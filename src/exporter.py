"""
Exportacion de modelos PyTorch a formatos Edge:
  - ONNX (FP32 base, FP16)
  - TFLite (INT8) via ONNX -> TF -> TFLite
  - TensorRT (FP16, INT8) - se construye en Jetson, no aqui

NOTA: TensorRT solo se construye en la propia Jetson (requiere CUDA/TRT).
Este script genera el ONNX que sera el input para 'trtexec' en Jetson.
"""
from pathlib import Path

import numpy as np
import torch

import config
from src.models import create_model, reparameterize_if_needed


def load_checkpoint(model_key: str, ckpt_path=None):
    """Carga un checkpoint entrenado en un modelo recien creado."""
    if ckpt_path is None:
        ckpt_path = config.CHECKPOINTS_DIR / f"{model_key}_best.pth"

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"No existe checkpoint: {ckpt_path}")

    model = create_model(model_key, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def export_onnx(model_key, opset=17, batch_size=1):
    """
    Exporta el modelo entrenado a ONNX FP32 con batch fijo a 1
    (configuracion optima para Edge).
    """
    out_dir = config.EXPORTED_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{model_key}.onnx"

    model, _ = load_checkpoint(model_key)
    reparam = reparameterize_if_needed(model)
    if reparam:
        print(f"  [{model_key}] re-parametrizacion aplicada antes de exportar")

    img_size = config.MODEL_REGISTRY[model_key]["img_size"]
    dummy = torch.randn(batch_size, 3, img_size, img_size)

    torch.onnx.export(
        model, dummy, str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,   # batch FIJO para Edge
    )
    print(f"  [{model_key}] ONNX exportado: {onnx_path}")
    return onnx_path


def verify_onnx(model_key, num_samples=5, tol=1e-4):
    """Comprueba que ONNX produce las mismas predicciones que PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime no instalado, saltando verificacion")
        return None

    onnx_path = config.EXPORTED_DIR / model_key / f"{model_key}.onnx"
    model, _ = load_checkpoint(model_key)
    reparameterize_if_needed(model)

    img_size = config.MODEL_REGISTRY[model_key]["img_size"]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0

    for _ in range(num_samples):
        x = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            y_pt = model(x).numpy()
        y_ort = sess.run(None, {"input": x.numpy()})[0]
        diff = np.abs(y_pt - y_ort).max()
        max_diff = max(max_diff, diff)

    ok = max_diff < tol
    print(f"  [{model_key}] verificacion ONNX: max_diff={max_diff:.2e} "
          f"{'OK' if ok else 'WARN (sobre tolerancia)'}")
    return max_diff


def export_tflite_int8(model_key, calibration_csv=None):
    """
    Exporta a TFLite-INT8 via onnx -> tf -> tflite.

    Requiere: onnx-tf, tensorflow.
    Para calibracion INT8 se usan ~100 imagenes del split de train.
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as e:
        print(f"  Faltan dependencias para TFLite: {e}")
        print("  Instalar: pip install onnx-tf tensorflow")
        return None

    out_dir = config.EXPORTED_DIR / model_key
    onnx_path = out_dir / f"{model_key}.onnx"
    tf_dir = out_dir / "tf_saved_model"
    tflite_path = out_dir / f"{model_key}_int8.tflite"

    if not onnx_path.exists():
        print(f"  [{model_key}] primero exporta a ONNX")
        return None

    # ONNX -> SavedModel
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(tf_dir))

    # SavedModel -> TFLite con cuantizacion INT8
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if calibration_csv is not None:
        import pandas as pd
        import cv2
        from src.transforms import IMAGENET_MEAN, IMAGENET_STD

        df = pd.read_csv(calibration_csv)
        sample_paths = df["abs_path"].sample(
            min(100, len(df)), random_state=config.SEED,
        ).tolist()

        img_size = config.MODEL_REGISTRY[model_key]["img_size"]

        def representative_dataset():
            for path in sample_paths:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                img = img.astype(np.float32) / 255.0
                img = (img - IMAGENET_MEAN) / IMAGENET_STD
                # NHWC -> NCHW
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0).astype(np.float32)
                yield [img]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"  [{model_key}] TFLite-INT8 exportado: {tflite_path} "
          f"({tflite_path.stat().st_size / 1024:.1f} KB)")
    return tflite_path


def build_tensorrt_command(model_key, precision="fp16"):
    """
    Genera el comando trtexec para construir el motor TensorRT EN LA JETSON.
    Este script NO ejecuta trtexec (no esta disponible en local).

    Devuelve el comando como string para que el usuario lo ejecute en
    la Jetson tras transferir el ONNX.
    """
    onnx_path = config.EXPORTED_DIR / model_key / f"{model_key}.onnx"
    engine_name = f"{model_key}_{precision}.engine"
    engine_path = config.EXPORTED_DIR / model_key / engine_name

    flags = "--fp16" if precision == "fp16" else "--int8"
    cmd = (
        f"trtexec --onnx={onnx_path.name} "
        f"--saveEngine={engine_name} "
        f"{flags} "
        f"--workspace=2048"
    )
    print(f"  [{model_key}] comando para Jetson:")
    print(f"    {cmd}")
    return cmd
