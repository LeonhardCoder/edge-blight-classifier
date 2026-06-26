#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_preds_rpi.py  ── Genera predicciones POR IMAGEN para los 8 artefactos ONNX
de la Raspberry Pi 4, usando EL MISMO preprocesado de evaluacion con el que se
entreno/valido (get_eval_transforms), para que la accuracy sea comparable.

CORRIGE el desajuste train/serve detectado:
  - Entrenamiento/validacion: Resize(255) -> CenterCrop(224) -> norma ImageNet
  - Despliegue RPi (04b):      Resize(224) directo, SIN center-crop  ← desajuste
Este script aplica el pipeline de evaluacion (Opcion A). Para volver al squash
(Opcion B, solo si fuera produccion deliberada), comenta CenterCrop/ajusta Resize.

SALIDA: CSV largo  image_id, source, model, precision, y_true, y_pred
        + informe de sanidad (accuracy y F1-macro global y por fuente).

EJECUTABLE EN CUALQUIER MAQUINA: ORT-CPU es determinista; y_pred no depende del
dispositivo. Solo latencia/energia exigian la Pi fisica.

DEPENDENCIAS:
  pip install onnxruntime albumentations opencv-python-headless pillow pandas numpy scikit-learn --break-system-packages
"""

import os, sys
import numpy as np
import pandas as pd
from PIL import Image

# ─── CONFIG (rutas del dispositivo) 
MODELS_DIR = "./models"
TEST_CSV   = "./data/test_edge.csv"     # columnas: filepath, label, source
PATH_BASE  = ""                         # prefijo si filepath es relativa; "" = abrir tal cual
OUT_CSV    = "./results/preds_rpi_per_image.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IDX_TO_CLASS  = {0: "healthy", 1: "early_blight", 2: "late_blight"}

# 8 artefactos. FP32 = nombre "a secas" (+ su .onnx.data); FP16 = sufijo _fp16.
ARTIFACTS = [
    ("mobilenetv4_conv_small", "FP32", "mobilenetv4_conv_small.onnx"),
    ("mobilenetv4_conv_small", "FP16", "mobilenetv4_conv_small_fp16.onnx"),
    ("efficientnetv2_b0",      "FP32", "efficientnetv2_b0.onnx"),
    ("efficientnetv2_b0",      "FP16", "efficientnetv2_b0_fp16.onnx"),
    ("efficientformerv2_s0",   "FP32", "efficientformerv2_s0.onnx"),
    ("efficientformerv2_s0",   "FP16", "efficientformerv2_s0_fp16.onnx"),
    ("repvit_m1",              "FP32", "repvit_m1.onnx"),
    ("repvit_m1",              "FP16", "repvit_m1_fp16.onnx"),
]
# 

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    import albumentations as A
    from sklearn.metrics import accuracy_score, f1_score
except ImportError as e:
    sys.exit(f"[ERROR] Falta dependencia: {e}")


def get_eval_transforms(img_size: int = 224):
    """Replica EXACTA del transform de evaluacion del entrenamiento (sin ToTensorV2,
    que es un no-op numerico: solo reordena a CHW, lo hacemos despues con numpy)."""
    return A.Compose([
        A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # max_pixel_value=255 por defecto
    ])


def load_manifest(path):
    df = pd.read_csv(path)
    need = {"filepath", "label"}
    if not need.issubset(df.columns):
        sys.exit(f"[ERROR] El manifest debe tener columnas {need}. Tiene: {list(df.columns)}")
    df = df.reset_index(drop=True)
    df["image_id"] = df.index          # ID estable = orden del manifest
    df["y_true"]   = df["label"].astype(int)
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df


def make_session(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = inp.shape[-1] if isinstance(inp.shape[-1], int) and inp.shape[-1] > 0 else 224
    np_dtype = np.float16 if "float16" in inp.type else np.float32
    return sess, inp.name, size, np_dtype


def run_artifact(onnx_path, df):
    if not os.path.isfile(onnx_path):
        print(f"  [AVISO] No existe {onnx_path} — se omite.")
        return None
    sess, in_name, size, np_dtype = make_session(onnx_path)
    tf = get_eval_transforms(size)
    print(f"  input_size={size} | dtype entrada={np_dtype.__name__}")

    preds = np.empty(len(df), dtype=np.int64)
    for pos, row in df.iterrows():
        p = os.path.join(PATH_BASE, row["filepath"]) if PATH_BASE else row["filepath"]
        img = np.array(Image.open(p).convert("RGB"))     # HWC RGB uint8
        x = tf(image=img)["image"]                       # HWC float32 normalizado
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np_dtype)   # 1,C,H,W
        logits = sess.run(None, {in_name: np.ascontiguousarray(x)})[0]
        preds[pos] = int(np.argmax(logits[0]))
    return preds


def main():
    df = load_manifest(TEST_CSV)
    print(f"[OK] Manifest: {len(df)} imagenes | fuentes: "
          f"{df['source'].value_counts().to_dict()}\n")

    rows, sanity = [], []
    y_true = df["y_true"].to_numpy()

    for model, prec, fname in ARTIFACTS:
        print(f"▶ {model} [{prec}]")
        preds = run_artifact(os.path.join(MODELS_DIR, fname), df)
        if preds is None:
            continue
        acc = accuracy_score(y_true, preds)
        f1m = f1_score(y_true, preds, average="macro")
        print(f"  → accuracy={acc:.4f}  F1-macro={f1m:.4f}")
        # accuracy por fuente (util para discusion por dominio)
        for src in sorted(df["source"].unique()):
            m = df["source"] == src
            print(f"     {src:<28} acc={accuracy_score(y_true[m], preds[m]):.4f} (n={m.sum()})")
        sanity.append({"model": model, "precision": prec,
                       "accuracy": round(acc, 4), "f1_macro": round(f1m, 4)})
        for iid, src, yt, yp in zip(df["image_id"], df["source"], y_true, preds):
            rows.append({"image_id": int(iid), "source": src, "model": model,
                         "precision": prec, "y_true": int(yt), "y_pred": int(yp)})
        print()

    if not rows:
        sys.exit("[ERROR] Sin predicciones. Revisa MODELS_DIR y nombres de ONNX.")

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("=" * 64)
    print(f"[OK] Escrito: {OUT_CSV}  ({len(rows)} filas)")
    print("\nSANIDAD — contrasta con 'F1 test':")
    print(pd.DataFrame(sanity).to_string(index=False))
    print("\nSi la accuracy ahora CONVERGE a tu F1 test (±0,005), el desajuste de")
    print("preprocesado era la unica causa de discrepancia. Si no, revisar mapeo de clases.")


if __name__ == "__main__":
    main()
