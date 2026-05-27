"""
Benchmark de modelos en dispositivos Edge.

Mide para cada combinacion (modelo, formato, dispositivo):
  - Exactitud y F1 sobre el split de test
  - Latencia media y P95 sobre N inferencias consecutivas
  - Uso de memoria RAM pico durante la inferencia
  - Tamano del modelo serializado

Runtime-agnostico: cada formato tiene su propio wrapper de inferencia,
todos exponen la misma API (predict_batch, predict_one).
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import config
from src.transforms import IMAGENET_MEAN, IMAGENET_STD


# =============================================================================
# WRAPPERS DE INFERENCIA POR FORMATO
# =============================================================================

class OnnxRunner:
    """Inferencia con onnxruntime (CPU o CUDA segun providers)."""

    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, batch_np):
        return self.session.run(None, {self.input_name: batch_np})[0]


class TFLiteRunner:
    """Inferencia con tensorflow lite (CPU optimizado para ARM)."""

    def __init__(self, tflite_path):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        self.interpreter = tflite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype    = self.input_details[0]["dtype"]
        self.input_quant    = self.input_details[0].get("quantization", (0.0, 0))
        self.output_quant   = self.output_details[0].get("quantization", (0.0, 0))

    def predict(self, batch_np):
        # batch_np viene en float32 normalizado (NCHW)
        if self.input_dtype == np.int8:
            scale, zero_point = self.input_quant
            if scale > 0:
                batch_np = (batch_np / scale + zero_point).astype(np.int8)
            else:
                batch_np = batch_np.astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]["index"], batch_np)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])

        if self.output_details[0]["dtype"] == np.int8:
            scale, zero_point = self.output_quant
            out = (out.astype(np.float32) - zero_point) * scale
        return out


class TensorRTRunner:
    """Inferencia con TensorRT (solo Jetson, requiere pycuda + tensorrt)."""

    def __init__(self, engine_path):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"TensorRT/pycuda no disponibles: {e}. "
                "Este runner solo funciona en Jetson."
            )
        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Preparar buffers
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = int(np.prod(shape))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    def predict(self, batch_np):
        np.copyto(self.inputs[0]["host"], batch_np.ravel())
        self.cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        out_shape = self.outputs[0]["shape"]
        return self.outputs[0]["host"].reshape(out_shape).copy()


def load_runner(model_key, format_name):
    """Factory de runners segun el formato solicitado."""
    out_dir = config.EXPORTED_DIR / model_key

    if format_name == "onnx_fp16" or format_name == "onnx_fp32":
        onnx_path = out_dir / f"{model_key}.onnx"
        # Para GPU usa CUDA provider si esta en jetson
        providers = ["CPUExecutionProvider"]
        if config.ENV == "edge_jetson":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return OnnxRunner(onnx_path, providers=providers)

    if format_name == "tflite_int8":
        tflite_path = out_dir / f"{model_key}_int8.tflite"
        return TFLiteRunner(tflite_path)

    if format_name == "trt_fp16":
        engine_path = out_dir / f"{model_key}_fp16.engine"
        return TensorRTRunner(engine_path)

    if format_name == "trt_int8":
        engine_path = out_dir / f"{model_key}_int8.engine"
        return TensorRTRunner(engine_path)

    raise ValueError(f"Formato desconocido: {format_name}")


# =============================================================================
# PRE-PROCESAMIENTO PARA INFERENCIA
# =============================================================================

def preprocess_image(path, img_size=224):
    """Mismo preprocesado que en validacion (deterministic)."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize + center crop (consistente con get_eval_transforms)
    h_resize = int(img_size * 1.14)
    img = cv2.resize(img, (h_resize, h_resize))
    cy, cx = h_resize // 2, h_resize // 2
    s = img_size // 2
    img = img[cy - s:cy + s, cx - s:cx + s]

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # NHWC -> NCHW
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0).astype(np.float32)


# =============================================================================
# MEDICION DE LATENCIA Y MEMORIA
# =============================================================================

def measure_latency(runner, img_size=224, warmup=20, iters=500):
    """Mide latencia de inferencia repetida sobre una imagen dummy."""
    dummy = np.random.rand(1, 3, img_size, img_size).astype(np.float32)

    # Warm-up
    for _ in range(warmup):
        runner.predict(dummy)

    # Medicion
    times_ms = []
    proc = psutil.Process()
    mem_samples = []

    for _ in range(iters):
        mem_samples.append(proc.memory_info().rss / (1024 * 1024))
        t0 = time.perf_counter()
        runner.predict(dummy)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_arr = np.array(times_ms)
    mem_arr = np.array(mem_samples)

    return {
        "lat_mean_ms":  float(times_arr.mean()),
        "lat_std_ms":   float(times_arr.std()),
        "lat_p50_ms":   float(np.percentile(times_arr, 50)),
        "lat_p95_ms":   float(np.percentile(times_arr, 95)),
        "lat_p99_ms":   float(np.percentile(times_arr, 99)),
        "lat_min_ms":   float(times_arr.min()),
        "lat_max_ms":   float(times_arr.max()),
        "throughput_fps": 1000.0 / float(times_arr.mean()),
        "mem_mean_mb":  float(mem_arr.mean()),
        "mem_peak_mb":  float(mem_arr.max()),
        "n_iters":      int(iters),
    }


def measure_accuracy(runner, test_csv, img_size=224, max_samples=None):
    """Evalua exactitud y F1 sobre las imagenes del CSV de test."""
    df = pd.read_csv(test_csv)
    if max_samples is not None:
        df = df.head(max_samples)

    y_true, y_pred = [], []
    for _, row in df.iterrows():
        x = preprocess_image(row["abs_path"], img_size)
        if x is None:
            continue
        logits = runner.predict(x)
        pred = int(np.argmax(logits.flatten()))
        y_pred.append(pred)
        y_true.append(int(row["label"]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "acc":      float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_cls": f1_score(y_true, y_pred, average=None,
                               labels=list(range(config.NUM_CLASSES)),
                               zero_division=0).tolist(),
        "cm":        confusion_matrix(y_true, y_pred,
                                      labels=list(range(config.NUM_CLASSES))).tolist(),
        "n_samples": int(len(y_true)),
    }


def get_model_size_kb(model_key, format_name):
    """Tamano del archivo del modelo en KB."""
    out_dir = config.EXPORTED_DIR / model_key
    paths = {
        "onnx_fp32":   out_dir / f"{model_key}.onnx",
        "onnx_fp16":   out_dir / f"{model_key}.onnx",
        "tflite_int8": out_dir / f"{model_key}_int8.tflite",
        "trt_fp16":    out_dir / f"{model_key}_fp16.engine",
        "trt_int8":    out_dir / f"{model_key}_int8.engine",
    }
    p = paths.get(format_name)
    if p and p.exists():
        return p.stat().st_size / 1024.0
    return None


def benchmark_combination(model_key, format_name, test_csv, device_label):
    """
    Benchmark completo de una combinacion (modelo, formato, dispositivo).
    Devuelve un diccionario con todas las metricas.
    """
    print(f"\n  [{device_label} | {model_key} | {format_name}]")

    runner = load_runner(model_key, format_name)
    img_size = config.MODEL_REGISTRY[model_key]["img_size"]

    print(f"    Midiendo exactitud...")
    acc_metrics = measure_accuracy(runner, test_csv, img_size)
    print(f"    acc={acc_metrics['acc']:.4f}  f1={acc_metrics['f1_macro']:.4f}")

    print(f"    Midiendo latencia ({config.EDGE_BENCHMARK['measured_iters']} iters)...")
    lat_metrics = measure_latency(
        runner, img_size,
        warmup=config.EDGE_BENCHMARK["warmup_iters"],
        iters=config.EDGE_BENCHMARK["measured_iters"],
    )
    print(f"    lat_mean={lat_metrics['lat_mean_ms']:.2f}ms  "
          f"p95={lat_metrics['lat_p95_ms']:.2f}ms  "
          f"FPS={lat_metrics['throughput_fps']:.1f}")

    size_kb = get_model_size_kb(model_key, format_name)

    return {
        "device":      device_label,
        "model_key":   model_key,
        "family":      config.MODEL_REGISTRY[model_key]["family"],
        "format":      format_name,
        "size_kb":     size_kb,
        **acc_metrics,
        **lat_metrics,
    }
