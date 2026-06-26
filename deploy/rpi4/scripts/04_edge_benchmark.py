#!/usr/bin/env python3
"""
04_edge_benchmark.py
====================
Benchmark de modelos ONNX en dispositivos Edge (Raspberry Pi 4 / Jetson) para el TFM:
"Evaluacion comparativa de modelos CNN y Vision Transformers para deteccion de tizon
 en hojas de papa en dispositivos Edge".


USO (en la Raspberry Pi, dentro del venv):
    cd ~/tfm-edge
    source venv/bin/activate
    python scripts/04_edge_benchmark.py \
        --models-dir ~/tfm-edge/models \
        --test-csv   ~/tfm-edge/data/edge_test_pkg/test_edge.csv \
        --images-root ~/tfm-edge/data/edge_test_pkg \
        --device-tag rpi4 \
        --warmup 50 \
        --out ~/tfm-edge/results/edge_results_rpi.csv
"""

import argparse
import csv
import gc
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---- Silenciar warnings de deteccion de GPU de ORT 1.26 en RPi ----
import onnxruntime as ort
ort.set_default_logger_severity(3)  # 3 = ERROR (oculta los warnings de /sys/class/drm)

from PIL import Image

# =====================================================================
# CONSTANTES METODOLOGICAS (deben coincidir con el entrenamiento)
# =====================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224  # se sobreescribe por modelo leyendo el grafo ONNX

# Mapeo FIJO confirmado desde el corpus (NO alfabetico):
IDX_TO_CLASS = {0: "healthy", 1: "early_blight", 2: "late_blight"}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}
NUM_CLASSES = 3


# =====================================================================
# UTILIDADES DE HARDWARE (Raspberry Pi)
# =====================================================================
def vcgencmd(arg):
    """Lee un valor de vcgencmd; devuelve None si no esta disponible (no-RPi)."""
    try:
        out = subprocess.check_output(["vcgencmd"] + arg.split(),
                                      stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return None


def parse_temp(s):
    # "temp=37.0'C" -> 37.0
    try:
        return float(s.split("=")[1].split("'")[0])
    except Exception:
        return None


def parse_clock(s):
    # "frequency(48)=1800457088" -> 1800.46 (MHz)
    try:
        return int(s.split("=")[1]) / 1e6
    except Exception:
        return None


class HardwareSampler(threading.Thread):
    """Muestrea temperatura, frecuencia y throttling en segundo plano durante la inferencia."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = False
        self.temps = []
        self.freqs = []
        self.throttled_flags = []

    def run(self):
        self.running = True
        while self.running:
            t = parse_temp(vcgencmd("measure_temp") or "")
            f = parse_clock(vcgencmd("measure_clock arm") or "")
            thr = vcgencmd("get_throttled")
            if t is not None:
                self.temps.append(t)
            if f is not None:
                self.freqs.append(f)
            if thr is not None:
                self.throttled_flags.append(thr)
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.join(timeout=2)

    def summary(self):
        return {
            "temp_mean_C": round(statistics.mean(self.temps), 1) if self.temps else None,
            "temp_max_C": round(max(self.temps), 1) if self.temps else None,
            "freq_mean_MHz": round(statistics.mean(self.freqs), 0) if self.freqs else None,
            "freq_min_MHz": round(min(self.freqs), 0) if self.freqs else None,
            # Si en algun momento el flag != 0x0, hubo throttling durante la corrida
            "throttling_observed": any(
                f not in ("throttled=0x0", "0x0") for f in self.throttled_flags
            ) if self.throttled_flags else None,
        }


def get_peak_rss_mb():
    """RAM pico del proceso en MB. Usa psutil si esta, si no resource."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        import resource
        # ru_maxrss en KB en Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


# =====================================================================
# PREPROCESAMIENTO (identico al entrenamiento)
# =====================================================================
def preprocess(img_path, size):
    """Replica eval del entrenamiento: Resize(round(size*1.14)) -> CenterCrop(size) -> ImageNet -> NCHW."""
    resize_to = int(round(size * 1.14))                   # 224 -> 256
    img = Image.open(img_path).convert("RGB").resize(
        (resize_to, resize_to), Image.BILINEAR)           # resize a 256 (deforma a cuadrado, igual que albumentations.Resize)
    left = (resize_to - size) // 2
    top  = (resize_to - size) // 2
    img = img.crop((left, top, left + size, top + size))  # center-crop a 224
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)


# =====================================================================
# METRICAS
# =====================================================================
def compute_metrics(y_true, y_pred, sources):
    """Exactitud global, F1-macro, recall por clase y exactitud por dominio."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float((y_true == y_pred).mean())

    # F1 macro y recall por clase (sin sklearn, para no anadir deps a la RPi)
    f1s = []
    recalls = {}
    for c in range(NUM_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        recalls[IDX_TO_CLASS[c]] = round(rec, 4)
    f1_macro = float(np.mean(f1s))

    # Exactitud por dominio (source)
    acc_by_source = {}
    sources = np.array(sources)
    for src in np.unique(sources):
        mask = sources == src
        if mask.sum() > 0:
            acc_by_source[str(src)] = round(float((y_true[mask] == y_pred[mask]).mean()), 4)

    # Falsos negativos criticos: late_blight (2) clasificado como healthy (0)
    fn_critical = int(((y_true == CLASS_TO_IDX["late_blight"]) &
                       (y_pred == CLASS_TO_IDX["healthy"])).sum())

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "recall_healthy": recalls["healthy"],
        "recall_early_blight": recalls["early_blight"],
        "recall_late_blight": recalls["late_blight"],
        "fn_critical_late_to_healthy": fn_critical,
        "acc_by_source": acc_by_source,
    }


# =====================================================================
# BENCHMARK DE UN MODELO
# =====================================================================
def benchmark_model(onnx_path, samples, warmup, device_tag):
    """Ejecuta inferencia sobre todas las muestras y mide todo."""
    model_size_mb = os.path.getsize(onnx_path) / 1e6

    # Sesion ORT, 4 hilos (Cortex-A72 tiene 4 nucleos)
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    inp_name = inp.name
    # Leer resolucion del grafo (robusto a tamanos distintos de 224)
    shape = inp.shape
    size = INPUT_SIZE
    try:
        # shape tipico [1,3,224,224]; tomamos la ultima dim si es entero
        if isinstance(shape[-1], int) and shape[-1] > 0:
            size = shape[-1]
    except Exception:
        pass

    # ---- Warmup (no se mide) ----
    dummy = np.random.rand(1, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp_name: dummy})

    # ---- Sampler de hardware (solo util en RPi) ----
    sampler = HardwareSampler(interval=0.5)
    if device_tag.startswith("rpi"):
        sampler.start()

    # ---- Inferencia medida ----
    latencies = []
    y_true, y_pred, sources = [], [], []

    t_start_wall = time.time()
    for s in samples:
        x = preprocess(s["path"], size)
        t0 = time.perf_counter()
        out = sess.run(None, {inp_name: x})[0]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        pred = int(np.argmax(out[0]))
        y_pred.append(pred)
        y_true.append(s["label"])
        sources.append(s["source"])
    t_end_wall = time.time()

    if device_tag.startswith("rpi"):
        sampler.stop()

    peak_rss = get_peak_rss_mb()

    # ---- Estadisticas de latencia ----
    lat = np.array(latencies)
    lat_stats = {
        "lat_mean_ms": round(float(lat.mean()), 3),
        "lat_std_ms": round(float(lat.std()), 3),
        "lat_p50_ms": round(float(np.percentile(lat, 50)), 3),
        "lat_p90_ms": round(float(np.percentile(lat, 90)), 3),
        "lat_p95_ms": round(float(np.percentile(lat, 95)), 3),
        "lat_p99_ms": round(float(np.percentile(lat, 99)), 3),
        "throughput_fps": round(1000.0 / float(lat.mean()), 2),
    }

    metrics = compute_metrics(y_true, y_pred, sources)
    hw = sampler.summary() if device_tag.startswith("rpi") else {}

    # Energia estimada RELATIVA (RPi): P_din ~ freq * tiempo. Indicador comparativo, NO watts absolutos.
    energy_proxy = None
    if hw.get("freq_mean_MHz"):
        wall_s = t_end_wall - t_start_wall
        energy_proxy = round(hw["freq_mean_MHz"] * wall_s / 1000.0, 2)  # unidad arbitraria (MHz*s/1000)

    result = {
        "model_file": Path(onnx_path).name,
        "input_size": size,
        "n_images": len(samples),
        "model_size_mb": round(model_size_mb, 2),
        "peak_rss_mb": round(peak_rss, 1),
        "wall_time_s": round(t_end_wall - t_start_wall, 2),
        "energy_proxy_rel": energy_proxy,
        **lat_stats,
        **{k: v for k, v in metrics.items() if k != "acc_by_source"},
        "acc_by_source": metrics["acc_by_source"],
        **hw,
    }
    # Liberar memoria entre modelos
    del sess
    gc.collect()
    return result


# =====================================================================
# CARGA DEL TEST SET
# =====================================================================
def load_samples(test_csv, images_root):
    samples = []
    with open(test_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["filepath"]
            path = Path(images_root) / rel
            if not path.exists():
                # intento alternativo: ruta tal cual
                path = Path(rel)
            samples.append({
                "path": str(path),
                "label": int(row["label"]),
                "source": row.get("source", "unknown"),
                "class_name": row.get("class_name", ""),
            })
    return samples


# =====================================================================
# MAIN
# =====================================================================
def parse_precision(fname):
    f = fname.lower()
    if "fp16" in f:
        return "fp16"
    if "int8" in f:
        return "int8"
    return "fp32"


def parse_model_family(fname):
    f = fname.lower()
    if "mobilenetv4" in f or "mobilenet" in f:
        return "MobileNetV4-Small", "CNN"
    if "efficientnetv2" in f or "efficientnet" in f:
        return "EfficientNetV2-B0", "CNN"
    if "efficientformer" in f:
        return "EfficientFormerV2-S0", "Hybrid"
    if "repvit" in f:
        return "RepViT-M1", "Hybrid"
    return Path(fname).stem, "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--images-root", required=True,
                    help="Raiz a la que son relativas las rutas del CSV (carpeta del paquete)")
    ap.add_argument("--device-tag", default="rpi4", help="Etiqueta del dispositivo (rpi4, jetson, ...)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Limitar n de imagenes (0=todas, para pruebas)")
    args = ap.parse_args()

    print("=" * 70)
    print("BENCHMARK EDGE - TFM deteccion de tizon")
    print("=" * 70)
    print(f"Dispositivo : {args.device_tag}")
    print(f"Plataforma  : {platform.platform()}")
    print(f"Python      : {platform.python_version()}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Nucleos CPU : {os.cpu_count()}")
    print(f"Mapeo clases: {IDX_TO_CLASS}")
    print("=" * 70)

    samples = load_samples(args.test_csv, args.images_root)
    if args.limit:
        samples = samples[:args.limit]
    # Verificar integridad: cuantas imagenes existen
    missing = sum(1 for s in samples if not Path(s["path"]).exists())
    print(f"Imagenes de test: {len(samples)}  (faltantes: {missing})")
    if missing > 0:
        print("ADVERTENCIA: hay imagenes faltantes. Revisa --images-root.")
    if len(samples) == 0:
        sys.exit("ERROR: no se cargaron imagenes. Revisa rutas.")

    # Distribucion por clase y source (control metodologico)
    from collections import Counter
    print("Por clase :", dict(Counter(IDX_TO_CLASS[s["label"]] for s in samples)))
    print("Por source:", dict(Counter(s["source"] for s in samples)))
    print("=" * 70)

    # Encontrar todos los ONNX
    model_files = sorted(Path(args.models_dir).glob("*.onnx"))
    if not model_files:
        sys.exit(f"ERROR: no se encontraron .onnx en {args.models_dir}")
    print(f"Modelos encontrados: {len(model_files)}")
    for m in model_files:
        print(f"  - {m.name}")
    print("=" * 70)

    all_results = []
    for i, mf in enumerate(model_files, 1):
        family, ftype = parse_model_family(mf.name)
        precision = parse_precision(mf.name)
        print(f"\n[{i}/{len(model_files)}] {mf.name}")
        print(f"    Familia: {family} ({ftype}) | Precision: {precision}")
        print(f"    Ejecutando {len(samples)} inferencias (warmup={args.warmup})...")

        try:
            res = benchmark_model(mf, samples, args.warmup, args.device_tag)
        except Exception as e:
            print(f"    ERROR al evaluar {mf.name}: {e}")
            continue

        res["family"] = family
        res["type"] = ftype
        res["precision"] = precision
        res["device"] = args.device_tag
        all_results.append(res)

        print(f"    -> acc={res['accuracy']:.4f}  f1={res['f1_macro']:.4f}  "
              f"lat={res['lat_mean_ms']:.2f}ms  fps={res['throughput_fps']:.1f}  "
              f"ram={res['peak_rss_mb']:.0f}MB")
        if res.get("temp_max_C"):
            print(f"    -> temp_max={res['temp_max_C']}C  "
                  f"freq_min={res['freq_min_MHz']}MHz  "
                  f"throttling={res.get('throttling_observed')}")
        print(f"    -> acc por dominio: {res['acc_by_source']}")
        print(f"    -> FN criticos (late->healthy): {res['fn_critical_late_to_healthy']}")

    # Guardar CSV (acc_by_source como string JSON)
    import json
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_results:
        # columnas: union de todas las claves, acc_by_source serializado
        fieldnames = [
            "device", "family", "type", "precision", "model_file", "input_size",
            "n_images", "accuracy", "f1_macro",
            "recall_healthy", "recall_early_blight", "recall_late_blight",
            "fn_critical_late_to_healthy",
            "lat_mean_ms", "lat_std_ms", "lat_p50_ms", "lat_p90_ms",
            "lat_p95_ms", "lat_p99_ms", "throughput_fps",
            "peak_rss_mb", "model_size_mb", "wall_time_s", "energy_proxy_rel",
            "temp_mean_C", "temp_max_C", "freq_mean_MHz", "freq_min_MHz",
            "throttling_observed", "acc_by_source",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                r = dict(r)
                r["acc_by_source"] = json.dumps(r.get("acc_by_source", {}))
                writer.writerow(r)
        print("\n" + "=" * 70)
        print(f"Resultados guardados en: {out_path}")
        print(f"Modelos evaluados: {len(all_results)}")
        print("=" * 70)
    else:
        print("\nNo se generaron resultados.")


if __name__ == "__main__":
    main()
