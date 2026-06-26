#!/usr/bin/env python3
"""
04_jetson_benchmark_energy.py
=============================
medicion de consumo energetico via tegrastats.

REQUISITOS:
  - sudo -n tegrastats debe funcionar sin password. Configurar:
      sudo visudo
      victor ALL=(ALL) NOPASSWD: /usr/bin/tegrastats, /usr/bin/killall
  - jetson_clocks aplicado: sudo nvpmodel -m 1 && sudo jetson_clocks

USO:
    source ~/tfm-edge/venv-jetson/bin/activate
    python ~/tfm-edge/scripts/04_jetson_benchmark_energy.py \\
        --engines-dir ~/tfm-edge/engines \\
        --test-csv    /home/victor/Documents/edge-blight-classifier/project/outputs/corpus/test.csv \\
        --images-root /home/victor/Documents/dataset \\
        --device-tag  jetson_orin_nano \\
        --warmup 50 \\
        --idle-seconds 30 \\
        --out ~/tfm-edge/results/edge_results_jetson_energy2.csv

     python ~/tfm-edge/scripts/04_jetson_benchmark_energy.py \\
        --engines-dir ~/tfm-edge/engines_new \\
        --test-csv    /home/victor/Documents/edge-blight-classifier/project/outputs/corpus/test.csv \\
        --images-root /home/victor/Documents/dataset \\
        --device-tag  jetson_orin_nano \\
        --warmup 50 \\
        --idle-seconds 30 \\
        --out ~/tfm-edge/results/edge_results_jetson_energy2.csv
"""

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from trt_runner import TRTRunner  
from tegrastats_sampler import TegrastatsSampler  #
import tensorrt as trt  


# =====================================================================
# CONSTANTES METODOLOGICAS (identicas al script original)
# =====================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IDX_TO_CLASS = {0: "healthy", 1: "early_blight", 2: "late_blight"}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}
NUM_CLASSES = 3


# =====================================================================
# UTILIDADES (copiadas del script original, sin cambios)
# =====================================================================
def get_peak_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def resolve_path(raw, images_root, remap_from):
    candidates = []
    if remap_from and raw.startswith(remap_from):
        candidates.append(raw.replace(remap_from, images_root))
    candidates.append(raw)
    candidates.append(str(Path(images_root) / raw.lstrip("/")))
    for c in list(candidates):
        candidates.append(unicodedata.normalize("NFC", c))
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if Path(c).exists():
            return c
    return None


def preprocess(img_path, size):
    """Replica get_eval_transforms de Colab: Resize(size*1.14) + CenterCrop(size)."""
    resize_to = int(round(size * 1.14))      # 224 -> 256
    img = Image.open(img_path).convert("RGB").resize(
        (resize_to, resize_to), Image.BILINEAR)
    # center-crop a size x size
    left = (resize_to - size) // 2
    top = (resize_to - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def compute_metrics(y_true, y_pred, sources):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sources = np.array(sources)
    acc = float((y_true == y_pred).mean())

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

    acc_by_source = {}
    for src in np.unique(sources):
        mask = sources == src
        if mask.sum() > 0:
            acc_by_source[str(src)] = round(
                float((y_true[mask] == y_pred[mask]).mean()), 4)

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
# BENCHMARK DE UN ENGINE (extendido con energia)
# =====================================================================
def benchmark_engine(engine_path, samples, warmup, sampler, p_idle_w):
    model_size_mb = os.path.getsize(engine_path) / 1e6
    runner = TRTRunner(engine_path)
    size = runner.input_shape[-1]

    # Warmup (no se mide, no se incluye en la ventana energetica)
    dummy = np.random.rand(1, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        runner.predict(dummy)

    latencies_total = []
    latencies_exec = []
    latencies_h2d = []
    latencies_d2h = []
    y_true, y_pred, sources = [], [], []

    # ---- Inicio de la ventana de medicion (latencia Y energia) ----
    t_start = time.perf_counter()
    t_start_wall = time.time()

    for s in samples:
        x = preprocess(s["path"], size)
        logits, t_h2d, t_exec, t_d2h, t_total = runner.predict_timed(x)
        latencies_h2d.append(t_h2d)
        latencies_exec.append(t_exec)
        latencies_d2h.append(t_d2h)
        latencies_total.append(t_total)
        pred = int(np.argmax(logits.flatten()))
        y_pred.append(pred)
        y_true.append(s["label"])
        sources.append(s["source"])

    t_end = time.perf_counter()
    t_end_wall = time.time()
    # ---- Fin de la ventana ----


    time.sleep(0.2)

    peak_rss = get_peak_rss_mb()
    gpu_mem_bytes = runner.device_memory_size or 0
    lat = np.array(latencies_total)
    lat_stats = {
        "lat_mean_ms": round(float(lat.mean()), 3),
        "lat_std_ms": round(float(lat.std()), 3),
        "lat_p50_ms": round(float(np.percentile(lat, 50)), 3),
        "lat_p90_ms": round(float(np.percentile(lat, 90)), 3),
        "lat_p95_ms": round(float(np.percentile(lat, 95)), 3),
        "lat_p99_ms": round(float(np.percentile(lat, 99)), 3),
        "throughput_fps": round(1000.0 / float(lat.mean()), 2),
        "lat_compute_only_ms": round(float(np.mean(latencies_exec)), 3),
        "lat_h2d_ms": round(float(np.mean(latencies_h2d)), 3),
        "lat_d2h_ms": round(float(np.mean(latencies_d2h)), 3),
        "gpu_mem_mb": round(gpu_mem_bytes / 1e6, 2),
    }

    # ---- Estadisticas energeticas ----
    energy_stats = sampler.get_window_stats(t_start, t_end)

    n_inf = len(samples)
    e_total_j = energy_stats.get("energy_in_j") or 0.0
    e_per_inf_j = e_total_j / n_inf if n_inf > 0 else 0.0

    # Energia neta: descontamos el coste idle
    duration_s = energy_stats.get("duration_s") or 0.0
    e_idle_during_window = (p_idle_w or 0.0) * duration_s
    e_net_j = max(0.0, e_total_j - e_idle_during_window)
    e_per_inf_net_j = e_net_j / n_inf if n_inf > 0 else 0.0


    inf_per_wh = (n_inf / (e_total_j / 3600.0)) if e_total_j > 0 else 0.0

    energy_cols = {
        "p_idle_w": round(p_idle_w, 3) if p_idle_w is not None else None,
        "p_avg_w": round(energy_stats.get("p_in_avg_w") or 0.0, 3),
        "p_max_w": round(energy_stats.get("p_in_max_w") or 0.0, 3),
        "p_compute_avg_w": round(energy_stats.get("p_compute_avg_w") or 0.0, 3),
        "energy_total_j": round(e_total_j, 3),
        "e_per_inf_j": round(e_per_inf_j, 4),
        "e_per_inf_net_j": round(e_per_inf_net_j, 4),
        "inf_per_wh": round(inf_per_wh, 1),
        "n_power_samples": energy_stats.get("n_samples") or 0,
    }

    metrics = compute_metrics(y_true, y_pred, sources)

    result = {
        "model_file": Path(engine_path).name,
        "input_size": size,
        "n_images": n_inf,
        "model_size_mb": round(model_size_mb, 2),
        "peak_rss_mb": round(peak_rss, 1),
        "wall_time_s": round(t_end_wall - t_start_wall, 2),
        "energy_proxy_rel": None,
        **lat_stats,
        **{k: v for k, v in metrics.items() if k != "acc_by_source"},
        "acc_by_source": metrics["acc_by_source"],
        "temp_mean_C": None,
        "temp_max_C": None,
        "freq_mean_MHz": None,
        "freq_min_MHz": None,
        "throttling_observed": None,
        **energy_cols,
    }

    del runner
    gc.collect()
    return result


def load_samples(test_csv, images_root, remap_from):
    samples = []
    missing = []
    with open(test_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["abs_path"] if "abs_path" in row else row["path"]
            resolved = resolve_path(raw, images_root, remap_from)
            if resolved is None:
                missing.append(raw)
                continue
            samples.append({
                "path": resolved,
                "label": int(row["label"]),
                "source": row.get("source", "unknown"),
                "class_name": row.get("class_name", ""),
            })
    return samples, missing


# =====================================================================
# IDENTIFICACION DE MODELOS (idem original)
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


# =====================================================================
# MAIN
# =====================================================================
def main():
    TEST_CSV  = Path("/home/victor/Documents/edge-blight-classifier/project/outputs/corpus/test.csv")
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines-dir", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--remap-from", default="/home/victor/Documents/dataset")
    ap.add_argument("--device-tag", default="jetson_orin_nano")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--idle-seconds", type=int, default=30,
                    help="Segundos de medicion idle previa para calibrar p_idle")
    ap.add_argument("--tegrastats-interval", type=int, default=100,
                    help="Intervalo de muestreo de tegrastats en ms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("BENCHMARK EDGE - Jetson + Energia - TFM deteccion de tizon")
    print("=" * 70)
    print(f"Dispositivo : {args.device_tag}")
    print(f"Plataforma  : {platform.platform()}")
    print(f"Python      : {platform.python_version()}")
    print(f"TensorRT    : {trt.__version__}")
    print(f"Mapeo clases: {IDX_TO_CLASS}")
    print(f"Remap: {args.remap_from}  ->  {args.images_root}")
    print("=" * 70)

    samples, missing = load_samples(args.test_csv, args.images_root, args.remap_from)
    if args.limit:
        samples = samples[:args.limit]
    print(f"Imagenes resueltas: {len(samples)}  (no encontradas: {len(missing)})")
    if not samples:
        sys.exit("ERROR: no se cargaron imagenes.")
    print("Por clase :", dict(Counter(IDX_TO_CLASS[s["label"]] for s in samples)))
    print("Por source:", dict(Counter(s["source"] for s in samples)))
    print("=" * 70)

    engine_files = sorted(Path(args.engines_dir).glob("*.engine"))
    if not engine_files:
        sys.exit(f"ERROR: no se encontraron engines en {args.engines_dir}")
    print(f"Engines encontrados: {len(engine_files)}")
    for e in engine_files:
        print(f"  - {e.name}")
    print("=" * 70)

    # ---- Arrancar sampler tegrastats ----
    print(f"\nArrancando tegrastats (intervalo {args.tegrastats_interval}ms)...")
    sampler = TegrastatsSampler(interval_ms=args.tegrastats_interval)
    try:
        sampler.start()
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Ventana idle para calibrar p_idle ----
    print(f"Midiendo {args.idle_seconds}s de idle para calibrar p_idle...")
    t_idle_start = time.perf_counter()
    time.sleep(args.idle_seconds)
    t_idle_end = time.perf_counter()
    idle_stats = sampler.get_window_stats(t_idle_start, t_idle_end)
    p_idle_w = idle_stats.get("p_in_avg_w") or 0.0
    print(f"  p_idle = {p_idle_w:.3f} W (n={idle_stats['n_samples']} muestras)")
    print("=" * 70)

    # ---- Benchmark de cada engine ----
    all_results = []
    for i, ef in enumerate(engine_files, 1):
        family, ftype = parse_model_family(ef.name)
        precision = parse_precision(ef.name)
        print(f"\n[{i}/{len(engine_files)}] {ef.name}")
        print(f"    Familia: {family} ({ftype}) | Precision: {precision}")
        print(f"    Ejecutando {len(samples)} inferencias (warmup={args.warmup})...")

        try:
            res = benchmark_engine(ef, samples, args.warmup, sampler, p_idle_w)
        except Exception as e:
            print(f"    ERROR al evaluar {ef.name}: {e}")
            continue

        res["family"] = family
        res["type"] = ftype
        res["precision"] = precision
        res["device"] = args.device_tag
        all_results.append(res)

        print(f"    -> acc={res['accuracy']:.4f}  f1={res['f1_macro']:.4f}  "
              f"lat={res['lat_mean_ms']:.2f}ms  fps={res['throughput_fps']:.1f}")
        print(f"    -> P_avg={res['p_avg_w']:.2f}W  P_max={res['p_max_w']:.2f}W  "
              f"P_compute={res['p_compute_avg_w']:.2f}W")
        print(f"    -> E_total={res['energy_total_j']:.2f}J  "
              f"E/inf={res['e_per_inf_j']*1000:.2f}mJ  "
              f"E/inf_net={res['e_per_inf_net_j']*1000:.2f}mJ  "
              f"inf/Wh={res['inf_per_wh']:.0f}")
        print(f"    -> acc por dominio: {res['acc_by_source']}")
        print(f"    -> FN criticos: {res['fn_critical_late_to_healthy']}")


        time.sleep(2)

    # ---- Parar sampler ----
    sampler.stop()

    # ---- Volcar CSV ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_results:
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
            # Columnas Jetson rendimiento:
            "lat_compute_only_ms", "lat_h2d_ms", "lat_d2h_ms", "gpu_mem_mb",
            # Columnas Jetson energia (NUEVAS):
            "p_idle_w", "p_avg_w", "p_max_w", "p_compute_avg_w",
            "energy_total_j", "e_per_inf_j", "e_per_inf_net_j", "inf_per_wh",
            "n_power_samples",
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
        print(f"Engines evaluados: {len(all_results)}")
        print(f"p_idle calibrado:  {p_idle_w:.3f} W")
        print("=" * 70)
    else:
        print("\nNo se generaron resultados.")


if __name__ == "__main__":
    main()
