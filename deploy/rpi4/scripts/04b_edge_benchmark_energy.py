#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04b_edge_benchmark_energy.py  (CORREGIDO)
=========================================
Variante del benchmark Edge para MEDICION ENERGETICA MANUAL con medidor USB
en linea (KWS-06C o similar, solo display, sin salida de datos a PC).

USO:
    python scripts/04b_edge_benchmark_energy.py \
        --models-dir ~/tfm-edge/models --test-csv ~/tfm-edge/data/test_edge.csv \
        --model mobilenetv4_conv_small.onnx --out ~/tfm-edge/results/energy_rpi.csv
    # o --all para todos en secuencia con pausas
"""

import argparse
import csv
import gc
import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IDX_TO_CLASS = {0: "healthy", 1: "early_blight", 2: "late_blight"}

# Transform de evaluacion: redimensionar a 1,14x224 = 255 y recortar el centro 224.
RESIZE_RATIO = 255.0 / 224.0   # = 1,1384 -> 255 cuando size=224


def preprocess(img_path, size):
    """Transform de EVALUACION (debe ser identico a gen_preds_rpi.py):
       Resize a (resize_to, resize_to) -> CenterCrop size -> [0,1] -> norma ImageNet."""
    resize_to = round(size * RESIZE_RATIO)              # 255 si size=224
    off = (resize_to - size) // 2                       # 15 para 255->224
    img = Image.open(img_path).convert("RGB").resize((resize_to, resize_to), Image.BILINEAR)
    img = img.crop((off, off, off + size, off + size))  # recorte central a size x size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_samples(test_csv):
    samples = []
    with open(test_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append({
                "path": row["filepath"],
                "label": int(row["label"]),
                "source": row.get("source", "unknown"),
            })
    return samples


def build_session(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    shp = sess.get_inputs()[0].shape
    size = shp[-1] if isinstance(shp[-1], int) and shp[-1] > 0 else 224
    return sess, name, size


def warmup(sess, name, size, n=50):
    """Se ejecuta ANTES del reset del medidor: NO debe entrar en la ventana medida."""
    dummy = np.random.rand(1, 3, size, size).astype(np.float32)
    for _ in range(n):
        sess.run(None, {name: dummy})


def preprocess_all(samples, size):
    """Cachea los tensores ya preprocesados ANTES del reset del medidor.
       ~451 MB para 750 imagenes float32 (cabe en RPi 4)."""
    cache = []
    for s in samples:
        cache.append((preprocess(s["path"], size), s["label"]))
    return cache


def timed_inference(sess, name, cache, repeat=1):
    """VENTANA MEDIDA: solo sess.run sobre tensores cacheados."""
    lat = []
    correct = 0
    total = 0
    t0 = time.time()
    for r in range(repeat):
        for x, label in cache:
            a = time.perf_counter()
            out = sess.run(None, {name: x})[0]
            b = time.perf_counter()
            lat.append((b - a) * 1000)
            total += 1
            if r == 0 and int(np.argmax(out[0])) == label:
                correct += 1
    wall = time.time() - t0
    return {
        "n": len(cache),
        "repeat": repeat,
        "total_inferences": total,
        "wall_time_s": round(wall, 2),
        "lat_mean_ms": round(statistics.mean(lat), 3),
        "accuracy": round(correct / len(cache), 4),
    }


def idle_phase(seconds):
    print("\n" + "#" * 64)
    print(f"# FASE IDLE ({seconds}s): la RPi esta EN REPOSO.")
    print("# >>> LEE la POTENCIA EN REPOSO (W) en el medidor KWS y anotala.")
    print("#" * 64)
    for r in range(seconds, 0, -1):
        print(f"\r  Reposo... {r:2d}s restantes ", end="", flush=True)
        time.sleep(1)
    print("\r  Reposo completado.            ")


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def measure_model(onnx_path, samples, idle_s, repeat=1, out_dir=None):
    name = Path(onnx_path).name
    print("\n" + "=" * 64)
    print(f"MODELO: {name}   (repeticiones del test: {repeat})")
    print("=" * 64)

    # --- TODO LO SIGUIENTE es PREVIO al reset del medidor (no se mide) --------
    sess, inp_name, size = build_session(onnx_path)
    print(f"  Preparando: warm-up (50) + preprocesado de {len(samples)} imagenes...")
    warmup(sess, inp_name, size, n=50)
    cache = preprocess_all(samples, size)
    print("  Listo. (warm-up y preprocesado completados FUERA de la medida)")

    # 1. Idle
    idle_phase(idle_s)
    p_idle = ask("\n  Introduce POTENCIA EN REPOSO leida (W) [ej. 2.8]: ")

    # 2. Reset contador
    print("\n" + "*" * 64)
    print("* AHORA: RESETEA el contador de ENERGIA (mWh) del medidor KWS.")
    print("* (manten pulsado el boton hasta que la energia acumulada vuelva a 0)")
    print("*" * 64)
    ask("  Pulsa ENTER cuando el contador este a 0 y listo para empezar... ")

    # 3. VENTANA MEDIDA: solo inferencia
    total = len(cache) * repeat
    print(f"\n  >>> EJECUTANDO {total} inferencias de {name} (solo sess.run)...")
    print("  >>> OBSERVA la potencia (W) durante la corrida. NO toques la RPi.")
    res = timed_inference(sess, inp_name, cache, repeat=repeat)
    print(f"\n  Corrida terminada en {res['wall_time_s']} s "
          f"({res['total_inferences']} inferencias, "
          f"lat media {res['lat_mean_ms']} ms, acc {res['accuracy']})")

    del sess, cache
    gc.collect()

    # 4. Lecturas finales
    print("\n" + "*" * 64)
    print("* AHORA: LEE en el medidor KWS:")
    print("*   - ENERGIA ACUMULADA (mWh) de toda esta corrida")
    print("*   - POTENCIA MEDIA aproximada observada (W)")
    print("*" * 64)
    mwh = ask("  Energia acumulada TOTAL (mWh) [ej. 12.6]: ")
    p_load = ask("  Potencia media bajo carga (W) [ej. 5.1]: ")

    # --- Cálculo energético con TODOS los intermedios para trazabilidad --------
    base_idle_mwh = ""          # energía atribuible al reposo durante la corrida
    energy_neta_total_mwh = ""  # energía neta de inferencia (total - idle)
    energy_per_inf_mwh = ""     # bruta / inferencia
    energy_neta_mwh = ""        # neta / inferencia
    try:
        mwh_f = float(mwh)
        energy_per_inf_mwh = round(mwh_f / res["total_inferences"], 6)
        if p_idle:
            base_mwh = float(p_idle) * (res["wall_time_s"] / 3600.0) * 1000.0
            base_idle_mwh = round(base_mwh, 4)
            energy_neta_total_mwh = round(mwh_f - base_mwh, 4)
            energy_neta_mwh = round((mwh_f - base_mwh) / res["total_inferences"], 6)
    except Exception:
        pass

    row = {
        "model_file": name,
        "repeat": repeat,
        "n_per_pass": res["n"],
        "total_inferences": res["total_inferences"],
        "wall_time_s": res["wall_time_s"],
        "lat_mean_ms": res["lat_mean_ms"],
        "accuracy": res["accuracy"],
        "input_size": size,
        "P_idle_W": p_idle,
        "P_load_W": p_load,
        "energy_total_mWh": mwh,
        "base_idle_mWh": base_idle_mwh,
        "energy_neta_total_mWh": energy_neta_total_mwh,
        "energy_per_inf_mWh": energy_per_inf_mwh,
        "energy_neta_per_inf_mWh": energy_neta_mwh,
    }
    write_audit(out_dir, row)
    return row


def write_audit(out_dir, r):
    """Escribe un *_audit.txt por modelo con la fórmula y los números sustituidos,
    para que la energía sea verificable a mano desde el propio fichero."""
    if not out_dir:
        return
    stem = Path(r["model_file"]).stem
    path = Path(out_dir) / f"energy_audit_{stem}.txt"
    L = []
    L.append(f"AUDITORIA ENERGETICA  -  {r['model_file']}")
    L.append("=" * 60)
    L.append(f"Inferencias totales (N)      : {r['total_inferences']}  "
             f"({r['repeat']} x {r['n_per_pass']})")
    L.append(f"Tiempo de la ventana medida  : {r['wall_time_s']} s")
    L.append(f"Potencia en reposo (P_idle)  : {r['P_idle_W']} W")
    L.append(f"Potencia bajo carga (P_load) : {r['P_load_W']} W")
    L.append(f"Energia acumulada (medidor)  : {r['energy_total_mWh']} mWh")
    L.append("-" * 60)
    L.append("[1] Energia base de reposo durante la corrida:")
    L.append(f"    base_idle = P_idle * (wall_s / 3600) * 1000")
    L.append(f"              = {r['P_idle_W']} * ({r['wall_time_s']} / 3600) * 1000")
    L.append(f"              = {r['base_idle_mWh']} mWh")
    L.append("[2] Energia neta de inferencia (total - reposo):")
    L.append(f"    neta_total = {r['energy_total_mWh']} - {r['base_idle_mWh']} "
             f"= {r['energy_neta_total_mWh']} mWh")
    L.append("[3] Energia por inferencia:")
    L.append(f"    bruta/inf = {r['energy_total_mWh']} / {r['total_inferences']} "
             f"= {r['energy_per_inf_mWh']} mWh")
    L.append(f"    neta/inf  = {r['energy_neta_total_mWh']} / {r['total_inferences']} "
             f"= {r['energy_neta_per_inf_mWh']} mWh")
    L.append("=" * 60)
    L.append("Nota: el medidor (KWS-06C) solo entrega energia ACUMULADA de la")
    L.append("corrida; la energia por inferencia es necesariamente un promedio")
    L.append("(total / N), no atribuible a una imagen concreta. La latencia si es")
    L.append("por inferencia (cronometro por sess.run).")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  [auditoria] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--model", default=None, help="Un solo archivo .onnx")
    ap.add_argument("--all", action="store_true", help="Todos los modelos en secuencia")
    ap.add_argument("--idle-seconds", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repetir el test N veces para acumular mas mWh (modelos rapidos)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = load_samples(args.test_csv)
    missing = sum(1 for s in samples if not Path(s["path"]).exists())
    print(f"Imagenes de test: {len(samples)} (faltantes: {missing})")
    if missing:
        print("ADVERTENCIA: faltan imagenes. Corrige antes de medir energia.")
        return

    models_dir = Path(args.models_dir)
    if args.model:
        targets = [models_dir / args.model]
    elif args.all:
        targets = sorted(models_dir.glob("*.onnx"))
    else:
        print("Especifica --model <archivo.onnx> o --all")
        return

    print("\nINSTRUCCIONES GENERALES:")
    print(" - El medidor KWS debe estar intercalado: fuente -> KWS -> RPi.")
    print(" - El warm-up y el preprocesado se hacen ANTES del reset (no se miden).")
    print(" - La ventana medida es SOLO inferencia (comparable con la latencia).")
    print(" - Mide ENERGIA ACUMULADA (mWh) por corrida. NO toques la RPi al medir.")

    results = []
    for t in targets:
        if not t.exists():
            print(f"  NO existe: {t}")
            continue
        results.append(measure_model(t, samples, args.idle_seconds,
                                     repeat=args.repeat,
                                     out_dir=str(Path(args.out).parent)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_file", "repeat", "n_per_pass", "total_inferences",
              "wall_time_s", "lat_mean_ms", "accuracy", "input_size",
              "P_idle_W", "P_load_W", "energy_total_mWh",
              "base_idle_mWh", "energy_neta_total_mWh",
              "energy_per_inf_mWh", "energy_neta_per_inf_mWh"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nResultados de energia guardados en: {out}")
    print("energy_neta_per_inf_mWh descuenta el consumo en reposo (idle).")
    print("La energia medida es de inferencia pura (warm-up y preprocesado excluidos).")


if __name__ == "__main__":
    main()
