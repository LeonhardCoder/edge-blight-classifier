"""
Benchmark en dispositivo Edge.

Este script se ejecuta DENTRO de cada dispositivo Edge (RPi4 o Jetson)
tras transferir los modelos exportados desde la maquina de entrenamiento.

Lee la configuracion del dispositivo activo (TFM_ENV=edge_rpi o edge_jetson),
itera sobre los formatos definidos en settings.conf, y para cada combinacion
(modelo, formato) mide exactitud, latencia y memoria.

Salida:
  outputs/metrics/edge_results.csv      <- una fila por combinacion
  outputs/metrics/edge_results_<env>.csv (nombre con el entorno)

Uso (en cada dispositivo Edge):
    export TFM_ENV=edge_rpi
    python scripts/04_edge_benchmark.py

    export TFM_ENV=edge_jetson
    python scripts/04_edge_benchmark.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

import config
from src.edge_bench import benchmark_combination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--formats", nargs="+", default=None,
                        help="Sobrescribe la lista del settings.conf")
    parser.add_argument("--test-csv", default=None,
                        help="CSV de test (por defecto outputs/corpus/test.csv)")
    parser.add_argument("--device-label", default=None,
                        help="Etiqueta del dispositivo (rpi/jetson). "
                             "Por defecto se infiere del entorno.")
    args = parser.parse_args()

    config.ensure_dirs()

    # Determinar etiqueta de dispositivo
    if args.device_label:
        device_label = args.device_label
    elif config.ENV == "edge_rpi":
        device_label = "rpi"
    elif config.ENV == "edge_jetson":
        device_label = "jetson"
    else:
        # Permitir corrida en local para pruebas (con onnx por ejemplo)
        device_label = "local_dev"

    # Determinar formatos a probar
    if args.formats is not None:
        formats = args.formats
    elif device_label == "rpi":
        formats = config.EDGE_MATRIX["rpi"]
    elif device_label == "jetson":
        formats = config.EDGE_MATRIX["jetson"]
    else:
        formats = ["onnx_fp32"]   # solo para pruebas en local

    models = args.models or list(config.MODEL_REGISTRY.keys())

    test_csv = args.test_csv or (config.CORPUS_DIR / "test.csv")
    if not Path(test_csv).exists():
        raise SystemExit(f"No existe test_csv: {test_csv}. "
                         f"Ejecuta primero 01_prepare_corpus.py")

    print("=" * 70)
    print(f"EDGE BENCHMARK | dispositivo: {device_label} | entorno: {config.ENV}")
    print("=" * 70)
    print(f"Modelos:  {models}")
    print(f"Formatos: {formats}")
    print(f"Test:     {test_csv}")
    print(f"Iters:    warmup={config.EDGE_BENCHMARK['warmup_iters']}, "
          f"medidas={config.EDGE_BENCHMARK['measured_iters']}")

    results = []
    for model_key in models:
        for fmt in formats:
            try:
                row = benchmark_combination(
                    model_key, fmt, str(test_csv), device_label,
                )
                results.append(row)
            except FileNotFoundError as e:
                print(f"  [SKIP] {model_key}/{fmt}: {e}")
            except Exception as e:
                print(f"  [FAIL] {model_key}/{fmt}: {type(e).__name__}: {e}")

    if not results:
        print("\nNo se obtuvieron resultados. Verifica que los modelos exportados existan.")
        return

    df = pd.DataFrame(results)
    out_csv = config.METRICS_DIR / f"edge_results_{device_label}.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print(f"RESULTADOS GUARDADOS: {out_csv}")
    print("=" * 70)
    cols = ["model_key", "format", "acc", "f1_macro",
            "lat_mean_ms", "lat_p95_ms", "mem_peak_mb", "size_kb"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
