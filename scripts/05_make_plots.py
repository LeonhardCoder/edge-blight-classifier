"""
Genera las figuras del TFM.

Lee:
  outputs/metrics/train_*.csv       <- curvas de entrenamiento
  outputs/metrics/edge_results_*.csv <- benchmark Edge

Produce en outputs/figures/:
  train_<model>.png                  <- loss, acc y F1 por epoca
  cm_<model>.png                     <- matriz de confusion
  edge_matrix_<metric>.png           <- heatmap por metrica
  pareto_lat_vs_acc.png              <- frontera Pareto
  size_vs_acc.png                    <- tamano vs exactitud
  family_bars_<metric>.png           <- barras CNN vs hibrido

Uso:
    python scripts/05_make_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json

import pandas as pd
import torch

import config
from src.plots import (
    plot_training_curves, plot_confusion_matrix,
    plot_edge_matrix, plot_pareto, plot_size_vs_accuracy,
    plot_family_bars,
)


def main():
    config.ensure_dirs()
    fig_dir = config.FIGURES_DIR
    metrics_dir = config.METRICS_DIR

    print("=" * 70)
    print("GENERACION DE FIGURAS")
    print("=" * 70)

    # 1) Curvas de entrenamiento
    print("\n[1] Curvas de entrenamiento")
    for csv_path in sorted(metrics_dir.glob("train_*.csv")):
        model_key = csv_path.stem.replace("train_", "")
        out = fig_dir / f"train_{model_key}.png"
        plot_training_curves(csv_path, save_path=out)
        print(f"  {out}")

    # 2) Matrices de confusion (desde checkpoints)
    print("\n[2] Matrices de confusion")
    for model_key in config.MODEL_REGISTRY:
        ckpt_path = config.CHECKPOINTS_DIR / f"{model_key}_best.pth"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cm = ckpt.get("cm")
        if cm is None:
            continue
        out = fig_dir / f"cm_{model_key}.png"
        plot_confusion_matrix(cm, model_key, save_path=out)
        print(f"  {out}")

    # 3) Consolidar resultados Edge (rpi + jetson) en uno solo
    print("\n[3] Consolidando resultados Edge")
    edge_dfs = []
    for csv_path in sorted(metrics_dir.glob("edge_results_*.csv")):
        edge_dfs.append(pd.read_csv(csv_path))
        print(f"  + {csv_path.name}")

    if not edge_dfs:
        print("  No hay resultados Edge todavia.")
        return

    df_edge = pd.concat(edge_dfs, ignore_index=True)
    consolidated = metrics_dir / "edge_results_all.csv"
    df_edge.to_csv(consolidated, index=False)
    print(f"  Consolidado: {consolidated}")

    # 4) Heatmaps de la matriz Edge para varias metricas
    print("\n[4] Matriz Edge (heatmaps)")
    for metric, cmap, title in [
        ("lat_mean_ms",   "YlOrRd",  "Latencia media (ms) - menor es mejor"),
        ("lat_p95_ms",    "YlOrRd",  "Latencia P95 (ms) - menor es mejor"),
        ("acc",           "Greens",  "Exactitud - mayor es mejor"),
        ("f1_macro",      "Greens",  "F1-macro - mayor es mejor"),
        ("mem_peak_mb",   "Purples", "Memoria pico (MB) - menor es mejor"),
        ("throughput_fps","Blues",   "Throughput (FPS) - mayor es mejor"),
        ("size_kb",       "Greys",   "Tamano (KB) - menor es mejor"),
    ]:
        if metric not in df_edge.columns:
            continue
        out = fig_dir / f"edge_matrix_{metric}.png"
        plot_edge_matrix(consolidated, metric=metric, save_path=out,
                          cmap=cmap, title=title)
        print(f"  {out}")

    # 5) Pareto y graficas adicionales
    print("\n[5] Pareto y graficas comparativas")
    plot_pareto(consolidated, save_path=fig_dir / "pareto_lat_vs_acc.png")
    print(f"  pareto_lat_vs_acc.png")

    plot_size_vs_accuracy(consolidated, save_path=fig_dir / "size_vs_acc.png")
    print(f"  size_vs_acc.png")

    for m in ("acc", "f1_macro", "lat_mean_ms"):
        out = fig_dir / f"family_bars_{m}.png"
        plot_family_bars(consolidated, metric=m, save_path=out)
        print(f"  {out}")

    print("\n" + "=" * 70)
    print(f"Todas las figuras en: {fig_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
