#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
generar_figuras.py  ── Genera las figuras del Capítulo 3 a partir de los CSV.
Salida en esta misma carpeta (figuras/), en PNG y PDF.

Figuras:
  1) matrices_confusion.{png,pdf}  <- results/confusion_matrices_rpi.csv
  2) pareto_latencia_f1.{png,pdf}  <- CSV consolidado (lat + F1 por artefacto)

Cada figura se regenera de forma independiente: si cambian los datos, se vuelve
a ejecutar este script y el capitulo (que la incluye con \includegraphics) se
actualiza sin tocar el .tex.

USO:  python generar_figuras.py --results ../../results
"""
import os, argparse  # noqa
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["healthy", "early_blight", "late_blight"]
NICE = {"mobilenetv4_conv_small": "MobileNetV4-Small",
        "efficientnetv2_b0": "EfficientNetV2-B0",
        "efficientformerv2_s0": "EfficientFormerV2-S0",
        "repvit_m1": "RepViT-M1"}
ORDER = ["mobilenetv4_conv_small", "efficientnetv2_b0",
         "efficientformerv2_s0", "repvit_m1"]


def fig_confusion(results_dir, out_dir):
    path = os.path.join(results_dir, "confusion_matrices_rpi.csv")
    if not os.path.isfile(path):
        print(f"[skip] no existe {path}"); return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, model in zip(axes.ravel(), ORDER):
        sub = df[df.model == model]
        M = np.zeros((3, 3))
        for _, r in sub.iterrows():
            M[CLASSES.index(r["true"]), CLASSES.index(r["pred"])] = r["count"]
        Mn = M / M.sum(axis=1, keepdims=True)        # normalizada por fila
        im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(NICE.get(model, model), fontsize=10)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel("Predicha", fontsize=8); ax.set_ylabel("Verdadera", fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                        color="white" if Mn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"matrices_confusion.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[ok] matrices_confusion")


def fig_pareto(results_dir, out_dir):
    """Espera un CSV con columnas: device, model, precision, latency_ms, f1_macro.
    Busca results/pareto_data.csv; si no existe, avisa."""
    path = os.path.join(results_dir, "pareto_data.csv")
    if not os.path.isfile(path):
        print(f"[skip] no existe {path}. Crea un CSV con columnas "
              f"device,model,precision,latency_ms,f1_macro (lat de edge_results_*, "
              f"F1 corregido).")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"RPi": "o", "Jetson": "s"}
    for dev, g in df.groupby("device"):
        ax.scatter(g["latency_ms"], g["f1_macro"], marker=markers.get(dev, "^"),
                   s=60, label=dev, alpha=0.8)
        for _, r in g.iterrows():
            ax.annotate(f"{r['model']}\n{r['precision']}",
                        (r["latency_ms"], r["f1_macro"]), fontsize=6,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Latencia media (ms, escala log)")
    ax.set_ylabel("F1-macro")
    ax.set_title("Frontera de Pareto: latencia frente a exactitud")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"pareto_latencia_f1.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[ok] pareto_latencia_f1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../../results",
                    help="carpeta con los CSV (confusion_matrices_rpi.csv, pareto_data.csv)")
    a = ap.parse_args()
    out = os.path.dirname(os.path.abspath(__file__))
    fig_confusion(a.results, out)
    fig_pareto(a.results, out)
    print("Figuras en:", out)
