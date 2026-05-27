"""
Generacion de figuras del TFM:
  - Curvas de entrenamiento (loss/acc/F1 por epoca)
  - Matrices de confusion por modelo
  - Matriz Edge: heatmap latencia / exactitud por (modelo, formato, dispositivo)
  - Pareto front: precision vs latencia
  - Barras comparativas por familia (CNN vs hibrido)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config


FAMILY_COLORS = {
    "CNN":             "#2196F3",
    "Hibrido_Attn":    "#FF5722",
}


def plot_training_curves(metrics_csv, save_path=None):
    """Curvas de loss y F1-macro por epoca a partir del CSV de entrenamiento."""
    df = pd.read_csv(metrics_csv)
    model_key = Path(metrics_csv).stem.replace("train_", "")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train", linewidth=2)
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val",   linewidth=2)
    axes[0].set(xlabel="Epoca", ylabel="Loss", title=f"{model_key} - Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"]*100, label="Train", linewidth=2)
    axes[1].plot(df["epoch"], df["val_acc"]*100,   label="Val",   linewidth=2)
    axes[1].set(xlabel="Epoca", ylabel="Accuracy (%)",
                title=f"{model_key} - Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(df["epoch"], df["val_f1_macro"], linewidth=2, color="purple")
    axes[2].set(xlabel="Epoca", ylabel="F1-macro",
                title=f"{model_key} - F1-macro validacion")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, model_key, save_path=None, normalize=True):
    """Heatmap de matriz de confusion."""
    cm = np.array(cm)
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt, data = ".2f", cm_norm
    else:
        fmt, data = "d", cm

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES, ax=ax,
    )
    ax.set(xlabel="Predicho", ylabel="Real",
           title=f"Matriz de confusion - {model_key}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_edge_matrix(results_csv, metric="lat_mean_ms", save_path=None,
                      cmap="YlOrRd_r", title=None):
    """
    Heatmap principal del TFM: filas=modelos, columnas=(dispositivo, formato).
    Por defecto muestra latencia media; cambiar metric para ver acc, f1, etc.
    """
    df = pd.read_csv(results_csv)
    df["combo"] = df["device"] + " / " + df["format"]

    pivot = df.pivot(index="model_key", columns="combo", values=metric)
    pivot = pivot.reindex(list(config.MODEL_REGISTRY.keys()))

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*1.6), 6))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax,
        cbar_kws={"label": metric},
    )
    if title is None:
        title = f"Matriz Edge - {metric}"
    ax.set(title=title, xlabel="dispositivo / formato", ylabel="modelo")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pareto(results_csv, save_path=None):
    """Pareto: latencia P95 (eje X) vs exactitud (eje Y) por combinacion."""
    df = pd.read_csv(results_csv)

    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"rpi": "o", "jetson": "^"}

    for _, row in df.iterrows():
        color = FAMILY_COLORS.get(row["family"], "gray")
        marker = markers.get(row["device"], "s")
        ax.scatter(
            row["lat_p95_ms"], row["acc"] * 100,
            s=180, color=color, marker=marker,
            edgecolors="black", linewidth=1.2,
        )
        label = f"{row['model_key']}\n{row['format']}"
        ax.annotate(
            label, (row["lat_p95_ms"], row["acc"] * 100),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )

    ax.set(xlabel="Latencia P95 (ms)", ylabel="Exactitud (%)",
           title="Trade-off Edge: latencia P95 vs exactitud")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=FAMILY_COLORS["CNN"], markersize=12,
               markeredgecolor="black", label="CNN - RPi"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=FAMILY_COLORS["Hibrido_Attn"], markersize=12,
               markeredgecolor="black", label="Hibrido - RPi"),
        Line2D([0],[0], marker="^", color="w",
               markerfacecolor=FAMILY_COLORS["CNN"], markersize=12,
               markeredgecolor="black", label="CNN - Jetson"),
        Line2D([0],[0], marker="^", color="w",
               markerfacecolor=FAMILY_COLORS["Hibrido_Attn"], markersize=12,
               markeredgecolor="black", label="Hibrido - Jetson"),
    ]
    ax.legend(handles=legend_items, loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_size_vs_accuracy(results_csv, save_path=None):
    """Tamano del modelo vs exactitud."""
    df = pd.read_csv(results_csv)

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in df.iterrows():
        if pd.isna(row.get("size_kb")):
            continue
        color = FAMILY_COLORS.get(row["family"], "gray")
        ax.scatter(
            row["size_kb"], row["acc"] * 100,
            s=180, color=color, marker="o",
            edgecolors="black", linewidth=1.2, alpha=0.8,
        )
        label = f"{row['model_key']}\n{row['device']}/{row['format']}"
        ax.annotate(
            label, (row["size_kb"], row["acc"] * 100),
            textcoords="offset points", xytext=(5, 4), fontsize=7,
        )
    ax.set(xlabel="Tamano del modelo (KB)", ylabel="Exactitud (%)",
           title="Tamano del modelo vs exactitud")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_family_bars(results_csv, metric="acc", save_path=None):
    """Barras agrupadas por modelo, coloreadas por familia."""
    df = pd.read_csv(results_csv)

    # Promedio por modelo a traves de combinaciones (formato + dispositivo)
    grp = df.groupby(["model_key", "family"], as_index=False).agg(
        val=(metric, "mean"),
        std=(metric, "std"),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [FAMILY_COLORS.get(f, "gray") for f in grp["family"]]
    vals = grp["val"]
    if metric in ("acc", "f1_macro"):
        vals = vals * 100
        ax.set_ylabel(f"{metric} (%)")
    else:
        ax.set_ylabel(metric)

    ax.bar(grp["model_key"], vals, color=colors,
           edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.set_title(f"Comparativa {metric} por modelo (promedio sobre la matriz Edge)")
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
