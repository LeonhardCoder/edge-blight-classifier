"""
Analisis de errores de clasificacion en el conjunto de test.

Genera:
  outputs/metrics/error_analysis_by_model.csv     <- FP/FN/TP/TN por clase
  outputs/metrics/error_analysis_misclassified.csv <- rutas de imgs mal clasificadas
  outputs/figures/cm_<model>.png                  <- matriz de confusion visual
  outputs/figures/cm_<model>_<dataset>.png        <- matriz por dataset

Uso:
    python scripts/07_error_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import BlightDataset
from src.models import create_model
from src.transforms import get_eval_transforms


def predict_all(model, loader, device):
    """Devuelve (y_true, y_pred) sobre todo el loader."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  eval", leave=False):
            images = images.to(device, non_blocking=True)
            out = model(images)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)


def compute_per_class_metrics(cm: np.ndarray, class_names: list[str]):
    """
    Calcula TP, FP, FN, TN por clase a partir de la matriz de confusion.

    En multiclase:
        TP[c] = cm[c, c]
        FN[c] = sum(cm[c, :]) - TP[c]      (real c, predicho otra cosa)
        FP[c] = sum(cm[:, c]) - TP[c]      (real otra cosa, predicho c)
        TN[c] = sum(cm) - TP[c] - FN[c] - FP[c]
    """
    rows = []
    total = cm.sum()
    for c, name in enumerate(class_names):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(total - tp - fn - fp)

        # Metricas derivadas (con proteccion ante division por cero)
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = sensibilidad
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        rows.append({
            "class":       name,
            "TP":          tp,
            "FP":          fp,
            "FN":          fn,
            "TN":          tn,
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "specificity": round(specificity, 4),
            "f1":          round(f1, 4),
            "support":     int(cm[c, :].sum()),
        })
    return rows


def plot_cm(cm: np.ndarray, class_names: list[str], title: str, save_path):
    """Heatmap de matriz de confusion con conteos y porcentajes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        cbar_kws={"label": "n imagenes"},
    )
    ax.set(xlabel="Prediccion", ylabel="Etiqueta real", title=title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    config.ensure_dirs()
    test_csv = config.CORPUS_DIR / "test.csv"
    if not test_csv.exists():
        raise SystemExit(f"No existe {test_csv}")

    df_test = pd.read_csv(test_csv)
    print(f"Test: {len(df_test)} imagenes")
    print(f"Por origen: {df_test['source'].value_counts().to_dict()}")

    device  = config.DEVICE
    val_tf  = get_eval_transforms(config.TRAINING["img_size"])
    test_ds = BlightDataset(test_csv, val_tf)
    test_dl = DataLoader(
        test_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    all_metrics = []
    all_misclassified = []

    for model_key in config.MODEL_REGISTRY:
        ckpt_path = config.CHECKPOINTS_DIR / f"{model_key}_best.pth"
        if not ckpt_path.exists():
            print(f"[SKIP] {model_key}: no existe checkpoint")
            continue

        print(f"\n>>> {model_key}")

        model = create_model(model_key, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        y_true, y_pred = predict_all(model, test_dl, device)

        # --- 1. Matriz de confusion GLOBAL ---
        labels = list(range(config.NUM_CLASSES))
        cm_global = confusion_matrix(y_true, y_pred, labels=labels)

        print(f"  Matriz de confusion global:")
        print(f"  {cm_global}")

        # Visual
        plot_cm(
            cm_global, config.CLASS_NAMES,
            f"{model_key} - Test global (n={len(y_true)})",
            config.FIGURES_DIR / f"cm_{model_key}_global.png",
        )

        # Metricas por clase
        per_class = compute_per_class_metrics(cm_global, config.CLASS_NAMES)
        for row in per_class:
            row.update({"model": model_key, "scope": "global"})
            all_metrics.append(row)

        # --- 2. Por dataset ---
        df_pred = df_test.copy().reset_index(drop=True)
        df_pred["y_pred"] = y_pred
        df_pred["correct"] = df_pred["label"] == df_pred["y_pred"]

        for src in sorted(df_pred["source"].unique()):
            mask = df_pred["source"] == src
            yt = df_pred.loc[mask, "label"].values
            yp = df_pred.loc[mask, "y_pred"].values
            if len(yt) == 0:
                continue

            cm_src = confusion_matrix(yt, yp, labels=labels)
            plot_cm(
                cm_src, config.CLASS_NAMES,
                f"{model_key} - {src} (n={len(yt)})",
                config.FIGURES_DIR / f"cm_{model_key}_{src}.png",
            )

            per_class_src = compute_per_class_metrics(cm_src, config.CLASS_NAMES)
            for row in per_class_src:
                row.update({"model": model_key, "scope": src})
                all_metrics.append(row)

        # --- 3. Listado de imagenes MAL clasificadas ---
        mis = df_pred[~df_pred["correct"]].copy()
        mis["model"] = model_key
        mis["real_class"] = mis["label"].map(config.IDX_TO_CLASS)
        mis["pred_class"] = mis["y_pred"].map(config.IDX_TO_CLASS)
        mis["error_type"] = mis.apply(
            lambda r: f"{r['real_class']} -> {r['pred_class']}", axis=1,
        )

        cols = ["model", "source", "abs_path", "real_class",
                "pred_class", "error_type"]
        all_misclassified.append(mis[cols])

        print(f"  Errores totales: {len(mis)} / {len(df_pred)} "
              f"({len(mis)/len(df_pred)*100:.2f}%)")
        print(f"  Por tipo de error:")
        print(mis["error_type"].value_counts().to_string().replace("\n", "\n    "))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Guardar -----
    df_metrics = pd.DataFrame(all_metrics)
    out_metrics = config.METRICS_DIR / "error_analysis_by_model.csv"
    df_metrics.to_csv(out_metrics, index=False)
    print(f"\nMetricas FP/FN/TP/TN: {out_metrics}")

    if all_misclassified:
        df_mis = pd.concat(all_misclassified, ignore_index=True)
        out_mis = config.METRICS_DIR / "error_analysis_misclassified.csv"
        df_mis.to_csv(out_mis, index=False)
        print(f"Imagenes mal clasificadas: {out_mis}")

    # ----- Resumen final -----
    print("\n" + "=" * 78)
    print("RESUMEN FP / FN GLOBAL POR MODELO Y CLASE")
    print("=" * 78)
    df_global = df_metrics[df_metrics["scope"] == "global"]
    pivot_fn = df_global.pivot(index="model", columns="class", values="FN")
    pivot_fp = df_global.pivot(index="model", columns="class", values="FP")
    print("\nFalsos NEGATIVOS (real=clase, predicho=otra) - critico para enfermedad:")
    print(pivot_fn.to_string())
    print("\nFalsos POSITIVOS (real=otra, predicho=clase) - falsa alarma:")
    print(pivot_fp.to_string())
    print("=" * 78)


if __name__ == "__main__":
    main()