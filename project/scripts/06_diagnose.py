"""
Diagnostico del conjunto de test: evalua los 4 modelos en test.csv
desglosando la exactitud y F1 por dataset de origen.

Responde a la pregunta: las metricas globales altas (~99.8%) reflejan
buen rendimiento en todos los dominios o estan dominadas por PlantVillage?

Salida:
  outputs/metrics/diagnose_test_by_dataset.csv
  outputs/metrics/diagnose_classification_reports.txt

Uso:
    python scripts/06_diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from tqdm import tqdm

import config
from src.dataset import BlightDataset
from src.models import create_model
from src.transforms import get_eval_transforms
from torch.utils.data import DataLoader


def evaluate_model_full(model, loader, device):
    """Devuelve y_true, y_pred sobre todo el DataLoader."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  eval", leave=False):
            images = images.to(device, non_blocking=True)
            out = model(images)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)


def main():
    config.ensure_dirs()
    test_csv = config.CORPUS_DIR / "test.csv"
    if not test_csv.exists():
        raise SystemExit(f"No existe {test_csv}. Ejecuta 01_prepare_corpus.py")

    df_test = pd.read_csv(test_csv)
    print(f"Test: {len(df_test)} imagenes")
    print(f"Por origen:")
    print(df_test["source"].value_counts().to_string())

    device   = config.DEVICE
    val_tf   = get_eval_transforms(config.TRAINING["img_size"])
    test_ds  = BlightDataset(test_csv, val_tf)
    test_dl  = DataLoader(
        test_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    rows = []
    reports = []

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

        y_true, y_pred = evaluate_model_full(model, test_dl, device)

        # ----- Global -----
        acc_global = accuracy_score(y_true, y_pred)
        f1_global  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_per_cls = f1_score(y_true, y_pred, average=None,
                              labels=list(range(config.NUM_CLASSES)),
                              zero_division=0)

        print(f"  GLOBAL  acc={acc_global:.4f}  f1_macro={f1_global:.4f}")
        rows.append({
            "model_key": model_key,
            "family":    config.MODEL_REGISTRY[model_key]["family"],
            "scope":     "global",
            "n":         len(y_true),
            "acc":       acc_global,
            "f1_macro":  f1_global,
            "f1_healthy":     f1_per_cls[0],
            "f1_early_blight":f1_per_cls[1],
            "f1_late_blight": f1_per_cls[2],
        })

        # ----- Por dataset (source) -----
        df_test_copy = df_test.copy().reset_index(drop=True)
        df_test_copy["y_pred"] = y_pred

        for src in sorted(df_test_copy["source"].unique()):
            mask = df_test_copy["source"] == src
            yt = df_test_copy.loc[mask, "label"].values
            yp = df_test_copy.loc[mask, "y_pred"].values
            if len(yt) == 0:
                continue
            acc_src = accuracy_score(yt, yp)
            f1_src  = f1_score(yt, yp, average="macro", zero_division=0)
            f1_cls  = f1_score(yt, yp, average=None,
                               labels=list(range(config.NUM_CLASSES)),
                               zero_division=0)
            print(f"  {src:<14} n={len(yt):>4}  "
                  f"acc={acc_src:.4f}  f1_macro={f1_src:.4f}")
            rows.append({
                "model_key": model_key,
                "family":    config.MODEL_REGISTRY[model_key]["family"],
                "scope":     src,
                "n":         len(yt),
                "acc":       acc_src,
                "f1_macro":  f1_src,
                "f1_healthy":     f1_cls[0],
                "f1_early_blight":f1_cls[1],
                "f1_late_blight": f1_cls[2],
            })

        # Reporte de clasificacion completo
        cls_report = classification_report(
            y_true, y_pred,
            target_names=config.CLASS_NAMES, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred,
                              labels=list(range(config.NUM_CLASSES)))
        reports.append(
            f"\n{'=' * 70}\n{model_key}\n{'=' * 70}\n"
            f"{cls_report}\nMatriz de confusion (filas=real, cols=pred):\n"
            f"{cm}\n"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Guardar resultados
    df_out = pd.DataFrame(rows)
    out_csv = config.METRICS_DIR / "diagnose_test_by_dataset.csv"
    df_out.to_csv(out_csv, index=False)

    out_txt = config.METRICS_DIR / "diagnose_classification_reports.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(reports)

    # Resumen final en consola
    print("\n" + "=" * 78)
    print("RESUMEN: exactitud por (modelo, dataset)")
    print("=" * 78)
    pivot = df_out.pivot(index="model_key", columns="scope", values="acc")
    print(pivot.round(4).to_string())
    print("=" * 78)
    print(f"\nResultados completos: {out_csv}")
    print(f"Reportes detallados:  {out_txt}")


if __name__ == "__main__":
    main()