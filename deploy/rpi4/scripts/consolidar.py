#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_consolidated_rpi.py
=========================
Construye la TABLA CONSOLIDADA de la Raspberry Pi 4 a partir de los CSV reales.
Cada columna es trazable a su fichero de origen:

  - accuracy / F1-macro / recall por clase  -> CALCULADAS desde preds_rpi_per_image.csv
  - latencia / RAM / energía                -> LEÍDAS de los CSV de benchmark (no se tocan)

DISEÑO TOLERANTE: detecta los nombres de columna por alias e IMPRIME las columnas
reales de cada CSV. Si no reconoce una métrica, la deja vacía y avisa (no revienta).

SALIDA: results/consolidated_rpi.csv + tabla por consola.
DEPENDENCIAS: pandas, numpy (ya disponibles).
"""

import os, sys, glob, re
import numpy as np
import pandas as pd

# ─── CONFIG ─────────────────────────────────────────────────────────────────
PREDS_CSV   = "./results/preds_rpi_per_image.csv"     # esquema conocido (lo generamos)
HW_CSV      = "./results/edge_results_rpi.csv"        # latencia/RAM (y quizá energía)
ENERGY_GLOB = "./results/energy_rpi.csv"                # opcional: si la energía va aparte
OUT_CSV     = "./results/consolidated_rpi.csv"
N_CLASSES   = 3
IDX_TO_CLASS = {0: "healthy", 1: "early_blight", 2: "late_blight"}
FAMILY = {
    "mobilenetv4_conv_small": "CNN",
    "efficientnetv2_b0":      "CNN",
    "efficientformerv2_s0":   "Hibrido c/atencion",
    "repvit_m1":              "Hibrido s/atencion",
}
# Alias de columnas en los CSV de hardware (minúsculas)
A_MODEL = ["model_file", "model", "artifact", "name", "engine", "modelo"]
A_PREC  = ["precision", "prec", "dtype", "precisión"]
A_LAT   = ["lat_mean_ms", "latency_ms", "lat_ms", "latency", "mean_latency_ms", "latencia_ms"]
#A_RAM   = ["ram_mb", "ram", "mem_mb", "memory_mb", "rss_mb", "peak_ram_mb", "ram_pico_mb"]
#A_FPS   = ["fps", "throughput", "ips"]
#A_ENER  = ["e_net_mj", "energy_net_mj", "energy_neta_mj", "e_neta_mj", "mj_per_inf",
#           "energy_per_inf_mj", "energy_mj", "mj", "energia_neta_mj", "e_net_mj_inf"]
A_RAM   = ["ram_mb", "ram", "mem_mb", "memory_mb", "rss_mb", "peak_ram_mb",
           "ram_pico_mb", "peak_rss_mb"]

A_FPS   = ["fps", "throughput", "ips", "throughput_fps"]

A_ENER  = ["e_net_mj", "energy_net_mj", "energy_neta_mj", "e_neta_mj", "mj_per_inf",
           "energy_per_inf_mj", "energy_mj", "mj", "energia_neta_mj", "e_net_mj_inf",
           "energy_neta_per_inf_mwh"]
# ────────────────────────────────────────────────────────────────────────────


def report_columns(path):
    if not os.path.isfile(path):
        print(f"  [falta] {path}"); return None
    df = pd.read_csv(path)
    print(f"  [{os.path.basename(path)}] filas={len(df)} columnas={list(df.columns)}")
    return df


def find_col(df, aliases):
    cols = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
    return None


def parse_artifact(raw):
    """De un identificador de hardware -> (model_key_canonico, precision)."""
    s = str(raw).strip()
    s = re.sub(r"\.(onnx|engine|plan)$", "", s, flags=re.I)
    low = s.lower()
    prec = "FP16" if "fp16" in low else ("INT8" if "int8" in low else "FP32")
    base = re.sub(r"[_\-]?(fp16|fp32|int8)", "", low)
    base = base.strip("_- ")
    # Casar con una clave canónica conocida
    for key in FAMILY:
        if key in base or base in key or key.replace("_", "") in base.replace("_", ""):
            return key, prec
    return base, prec


# ── 1) MÉTRICAS PREDICTIVAS desde preds (cálculo, no copia) ──────────────────
def predictive_metrics(preds_csv):
    df = pd.read_csv(preds_csv)
    need = {"model", "precision", "y_true", "y_pred"}
    if not need.issubset(df.columns):
        sys.exit(f"[ERROR] {preds_csv} necesita {need}. Tiene {list(df.columns)}")
    out = []
    for (model, prec), g in df.groupby(["model", "precision"]):
        yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        acc = float(np.mean(yt == yp))
        f1s, recalls = [], {}
        for c in range(N_CLASSES):
            tp = np.sum((yp == c) & (yt == c))
            fp = np.sum((yp == c) & (yt != c))
            fn = np.sum((yp != c) & (yt == c))
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
            recalls[f"recall_{IDX_TO_CLASS[c]}"] = round(float(r), 4)
        out.append({"model": model, "precision": prec,
                    "accuracy": round(acc, 4), "f1_macro": round(float(np.mean(f1s)), 4),
                    **recalls})
    return pd.DataFrame(out)


# ── 2) MÉTRICAS DE HARDWARE desde los CSV de benchmark (lectura) ─────────────
def hardware_metrics(hw_csv, energy_glob):
    df = report_columns(hw_csv)
    if df is None:
        return pd.DataFrame()
    c_model = find_col(df, A_MODEL)
    if c_model is None:
        print(f"  [aviso] no encuentro columna de modelo en {hw_csv}; alias={A_MODEL}")
        return pd.DataFrame()
    c_prec = find_col(df, A_PREC)
    c_lat, c_ram, c_fps, c_ener = (find_col(df, A) for A in (A_LAT, A_RAM, A_FPS, A_ENER))
    print(f"  [cols hw] modelo='{c_model}' precision='{c_prec}' "
          f"lat='{c_lat}' ram='{c_ram}' fps='{c_fps}' energia='{c_ener}'")

    # Factor de conversión: si la columna de energía viene en mWh, pasar a mJ.
    # 1 mWh = 3.6 J = 3600 mJ
    def energy_to_mj(value, col_name):
        if "mwh" in col_name.lower():
            return value * 3600.0
        return value

    rows = []
    for _, r in df.iterrows():
        if c_prec and pd.notna(r[c_prec]):
            model, _ = parse_artifact(r[c_model])
            prec = str(r[c_prec]).upper().replace("FLOAT", "FP")
            if prec not in ("FP32", "FP16", "INT8"):
                model, prec = parse_artifact(r[c_model])
        else:
            model, prec = parse_artifact(r[c_model])

        if c_ener and pd.notna(r[c_ener]):
            e_mj = energy_to_mj(float(r[c_ener]), c_ener)
        else:
            e_mj = np.nan

        rows.append({
            "model": model, "precision": prec,
            "latency_ms": float(r[c_lat]) if c_lat and pd.notna(r[c_lat]) else np.nan,
            "ram_mb":     float(r[c_ram]) if c_ram and pd.notna(r[c_ram]) else np.nan,
            "fps":        float(r[c_fps]) if c_fps and pd.notna(r[c_fps]) else np.nan,
            "energy_mj":  e_mj,
        })
    hw = pd.DataFrame(rows)

    # Energía aparte, solo si no vino en el HW principal
    if hw["energy_mj"].isna().all():
        files = sorted(glob.glob(energy_glob))
        if files:
            print(f"  [energia] leyendo {len(files)} ficheros de {energy_glob}")
            emap = {}
            for f in files:
                edf = report_columns(f)
                if edf is None or len(edf) == 0:
                    continue
                cm = find_col(edf, A_MODEL) or edf.columns[0]
                ce = find_col(edf, A_ENER)
                if ce is None:
                    print(f"    [aviso] sin columna de energia en {os.path.basename(f)}")
                    continue
                convert = "mwh" in ce.lower()
                if convert:
                    print(f"    [energia] columna '{ce}' en mWh -> convirtiendo a mJ (x3600)")
                for _, er in edf.iterrows():
                    if pd.notna(er[ce]):
                        val = float(er[ce])
                        if convert:
                            val *= 3600.0
                        emap[parse_artifact(er[cm])] = val
            hw["energy_mj"] = hw.apply(
                lambda x: emap.get((x["model"], x["precision"]), x["energy_mj"]), axis=1)
    return hw

def main():
    print("== Inspección de CSV de entrada ==")
    pred = predictive_metrics(PREDS_CSV)
    print(f"  [preds] artefactos con métricas predictivas: {len(pred)}")
    print("== Hardware ==")
    hw = hardware_metrics(HW_CSV, ENERGY_GLOB)

    if hw.empty:
        print("\n[AVISO] Sin métricas de hardware: la tabla saldrá solo con predictivas.")
        merged = pred.copy()
        for c in ["latency_ms", "ram_mb", "fps", "energy_mj"]:
            merged[c] = np.nan
    else:
        merged = pred.merge(hw, on=["model", "precision"], how="outer")

    # Avisar de desajustes de emparejamiento
    miss_hw = merged[merged["latency_ms"].isna()][["model", "precision"]].values.tolist()
    miss_pr = merged[merged["accuracy"].isna()][["model", "precision"]].values.tolist()
    if miss_hw: print(f"  [aviso] sin hardware: {miss_hw}")
    if miss_pr: print(f"  [aviso] sin predictivas: {miss_pr}")

    merged["family"] = merged["model"].map(FAMILY).fillna("?")
    merged["_pord"] = merged["precision"].map({"FP32": 0, "FP16": 1, "INT8": 2}).fillna(9)
    cols = ["model", "family", "precision", "accuracy", "f1_macro",
            "recall_healthy", "recall_early_blight", "recall_late_blight",
            "latency_ms", "ram_mb", "fps", "energy_mj"]
    merged = merged.sort_values(["model", "_pord"]).reindex(columns=cols)

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 78)
    print("TABLA CONSOLIDADA — RASPBERRY PI 4")
    print("=" * 78)
    print(merged.to_string(index=False))
    print("\nORIGEN POR COLUMNA:")
    print("  accuracy, f1_macro, recall_*  <- CALCULADO de", PREDS_CSV)
    print("  latency_ms, ram_mb, fps       <- LEÍDO de", HW_CSV)
    print("  energy_mj                     <- LEÍDO de", HW_CSV, "o", ENERGY_GLOB)
    print(f"\n[OK] Guardado: {OUT_CSV}")
    if merged[["latency_ms", "ram_mb", "energy_mj"]].isna().any().any():
        print("\n[!] Hay celdas vacías: pásame el head -1 del CSV correspondiente y")
        print("    añado el nombre real a los alias (A_LAT/A_RAM/A_ENER).")


if __name__ == "__main__":
    main()
