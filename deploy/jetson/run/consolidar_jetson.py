#!/usr/bin/env python3
"""Consolidada los resultados de Jetson.
Lee los cls_*.json (exactitud) y jetson_*.json (latencia/energia)."""
import json, glob, os, re
import pandas as pd

RESULTS = "/home/victor/tfm-edge/results"
OUT = f"{RESULTS}/consolidated_jetson.csv"
FAMILY = {
    "mobilenetv4_conv_small": "CNN",
    "efficientnetv2_b0":      "CNN",
    "efficientformerv2_s0":   "Hibrido c/atencion",
    "repvit_m1":              "Hibrido s/atencion",
}

def parse(stem):
    low = stem.lower()
    prec = "FP16" if "fp16" in low else ("INT8" if "int8" in low else "FP32")
    base = re.sub(r"[_\-]?(fp16|fp32|int8)", "", low).strip("_- ")
    base = base.replace("jetson_", "").replace("cls_", "")
    return base, prec

rows = {}
# Exactitud desde cls_*.json
for f in glob.glob(f"{RESULTS}/cls_jetson_*.json"):
    stem = os.path.basename(f).replace("cls_jetson_", "").replace(".json", "")
    model, prec = parse(stem)
    d = json.load(open(f))
    rows[(model, prec)] = {
        "model": model, "precision": prec,
        "accuracy": round(d["accuracy"], 4),
        "f1_macro": round(d["f1_macro"], 4),
        "recall_healthy": round(d["recall_per_class"]["healthy"], 4),
        "recall_early_blight": round(d["recall_per_class"]["early_blight"], 4),
        "recall_late_blight": round(d["recall_per_class"]["late_blight"], 4),
    }
# Latencia/energia desde jetson_*.json
for f in glob.glob(f"{RESULTS}/jetson_*.json"):
    stem = os.path.basename(f).replace("jetson_", "").replace(".json", "")
    model, prec = parse(stem)
    d = json.load(open(f))
    r = rows.setdefault((model, prec), {"model": model, "precision": prec})
    r["latency_ms"] = round(d["mean_ms"], 3)
    r["ram_mb"]     = round(d["rss_peak_mb"], 1)
    r["fps"]        = round(d["fps"], 1)
    r["energy_mj"]  = round(d["energy_net_j_per_inf"] * 1000, 4)  # J -> mJ

df = pd.DataFrame(rows.values())
df["family"] = df["model"].map(FAMILY).fillna("?")
df["_p"] = df["precision"].map({"FP32":0,"FP16":1,"INT8":2}).fillna(9)
cols = ["model","family","precision","accuracy","f1_macro",
        "recall_healthy","recall_early_blight","recall_late_blight",
        "latency_ms","ram_mb","fps","energy_mj"]
df = df.sort_values(["model","_p"]).reindex(columns=cols)
df.to_csv(OUT, index=False)
print(df.to_string(index=False))
print(f"\n[OK] {OUT}")