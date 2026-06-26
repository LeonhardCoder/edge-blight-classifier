#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcnemar_rpi.py
==============
Test de McNemar pareado + matrices de confusion sobre las predicciones por imagen
de la Raspberry Pi 4 (results/preds_rpi_per_image.csv).


SALIDA:
  - results/mcnemar_rpi.csv            (6 contrastes con b,c,p,Holm,ganador)
  - results/confusion_matrices_rpi.csv (matrices 3x3 en formato largo)
  - tabla + matrices por consola
"""

import os, sys, math, itertools
import numpy as np
import pandas as pd

PREDS_CSV = "./results/preds_rpi_per_image.csv"
OUT_MCN   = "./results/mcnemar_rpi.csv"
OUT_CM    = "./results/confusion_matrices_rpi.csv"
ALPHA     = 0.05
EXACT_THR = 25
N_CLASSES = 3
IDX_TO_CLASS = {0: "healthy", 1: "early_blight", 2: "late_blight"}


# ── McNemar (stdlib puro) ────────────────────────────────────────────────────
def mcnemar_exact_p(b, c):
    """p bilateral exacto bajo Binom(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return min(2.0 * tail, 1.0)


def chi2_sf_df1(x):
    """Supervivencia de chi-cuadrado con 1 g.l. = erfc(sqrt(x/2))."""
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar_test(b, c):
    n = b + c
    if n == 0:
        return {"n_disc": 0, "test": "n/a", "stat": float("nan"), "p": 1.0}
    if n < EXACT_THR:
        return {"n_disc": n, "test": "exacto", "stat": float("nan"),
                "p": mcnemar_exact_p(b, c)}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"n_disc": n, "test": "chi2_cc", "stat": stat, "p": chi2_sf_df1(stat)}


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def confusion(yt, yp, n=N_CLASSES):
    M = np.zeros((n, n), dtype=int)
    for t, p in zip(yt, yp):
        M[t, p] += 1
    return M


def main():
    if not os.path.isfile(PREDS_CSV):
        sys.exit(f"[ERROR] No existe {PREDS_CSV}")
    df = pd.read_csv(PREDS_CSV)
    need = {"image_id", "model", "precision", "y_true", "y_pred"}
    if not need.issubset(df.columns):
        sys.exit(f"[ERROR] faltan columnas {need - set(df.columns)}")

    # 1) Verificar FP16 == FP32 (justifica trabajar solo en FP32)
    print("== Verificacion FP16 vs FP32 ==")
    for m in sorted(df["model"].unique()):
        a = df[(df.model == m) & (df.precision == "FP32")].sort_values("image_id")["y_pred"].to_numpy()
        h = df[(df.model == m) & (df.precision == "FP16")].sort_values("image_id")["y_pred"].to_numpy()
        if len(a) and len(h):
            ok = np.array_equal(a, h)
            print(f"  {m:<26} FP16{'==' if ok else '!='}FP32  ({'identicas' if ok else 'DIFIEREN'})")
    print()

    fp32 = df[df.precision == "FP32"].copy()
    models = sorted(fp32["model"].unique())

    # Alinear por image_id
    ref_ids, ref_true, correct, ypred = None, None, {}, {}
    for m in models:
        g = fp32[fp32.model == m].sort_values("image_id")
        ids, yt, yp = g.image_id.to_numpy(), g.y_true.to_numpy(), g.y_pred.to_numpy()
        if ref_ids is None:
            ref_ids, ref_true = ids, yt
        elif not np.array_equal(ids, ref_ids) or not np.array_equal(yt, ref_true):
            sys.exit(f"[ERROR] {m}: image_id/y_true no alinean con el resto.")
        correct[m] = (yp == yt)
        ypred[m] = yp

    # 2) Matrices de confusion + recall/precision por clase
    print("== Matrices de confusion (FP32) ==")
    cm_rows = []
    for m in models:
        M = confusion(ref_true, ypred[m])
        print(f"\n{m}  (filas=verdadera, columnas=predicha)")
        hdr = "        " + " ".join(f"{IDX_TO_CLASS[j][:9]:>10}" for j in range(N_CLASSES))
        print(hdr)
        for i in range(N_CLASSES):
            print(f"{IDX_TO_CLASS[i][:7]:>7} " + " ".join(f"{M[i, j]:>10d}" for j in range(N_CLASSES)))
        for i in range(N_CLASSES):
            rec = M[i, i] / M[i, :].sum() if M[i, :].sum() else float("nan")
            prc = M[i, i] / M[:, i].sum() if M[:, i].sum() else float("nan")
            print(f"    {IDX_TO_CLASS[i]:<14} recall={rec:.4f}  precision={prc:.4f}")
            for j in range(N_CLASSES):
                cm_rows.append({"model": m, "true": IDX_TO_CLASS[i],
                                "pred": IDX_TO_CLASS[j], "count": int(M[i, j])})
    pd.DataFrame(cm_rows).to_csv(OUT_CM, index=False)

    # 3) McNemar 6 contrastes
    print("\n== McNemar pareado (6 contrastes de arquitectura, FP32) ==")
    pairs = list(itertools.combinations(models, 2))
    res = []
    for A, B in pairs:
        b = int(np.sum(correct[A] & ~correct[B]))   # A acierta, B falla
        c = int(np.sum(~correct[A] & correct[B]))   # A falla, B acierta
        t = mcnemar_test(b, c)
        d_acc = float(np.mean(correct[A]) - np.mean(correct[B]))
        winner = A if b > c else (B if c > b else "empate")
        res.append({"A": A, "B": B, "b(A>B)": b, "c(B>A)": c,
                    "n_disc": t["n_disc"], "test": t["test"],
                    "p_raw": t["p"], "delta_acc": round(d_acc, 4), "ganador": winner})

    p_holm = holm([r["p_raw"] for r in res])
    for r, ph in zip(res, p_holm):
        r["p_holm"] = ph
        r["sig_0.05"] = "si" if ph < ALPHA else "no"
        r["p_raw"] = round(r["p_raw"], 5)
        r["p_holm"] = round(ph, 5)

    out = pd.DataFrame(res)[["A", "B", "b(A>B)", "c(B>A)", "n_disc", "test",
                             "p_raw", "p_holm", "sig_0.05", "ganador", "delta_acc"]]
    out.to_csv(OUT_MCN, index=False)
    print(out.to_string(index=False))
    print(f"\n[OK] Guardado: {OUT_MCN} y {OUT_CM}")


if __name__ == "__main__":
    main()
