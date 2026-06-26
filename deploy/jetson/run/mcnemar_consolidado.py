"""mcnemar_consolidado.py — McNemar pareado sobre el CSV consolidado."""
import pandas as pd
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar

df = pd.read_csv("/home/victor/tfm-edge/results/preds_jetson.csv")  # ajusta

models = ["efficientformerv2_s0_fp16","efficientformerv2_s0_fp32","efficientformerv2_s0_int8",
          "efficientnetv2_b0_fp16","efficientnetv2_b0_fp32","efficientnetv2_b0_int8",
          "mobilenetv4_conv_small_fp16","mobilenetv4_conv_small_fp32","mobilenetv4_conv_small_int8",
          "repvit_m1_fp16","repvit_m1_fp32","repvit_m1_int8"]

# Acierto/fallo por imagen
correct = {m: (df[m] == df["y_true"]) for m in models}

# Accuracy global de cada modelo (para la tabla del capítulo)
print("=== Accuracy por modelo (preprocesado correcto) ===")
for m in models:
    print(f"  {m:<32} {correct[m].mean():.4f}")

# Pares de interés para las hipótesis (NO los 66)
pares = [
    ("efficientformerv2_s0_fp32","efficientformerv2_s0_int8"),  # ¿INT8 degrada el híbrido-atención?
    ("repvit_m1_fp32","repvit_m1_int8"),                         # ¿INT8 degrada el híbrido sin atención?
    ("efficientnetv2_b0_fp32","efficientnetv2_b0_int8"),         # ¿INT8 degrada CNN?
    ("mobilenetv4_conv_small_fp32","mobilenetv4_conv_small_int8"),
    ("repvit_m1_fp16","mobilenetv4_conv_small_fp16"),            # híbrido vs CNN (igual precisión)
    ("efficientformerv2_s0_fp16","efficientnetv2_b0_fp16"),      # híbrido vs CNN
]

print(f"\n{'A':<30}{'B':<30}{'b':>4}{'c':>4}{'p':>9}")
for a, b in pares:
    n01 = int((~correct[a] &  correct[b]).sum())
    n10 = int(( correct[a] & ~correct[b]).sum())
    res = mcnemar([[0, n01],[n10, 0]], exact=(n01+n10) < 25)
    sig = "*" if res.pvalue < 0.05 else ""
    print(f"{a:<30}{b:<30}{n01:>4}{n10:>4}{res.pvalue:>9.4f}{sig}")