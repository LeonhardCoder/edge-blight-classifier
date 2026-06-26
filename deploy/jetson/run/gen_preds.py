"""gen_preds.py — predicciones por imagen en Jetson (clasificación).

Regenera las predicciones de un engine con el preprocesado de evaluación
correcto. Resuelve el PENDIENTE del TFM: reverificar las métricas INT8/FP16/FP32
de la Jetson de forma coherente con la referencia. Métricas con scikit-learn.

Salida:
  results/preds_jetson_<engine>.csv   (path, source, y_true, y_pred)
  results/preds_jetson.csv            (ancho: una columna por engine)
  results/cls_jetson_<engine>.json

Uso:
  python run/gen_preds.py --engine engines/repvit_m1_int8.engine
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("/home/victor/tfm-edge")
sys.path.insert(0, str(ROOT))
from common.labels import CLASS_TO_IDX
from common.preprocess import preprocess_numpy
from common.evalmetrics import compute_metrics, subdomain_accuracy

sys.path.insert(0, str(ROOT / "run"))
from trt_infer import TRTRunner


def main():
    TEST_CSV  = Path("/home/victor/Documents/edge-blight-classifier/project/outputs/corpus/test.csv")

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--test", default=str(TEST_CSV))
    args = ap.parse_args()
    eng_key = Path(args.engine).stem

    df = pd.read_csv(args.test)
    runner = TRTRunner(args.engine)

    #y_true = [int(CLASS_TO_IDX[c]) for c in df["label"]]
    y_true = [CLASS_TO_IDX[c] for c in df["class_name"]]
    y_pred = [int(np.argmax(runner.infer(preprocess_numpy(p))))
              for p in df["abs_path"]]

    results = ROOT / "results"

    per = df[["abs_path", "source"]].copy()
    per["y_true"] = y_true
    per["y_pred"] = y_pred
    per.to_csv(results / f"preds_jetson_{eng_key}.csv", index=False)

    wide_path = results / "preds_jetson.csv"
    if wide_path.exists():
        wide = pd.read_csv(wide_path)
    else:
        wide = df[["abs_path", "source"]].copy()
        wide["y_true"] = y_true
    wide[eng_key] = y_pred
    wide.to_csv(wide_path, index=False)

    metrics = compute_metrics(y_true, y_pred)
    metrics["by_subdomain"] = subdomain_accuracy(
        y_true, y_pred, df["source"].tolist())
    (results / f"cls_jetson_{eng_key}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
