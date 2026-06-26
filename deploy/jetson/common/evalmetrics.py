import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                            confusion_matrix)

from common.labels import CLASSES, CLASS_TO_IDX, CRITICAL_CLASS


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(CLASSES)))

    rec = recall_score(y_true, y_pred, average=None,
                       labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    li = CLASS_TO_IDX[CRITICAL_CLASS]
    hi = CLASS_TO_IDX["healthy"]
    fn_critical = int(cm[li, hi])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                  labels=labels, zero_division=0)),
        "recall_per_class": {CLASSES[i]: float(rec[i]) for i in labels},
        "fn_critical": fn_critical,
        "confusion_matrix": cm.tolist(),
    }


def subdomain_accuracy(y_true, y_pred, sources):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sources = np.asarray(sources)
    out = {}
    for src in np.unique(sources):
        m = sources == src
        out[str(src)] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y_true[m], y_pred[m])),
        }
    return out
