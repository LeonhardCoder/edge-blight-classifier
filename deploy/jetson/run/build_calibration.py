"""
build_calibration.py — genera calib_npy/ para cuantización INT8.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/victor/tfm-edge")
sys.path.insert(0, str(ROOT))
from common.preprocess import preprocess_numpy 

from common.labels import IMG_SIZE, RESIZE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED


# --- rutas ---
TRAIN_CSV = Path("/home/victor/Documents/edge-blight-classifier/project/outputs/corpus/train.csv")
TEST_CSV  = Path("/home/victor/Documents/edge-blight-classifier/project/outputs/corpus/test.csv")
OUT       = ROOT / "calib_npy"
PER_CLASS, PATH_COL = 50, "abs_path"

OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(TRAIN_CSV)
if "md5" not in train.columns:
    raise SystemExit("La fuente no tiene md5: no puedo verificar no-fuga.")
test = pd.read_csv(TEST_CSV)
if "md5" in test.columns:
    overlap = set(train["md5"]) & set(test["md5"])
    train = train[~train["md5"].isin(test["md5"])]
    print(f"Imágenes que estaban en train Y test (excluidas): {len(overlap)}")
else:
    raise SystemExit("test sin md5: no puedo garantizar ausencia de fuga. Para.")
sample = (train.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=PER_CLASS, random_state=SEED)))
assert len(sample) == PER_CLASS * sample["label"].nunique(), "faltan imágenes en alguna clase"

shapes = set()
for i, row in enumerate(sample.itertuples()):
    x = preprocess_numpy(getattr(row, PATH_COL))
    x = np.ascontiguousarray(x, dtype=np.float32)
    shapes.add(x.shape)
    np.save(OUT / f"calib_{i:04d}.npy", x)

print(f"{len(sample)} tensores -> {OUT}")
print(f"Shapes: {shapes}  (debe ser UNA sola y = entrada del ONNX)")