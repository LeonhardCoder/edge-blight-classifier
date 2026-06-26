import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from common.labels import IMG_SIZE, RESIZE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _safe_open_beta(path):
    p = Path(unicodedata.normalize("NFC", str(path)))
    if not p.exists():
        alt = Path(unicodedata.normalize("NFD", str(path)))
        p = alt if alt.exists() else p
    return Image.open(p).convert("RGB")


def preprocess_numpy_beta(path):
    img = _safe_open_beta(path).resize((RESIZE_SIZE, RESIZE_SIZE), Image.BILINEAR)
    off = (RESIZE_SIZE - IMG_SIZE) // 2
    img = img.crop((off, off, off + IMG_SIZE, off + IMG_SIZE))
    arr = np.asarray(img, np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / \
        np.array(IMAGENET_STD, np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], np.float32)


import numpy as np
import albumentations as A
from common.labels import IMG_SIZE, RESIZE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from PIL import Image
import unicodedata
from pathlib import Path

_TF = A.Compose([
    A.Resize(RESIZE_SIZE, RESIZE_SIZE),          
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def _safe_open(path):
    p = Path(unicodedata.normalize("NFC", str(path)))
    if not p.exists():
        alt = Path(unicodedata.normalize("NFD", str(path)))
        p = alt if alt.exists() else p
    return np.array(Image.open(p).convert("RGB"))   

def preprocess_numpy(path):
    img = _safe_open(path)
    x = _TF(image=img)["image"]                     
    x = np.transpose(x, (2, 0, 1))                 
    return np.ascontiguousarray(x[None, ...], np.float32)
