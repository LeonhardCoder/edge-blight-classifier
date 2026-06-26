"""
Dataset PyTorch que lee los CSVs consolidados producidos por
scripts/01_prepare_corpus.py.

Cada CSV tiene columnas: abs_path, label, class_name, source, md5
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import config


class BlightDataset(Dataset):
    """Lee imagenes desde rutas absolutas en CSV y aplica transformaciones."""

    def __init__(self, csv_path, transforms=None):
        self.df = pd.read_csv(csv_path)
        required = {"abs_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en {csv_path}: {missing}")
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["abs_path"]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img, int(row["label"])

    def get_class_weights(self):
        """Pesos inversamente proporcionales a la frecuencia de cada clase."""
        counts = self.df["label"].value_counts().sort_index()
        n_total = counts.sum()
        n_classes = len(counts)
        weights = n_total / (n_classes * counts)
        return weights.values.astype(np.float32)


def create_dataloaders(train_csv, val_csv, train_transforms, val_transforms,
                       batch_size=None, num_workers=None):
    """Crea train/val DataLoaders a partir de CSVs y transformaciones."""
    if batch_size is None:
        batch_size = config.TRAIN_BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    train_ds = BlightDataset(train_csv, train_transforms)
    val_ds   = BlightDataset(val_csv,   val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds
