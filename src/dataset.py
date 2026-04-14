import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

CLASS_TO_IDX = {
    'healthy':      0,
    'early_blight': 1,
    'late_blight':  2,
}

FOLDER_TO_CLASS = {
    'Potato___Early_blight': 'early_blight',
    'Potato___healthy':      'healthy',
    'Potato___Late_blight':  'late_blight',
}


class PlantDataset(Dataset):
    def __init__(self, csv_path, data_path, transforms=None):
        df = pd.read_csv(csv_path, header=None)
        self.images     = df.iloc[:, 0].values
        self.labels     = df.iloc[:, 3].values.astype(int)
        self.data_path  = data_path
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.images[idx])
        img  = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, self.labels[idx]


def create_csv(plantville_root, csv_train, csv_val, csv_test,
               train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Lee las 3 carpetas de PlantVillage y genera los 3 CSVs
    sin mover ninguna imagen.
    """
    records = []
    for folder_name, canonical_class in FOLDER_TO_CLASS.items():
        folder_path = os.path.join(plantville_root, folder_name)
        if not os.path.exists(folder_path):
            print(f"ADVERTENCIA: {folder_path} no existe")
            continue
        imgs = [f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        for img_name in imgs:
            records.append({
                'rel_path': os.path.join(folder_name, img_name),
                'label':    CLASS_TO_IDX[canonical_class],
            })
        print(f"  {canonical_class}: {len(imgs)} imgs")

    df = pd.DataFrame(records)
    print(f"Total: {len(df)} imágenes")

    val_test = val_ratio + (1 - train_ratio - val_ratio)
    df_train, df_temp = train_test_split(
        df, test_size=val_test, stratify=df['label'], random_state=seed)
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, stratify=df_temp['label'], random_state=seed)

    for df_split, path in [(df_train, csv_train),
                           (df_val,   csv_val),
                           (df_test,  csv_test)]:
        pd.DataFrame({
            0: df_split['rel_path'].values,
            1: '', 2: '',
            3: df_split['label'].values,
        }).to_csv(path, index=False, header=False)
        print(f"  {os.path.basename(path)}: {len(df_split)} imgs")


def create_dataloaders(csv_train, csv_val, data_path,
                       train_transforms, val_transforms,
                       batch_size=32, num_workers=2):
    train_ds = PlantDataset(csv_train, data_path, train_transforms)
    val_ds   = PlantDataset(csv_val,   data_path, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader