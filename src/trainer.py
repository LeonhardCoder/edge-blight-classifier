"""
Bucle de entrenamiento generico.

Diseno: una sola fase de fine-tuning desde pesos ImageNet, identica para
los 4 modelos. Las diferencias observadas en Edge seran atribuibles a
la arquitectura (control de variables).
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

import config


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    #scaler = torch.cuda.amp.GradScaler() if use_amp else None
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            #with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda'):
                out = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out = model(images)
            total_loss += criterion(out, labels).item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return {
        "loss":       total_loss / len(loader),
        "acc":        accuracy_score(y_true, y_pred),
        "f1_macro":   f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_cls": f1_score(y_true, y_pred, average=None, zero_division=0),
        "cm":         confusion_matrix(y_true, y_pred,
                                       labels=list(range(config.NUM_CLASSES))),
    }


def train_model(model, model_key, train_loader, val_loader,
                class_weights=None, use_amp=None):
    """
    Entrena un modelo durante config.TRAINING['epochs'] con early stopping.

    Args:
        model: instancia de modelo (timm)
        model_key: clave del modelo (para checkpoints y logs)
        train_loader, val_loader: DataLoaders
        class_weights: torch.Tensor con pesos de clase para CrossEntropy
                       (recomendado si hay desbalance fuerte)
        use_amp: si None, usa config.MIXED_PRECISION
    """
    if use_amp is None:
        use_amp = config.MIXED_PRECISION

    device = config.DEVICE
    model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAINING["lr"],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6,
    )

    ckpt_path = config.CHECKPOINTS_DIR / f"{model_key}_best.pth"
    csv_path  = config.METRICS_DIR / f"train_{model_key}.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "val_f1_macro",
            "f1_healthy", "f1_early", "f1_late",
            "lr", "is_best", "epoch_time_s",
        ])

    best_f1, no_improve = 0.0, 0

    for epoch in range(config.TRAINING["epochs"]):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp,
        )
        val_m = validate(model, val_loader, criterion, device)

        vl_f1 = val_m["f1_macro"]
        f1_cls = val_m["f1_per_cls"]
        cur_lr = optimizer.param_groups[0]["lr"]

        is_best = vl_f1 > best_f1 + 1e-4
        if is_best:
            best_f1, no_improve = vl_f1, 0
            torch.save({
                "model_key":        model_key,
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "best_f1_macro":    best_f1,
                "val_acc":          val_m["acc"],
                "val_loss":         val_m["loss"],
                "cm":               val_m["cm"].tolist(),
            }, ckpt_path)
        else:
            no_improve += 1

        scheduler.step(vl_f1)
        ep_time = time.time() - t0

        # Asegurar 3 valores aunque alguna clase no aparezca en val
        f1_h = float(f1_cls[0]) if len(f1_cls) > 0 else 0.0
        f1_e = float(f1_cls[1]) if len(f1_cls) > 1 else 0.0
        f1_l = float(f1_cls[2]) if len(f1_cls) > 2 else 0.0

        print(
            f"  Ep {epoch+1:02d}/{config.TRAINING['epochs']}  "
            f"tr={tr_loss:.4f}/{tr_acc*100:.1f}%  "
            f"val={val_m['loss']:.4f}/{val_m['acc']*100:.1f}%  "
            f"F1={vl_f1:.4f}  lr={cur_lr:.1e}  "
            f"{'[BEST]' if is_best else f'wait={no_improve}'}  "
            f"({ep_time:.0f}s)"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, tr_loss, tr_acc,
                val_m["loss"], val_m["acc"], vl_f1,
                f1_h, f1_e, f1_l,
                cur_lr, int(is_best), ep_time,
            ])

        if no_improve >= config.TRAINING["es_patience"]:
            print(f"  Early stopping en epoca {epoch+1}. Mejor F1: {best_f1:.4f}")
            break

    print(f"  Checkpoint final: {ckpt_path}")
    return ckpt_path, best_f1
