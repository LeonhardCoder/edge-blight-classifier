import os, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CLASS_NAMES = ['healthy', 'early_blight', 'late_blight']


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc='  Train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='  Val  ', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            total_loss += criterion(out, labels).item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return {
        'loss':       total_loss / len(loader),
        'acc':        accuracy_score(y_true, y_pred),
        'f1_macro':   f1_score(y_true, y_pred, average='macro',
                               zero_division=0),
        'f1_per_cls': f1_score(y_true, y_pred, average=None,
                               zero_division=0),
        'cm':         confusion_matrix(y_true, y_pred),
    }


def run_training(model, model_name, phase,
                 train_loader, val_loader, cfg):
    """
    Función genérica usada por las 3 fases.
    cfg debe tener: device, epochs, lr, es_patience, output_dir
    Opcionalmente: prev_checkpoint (para fases 2 y 3)
    """
    device     = cfg['device']
    output_dir = cfg['output_dir']

    ckpt_dir = os.path.join(output_dir, model_name, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Cargar checkpoint previo si existe (fases 2 y 3)
    if cfg.get('prev_checkpoint') and \
       os.path.exists(cfg['prev_checkpoint']):
        print(f"Cargando pesos: {cfg['prev_checkpoint']}")
        ckpt = torch.load(cfg['prev_checkpoint'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['lr'], weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5, min_lr=1e-6
    )

    csv_path = os.path.join(output_dir, model_name,
                            f'phase{phase}_{model_name}_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'train_acc',
            'val_loss', 'val_acc', 'val_f1_macro',
            'f1_healthy', 'f1_early', 'f1_late', 'lr', 'is_best'
        ])

    best_f1, no_improve = 0.0, 0
    ckpt_path = os.path.join(ckpt_dir,
                             f'phase{phase}_{model_name}_best.pth')

    for epoch in range(cfg['epochs']):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        val_m = val_epoch(model, val_loader, criterion, device)

        vl_f1  = val_m['f1_macro']
        f1_cls = val_m['f1_per_cls']
        cur_lr = optimizer.param_groups[0]['lr']

        is_best = vl_f1 > best_f1 + 1e-4
        if is_best:
            best_f1, no_improve = vl_f1, 0
            torch.save({
                'phase':            phase,
                'epoch':            epoch,
                'model_name':       model_name,
                'model_state_dict': model.state_dict(),
                'best_f1_macro':    best_f1,
                'val_acc':          val_m['acc'],
                'val_loss':         val_m['loss'],
                'cm':               val_m['cm'].tolist(),
            }, ckpt_path)
        else:
            no_improve += 1

        scheduler.step(vl_f1)

        print(f"Ep {epoch+1:02d}/{cfg['epochs']}  "
              f"tr={tr_loss:.4f}/{tr_acc*100:.1f}%  "
              f"val={val_m['loss']:.4f}/{val_m['acc']*100:.1f}%  "
              f"F1={vl_f1:.4f}  lr={cur_lr:.1e}  "
              f"{'[BEST]' if is_best else f'wait={no_improve}'}  "
              f"({time.time()-t0:.0f}s)")
        for i, cn in enumerate(CLASS_NAMES):
            print(f"    {cn}: F1={f1_cls[i]:.4f}")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, tr_loss, tr_acc,
                val_m['loss'], val_m['acc'], vl_f1,
                f1_cls[0], f1_cls[1], f1_cls[2],
                cur_lr, int(is_best)
            ])

        if no_improve >= cfg['es_patience']:
            print(f"Early stopping. Mejor F1: {best_f1:.4f}")
            break

    print(f"\nCompletado. Checkpoint: {ckpt_path}")
    return ckpt_path