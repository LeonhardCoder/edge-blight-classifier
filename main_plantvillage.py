import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy import stats

from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose,
    HueSaturationValue, RandomResizedCrop, RandomBrightnessContrast,
    Compose, Normalize, CoarseDropout, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

from src.dataset import create_csv, create_dataloaders
from src.transforms import get_transforms_phase1
import os
import logging

localDataset = True

DATASET_PATH = 'datasets'       # fallback local
OUTPUT_PATH  = '/home/victor/Documents/edge-blight-classifier/resultados' # fallback local
ROOT_PATH    = '/home/victor/Documents'           # fallback local

if not localDataset:
    DATASET_PATH = 'Data'       # fallback local
    OUTPUT_PATH  = 'resultados' # fallback local
    ROOT_PATH    = '/'           # fallback local
    #from google.colab import drive
    mountPoint = '/content/drive'
    #drive.mount(mountPoint, force_remount=True)

    remotePath = 'MyDrive/UOC/TFM/LAB'  # ajusta si tu carpeta difiere

    # dataset
    DATASET_PATH = os.path.join(mountPoint, remotePath, 'data')

    # resultados
    OUTPUT_PATH = os.path.join(mountPoint, remotePath, 'resultados')
    ROOT_PATH = os.path.join(mountPoint, remotePath)
else:
    DATASET_PATH = os.path.join(ROOT_PATH, DATASET_PATH, 'PlantVille')
    os.makedirs(os.path.join(ROOT_PATH, DATASET_PATH, 'PlantVille', 'train'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_PATH, DATASET_PATH, 'PlantVille', 'valid'), exist_ok=True)

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f'Dataset: {DATASET_PATH}')
print(f'Resultados: {OUTPUT_PATH}')
print("Contenido raíz:", os.listdir(DATASET_PATH))

plantville_root = DATASET_PATH

create_archivo = False
if create_archivo:
    create_csv(
        plantville_root = plantville_root,
        csv_train = os.path.join(DATASET_PATH, 'data/train.csv'),
        csv_val   = os.path.join(DATASET_PATH,'data/val.csv'),
        csv_test  = os.path.join(DATASET_PATH,'data/test.csv'),
    )
# --- Clases PlantVillage — PAPA (3 clases) ---
NUM_CLASSES  = 3
CLASS_NAMES  = ['Early_blight', 'Late_blight', 'Healthy']
CFG = {
    # Hiperparámetros Fase 1
    # LR normal, todas las capas entrenables desde ImageNet
    "lr":            1e-3,
    "weight_decay":  1e-5,
    "optimizer":     "adam",
    "epochs":        30,
    "batch_size":    48,
    "num_workers":   2,
    "img_size":      224,

    # Scheduler
    "lr_factor":     0.7,
    "lr_patience":   5,
    "min_lr":        1e-6,

    # Early stopping
    "es_patience":   7,
    "min_delta":     1e-4,

    # Augmentation fase 1: moderado (PlantVillage ya es limpio)
    "augmentation":  True,
    "aggressive_aug": False,  # False en fase 1, True en fase 2

    # Device
    "device":        "cuda" if torch.cuda.is_available() else "cpu",

    # Semilla para reproducibilidad
    "seed":          42,
}

log = logging.getLogger(__name__)

train_tf, val_tf = get_transforms_phase1()
print(DATASET_PATH)
train_loader, val_loader  = create_dataloaders(os.path.join(DATASET_PATH, 'data/train.csv'),
                   os.path.join(DATASET_PATH,'data/val.csv'),
                   DATASET_PATH,
                   train_tf, val_tf
    )
log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# EDGE_MODELS = {
#     'mobilenetv3_small': 'mobilenetv3_small_100',   # CNN   ~2.5M params
#     'efficientnet_b0':   'efficientnet_b0',          # CNN   ~5.3M params
#     'vit_tiny':          'vit_tiny_patch16_224',     # ViT   ~5.7M params
#     'mobilevit_s':       'mobilevit_s',              # MViT  ~5.6M params
# }
# config/settings.py — Opción A (modelos 2023-2024)
# EDGE_MODELS = {
#     "efficientnetv2_s":   "tf_efficientnetv2_s.in21k",  
                     
#     "efficientformerv2_s0":  "efficientformerv2_s0.snap_dist_in1k",              
#     "repvit_m1":             "repvit_m1.dist_in1k",                       
#     "mobilenetv4_small":     "mobilenetv4_conv_small.e2400_r224_in1k", 
   
# }

EDGE_MODELS = {
    # CNN — familia convolucional
    "mobilenetv4_small": "mobilenetv4_conv_small.e2400_r224_in1k",   # ~2.5M, 2024
    "tf_efficientnetv2_b0":   "tf_efficientnetv2_b0.in1k",                # ~5.9M, 2021

    # ViT — familia transformers híbridos
   "efficientformerv2_s0":   "efficientformerv2_s0.snap_dist_in1k",      # ~3.2M, 2023
   "repvit_m1":              "repvit_m1.dist_in1k",                       # ~4.7M, 2024
}

def print_model_summary():
    """Imprime tabla comparativa de todos los modelos del TFM."""
    print("\n" + "=" * 70)
    print("MODELOS DEL TFM — COMPARATIVA DE PARÁMETROS")
    print(f"{'Modelo':<22} {'Familia':<8} {'timm_name':<30} {'Params (M)':>10}")
    print("-" * 70)
    for key, timm_name in EDGE_MODELS.items():
        familia = 'CNN' if key in ('mobilenetv3_small', 'efficientnet_b0') else 'ViT'
        try:
            m = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)
            params = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"  {key:<20} {familia:<8} {timm_name:<30} {params:>10.2f}")
            del m
        except Exception as e:
            print(f"  {key:<20} ERROR: {e}")
    print("=" * 70)

print_model_summary()

def create_edge_model(model_key, num_classes=3, pretrained=True):
    """
    Factory de modelos ligeros para Edge.
    Todos usan fine-tuning desde ImageNet (transfer learning).
    """
    if model_key not in EDGE_MODELS:
        raise ValueError(f"Modelo desconocido: {model_key}. Opciones: {list(EDGE_MODELS.keys())}")

    timm_name = EDGE_MODELS[model_key]
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  [{model_key}] timm: '{timm_name}' | Params: {total_params:.2f}M | Entrenables: {trainable:.2f}M")

    return model
# ============================================================
# 6. ENTRENAMIENTO
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='  Train'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Val'):
            images, labels = images.to(device), labels.to(device)
            out  = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item()
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), correct / total


def train_model(model, train_loader, val_loader, epochs=30, lr=0.001,
                device='cuda', save_path='./', model_name='model'):
    """
    Entrena un modelo con LR scheduler y guarda el mejor checkpoint.
    Se usa Adam + ReduceLROnPlateau (estrategia conservadora para modelos ligeros).
    """
    from datetime import datetime

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc  = 0.0

    for epoch in range(epochs):
        print(f"\n{'='*60}\n  Epoch {epoch+1}/{epochs} — {model_name}")
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        scheduler.step(v_acc)

        print(f"  Train — Loss: {t_loss:.4f} | Acc: {t_acc:.4f}")
        print(f"  Val   — Loss: {v_loss:.4f} | Acc: {v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            ckpt_path = os.path.join(save_path, f'{model_name}_best.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': v_acc, 'optimizer_state_dict': optimizer.state_dict()},
                       ckpt_path)
            print(f"  Guardado {ckpt_path} (val_acc={v_acc:.4f})")

    return history


# --- Hiperparámetros ---
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
IMG_SIZE      = 224
LEARNING_RATE = 0.001

# --- Clases PlantVillage — PAPA (3 clases) ---
NUM_CLASSES  = 3
CLASS_NAMES  = ['Early_blight', 'Late_blight', 'Healthy']
# Mapeo de carpetas PlantVillage:
#   Potato___Early_blight  → label 0
#   Potato___Late_blight   → label 1
#   Potato___healthy       → label 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_training_history(history, model_name, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key_pair, ylabel, title in [
        (axes[0], ('train_loss', 'val_loss'), 'Loss', 'Loss'),
        (axes[1], ('train_acc',  'val_acc'),  'Accuracy', 'Accuracy'),
    ]:
        ax.plot(history[key_pair[0]], label='Train', linewidth=2)
        ax.plot(history[key_pair[1]], label='Val',   linewidth=2)
        ax.set(xlabel='Epoch', ylabel=ylabel,
               title=f'{model_name} — {title}')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names, model_name, save_path=None):
    plt.figure(figsize=(7, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {model_name}', fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparative_bar(results_comp, metric='accuracy', save_path=None):
    """Barras comparativas por familia (CNN vs ViT)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    keys    = list(results_comp.keys())
    vals    = [results_comp[k][metric] * 100 for k in keys]
    colors  = ['#2196F3' if results_comp[k]['familia'] == 'CNN' else '#FF5722' for k in keys]
    bars    = ax.bar(range(len(keys)), vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=20, ha='right')
    ax.set_ylabel(f'{metric.capitalize()} (%)'); ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'Comparativa {metric.capitalize()} — CNN (azul) vs ViT (naranja)', fontweight='bold')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2196F3', label='CNN'),
                        Patch(color='#FF5722', label='ViT')])
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pareto_accuracy_vs_params(results_comp, save_path=None):
    """
    Gráfico Pareto: Accuracy vs Parámetros (proxy de eficiencia para Edge).
    En el TFM complementar con latencia real medida en Raspberry Pi 4 / Jetson.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    for key, r in results_comp.items():
        color  = '#2196F3' if r['familia'] == 'CNN' else '#FF5722'
        marker = 'o' if r['familia'] == 'CNN' else '^'
        ax.scatter(r['params_M'], r['accuracy'] * 100,
                   s=200, color=color, marker=marker, zorder=5,
                   edgecolors='black', linewidth=1.5)
        ax.annotate(key, (r['params_M'], r['accuracy'] * 100),
                    textcoords='offset points', xytext=(6, 4), fontsize=10, fontweight='bold')
    ax.set_xlabel('Parámetros (M)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Trade-off Accuracy vs Complejidad del Modelo\n'
                 '(Raspberry Pi 4 / Jetson Orin Nano — latencia real en edge_benchmark.py)',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2196F3',
               markersize=12, markeredgecolor='black', label='CNN'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#FF5722',
               markersize=12, markeredgecolor='black', label='ViT'),
    ])
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



import time as time_mod
from datetime import datetime
trained = {}
times   = {}
total_t0 = time_mod.time()


entrenar_modelos = False  # Cambia a True para entrenar los modelos (recomienda GPU)
if entrenar_modelos:
    for model_key in EDGE_MODELS.keys():
        print(f"\n{'='*70}\n🚀 ENTRENANDO: {model_key.upper()}\n{'='*70}")
        t0 = time_mod.time()
        model   = create_edge_model(model_key, num_classes=NUM_CLASSES, pretrained=True)
        history = train_model(
            model       = model,
            train_loader= train_loader,
            val_loader  = val_loader,
            epochs      = NUM_EPOCHS,
            lr          = LEARNING_RATE,
            device      = DEVICE,
            save_path   = os.path.join(OUTPUT_PATH, 'models'),
            model_name  = model_key,
        )
        elapsed = time_mod.time() - t0
        h, m, s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)
        times[model_key] = f"{h:02d}h {m:02d}m {s:02d}s"

        trained[model_key] = {'model': model, 'history': history}
        plot_training_history(history, model_key,
                                save_path=os.path.join(OUTPUT_PATH, 'figures',
                                                        f'{model_key}_training.png'))

    print(f"\n{'='*70}\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Tiempo total: {(time_mod.time()-total_t0)/60:.1f} min")
    for k, t in times.items():
        print(f"   {k:<25}: {t}")
    print("=" * 70)
#return trained

# TODO crear todas las carpetas necesarias en OUTPUT_PATH (models, figures, reports) al inicio del script


def load_all_models(output_path, device, num_classes=3):
    """Carga los 4 checkpoints guardados tras el entrenamiento."""
    loaded = {}
    print("\n" + "=" * 70)
    print("CARGA DE MODELOS ENTRENADOS")
    print("=" * 70)

    for model_key in EDGE_MODELS.keys():
        ckpt_path = os.path.join(output_path, 'models', f'{model_key}_best.pth')
        if os.path.exists(ckpt_path):
            model = create_edge_model(model_key, num_classes=num_classes, pretrained=False)
            ckpt  = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)
            model.eval()
            loaded[model_key] = {
                'model':   model,
                'val_acc': ckpt.get('val_acc', 'N/A'),
                'epoch':   ckpt.get('epoch', 'N/A'),
            }
            print(f"  {model_key:<25} | val_acc={ckpt.get('val_acc','?'):.4f} | epoch={ckpt.get('epoch','?')}")
        else:
            print(f"  NO encontrado: {ckpt_path}")

    print(f"\n  Modelos cargados: {len(loaded)}/{len(EDGE_MODELS)}")
    return loaded

loaded_models = load_all_models(OUTPUT_PATH, DEVICE, NUM_CLASSES)


# ============================================================
# 8. MÉTRICAS DE EFICIENCIA COMPUTACIONAL (Colab/GPU)
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_gflops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """
    Mide GFLOPs usando thop.
    NOTA: Los FLOPs son independientes del hardware; sirven como proxy
    de complejidad computacional comparativa entre modelos.
    """
    try:
        from thop import profile
        dummy = torch.randn(input_size).to(device)
        macs, params = profile(model.to(device), inputs=(dummy,), verbose=False)
        return (macs * 2) / 1e9  # GFLOPs
    except ImportError:
        print("  ⚠ Instalar thop: pip install thop")
        return None


def measure_inference_time_gpu(model, input_size=(1, 3, 224, 224),
                                iterations=200, device='cuda'):
    """
    Latencia de inferencia en GPU (Colab).
    ⚠ ADVERTENCIA: Esta latencia NO es representativa de Raspberry Pi 4
    ni Jetson Orin Nano. Los valores en Edge deben medirse en los
    dispositivos físicos usando el script 'edge_benchmark.py'.
    """
    model.eval()
    model.to(device)
    dummy = torch.randn(input_size).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _  = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms':  np.std(times),
        'min_ms':  np.min(times),
        'max_ms':  np.max(times),
        'device':  device,
        'note':    'GPU Colab — NO equivalente a dispositivos Edge'
    }


def measure_memory_usage_gpu(model, input_size=(1, 3, 224, 224)):
    """
    Uso de memoria GPU durante inferencia.
    Para RAM en Raspberry Pi / Jetson → ver edge_benchmark.py
    """
    if not torch.cuda.is_available():
        return {'note': 'GPU no disponible'}
    model.to('cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dummy = torch.randn(input_size).to('cuda')
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return {'peak_gpu_mb': mem_mb, 'note': 'GPU Colab — para RAM Edge ver edge_benchmark.py'}



def run_full_comparison(loaded_models, val_loader, device, output_path):
    """
    Evaluación comparativa de los 4 modelos en todas las dimensiones del TFM:
      - Precisión / Exactitud (accuracy, F1, precision, recall)
      - GFLOPs (proxy de eficiencia computacional)
      - Latencia GPU (referencia; los valores Edge vienen de edge_benchmark.py)
      - Uso de memoria GPU (idem)
    """
    print("\n" + "=" * 70)
    print("EVALUACIÓN COMPARATIVA — MODELOS EDGE")
    print("(CNN: MobileNetV3-Small, EfficientNet-B0)")
    print("(ViT: ViT-Tiny, MobileViT-S)")
    print("=" * 70)

    results = {}

    for model_key, info in loaded_models.items():
        print(f"\n{'─'*60}\n  📊 {model_key}")
        model = info['model']

        # Métricas de clasificación
        metrics = evaluate_model(model, val_loader, device)

        # Parámetros
        params_M = count_parameters(model) / 1e6

        # GFLOPs
        gflops = measure_gflops(model, device='cpu')

        # Latencia GPU (solo referencia)
        timing = measure_inference_time_gpu(model, device=device)

        # Memoria GPU
        mem_gpu = measure_memory_usage_gpu(model)

        familia = 'CNN' if model_key in ('mobilenetv3_small', 'efficientnet_b0') else 'ViT'

        results[model_key] = {
            'familia':        familia,
            'accuracy':       metrics['accuracy'],
            'precision':      metrics['precision'],
            'recall':         metrics['recall'],
            'f1':             metrics['f1'],
            'params_M':       params_M,
            'gflops':         gflops,
            'lat_gpu_ms':     timing['mean_ms'],
            'lat_gpu_std_ms': timing['std_ms'],
            'mem_gpu_mb':     mem_gpu.get('peak_gpu_mb', None),
        }

        print(f"  Familia      : {familia}")
        print(f"  Accuracy     : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision    : {metrics['precision']:.4f}")
        print(f"  Recall       : {metrics['recall']:.4f}")
        print(f"  F1-Score     : {metrics['f1']:.4f}")
        print(f"  Params       : {params_M:.2f}M")
        if gflops: print(f"  GFLOPs       : {gflops:.3f}")
        print(f"  Lat GPU      : {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms  ⚠ Solo referencia Colab")
        print(f"\n{metrics['report']}")

        # Guardar confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'], CLASS_NAMES, model_key,
            save_path=os.path.join(output_path, 'figures', f'cm_{model_key}.png')
        )

    # Tabla resumen
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(output_path, 'results', 'comparative_results.csv'))
    print("\n" + "=" * 70)
    print("TABLA RESUMEN")
    print(df[['familia','accuracy','f1','params_M','gflops','lat_gpu_ms']].to_string())
    print("=" * 70)
    print(f"\n{output_path}/results/comparative_results.csv")

    return results


# Cargar dataloaders para evaluación
print(DATASET_PATH)
# train_loader, val_loader = create_dataloaders(
#     train_csv  = os.path.join(DATASET_PATH, 'data/train.csv'),
#     val_csv    = os.path.join(DATASET_PATH, 'data/val.csv'),
#     data_path  = DATASET_PATH,
#     batch_size = BATCH_SIZE,
#     num_workers= 2,
# )

def evaluate_model(model, loader, device):
    """Evalúa el modelo y retorna métricas de clasificación."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Eval'):
            out = model(images.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return {
        'accuracy':         accuracy_score(y_true, y_pred),
        'precision':        precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':           recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1':               f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'report':           classification_report(y_true, y_pred, target_names=CLASS_NAMES),
        'y_true': y_true,
        'y_pred': y_pred,
    }
train_loader, val_loader  = create_dataloaders(
    os.path.join(DATASET_PATH, 'data/train.csv'),
                   os.path.join(DATASET_PATH,'data/val.csv'),
                   DATASET_PATH,
                   train_tf, val_tf
    )

if loaded_models:
    results_comparative = run_full_comparison(loaded_models, val_loader, DEVICE, OUTPUT_PATH)
else:
    print("No hay modelos cargados. Entrena primero con RUN_TRAINING = True")
    results_comparative = {}

if results_comparative:
    plot_comparative_bar(results_comparative, metric='accuracy',
                         save_path=os.path.join(OUTPUT_PATH, 'figures', 'bar_accuracy.png'))
    plot_comparative_bar(results_comparative, metric='f1',
                         save_path=os.path.join(OUTPUT_PATH, 'figures', 'bar_f1.png'))
    plot_pareto_accuracy_vs_params(results_comparative,
                                   save_path=os.path.join(OUTPUT_PATH, 'figures', 'pareto_accuracy_params.png'))
