"""
Configuracion central del Edge Blight Classifier.

Lee 'configs/settings.conf' y expone las constantes al resto del proyecto.

Para cambiar de entorno:
        export TFM_ENV=edge_rpi
        python scripts/02_train.py
"""
from pathlib import Path
import configparser
import os
import platform

# =============================================================================
# CARGA DEL FICHERO DE CONFIGURACION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
print(f"Cargando configuracion desde: {PROJECT_ROOT}")
CONF_PATH    = PROJECT_ROOT / "configs" / "settings.conf"

if not CONF_PATH.exists():
    raise FileNotFoundError(
        f"No se encuentra el fichero de configuracion: {CONF_PATH}"
    )

_cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
_cfg.read(CONF_PATH, encoding="utf-8")

ENV = os.environ.get("TFM_ENV", _cfg["active"]["env"]).lower()
assert ENV in {"local", "colab", "edge_rpi", "edge_jetson", "edge-train"}, f"Entorno invalido: {ENV}"

if ENV not in _cfg:
    raise ValueError(f"La seccion [{ENV}] no existe en {CONF_PATH}")

_env = _cfg[ENV]

# =============================================================================
# RUTAS
# =============================================================================
LOCAL_BASE      = _env["data_base"]
DATA_DIR        = Path(LOCAL_BASE) / "dataset"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"

CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
EXPORTED_DIR    = OUTPUTS_DIR / "exported"
METRICS_DIR     = OUTPUTS_DIR / "metrics"
FIGURES_DIR     = OUTPUTS_DIR / "figures"
CORPUS_DIR      = OUTPUTS_DIR / "corpus"
LOGS_DIR        = OUTPUTS_DIR / "logs"

# =============================================================================
# DISPOSITIVO Y PRECISION
# =============================================================================
_device_pref = _env.get("device", "auto").lower()
if _device_pref == "auto":
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"
else:
    DEVICE = _device_pref

MIXED_PRECISION = _env.getboolean("mixed_precision") and (DEVICE == "cuda")

# =============================================================================
# BATCH SIZES Y WORKERS
# =============================================================================
TRAIN_BATCH_SIZE = _env.getint("train_batch_size")
EVAL_BATCH_SIZE  = _env.getint("eval_batch_size")
NUM_WORKERS      = _env.getint("num_workers")

if platform.system() == "Windows":
    NUM_WORKERS = 0

# =============================================================================
# DATASETS
# Tras homogeneizacion de etiquetas, los datasets con 3 clases comparten
# el esquema de subcarpetas. Documentado en la memoria como paso de
# pre-procesamiento de la seccion 'Materiales y metodos'.
# =============================================================================
HOMOGENEOUS_MAP = {
    "Potato___healthy":      "healthy",
    "Potato___Early_blight": "early_blight",
    "Potato___Late_blight":  "late_blight",
}

# Tilahun no tiene early_blight (mapeo distinto)
TILAHUN_MAP = {
    "Potato_Healthy":     "healthy",
    "Potato_Late_Blight": "late_blight",
}

MENDELEY_UE_MAP = {
    "Potato___healthy":      "healthy",
    "Potato___Early_blight_Fungi": "early_blight",
    "Potato___Late_blight_Phytopthora":  "late_blight",
}

DATASETS = {
    "plantvillage": {
        "path":  DATA_DIR / "Plantville",
        "map":   dict(HOMOGENEOUS_MAP),
        "origin": "USA / laboratorio",
    },
    "mendeley_ue": {
        "path":  DATA_DIR / "mendeley_uncontrolled_Environment",
        "map":   dict(MENDELEY_UE_MAP),
        "origin": "Indonesia / campo (Shabrina et al. 2023)",
    },
    "tilahun": {
        "path":  DATA_DIR / "Addis",
        "map":   TILAHUN_MAP,
        "origin": "Etiopia / semicontrolado",
    },
    "quizhpe": {
        "path":  DATA_DIR / "GerardoQuizhpe",
        "map":   dict(HOMOGENEOUS_MAP),
        "origin": "Ecuador / campo andino",
    },
    "propio": {
        "path":  DATA_DIR / "PropioVictor",
        "map":   None,   # sin etiquetar, solo inventario
        "origin": "Ecuador / propio (sin etiquetar)",
    },
}

# =============================================================================
# CLASES
# =============================================================================
CLASS_NAMES    = ["healthy", "early_blight", "late_blight"]
NUM_CLASSES    = len(CLASS_NAMES)
CLASS_TO_IDX   = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS   = {i: name for i, name in enumerate(CLASS_NAMES)}
PRIORITY_CLASS = _cfg["experiment"]["priority_class"]

# =============================================================================
# MODELOS (4 arquitecturas modernas 2021-2024)
# =============================================================================
MODEL_REGISTRY = {
    "mobilenetv4_conv_small": {
        "timm_name": "mobilenetv4_conv_small.e2400_r224_in1k",
        "family":    "CNN",
        "year":      2024,
        "img_size":  224,
        "params_M":  3.8,
    },
    "efficientnetv2_b0": {
        "timm_name": "tf_efficientnetv2_b0.in1k",
        "family":    "CNN",
        "year":      2021,
        "img_size":  224,
        "params_M":  7.1,
    },
    "efficientformerv2_s0": {
        "timm_name": "efficientformerv2_s0.snap_dist_in1k",
        "family":    "Hibrido_Attn",
        "year":      2023,
        "img_size":  224,
        "params_M":  3.5,
    },
    "repvit_m1": {
        "timm_name": "repvit_m1.dist_in1k",
        "family":    "Hibrido_Attn",
        "year":      2024,
        "img_size":  224,
        "params_M":  5.5,
    },
}

# =============================================================================
# ENTRENAMIENTO (un unico entrenamiento por modelo)
# =============================================================================
TRAINING = {
    "epochs":      _cfg.getint("training", "epochs"),
    "lr":          _cfg.getfloat("training", "lr"),
    "es_patience": _cfg.getint("training", "es_patience"),
    "img_size":    _cfg.getint("training", "img_size"),
}

WEIGHT_DECAY = _cfg.getfloat("experiment", "weight_decay")
SEED         = _cfg.getint("experiment", "seed")

# =============================================================================
# SPLITS
# =============================================================================
SPLITS = {
    "train": _cfg.getfloat("splits", "train"),
    "val":   _cfg.getfloat("splits", "val"),
    "test":  _cfg.getfloat("splits", "test"),
}

# =============================================================================
# EDGE BENCHMARK
# =============================================================================
EDGE_BENCHMARK = {
    "warmup_iters":   _cfg.getint("edge_benchmark", "warmup_iters"),
    "measured_iters": _cfg.getint("edge_benchmark", "measured_iters"),
    "subset_size":    _cfg.getint("edge_benchmark", "subset_size"),
}

QUANT_THRESHOLD = _cfg.getfloat("quantization", "threshold")

# =============================================================================
# MATRIZ DE FORMATOS POR DISPOSITIVO EDGE
# =============================================================================
def _parse_list(value: str):
    return [s.strip() for s in value.split(",") if s.strip()]

EDGE_MATRIX = {
    "rpi":    _parse_list(_cfg["matrix_rpi"]["formats"]),
    "jetson": _parse_list(_cfg["matrix_jetson"]["formats"]),
}

# =============================================================================
# UTILIDADES
# =============================================================================
def ensure_dirs():
    """Crea carpetas de salida si no existen."""
    for d in (OUTPUTS_DIR, CHECKPOINTS_DIR, EXPORTED_DIR,
              METRICS_DIR, FIGURES_DIR, CORPUS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_labeled_datasets(names=None):
    """Devuelve los datasets con mapeo definido (excluye los sin etiquetar)."""
    if names is None:
        names = list(DATASETS.keys())
    return [n for n in names if DATASETS[n]["map"] is not None]


def get_pending_datasets(names=None):
    """Devuelve los datasets sin etiquetar (map=None)."""
    if names is None:
        names = list(DATASETS.keys())
    return [n for n in names if DATASETS[n]["map"] is None]


def _count_images(folder: Path) -> int:
    """Cuenta imagenes JPG/JPEG/PNG en una carpeta (case-insensitive)."""
    if not folder.exists():
        return 0
    exts = (".jpg", ".jpeg", ".png")
    return sum(1 for p in folder.iterdir()
               if p.is_file() and p.suffix.lower() in exts)


def validate_class_mappings(verbose: bool = True) -> bool:
    """Verifica que cada subcarpeta declarada existe y tiene imagenes."""
    all_ok = True
    issues = []

    for name, info in DATASETS.items():
        path, mapping = info["path"], info["map"]

        if mapping is None:
            full = path
            n = _count_images(full)
            if verbose:
                print(f"  [SKIP] {name}: sin mapeo (pendiente de etiquetar)  {n} imagenes")
            continue

        if not path.exists():
            issues.append(f"  [MISS] {name}: ruta no existe -> {path}")
            all_ok = False
            continue

        for subdir in mapping.keys():
            full = path / subdir
            if not full.exists():
                issues.append(f"  [MISS] {name}: subcarpeta '{subdir}' no encontrada")
                all_ok = False
                continue
            n = _count_images(full)
            if n == 0:
                issues.append(f"  [EMPTY] {name}/{subdir}: 0 imagenes")
                all_ok = False
            elif verbose:
                print(f"  [OK] {name}/{subdir}: {n} imagenes")

    if issues:
        print("\nPROBLEMAS DETECTADOS:")
        for i in issues:
            print(i)
    elif verbose:
        print("\nTodos los datasets verificados correctamente.")

    return all_ok


def show_config():
    """Imprime un resumen para depuracion."""
    print("=" * 60)
    print("EDGE BLIGHT CLASSIFIER - CONFIGURACION")
    print("=" * 60)
    print(f"ENV:             {ENV}")
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"DATA_DIR:        {DATA_DIR}")
    print(f"OUTPUTS_DIR:     {OUTPUTS_DIR}")
    print(f"DEVICE:          {DEVICE}")
    print(f"MIXED_PRECISION: {MIXED_PRECISION}")
    print(f"NUM_WORKERS:     {NUM_WORKERS}")
    print(f"TRAIN_BATCH:     {TRAIN_BATCH_SIZE}")
    print(f"EVAL_BATCH:      {EVAL_BATCH_SIZE}")
    print()
    print("Datasets:")
    for name, info in DATASETS.items():
        path = info["path"]
        if info["map"] is None:
            status = "PEND"
        elif path.exists():
            status = "OK"
        else:
            status = "FALTA"
        print(f"  [{status}] {name} ({info['origin']}): {path}")
    print()
    print(f"Modelos: {list(MODEL_REGISTRY.keys())}")
    print(f"Clases:  {CLASS_NAMES}")
    print(f"Seed:    {SEED}")
    print(f"Training: {TRAINING['epochs']} epochs, lr={TRAINING['lr']}")
    print(f"Splits:  {SPLITS}")
    print()
    print("Matriz Edge:")
    print(f"  RPi:    {EDGE_MATRIX['rpi']}")
    print(f"  Jetson: {EDGE_MATRIX['jetson']}")
    print("=" * 60)


if __name__ == "__main__":
    show_config()
    ensure_dirs()
    print("\nDirectorios de outputs creados.")
    validate_class_mappings()
