"""
Configuracion central del TFM Edge Blight Classifier.

Lee 'tfm.conf' (formato INI estandar) y expone las constantes al resto
del proyecto. Para cambiar de entorno:

    Opcion A: editar la seccion [active] en tfm.conf
    Opcion B: exportar la variable de entorno TFM_ENV
        export TFM_ENV=edge
        python train.py

La variable de entorno TFM_ENV tiene prioridad sobre settings.conf.
"""
from pathlib import Path
import configparser
import os
import platform

# =============================================================================
# CARGA DEL FICHERO DE CONFIGURACION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
CONF_PATH    = PROJECT_ROOT / "settings.conf"

if not CONF_PATH.exists():
    raise FileNotFoundError(
        f"No se encuentra el fichero de configuracion: {CONF_PATH}\n"
        f"Copia settings.conf.example a settings.conf y ajustalo a tu maquina."
    )

_cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
_cfg.read(CONF_PATH, encoding="utf-8")

# Resolucion del entorno activo: variable de entorno > [active] en settings.conf
ENV = os.environ.get("TFM_ENV", _cfg["active"]["env"]).lower()
assert ENV in {"local", "colab", "edge", "edge-train"}, f"Entorno invalido: {ENV}"

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
# Tras la homogeneizacion de etiquetas, los datasets con tres clases comparten
# el mismo esquema de subcarpetas. Documentar en 'Materiales y metodos'.
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
    "Potato___Early_blight Fungi": "early_blight",
    "Potato___Late_blight Phytopthora":  "late_blight",
}

DATASETS = {
    "plantvillage": {
        "path":  DATA_DIR / "Plantville",
        "map":   dict(HOMOGENEOUS_MAP),
        "phase": "phase1_domain",
    },
    "mendeley_ue": {
        "path":  DATA_DIR / "mendeley_uncontrolled_Environment",
        "map":   dict(MENDELEY_UE_MAP),
        "phase": "phase1_domain",
    },
    "tilahun": {
        "path":  DATA_DIR / "Addis",
        "map":   TILAHUN_MAP,
        "phase": "phase1_domain",
    },
    "quizhpe": {
        "path":  DATA_DIR / "GerardoQuizhpe",
        "map":   dict(HOMOGENEOUS_MAP),
        "phase": "phase2_field",
    },
    "propio": {
        "path":  DATA_DIR / "PropioVictor",
        "map":   None,  # Pendiente de etiquetar, se asignara a mano en el futuro
        "phase":  "phase2_field_pending" #"phase2_field",
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
# MODELOS (constantes del proyecto, no varian por entorno)
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
        "family":    "ViT",
        "year":      2023,
        "img_size":  224,
        "params_M":  3.5,
    },
    "repvit_m1": {
        "timm_name": "repvit_m1.dist_in1k",
        "family":    "Hibrido",
        "year":      2024,
        "img_size":  224,
        "params_M":  5.5,
    },
}

# =============================================================================
# HIPERPARAMETROS DE ENTRENAMIENTO
# =============================================================================
def _parse_list(value: str):
    return [s.strip() for s in value.split(",") if s.strip()]


PHASE1 = {
    "epochs":      _cfg.getint("phase1", "epochs"),
    "lr":          _cfg.getfloat("phase1", "lr"),
    "freeze":      _cfg.getboolean("phase1", "freeze"),
    "es_patience": _cfg.getint("phase1", "es_patience"),
    "datasets":    _parse_list(_cfg["phase1"]["datasets"]),
}

PHASE2 = {
    "epochs":      _cfg.getint("phase2", "epochs"),
    "lr":          _cfg.getfloat("phase2", "lr"),
    "freeze":      _cfg.getboolean("phase2", "freeze"),
    "es_patience": _cfg.getint("phase2", "es_patience"),
    "datasets":    _parse_list(_cfg["phase2"]["datasets"]),
}

MAX_TOTAL_EPOCHS = _cfg.getint("experiment", "max_total_epochs")
WEIGHT_DECAY     = _cfg.getfloat("experiment", "weight_decay")
SEED             = _cfg.getint("experiment", "seed")
QUANT_THRESHOLD  = _cfg.getfloat("experiment", "quant_threshold")

# =============================================================================
# SPLITS POR FASE
# =============================================================================
SPLITS = {
    "phase1_domain": {
        "train": _cfg.getfloat("splits", "phase1_train"),
        "val":   _cfg.getfloat("splits", "phase1_val"),
        "test":  _cfg.getfloat("splits", "phase1_test"),
    },
    "phase2_field": {
        "train": _cfg.getfloat("splits", "phase2_train"),
        "val":   _cfg.getfloat("splits", "phase2_val"),
        "test":  _cfg.getfloat("splits", "phase2_test"),
    },
}

FINAL_TEST_DATASETS = _parse_list(_cfg["splits"]["final_test_datasets"])

# =============================================================================
# AUGMENTATIONS
# =============================================================================
AUGMENTATIONS = {
    "phase1": [
        "random_horizontal_flip",
        "random_rotation_15",
        "color_jitter_0.2",
    ],
    "phase2": [
        "random_horizontal_flip",
        "random_rotation_30",
        "color_jitter_0.3",
        "random_resized_crop",
        "random_erasing",
    ],
    "eval": [],
}

# =============================================================================
# EVALUACION EN EDGE
# =============================================================================
EDGE_BENCHMARK = {
    "warmup_iters":      _cfg.getint("edge_benchmark", "warmup_iters"),
    "measured_iters":    _cfg.getint("edge_benchmark", "measured_iters"),
    "subset_size":       _cfg.getint("edge_benchmark", "subset_size"),
    "energy_duration_s": _cfg.getint("edge_benchmark", "energy_duration_s"),
}

# =============================================================================
# UTILIDADES
# =============================================================================
def ensure_dirs():
    """Crea todas las carpetas de salida si no existen."""
    for d in (OUTPUTS_DIR, CHECKPOINTS_DIR, EXPORTED_DIR,
              METRICS_DIR, FIGURES_DIR, CORPUS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


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
    print(f"CONF_FILE:       {CONF_PATH}")
    print(f"ENV:             {ENV}")
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"LOCAL_BASE:      {LOCAL_BASE}")
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
        ok = "OK" if path.exists() else "FALTA"
        print(f"  [{ok}] {name} ({info['phase']}): {path}")
    print()
    print(f"Modelos: {list(MODEL_REGISTRY.keys())}")
    print(f"Clases:  {CLASS_NAMES}")
    print(f"Seed:    {SEED}")
    print(f"Phase1:  {PHASE1['epochs']} epochs, lr={PHASE1['lr']}, freeze={PHASE1['freeze']}")
    print(f"Phase2:  {PHASE2['epochs']} epochs, lr={PHASE2['lr']}, freeze={PHASE2['freeze']}")
    print("=" * 60)


if __name__ == "__main__":
    show_config()
    ensure_dirs()
    print("\nDirectorios de outputs creados.\n")
    print("Validando estructura de datasets...")
    validate_class_mappings()