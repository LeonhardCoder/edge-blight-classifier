"""
Entrena los 4 modelos del TFM con protocolo comun.

Lee train.csv y val.csv producidos por 01_prepare_corpus.py,
entrena cada modelo, guarda el mejor checkpoint y las curvas de
entrenamiento en outputs/checkpoints/ y outputs/metrics/.

Uso:
    python scripts/02_train.py                  # los 4 modelos
    python scripts/02_train.py --models repvit_m1 mobilenetv4_conv_small
    python scripts/02_train.py --skip-existing  # no re-entrenar si ya hay ckpt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch

import config
from src.dataset import create_dataloaders
from src.models import create_model, count_parameters, print_models_summary
from src.transforms import get_train_transforms, get_eval_transforms
from src.trainer import train_model


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset de modelos a entrenar (por defecto todos)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Saltar modelos que ya tienen checkpoint")
    args = parser.parse_args()

    config.ensure_dirs()
    set_seed(config.SEED)

    print("=" * 70)
    print("ENTRENAMIENTO DE LOS MODELOS DEL TFM")
    print("=" * 70)
    config.show_config()

    train_csv = config.CORPUS_DIR / "train.csv"
    val_csv   = config.CORPUS_DIR / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise SystemExit(
            "ERROR: faltan CSVs. Ejecuta primero scripts/01_prepare_corpus.py"
        )

    img_size = config.TRAINING["img_size"]
    train_tf = get_train_transforms(img_size)
    val_tf   = get_eval_transforms(img_size)

    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        train_csv, val_csv, train_tf, val_tf,
    )
    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)}")

    # Class weights basados en frecuencia del split de train
    class_weights = torch.tensor(train_ds.get_class_weights(), dtype=torch.float)
    print(f"Class weights: {class_weights.tolist()}")

    print_models_summary()

    models_to_train = args.models or list(config.MODEL_REGISTRY.keys())
    timings = {}

    for model_key in models_to_train:
        ckpt_path = config.CHECKPOINTS_DIR / f"{model_key}_best.pth"
        if args.skip_existing and ckpt_path.exists():
            print(f"\n[SKIP] {model_key}: ya existe {ckpt_path}")
            continue

        print(f"\n{'=' * 70}\n>>> ENTRENANDO {model_key}\n{'=' * 70}")
        t0 = time.time()
        model = create_model(model_key, pretrained=True)
        params = count_parameters(model)
        print(f"  Params: {params['total_M']:.2f}M | Trainable: {params['trainable_M']:.2f}M")

        ckpt, best_f1 = train_model(
            model, model_key, train_loader, val_loader,
            class_weights=class_weights,
        )

        elapsed = time.time() - t0
        timings[model_key] = elapsed
        print(f"  Tiempo: {elapsed/60:.1f} min | Mejor F1: {best_f1:.4f}")

        # Liberar memoria entre modelos
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETO")
    for k, t in timings.items():
        print(f"  {k:<28} {t/60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
