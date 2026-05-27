"""
Transformaciones de imagenes para entrenamiento y evaluacion.

Como el foco del TFM esta en Edge y NO en domain adaptation, se usa
una unica configuracion de augmentacion estandar para todos los modelos.
"""
from albumentations import (
    Compose, Resize, CenterCrop, Normalize,
    RandomResizedCrop, HorizontalFlip, VerticalFlip,
    ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast,
    CoarseDropout,
)
from albumentations.pytorch import ToTensorV2

# Normalizacion ImageNet (necesaria porque partimos de pesos ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224):
    """Augmentaciones de entrenamiento."""
    return Compose([
        RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.3),
        ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
            border_mode=0, p=0.6,
        ),
        HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=20,
            val_shift_limit=15, p=0.5,
        ),
        RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5,
        ),
        CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.3,
        ),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(img_size: int = 224):
    """Transformaciones para validacion y test (deterministas)."""
    return Compose([
        Resize(int(img_size * 1.14), int(img_size * 1.14)),
        CenterCrop(img_size, img_size),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
