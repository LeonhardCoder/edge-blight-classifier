from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip,
    ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast,
    Normalize, Resize, CenterCrop, Transpose,
    GaussNoise, MotionBlur, GaussianBlur, ImageCompression,
    CoarseDropout, OneOf
)
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
SIZE = 224


def get_val_transforms():
    return Compose([
        Resize(SIZE, SIZE),
        CenterCrop(SIZE, SIZE),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_transforms_phase1():
    train = Compose([
        RandomResizedCrop(size=(SIZE, SIZE), scale=(0.8, 1.0)),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.3),
        Transpose(p=0.3),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                         rotate_limit=20, border_mode=0, p=0.5),
        HueSaturationValue(10, 15, 10, p=0.5),
        RandomBrightnessContrast(0.1, 0.1, p=0.5),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    return train, get_val_transforms()


def get_transforms_phase2():
    train = Compose([
        RandomResizedCrop(size=(SIZE, SIZE), scale=(0.6, 1.0)),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2,
                         rotate_limit=45, border_mode=0, p=0.8),
        HueSaturationValue(25, 35, 25, p=0.7),
        RandomBrightnessContrast(0.3, 0.3, p=0.7),
        OneOf([
            GaussNoise(var_limit=(20.0, 80.0), p=1.0),
            MotionBlur(blur_limit=7, p=1.0),
            GaussianBlur(blur_limit=(3, 9), p=1.0),
        ], p=0.6),
        ImageCompression(quality_range=(20, 70), p=0.5),
        CoarseDropout(num_holes_range=(1, 8),
                      hole_height_range=(16, 48),
                      hole_width_range=(16, 48),
                      mask_fill_value=0, p=0.4),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    return train, get_val_transforms()