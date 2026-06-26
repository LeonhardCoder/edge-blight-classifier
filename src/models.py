"""
Factory de modelos del TFM.

Los 4 modelos se crean desde timm con pesos preentrenados en ImageNet.
Todos comparten el mismo numero de clases y se exponen mediante una
interfaz comun para garantizar comparabilidad.
"""
import torch.nn as nn
import timm

import config


def create_model(model_key: str, pretrained: bool = True, num_classes: int = None):
    """
    Crea un modelo del registry usando timm.

    Args:
        model_key: clave del modelo (ver config.MODEL_REGISTRY)
        pretrained: si True, descarga pesos ImageNet
        num_classes: numero de clases del clasificador final
                     (por defecto config.NUM_CLASSES)
    """
    if model_key not in config.MODEL_REGISTRY:
        raise ValueError(
            f"Modelo desconocido: {model_key}. "
            f"Opciones: {list(config.MODEL_REGISTRY.keys())}"
        )

    if num_classes is None:
        num_classes = config.NUM_CLASSES

    info = config.MODEL_REGISTRY[model_key]
    model = timm.create_model(
        info["timm_name"],
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Metadatos adheridos al modelo para trazabilidad
    model.model_key = model_key
    model.family    = info["family"]
    model.img_size  = info["img_size"]
    return model


def reparameterize_if_needed(model):
    """
    RepViT y arquitecturas re-parametrizables deben fundir sus ramas
    antes de exportar a ONNX/TFLite/TensorRT. Si no se hace, se exporta
    la version multi-rama (mas lenta y pesada).

    timm expone esto via model.fuse() o model.reparameterize() segun version.
    """
    if hasattr(model, "reparameterize"):
        model.reparameterize()
        return True
    if hasattr(model, "fuse"):
        model.fuse()
        return True
    return False


def count_parameters(model) -> dict:
    """Cuenta parametros totales y entrenables."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_M":     total / 1e6,
        "trainable_M": trainable / 1e6,
    }


def print_models_summary():
    """Imprime tabla resumen de los 4 modelos del TFM."""
    print("\n" + "=" * 78)
    print(f"{'Modelo':<25} {'Familia':<15} {'Anio':<6} {'Params (M)':>12}")
    print("-" * 78)
    for key, info in config.MODEL_REGISTRY.items():
        try:
            m = create_model(key, pretrained=False)
            params = count_parameters(m)
            print(f"  {key:<23} {info['family']:<15} {info['year']:<6} {params['total_M']:>12.2f}")
            del m
        except Exception as e:
            print(f"  {key:<23} ERROR: {e}")
    print("=" * 78)
