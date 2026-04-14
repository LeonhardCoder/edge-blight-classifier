# edge-blight-classifier
Este repositorio contiene el código desarrollado como parte del Trabajo Final de Máster (TFM)  Máster Universitario en Ingeniería Informática , en el  área de Inteligencia Artificial, de la Universitat Oberta de Catalunya (UOC). de la Universitat Oberta de Catalunya (UOC).

## Modelos evaluados

| Familia | Modelo |
|---------|--------|
| CNN | MobileNetV3 |
| CNN | EfficientNet-B0 |
| Vision Transformer | ViT-Tiny |
| Vision Transformer | MobileViT |

## Dataset

Se utiliza el dataset **PlantVillage**, con tres clases de clasificación:

- `Healthy` — hoja sana
- `Early Blight` — tizón temprano (*Alternaria solani*)
- `Late Blight` — tizón tardío (*Phytophthora infestans*)

> Las imágenes no se incluyen en este repositorio.  
> Para reproducir los experimentos, descarga el dataset desde:  
> 

## Hardware de evaluación Edge

- Raspberry Pi 4 Model B (4GB RAM)
- NVIDIA Jetson Orin Nano

## Métricas de evaluación

**Calidad predictiva:** Accuracy, F1-Score macro, Recall por clase  
**Umbrales mínimos de Recall:** Late Blight ≥ 90% · Early Blight ≥ 80%  
**Eficiencia Edge:** latencia de inferencia, uso de memoria RAM, consumo energético  
**Optimización:** cuantización PTQ-FP16 con umbral de degradación máxima del 2% en F1
## Entorno de desarrollo

- Python 3.12
- PyTorch / TensorFlow
- Google Colab (entrenamiento)
- TensorFlow Lite (conversión y cuantización para Edge)

## Autor

**[Tu nombre completo]**  
Máster Universitario de Ingeniería Informática— UOC 