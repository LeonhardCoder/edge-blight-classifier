# Edge Blight Classifier

Evaluacion comparativa del rendimiento de modelos CNN y Vision Transformers
para la deteccion de tizon en hojas de papa en dispositivos Edge
(Raspberry Pi 4 y NVIDIA Jetson Orin Nano).

## Enfoque del TFM

El nucleo del trabajo es el **analisis exhaustivo en Edge**, no el entrenamiento.
El entrenamiento es un instrumento metodologico para producir modelos
comparables; lo que se compara realmente es su comportamiento al desplegarse
con distintos formatos y cuantizaciones en cada dispositivo.

```
DATASET CONSOLIDADO (mezcla de 4 fuentes):
  PlantVillage + Tilahun + Mendeley UE + Quizhpe
  Split unico 70/15/15 estratificado por (clase, origen)

ENTRENAMIENTO (un solo run por modelo, mismo protocolo):
  Fine-tuning desde ImageNet
  4 modelos: MobileNetV4, EfficientNetV2-B0, EfficientFormerV2-S0, RepViT-M1

MATRIZ EDGE (corazon del TFM):
  RPi4:    TFLite-INT8, ONNX-FP16
  Jetson:  TensorRT-FP16, TensorRT-INT8, ONNX-FP16

METRICAS POR COMBINACION:
  Exactitud, F1-macro, recall por clase
  Latencia media, P95, P99 sobre 500 inferencias
  Memoria RAM pico, tamano del modelo
```

## Estructura

```
.
├── configs/
│   └── settings.conf          # ruta y parametros por entorno
├── config.py                  # carga settings.conf y expone constantes
├── src/
│   ├── dataset.py             # PyTorch Dataset desde CSV
│   ├── transforms.py          # augmentaciones
│   ├── models.py              # factory timm
│   ├── trainer.py             # bucle de entrenamiento
│   ├── exporter.py            # PyTorch -> ONNX/TFLite/comandos TRT
│   ├── edge_bench.py          # runners y mediciones Edge
│   └── plots.py               # figuras
├── scripts/
│   ├── 01_prepare_corpus.py   # construye train/val/test.csv
│   ├── 02_train.py            # entrena los 4 modelos
│   ├── 03_export.py           # exporta a ONNX y TFLite
│   ├── 04_edge_benchmark.py   # ejecuta en RPi4 / Jetson
│   └── 05_make_plots.py       # genera todas las figuras
└── outputs/
    ├── checkpoints/           # *.pth entrenados
    ├── exported/              # *.onnx, *.tflite, *.engine
    ├── metrics/               # train_*.csv, edge_results_*.csv
    ├── figures/               # *.png para la memoria
    └── corpus/                # train/val/test.csv, corpus_summary.json
```

## Flujo de uso

### Configurar entorno

Editar `configs/settings.conf` para que `[local].data_base` apunte a la
carpeta donde tienes los datasets. Alternativamente:

```bash
export TFM_ENV=local           # o colab / edge_rpi / edge_jetson
export TFM_DATA_BASE=/ruta/a/tus/datasets   # opcional
python config.py               # verificar configuracion
```

### Pipeline completo (en tu maquina de entrenamiento)

```bash
# 1. Construir corpus consolidado
python scripts/01_prepare_corpus.py

# 2. Entrenar los 4 modelos
python scripts/02_train.py

# 3. Exportar a ONNX y TFLite
python scripts/03_export.py
```

### En la Raspberry Pi 4

```bash
# Transferir los modelos exportados a la RPi (scp/rsync)
# Instalar tflite-runtime y onnxruntime
export TFM_ENV=edge_rpi
python scripts/04_edge_benchmark.py
# Genera outputs/metrics/edge_results_rpi.csv
```

### En la Jetson Orin Nano

```bash
# Transferir los .onnx a la Jetson
# Construir los .engine con trtexec (comando lo imprime 03_export.py):
trtexec --onnx=mobilenetv4_conv_small.onnx \
        --saveEngine=mobilenetv4_conv_small_fp16.engine --fp16

export TFM_ENV=edge_jetson
python scripts/04_edge_benchmark.py
# Genera outputs/metrics/edge_results_jetson.csv
```

### Generar figuras de la memoria

```bash
# De vuelta en tu maquina principal, consolida ambos resultados
python scripts/05_make_plots.py
# Genera todas las figuras en outputs/figures/
```

## Cambio de entorno

Tres formas equivalentes de seleccionar entorno:

```bash
# Opcion A: editar [active].env en configs/settings.conf
# Opcion B: por comando puntual
TFM_ENV=edge_rpi python scripts/04_edge_benchmark.py
# Opcion C: en .bashrc del dispositivo
echo 'export TFM_ENV=edge_rpi' >> ~/.bashrc
```

## Reproducibilidad

- Semilla fija en `settings.conf` ([experiment].seed)
- Splits estratificados deterministicos
- Hash SHA256 del corpus completo en `corpus_summary.json`
- Hash MD5 por imagen en los CSVs
- Verificacion ONNX vs PyTorch tras exportacion
