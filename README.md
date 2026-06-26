# Edge Blight Classifier

Este repositorio contiene el código desarrollado como parte del Trabajo Final de Máster (TFM) Máster Universitario en Ingeniería Informática , en el área de Inteligencia Artificial, de la Universitat Oberta de Catalunya (UOC). 
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
  RPi4:    ONNX-FP16, FP32
  Jetson:  FP32, TensorRT-INT8, ONNX-FP16

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
│   ├── exporter.py            
│   ├── edge_bench.py          # runners y mediciones Edge
├── scripts/
│   ├── 01_prepare_corpus.py   # construye train/val/test.csv
│   ├── 02_train.py            # entrena los 4 modelos
│   ├── 03_export.py           # exporta a ONNX y TFLite
│   ├── 04_edge_benchmark.py   # ejecuta en RPi4 / Jetson
│   └── 05_make_plots.py       # genera todas las figuras
├── notebooks/
│   ├── EntrenamientoModelos.py   # flujo de entrenamiento y exportacion en Colab
├── deploy/
│   ├── jetson/
|   │   └── script/
|   └── rpi4/
|       └── script/
├── scripts/    
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
export TFM_ENV=local           # o colab
export TFM_DATA_BASE=/ruta/a/tus/datasets   # opcional
python config.py               # verificar configuracion
```

### Pipeline completo (en tu maquina de entrenamiento)

```bash
# 1. Construir corpus consolidado
python scripts/01_prepare_corpus.py

# 2. Entrenar los 4 modelos
python scripts/02_train.py

# 3. Exportar a ONNX FP32 y FP16
python scripts/03_export.py --formats onnx 
```

### En la Jetson Orin Nano

```bash
# Transferir los modelos exportados 
# Construir los .engine int8
python run/build_engines.py --all
```

```bash

python deploy/jetson/run/04_jetson_benchmark_energy.py \
  --engines-dir ~/tfm-edge/engines_new \
  --test-csv    /home/victor/Documents/edge-blight-classifier/project/outputs/corpus/test.csv \
  --images-root /home/victor/Documents/dataset \
  --device-tag  jetson_orin_nano \
  --warmup 50 --idle-seconds 30 \
  --out ~/tfm-edge/results/edge_results_jetson_energy.csv
```

```bash
for e in deploy/jetson/engines_new/*.engine; do
  echo "=== $(basename $e) ==="
  python run/gen_preds.py --engine "$e"
done

```

```bash
python deploy/jetson/script/mcnemar_consolidado.py 
```
###  En la Raspberry Pi 4 

```bash
python scripts/04_edge_benchmark.py \
        --models-dir ~/tfm-edge/models \
        --test-csv   ~/tfm-edge/data/test_edge.csv \
        --images-root ~/tfm-edge/data/dataset \
        --device-tag rpi4 \
        --warmup 50 \
        --out ~/tfm-edge/results/edge_results_rpi.csv

python scripts/mcnemar_rpi.py 
```

### Generar figuras de la memoria

```bash
# De vuelta en tu maquina principal, consolida ambos resultados
python scripts/05_make_plots.py
# Genera todas las figuras en outputs/figures/
```

## Reproducibilidad

- Semilla fija en `settings.conf` ([experiment].seed)
- Splits estratificados deterministicos
- Hash SHA256 del corpus completo en `corpus_summary.json`
- Hash MD5 por imagen en los CSVs
- Verificacion ONNX vs PyTorch tras exportacion
