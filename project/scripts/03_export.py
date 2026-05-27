"""
Exporta los modelos entrenados a formatos Edge.

Por defecto exporta ONNX y TFLite-INT8. El motor TensorRT se construye
EN LA JETSON (este script imprime el comando trtexec necesario).

Uso:
    python scripts/03_export.py                       # todos los modelos
    python scripts/03_export.py --models repvit_m1
    python scripts/03_export.py --formats onnx        # solo ONNX
    python scripts/03_export.py --skip-verify         # no verificar ONNX
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from src.exporter import (
    export_onnx, verify_onnx,
    export_tflite_int8, build_tensorrt_command,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--formats", nargs="+",
                        default=["onnx", "tflite", "trt_cmd"],
                        help="Formatos a generar: onnx, tflite, trt_cmd")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--calibration-csv", default=None,
                        help="CSV con imagenes de calibracion INT8 "
                             "(por defecto: train.csv)")
    args = parser.parse_args()

    config.ensure_dirs()
    models = args.models or list(config.MODEL_REGISTRY.keys())
    calib_csv = args.calibration_csv or str(config.CORPUS_DIR / "train.csv")

    print("=" * 70)
    print("EXPORTACION DE MODELOS A FORMATOS EDGE")
    print("=" * 70)
    print(f"Modelos: {models}")
    print(f"Formatos: {args.formats}")
    print(f"Calibracion INT8: {calib_csv}")

    for model_key in models:
        ckpt = config.CHECKPOINTS_DIR / f"{model_key}_best.pth"
        if not ckpt.exists():
            print(f"\n[SKIP] {model_key}: no existe checkpoint ({ckpt})")
            continue

        print(f"\n>>> {model_key}")

        # ONNX
        if "onnx" in args.formats:
            onnx_path = export_onnx(model_key)
            if not args.skip_verify:
                verify_onnx(model_key)

        # TFLite INT8
        if "tflite" in args.formats:
            export_tflite_int8(model_key, calibration_csv=calib_csv)

        # Comando trtexec (no se ejecuta aqui)
        if "trt_cmd" in args.formats:
            build_tensorrt_command(model_key, precision="fp16")
            build_tensorrt_command(model_key, precision="int8")

    print("\n" + "=" * 70)
    print("EXPORTACION COMPLETA")
    print(f"Artefactos en: {config.EXPORTED_DIR}")
    print("Siguiente paso: transferir los .onnx a la Jetson y ejecutar trtexec")
    print("=" * 70)


if __name__ == "__main__":
    main()
