"""
construye engines TensorRT en la Jetson (FP32/FP16/INT8).

Lee los .onnx de onnx/ y, para INT8, los tensores de calibración de
onnx/calib_npy/. Los engines NO son portables: se construyen en el propio dispositivo.

Uso:
  python run/build_engines.py --all
  python run/build_engines.py --onnx models/repvit_m1.onnx --precisions fp16 int8
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("/home/victor/tfm-edge") 
sys.path.insert(0, str(ROOT))

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_DIR = ROOT / "models"
CALIB_DIR = ONNX_DIR / "calib_npy"
CALIB_DIR = ROOT / "calib_npy"
ENG_DIR = ROOT / "engines_new"


def _calibrator(cache_path):
    class Cal(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.files = sorted(CALIB_DIR.glob("calib_*.npy"))
            if not self.files:
                raise FileNotFoundError(
                    f"No hay tensores de calibración en {CALIB_DIR}. "
                    "Cópialos desde Colab (artifacts/onnx/calib_npy/).")
            self.idx = 0
            sample = np.load(self.files[0])
            self.dbuf = cuda.mem_alloc(sample.nbytes)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.files):
                return None
            arr = np.ascontiguousarray(np.load(self.files[self.idx]), np.float32)
            cuda.memcpy_htod(self.dbuf, arr)
            self.idx += 1
            return [int(self.dbuf)]

        def read_calibration_cache(self):
            return cache_path.read_bytes() if cache_path.exists() else None

        def write_calibration_cache(self, data):
            cache_path.write_bytes(data)

    return Cal()


def build(onnx_path, precision):
    onnx_path = Path(onnx_path)
    ENG_DIR.mkdir(parents=True, exist_ok=True)
    eng_path = ENG_DIR / f"{onnx_path.stem}_{precision}.engine"

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    #parser = trt.OnnxParser(network, TRT_LOGGER)
    #with open(onnx_path, "rb") as f:
    #    if not parser.parse(f.read()):
    #        for i in range(parser.num_errors):
    #            print(parser.get_error(i))
    #        raise RuntimeError(f"Fallo al parsear {onnx_path.name}")
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Fallo al parsear {onnx_path.name}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = _calibrator(
            ENG_DIR / f"{onnx_path.stem}_int8_calib.cache")

    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError(f"build devolvió None ({onnx_path.name}/{precision})")
    eng_path.write_bytes(serialized)
    print(f"{onnx_path.stem} {precision.upper():4s} -> {eng_path.name} "
          f"({eng_path.stat().st_size/1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--precisions", nargs="+",
                    default=["fp32", "fp16", "int8"],
                    choices=["fp32", "fp16", "int8"])
    args = ap.parse_args()
    onnxs = (sorted(glob.glob(str(ONNX_DIR / "*.onnx"))) if args.all
             else ([args.onnx] if args.onnx else []))
    if not onnxs:
        ap.error("Indica --onnx <ruta> o --all")
    for o in onnxs:
        for p in args.precisions:
            build(o, p)


if __name__ == "__main__":
    main()
