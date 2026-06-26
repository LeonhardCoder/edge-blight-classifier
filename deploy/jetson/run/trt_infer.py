"""trt_infer.py — runner de inferencia TensorRT 10.x para la Jetson.

API moderna (set_tensor_address + execute_async_v3). Buffers fijos batch=1.
"""
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTRunner:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        self.in_shape = (1, 3, 224, 224)
        self.context.set_input_shape(self.input_name, self.in_shape)
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))

        self.h_in = cuda.pagelocked_empty(
            int(np.prod(self.in_shape)), np.float32)
        self.h_out = cuda.pagelocked_empty(
            int(np.prod(out_shape)), np.float32)
        self.d_in = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.d_in))
        self.context.set_tensor_address(self.output_name, int(self.d_out))

    def infer(self, tensor):
        np.copyto(self.h_in, tensor.ravel())
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return self.h_out.copy()

    def sync(self):
        self.stream.synchronize()
