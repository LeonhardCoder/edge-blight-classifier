import time
import numpy as np


def measure_latency(infer_fn, n_warmup=50, n_runs=750, cuda_sync=None):
    for _ in range(n_warmup):
        infer_fn()
    if cuda_sync:
        cuda_sync()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        infer_fn()
        if cuda_sync:
            cuda_sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std()),
            "p50_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "fps": float(1000.0 / arr.mean()), "n": int(len(arr))}
