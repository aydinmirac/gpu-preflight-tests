#!/usr/bin/env python3
import os, json, time
from datetime import datetime

NODE_NAME = os.environ.get("NODE_NAME", "unknown")
RESULTS_FILE = f"/results/{NODE_NAME}-results.json"

# Test parameters (override via env)
MATMUL_N = int(os.environ.get("MATMUL_N", "16384"))     # matrix dimension
WARMUP = int(os.environ.get("WARMUP", "5"))
ITERS = int(os.environ.get("ITERS", "20"))
DTYPE = os.environ.get("DTYPE", "fp16").lower()         # fp16 or bf16
MIN_TFLOPS = float(os.environ.get("MIN_TFLOPS", "50"))  # per GPU, very conservative

compute_result = {
    "test": "gpu_compute_matmul",
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "metrics": {
        "matmul_n": MATMUL_N,
        "dtype": DTYPE,
        "per_gpu": []
    },
    "errors": []
}

def record_error(msg):
    compute_result["errors"].append(msg)
    compute_result["status"] = "FAIL"

def load_existing():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {"node": NODE_NAME, "timestamp": datetime.now().isoformat(), "status": "PASS", "tests": []}

def save(existing):
    existing["timestamp"] = datetime.now().isoformat()
    existing["tests"].append(compute_result)
    existing["status"] = "PASS" if all(t.get("status") == "PASS" for t in existing["tests"]) else "FAIL"
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, RESULTS_FILE)

def main():
    try:
        import torch
    except Exception as e:
        record_error(f"PyTorch not available in image: {e}")
        existing = load_existing()
        save(existing)
        return 0

    if not torch.cuda.is_available():
        record_error("CUDA not available in PyTorch (torch.cuda.is_available() is False)")
        existing = load_existing()
        save(existing)
        return 0

    dtype = torch.float16 if DTYPE == "fp16" else torch.bfloat16
    ngpu = torch.cuda.device_count()

    torch.backends.cuda.matmul.allow_tf32 = True

    for dev in range(ngpu):
        try:
            torch.cuda.set_device(dev)
            device = torch.device(f"cuda:{dev}")

            # Allocate
            a = torch.randn((MATMUL_N, MATMUL_N), device=device, dtype=dtype)
            b = torch.randn((MATMUL_N, MATMUL_N), device=device, dtype=dtype)

            # Warmup
            for _ in range(WARMUP):
                c = a @ b
            torch.cuda.synchronize(device)

            # Timed
            t0 = time.time()
            for _ in range(ITERS):
                c = a @ b
            torch.cuda.synchronize(device)
            t1 = time.time()

            avg_s = (t1 - t0) / ITERS

            # FLOPs for GEMM ~ 2*N^3
            flops = 2.0 * (MATMUL_N ** 3)
            tflops = (flops / avg_s) / 1e12

            entry = {
                "gpu": dev,
                "avg_time_s": avg_s,
                "tflops": tflops
            }
            compute_result["metrics"]["per_gpu"].append(entry)

            if tflops < MIN_TFLOPS:
                record_error(f"GPU {dev} matmul too slow: {tflops:.2f} TFLOP/s < {MIN_TFLOPS}")

        except RuntimeError as e:
            record_error(f"GPU {dev} runtime error during matmul: {e}")
        except Exception as e:
            record_error(f"GPU {dev} unexpected error: {e}")

    existing = load_existing()
    save(existing)
    return 0 

if __name__ == "__main__":
    raise SystemExit(main())