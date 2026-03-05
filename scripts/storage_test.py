#!/usr/bin/env python3
import os
import json
import subprocess
from datetime import datetime

NODE_NAME = os.environ.get("NODE_NAME", "unknown")

# Benchmark directory on shared filesystem (PVC mounted to /results in your workflow)
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/results/preflight-storage")

# Thresholds (tune via workflow parameters)
MIN_SEQ_READ_MBPS = float(os.environ.get("MIN_SEQ_READ_MBPS", "500"))      # MB/s
MIN_RAND_READ_IOPS = float(os.environ.get("MIN_RAND_READ_IOPS", "2000"))   # IOPS
MAX_RAND_P95_LAT_MS = float(os.environ.get("MAX_RAND_P95_LAT_MS", "5.0"))  # ms

# fio job params
SEQ_SIZE = os.environ.get("SEQ_SIZE", "8G")
SEQ_RUNTIME = int(os.environ.get("SEQ_RUNTIME", "30"))
RAND_SIZE = os.environ.get("RAND_SIZE", "8G")
RAND_RUNTIME = int(os.environ.get("RAND_RUNTIME", "30"))
RAND_NUMJOBS = int(os.environ.get("RAND_NUMJOBS", "4"))
RAND_IODEPTH = int(os.environ.get("RAND_IODEPTH", "32"))

RESULTS_FILE = f"/results/{NODE_NAME}-results.json"


storage_result = {
    "test": "storage_preflight",
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "metrics": {
        "storage_dir": STORAGE_DIR,
        "seq_read": {},
        "rand_read": {}
    },
    "errors": []
}

def record_error(msg):
    storage_result["errors"].append(msg)
    storage_result["status"] = "FAIL"

def run_cmd(cmd_list):
    try:
        out = subprocess.check_output(cmd_list, stderr=subprocess.STDOUT).decode("utf-8", "replace")
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output.decode("utf-8", "replace")

def fio_run_and_parse(name, extra_args):
    """
    Runs fio with --output-format=json and returns parsed JSON dict.
    """
    cmd = [
        "fio",
        f"--name={name}",
        f"--directory={STORAGE_DIR}",
        "--output-format=json",
        "--group_reporting=1",
    ] + extra_args

    rc, out = run_cmd(cmd)
    if rc != 0:
        record_error(f"fio job '{name}' failed (rc={rc}). Output:\n{out}")
        return None

    try:
        return json.loads(out)
    except Exception as e:
        record_error(f"Failed to parse fio JSON output for job '{name}': {e}. Raw output:\n{out}")
        return None

def fio_extract_read_metrics(fio_json):
    """
    Returns (mbps, iops, p95_lat_ms) from fio JSON for read workload.
    fio JSON is usually: {"jobs":[{"read":{...}, "job options":...}], ...}
    """
    job = fio_json["jobs"][0]
    read = job.get("read", {})

    # Bandwidth: fio's bw is typically KiB/s (in many versions), bw_bytes is bytes/s.
    bw_bytes = read.get("bw_bytes", None)
    bw_kib = read.get("bw", None)

    if bw_bytes is not None:
        mbps = float(bw_bytes) / (1024.0 * 1024.0)
    elif bw_kib is not None:
        mbps = float(bw_kib) / 1024.0
    else:
        mbps = 0.0

    iops = float(read.get("iops", 0.0))

    # Latency percentiles:
    # Prefer clat_ns percentiles if present, fallback to lat_ns.
    p95_ms = 0.0
    clat = read.get("clat_ns", {})
    lat = read.get("lat_ns", {})

    pct = None
    if isinstance(clat, dict):
        pct = clat.get("percentile", None)
    if pct is None and isinstance(lat, dict):
        pct = lat.get("percentile", None)

    if isinstance(pct, dict):
        # fio percentile keys look like "95.000000"
        v = pct.get("95.000000", None)
        if v is not None:
            p95_ms = float(v) / 1e6  # ns -> ms

    return mbps, iops, p95_ms

def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    # if it doesn't exist, create a minimal structure
    return {
        "node": NODE_NAME,
        "timestamp": datetime.now().isoformat(),
        "status": "PASS",
        "tests": []
    }

def save_results(existing):
    existing["timestamp"] = datetime.now().isoformat()
    existing["tests"].append(storage_result)
    existing["status"] = "PASS" if all(t.get("status") == "PASS" for t in existing["tests"]) else "FAIL"

    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, RESULTS_FILE)

def main():
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # Ensure fio exists
    rc, _ = run_cmd(["sh", "-c", "command -v fio"])
    if rc != 0:
        record_error("fio not found in container. Install fio or use an image that includes it.")
        existing = load_existing_results()
        save_results(existing)
        return 0

    # Sequential read test (time-based, direct I/O)
    seq_json = fio_run_and_parse("seqread", [
        "--rw=read",
        "--bs=4m",
        f"--size={SEQ_SIZE}",
        "--numjobs=1",
        "--iodepth=16",
        "--direct=1",
        "--time_based=1",
        f"--runtime={SEQ_RUNTIME}",
        f"--filename=preflight_{NODE_NAME}_randread.dat"
    ])
    if seq_json:
        mbps, iops, p95 = fio_extract_read_metrics(seq_json)
        storage_result["metrics"]["seq_read"] = {
            "mbps": mbps,
            "iops": iops,
            "p95_lat_ms": p95,
            "size": SEQ_SIZE,
            "runtime_s": SEQ_RUNTIME
        }
        if mbps < MIN_SEQ_READ_MBPS:
            record_error(f"Sequential read MB/s below threshold: {mbps:.2f} < {MIN_SEQ_READ_MBPS}")

    # Random read test
    rand_json = fio_run_and_parse("randread", [
        "--rw=randread",
        "--bs=4k",
        f"--size={RAND_SIZE}",
        f"--numjobs={RAND_NUMJOBS}",
        f"--iodepth={RAND_IODEPTH}",
        "--direct=1",
        "--time_based=1",
        f"--runtime={RAND_RUNTIME}",
        f"--filename=preflight_{NODE_NAME}_randread.dat"
    ])
    if rand_json:
        mbps, iops, p95 = fio_extract_read_metrics(rand_json)
        storage_result["metrics"]["rand_read"] = {
            "mbps": mbps,
            "iops": iops,
            "p95_lat_ms": p95,
            "size": RAND_SIZE,
            "runtime_s": RAND_RUNTIME,
            "numjobs": RAND_NUMJOBS,
            "iodepth": RAND_IODEPTH,
            "bs": "4k"
        }
        if iops < MIN_RAND_READ_IOPS:
            record_error(f"Random read IOPS below threshold: {iops:.2f} < {MIN_RAND_READ_IOPS}")
        if p95 > 0 and p95 > MAX_RAND_P95_LAT_MS:
            record_error(f"Random read p95 latency above threshold: {p95:.3f} ms > {MAX_RAND_P95_LAT_MS} ms")

    existing = load_existing_results()
    save_results(existing)
    return 0  # always success for Argo

if __name__ == "__main__":
    raise SystemExit(main())