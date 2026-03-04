#!/usr/bin/env python3
import subprocess
import json
import os
import sys
from datetime import datetime

# Configuration from environment variables
NODE_NAME = os.environ.get("NODE_NAME", "unknown")
EXPECTED_DRIVER_VERSION = os.environ.get("DRIVER_VERSION", "530")
EXPECTED_CUDA_VERSION = os.environ.get("CUDA_VERSION", "12.2")
EXPECTED_GPU_COUNT = int(os.environ.get("MIN_GPU_COUNT", "8"))
MAX_TEMPERATURE = int(os.environ.get("MAX_TEMPERATURE", "85"))
MAX_ECC_ERRORS = int(os.environ.get("MAX_ECC_ERRORS", "0"))

# Initialize result structure for this test
gpu_health_result = {
    "test": "gpu_health",
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "metrics": {
        "driver_version": None,
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": []
    },
    "errors": []
}

# Initialize the aggregated results structure
result = {
    "node": NODE_NAME,
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "tests": []
}

def record_error(msg):
    """Record an error (doesn't exit, just collects errors)"""
    gpu_health_result["errors"].append(msg)
    gpu_health_result["status"] = "FAIL"
    result["status"] = "FAIL"

def run_command(cmd):
    """Run a shell command and return output"""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    except subprocess.CalledProcessError as e:
        record_error(f"Command failed: {cmd}\nError: {e.output.decode()}")
        return None
    except Exception as e:
        record_error(f"Unexpected error running {cmd}: {str(e)}")
        return None

# Check NVIDIA Driver Version
try:
    driver_version = run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if driver_version:
        # Take first line if multiple GPUs
        driver_version = driver_version.split('\n')[0]
        gpu_health_result["metrics"]["driver_version"] = driver_version
        if not driver_version.startswith(EXPECTED_DRIVER_VERSION):
            record_error(f"Driver version mismatch. Expected: {EXPECTED_DRIVER_VERSION}, Found: {driver_version}")
    else:
        record_error("Failed to get driver version")
except Exception as e:
    record_error(f"Driver version check failed: {str(e)}")

# Check CUDA Version
try:
    # Try nvcc first
    cuda_version = run_command(["nvcc", "--version"])
    if cuda_version and "release" in cuda_version.lower():
        # Extract version from nvcc output
        for line in cuda_version.split('\n'):
            if "release" in line.lower():
                version = line.split()[-1]
                gpu_health_result["metrics"]["cuda_version"] = version
                if EXPECTED_CUDA_VERSION not in version:
                    record_error(f"CUDA version mismatch. Expected: {EXPECTED_CUDA_VERSION}, Found: {version}")
                break
    else:
        # Fall back to nvidia-smi
        cuda_version = run_command(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])
        if cuda_version:
            cuda_version = cuda_version.split('\n')[0]
            gpu_health_result["metrics"]["cuda_version"] = cuda_version
            if EXPECTED_CUDA_VERSION not in cuda_version:
                record_error(f"CUDA version mismatch. Expected: {EXPECTED_CUDA_VERSION}, Found: {cuda_version}")
        else:
            record_error("Failed to get CUDA version")
except Exception as e:
    record_error(f"CUDA version check failed: {str(e)}")

# Check GPU Count and Health
try:
    gpu_list = run_command(["nvidia-smi", "-L"])
    if gpu_list:
        gpu_count = len(gpu_list.strip().split('\n'))
        gpu_health_result["metrics"]["gpu_count"] = gpu_count
        if gpu_count < EXPECTED_GPU_COUNT:
            record_error(f"Insufficient GPUs. Expected: {EXPECTED_GPU_COUNT}, Found: {gpu_count}")
    else:
        record_error("Failed to get GPU count")

    # Get detailed GPU info
    gpu_info = run_command([
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,ecc.errors.uncorrected.volatile.device_memory,utilization.gpu,power.draw",
        "--format=csv,noheader"
    ])

    if gpu_info:
        for line in gpu_info.split('\n'):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                record_error(f"Unexpected GPU info format: {line}")
                continue

            try:
                gpu_index, name, temp, ecc_errors, utilization, power = parts
                gpu_data = {
                    "index": int(gpu_index),
                    "name": name,
                    "temperature": int(temp.split()[0]),
                    "ecc_errors": int(ecc_errors),
                    "utilization": utilization,
                    "power": power
                }
                gpu_health_result["metrics"]["gpus"].append(gpu_data)

                # Check temperature
                if int(temp.split()[0]) > MAX_TEMPERATURE:
                    record_error(f"GPU {gpu_index} temperature too high: {temp}")

                # Check ECC errors
                if int(ecc_errors) > MAX_ECC_ERRORS:
                    record_error(f"GPU {gpu_index} has ECC errors: {ecc_errors}")
            except Exception as e:
                record_error(f"Error processing GPU info: {line}\nError: {str(e)}")
    else:
        record_error("Failed to get GPU health information")
except Exception as e:
    record_error(f"GPU health check failed: {str(e)}")

# Add the test result to the aggregated results
result["tests"].append(gpu_health_result)

# Write results to JSON file
try:
    os.makedirs("/results", exist_ok=True)
    output_file = f"/results/{NODE_NAME}-results.json"

    temp_file = f"{output_file}.tmp"
    with open(temp_file, "w") as f:
        json.dump(result, f, indent=2)

    os.rename(temp_file, output_file)
    print(f"Results written to {output_file}")
except Exception as e:
    print(f"Failed to write results: {str(e)}", file=sys.stderr)
    result["status"] = "FAIL"
    if "errors" not in gpu_health_result:
        gpu_health_result["errors"] = []
    gpu_health_result["errors"].append(f"Failed to write results: {str(e)}")

# Print summary for logging
print(f"Node {NODE_NAME} - GPU Health Check: {gpu_health_result['status']}")
if gpu_health_result["errors"]:
    print("Errors encountered:")
    for error in gpu_health_result["errors"]:
        print(f"  - {error}")

# Exit with status code (0 for PASS, 1 for FAIL)
sys.exit(0)