#!/usr/bin/env python3
import subprocess
import json
import os
import sys
from datetime import datetime

# Configuration from environment variables
NODE_NAME = os.environ.get("NODE_NAME", "unknown")
MIN_GPU_COUNT = int(os.environ.get("MIN_GPU_COUNT", "2"))  # Minimum GPUs for NCCL
EXPECTED_BANDWIDTH = float(os.environ.get("EXPECTED_BANDWIDTH", "100"))  # GB/s
EXPECTED_LATENCY = float(os.environ.get("EXPECTED_LATENCY", "10"))  # microseconds

# Initialize result structure
result = {
    "node": NODE_NAME,
    "test": "nccl_intra_node",
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "metrics": {
        "gpu_count": 0,
        "tests": [],
        "summary": {
            "avg_bandwidth": 0,
            "min_bandwidth": 0,
            "avg_latency": 0,
            "max_latency": 0
        }
    },
    "errors": []
}

def record_error(msg):
    """Record an error"""
    result["errors"].append(msg)
    result["status"] = "FAIL"

def run_command(cmd):
    """Run a shell command and return output"""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode().strip()
    except subprocess.CalledProcessError as e:
        record_error(f"Command failed: {cmd}\nError: {e.output.decode()}")
        return None
    except Exception as e:
        record_error(f"Unexpected error running {cmd}: {str(e)}")
        return None

def run_nccl_test(size="1G"):
    """Run NCCL all_reduce_perf test directly"""
    # Ensure we have a valid GPU count
    gpu_count = result['metrics']['gpu_count']
    if gpu_count < MIN_GPU_COUNT:
        record_error(f"Insufficient GPUs for NCCL test. Need at least {MIN_GPU_COUNT}, found: {gpu_count}")
        return None

    # Set up GPU selection
    gpu_list = ",".join(str(i) for i in range(gpu_count))

    # Run the test directly using the correct path
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_list} /opt/nccl-tests/build/all_reduce_perf -b 8 -e {size} -f 2 -g {gpu_count}"

    output = run_command(cmd)
    if not output:
        return None

    # Parse output
    test_result = {
        "size": size,
        "bandwidth": 0,
        "latency": 0,
        "raw_output": output
    }

    for line in output.split('\n'):
        if "out-of-place" in line and "Bandwidth" in line:
            parts = line.split()
            try:
                test_result["bandwidth"] = float(parts[6])
                test_result["latency"] = float(parts[8])
            except (IndexError, ValueError):
                record_error(f"Failed to parse NCCL output: {line}")

    return test_result

def load_existing_results():
    """Load existing results file if it exists"""
    results_file = f"/results/{NODE_NAME}-results.json"
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            record_error(f"Failed to load existing results: {str(e)}")
    return None

def save_results():
    """Save results to JSON file"""
    results_file = f"/results/{NODE_NAME}-results.json"
    temp_file = f"{results_file}.tmp"

    # Load existing results if they exist
    existing_results = load_existing_results()

    if existing_results:
        # Merge with existing results
        merged_results = existing_results
        merged_results["timestamp"] = datetime.now().isoformat()

        # Get GPU count from existing results
        if "tests" in merged_results:
            for test in merged_results["tests"]:
                if test.get("test") == "gpu_health" and "metrics" in test and "gpu_count" in test["metrics"]:
                    result["metrics"]["gpu_count"] = test["metrics"]["gpu_count"]
                    break

        # If we still don't have GPU count, use fallback
        if result["metrics"]["gpu_count"] == 0:
            result["metrics"]["gpu_count"] = MIN_GPU_COUNT
            record_error(f"Could not find GPU count in existing results, using fallback: {MIN_GPU_COUNT}")

        # Add or update our test results
        test_found = False
        for i, test in enumerate(merged_results.get("tests", [])):
            if test.get("test") == "nccl_intra_node":
                merged_results["tests"][i] = result
                test_found = True
                break

        if not test_found:
            if "tests" not in merged_results:
                merged_results["tests"] = []
            merged_results["tests"].append(result)

        # Update overall status
        merged_results["status"] = "PASS" if all(
            test.get("status") == "PASS" for test in merged_results.get("tests", [])
        ) else "FAIL"

        # Write merged results
        with open(temp_file, 'w') as f:
            json.dump(merged_results, f, indent=2)
    else:
        # Create new results structure
        result["metrics"]["gpu_count"] = MIN_GPU_COUNT
        record_error(f"No existing results found, using fallback GPU count: {MIN_GPU_COUNT}")

        merged_results = {
            "node": NODE_NAME,
            "timestamp": datetime.now().isoformat(),
            "status": result["status"],
            "tests": [result]
        }

        # Write new results
        with open(temp_file, 'w') as f:
            json.dump(merged_results, f, indent=2)

    # Atomic rename
    os.rename(temp_file, results_file)
    print(f"Results written to {results_file}")

try:
    # Run NCCL tests with different message sizes
    sizes = ["1M", "32M", "256M", "1G", "4G"]
    bandwidths = []
    latencies = []

    for size in sizes:
        test_result = run_nccl_test(size=size)
        if test_result:
            result["metrics"]["tests"].append(test_result)
            bandwidths.append(test_result["bandwidth"])
            latencies.append(test_result["latency"])

            # Check against thresholds
            if test_result["bandwidth"] < EXPECTED_BANDWIDTH * 0.8:
                record_error(f"Bandwidth below threshold for size {size}: {test_result['bandwidth']} GB/s")
            if test_result["latency"] > EXPECTED_LATENCY * 1.2:
                record_error(f"Latency above threshold for size {size}: {test_result['latency']} us")

    # Calculate summary metrics
    if bandwidths:
        result["metrics"]["summary"]["avg_bandwidth"] = sum(bandwidths) / len(bandwidths)
        result["metrics"]["summary"]["min_bandwidth"] = min(bandwidths)
    if latencies:
        result["metrics"]["summary"]["avg_latency"] = sum(latencies) / len(latencies)
        result["metrics"]["summary"]["max_latency"] = max(latencies)

except Exception as e:
    record_error(f"NCCL test failed: {str(e)}")

# Save results
save_results()

# Print summary
print(f"Node {NODE_NAME} - NCCL Intra-Node Test: {result['status']}")
if result["errors"]:
    print("Errors encountered:")
    for error in result["errors"]:
        print(f"  - {error}")

sys.exit(0 if result["status"] == "PASS" else 1)