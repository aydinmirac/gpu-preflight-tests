#!/usr/bin/env python3
import os
import json
import subprocess
import sys

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")
NODES = json.loads(os.environ.get("NODES", "[]"))

LABEL_KEY = os.environ.get("PREFLIGHT_LABEL_KEY", "preflight-status")
TAINT_KEY = os.environ.get("PREFLIGHT_TAINT_KEY", "preflight-not-ready")
TAINT_EFFECT = os.environ.get("PREFLIGHT_TAINT_EFFECT", "NoSchedule")

LAST_RUN_ANN_KEY = os.environ.get("PREFLIGHT_LAST_RUN_ANNOTATION_KEY", "preflight-last-run")
FAILURES_ANN_KEY = os.environ.get("PREFLIGHT_FAILURES_ANNOTATION_KEY", "preflight-failing-tests")

def run(cmd):
    subprocess.check_call(cmd)

def node_result_path(node):
    return os.path.join(RESULTS_DIR, f"{node}-results.json")

def extract_failing_tests(data):
    failing = []
    for t in data.get("tests", []):
        if t.get("status") == "FAIL":
            failing.append(t.get("test", "unknown"))
    return failing

def get_node_taints(node):
    """Return list of taint strings currently on the node."""
    result = subprocess.run(
        ["kubectl", "get", "node", node, "-o", "jsonpath={.spec.taints}"],
        capture_output=True, text=True
    )
    raw = result.stdout.strip()
    if not raw or raw == "null":
        return []
    taints = json.loads(raw)
    return [f"{t['key']}={t.get('value', '')}:{t['effect']}" for t in taints]

def taint_exists(node, key, effect):
    """Check if a specific taint key+effect is already applied."""
    result = subprocess.run(
        ["kubectl", "get", "node", node, "-o", "jsonpath={.spec.taints}"],
        capture_output=True, text=True
    )
    raw = result.stdout.strip()
    if not raw or raw == "null":
        return False
    taints = json.loads(raw)
    return any(t.get("key") == key and t.get("effect") == effect for t in taints)

def apply_taint(node, key, value, effect):
    """Apply taint, overwriting if it already exists."""
    if taint_exists(node, key, effect):
        subprocess.call(["kubectl", "taint", "node", node, f"{key}:{effect}-"])
    run(["kubectl", "taint", "node", node, f"{key}={value}:{effect}"])

def remove_taint(node, key, effect):
    """Remove taint only if it exists, avoiding 'not found' errors."""
    if taint_exists(node, key, effect):
        run(["kubectl", "taint", "node", node, f"{key}:{effect}-"])
    else:
        print(f"[INFO] Taint {key}:{effect} not present on {node}, skipping removal.")

def main():
    if not NODES:
        print("No nodes provided in NODES env", file=sys.stderr)
        return 0

    for node in NODES:
        path = node_result_path(node)

        status = "not-ready"
        ts = ""
        failing_tests = []

        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            status = "ready" if data.get("status") == "PASS" else "not-ready"
            ts = data.get("timestamp", "")
            failing_tests = extract_failing_tests(data)
        else:
            failing_tests = ["missing-results"]

        # Label node with status
        run(["kubectl", "label", "node", node, f"{LABEL_KEY}={status}", "--overwrite"])

        # Annotations for metadata
        if ts:
            run(["kubectl", "annotate", "node", node, f"{LAST_RUN_ANN_KEY}={ts}", "--overwrite"])

        if failing_tests:
            run(["kubectl", "annotate", "node", node, f"{FAILURES_ANN_KEY}={','.join(failing_tests)}", "--overwrite"])
        else:
            subprocess.call(["kubectl", "annotate", "node", node, f"{FAILURES_ANN_KEY}-"])

        # Taint policy
        if status != "ready":
            apply_taint(node, TAINT_KEY, "true", TAINT_EFFECT)
        else:
            remove_taint(node, TAINT_KEY, TAINT_EFFECT)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())