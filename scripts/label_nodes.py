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
            try:
                run(["kubectl", "taint", "node", node, f"{TAINT_KEY}=true:{TAINT_EFFECT}", "--overwrite"])
            except subprocess.CalledProcessError:
                run(["kubectl", "taint", "node", node, f"{TAINT_KEY}=true:{TAINT_EFFECT}"])
        else:
            subprocess.call(["kubectl", "taint", "node", node, f"{TAINT_KEY}:{TAINT_EFFECT}-"])

    return 0

if __name__ == "__main__":
    raise SystemExit(main())