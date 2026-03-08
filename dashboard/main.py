import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

app = FastAPI(title="GPU Cluster Health Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_node_results():
    nodes = []
    if not RESULTS_DIR.exists():
        return nodes
    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                nodes.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    return nodes


def summarize_cluster(nodes: list) -> dict:
    if not nodes:
        return {
            "total_nodes": 0,
            "healthy_nodes": 0,
            "failed_nodes": 0,
            "total_gpus": 0,
            "cluster_status": "NO_DATA",
            "gpu_model": None,
        }

    total_nodes = len(nodes)
    healthy_nodes = sum(1 for n in nodes if n.get("status") == "PASS")
    failed_nodes = total_nodes - healthy_nodes

    total_gpus = 0
    gpu_model = None
    for node in nodes:
        for test in node.get("tests", []):
            if test.get("test") == "gpu_health":
                total_gpus += test["metrics"].get("gpu_count", 0)
                gpus = test["metrics"].get("gpus", [])
                if gpus and gpu_model is None:
                    gpu_model = gpus[0].get("name")
                break

    cluster_status = "PASS" if failed_nodes == 0 else ("FAIL" if healthy_nodes == 0 else "DEGRADED")

    return {
        "total_nodes": total_nodes,
        "healthy_nodes": healthy_nodes,
        "failed_nodes": failed_nodes,
        "total_gpus": total_gpus,
        "cluster_status": cluster_status,
        "gpu_model": gpu_model,
    }


@app.get("/api/cluster")
def get_cluster():
    nodes = load_node_results()
    summary = summarize_cluster(nodes)
    return {
        "summary": summary,
        "nodes": nodes,
    }


@app.get("/api/nodes/{node_id}")
def get_node(node_id: str):
    nodes = load_node_results()
    for node in nodes:
        if node.get("node") == node_id:
            return node
    raise HTTPException(status_code=404, detail=f"Node {node_id} not found")


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve static frontend
frontend_dir = Path(__file__).parent / "static"

@app.get("/")
def serve_frontend():
    index = frontend_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail=f"Frontend not found. Expected: {index}")
    return FileResponse(str(index))

app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")