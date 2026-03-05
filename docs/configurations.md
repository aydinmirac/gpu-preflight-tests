# Workflow Configuration Guide

This document explains the required configurations for running the **GPU
Preflight Test Workflow** using **Argo Workflows**.

The guide covers:

-   Installing Argo Workflows on your Kubernetes cluster
-   Installing the Argo CLI
-   Configuring `params.yaml`
-   Configuring the `PersistentVolumeClaim` used by the workflows

## Installing Argo Workflows

Create a namespace for Argo Workflows:

``` bash
kubectl create namespace argo
```

Export the desired version:

``` bash
ARGO_WORKFLOWS_VERSION="v3.7.10"
```

Install Argo Workflows:

``` bash
kubectl apply --server-side -n argo -f "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_WORKFLOWS_VERSION}/quick-start-minimal.yaml"
```

This installs a minimal setup of Argo Workflows inside the `argo` namespace.


## Installing the Argo CLI

The **Argo CLI** allows you to easily submit, monitor, and manage workflows from your local machine.

### macOS Installation

Download the binary:

``` bash
curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.2.6/argo-darwin-amd64.gz
```

Unzip the binary:

``` bash
gunzip argo-darwin-amd64.gz
```

Make the binary executable:

``` bash
chmod +x argo-darwin-amd64
```

Move the binary into your system path:

``` bash
mv ./argo-darwin-amd64 /usr/local/bin/argo
```

Verify the installation:

``` bash
argo version
```

### Linux Installation

Download the binary:

``` bash
curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.2.6/argo-linux-amd64.gz
```

Unzip the binary:

``` bash
gunzip argo-linux-amd64.gz
```

Make the binary executable:

``` bash
chmod +x argo-linux-amd64
```

Move the binary into your system path:

``` bash
mv ./argo-linux-amd64 /usr/local/bin/argo
```

Verify the installation:

``` bash
argo version
```

## `params.yaml` Configuration

The `params.yaml` file contains runtime parameters used by the workflow templates.

``` yaml
githubRepo: "https://github.com/aydinmirac/gpu-preflight-tests.git" # Repository containing workflow scripts or test code

nodes: '["computeinstance","computeinstance"]'  # JSON array of node names where tests will run

# GPU health check configuration
gpuHealthImage: "ghcr.io/coreweave/nccl-tests:12.8.1-devel-ubuntu22.04-nccl2.29.2-1-2276a5e"  # Container image used for GPU diagnostics
minGpuCount: "8"  # Minimum number of GPUs expected on each node
driverVersion: "570"  # Required NVIDIA driver version
cudaVersion: "12.8"  # Required CUDA version
maxTemperature: "85"  # Maximum allowed GPU temperature in °C
maxEccErrors: "0"  # Maximum allowed ECC memory errors

# NCCL Tests
expectedBandwidth: "100"  # Expected NCCL bandwidth in GB/s (adjust for your GPU model)
expectedLatency: "100"   # Expected NCCL latency in microseconds
minSeqReadMBps: "500" # Minimum sequential disk read throughput in MB/s
minRandReadIOPS: "2000" # Minimum random read IOPS for storage validation
maxRandP95LatMs: "5.0" # Maximum 95th percentile latency for random reads (milliseconds)

# PyTorch Compute Test
computeImage: "pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime"  # Container image used for GPU compute tests
minTflops: "50"  # Minimum expected GPU compute performance in TFLOPS

# Node Policies
preflightLabelKey: "preflight-status" # Node label used to mark preflight test status
preflightTaintKey: "preflight-not-ready" # Taint applied to nodes that fail tests (must exist in cluster policies)
preflightTaintEffect: "NoSchedule" # Kubernetes taint effect preventing scheduling
preflightLastRunAnnotationKey: "preflight-last-run" # Annotation storing timestamp of last preflight test
preflightFailuresAnnotationKey: "preflight-failing-tests" # Annotation storing names of failed tests
```

These parameters allow you to customize the behavior of the workflow without modifying the workflow templates directly.

## `pvc.yaml` Configuration

The workflow stores intermediate files and test results in a **PersistentVolumeClaim (PVC)**.

You can adjust:

-   storage size
-   storage class

⚠️ **Important:**\
Do **not change the PVC name**, because it is currently **hardcoded inside the workflow templates**.

Storage tests generate files of approximately **8 GB per node**.\

It is recommended to configure storage size using:

    (number_of_nodes × 8 GB) + 1 GB

Example configuration:

``` yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: preflight-tests-workspace
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
  storageClassName: csi-mounted-fs-path-sc
```
