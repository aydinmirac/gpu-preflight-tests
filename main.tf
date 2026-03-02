resource "kubernetes_namespace" "preflight" {
  metadata {
    name = "preflight"
  }
}

resource "kubernetes_persistent_volume_claim" "preflight_results" {
  metadata {
    name      = "preflight-results"
    namespace = kubernetes_namespace.preflight.metadata[0].name
  }

  spec {
    access_modes = ["ReadWriteMany"]
    storage_class_name = "compute-csi-default-sc"
    resources {
      requests = {
        storage = "1Gi"
      }
    }
  }
}


resource "helm_release" "gpu_node_health" {
  name       = "gpu-node-health"
  chart      = "./charts/gpu-node-health"
  namespace  = "preflight"

  create_namespace = true

  set {
    name  = "expected.minGpuCount"
    value = "1"
  }

  set {
    name  = "nodeSelector.nvidia\\.com/gpu\\.count"
    value = "1"
  }

}

resource "helm_release" "nccl_intranode" {
  name       = "nccl-intranode-test"
  chart      = "./charts/nccl-intranode-test"
  namespace  = "preflight"

  create_namespace = true

  depends_on = [
    helm_release.gpu_node_health
  ]
}

resource "helm_release" "storage_test" {
  name       = "storage-performance-test"
  chart      = "./charts/storage-performance-test"
  namespace  = "preflight"

  create_namespace = true

  depends_on = [
    helm_release.nccl_intranode
  ]
}

resource "helm_release" "telemetry_aggregator" {
  name       = "telemetry-aggregator"
  chart      = "./charts/telemetry-aggregator"
  namespace  = "preflight"

  create_namespace = true

  depends_on = [
    helm_release.storage_test
  ]
}

resource "helm_release" "cluster_dashboard" {
  name       = "cluster-status"
  chart      = "./charts/cluster-status"
  namespace  = "preflight"

  create_namespace = true

  depends_on = [
    helm_release.telemetry_aggregator
  ]
}