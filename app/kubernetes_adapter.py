"""Kubernetes GPU Job adapter used by the API and controller."""
from datetime import datetime
import os

class KubernetesAdapter:
    def __init__(self):
        self.namespace = os.getenv("K8S_NAMESPACE", "gpuops")
        try:
            from kubernetes import client, config
            try: config.load_incluster_config()
            except Exception: config.load_kube_config()
            self.batch = client.BatchV1Api(); self.core = client.CoreV1Api(); self.available = True
        except Exception as exc:
            self.available = False; self.error = type(exc).__name__

    def submit_gpu_job(self, name, image="gpu-cluster-ops:dev", steps=100, gpu_count=1):
        if not self.available: return {"backend":"kubernetes", "status":"unavailable", "error":getattr(self,"error","client_missing")}
        from kubernetes import client
        job = client.V1Job(metadata=client.V1ObjectMeta(name=name, labels={"app":"gpu-training", "gpuops/job":name}), spec=client.V1JobSpec(backoff_limit=2, template=client.V1PodTemplateSpec(metadata=client.V1ObjectMeta(labels={"app":"gpu-training", "gpuops/job":name}), spec=client.V1PodSpec(restart_policy="Never", containers=[client.V1Container(name="trainer", image=image, command=["python3","training/train.py","--steps",str(steps)], resources=client.V1ResourceRequirements(limits={"nvidia.com/gpu": str(gpu_count)}))]))))
        created = self.batch.create_namespaced_job(self.namespace, job)
        return {"backend":"kubernetes", "status":"submitted", "name":created.metadata.name, "namespace":self.namespace}

    def list_training_jobs(self):
        if not self.available: return {"backend":"kubernetes", "status":"unavailable", "error":getattr(self,"error","client_missing")}
        rows = self.batch.list_namespaced_job(self.namespace, label_selector="app=gpu-training").items
        return {"backend":"kubernetes", "status":"ok", "items":[{"name":j.metadata.name, "active":j.status.active or 0, "succeeded":j.status.succeeded or 0, "failed":j.status.failed or 0} for j in rows]}
