"""Idempotent GpuWorkload controller with Kubernetes Lease leadership."""
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone

from kubernetes import client, config

GROUP = "gpuops.nvidia.example"
VERSION = "v1alpha1"
PLURAL = "gpuworkloads"
LEASE_NAME = os.getenv("LEADER_ELECTION_LEASE", "gpuops-workload-controller")
NAMESPACE = os.getenv("K8S_NAMESPACE", "gpuops")
log = logging.getLogger("gpuworkload-controller")


class LeaseElector:
    def __init__(self, api):
        self.api = api
        self.identity = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self.duration = int(os.getenv("LEADER_LEASE_SECONDS", "30"))

    def is_leader(self):
        now = datetime.now(timezone.utc)
        try:
            lease = self.api.read_namespaced_lease(LEASE_NAME, NAMESPACE)
            spec = lease.spec
            holder = spec.holder_identity
            renewed = spec.renew_time
            expired = not renewed or (now - renewed) > timedelta(seconds=self.duration)
            if holder != self.identity and not expired:
                return False
            spec.holder_identity = self.identity
            spec.lease_duration_seconds = self.duration
            spec.renew_time = now
            self.api.replace_namespaced_lease(LEASE_NAME, NAMESPACE, lease)
            return True
        except client.exceptions.ApiException as exc:
            if exc.status != 404:
                log.warning("leader_election_error status=%s", exc.status)
                return False
            try:
                body = client.V1Lease(
                    metadata=client.V1ObjectMeta(name=LEASE_NAME, namespace=NAMESPACE),
                    spec=client.V1LeaseSpec(holder_identity=self.identity, lease_duration_seconds=self.duration, renew_time=now),
                )
                self.api.create_namespaced_lease(NAMESPACE, body)
                return True
            except client.exceptions.ApiException:
                return False


class GpuWorkloadController:
    def __init__(self):
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self.custom = client.CustomObjectsApi()
        self.batch = client.BatchV1Api()
        self.leases = client.CoordinationV1Api()
        self.elector = LeaseElector(self.leases)

    def list_workloads(self):
        return self.custom.list_namespaced_custom_object(GROUP, VERSION, NAMESPACE, PLURAL).get("items", [])

    def desired_job(self, workload):
        metadata = workload["metadata"]
        spec = workload.get("spec", {})
        name = f"gw-{metadata['name']}"[:50].rstrip("-")
        labels = {"app": "gpu-workload", "gpuops/workload": metadata["name"]}
        command = spec.get("command") or ["python3", "training/train.py", "--steps", "100"]
        owner = client.V1OwnerReference(api_version=f"{GROUP}/{VERSION}", kind="GpuWorkload", name=metadata["name"], uid=metadata["uid"], controller=True, block_owner_deletion=True)
        return client.V1Job(
            metadata=client.V1ObjectMeta(name=name, namespace=NAMESPACE, labels=labels, owner_references=[owner]),
            spec=client.V1JobSpec(
                backoff_limit=2,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels=labels),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        runtime_class_name="nvidia",
                        containers=[client.V1Container(name="workload", image=spec["image"], image_pull_policy="IfNotPresent", command=command, resources=client.V1ResourceRequirements(limits={"nvidia.com/gpu": str(spec.get("gpuCount", 1))}))],
                    ),
                ),
            ),
        )

    def reconcile(self, workload):
        metadata = workload["metadata"]
        name = f"gw-{metadata['name']}"[:50].rstrip("-")
        try:
            job = self.batch.read_namespaced_job(name, NAMESPACE)
        except client.exceptions.ApiException as exc:
            if exc.status != 404:
                raise
            self.batch.create_namespaced_job(NAMESPACE, self.desired_job(workload))
            self.patch_status(metadata["name"], {"phase": "Submitted", "jobName": name, "attempt": 1, "observedGeneration": metadata.get("generation", 0)})
            log.info("workload_submitted name=%s job=%s", metadata["name"], name)
            return
        status = job.status
        if (status.succeeded or 0) > 0:
            phase, message = "Succeeded", "Kubernetes Job completed"
        elif (status.failed or 0) > 0:
            phase, message = "Failed", "Kubernetes Job failed"
        elif (status.active or 0) > 0:
            phase, message = "Running", "Kubernetes Job is active"
        else:
            phase, message = "Pending", "Waiting for GPU capacity"
        self.patch_status(metadata["name"], {"phase": phase, "jobName": name, "message": message, "observedGeneration": metadata.get("generation", 0)})

    def patch_status(self, name, status):
        self.custom.patch_namespaced_custom_object_status(GROUP, VERSION, NAMESPACE, PLURAL, name, {"status": status})

    def run(self):
        interval = int(os.getenv("CONTROLLER_INTERVAL", "10"))
        while True:
            if self.elector.is_leader():
                for workload in self.list_workloads():
                    try:
                        self.reconcile(workload)
                    except Exception:
                        log.exception("reconcile_failed name=%s", workload.get("metadata", {}).get("name"))
            time.sleep(interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    GpuWorkloadController().run()
