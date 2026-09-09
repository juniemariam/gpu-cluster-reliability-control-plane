"""Minimal Kubernetes reliability controller: cordon unhealthy nodes and report failed GPU Jobs."""
import logging, os, time
from app.kubernetes_adapter import KubernetesAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s reliability-controller %(message)s")
log = logging.getLogger("reliability-controller")

def run(interval=15):
    adapter = KubernetesAdapter()
    if not adapter.available: raise RuntimeError("Kubernetes client unavailable")
    while True:
        for job in adapter.batch.list_namespaced_job(adapter.namespace, label_selector="app=gpu-training").items:
            if (job.status.failed or 0) > 0: log.warning("gpu_job_failed name=%s failed=%s remediation=inspect_logs_and_retry", job.metadata.name, job.status.failed)
        for node in adapter.core.list_node().items:
            conditions = {c.type:c.status for c in (node.status.conditions or [])}
            if conditions.get("Ready") == "False":
                log.warning("node_unhealthy name=%s remediation=cordon_node", node.metadata.name)
                try: adapter.core.patch_node(node.metadata.name, {"spec":{"unschedulable":True}})
                except Exception as exc: log.error("cordon_failed name=%s error=%s", node.metadata.name, type(exc).__name__)
        time.sleep(interval)

if __name__ == "__main__": run(int(os.getenv("CONTROLLER_INTERVAL", "15")))
