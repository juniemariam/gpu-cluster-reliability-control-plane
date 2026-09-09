from .cluster import Cluster
from .models import NodeState, Severity

class Validator:
    def validate(self, cluster: Cluster):
        results = []
        for node in cluster.nodes.values():
            checks = []
            if node.state != NodeState.READY:
                checks.append(("node_state", Severity.CRITICAL, f"node is {node.state.value}", "cordon_and_recover_node"))
            if not node.gpus:
                checks.append(("gpu_inventory", Severity.CRITICAL, "no GPUs discovered", "inspect_driver_and_runtime"))
            for gpu in node.gpus:
                if not gpu.healthy:
                    checks.append((f"gpu_{gpu.id}", Severity.CRITICAL, "GPU marked unhealthy", "reset_gpu_and_retry_jobs"))
                if gpu.memory_used_mb / max(gpu.memory_total_mb, 1) > .9:
                    checks.append((f"memory_{gpu.id}", Severity.WARNING, f"memory usage {gpu.memory_used_mb}/{gpu.memory_total_mb} MB", "drain_node_and_clear_workloads"))
                if gpu.temperature_c >= 85:
                    checks.append((f"temperature_{gpu.id}", Severity.CRITICAL, f"temperature {gpu.temperature_c}C", "drain_node_and_check_cooling"))
            results.append({"node_id": node.id, "healthy": not checks, "checks": [{"name": n, "severity": s.value, "evidence": e, "action": a} for n,s,e,a in checks]})
        return results

