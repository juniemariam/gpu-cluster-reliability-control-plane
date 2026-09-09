import os
import shutil
import subprocess
from .models import Gpu, Node, NodeState

class Cluster:
    def __init__(self, mode: str | None = None):
        self.mode = mode or os.getenv("CLUSTER_MODE", "simulated")
        self.nodes: dict[str, Node] = {}
        self._injected: dict[str, str] = {}
        self._seed()

    def _seed(self):
        if self.mode == "real" and shutil.which("nvidia-smi"):
            try:
                query = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,driver_version", "--format=csv,noheader,nounits"]
                rows = subprocess.check_output(query, text=True, timeout=5).strip().splitlines()
                gpus = []
                driver = "unknown"
                for row in rows:
                    idx, name, total, used, util, temp, power, driver = [x.strip() for x in row.split(",")]
                    gpus.append(Gpu(idx, name, int(float(total)), int(float(used)), float(util), float(temp), float(power)))
                if gpus:
                    self.nodes["gpu-local-01"] = Node("gpu-local-01", "localhost", gpus=gpus, driver_version=driver, cuda_version="host")
                    return
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
        for i in range(1, 3):
            self.nodes[f"gpu-sim-{i:02d}"] = Node(
                f"gpu-sim-{i:02d}", f"sim-worker-{i}",
                gpus=[Gpu(f"{i}-0", "NVIDIA Simulated GPU", 16384), Gpu(f"{i}-1", "NVIDIA Simulated GPU", 16384)],
                labels={"accelerator": "nvidia", "pool": "research"})

    def refresh_telemetry(self):
        """Refresh real GPU readings without changing simulated state."""
        if self.mode != "real" or not self.nodes or not shutil.which("nvidia-smi"):
            return
        try:
            query = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
            rows = subprocess.check_output(query, text=True, timeout=3).strip().splitlines()
            readings = {}
            for row in rows:
                idx, used, util, temp, power = [x.strip() for x in row.split(",")]
                readings[idx] = (int(float(used)), float(util), float(temp), float(power))
            for node in self.nodes.values():
                for gpu in node.gpus:
                    if gpu.id in readings and gpu.id not in self._injected:
                        gpu.memory_used_mb, gpu.utilization_pct, gpu.temperature_c, gpu.power_usage_w = readings[gpu.id]
        except (OSError, subprocess.SubprocessError, ValueError):
            return

    def inject(self, node_id: str, kind: str, value: float | None = None):
        node = self.nodes[node_id]
        if kind == "node_unhealthy": node.state = NodeState.UNHEALTHY
        elif kind == "gpu_memory_pressure":
            for gpu in node.gpus:
                gpu.memory_used_mb = int(gpu.memory_total_mb * ((value or 95) / 100)); self._injected[gpu.id] = kind
        elif kind == "temperature":
            for gpu in node.gpus:
                gpu.temperature_c = value or 95; self._injected[gpu.id] = kind
        else: raise ValueError(f"unsupported failure kind: {kind}")

    def recover(self, node_id: str):
        node = self.nodes[node_id]; node.state = NodeState.READY
        for gpu in node.gpus:
            gpu.healthy = True; gpu.memory_used_mb = 0; gpu.temperature_c = 35; gpu.utilization_pct = 0; gpu.power_usage_w = 0
            self._injected.pop(gpu.id, None)
