"""Inference routing policies for real and simulated replicas.

The router makes placement decisions independently of model execution. On a
single GPU, replicas are logical control-plane objects; one real vLLM backend
can still be used to exercise and compare the policies.
"""
from dataclasses import dataclass
from enum import Enum


class RoutingPolicy(str, Enum):
    ROUND_ROBIN = "round-robin"
    LOAD_AWARE = "load-aware"
    KV_CACHE_AWARE = "kv-cache-aware"


@dataclass
class ReplicaSnapshot:
    name: str
    queue_depth: int = 0
    kv_cache_utilization: float = 0.0
    state: str = "ready"
    requests: int = 0


class InferenceRouter:
    def __init__(self):
        self.replicas: dict[str, list[ReplicaSnapshot]] = {}
        self.next_index: dict[str, int] = {}
        self.decisions: dict[tuple[str, str], int] = {}

    def register(self, service_name: str, replica_count: int = 1):
        if replica_count < 1:
            raise ValueError("replica_count must be at least 1")
        self.replicas[service_name] = [
            ReplicaSnapshot(f"{service_name}-replica-{index}")
            for index in range(replica_count)
        ]
        self.next_index[service_name] = 0

    def update_replica(
        self,
        service_name: str,
        replica_name: str,
        *,
        queue_depth: int = 0,
        kv_cache_utilization: float = 0.0,
        state: str = "ready",
    ) -> ReplicaSnapshot:
        replica = self._get_replica(service_name, replica_name)
        if queue_depth < 0:
            raise ValueError("queue depth must be non-negative")
        if not 0 <= kv_cache_utilization <= 1:
            raise ValueError("kv-cache utilization must be between 0 and 1")
        replica.queue_depth = queue_depth
        replica.kv_cache_utilization = kv_cache_utilization
        replica.state = state
        return replica

    def choose(self, service_name: str, policy: RoutingPolicy | str = RoutingPolicy.ROUND_ROBIN) -> dict:
        if service_name not in self.replicas:
            raise KeyError(service_name)
        policy = RoutingPolicy(policy)
        candidates = [r for r in self.replicas[service_name] if r.state == "ready"]
        if not candidates:
            raise RuntimeError(f"no ready inference replicas for {service_name}")

        if policy == RoutingPolicy.ROUND_ROBIN:
            index = self.next_index[service_name] % len(candidates)
            selected = candidates[index]
            self.next_index[service_name] += 1
            reason = "next ready logical replica"
        elif policy == RoutingPolicy.LOAD_AWARE:
            selected = min(candidates, key=lambda r: (r.queue_depth, r.requests, r.name))
            reason = "lowest queue depth, then request count"
        else:
            selected = min(candidates, key=lambda r: (r.queue_depth, -r.kv_cache_utilization, r.name))
            reason = "lowest queue depth, then highest KV-cache utilization"

        selected.requests += 1
        key = (service_name, policy.value)
        self.decisions[key] = self.decisions.get(key, 0) + 1
        return {
            "service": service_name,
            "replica": selected.name,
            "policy": policy.value,
            "reason": reason,
            "queue_depth": selected.queue_depth,
            "kv_cache_utilization": selected.kv_cache_utilization,
            "execution_mode": "logical-replica-simulation",
        }

    def _get_replica(self, service_name: str, replica_name: str) -> ReplicaSnapshot:
        for replica in self.replicas.get(service_name, []):
            if replica.name == replica_name:
                return replica
        raise KeyError(replica_name)
