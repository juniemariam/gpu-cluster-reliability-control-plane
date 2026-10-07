"""Concurrent benchmark for logical inference routing policies."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import statistics
import time

from .routing import InferenceRouter, RoutingPolicy


@dataclass
class RoutingBenchmarkResult:
    policy: str
    requests: int
    workers: int
    duration_seconds: float
    decisions_per_second: float
    distribution: dict[str, int]


def benchmark_policy(policy: str, requests: int = 100, workers: int = 8, replicas: int = 3) -> RoutingBenchmarkResult:
    if requests < 1 or workers < 1 or replicas < 1:
        raise ValueError("requests, workers, and replicas must be at least 1")
    router = InferenceRouter()
    router.register("benchmark", replicas)
    for index in range(replicas):
        router.update_replica(
            "benchmark",
            f"benchmark-replica-{index}",
            queue_depth=[7, 2, 5][index % 3],
            kv_cache_utilization=[0.2, 0.8, 0.4][index % 3],
        )
    started = time.perf_counter()

    def choose(_):
        return router.choose("benchmark", RoutingPolicy(policy))["replica"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        selected = list(pool.map(choose, range(requests)))
    duration = time.perf_counter() - started
    distribution = {name: selected.count(name) for name in sorted(set(selected))}
    return RoutingBenchmarkResult(policy, requests, workers, duration, requests / duration if duration else 0.0, distribution)


def run_routing_benchmark(requests: int = 100, workers: int = 8):
    return [asdict(benchmark_policy(policy, requests, workers)) for policy in RoutingPolicy]


if __name__ == "__main__":
    import json

    print(json.dumps(run_routing_benchmark(), indent=2))
