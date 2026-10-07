"""Repeatable CPU-only reliability experiment for the portfolio demo.

This measures the control-plane workflow, not physical GPU recovery time. The
simulated cluster makes the experiment deterministic and runnable without a
second GPU or cloud resources.
"""
from dataclasses import asdict, dataclass
import statistics
import time

from .agent import AIOpsAgent
from .cluster import Cluster
from .models import Job
from .scheduler import Scheduler


@dataclass
class ReliabilityExperimentResult:
    trials: int
    healthy_control_trials: int
    failure_detection_seconds: float
    drain_initiation_seconds: float
    recovery_seconds: float
    workloads_affected: int
    false_positive_rate: float


def run_reliability_experiment(trials: int = 5) -> ReliabilityExperimentResult:
    if trials < 1:
        raise ValueError("trials must be at least 1")

    detection = []
    drain = []
    recovery = []
    affected = []
    false_positives = 0

    for index in range(trials):
        cluster = Cluster("simulated")
        scheduler = Scheduler(cluster)
        agent = AIOpsAgent(cluster, scheduler)
        target = "gpu-sim-01"
        scheduler.submit(Job(f"reliability-trial-{index}", "demo", 1, priority=10))

        healthy_started = time.perf_counter()
        healthy_plan = agent.plan()
        if healthy_plan:
            false_positives += 1
        _ = time.perf_counter() - healthy_started

        injected_at = time.perf_counter()
        cluster.inject(target, "gpu_memory_pressure", 95)
        plans = agent.plan()
        detected_at = time.perf_counter()
        detection.append(detected_at - injected_at)

        result = agent.reconcile(plans)
        drain.append(agent.last_drain_initiation_seconds)
        recovery.append(time.perf_counter() - injected_at)
        affected.append(sum(action.get("workloads_affected", 0) for action in result["actions"]))

    return ReliabilityExperimentResult(
        trials=trials,
        healthy_control_trials=trials,
        failure_detection_seconds=statistics.mean(detection),
        drain_initiation_seconds=statistics.mean(drain),
        recovery_seconds=statistics.mean(recovery),
        workloads_affected=max(affected),
        false_positive_rate=false_positives / trials,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(asdict(run_reliability_experiment()), indent=2))
