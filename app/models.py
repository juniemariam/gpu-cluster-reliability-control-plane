from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class NodeState(str, Enum):
    READY = "ready"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"

class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    PREEMPTING = "preempting"
    REQUEUED = "requeued"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Gpu:
    id: str
    name: str
    memory_total_mb: int
    memory_used_mb: int = 0
    utilization_pct: float = 0
    temperature_c: float = 35
    power_usage_w: float = 0
    healthy: bool = True

@dataclass
class Node:
    id: str
    hostname: str
    state: NodeState = NodeState.READY
    gpus: list[Gpu] = field(default_factory=list)
    driver_version: str = "simulated"
    cuda_version: str = "simulated"
    last_heartbeat: datetime = field(default_factory=utc_now)
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def free_gpus(self) -> int:
        return sum(1 for g in self.gpus if g.healthy and g.memory_used_mb < g.memory_total_mb * .9)

@dataclass
class Job:
    name: str
    image: str
    gpu_count: int
    priority: int = 0
    command: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: JobState = JobState.QUEUED
    node_id: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    idempotency_key: Optional[str] = None
    checkpoint_uri: Optional[str] = None
    checkpoint_step: int = 0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    error: Optional[str] = None

@dataclass
class Incident:
    node_id: str
    severity: Severity
    title: str
    evidence: list[str]
    recommended_action: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resolved: bool = False
    created_at: datetime = field(default_factory=utc_now)
