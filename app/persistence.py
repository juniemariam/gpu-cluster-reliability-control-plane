"""PostgreSQL-ready persistence and audit trail with SQLite local fallback."""
import json
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Integer, DateTime, Text, Boolean, Index, UniqueConstraint, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./gpu_cluster_ops.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase): pass

class JobRecord(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    state: Mapped[str] = mapped_column(String(30), index=True)
    node_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    gpu_count: Mapped[int] = mapped_column(Integer)
    owner: Mapped[str] = mapped_column(String(200), default="demo")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class WorkloadStateRecord(Base):
    """Durable control-plane state used for idempotency and recovery."""
    __tablename__ = "workload_state"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_workload_idempotency_key"),)
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    state: Mapped[str] = mapped_column(String(30), index=True)
    node_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    gpu_count: Mapped[int] = mapped_column(Integer)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    checkpoint_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    checkpoint_step: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class IncidentRecord(Base):
    __tablename__ = "incidents"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(100), index=True)
    severity: Mapped[str] = mapped_column(String(30), index=True)
    title: Mapped[str] = mapped_column(String(200))
    evidence: Mapped[str] = mapped_column(Text)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(200), index=True)
    action: Mapped[str] = mapped_column(String(100), index=True)
    resource: Mapped[str] = mapped_column(String(200))
    details: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

Index("ix_audit_action_created", AuditEvent.action, AuditEvent.created_at)

def init_db():
    Base.metadata.create_all(engine)

def record_job(job, actor="demo"):
    with SessionLocal.begin() as db:
        db.merge(JobRecord(id=job.id, name=job.name, state=job.state.value, node_id=job.node_id, gpu_count=job.gpu_count, owner=actor, created_at=job.created_at))
        db.merge(WorkloadStateRecord(job_id=job.id, idempotency_key=job.idempotency_key, state=job.state.value, node_id=job.node_id, gpu_count=job.gpu_count, priority=job.priority, attempt=job.retries, max_retries=job.max_retries, checkpoint_uri=job.checkpoint_uri, checkpoint_step=job.checkpoint_step, error=job.error, updated_at=job.updated_at))
        db.add(AuditEvent(actor=actor, action="job.submitted", resource=job.id, details=json.dumps({"name": job.name, "gpu_count": job.gpu_count})))

def find_job_by_idempotency_key(key):
    if not key:
        return None
    with SessionLocal() as db:
        return db.scalar(select(WorkloadStateRecord).where(WorkloadStateRecord.idempotency_key == key))

def update_job_state(job, actor="system", action="job.state_changed"):
    with SessionLocal.begin() as db:
        db.merge(JobRecord(id=job.id, name=job.name, state=job.state.value, node_id=job.node_id, gpu_count=job.gpu_count, owner=actor, created_at=job.created_at))
        db.merge(WorkloadStateRecord(job_id=job.id, idempotency_key=job.idempotency_key, state=job.state.value, node_id=job.node_id, gpu_count=job.gpu_count, priority=job.priority, attempt=job.retries, max_retries=job.max_retries, checkpoint_uri=job.checkpoint_uri, checkpoint_step=job.checkpoint_step, error=job.error, updated_at=job.updated_at))
        db.add(AuditEvent(actor=actor, action=action, resource=job.id, details=json.dumps({"state": job.state.value, "attempt": job.retries, "checkpoint_step": job.checkpoint_step})))

def record_checkpoint(job, uri, step, actor="system"):
    job.checkpoint_uri = uri
    job.checkpoint_step = step
    update_job_state(job, actor=actor, action="job.checkpoint_created")

def record_incident(incident, actor="aiops-agent"):
    with SessionLocal.begin() as db:
        db.merge(IncidentRecord(id=incident.id, node_id=incident.node_id, severity=incident.severity.value, title=incident.title, evidence=json.dumps(incident.evidence), resolved=incident.resolved, created_at=incident.created_at))
        db.add(AuditEvent(actor=actor, action="incident.detected", resource=incident.id, details=json.dumps({"node_id": incident.node_id, "severity": incident.severity.value})))

def record_audit(actor, action, resource, details=None):
    with SessionLocal.begin() as db:
        db.add(AuditEvent(actor=actor, action=action, resource=resource, details=json.dumps(details or {})))
