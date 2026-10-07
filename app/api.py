from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from .agent import AIOpsAgent
from .cluster import Cluster
from .models import Job
from .scheduler import Scheduler
from .validator import Validator
from .auth import USERS, issue_token, current_user, require_role
from .persistence import init_db, record_audit, record_job, record_incident, update_job_state, find_job_by_idempotency_key, record_checkpoint, SessionLocal, AuditEvent
from .llm_agent import AgenticPlanner
from .slurm import SlurmScheduler
from .kubernetes_adapter import KubernetesAdapter
from .inference import InferenceManager, InferenceService, default_endpoint
from pathlib import Path
import subprocess
import sys
import logging

BASE_DIR = Path(__file__).parent
logger = logging.getLogger("gpu-cluster-ops")

app = FastAPI(title="GPU Cluster Operations Platform", version="0.3.0")
cluster = Cluster(); scheduler = Scheduler(cluster); agent = AIOpsAgent(cluster)
inference = InferenceManager()
processes = {}

@app.on_event("startup")
def startup():
    init_db()

def node_json(node):
    return {
        "id": node.id, "hostname": node.hostname, "state": node.state.value,
        "driver_version": node.driver_version, "cuda_version": node.cuda_version,
        "last_heartbeat": node.last_heartbeat.isoformat(), "labels": node.labels,
        "gpus": [g.__dict__ for g in node.gpus], "free_gpus": node.free_gpus,
    }

def job_json(job):
    return {**job.__dict__, "state": job.state.value, "created_at": job.created_at.isoformat()}

def inference_json(service):
    return {**service.__dict__, "state": service.state.value, "created_at": service.created_at.isoformat(), "updated_at": service.updated_at.isoformat()}

def prom_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

class JobRequest(BaseModel):
    name: str; image: str = "python:3.12-slim"; gpu_count: int = Field(ge=1, le=16); priority: int = Field(default=0, ge=0, le=100); command: list[str] = []; idempotency_key: str | None = Field(default=None, max_length=200); max_retries: int = Field(default=3, ge=0, le=10)
class FailureRequest(BaseModel):
    node_id: str; kind: str; value: float | None = None
class TokenRequest(BaseModel): username: str; password: str
class InferenceServiceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    model: str = Field(min_length=1, max_length=300)
    runtime: str = "mock"
    endpoint: str | None = None
    replicas: int = Field(default=1, ge=1, le=100)
    gpu_count: int = Field(default=1, ge=1, le=16)
class ChatCompletionRequest(BaseModel):
    messages: list[dict]
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    stream: bool = False

@app.get("/api/v1/cluster/health")
def cluster_health(user=Depends(current_user)):
    cluster.refresh_telemetry()
    refresh_processes()
    results = Validator().validate(cluster)
    return {"mode": cluster.mode, "healthy": all(x["healthy"] for x in results), "nodes": results}

@app.get("/api/v1/nodes")
def nodes(user=Depends(current_user)):
    cluster.refresh_telemetry()
    refresh_processes(); return {"items": [node_json(n) for n in cluster.nodes.values()]}

@app.post("/api/v1/jobs")
def create_job(req: JobRequest, user=Depends(require_role("researcher", "operator", "admin"))):
    existing = find_job_by_idempotency_key(req.idempotency_key)
    if existing:
        return {"id": existing.job_id, "state": existing.state, "node_id": existing.node_id, "gpu_count": existing.gpu_count, "idempotent_replay": True}
    job = scheduler.submit(Job(req.name, req.image, req.gpu_count, req.priority, req.command, idempotency_key=req.idempotency_key, max_retries=req.max_retries))
    record_job(job, user["sub"])
    return job_json(job)

@app.get("/api/v1/jobs")
def jobs(user=Depends(current_user)):
    refresh_processes(); return {"items": [job_json(j) for j in scheduler.jobs.values()]}

@app.post("/api/v1/jobs/{job_id}/execute")
def execute_job(job_id: str, seconds: int = 10, user=Depends(require_role("operator", "admin"))):
    if job_id not in scheduler.jobs: raise HTTPException(404, "job not found")
    job = scheduler.jobs[job_id]
    if job.state.value != "running": raise HTTPException(409, "job must be running")
    if job_id in processes and processes[job_id].poll() is None: raise HTTPException(409, "job already executing")
    processes[job_id] = subprocess.Popen([sys.executable, "-m", "app.gpu_worker", "--seconds", str(max(1, min(seconds, 300)))], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    record_audit(user["sub"], "job.cuda_started", job_id, {"seconds": seconds})
    return {"job_id": job_id, "status": "launched", "node_id": job.node_id, "workload": "cuda_matrix_multiply"}

def refresh_processes():
    for job_id, process in list(processes.items()):
        if process.poll() is not None:
            completed = processes.pop(job_id, None)
            if completed is None:
                continue
            process = completed
            job = scheduler.jobs[job_id]
            job.state = __import__("app.models", fromlist=["JobState"]).JobState.SUCCEEDED if process.returncode == 0 else __import__("app.models", fromlist=["JobState"]).JobState.FAILED
            job.error = None if process.returncode == 0 else (process.stdout.read()[-500:] if process.stdout else "CUDA workload failed")
            logger.info("job.completed job_id=%s state=%s return_code=%s", job_id, job.state.value, process.returncode)
            record_audit("system", "job.completed", job_id, {"state": job.state.value, "return_code": process.returncode})
            scheduler.release(job_id)
            update_job_state(job)

@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user=Depends(require_role("operator", "admin"))):
    if job_id not in scheduler.jobs: raise HTTPException(404, "job not found")
    job = scheduler.cancel(job_id); update_job_state(job, user["sub"], "job.cancelled"); return job_json(job)

@app.post("/api/v1/jobs/{job_id}/checkpoint")
def checkpoint_job(job_id: str, step: int = 0, user=Depends(require_role("operator", "admin"))):
    if job_id not in scheduler.jobs: raise HTTPException(404, "job not found")
    if step < 0: raise HTTPException(400, "step must be non-negative")
    job = scheduler.jobs[job_id]
    job.state = __import__("app.models", fromlist=["JobState"]).JobState.CHECKPOINTING
    record_checkpoint(job, f"checkpoint://jobs/{job.id}/step-{step}", step, user["sub"])
    job.state = __import__("app.models", fromlist=["JobState"]).JobState.PREEMPTING
    update_job_state(job, user["sub"], "job.preempt_requested")
    job.state = __import__("app.models", fromlist=["JobState"]).JobState.REQUEUED
    job.retries += 1
    update_job_state(job, user["sub"], "job.requeued")
    job.state = __import__("app.models", fromlist=["JobState"]).JobState.QUEUED
    scheduler.reconcile()
    update_job_state(job, user["sub"], "job.requeued_for_recovery")
    return job_json(job)

@app.post("/api/v1/failures")
def inject_failure(req: FailureRequest, user=Depends(require_role("operator", "admin"))):
    node_id = req.node_id
    if node_id not in cluster.nodes and node_id == "gpu-local-01" and cluster.nodes:
        node_id = next(iter(cluster.nodes))
    if node_id not in cluster.nodes: raise HTTPException(404, "node not found")
    try: cluster.inject(node_id, req.kind, req.value)
    except ValueError as e: raise HTTPException(400, str(e))
    logger.warning("failure.injected node_id=%s kind=%s value=%s", node_id, req.kind, req.value)
    record_audit(user["sub"], "failure.injected", node_id, {"kind": req.kind, "value": req.value})
    return {"status": "injected", "node_id": node_id, "kind": req.kind}

@app.post("/api/v1/agent/reconcile")
def reconcile(user=Depends(require_role("operator", "admin"))):
    result = agent.reconcile()
    for incident in result["incidents"]: record_incident(incident, user["sub"])
    record_audit(user["sub"], "aiops.reconcile", "cluster", {"actions": len(result["actions"])})
    for action in result["actions"]:
        logger.info("aiops.remediation incident_id=%s node_id=%s action=%s status=%s", action["incident_id"], action["node_id"], action["action"], action["status"])
        record_audit(user["sub"], "aiops.remediation", action["node_id"], action)
    return result

@app.post("/api/v1/auth/token")
def token(req: TokenRequest):
    user = USERS.get(req.username)
    if not user or user["password"] != req.password: raise HTTPException(401, "invalid credentials")
    return {"access_token": issue_token(req.username, user["role"]), "token_type": "bearer", "role": user["role"]}

@app.get("/api/v1/agent/plan")
def agent_plan(user=Depends(current_user)):
    return AgenticPlanner(cluster).llm_explanation(AgenticPlanner(cluster).plan())

@app.post("/api/v1/slurm/submit")
def slurm_submit(req: JobRequest, user=Depends(require_role("operator", "admin"))):
    result = SlurmScheduler().submit(req.command or ["python", "-c", "print('slurm job')"], req.gpu_count)
    record_audit(user["sub"], "slurm.submit", req.name, result)
    return result

@app.post("/api/v1/kubernetes/jobs")
def kubernetes_submit(req: JobRequest, steps: int = 100, user=Depends(require_role("researcher", "operator", "admin"))):
    result = KubernetesAdapter().submit_gpu_job(req.name, req.image, steps, req.gpu_count)
    record_audit(user["sub"], "kubernetes.submit", req.name, result)
    return result

@app.get("/api/v1/kubernetes/jobs")
def kubernetes_jobs(user=Depends(current_user)): return KubernetesAdapter().list_training_jobs()

@app.post("/api/v1/inference/services")
def create_inference_service(req: InferenceServiceRequest, user=Depends(require_role("researcher", "operator", "admin"))):
    try:
        service = inference.register(InferenceService(req.name, req.model, req.runtime, req.endpoint or default_endpoint(), req.replicas, req.gpu_count))
    except ValueError as exc:
        raise HTTPException(409 if "already exists" in str(exc) else 400, str(exc))
    record_audit(user["sub"], "inference.service_created", service.name, {"model": service.model, "runtime": service.runtime, "gpu_count": service.gpu_count})
    return inference_json(service)

@app.get("/api/v1/inference/services")
def list_inference_services(user=Depends(current_user)):
    return {"items": [inference_json(service) for service in inference.services.values()]}

@app.get("/api/v1/inference/services/{name}")
def get_inference_service(name: str, user=Depends(current_user)):
    try:
        return inference_json(inference.get(name))
    except KeyError:
        raise HTTPException(404, "inference service not found")

@app.post("/api/v1/inference/services/{name}/stop")
def stop_inference_service(name: str, user=Depends(require_role("operator", "admin"))):
    try:
        service = inference.stop(name)
    except KeyError:
        raise HTTPException(404, "inference service not found")
    record_audit(user["sub"], "inference.service_stopped", name)
    return inference_json(service)

@app.post("/api/v1/inference/services/{name}/chat/completions")
def chat_completion(name: str, req: ChatCompletionRequest, user=Depends(current_user)):
    if req.stream:
        try:
            inference.get(name)
        except KeyError:
            raise HTTPException(404, "inference service not found")
        return StreamingResponse(
            inference.stream_chat(name, req.model_dump(exclude_none=True)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    try:
        response = inference.chat(name, req.model_dump(exclude_none=True))
    except KeyError:
        raise HTTPException(404, "inference service not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    except Exception as exc:
        logger.exception("inference_request_failed service=%s", name)
        raise HTTPException(502, f"inference runtime failed: {type(exc).__name__}")
    return response

@app.get("/api/v1/audit")
def audit(user=Depends(require_role("admin"))):
    with SessionLocal() as db:
        rows = db.query(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(100).all()
        return {"items": [{"id": x.id, "actor": x.actor, "action": x.action, "resource": x.resource, "details": x.details, "created_at": x.created_at.isoformat()} for x in rows]}

@app.get("/metrics")
def metrics():
    cluster.refresh_telemetry()
    refresh_processes()
    lines = ["# HELP cluster_nodes_total Number of cluster nodes", "# TYPE cluster_nodes_total gauge", f"cluster_nodes_total {len(cluster.nodes)}"]
    for n in cluster.nodes.values():
        for g in n.gpus:
            lines += [f'gpu_utilization_percent{{node="{n.id}",gpu="{g.id}"}} {g.utilization_pct}', f'gpu_memory_used_mb{{node="{n.id}",gpu="{g.id}"}} {g.memory_used_mb}', f'gpu_temperature_celsius{{node="{n.id}",gpu="{g.id}"}} {g.temperature_c}', f'gpu_power_watts{{node="{n.id}",gpu="{g.id}"}} {g.power_usage_w}']
    lines += [
        "# HELP inference_requests_total Total inference requests",
        "# TYPE inference_requests_total counter",
        f"inference_requests_total {inference.request_count}",
        "# HELP inference_request_errors_total Total failed inference requests",
        "# TYPE inference_request_errors_total counter",
        f"inference_request_errors_total {inference.error_count}",
        "# HELP inference_prompt_tokens_total Total prompt tokens processed",
        "# TYPE inference_prompt_tokens_total counter",
        f"inference_prompt_tokens_total {inference.total_prompt_tokens}",
        "# HELP inference_completion_tokens_total Total completion tokens generated",
        "# TYPE inference_completion_tokens_total counter",
        f"inference_completion_tokens_total {inference.total_completion_tokens}",
        "# HELP inference_services_total Registered inference services",
        "# TYPE inference_services_total gauge",
        f"inference_services_total {len(inference.services)}",
        "# HELP inference_request_duration_seconds_total Total inference request duration",
        "# TYPE inference_request_duration_seconds_total counter",
        f"inference_request_duration_seconds_total {inference.total_duration_seconds}",
        "# HELP inference_time_to_first_token_seconds_last Last observed time to first token",
        "# TYPE inference_time_to_first_token_seconds_last gauge",
        f"inference_time_to_first_token_seconds_last {max((s.last_ttft_seconds for s in inference.services.values()), default=0.0)}",
        "# HELP inference_tokens_per_second_last Last observed generated token rate",
        "# TYPE inference_tokens_per_second_last gauge",
        f"inference_tokens_per_second_last {max((s.last_tokens_per_second for s in inference.services.values()), default=0.0)}",
    ]
    for service in inference.services.values():
        labels = f'service="{prom_escape(service.name)}",runtime="{prom_escape(service.runtime)}"'
        model_labels = f'{labels},model="{prom_escape(service.model)}"'
        lines += [
            f"inference_service_requests_total{{{labels}}} {service.requests}",
            f"inference_service_errors_total{{{labels}}} {service.errors}",
            f"inference_service_prompt_tokens_total{{{labels}}} {service.prompt_tokens}",
            f"inference_service_completion_tokens_total{{{labels}}} {service.completion_tokens}",
            f"inference_service_last_duration_seconds{{{labels}}} {service.last_duration_seconds}",
            f"inference_service_last_ttft_seconds{{{labels}}} {service.last_ttft_seconds}",
            f"inference_service_last_tpot_seconds{{{labels}}} {service.last_tpot_seconds}",
            f"inference_service_last_tokens_per_second{{{labels}}} {service.last_tokens_per_second}",
            f'inference_service_info{{{model_labels},state="{prom_escape(service.state.value)}"}} 1',
        ]
    return "\n".join(lines) + "\n"

@app.get("/")
def dashboard(): return FileResponse(BASE_DIR / "static" / "index.html")
