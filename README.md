# GPU Cluster Reliability Control Plane

An NVIDIA GPU cluster operations platform for safely scheduling research workloads, observing GPU health, detecting failures, and executing automated remediation.

The project models the control-plane responsibilities that surround machine-learning infrastructure: GPU discovery, workload admission, CUDA execution, telemetry refresh, incident generation, audit history, and AIOps reconciliation.

## Why this project exists

Research teams need reliable access to GPU clusters without manually inspecting every node or debugging every failed workload. This project provides a small but realistic control plane that:

- discovers NVIDIA GPUs through `nvidia-smi` in real mode;
- supports a simulated multi-node mode for development;
- schedules GPU jobs based on available capacity and priority;
- runs a CUDA matrix-multiplication smoke workload;
- records jobs, incidents, and audit events in PostgreSQL;
- injects GPU-memory pressure to exercise failure handling;
- detects incidents and executes drain/recovery actions through an AIOps agent;
- exposes a FastAPI API and operator dashboard;
- runs locally with Docker Compose or on Kubernetes/k3s.

## Architecture at a glance

```text
                         +----------------------+
                         |  Operator Dashboard  |
                         |  FastAPI static UI   |
                         +----------+-----------+
                                    |
                                    v
+----------------+       +----------+-----------+       +----------------+
| Researcher /  |------>| FastAPI Control Plane |<----->|  PostgreSQL    |
| Operator API   |       | jobs, health, AIOps   |       | jobs/audit     |
+----------------+       +----+-------------+---+       +----------------+
                              |             |
                              v             v
                    +---------+--+   +------+----------------+
                    | GPU Adapter |   | Reliability Controller |
                    | nvidia-smi  |   | reconcile/remediation  |
                    +------+------+
                           |
                           v
                    +------+------+
                    | NVIDIA GPU  |
                    | RTX 5070 Ti |
                    +-------------+
```

## Runtime modes

### Real GPU mode

Used by the native k3s deployment. The application executes `nvidia-smi` to discover the host GPU and refresh memory, utilization, temperature, power, driver, and CUDA information.

The current local environment uses:

- NVIDIA GeForce RTX 5070 Ti;
- native k3s inside Ubuntu 24.04 on WSL2;
- NVIDIA Container Toolkit;
- NVIDIA Kubernetes device plugin;
- `runtimeClassName: nvidia`;
- Kubernetes extended resource `nvidia.com/gpu`.

### Simulated mode

Used by the CPU Docker Compose profile and Docker Desktop Kubernetes demonstrations. It creates deterministic simulated nodes and GPUs so the API, scheduler, incident workflow, and dashboard can be developed without a GPU cluster.

## Main capabilities

### GPU inventory and telemetry

The cluster adapter reports node state and per-GPU telemetry. In real mode, values come from `nvidia-smi`; in simulated mode, values come from the in-memory test cluster.

### GPU-aware scheduling

Jobs declare a GPU count and priority. The scheduler selects a ready node with sufficient free GPUs, marks the workload as running, and releases capacity when the workload completes or is cancelled.

### CUDA workload execution

The dashboard can submit a CUDA job and launch the packaged GPU worker. The worker performs a matrix-multiplication smoke test, making GPU execution visible in the application lifecycle and with `nvidia-smi`.

### Reliability and AIOps

Operators can inject GPU-memory pressure. The validator turns abnormal telemetry into incidents with evidence and a recommended action. The AIOps agent reconciles incidents and records remediation actions such as `drain_node_and_clear_workloads`.

### Persistence and auditability

PostgreSQL stores job, incident, and audit records. This gives the demo an operational history instead of relying only on transient console output.

## Repository layout

```text
app/
  api.py                    FastAPI routes and lifecycle
  cluster.py                Real/simulated GPU discovery and telemetry
  scheduler.py              GPU-aware job admission and lifecycle
  agent.py                  Reliability reconciliation logic
  validator.py              Health and incident validation
  persistence.py            PostgreSQL persistence and audit events
  gpu_worker.py             CUDA workload worker
  kubernetes_adapter.py     Kubernetes integration boundary
  static/index.html         Operator dashboard

controllers/
  reliability_controller.py Kubernetes reliability controller
  workload_controller.py    GpuWorkload reconciler and Lease leadership

training/
  train.py                  GPU training smoke workload

deploy/
  docker-compose.yml        CPU and GPU Compose services
  Dockerfile.gpu            NVIDIA CUDA runtime image
  gpuworkload-crd.yaml      GpuWorkload custom resource definition
  k8s.yaml                  API Deployment and Service
  postgres.yaml              PostgreSQL StatefulSet and Service
  controller.yaml            RBAC and reliability controller
  training-job.yaml          Kubernetes GPU training Job
  observability.yaml         Observability resources
```

## Run the real GPU Kubernetes demo

These commands target the native k3s cluster inside Ubuntu WSL2. Keep the existing project directory on Windows; WSL accesses it through `/mnt/c`.

```bash
cd "/mnt/c/Users/George/Downloads/GPU Cluster Reliability Control Plane"

# The image must already be imported into k3s containerd.
sudo k3s kubectl apply -f deploy/postgres.yaml
sudo k3s kubectl apply -f deploy/gpuworkload-crd.yaml
sudo k3s kubectl apply -f deploy/k8s.yaml
sudo k3s kubectl apply -f deploy/controller.yaml

sudo k3s kubectl get pods -n gpuops
sudo k3s kubectl exec -n gpuops deployment/gpu-cluster-ops -- nvidia-smi

sudo k3s kubectl port-forward \
  --address 0.0.0.0 \
  -n gpuops service/gpu-cluster-ops 8003:8000
```

Open `http://127.0.0.1:8003` in a browser.

## Run the Docker Compose GPU demo

```powershell
cd "C:\Users\George\Downloads\GPU Cluster Reliability Control Plane"
docker compose -f deploy/docker-compose.yml down
docker compose -f deploy/docker-compose.yml --profile gpu up --build
```

The Compose GPU service is exposed at `http://127.0.0.1:8002`.

## Demo script

1. Open the dashboard and show the RTX 5070 Ti inventory.
2. Submit a CUDA job and run it.
3. Show the job transition to `succeeded`.
4. Click **Inject GPU pressure**.
5. Click **Run AIOps reconcile**.
6. Show the incident evidence and remediation action.
7. Explain that Kubernetes handles placement and GPU allocation while the application provides the control-plane and reliability workflow.

## API surface

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/cluster/health` | Aggregate cluster health |
| `GET /api/v1/nodes` | Node and GPU inventory |
| `GET /api/v1/jobs` | List jobs and states |
| `POST /api/v1/jobs` | Submit a GPU job |
| `POST /api/v1/jobs/{id}/execute` | Launch CUDA execution |
| `POST /api/v1/jobs/{id}/cancel` | Cancel a running job |
| `POST /api/v1/jobs/{id}/checkpoint` | Record checkpoint metadata and requeue state |
| `POST /api/v1/failures` | Inject a test failure |
| `POST /api/v1/agent/reconcile` | Detect and remediate incidents |
| `GET /metrics` | Prometheus-compatible metrics |

Interactive API documentation is available at `/docs`.

## Limitations and production next steps

This is a portfolio-scale control plane, not a replacement for a production scheduler. Durable workload state, idempotent replay protection, a Kubernetes workload CRD, and Lease-based controller leadership are implemented. Remaining production-hardening work includes a real queue, actual model checkpoint save/restore, highly available PostgreSQL, stronger authentication and authorization, structured logs, Prometheus/Grafana dashboards, alert routing, network policies, image signing, and multi-node testing.

## Resume-ready description

**GPU Cluster Reliability Control Plane** — Built a FastAPI and Kubernetes-based GPU operations platform that discovers NVIDIA hardware, schedules prioritized CUDA workloads, persists workload and audit state in PostgreSQL, protects submissions with idempotency keys, injects GPU-memory failures, and reconciles incidents through reliability controllers with Kubernetes Lease leadership. Deployed real GPU workloads on native k3s with NVIDIA Container Toolkit and the Kubernetes device plugin.

## Name recommendation

Recommended product name: **GPU Cluster Reliability Control Plane**

Recommended repository slug: `gpu-cluster-reliability-control-plane`

For now, keeping the existing local folder name avoids breaking your current commands. Rename the GitHub repository and folder only after the demo is stable.
