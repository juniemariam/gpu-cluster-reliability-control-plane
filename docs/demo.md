# Demo script

1. Start the service in simulated mode.
2. Open `/api/v1/cluster/health` and `/metrics`.
3. Submit two jobs with different priorities.
4. Inject `gpu_memory_pressure` into `gpu-sim-01`.
5. Show the health endpoint reporting evidence and severity.
6. Call `/api/v1/agent/reconcile` and show the allow-listed recovery action.
7. Re-check health and job placement.
8. Repeat with `CLUSTER_MODE=real` on an NVIDIA-enabled Docker host and capture GPU discovery output.

The interview story is: "I built an operations control plane that makes cluster state observable, schedules researcher workloads, detects failure with evidence, and performs bounded remediation with an audit trail."
