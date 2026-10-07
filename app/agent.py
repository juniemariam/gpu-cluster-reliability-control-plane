from .cluster import Cluster
from .models import Incident, Severity, NodeState
from .validator import Validator
import time

class AIOpsAgent:
    """Rule-backed agent scaffold: evidence first, allow-listed actions only."""
    def __init__(self, cluster: Cluster, scheduler=None):
        self.cluster = cluster
        self.scheduler = scheduler
        self.validator = Validator()
        self.incidents: list[Incident] = []
        self.last_drain_initiation_seconds = 0.0
    def plan(self):
        incidents = []
        for result in self.validator.validate(self.cluster):
            for check in result["checks"]:
                incidents.append(Incident(result["node_id"], Severity(check["severity"]), check["name"], [check["evidence"]], check["action"]))
        self.incidents.extend(incidents); return incidents
    def reconcile(self, plans=None):
        plans = self.plan() if plans is None else plans
        actions = []
        started = time.perf_counter()
        for incident in plans:
            if incident.recommended_action in {"cordon_and_recover_node", "drain_node_and_clear_workloads", "drain_node_and_check_cooling"}:
                node = self.cluster.nodes[incident.node_id]; node.state = NodeState.DRAINING
                self.last_drain_initiation_seconds = time.perf_counter() - started
                affected = []
                if self.scheduler and incident.recommended_action != "drain_node_and_check_cooling":
                    for job in self.scheduler.jobs.values():
                        if job.node_id == incident.node_id and job.state.value == "running":
                            job.state = type(job.state).REQUEUED
                            job.retries += 1
                            job.node_id = None
                            affected.append(job.id)
                if incident.recommended_action != "drain_node_and_check_cooling":
                    self.cluster.recover(incident.node_id)
                    if self.scheduler:
                        self.scheduler.reconcile()
                incident.resolved = True
                actions.append({"incident_id": incident.id, "node_id": incident.node_id, "action": incident.recommended_action, "status": "executed", "workloads_affected": len(affected), "workload_ids": affected})
            else: actions.append({"incident_id": incident.id, "node_id": incident.node_id, "action": incident.recommended_action, "status": "planned"})
        return {"incidents": plans, "actions": actions}
