from .cluster import Cluster
from .models import Incident, Severity, NodeState
from .validator import Validator

class AIOpsAgent:
    """Rule-backed agent scaffold: evidence first, allow-listed actions only."""
    def __init__(self, cluster: Cluster): self.cluster = cluster; self.validator = Validator(); self.incidents: list[Incident] = []
    def plan(self):
        incidents = []
        for result in self.validator.validate(self.cluster):
            for check in result["checks"]:
                incidents.append(Incident(result["node_id"], Severity(check["severity"]), check["name"], [check["evidence"]], check["action"]))
        self.incidents.extend(incidents); return incidents
    def reconcile(self):
        plans = self.plan(); actions = []
        for incident in plans:
            if incident.recommended_action in {"cordon_and_recover_node", "drain_node_and_clear_workloads", "drain_node_and_check_cooling"}:
                node = self.cluster.nodes[incident.node_id]; node.state = NodeState.DRAINING
                if incident.recommended_action != "drain_node_and_check_cooling": self.cluster.recover(incident.node_id)
                incident.resolved = True; actions.append({"incident_id": incident.id, "node_id": incident.node_id, "action": incident.recommended_action, "status": "executed"})
            else: actions.append({"incident_id": incident.id, "node_id": incident.node_id, "action": incident.recommended_action, "status": "planned"})
        return {"incidents": plans, "actions": actions}

