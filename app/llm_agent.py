"""Approval-gated Agentic AI adapter with deterministic fallback."""
import json, os
from .validator import Validator

ALLOW_LIST = {"cordon_and_recover_node", "drain_node_and_clear_workloads", "drain_node_and_check_cooling", "inspect_driver_and_runtime"}

class AgenticPlanner:
    def __init__(self, cluster): self.cluster = cluster
    def plan(self):
        evidence = Validator().validate(self.cluster)
        incidents = []
        for node in evidence:
            for check in node["checks"]:
                incidents.append({"node_id": node["node_id"], "diagnosis": check["evidence"], "action": check["action"], "requires_approval": check["action"] not in {"inspect_driver_and_runtime"}})
        return {"mode": os.getenv("AGENT_MODE", "rules"), "incidents": incidents, "allow_list": sorted(ALLOW_LIST)}

    def llm_explanation(self, plan):
        if os.getenv("AGENT_MODE", "rules") != "openai": return plan
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.responses.create(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), input="Return concise JSON explaining this GPU incident plan. Do not invent actions:\n" + json.dumps(plan))
            plan["explanation"] = response.output_text
        except Exception as exc:
            plan["explanation"] = "LLM unavailable; deterministic evidence-based plan retained."
            plan["llm_error"] = type(exc).__name__
        return plan
