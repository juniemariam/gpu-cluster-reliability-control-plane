import unittest
from app.agent import AIOpsAgent
from app.cluster import Cluster
from app.models import Job, JobState
from app.scheduler import Scheduler
from app.validator import Validator

class PlatformTests(unittest.TestCase):
    def setUp(self): self.cluster = Cluster("simulated")
    def test_scheduler_assigns_priority_job(self):
        s = Scheduler(self.cluster); high = s.submit(Job("high", "demo", 1, priority=10)); low = s.submit(Job("low", "demo", 1, priority=1))
        self.assertEqual(high.state, JobState.RUNNING); self.assertEqual(low.state, JobState.RUNNING)
    def test_validator_detects_pressure(self):
        self.cluster.inject("gpu-sim-01", "gpu_memory_pressure", 95)
        results = Validator().validate(self.cluster)
        self.assertFalse(results[0]["healthy"]); self.assertTrue(any("memory" in x["name"] for x in results[0]["checks"]))
    def test_agent_recovers_unhealthy_node(self):
        self.cluster.inject("gpu-sim-01", "node_unhealthy")
        result = AIOpsAgent(self.cluster).reconcile()
        self.assertTrue(result["actions"]); self.assertEqual(self.cluster.nodes["gpu-sim-01"].state.value, "ready")

if __name__ == "__main__": unittest.main()
