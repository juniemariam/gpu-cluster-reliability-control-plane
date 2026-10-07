import unittest
from app.agent import AIOpsAgent
from app.cluster import Cluster
from app.models import Job, JobState
from app.scheduler import Scheduler
from app.validator import Validator
from app.inference import InferenceManager, InferenceService
from app.reliability_experiment import run_reliability_experiment

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

    def test_mock_inference_service_returns_openai_shape(self):
        manager = InferenceManager()
        manager.register(InferenceService("rick", "llama-3.1-8b", runtime="mock"))
        response = manager.chat("rick", {"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["model"], "llama-3.1-8b")
        self.assertGreater(response["usage"]["completion_tokens"], 0)
        self.assertEqual(manager.request_count, 1)

    def test_duplicate_inference_service_is_rejected(self):
        manager = InferenceManager()
        manager.register(InferenceService("demo", "model"))
        with self.assertRaises(ValueError):
            manager.register(InferenceService("demo", "model"))

    def test_mock_inference_health_check_and_queue_depth(self):
        manager = InferenceManager()
        manager.register(InferenceService("health", "demo-model"))

        manager.set_queue_depth("health", 3)
        service = manager.health_check("health")

        self.assertEqual(service.state.value, "ready")
        self.assertEqual(service.queue_depth, 3)
        self.assertEqual(service.health_checks, 1)
        self.assertEqual(service.health_failures, 0)

    def test_failed_inference_health_check_marks_service_unavailable(self):
        manager = InferenceManager()
        manager.register(InferenceService("down", "demo-model"))

        class BrokenRuntime:
            def health(self, service):
                return False, "ConnectError"

        manager.runtimes["down"] = BrokenRuntime()
        service = manager.health_check("down")

        self.assertEqual(service.state.value, "unavailable")
        self.assertEqual(service.health_failures, 1)
        self.assertEqual(service.last_health_error, "ConnectError")

    def test_openai_runtime_requires_endpoint_at_request_time(self):
        manager = InferenceManager()
        manager.register(InferenceService("vllm", "model", runtime="vllm"))
        with self.assertRaises(ValueError):
            manager.chat("vllm", {"messages": [{"role": "user", "content": "hello"}]})

    def test_mock_stream_records_latency_and_throughput(self):
        manager = InferenceManager()
        manager.register(InferenceService("stream", "demo-model"))
        chunks = list(manager.stream_chat("stream", {"messages": [{"role": "user", "content": "hello"}]}))
        self.assertTrue(any("chat.completion.chunk" in chunk for chunk in chunks))
        self.assertTrue(chunks[-1].startswith("data: [DONE]"))
        service = manager.get("stream")
        self.assertEqual(service.requests, 1)
        self.assertGreater(service.completion_tokens, 0)
        self.assertGreaterEqual(service.last_ttft_seconds, 0)
        self.assertGreaterEqual(service.last_tokens_per_second, 0)

    def test_reliability_experiment_measures_recovery_and_no_false_positives(self):
        result = run_reliability_experiment(trials=3)
        self.assertEqual(result.trials, 3)
        self.assertGreaterEqual(result.failure_detection_seconds, 0)
        self.assertGreaterEqual(result.drain_initiation_seconds, 0)
        self.assertGreaterEqual(result.recovery_seconds, 0)
        self.assertEqual(result.workloads_affected, 1)
        self.assertEqual(result.false_positive_rate, 0)

if __name__ == "__main__": unittest.main()
