import unittest
from app.agent import AIOpsAgent
from app.cluster import Cluster
from app.models import Job, JobState
from app.scheduler import Scheduler
from app.validator import Validator
from app.inference import InferenceManager, InferenceService
from app.reliability_experiment import run_reliability_experiment
from app.routing import InferenceRouter
from app.routing_benchmark import benchmark_policy

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

    def test_round_robin_router_rotates_logical_replicas(self):
        router = InferenceRouter()
        router.register("demo", 3)

        selected = [router.choose("demo", "round-robin")["replica"] for _ in range(4)]

        self.assertEqual(selected, ["demo-replica-0", "demo-replica-1", "demo-replica-2", "demo-replica-0"])

    def test_load_aware_router_selects_lowest_queue(self):
        router = InferenceRouter()
        router.register("demo", 2)
        router.update_replica("demo", "demo-replica-0", queue_depth=5)
        router.update_replica("demo", "demo-replica-1", queue_depth=1)

        decision = router.choose("demo", "load-aware")

        self.assertEqual(decision["replica"], "demo-replica-1")

    def test_kv_cache_aware_router_prefers_cached_replica(self):
        router = InferenceRouter()
        router.register("demo", 2)
        router.update_replica("demo", "demo-replica-0", queue_depth=1, kv_cache_utilization=0.2)
        router.update_replica("demo", "demo-replica-1", queue_depth=1, kv_cache_utilization=0.8)

        decision = router.choose("demo", "kv-cache-aware")

        self.assertEqual(decision["replica"], "demo-replica-1")
        self.assertEqual(decision["execution_mode"], "logical-replica-simulation")

    def test_routing_decision_can_precede_model_request(self):
        router = InferenceRouter()
        manager = InferenceManager()
        manager.register(InferenceService("chat", "demo-model", replicas=2))
        router.register("chat", 2)
        router.update_replica("chat", "chat-replica-0", queue_depth=4)
        router.update_replica("chat", "chat-replica-1", queue_depth=1)

        decision = router.choose("chat", "load-aware")
        response = manager.chat(
            "chat",
            {"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(decision["replica"], "chat-replica-1")
        self.assertEqual(response["object"], "chat.completion")

    def test_vllm_metrics_parser_reads_queue_and_kv_cache(self):
        from app.inference import OpenAICompatibleRuntime

        text = """
        vllm:num_requests_waiting 4
        vllm:kv_cache_usage_perc 72.5
        """

        metrics = OpenAICompatibleRuntime.parse_metrics_text(text)

        self.assertEqual(metrics["queue_depth"], 4)
        self.assertAlmostEqual(metrics["kv_cache_utilization"], 0.725)

    def test_router_drains_non_ready_service_replicas(self):
        router = InferenceRouter()
        router.register("demo", 2)
        router.set_service_state("demo", "degraded")

        with self.assertRaises(RuntimeError):
            router.choose("demo", "load-aware")

    def test_concurrent_routing_benchmark_reports_distribution(self):
        result = benchmark_policy("round-robin", requests=30, workers=4, replicas=3)

        self.assertEqual(result.requests, 30)
        self.assertEqual(sum(result.distribution.values()), 30)
        self.assertGreater(result.decisions_per_second, 0)

    def test_admission_limit_rejects_second_in_flight_request(self):
        manager = InferenceManager()
        manager.register(InferenceService("limited", "demo-model", max_concurrent_requests=1))
        service = manager.get("limited")
        service.in_flight_requests = 1

        with self.assertRaisesRegex(RuntimeError, "admission limit"):
            manager.chat("limited", {"messages": [{"role": "user", "content": "hello"}]})

        self.assertEqual(service.admission_rejections, 1)

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
