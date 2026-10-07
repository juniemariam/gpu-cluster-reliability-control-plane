"""Inference-service control-plane primitives.

The control plane registers services and delegates model execution to an
OpenAI-compatible runtime such as vLLM. A deterministic mock runtime keeps
local development and tests CPU-only.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
import time
from typing import Any

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InferenceServiceState(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


@dataclass
class InferenceService:
    name: str
    model: str
    runtime: str = "mock"
    endpoint: str | None = None
    replicas: int = 1
    gpu_count: int = 1
    state: InferenceServiceState = InferenceServiceState.READY
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_error: str | None = None
    last_duration_seconds: float = 0.0
    last_ttft_seconds: float = 0.0
    last_tpot_seconds: float = 0.0
    last_tokens_per_second: float = 0.0
    queue_depth: int = 0
    in_flight_requests: int = 0
    health_checks: int = 0
    health_failures: int = 0
    last_health_check_at: datetime | None = None
    last_health_error: str | None = None


class InferenceRuntime:
    def chat(self, service: InferenceService, payload: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        raise NotImplementedError

    def health(self, service: InferenceService) -> tuple[bool, str | None]:
        return True, None


class MockRuntime(InferenceRuntime):
    """Deterministic runtime for CPU development and API tests."""

    def health(self, service: InferenceService):
        return True, None

    def chat(self, service: InferenceService, payload: dict[str, Any]):
        messages = payload.get("messages") or []
        last = messages[-1].get("content", "") if messages else ""
        content = f"{service.model} mock response: {last}".strip()
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
        completion_tokens = len(content.split())
        response = {
            "id": f"chatcmpl-{service.name}-{service.requests + 1}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": service.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        }
        return response, prompt_tokens, completion_tokens

    def stream(self, service: InferenceService, payload: dict[str, Any]):
        messages = payload.get("messages") or []
        last = messages[-1].get("content", "") if messages else ""
        content = f"{service.model} mock response: {last}".strip()
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
        words = content.split()
        response_id = f"chatcmpl-{service.name}-{service.requests}"
        for index, word in enumerate(words):
            yield {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": service.model,
                "choices": [{"index": 0, "delta": {"content": word + (" " if index < len(words) - 1 else "")}, "finish_reason": None}],
                "_prompt_tokens": prompt_tokens,
            }
        yield {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": service.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(words), "total_tokens": prompt_tokens + len(words)},
        }


class OpenAICompatibleRuntime(InferenceRuntime):
    """Adapter for vLLM, SGLang, TGI-compatible gateways, or another server."""

    def health(self, service: InferenceService):
        if not service.endpoint:
            return False, "endpoint is required for an OpenAI-compatible runtime"
        try:
            import httpx
        except ImportError as exc:
            return False, type(exc).__name__
        endpoint = service.endpoint.rstrip("/")
        base = endpoint[:-3].rstrip("/") if endpoint.endswith("/v1") else endpoint
        try:
            response = httpx.get(base + "/health", timeout=5, trust_env=False)
            response.raise_for_status()
            return True, None
        except Exception as exc:
            return False, type(exc).__name__

    def chat(self, service: InferenceService, payload: dict[str, Any]):
        if not service.endpoint:
            raise ValueError("endpoint is required for an OpenAI-compatible runtime")
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for an OpenAI-compatible runtime") from exc
        body = {**payload, "model": payload.get("model", service.model), "stream": False}
        response = httpx.post(service.endpoint.rstrip("/") + "/chat/completions", json=body, timeout=120, trust_env=False)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return data, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    def stream(self, service: InferenceService, payload: dict[str, Any]):
        if not service.endpoint:
            raise ValueError("endpoint is required for an OpenAI-compatible runtime")
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for an OpenAI-compatible runtime") from exc
        body = {**payload, "model": payload.get("model", service.model), "stream": True}
        body.setdefault("stream_options", {"include_usage": True})
        with httpx.stream("POST", service.endpoint.rstrip("/") + "/chat/completions", json=body, timeout=120, trust_env=False) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    continue
                yield json.loads(data)


class InferenceManager:
    def __init__(self):
        self.services: dict[str, InferenceService] = {}
        self.runtimes: dict[str, InferenceRuntime] = {}
        self.request_count = 0
        self.error_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_duration_seconds = 0.0

    def register(self, service: InferenceService) -> InferenceService:
        if service.name in self.services:
            raise ValueError(f"inference service already exists: {service.name}")
        if service.replicas < 1 or service.gpu_count < 1:
            raise ValueError("replicas and gpu_count must be at least 1")
        runtime = service.runtime.lower()
        if runtime == "mock":
            self.runtimes[service.name] = MockRuntime()
        elif runtime in {"openai", "openai-compatible", "vllm", "sglang", "tgi"}:
            self.runtimes[service.name] = OpenAICompatibleRuntime()
        else:
            raise ValueError(f"unsupported inference runtime: {service.runtime}")
        service.runtime = runtime
        self.services[service.name] = service
        return service

    def get(self, name: str) -> InferenceService:
        if name not in self.services:
            raise KeyError(name)
        return self.services[name]

    def set_queue_depth(self, name: str, depth: int) -> InferenceService:
        service = self.get(name)
        if depth < 0:
            raise ValueError("queue depth must be non-negative")
        service.queue_depth = depth
        service.updated_at = utc_now()
        return service

    def health_check(self, name: str) -> InferenceService:
        service = self.get(name)
        if service.state == InferenceServiceState.STOPPED:
            return service
        service.health_checks += 1
        service.last_health_check_at = utc_now()
        healthy, error = self.runtimes[name].health(service)
        service.last_health_error = error
        if healthy:
            service.state = InferenceServiceState.READY
        else:
            service.health_failures += 1
            service.state = InferenceServiceState.UNAVAILABLE
        service.updated_at = utc_now()
        return service

    def chat(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        service = self.get(name)
        if service.state == InferenceServiceState.STOPPED:
            raise RuntimeError(f"inference service is stopped: {name}")
        started = time.perf_counter()
        self.request_count += 1
        service.requests += 1
        service.in_flight_requests += 1
        try:
            response, prompt_tokens, completion_tokens = self.runtimes[name].chat(service, payload)
            service.prompt_tokens += prompt_tokens
            service.completion_tokens += completion_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            return response
        except Exception as exc:
            self.error_count += 1
            service.errors += 1
            service.state = InferenceServiceState.DEGRADED
            service.last_error = type(exc).__name__
            raise
        finally:
            service.in_flight_requests = max(service.in_flight_requests - 1, 0)
            self.total_duration_seconds += time.perf_counter() - started
            service.updated_at = utc_now()

    def stream_chat(self, name: str, payload: dict[str, Any]):
        service = self.get(name)
        if service.state == InferenceServiceState.STOPPED:
            raise RuntimeError(f"inference service is stopped: {name}")
        started = time.perf_counter()
        first_token_at = None
        completion_tokens = 0
        prompt_tokens = 0
        self.request_count += 1
        service.requests += 1
        service.in_flight_requests += 1
        try:
            for chunk in self.runtimes[name].stream(service, payload):
                choices = chunk.get("choices") or []
                delta = choices[0].get("delta") or {} if choices else {}
                content = delta.get("content") or ""
                if content and first_token_at is None:
                    first_token_at = time.perf_counter()
                if content:
                    completion_tokens += 1
                usage = chunk.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", chunk.get("_prompt_tokens", prompt_tokens)))
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                public_chunk = {key: value for key, value in chunk.items() if not key.startswith("_")}
                yield f"data: {json.dumps(public_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            self.error_count += 1
            service.errors += 1
            service.state = InferenceServiceState.DEGRADED
            service.last_error = type(exc).__name__
            yield f"event: error\ndata: {json.dumps({'error': type(exc).__name__})}\n\n"
        finally:
            service.in_flight_requests = max(service.in_flight_requests - 1, 0)
            finished = time.perf_counter()
            duration = finished - started
            ttft = (first_token_at - started) if first_token_at is not None else duration
            decode_duration = max(duration - ttft, 0.0)
            tpot = decode_duration / completion_tokens if completion_tokens else 0.0
            tokens_per_second = completion_tokens / decode_duration if decode_duration > 0 and completion_tokens else 0.0
            service.prompt_tokens += prompt_tokens
            service.completion_tokens += completion_tokens
            service.last_duration_seconds = duration
            service.last_ttft_seconds = ttft
            service.last_tpot_seconds = tpot
            service.last_tokens_per_second = tokens_per_second
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_duration_seconds += duration
            service.updated_at = utc_now()

    def stop(self, name: str) -> InferenceService:
        service = self.get(name)
        service.state = InferenceServiceState.STOPPED
        service.updated_at = utc_now()
        return service


def default_endpoint() -> str | None:
    return os.getenv("INFERENCE_ENDPOINT")
