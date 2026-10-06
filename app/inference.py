"""Inference-service control-plane primitives.

The control plane registers services and delegates model execution to an
OpenAI-compatible runtime such as vLLM. A deterministic mock runtime keeps
local development and tests CPU-only.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import os
import time
from typing import Any

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InferenceServiceState(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
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


class InferenceRuntime:
    def chat(self, service: InferenceService, payload: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        raise NotImplementedError


class MockRuntime(InferenceRuntime):
    """Deterministic runtime for CPU development and API tests."""

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


class OpenAICompatibleRuntime(InferenceRuntime):
    """Adapter for vLLM, SGLang, TGI-compatible gateways, or another server."""

    def chat(self, service: InferenceService, payload: dict[str, Any]):
        if not service.endpoint:
            raise ValueError("endpoint is required for an OpenAI-compatible runtime")
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for an OpenAI-compatible runtime") from exc
        body = {**payload, "model": payload.get("model", service.model), "stream": False}
        response = httpx.post(service.endpoint.rstrip("/") + "/chat/completions", json=body, timeout=120)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return data, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


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

    def chat(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        service = self.get(name)
        if service.state == InferenceServiceState.STOPPED:
            raise RuntimeError(f"inference service is stopped: {name}")
        started = time.perf_counter()
        self.request_count += 1
        service.requests += 1
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
            self.total_duration_seconds += time.perf_counter() - started
            service.updated_at = utc_now()

    def stop(self, name: str) -> InferenceService:
        service = self.get(name)
        service.state = InferenceServiceState.STOPPED
        service.updated_at = utc_now()
        return service


def default_endpoint() -> str | None:
    return os.getenv("INFERENCE_ENDPOINT")
