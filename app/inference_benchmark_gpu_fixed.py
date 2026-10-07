"""Single-GPU inference benchmark for the running control-plane API.

Run this against a live FastAPI + vLLM deployment. It intentionally measures
the complete HTTP path through the control plane, not just direct vLLM calls.

Example:
    python -m app.inference_benchmark --service local-qwen --concurrency 1,4,8,16 \
        --prompt-tokens 128,2048 --requests 16 --max-tokens 64
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RequestSample:
    ok: bool
    status_code: int
    duration_seconds: float
    ttft_seconds: float | None
    completion_tokens: int
    prompt_tokens: int
    error: str | None = None

    @property
    def tpot_seconds(self) -> float | None:
        if self.ttft_seconds is None or self.completion_tokens <= 1:
            return None
        decode_seconds = self.duration_seconds - self.ttft_seconds
        return decode_seconds / (self.completion_tokens - 1) if decode_seconds >= 0 else None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def make_prompt(target_tokens: int) -> str:
    # This is a reproducible approximation. The response reports actual prompt
    # tokens from the runtime, which should be used for final analysis.
    words = ("GPU inference benchmark measures latency throughput queue and "
             "memory behavior under concurrent requests ")
    return (words * math.ceil(target_tokens / len(words.split())))[:target_tokens * 6].strip()


def parse_sse(response: httpx.Response, started: float) -> RequestSample:
    first_token_at: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    try:
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            choices = payload.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    first_token_at = first_token_at or time.perf_counter()
            usage = payload.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or 0)
            completion_tokens = int(usage.get("completion_tokens", completion_tokens) or 0)
        finished = time.perf_counter()
        if not completion_tokens:
            # Some gateways omit usage on streamed responses. This keeps the
            # sample valid while making the limitation visible in the output.
            completion_tokens = 0
        return RequestSample(True, response.status_code, finished - started,
                             (first_token_at - started) if first_token_at else None,
                             completion_tokens, prompt_tokens)
    except Exception as exc:
        return RequestSample(False, response.status_code, time.perf_counter() - started,
                             None, completion_tokens, prompt_tokens, type(exc).__name__)


def run_request(base_url: str, service: str, prompt: str, max_tokens: int,
                headers: dict[str, str]) -> RequestSample:
    started = time.perf_counter()
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }
    try:
        with httpx.Client(timeout=180, trust_env=False) as client:
            with client.stream(
                "POST",
                f"{base_url.rstrip('/')}/api/v1/inference/services/{service}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    return RequestSample(False, response.status_code,
                                         time.perf_counter() - started, None, 0, 0,
                                         body[:200])
                return parse_sse(response, started)
    except Exception as exc:
        return RequestSample(False, 0, time.perf_counter() - started, None, 0, 0,
                             type(exc).__name__)


def calibrate_prompt_tokens(base_url: str, service: str, prompt: str,
                            headers: dict[str, str]) -> int | None:
    """Get an authoritative prompt-token count outside streaming usage."""
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/api/v1/inference/services/{service}/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}],
                  "stream": False, "max_tokens": 1},
            headers=headers,
            timeout=120,
            trust_env=False,
        )
        response.raise_for_status()
        usage = response.json().get("usage") or {}
        value = usage.get("prompt_tokens")
        return int(value) if value is not None else None
    except Exception:
        return None


def read_prometheus_metrics(base_url: str, runtime_metrics_url: str | None = None) -> dict[str, float]:
    """Read the latest control-plane gauges and best-effort GPU signals."""
    try:
        text = httpx.get(f"{base_url.rstrip('/')}/metrics", timeout=10, trust_env=False).text
    except Exception:
        return {}
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, raw = line.rsplit(" ", 1)
        try:
            values[name] = float(raw)
        except ValueError:
            continue
    result = {
        "queue_depth": next((v for k, v in values.items() if k.startswith("inference_service_queue_depth")), 0.0),
        "kv_cache_utilization": 0.0,
        "gpu_utilization_percent": next((v for k, v in values.items() if k.startswith("gpu_utilization_percent")), 0.0),
    }
    if runtime_metrics_url:
        try:
            runtime_text = httpx.get(runtime_metrics_url, timeout=10, trust_env=False).text
            runtime_values = {}
            for line in runtime_text.splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                name, raw = line.rsplit(" ", 1)
                try:
                    runtime_values[name] = float(raw)
                except ValueError:
                    continue
            result["queue_depth"] = next(
                (v for k, v in runtime_values.items() if "num_requests_waiting" in k),
                result["queue_depth"],
            )
            cache = next(
                (v for k, v in runtime_values.items()
                 if "kv_cache_usage_perc" in k or "gpu_cache_usage_perc" in k),
                None,
            )
            if cache is not None:
                result["kv_cache_utilization"] = cache / 100.0 if cache > 1 else cache
        except Exception:
            pass
    return result


def read_nvidia_smi() -> dict[str, float]:
    """Read host GPU telemetry when nvidia-smi is available."""
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        rows = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        if not rows:
            return {}
        values = [float(part.strip()) for part in rows[0].split(",")]
        if len(values) < 4:
            return {}
        return {
            "gpu_utilization_percent": values[0],
            "gpu_memory_used_mb": values[1],
            "gpu_memory_total_mb": values[2],
            "gpu_power_watts": values[3],
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {}


class TelemetrySampler:
    """Poll telemetry while a benchmark case is actively running."""

    def __init__(self, base_url: str, runtime_metrics_url: str | None = None):
        self.base_url = base_url
        self.runtime_metrics_url = runtime_metrics_url
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            metrics = read_prometheus_metrics(self.base_url, self.runtime_metrics_url)
            metrics.update(read_nvidia_smi())
            if metrics:
                self.samples.append(metrics)
            self._stop.wait(0.25)

    def summary(self, key: str) -> tuple[float, float]:
        values = [sample[key] for sample in self.samples if key in sample]
        return (statistics.mean(values), max(values)) if values else (0.0, 0.0)


def run_case(base_url: str, service: str, concurrency: int, prompt_tokens: int,
             requests: int, max_tokens: int, headers: dict[str, str],
             runtime_metrics_url: str | None = None,
             calibrated_prompt_tokens: int | None = None) -> dict[str, Any]:
    prompt = make_prompt(prompt_tokens)
    started = time.perf_counter()
    samples: list[RequestSample] = []
    telemetry = TelemetrySampler(base_url, runtime_metrics_url)
    telemetry.start()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_request, base_url, service, prompt, max_tokens, headers)
                   for _ in range(requests)]
        for future in as_completed(futures):
            samples.append(future.result())
    telemetry.stop()
    wall = time.perf_counter() - started
    good = [sample for sample in samples if sample.ok]
    durations = [sample.duration_seconds for sample in good]
    ttfts = [sample.ttft_seconds for sample in good if sample.ttft_seconds is not None]
    tpots = [sample.tpot_seconds for sample in good if sample.tpot_seconds is not None]
    completion_tokens = sum(sample.completion_tokens for sample in good)
    prompt_samples = [sample.prompt_tokens for sample in good if sample.prompt_tokens > 0]
    gpu_avg, gpu_max = telemetry.summary("gpu_utilization_percent")
    kv_avg, kv_max = telemetry.summary("kv_cache_utilization")
    queue_avg, queue_max = telemetry.summary("queue_depth")
    return {
        "concurrency": concurrency,
        "prompt_target_tokens": prompt_tokens,
        "prompt_observed_tokens": calibrated_prompt_tokens if calibrated_prompt_tokens is not None else (round(statistics.mean(prompt_samples), 2) if prompt_samples else None),
        "prompt_usage_samples": len(prompt_samples),
        "prompt_usage_coverage": round(len(prompt_samples) / len(good), 3) if good else 0.0,
        "requests": requests,
        "successful_requests": len(good),
        "failed_requests": len(samples) - len(good),
        "ttft_seconds": round(statistics.mean(ttfts), 6) if ttfts else None,
        "tpot_seconds": round(statistics.mean(tpots), 6) if tpots else None,
        "throughput_tokens_per_second": round(completion_tokens / wall, 3) if wall else 0.0,
        "p95_seconds": round(percentile(durations, 0.95), 6),
        "wall_seconds": round(wall, 6),
        "gpu_utilization_percent_avg": round(gpu_avg, 3),
        "gpu_utilization_percent_max": round(gpu_max, 3),
        "gpu_memory_used_mb_avg": round(telemetry.summary("gpu_memory_used_mb")[0], 1),
        "gpu_power_watts_avg": round(telemetry.summary("gpu_power_watts")[0], 1),
        "kv_cache_utilization_avg": round(kv_avg, 4),
        "kv_cache_utilization_max": round(kv_max, 4),
        "queue_depth_avg": round(queue_avg, 3),
        "queue_depth_max": round(queue_max, 3),
        "telemetry_samples": len(telemetry.samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--runtime-metrics-url", default=None,
                        help="Optional vLLM /metrics URL, e.g. http://127.0.0.1:8002/metrics")
    parser.add_argument("--service", default="local-qwen")
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--prompt-tokens", default="128,2048")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--bearer-token", default=None)
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.bearer_token}"} if args.bearer_token else {}
    concurrency_values = [int(value) for value in args.concurrency.split(",")]
    prompt_values = [int(value) for value in args.prompt_tokens.split(",")]
    calibrations = {
        prompt_value: calibrate_prompt_tokens(
            args.base_url, args.service, make_prompt(prompt_value), headers
        )
        for prompt_value in prompt_values
    }
    results = []
    for concurrency in concurrency_values:
        for prompt_value in prompt_values:
            results.append(run_case(
                args.base_url, args.service, concurrency, prompt_value,
                args.requests, args.max_tokens, headers, args.runtime_metrics_url,
                calibrations[prompt_value],
            ))
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
