"""A deterministic, dependency-free mock provider.

Why this exists: an evaluation harness must be runnable and testable
*without* network access or paid API keys. The mock lets the whole pipeline
— runner, scorers, cost model, reports — execute end to end in CI and in
the default demo config. Output is a pure function of ``(seed, prompt)`` so
runs are reproducible, and two mock models with different seeds produce
different answers, which makes the demo leaderboard meaningful.

It is a stand-in, not a real model: it does not understand the task. Swap in
a real provider (and an API key) for genuine evaluation numbers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from typing import Any

from assay.models import CompletionRequest, Prediction
from assay.providers.base import LLMProvider


def _digest(*parts: str) -> int:
    raw = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest(), 16)


class MockProvider(LLMProvider):
    """Reproducible fake model.

    Options:
        seed: int            -- changes the deterministic answers
        latency_ms: float    -- base simulated latency
        label_pool: list[str] -- valid labels for classification-style tasks
        json_template: dict   -- shape returned for extraction-style tasks
        canned: str           -- fixed response for everything else
    """

    backend = "mock"

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.seed = int(options.get("seed", 0))
        self.latency_ms = float(options.get("latency_ms", 25.0))
        self.label_pool: list[str] | None = options.get("label_pool")
        self.json_template: dict[str, Any] | None = options.get("json_template")
        self.canned: str | None = options.get("canned")

    async def complete(self, request: CompletionRequest) -> Prediction:
        # Deterministic-but-varied simulated latency so latency stats are
        # non-degenerate across cases (identical across repeats by design).
        rng = random.Random(_digest(str(self.seed), request.prompt, "lat"))
        jitter = 0.6 + rng.random() * 0.8
        await asyncio.sleep(self.latency_ms / 1000.0 * jitter)

        output = self._render(request)
        return Prediction(
            output=output,
            input_tokens=max(1, len(request.prompt) // 4),
            output_tokens=max(1, len(output) // 4),
            raw={"mock": True, "seed": self.seed},
        )

    def _render(self, request: CompletionRequest) -> str:
        task = str(request.extra.get("task", "")).lower()
        h = _digest(str(self.seed), request.prompt)

        if "class" in task and self.label_pool:
            return self.label_pool[h % len(self.label_pool)]

        if "extract" in task and self.json_template is not None:
            obj: dict[str, Any] = {}
            for key, value in self.json_template.items():
                # Drop the occasional field to vary structural-match scores.
                if _digest(str(self.seed), request.prompt, key) % 5 == 0:
                    obj[key] = "UNKNOWN"
                else:
                    obj[key] = value
            return json.dumps(obj)

        if self.canned is not None:
            return self.canned

        return f"[mock-{self.seed}] {request.prompt[:120]}"
