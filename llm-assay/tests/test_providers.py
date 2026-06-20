"""Tests for the mock provider and the provider registry."""

from __future__ import annotations

import json

import pytest

from assay.models import CompletionRequest
from assay.providers.anthropic_provider import AnthropicProvider
from assay.providers.huggingface_provider import HuggingFaceProvider
from assay.providers.mock_provider import MockProvider
from assay.providers.openai_provider import OpenAIProvider
from assay.providers.registry import available_backends, build_provider


async def test_mock_is_deterministic():
    p = MockProvider("smart", seed=1, latency_ms=1)
    req = CompletionRequest(prompt="hello world")
    a = await p.complete(req)
    b = await p.complete(req)
    assert a.output == b.output
    # The provider does not set latency — the runner times each call — so a
    # direct provider call leaves latency at 0.0 by design.
    assert a.latency_ms == 0.0
    assert a.input_tokens > 0 and a.output_tokens > 0


async def test_mock_label_pool_for_classification():
    pool = ["Critical", "Major", "Minor"]
    p = MockProvider("smart", seed=3, latency_ms=1, label_pool=pool)
    req = CompletionRequest(prompt="some deviation", extra={"task": "classification"})
    out = (await p.complete(req)).output
    assert out in pool


async def test_mock_json_template_for_extraction():
    template = {"product": "X", "batch_number": "Y", "deviation_type": "Z"}
    p = MockProvider("smart", seed=5, latency_ms=1, json_template=template)
    req = CompletionRequest(prompt="extract this", extra={"task": "extraction"})
    out = (await p.complete(req)).output
    parsed = json.loads(out)  # must be valid JSON
    assert set(parsed.keys()) == set(template.keys())


async def test_different_seeds_can_diverge():
    pool = ["Critical", "Major", "Minor"]
    p1 = MockProvider("a", seed=1, latency_ms=1, label_pool=pool)
    p2 = MockProvider("b", seed=2, latency_ms=1, label_pool=pool)
    prompts = [f"case {i}" for i in range(12)]

    async def labels(provider):
        out = []
        for prompt in prompts:
            req = CompletionRequest(prompt=prompt, extra={"task": "classification"})
            out.append((await provider.complete(req)).output)
        return out

    assert await labels(p1) != await labels(p2)  # seeds produce different patterns


def test_registry_builds_each_backend():
    assert isinstance(build_provider("mock:smart"), MockProvider)
    assert isinstance(build_provider("anthropic:claude-3-5-haiku-latest"), AnthropicProvider)
    assert isinstance(build_provider("openai:gpt-4o-mini"), OpenAIProvider)
    hf = build_provider("huggingface:meta-llama/Meta-Llama-3-8B-Instruct")
    assert isinstance(hf, HuggingFaceProvider)
    # The slash in the model id must survive the backend split.
    assert hf.model == "meta-llama/Meta-Llama-3-8B-Instruct"


def test_registry_rejects_bad_specs():
    with pytest.raises(ValueError):
        build_provider("noseparator")
    with pytest.raises(ValueError):
        build_provider("unknownbackend:model")


def test_available_backends_listed():
    backends = available_backends()
    assert {"anthropic", "openai", "google", "huggingface", "mock"} <= set(backends)
