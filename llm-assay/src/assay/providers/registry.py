"""Provider factory.

Maps a ``backend:model`` spec to a concrete provider instance. This is the
single place that knows about every backend, so the runner stays generic.
"""

from __future__ import annotations

from typing import Any

from assay.providers.anthropic_provider import AnthropicProvider
from assay.providers.base import LLMProvider
from assay.providers.google_provider import GoogleProvider
from assay.providers.huggingface_provider import HuggingFaceProvider
from assay.providers.mock_provider import MockProvider
from assay.providers.openai_provider import OpenAIProvider

BACKENDS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
    "huggingface": HuggingFaceProvider,
    "mock": MockProvider,
}


def build_provider(spec: str, options: dict[str, Any] | None = None) -> LLMProvider:
    """Construct a provider from a ``backend:model`` spec.

    ``huggingface:meta-llama/Meta-Llama-3-8B`` is supported — only the first
    colon separates backend from model id.
    """

    backend, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(
            f"invalid model spec {spec!r}; expected 'backend:model' "
            f"(e.g. 'anthropic:claude-3-5-sonnet-latest')"
        )
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; available: {', '.join(sorted(BACKENDS))}"
        )
    return BACKENDS[backend](model, **(options or {}))


def available_backends() -> list[str]:
    return sorted(BACKENDS)
