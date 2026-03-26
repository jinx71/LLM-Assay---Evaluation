"""Approximate model pricing (USD per 1M tokens) and cost calculation.

NOTE: list prices change frequently. The numbers below are a sensible
default for relative comparison, not a billing source of truth. Override
any of them from your run config under the top-level ``pricing:`` key to
keep cost estimates accurate. Last reviewed: early 2025.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """Per-million-token prices for a single model."""

    input_per_mtok: float
    output_per_mtok: float


# provider:model -> Price  (USD per 1,000,000 tokens)
DEFAULT_PRICING: dict[str, Price] = {
    "anthropic:claude-3-5-sonnet-latest": Price(3.00, 15.00),
    "anthropic:claude-3-5-haiku-latest": Price(0.80, 4.00),
    "anthropic:claude-3-opus-latest": Price(15.00, 75.00),
    "openai:gpt-4o": Price(2.50, 10.00),
    "openai:gpt-4o-mini": Price(0.15, 0.60),
    "google:gemini-1.5-pro": Price(1.25, 5.00),
    "google:gemini-1.5-flash": Price(0.075, 0.30),
    "huggingface:meta-llama/Meta-Llama-3-8B-Instruct": Price(0.0, 0.0),
    # Deterministic mock models are always free.
    "mock:smart": Price(0.0, 0.0),
    "mock:weak": Price(0.0, 0.0),
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, Price] | None = None,
) -> float:
    """Estimate the USD cost of a single call.

    Unknown models cost 0.0 (the runner surfaces this so you know to add a
    price). ``pricing`` lets a caller pass a merged table that includes
    config overrides.
    """

    table = pricing if pricing is not None else DEFAULT_PRICING
    price = table.get(model)
    if price is None:
        return 0.0
    return (
        input_tokens / 1_000_000 * price.input_per_mtok
        + output_tokens / 1_000_000 * price.output_per_mtok
    )
