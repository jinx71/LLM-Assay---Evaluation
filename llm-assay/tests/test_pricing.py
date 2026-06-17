"""Tests for the pricing / cost model."""

from __future__ import annotations

from assay.pricing import DEFAULT_PRICING, Price, estimate_cost


def test_known_model_cost():
    # gpt-4o-mini: $0.15 in / $0.60 out per 1M tokens.
    cost = estimate_cost("openai:gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == 0.15 + 0.60


def test_partial_token_cost():
    cost = estimate_cost("openai:gpt-4o-mini", 500_000, 0)
    assert abs(cost - 0.075) < 1e-9


def test_unknown_model_is_free():
    assert estimate_cost("openai:does-not-exist", 1_000, 1_000) == 0.0


def test_mock_models_are_free():
    assert estimate_cost("mock:smart", 10_000, 10_000) == 0.0


def test_pricing_override():
    table = dict(DEFAULT_PRICING)
    table["custom:model"] = Price(10.0, 20.0)
    cost = estimate_cost("custom:model", 1_000_000, 1_000_000, table)
    assert cost == 30.0
