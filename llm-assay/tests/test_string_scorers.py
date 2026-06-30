"""Tests for the string-comparison scorers."""

from __future__ import annotations

from assay.models import Prediction, TestCase
from assay.scorers.base import ScoringContext
from assay.scorers.string_scorers import Contains, ExactMatch, NormalizedMatch, Regex

CTX = ScoringContext()


def _case(expected):
    return TestCase(id="t", input="x", expected=expected)


def _pred(output):
    return Prediction(output=output)


async def test_exact_match_pass_ignorecase():
    result = await ExactMatch().score(_case("Critical"), _pred("critical"), CTX)
    assert result.passed and result.score == 1.0


async def test_exact_match_trailing_period():
    # Default normalisation strips a trailing period.
    result = await ExactMatch().score(_case("Major"), _pred("Major."), CTX)
    assert result.passed


async def test_exact_match_fail():
    result = await ExactMatch().score(_case("Minor"), _pred("Major"), CTX)
    assert not result.passed and result.score == 0.0
    assert "expected" in result.detail


async def test_normalized_match_strips_punct_and_space():
    result = await NormalizedMatch().score(
        _case("dead leg"), _pred("  Dead-Leg!! "), CTX
    )
    assert result.passed


async def test_contains_uses_expected():
    result = await Contains().score(_case("turbulent"), _pred("maintain turbulent flow"), CTX)
    assert result.passed


async def test_contains_explicit_value_miss():
    result = await Contains(value="biofilm").score(_case(None), _pred("just water"), CTX)
    assert not result.passed


async def test_regex_match_and_flags():
    scorer = Regex(pattern=r"\bMAJOR\b", flags="i")
    ok = await scorer.score(_case(None), _pred("severity: major"), CTX)
    assert ok.passed
    miss = await Regex(pattern=r"\d{4}").score(_case(None), _pred("no digits"), CTX)
    assert not miss.passed


async def test_regex_requires_pattern():
    result = await Regex().score(_case(None), _pred("x"), CTX)
    assert not result.passed and "pattern" in result.detail
