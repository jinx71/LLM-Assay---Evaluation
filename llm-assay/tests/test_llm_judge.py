"""Tests for the LLM-as-judge scorer (using a stub judge provider)."""

from __future__ import annotations

from assay.models import Prediction, TestCase
from assay.scorers.base import ScoringContext
from assay.scorers.llm_judge import LLMJudge
from tests.conftest import StubProvider


def _case():
    return TestCase(id="t", input="What is CAPA?", expected="Corrective and Preventive Action")


async def test_judge_parses_score_and_passes():
    judge = StubProvider(output='{"score": 0.9, "reason": "accurate and complete"}')
    ctx = ScoringContext(judge_provider=judge)
    result = await LLMJudge(threshold=0.7).score(_case(), Prediction(output="CAPA = ..."), ctx)
    assert result.score == 0.9 and result.passed
    assert "accurate" in result.detail
    assert judge.calls == 1


async def test_judge_below_threshold_fails():
    judge = StubProvider(output='{"score": 0.4, "reason": "vague"}')
    ctx = ScoringContext(judge_provider=judge)
    result = await LLMJudge(threshold=0.7).score(_case(), Prediction(output="dunno"), ctx)
    assert result.score == 0.4 and not result.passed


async def test_judge_missing_provider():
    result = await LLMJudge().score(_case(), Prediction(output="x"), ScoringContext())
    assert result.score == 0.0 and not result.passed
    assert "no judge" in result.detail


async def test_judge_unparseable_verdict():
    judge = StubProvider(output="the answer is pretty good honestly")
    ctx = ScoringContext(judge_provider=judge)
    result = await LLMJudge().score(_case(), Prediction(output="x"), ctx)
    assert result.score == 0.0 and not result.passed


async def test_judge_clamps_out_of_range_score():
    judge = StubProvider(output='{"score": 1.7, "reason": "over"}')
    ctx = ScoringContext(judge_provider=judge)
    result = await LLMJudge().score(_case(), Prediction(output="x"), ctx)
    assert result.score == 1.0
