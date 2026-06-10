"""Tests for the JSON structural scorer and the extract_json helper."""

from __future__ import annotations

import pytest

from assay.models import Prediction, TestCase
from assay.scorers.base import ScoringContext, extract_json
from assay.scorers.json_match import JSONMatch

CTX = ScoringContext()
EXPECTED = {"product": "Aspirin", "batch_number": "ASP-1", "deviation_type": "OOS"}


def _case():
    return TestCase(id="t", input="x", expected=EXPECTED, task="extraction")


async def test_full_match_scores_one():
    pred = Prediction(
        output='{"product": "Aspirin", "batch_number": "ASP-1", "deviation_type": "OOS"}'
    )
    result = await JSONMatch(keys=list(EXPECTED)).score(_case(), pred, CTX)
    assert result.score == 1.0 and result.passed


async def test_partial_match_scores_fraction():
    pred = Prediction(
        output='{"product": "Aspirin", "batch_number": "WRONG", "deviation_type": "OOS"}'
    )
    result = await JSONMatch(keys=list(EXPECTED)).score(_case(), pred, CTX)
    assert result.score == pytest.approx(2 / 3)
    assert not result.passed


async def test_invalid_json_scores_zero():
    pred = Prediction(output="not json at all")
    result = await JSONMatch(keys=list(EXPECTED)).score(_case(), pred, CTX)
    assert result.score == 0.0 and "invalid JSON" in result.detail


async def test_keys_mode_ignores_values():
    pred = Prediction(output='{"product": "x", "batch_number": "y", "deviation_type": "z"}')
    result = await JSONMatch(keys=list(EXPECTED), mode="keys").score(_case(), pred, CTX)
    assert result.score == 1.0


def test_extract_json_from_fenced_block():
    text = '```json\n{"a": 1, "b": 2}\n```'
    assert extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_with_preamble():
    text = 'Here you go: {"a": 1} hope that helps'
    assert extract_json(text) == {"a": 1}


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("there is no json here")
