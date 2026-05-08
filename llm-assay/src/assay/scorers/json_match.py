"""Structural JSON scorer for extraction tasks.

Gives partial credit: the score is the fraction of expected fields the model
got right, so a model that extracts 3 of 4 fields scores 0.75 rather than a
flat zero. Two modes:

    value  -- field value must match the expected value (default)
    keys   -- field need only be present (structure check)
"""

from __future__ import annotations

from typing import Any

from assay.models import Prediction, ScoreResult, TestCase
from assay.scorers.base import Scorer, ScoringContext, extract_json


def _norm(value: Any) -> str:
    return str(value).strip().lower()


class JSONMatch(Scorer):
    """Field-level match against an expected JSON object.

    Params:
        keys: list[str]  -- which fields to check (default: all expected keys)
        mode: 'value' | 'keys'
    """

    name = "json_match"

    async def score(
        self, test_case: TestCase, prediction: Prediction, context: ScoringContext
    ) -> ScoreResult:
        expected = test_case.expected
        if not isinstance(expected, dict):
            return ScoreResult(self.name, 0.0, False, "expected value is not a JSON object")

        try:
            got = extract_json(prediction.output)
        except ValueError as exc:
            return ScoreResult(self.name, 0.0, False, f"invalid JSON: {exc}")
        if not isinstance(got, dict):
            return ScoreResult(self.name, 0.0, False, "model output is not a JSON object")

        keys: list[str] = self.params.get("keys") or list(expected.keys())
        mode = str(self.params.get("mode", "value"))
        if not keys:
            return ScoreResult(self.name, 0.0, False, "no keys to compare")

        correct = 0
        misses: list[str] = []
        for key in keys:
            if mode == "keys":
                ok = key in got
            else:
                ok = key in got and _norm(got.get(key)) == _norm(expected.get(key))
            if ok:
                correct += 1
            else:
                misses.append(key)

        score = correct / len(keys)
        detail = f"{correct}/{len(keys)} fields"
        if misses:
            detail += f" (missed: {', '.join(misses)})"
        return ScoreResult(self.name, score, score == 1.0, detail)
