"""String-comparison scorers for classification and short-answer tasks."""

from __future__ import annotations

import re
import string
from typing import Any

from assay.models import Prediction, ScoreResult, TestCase
from assay.scorers.base import Scorer, ScoringContext

_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def _normalize(text: str, *, ignore_case: bool, strip_punct: bool) -> str:
    out = text.strip()
    if ignore_case:
        out = out.lower()
    if strip_punct:
        out = out.translate(_PUNCT_TABLE)
        out = " ".join(out.split())
    else:
        out = out.rstrip(".")
    return out


class ExactMatch(Scorer):
    """Output must equal the expected value after light normalisation.

    Params: ``ignore_case`` (default True), ``strip_punct`` (default False).
    Light by design — strict format adherence is part of what a benchmark
    measures.
    """

    name = "exact_match"

    async def score(
        self, test_case: TestCase, prediction: Prediction, context: ScoringContext
    ) -> ScoreResult:
        ignore_case = bool(self.params.get("ignore_case", True))
        strip_punct = bool(self.params.get("strip_punct", False))
        expected = str(test_case.expected)
        got = _normalize(prediction.output, ignore_case=ignore_case, strip_punct=strip_punct)
        want = _normalize(expected, ignore_case=ignore_case, strip_punct=strip_punct)
        passed = got == want
        return ScoreResult(
            scorer=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            detail="" if passed else f"expected {want!r}, got {got!r}",
        )


class NormalizedMatch(Scorer):
    """Like :class:`ExactMatch` but also strips punctuation and collapses
    whitespace. Useful when only the content, not the formatting, matters."""

    name = "normalized_match"

    async def score(
        self, test_case: TestCase, prediction: Prediction, context: ScoringContext
    ) -> ScoreResult:
        expected = str(test_case.expected)
        got = _normalize(prediction.output, ignore_case=True, strip_punct=True)
        want = _normalize(expected, ignore_case=True, strip_punct=True)
        passed = got == want
        return ScoreResult(
            scorer=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            detail="" if passed else f"expected {want!r}, got {got!r}",
        )


class Contains(Scorer):
    """Pass if the expected text (or an explicit ``value``) appears in the
    output. Params: ``value`` (overrides expected), ``ignore_case`` (True)."""

    name = "contains"

    async def score(
        self, test_case: TestCase, prediction: Prediction, context: ScoringContext
    ) -> ScoreResult:
        value: Any = self.params.get("value", test_case.expected)
        ignore_case = bool(self.params.get("ignore_case", True))
        needle = str(value)
        haystack = prediction.output
        if ignore_case:
            needle = needle.lower()
            haystack = haystack.lower()
        passed = needle in haystack
        return ScoreResult(
            scorer=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            detail="" if passed else f"missing substring {value!r}",
        )


class Regex(Scorer):
    """Pass if ``pattern`` matches the output. Params: ``pattern`` (required),
    ``flags`` ('i' for ignore-case)."""

    name = "regex"

    async def score(
        self, test_case: TestCase, prediction: Prediction, context: ScoringContext
    ) -> ScoreResult:
        pattern = self.params.get("pattern")
        if not pattern:
            return ScoreResult(self.name, 0.0, False, "regex scorer needs a 'pattern'")
        flags = re.IGNORECASE if "i" in str(self.params.get("flags", "")) else 0
        passed = re.search(pattern, prediction.output, flags) is not None
        return ScoreResult(
            scorer=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            detail="" if passed else f"no match for /{pattern}/",
        )
