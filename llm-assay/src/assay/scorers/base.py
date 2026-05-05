"""Scorer interface and shared helpers.

A scorer turns one prediction into a normalised 0.0–1.0 score plus a
pass/fail flag. Scorers are async because some (the LLM judge) need to make
their own model call; the simple string scorers complete synchronously but
keep the async signature for a uniform interface.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any

from assay.models import Prediction, ScoreResult, TestCase
from assay.providers.base import LLMProvider


@dataclass
class ScoringContext:
    """Side channel passed to every scorer.

    Most scorers ignore it; :class:`LLMJudge` reads ``judge_provider``.
    """

    judge_provider: LLMProvider | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Scorer(abc.ABC):
    """Base class for all scorers."""

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abc.abstractmethod
    async def score(
        self, test_case: TestCase, prediction: Prediction, context: ScoringContext
    ) -> ScoreResult:
        """Return a :class:`ScoreResult` for one prediction."""


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response.

    Tolerates ```json fences and leading/trailing prose by grabbing the
    outermost balanced ``{...}`` or ``[...]`` block. Raises ``ValueError``
    if nothing parseable is found.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip a fenced block: ```json\n...\n```
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced object/array in the string.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError("no parseable JSON found in output")
