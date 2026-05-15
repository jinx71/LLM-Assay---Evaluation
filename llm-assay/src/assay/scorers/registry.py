"""Scorer factory.

A scorer config is a dict like ``{"type": "exact_match", "ignore_case": true}``.
The ``type`` selects the class; the remaining keys become constructor params.
"""

from __future__ import annotations

from typing import Any

from assay.scorers.base import Scorer
from assay.scorers.json_match import JSONMatch
from assay.scorers.llm_judge import LLMJudge
from assay.scorers.string_scorers import Contains, ExactMatch, NormalizedMatch, Regex

SCORERS: dict[str, type[Scorer]] = {
    "exact_match": ExactMatch,
    "normalized_match": NormalizedMatch,
    "contains": Contains,
    "regex": Regex,
    "json_match": JSONMatch,
    "llm_judge": LLMJudge,
}


def build_scorer(config: dict[str, Any]) -> Scorer:
    cfg = dict(config)
    scorer_type = cfg.pop("type", None)
    if scorer_type is None:
        raise ValueError("scorer config must include a 'type' field")
    if scorer_type not in SCORERS:
        raise ValueError(
            f"unknown scorer {scorer_type!r}; available: {', '.join(sorted(SCORERS))}"
        )
    return SCORERS[scorer_type](**cfg)


def available_scorers() -> list[str]:
    return sorted(SCORERS)
