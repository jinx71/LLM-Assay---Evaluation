"""Core data structures for the Assay evaluation harness.

Everything that flows through the pipeline (requests, predictions, scores,
and per-evaluation records) is a plain dataclass defined here. Keeping the
data model in one place makes the rest of the package easy to reason about
and trivial to serialise for reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompletionRequest:
    """A single request sent to a provider.

    ``extra`` carries non-essential hints (for example the task type) that
    some providers — notably the deterministic mock — use to shape output.
    Real providers ignore keys they do not recognise.
    """

    prompt: str
    system: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """A normalised completion result returned by any provider.

    Latency is populated by the runner (which times the call including
    retries), not by the provider, so the number is consistent across every
    backend.
    """

    output: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prediction:
        return cls(
            output=data["output"],
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            latency_ms=data.get("latency_ms", 0.0),
            error=data.get("error"),
        )


@dataclass
class TestCase:
    """A single evaluation example loaded from a dataset."""

    __test__ = False  # this is a domain model, not a pytest test class

    id: str
    input: str
    expected: Any = None
    task: str = "generic"
    system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """The outcome of applying one scorer to one prediction.

    ``score`` is always normalised to the 0.0–1.0 range so that different
    scorers can be averaged and ranked on the same scale.
    """

    scorer: str
    score: float
    passed: bool
    detail: str = ""


@dataclass
class EvalRecord:
    """Everything about one (model, dataset, test case, repeat) evaluation."""

    model: str
    dataset: str
    test_case_id: str
    repeat: int
    prediction: Prediction
    scores: list[ScoreResult] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def primary_score(self) -> float:
        """The headline score = the first configured scorer for the dataset."""
        return self.scores[0].score if self.scores else 0.0

    @property
    def primary_passed(self) -> bool:
        return self.scores[0].passed if self.scores else False
