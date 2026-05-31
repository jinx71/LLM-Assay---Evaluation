"""Shared fixtures and stubs for the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assay.models import CompletionRequest, Prediction
from assay.providers.base import LLMProvider


class StubProvider(LLMProvider):
    """A provider that always returns a fixed output (for judge tests)."""

    backend = "stub"

    def __init__(self, model: str = "fixed", *, output: str = "ok", **options: object) -> None:
        super().__init__(model, **options)
        self.output = output
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> Prediction:
        self.calls += 1
        return Prediction(output=self.output, input_tokens=5, output_tokens=5)


@pytest.fixture
def classification_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "mini_class.jsonl"
    rows = [
        {"id": "c1", "task": "classification", "expected": "Critical", "input": "sterility fail"},
        {"id": "c2", "task": "classification", "expected": "Minor", "input": "undated initial"},
        {"id": "c3", "task": "classification", "expected": "Major", "input": "pressure lost"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path
