"""Dataset loading.

Datasets are JSON Lines: one JSON object per line. Recognised fields:

    input     (required) -- the prompt text shown to the model
    expected             -- ground truth (string for classification, object
                            for extraction, reference text for judged Q&A)
    id                   -- stable identifier (auto-assigned if omitted)
    task                 -- task type hint, e.g. "classification"
    system               -- per-case system prompt override
    metadata             -- free-form dict

Blank lines and ``#`` comment lines are ignored so datasets can be annotated.
"""

from __future__ import annotations

import json
from pathlib import Path

from assay.models import TestCase


def load_dataset(path: str | Path) -> list[TestCase]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"dataset not found: {file_path}")

    cases: list[TestCase] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_no}: invalid JSON ({exc})") from exc
            if "input" not in row:
                raise ValueError(f"{file_path}:{line_no}: row is missing required 'input' field")

            cases.append(
                TestCase(
                    id=str(row.get("id", f"{file_path.stem}-{line_no}")),
                    input=row["input"],
                    expected=row.get("expected"),
                    task=row.get("task", "generic"),
                    system=row.get("system"),
                    metadata=row.get("metadata", {}),
                )
            )

    if not cases:
        raise ValueError(f"{file_path}: no test cases found")
    return cases
