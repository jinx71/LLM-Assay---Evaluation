"""Tests for the reporters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assay.config import (
    Config,
    DatasetConfig,
    ModelConfig,
    ReportConfig,
    RunConfig,
)
from assay.report import run_to_dict, to_html, to_json, to_markdown
from assay.runner import RunResult, run_eval


@pytest.fixture
async def small_run(classification_dataset: Path, tmp_path: Path) -> RunResult:
    pool = ["Critical", "Major", "Minor"]
    config = Config(
        models=[
            ModelConfig(
                spec="mock:smart",
                options={"seed": 7, "latency_ms": 1, "label_pool": pool},
            ),
            ModelConfig(
                spec="mock:weak",
                options={"seed": 99, "latency_ms": 1, "label_pool": pool},
            ),
        ],
        datasets=[
            DatasetConfig(
                path=str(classification_dataset),
                name="mini",
                scorers=[{"type": "exact_match"}],
            )
        ],
        run=RunConfig(cache=False, cache_dir=str(tmp_path / "cache"), max_retries=0),
        report=ReportConfig(formats=[]),
    )
    return await run_eval(config)


def test_run_to_dict_shape(small_run: RunResult):
    data = run_to_dict(small_run)
    assert set(data) >= {"meta", "leaderboard", "cells", "records", "datasets"}
    assert len(data["leaderboard"]) == 2
    assert len(data["records"]) == 6


def test_json_envelope_written_and_reloadable(small_run: RunResult, tmp_path: Path):
    out = to_json(small_run, tmp_path / "r.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert "data" in payload and "leaderboard" in payload["data"]
    assert isinstance(payload["message"], str)


def test_markdown_contains_models_and_leaderboard(small_run: RunResult):
    md = to_markdown(small_run)
    assert "Leaderboard" in md
    assert "mock:smart" in md and "mock:weak" in md
    assert "| # | Model |" in md


def test_html_is_self_contained(small_run: RunResult, tmp_path: Path):
    out = to_html(small_run, tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "<html" in html
    assert "run-data" in html  # embedded JSON data island
    assert "mock:smart" in html
    assert "Chart" in html  # chart wiring present


def test_html_escapes_script_breakout(classification_dataset: Path, tmp_path: Path):
    # A model output containing </script> must not break the data island.
    from assay.models import EvalRecord, Prediction, ScoreResult

    rec = EvalRecord(
        model="mock:x",
        dataset="mini",
        test_case_id="c1",
        repeat=0,
        prediction=Prediction(output="danger </script> text"),
        scores=[ScoreResult("exact_match", 1.0, True, "")],
    )
    run = RunResult(
        records=[rec],
        leaderboard=[],
        cells=[],
        datasets=["mini"],
        started_at="t",
        finished_at="t",
        duration_s=0.0,
        meta={"total_evaluations": 1, "repeats": 1},
    )
    out = to_html(run, tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    # The raw closing tag must have been neutralised inside the data island.
    assert "danger </script> text" not in html
    assert "<\\/script>" in html
