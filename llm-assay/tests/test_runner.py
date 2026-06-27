"""End-to-end runner tests, using deterministic mock providers."""

from __future__ import annotations

from pathlib import Path

from assay.config import (
    Config,
    DatasetConfig,
    ModelConfig,
    ReportConfig,
    RunConfig,
)
from assay.runner import run_eval


def _config(dataset_path: Path, cache_dir: Path) -> Config:
    pool = ["Critical", "Major", "Minor"]
    return Config(
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
                path=str(dataset_path),
                name="mini",
                system="Classify severity.",
                scorers=[{"type": "exact_match", "ignore_case": True}],
            )
        ],
        run=RunConfig(concurrency=4, cache=False, cache_dir=str(cache_dir), max_retries=0),
        report=ReportConfig(formats=[]),
    )


async def test_runner_produces_full_matrix(classification_dataset: Path, tmp_path: Path):
    config = _config(classification_dataset, tmp_path / "cache")
    run = await run_eval(config)

    # 3 cases x 2 models x 1 repeat = 6 records.
    assert len(run.records) == 6
    assert run.meta["total_evaluations"] == 6
    assert len(run.leaderboard) == 2
    assert run.duration_s >= 0
    assert run.datasets == ["mini"]


async def test_runner_zero_cost_and_no_errors(classification_dataset: Path, tmp_path: Path):
    run = await run_eval(_config(classification_dataset, tmp_path / "cache"))
    assert sum(r.cost_usd for r in run.records) == 0.0  # mock is free
    assert all(r.prediction.ok for r in run.records)
    assert all(m.error_rate == 0.0 for m in run.leaderboard)


async def test_runner_leaderboard_sorted_by_score(classification_dataset: Path, tmp_path: Path):
    run = await run_eval(_config(classification_dataset, tmp_path / "cache"))
    scores = [m.mean_score for m in run.leaderboard]
    assert scores == sorted(scores, reverse=True)


async def test_runner_limit(classification_dataset: Path, tmp_path: Path):
    config = _config(classification_dataset, tmp_path / "cache")
    run = await run_eval(config, limit=1)
    # 1 case x 2 models = 2 records.
    assert len(run.records) == 2


async def test_runner_progress_callback(classification_dataset: Path, tmp_path: Path):
    config = _config(classification_dataset, tmp_path / "cache")
    seen = {"n": 0}
    await run_eval(config, progress=lambda: seen.__setitem__("n", seen["n"] + 1))
    assert seen["n"] == 6


async def test_cells_have_scorer_means(classification_dataset: Path, tmp_path: Path):
    run = await run_eval(_config(classification_dataset, tmp_path / "cache"))
    assert run.cells
    for cell in run.cells:
        assert "exact_match" in cell.scorer_means
        assert 0.0 <= cell.scorer_means["exact_match"] <= 1.0
