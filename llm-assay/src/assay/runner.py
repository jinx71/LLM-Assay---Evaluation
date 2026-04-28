"""The evaluation runner.

Orchestrates the full matrix of (dataset x test case x model x repeat), with
bounded concurrency, per-call timeout, retry-with-backoff, and caching. It
owns latency measurement so timings are comparable across backends, then
aggregates everything into a :class:`RunResult` the reporters can render.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from assay.cache import Cache
from assay.config import Config
from assay.dataset import load_dataset
from assay.models import (
    CompletionRequest,
    EvalRecord,
    Prediction,
    ScoreResult,
    TestCase,
)
from assay.pricing import estimate_cost
from assay.providers.base import LLMProvider, ProviderError
from assay.providers.registry import build_provider
from assay.scorers.base import Scorer, ScoringContext
from assay.scorers.registry import build_scorer
from assay.utils import percentile


# --------------------------------------------------------------------------- #
# Aggregated result types
# --------------------------------------------------------------------------- #
@dataclass
class CellStats:
    """Stats for one model on one dataset."""

    model: str
    dataset: str
    n: int
    mean_score: float
    pass_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    error_rate: float
    scorer_means: dict[str, float]


@dataclass
class ModelStats:
    """Aggregate stats for one model across every dataset (leaderboard row)."""

    model: str
    n: int
    mean_score: float
    pass_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    error_rate: float


@dataclass
class RunResult:
    records: list[EvalRecord]
    leaderboard: list[ModelStats]
    cells: list[CellStats]
    datasets: list[str]
    started_at: str
    finished_at: str
    duration_s: float
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Core call path
# --------------------------------------------------------------------------- #
def _backoff_seconds(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    return min(cap, base * (2**attempt)) + random.random() * 0.25


async def _call_with_retries(
    provider: LLMProvider,
    request: CompletionRequest,
    cache: Cache,
    repeat: int,
    *,
    timeout_s: float,
    max_retries: int,
) -> Prediction:
    cache_key = cache.key(provider.id, request, repeat)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    last_error = "unknown error"
    last_latency = 0.0

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            prediction = await asyncio.wait_for(
                provider.complete(request), timeout=timeout_s
            )
            prediction.latency_ms = (time.perf_counter() - start) * 1000.0
            cache.set(cache_key, prediction)
            return prediction
        except ProviderError as exc:
            last_latency = (time.perf_counter() - start) * 1000.0
            last_error = str(exc)
            if not exc.retryable or attempt == max_retries:
                break
        except asyncio.TimeoutError:
            last_latency = (time.perf_counter() - start) * 1000.0
            last_error = f"timeout after {timeout_s}s"
            if attempt == max_retries:
                break
        except Exception as exc:  # noqa: BLE001 - record anything unexpected
            last_latency = (time.perf_counter() - start) * 1000.0
            last_error = f"{type(exc).__name__}: {exc}"
            break

        await asyncio.sleep(_backoff_seconds(attempt))

    return Prediction(output="", error=last_error, latency_ms=last_latency)


async def _evaluate_one(
    provider: LLMProvider,
    case: TestCase,
    scorers: list[Scorer],
    dataset_name: str,
    dataset_system: str | None,
    repeat: int,
    *,
    sem: asyncio.Semaphore,
    cache: Cache,
    judge: LLMProvider | None,
    pricing: dict,
    config: Config,
) -> EvalRecord:
    async with sem:
        request = CompletionRequest(
            prompt=case.input,
            system=case.system or dataset_system,
            max_tokens=config.run.max_tokens,
            temperature=config.run.temperature,
            extra={"task": case.task},
        )
        prediction = await _call_with_retries(
            provider,
            request,
            cache,
            repeat,
            timeout_s=config.run.timeout_s,
            max_retries=config.run.max_retries,
        )

        cost = estimate_cost(
            provider.id, prediction.input_tokens, prediction.output_tokens, pricing
        )

        context = ScoringContext(judge_provider=judge)
        scores: list[ScoreResult] = []
        if prediction.ok:
            for scorer in scorers:
                try:
                    scores.append(await scorer.score(case, prediction, context))
                except Exception as exc:  # noqa: BLE001
                    scores.append(
                        ScoreResult(scorer.name, 0.0, False, f"scorer error: {exc}")
                    )
        else:
            # A failed call scores zero on every configured scorer.
            for scorer in scorers:
                scores.append(ScoreResult(scorer.name, 0.0, False, "no prediction"))

        return EvalRecord(
            model=provider.id,
            dataset=dataset_name,
            test_case_id=case.id,
            repeat=repeat,
            prediction=prediction,
            scores=scores,
            cost_usd=cost,
        )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(records: list[EvalRecord], datasets: list[str]) -> tuple[
    list[ModelStats], list[CellStats]
]:
    models = sorted({r.model for r in records})

    cells: list[CellStats] = []
    for model in models:
        for dataset in datasets:
            subset = [r for r in records if r.model == model and r.dataset == dataset]
            if not subset:
                continue
            latencies = [r.prediction.latency_ms for r in subset]
            scorer_names = subset[0].scores and [s.scorer for s in subset[0].scores] or []
            scorer_means = {
                name: _mean(
                    [s.score for r in subset for s in r.scores if s.scorer == name]
                )
                for name in scorer_names
            }
            cells.append(
                CellStats(
                    model=model,
                    dataset=dataset,
                    n=len(subset),
                    mean_score=_mean([r.primary_score for r in subset]),
                    pass_rate=_mean([1.0 if r.primary_passed else 0.0 for r in subset]),
                    mean_latency_ms=_mean(latencies),
                    p95_latency_ms=percentile(latencies, 95),
                    total_cost_usd=sum(r.cost_usd for r in subset),
                    error_rate=_mean(
                        [0.0 if r.prediction.ok else 1.0 for r in subset]
                    ),
                    scorer_means=scorer_means,
                )
            )

    leaderboard: list[ModelStats] = []
    for model in models:
        subset = [r for r in records if r.model == model]
        latencies = [r.prediction.latency_ms for r in subset]
        leaderboard.append(
            ModelStats(
                model=model,
                n=len(subset),
                mean_score=_mean([r.primary_score for r in subset]),
                pass_rate=_mean([1.0 if r.primary_passed else 0.0 for r in subset]),
                mean_latency_ms=_mean(latencies),
                p95_latency_ms=percentile(latencies, 95),
                total_cost_usd=sum(r.cost_usd for r in subset),
                error_rate=_mean([0.0 if r.prediction.ok else 1.0 for r in subset]),
            )
        )

    leaderboard.sort(key=lambda m: (-m.mean_score, m.mean_latency_ms, m.total_cost_usd))
    return leaderboard, cells


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
async def run_eval(
    config: Config,
    *,
    limit: int | None = None,
    progress: Any = None,
) -> RunResult:
    """Execute the full evaluation matrix and return aggregated results.

    ``limit`` caps the number of test cases per dataset (handy for smoke
    runs). ``progress`` is an optional callable invoked once per completed
    evaluation, used by the CLI to drive a progress bar.
    """

    started = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    providers: dict[str, LLMProvider] = {
        m.spec: build_provider(m.spec, m.options) for m in config.models
    }
    judge: LLMProvider | None = (
        build_provider(config.judge.spec, config.judge.options)
        if config.judge.spec
        else None
    )

    cache = Cache(enabled=config.run.cache, directory=config.run.cache_dir)
    sem = asyncio.Semaphore(config.run.concurrency)

    loaded: list[tuple[str, str | None, list[TestCase], list[Scorer]]] = []
    for dataset_cfg in config.datasets:
        cases = load_dataset(dataset_cfg.path)
        if limit is not None:
            cases = cases[:limit]
        scorers = [build_scorer(s) for s in dataset_cfg.scorers]
        loaded.append((dataset_cfg.name, dataset_cfg.system, cases, scorers))

    async def _wrapped(coro: Any) -> EvalRecord:
        record = await coro
        if progress is not None:
            progress()
        return record

    tasks: list[asyncio.Task[EvalRecord]] = []
    for dataset_name, dataset_system, cases, scorers in loaded:
        for case in cases:
            for provider in providers.values():
                for repeat in range(config.run.repeats):
                    coro = _evaluate_one(
                        provider,
                        case,
                        scorers,
                        dataset_name,
                        dataset_system,
                        repeat,
                        sem=sem,
                        cache=cache,
                        judge=judge,
                        pricing=config.pricing,
                        config=config,
                    )
                    tasks.append(asyncio.create_task(_wrapped(coro)))

    try:
        records = await asyncio.gather(*tasks)
    finally:
        for provider in providers.values():
            await provider.aclose()
        if judge is not None:
            await judge.aclose()

    datasets = [name for name, _, _, _ in loaded]
    leaderboard, cells = _aggregate(list(records), datasets)
    finished = datetime.now(timezone.utc)

    return RunResult(
        records=list(records),
        leaderboard=leaderboard,
        cells=cells,
        datasets=datasets,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_s=time.perf_counter() - start_perf,
        meta={
            "models": [p.id for p in providers.values()],
            "judge": judge.id if judge else None,
            "repeats": config.run.repeats,
            "concurrency": config.run.concurrency,
            "total_evaluations": len(records),
        },
    )
