"""Tiny shared helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def stopwatch() -> Iterator[dict[str, float]]:
    """Measure wall-clock elapsed time in milliseconds.

    Usage::

        with stopwatch() as t:
            do_work()
        print(t["ms"])  # read after the block exits
    """

    box = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - start) * 1000.0


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no external dependency).

    ``pct`` is expressed 0–100. Returns 0.0 for an empty input.
    """

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac
