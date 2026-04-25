"""Reporters.

Four output formats, all driven from the same :class:`RunResult`:

    terminal  -- rich tables for an at-a-glance leaderboard in the shell
    json      -- machine-readable dump (wrapped in a success/data envelope)
    markdown  -- a leaderboard table ready to paste into a README or PR
    html      -- a self-contained, chart-rich report for sharing
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from rich.table import Table

from assay.runner import RunResult

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def run_to_dict(run: RunResult) -> dict[str, Any]:
    """Flatten a RunResult into JSON-friendly primitives."""

    return {
        "meta": run.meta,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "duration_s": round(run.duration_s, 3),
        "datasets": run.datasets,
        "leaderboard": [
            {
                "model": m.model,
                "evaluations": m.n,
                "mean_score": round(m.mean_score, 4),
                "pass_rate": round(m.pass_rate, 4),
                "mean_latency_ms": round(m.mean_latency_ms, 1),
                "p95_latency_ms": round(m.p95_latency_ms, 1),
                "total_cost_usd": round(m.total_cost_usd, 6),
                "error_rate": round(m.error_rate, 4),
            }
            for m in run.leaderboard
        ],
        "cells": [
            {
                "model": c.model,
                "dataset": c.dataset,
                "evaluations": c.n,
                "mean_score": round(c.mean_score, 4),
                "pass_rate": round(c.pass_rate, 4),
                "mean_latency_ms": round(c.mean_latency_ms, 1),
                "p95_latency_ms": round(c.p95_latency_ms, 1),
                "total_cost_usd": round(c.total_cost_usd, 6),
                "error_rate": round(c.error_rate, 4),
                "scorer_means": {k: round(v, 4) for k, v in c.scorer_means.items()},
            }
            for c in run.cells
        ],
        "records": [
            {
                "model": r.model,
                "dataset": r.dataset,
                "test_case_id": r.test_case_id,
                "repeat": r.repeat,
                "output": r.prediction.output,
                "error": r.prediction.error,
                "latency_ms": round(r.prediction.latency_ms, 1),
                "input_tokens": r.prediction.input_tokens,
                "output_tokens": r.prediction.output_tokens,
                "cost_usd": round(r.cost_usd, 6),
                "scores": [
                    {"scorer": s.scorer, "score": s.score, "passed": s.passed, "detail": s.detail}
                    for s in r.scores
                ],
            }
            for r in run.records
        ],
    }


# --------------------------------------------------------------------------- #
# Terminal
# --------------------------------------------------------------------------- #
def to_terminal(run: RunResult, console: Console | None = None) -> None:
    console = console or Console()

    table = Table(title="Leaderboard  (ranked by mean score)", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model", style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("Pass", justify="right")
    table.add_column("Mean ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("Cost $", justify="right")
    table.add_column("Err", justify="right")

    for rank, m in enumerate(run.leaderboard, start=1):
        err_style = "red" if m.error_rate > 0 else "green"
        table.add_row(
            str(rank),
            m.model,
            f"{m.mean_score:.1%}",
            f"{m.pass_rate:.1%}",
            f"{m.mean_latency_ms:.0f}",
            f"{m.p95_latency_ms:.0f}",
            f"{m.total_cost_usd:.4f}",
            f"[{err_style}]{m.error_rate:.0%}[/{err_style}]",
        )
    console.print(table)

    for dataset in run.datasets:
        cells = [c for c in run.cells if c.dataset == dataset]
        if not cells:
            continue
        cells.sort(key=lambda c: -c.mean_score)
        dt = Table(title=f"Dataset: {dataset}", title_style="bold")
        dt.add_column("Model", style="cyan")
        dt.add_column("Score", justify="right")
        dt.add_column("Pass", justify="right")
        dt.add_column("Mean ms", justify="right")
        dt.add_column("Cost $", justify="right")
        for c in cells:
            dt.add_row(
                c.model,
                f"{c.mean_score:.1%}",
                f"{c.pass_rate:.1%}",
                f"{c.mean_latency_ms:.0f}",
                f"{c.total_cost_usd:.4f}",
            )
        console.print(dt)

    console.print(
        f"[dim]{run.meta.get('total_evaluations', 0)} evaluations "
        f"in {run.duration_s:.1f}s[/dim]"
    )


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def to_json(run: RunResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "success": True,
        "message": f"Evaluated {len(run.leaderboard)} model(s) "
        f"across {len(run.datasets)} dataset(s).",
        "data": run_to_dict(run),
    }
    out.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def to_markdown(run: RunResult, path: str | Path | None = None) -> str:
    lines: list[str] = []
    lines.append("# LLM Assay — Evaluation Report")
    lines.append("")
    lines.append(f"- **Run:** {run.started_at}")
    lines.append(f"- **Datasets:** {', '.join(run.datasets)}")
    lines.append(f"- **Evaluations:** {run.meta.get('total_evaluations', 0)}")
    lines.append(f"- **Duration:** {run.duration_s:.1f}s")
    if run.meta.get("judge"):
        lines.append(f"- **Judge:** {run.meta['judge']}")
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| # | Model | Score | Pass | Mean ms | p95 ms | Cost $ | Err |")
    lines.append("|---|-------|------:|-----:|--------:|-------:|-------:|----:|")
    for rank, m in enumerate(run.leaderboard, start=1):
        lines.append(
            f"| {rank} | `{m.model}` | {m.mean_score:.1%} | {m.pass_rate:.1%} | "
            f"{m.mean_latency_ms:.0f} | {m.p95_latency_ms:.0f} | "
            f"{m.total_cost_usd:.4f} | {m.error_rate:.0%} |"
        )
    lines.append("")

    for dataset in run.datasets:
        cells = [c for c in run.cells if c.dataset == dataset]
        if not cells:
            continue
        cells.sort(key=lambda c: -c.mean_score)
        lines.append(f"## Dataset: {dataset}")
        lines.append("")
        lines.append("| Model | Score | Pass | Mean ms | Cost $ |")
        lines.append("|-------|------:|-----:|--------:|-------:|")
        for c in cells:
            lines.append(
                f"| `{c.model}` | {c.mean_score:.1%} | {c.pass_rate:.1%} | "
                f"{c.mean_latency_ms:.0f} | {c.total_cost_usd:.4f} |"
            )
        lines.append("")

    text = "\n".join(lines)
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def to_html(run: RunResult, path: str | Path) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")
    data = run_to_dict(run)
    # Prevent a "</script>" inside any model output from closing the data tag.
    safe_json = json.dumps(data).replace("</", "<\\/")
    html = template.render(
        run=data,
        run_json=safe_json,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def generate_reports(
    run: RunResult, formats: list[str], output_dir: str | Path, stem: str = "report"
) -> dict[str, Path]:
    """Write every requested format and return {format: path}."""

    out_dir = Path(output_dir)
    written: dict[str, Path] = {}
    if "terminal" in formats:
        to_terminal(run)
    if "json" in formats:
        written["json"] = to_json(run, out_dir / f"{stem}.json")
    if "markdown" in formats:
        written["markdown"] = Path(out_dir / f"{stem}.md")
        to_markdown(run, written["markdown"])
    if "html" in formats:
        written["html"] = to_html(run, out_dir / f"{stem}.html")
    return written
