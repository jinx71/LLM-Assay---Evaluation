"""Command-line interface for the Assay harness.

    assay run --config config.yaml      run an evaluation and write reports
    assay providers                     list available model backends
    assay scorers                       list available scorers
    assay datasets <path.jsonl>         inspect a dataset
    assay version                       print the version
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

from assay import __version__
from assay.config import load_config
from assay.dataset import load_dataset
from assay.providers.registry import available_backends
from assay.report import generate_reports
from assay.runner import run_eval
from assay.scorers.registry import available_scorers

console = Console()


@click.group()
@click.version_option(__version__, prog_name="assay")
def main() -> None:
    """Assay — measure LLM quality, latency, and cost on domain tasks."""


@main.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default="config.yaml",
    show_default=True,
    help="Path to the YAML run config.",
)
@click.option(
    "--models", default=None, help="Comma-separated backend:model specs to override the config."
)
@click.option("--limit", type=int, default=None, help="Max test cases per dataset (smoke runs).")
@click.option("--no-cache", is_flag=True, help="Ignore and bypass the response cache.")
@click.option(
    "--format", "fmt", default=None, help="Report formats: terminal,json,markdown,html."
)
@click.option("--output-dir", default=None, help="Where to write report files.")
@click.option("--stem", default="report", show_default=True, help="Base filename for report files.")
def run_command(
    config_path: str,
    models: str | None,
    limit: int | None,
    no_cache: bool,
    fmt: str | None,
    output_dir: str | None,
    stem: str,
) -> None:
    """Run an evaluation defined by a config file."""

    config = load_config(config_path)

    if models:
        from assay.config import ModelConfig

        config.models = [ModelConfig(spec=s.strip()) for s in models.split(",") if s.strip()]
    if no_cache:
        config.run.cache = False
    if fmt:
        config.report.formats = [f.strip() for f in fmt.split(",") if f.strip()]
    if output_dir:
        config.report.output_dir = output_dir

    total_cases = 0
    for ds in config.datasets:
        cases = load_dataset(ds.path)
        total_cases += min(len(cases), limit) if limit is not None else len(cases)
    total_evals = total_cases * len(config.models) * config.run.repeats

    console.print(
        f"[bold]Running[/bold] {len(config.models)} model(s) × {total_cases} case(s) "
        f"× {config.run.repeats} repeat(s) = [amber]{total_evals}[/amber] evaluations"
    )

    with console.status("[bold]Evaluating…", spinner="dots") as status:
        done = {"n": 0}

        def tick() -> None:
            done["n"] += 1
            status.update(f"[bold]Evaluating…[/bold] {done['n']}/{total_evals}")

        run = asyncio.run(run_eval(config, limit=limit, progress=tick))

    console.print()
    written = generate_reports(run, config.report.formats, config.report.output_dir, stem=stem)
    for fmt_name, path in written.items():
        console.print(f"[green]✓[/green] wrote {fmt_name}: [bold]{path}[/bold]")


@main.command("providers")
def providers_command() -> None:
    """List available model backends."""
    console.print("[bold]Available backends:[/bold]")
    for backend in available_backends():
        console.print(f"  • {backend}")
    console.print("\nUse them as [dim]backend:model[/dim], e.g. anthropic:claude-3-5-sonnet-latest")


@main.command("scorers")
def scorers_command() -> None:
    """List available scorers."""
    console.print("[bold]Available scorers:[/bold]")
    for scorer in available_scorers():
        console.print(f"  • {scorer}")


@main.command("datasets")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def datasets_command(path: str) -> None:
    """Inspect a JSONL dataset."""
    cases = load_dataset(path)
    tasks: dict[str, int] = {}
    for case in cases:
        tasks[case.task] = tasks.get(case.task, 0) + 1
    console.print(f"[bold]{Path(path).name}[/bold]: {len(cases)} cases")
    for task, count in sorted(tasks.items()):
        console.print(f"  • task '{task}': {count}")
    console.print(f"\n[dim]first case:[/dim] {cases[0].input[:140]}")


@main.command("version")
def version_command() -> None:
    """Print the version."""
    console.print(f"assay {__version__}")


if __name__ == "__main__":
    main()
