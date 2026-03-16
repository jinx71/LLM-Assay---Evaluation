"""Run configuration: dataclasses + YAML loader.

A config file declares which models to run, which datasets to score (and
with which scorers), an optional judge model, and runtime/report settings.
See ``config.example.yaml`` for an annotated reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from assay.pricing import DEFAULT_PRICING, Price


@dataclass
class ModelConfig:
    spec: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    path: str
    name: str
    scorers: list[dict[str, Any]]
    system: str | None = None


@dataclass
class JudgeConfig:
    spec: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    concurrency: int = 4
    repeats: int = 1
    cache: bool = True
    cache_dir: str = ".assay_cache"
    timeout_s: float = 60.0
    max_retries: int = 3
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class ReportConfig:
    formats: list[str] = field(default_factory=lambda: ["terminal", "json", "markdown", "html"])
    output_dir: str = "reports"


@dataclass
class Config:
    models: list[ModelConfig]
    datasets: list[DatasetConfig]
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    run: RunConfig = field(default_factory=RunConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    pricing: dict[str, Price] = field(default_factory=lambda: dict(DEFAULT_PRICING))


def _parse_models(raw: list[Any]) -> list[ModelConfig]:
    models: list[ModelConfig] = []
    for entry in raw:
        if isinstance(entry, str):
            models.append(ModelConfig(spec=entry))
        elif isinstance(entry, dict):
            spec = entry.get("id") or entry.get("spec")
            if not spec:
                raise ValueError("each model entry needs an 'id' (the backend:model spec)")
            models.append(ModelConfig(spec=spec, options=entry.get("options", {})))
        else:
            raise ValueError(f"invalid model entry: {entry!r}")
    return models


def _parse_datasets(raw: list[Any]) -> list[DatasetConfig]:
    datasets: list[DatasetConfig] = []
    for entry in raw:
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError("each dataset entry must be a mapping with a 'path'")
        path = entry["path"]
        name = entry.get("name") or Path(path).stem
        scorers = entry.get("scorers")
        if not scorers:
            raise ValueError(f"dataset {name!r} must declare at least one scorer")
        datasets.append(
            DatasetConfig(
                path=path,
                name=name,
                scorers=scorers,
                system=entry.get("system"),
            )
        )
    return datasets


def _parse_pricing(raw: dict[str, Any] | None) -> dict[str, Price]:
    table = dict(DEFAULT_PRICING)
    for spec, value in (raw or {}).items():
        table[spec] = Price(
            input_per_mtok=float(value["input_per_mtok"]),
            output_per_mtok=float(value["output_per_mtok"]),
        )
    return table


def parse_config(data: dict[str, Any]) -> Config:
    if "models" not in data or "datasets" not in data:
        raise ValueError("config must define 'models' and 'datasets'")

    run_raw = data.get("run", {})
    report_raw = data.get("report", {})
    judge_raw = data.get("judge", {})

    return Config(
        models=_parse_models(data["models"]),
        datasets=_parse_datasets(data["datasets"]),
        judge=JudgeConfig(
            spec=judge_raw.get("id") or judge_raw.get("spec"),
            options=judge_raw.get("options", {}),
        ),
        run=RunConfig(
            concurrency=int(run_raw.get("concurrency", 4)),
            repeats=int(run_raw.get("repeats", 1)),
            cache=bool(run_raw.get("cache", True)),
            cache_dir=run_raw.get("cache_dir", ".assay_cache"),
            timeout_s=float(run_raw.get("timeout_s", 60.0)),
            max_retries=int(run_raw.get("max_retries", 3)),
            max_tokens=int(run_raw.get("max_tokens", 512)),
            temperature=float(run_raw.get("temperature", 0.0)),
        ),
        report=ReportConfig(
            formats=report_raw.get("formats", ["terminal", "json", "markdown", "html"]),
            output_dir=report_raw.get("output_dir", "reports"),
        ),
        pricing=_parse_pricing(data.get("pricing")),
    )


def load_config(path: str | Path) -> Config:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"config not found: {file_path}")
    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{file_path}: config root must be a mapping")
    return parse_config(data)
