"""Tests for dataset loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from assay.dataset import load_dataset


def test_loads_cases(classification_dataset: Path):
    cases = load_dataset(classification_dataset)
    assert len(cases) == 3
    assert cases[0].id == "c1"
    assert cases[0].expected == "Critical"
    assert cases[0].task == "classification"


def test_auto_id_and_comments(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        '# a comment line\n'
        '\n'
        '{"input": "no id here", "expected": "x"}\n',
        encoding="utf-8",
    )
    cases = load_dataset(path)
    assert len(cases) == 1
    assert cases[0].id == "d-3"  # line number used for the auto id


def test_missing_input_raises(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"expected": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'input'"):
        load_dataset(path)


def test_invalid_json_raises(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_dataset(path)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("/no/such/file.jsonl")
