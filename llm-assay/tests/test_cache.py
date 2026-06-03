"""Tests for the on-disk response cache."""

from __future__ import annotations

from pathlib import Path

from assay.cache import Cache
from assay.models import CompletionRequest, Prediction


def _req():
    return CompletionRequest(prompt="hello", system="sys", max_tokens=64, temperature=0.0)


def test_roundtrip(tmp_path: Path):
    cache = Cache(enabled=True, directory=tmp_path / "c")
    key = cache.key("mock:smart", _req(), 0)
    assert cache.get(key) is None
    cache.set(key, Prediction(output="cached!", input_tokens=2, output_tokens=3))
    got = cache.get(key)
    assert got is not None and got.output == "cached!"
    assert got.input_tokens == 2 and got.output_tokens == 3


def test_errors_are_not_cached(tmp_path: Path):
    cache = Cache(enabled=True, directory=tmp_path / "c")
    key = cache.key("mock:smart", _req(), 0)
    cache.set(key, Prediction(output="", error="boom"))
    assert cache.get(key) is None


def test_key_changes_with_repeat(tmp_path: Path):
    cache = Cache(enabled=True, directory=tmp_path / "c")
    assert cache.key("m", _req(), 0) != cache.key("m", _req(), 1)


def test_key_changes_with_model(tmp_path: Path):
    cache = Cache(enabled=True, directory=tmp_path / "c")
    assert cache.key("model-a", _req(), 0) != cache.key("model-b", _req(), 0)


def test_disabled_cache_is_noop(tmp_path: Path):
    cache = Cache(enabled=False, directory=tmp_path / "c")
    key = cache.key("m", _req(), 0)
    cache.set(key, Prediction(output="x"))
    assert cache.get(key) is None
