"""On-disk response cache.

LLM calls are slow and cost money, so identical ``(model, request, repeat)``
calls are cached as one JSON file per key. Re-running an eval after tweaking
only the report or the scoring config is then nearly instant and free.

Errors are never cached (a transient failure should be retried next run).
The cache directory is gitignored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from assay.models import CompletionRequest, Prediction


class Cache:
    def __init__(self, enabled: bool = True, directory: str | Path = ".assay_cache") -> None:
        self.enabled = enabled
        self.directory = Path(directory)
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, model: str, request: CompletionRequest, repeat: int) -> str:
        payload = json.dumps(
            {
                "model": model,
                "prompt": request.prompt,
                "system": request.system,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "repeat": repeat,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Prediction | None:
        if not self.enabled:
            return None
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Prediction.from_dict(data)

    def set(self, key: str, prediction: Prediction) -> None:
        # Never persist failures.
        if not self.enabled or not prediction.ok:
            return
        path = self.directory / f"{key}.json"
        try:
            path.write_text(json.dumps(prediction.to_dict()), encoding="utf-8")
        except OSError:
            pass  # cache is a best-effort optimisation, never fatal
