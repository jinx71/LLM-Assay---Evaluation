"""Hugging Face Inference API backend.

Covers open-weight models (Llama, Mistral, Qwen, ...) through the hosted
Inference API, so the harness can compare proprietary and open models side
by side — exactly the multi-LLM coverage a benchmark needs.

The text-generation endpoint does not return token usage, so token counts
are estimated (chars / 4). That approximation is flagged in reports and only
affects open models, whose cost is typically tracked per-hour rather than
per-token anyway.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from assay.models import CompletionRequest, Prediction
from assay.providers.base import LLMProvider, ProviderError

BASE = "https://api-inference.huggingface.co/models"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class HuggingFaceProvider(LLMProvider):
    """Open-weight models via the Hugging Face Inference API."""

    backend = "huggingface"

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.api_key = os.environ.get("HUGGINGFACE_API_KEY")
        self.base = options.get("base_url", BASE)
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
        return self._client

    async def complete(self, request: CompletionRequest) -> Prediction:
        if not self.api_key:
            raise ProviderError("HUGGINGFACE_API_KEY is not set", retryable=False)

        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\n{request.prompt}"

        url = f"{self.base}/{self.model}"
        body: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "temperature": max(request.temperature, 0.01),
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            resp = await self._http().post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"request timed out: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"transport error: {exc}", retryable=True) from exc

        if resp.status_code >= 400:
            retryable = resp.status_code in RETRYABLE_STATUS
            raise ProviderError(
                f"HTTP {resp.status_code}: {resp.text[:300]}", retryable=retryable
            )

        data = resp.json()
        try:
            if isinstance(data, list):
                text = data[0].get("generated_text", "")
            else:
                text = data.get("generated_text", "")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}", retryable=False) from exc

        return Prediction(
            output=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            raw={"estimated_tokens": True},
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
