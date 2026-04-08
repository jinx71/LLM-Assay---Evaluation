"""Google Gemini backend (Generative Language API)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from assay.models import CompletionRequest, Prediction
from assay.providers.base import LLMProvider, ProviderError

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class GoogleProvider(LLMProvider):
    """Gemini models via the Generative Language API."""

    backend = "google"

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.base = options.get("base_url", BASE)
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def complete(self, request: CompletionRequest) -> Prediction:
        if not self.api_key:
            raise ProviderError("GOOGLE_API_KEY is not set", retryable=False)

        url = f"{self.base}/{self.model}:generateContent?key={self.api_key}"
        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if request.system:
            body["systemInstruction"] = {"parts": [{"text": request.system}]}

        try:
            resp = await self._http().post(url, json=body)
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
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            usage = data.get("usageMetadata", {})
            return Prediction(
                output=text,
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
                raw=data,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}", retryable=False) from exc

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
