"""OpenAI GPT backend (Chat Completions API)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from assay.models import CompletionRequest, Prediction
from assay.providers.base import LLMProvider, ProviderError

ENDPOINT = "https://api.openai.com/v1/chat/completions"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAIProvider(LLMProvider):
    """GPT models via the OpenAI Chat Completions API."""

    backend = "openai"

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = options.get("base_url", ENDPOINT)
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def complete(self, request: CompletionRequest) -> Prediction:
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set", retryable=False)

        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = await self._http().post(self.base_url, json=body, headers=headers)
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
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            return Prediction(
                output=text,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                raw=data,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}", retryable=False) from exc

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
