"""Anthropic Claude backend (direct Messages API).

We call the REST endpoint directly with httpx rather than wrapping the SDK.
That keeps the dependency surface tiny and makes the exact request/response
contract visible — useful both for cost accounting (we read the real
``usage`` block) and for explaining the integration in interviews.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from assay.models import CompletionRequest, Prediction
from assay.providers.base import LLMProvider, ProviderError

ENDPOINT = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class AnthropicProvider(LLMProvider):
    """Claude models via the Anthropic Messages API."""

    backend = "anthropic"

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = options.get("base_url", ENDPOINT)
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def complete(self, request: CompletionRequest) -> Prediction:
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set", retryable=False)

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            body["system"] = request.system

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
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
            blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            usage = data.get("usage", {})
            return Prediction(
                output=text,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                raw=data,
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}", retryable=False) from exc

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
