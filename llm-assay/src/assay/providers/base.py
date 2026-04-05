"""Abstract provider interface.

Every model backend implements :class:`LLMProvider`. The runner only ever
talks to this interface, which is what makes the harness model-agnostic:
adding a new vendor means writing one class, not touching the pipeline.
"""

from __future__ import annotations

import abc

from assay.models import CompletionRequest, Prediction


class ProviderError(Exception):
    """Raised by providers on a failed call.

    ``retryable`` tells the runner whether a backoff-and-retry is worth
    attempting (rate limits, timeouts, transient 5xx) versus a hard failure
    (bad request, auth error) that should fail fast.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(abc.ABC):
    """Common interface for a single model backend.

    A provider is identified by an id of the form ``backend:model`` (for
    example ``anthropic:claude-3-5-sonnet-latest``).
    """

    backend: str = "base"

    def __init__(self, model: str, **options: object) -> None:
        self.model = model
        self.options = options

    @property
    def id(self) -> str:
        return f"{self.backend}:{self.model}"

    @abc.abstractmethod
    async def complete(self, request: CompletionRequest) -> Prediction:
        """Run a single completion and return a normalised :class:`Prediction`.

        Implementations must raise :class:`ProviderError` on failure rather
        than returning an error Prediction — the runner owns retry and
        timing so latency is measured consistently across backends.
        """

    async def aclose(self) -> None:
        """Release any held resources (HTTP clients, etc.)."""
        return None
