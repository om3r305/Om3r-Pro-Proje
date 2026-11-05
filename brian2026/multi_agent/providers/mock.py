"""Free, deterministic stand-in provider for tests and dry runs. Its output is marked ``mock``."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from .base import CostLedger, GuardedProvider, ProviderRequest, ProviderResponse


class MockProvider(GuardedProvider):
    paid = False

    def __init__(self, name: str, responder: Callable[[ProviderRequest], Mapping[str, Any] | Exception],
                 *, ledger: CostLedger | None = None, model: str = "mock-model") -> None:
        super().__init__(model, ledger=ledger or CostLedger(0.0), min_interval_seconds=0.0)
        self.name = name
        self._responder = responder
        self.requests: list[ProviderRequest] = []

    def _invoke(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        result = self._responder(request)
        if isinstance(result, Exception):
            raise result
        return ProviderResponse(self.name, self.model, "OK", dict(result), input_tokens=0, output_tokens=0, mock=True)
