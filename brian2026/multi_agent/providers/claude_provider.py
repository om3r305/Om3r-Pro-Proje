"""Claude through the official ``anthropic`` Python SDK (optional dependency, imported lazily).

The key is read by the SDK from ``ANTHROPIC_API_KEY``; this module only checks that it is set.
Structured output uses ``output_config.format`` (JSON schema); the answer is validated again here.
Not exercised against the live API in this change: the default budget is zero.
"""
from __future__ import annotations

import importlib
import os
from typing import Any

from ..schemas import validate_assessment_payload
from .base import GuardedProvider, ProviderRefused, ProviderRequest, ProviderResponse, parse_json_payload


class ClaudeProvider(GuardedProvider):
    name = "claude"
    paid = True

    def __init__(self, model: str, *, sdk: Any = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._sdk = sdk  # injectable for tests; the real SDK is imported on first use
        self._client: Any = None

    def available(self) -> bool:
        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("anthropic")
            except ImportError:
                return False
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    def _invoke(self, request: ProviderRequest) -> ProviderResponse:
        if self._client is None:
            # Retries stay inside the per-request budget reservation: at most one retry.
            self._client = self._sdk.Anthropic(timeout=self.timeout_seconds, max_retries=1)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=request.max_output_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": dict(request.json_schema)}},
        )
        usage = getattr(message, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if getattr(message, "stop_reason", None) == "refusal":
            raise ProviderRefused("MODEL_REFUSED")
        if getattr(message, "stop_reason", None) == "max_tokens":
            raise ProviderRefused("TRUNCATED_OUTPUT")
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        payload = parse_json_payload(text)
        if validate_assessment_payload(payload):
            raise ProviderRefused("SCHEMA_INVALID")
        return ProviderResponse(self.name, self.model, "OK", payload,
                                input_tokens=input_tokens, output_tokens=output_tokens)
