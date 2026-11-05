"""Grok through the official ``xai-sdk`` Python package (optional dependency, imported lazily).

X Search is enabled only when the request asks for it, with ``to_date`` set to the decision day so the
search cannot return later posts; the date filter is day-granular, so the data verifier still drops
any cited post published after the decision time. The key is read from ``XAI_API_KEY``.
Not exercised against the live API in this change: the default budget is zero.
"""
from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from ..schemas import validate_assessment_payload
from .base import GuardedProvider, ProviderRefused, ProviderRequest, ProviderResponse, parse_json_payload


class GrokProvider(GuardedProvider):
    name = "grok"
    paid = True

    def __init__(self, model: str, *, sdk: Any = None, chat_helpers: Any = None, tools_module: Any = None,
                 **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._sdk, self._chat, self._tools = sdk, chat_helpers, tools_module
        self._client: Any = None

    def available(self) -> bool:
        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("xai_sdk")
                self._chat = importlib.import_module("xai_sdk.chat")
                self._tools = importlib.import_module("xai_sdk.tools")
            except ImportError:
                return False
        return bool(os.environ.get("XAI_API_KEY", "").strip())

    def _invoke(self, request: ProviderRequest) -> ProviderResponse:
        if self._client is None:
            self._client = self._sdk.Client(timeout=self.timeout_seconds)
        tools = []
        if request.x_search:
            if not request.x_search_to_date:
                raise ProviderRefused("X_SEARCH_WITHOUT_CUTOFF")
            to_date = datetime.strptime(request.x_search_to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            tools.append(self._tools.x_search(to_date=to_date))
        options: dict[str, Any] = {"model": self.model, "max_tokens": request.max_output_tokens}
        if tools:
            options["tools"] = tools
        # The answer format is enforced by the prompt and validated below, not trusted.
        chat = self._client.chat.create(**options)
        chat.append(self._chat.system(request.system))
        chat.append(self._chat.user(request.prompt + "\n\nAnswer only with JSON matching this schema:\n"
                                    + json.dumps(request.json_schema)))
        response = chat.sample()
        usage = getattr(response, "usage", None)
        payload = parse_json_payload(getattr(response, "content", "") or "")
        if validate_assessment_payload(payload):
            raise ProviderRefused("SCHEMA_INVALID")
        citations = tuple(str(c) for c in (getattr(response, "citations", None) or ()))[:50]
        return ProviderResponse(
            self.name, self.model, "OK", payload, citations=citations,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            reported_cost_usd=_finite_cost(getattr(response, "cost_usd", None)),
        )


def _finite_cost(value: Any) -> float | None:
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if cost >= 0 and cost == cost and cost != float("inf") else None
