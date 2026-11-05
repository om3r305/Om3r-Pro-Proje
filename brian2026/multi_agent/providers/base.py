"""Common provider interface: budget, call cap, cache, rate limit, usage and sanitised errors.

``GuardedProvider.call`` is the only way the layer reaches a model. Before any paid request it
reserves the worst-case cost of that request against the run budget; a request that could exceed the
budget, a model without a known price, a missing key or an exhausted per-candidate call cap is refused
without touching the network. Errors are reduced to a fixed class name so that keys, prompts and raw
provider output never reach logs or stored records.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

# USD per million tokens (input, output). Sources: Anthropic model table (2026-10-06) and
# docs.x.ai/docs/models (read 2026-11-05). A model missing here cannot be called: its cost is unknown.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "grok-4.7": (2.00, 6.00),
    "grok-4.6": (2.00, 6.00),
    "grok-4.3": (1.25, 2.50),
    "grok-4.20-0309-non-reasoning": (1.25, 2.50),
}
# X Search bills $5 per 1,000 posts fetched (docs.x.ai, read 2026-11-05); reserved per search request.
X_SEARCH_USD_PER_POST = 0.005
X_SEARCH_RESERVED_POSTS = 100
CHARS_PER_TOKEN_ESTIMATE = 3.0  # deliberately low, so the input estimate errs high


class ProviderRefused(Exception):
    """The guard refused the request before any network call."""

    def __init__(self, error_class: str):
        super().__init__(error_class)
        self.error_class = error_class


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    agent: str
    system: str
    prompt: str
    json_schema: Mapping[str, Any]
    max_output_tokens: int = 2_000
    x_search: bool = False
    x_search_to_date: str | None = None  # YYYY-MM-DD, exclusive upper bound (point-in-time cut)

    def cache_key(self, provider: str, model: str) -> str:
        blob = json.dumps(
            [provider, model, self.system, self.prompt, self.json_schema, self.max_output_tokens,
             self.x_search, self.x_search_to_date],
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    provider: str
    model: str
    status: str  # OK | FAILED | SKIPPED
    payload: Mapping[str, Any] | None
    citations: tuple[str, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float = 0.0
    reported_cost_usd: float | None = None
    latency_ms: int | None = None
    error_class: str | None = None
    cached: bool = False
    mock: bool = False


def worst_case_cost_usd(model: str, request: ProviderRequest) -> float:
    price = MODEL_PRICES_USD_PER_MTOK.get(model)
    if price is None:
        raise ProviderRefused("UNKNOWN_MODEL_PRICE")
    chars = len(request.system) + len(request.prompt) + len(json.dumps(request.json_schema))
    input_tokens = chars / CHARS_PER_TOKEN_ESTIMATE + 500  # message framing / schema overhead
    cost = input_tokens * price[0] / 1e6 + request.max_output_tokens * price[1] / 1e6
    if request.x_search:
        cost += X_SEARCH_RESERVED_POSTS * X_SEARCH_USD_PER_POST
    return cost


def actual_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    price = MODEL_PRICES_USD_PER_MTOK.get(model)
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return input_tokens * price[0] / 1e6 + output_tokens * price[1] / 1e6


@dataclass
class CostLedger:
    """Run budget. Spend is reserved before a call and settled after it; never goes above the limit."""
    limit_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self, amount: float) -> None:
        with self._lock:
            if amount < 0 or self.spent_usd + self.reserved_usd + amount > self.limit_usd + 1e-12:
                raise ProviderRefused("BUDGET_EXCEEDED")
            self.reserved_usd += amount

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reserved)
            # When the provider reports nothing, the reservation is the spend we must assume.
            self.spent_usd += max(0.0, actual)


@dataclass
class CallCounter:
    limit: int
    used: int = 0

    def take(self) -> None:
        if self.used >= self.limit:
            raise ProviderRefused("CALL_CAP_REACHED")
        self.used += 1


class GuardedProvider:
    """Base class. Subclasses implement ``_invoke`` and set ``name``/``model``/``paid``."""

    name = "base"
    paid = True

    def __init__(self, model: str, *, ledger: CostLedger, min_interval_seconds: float = 1.0,
                 timeout_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.model = model
        self.ledger = ledger
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_call: float | None = None
        self._cache: dict[str, ProviderResponse] = {}

    # --- subclass hooks -------------------------------------------------------------------------
    def available(self) -> bool:
        """True when the SDK is importable and a key is configured (never reads the key out)."""
        return True

    def _invoke(self, request: ProviderRequest) -> ProviderResponse:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- guarded entry point --------------------------------------------------------------------
    def call(self, request: ProviderRequest, *, calls: CallCounter, allow_paid: bool) -> ProviderResponse:
        key = request.cache_key(self.name, self.model)
        cached = self._cache.get(key)
        if cached is not None:
            return ProviderResponse(**{**_fields(cached), "cached": True, "estimated_cost_usd": 0.0,
                                       "reported_cost_usd": None, "latency_ms": 0})
        reserved = 0.0
        try:
            if self.paid and not allow_paid:
                raise ProviderRefused("PAID_CALLS_DISABLED")
            if not self.available():
                raise ProviderRefused("PROVIDER_UNAVAILABLE")
            if self.paid:
                reserved = worst_case_cost_usd(self.model, request)
            calls.take()
            self.ledger.reserve(reserved)
        except ProviderRefused as refused:
            return ProviderResponse(self.name, self.model, "SKIPPED", None, error_class=refused.error_class)

        self._respect_rate_limit()
        started = self._clock()
        try:
            response = self._invoke(request)
        except ProviderRefused as refused:
            response = ProviderResponse(self.name, self.model, "FAILED", None, error_class=refused.error_class)
        except Exception as exc:  # noqa: BLE001 - provider errors are reduced to a class name only
            response = ProviderResponse(self.name, self.model, "FAILED", None, error_class=_error_class(exc))
        latency_ms = int((self._clock() - started) * 1000)

        actual = response.reported_cost_usd
        if actual is None:
            actual = actual_cost_usd(self.model, response.input_tokens, response.output_tokens)
        if actual is None:
            actual = reserved if self.paid else 0.0
        self.ledger.settle(reserved, actual)
        response = ProviderResponse(**{**_fields(response), "estimated_cost_usd": actual, "latency_ms": latency_ms})
        if response.status == "OK":
            self._cache[key] = response
        return response

    def _respect_rate_limit(self) -> None:
        now = self._clock()
        if self._last_call is not None:
            wait = self.min_interval_seconds - (now - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()


def _fields(response: ProviderResponse) -> dict[str, Any]:
    return {name: getattr(response, name) for name in ProviderResponse.__dataclass_fields__}


def _error_class(exc: BaseException) -> str:
    """Only the exception type name: messages can echo prompts, headers or keys."""
    name = type(exc).__name__
    if "Timeout" in name:
        return "TIMEOUT"
    if "RateLimit" in name:
        return "RATE_LIMITED"
    if "Authentication" in name or "PermissionDenied" in name:
        return "AUTH_FAILED"
    if "Connection" in name:
        return "CONNECTION_FAILED"
    return f"PROVIDER_ERROR:{name}"[:80]


def parse_json_payload(text: str) -> Mapping[str, Any]:
    """Parse a model's JSON answer; raises ProviderRefused('INVALID_JSON') on anything else."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ProviderRefused("INVALID_JSON") from exc
    if not isinstance(payload, dict):
        raise ProviderRefused("INVALID_JSON")
    return payload
