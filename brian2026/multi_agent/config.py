"""Environment configuration for the BRIAN multi-agent layer.

Every default is the safe one: the layer is off, shadow only, real trading off and the paid-call
budget is zero (no provider call can be made). Values that would weaken a guarantee are clamped or
refused here instead of being trusted downstream.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

MAX_DEBATE_ROUNDS_CAP = 2
MAX_PROVIDER_CALLS_CAP = 6
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_GROK_MODEL = "grok-4.7"


class UnsafeConfigError(ValueError):
    """Raised when the environment asks for something this layer must never do."""


def _flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise UnsafeConfigError(f"{key} must be a boolean, got an unrecognised value")


def _int(env: Mapping[str, str], key: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise UnsafeConfigError(f"{key} must be an integer") from exc
    return max(lo, min(hi, value))


def _money(env: Mapping[str, str], key: str) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return 0.0
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise UnsafeConfigError(f"{key} must be a number") from exc
    if value != value or value < 0 or value == float("inf"):
        raise UnsafeConfigError(f"{key} must be a finite non-negative number")
    return value


@dataclass(frozen=True, slots=True)
class MultiAgentConfig:
    enabled: bool = False
    shadow_only: bool = True
    real_trading_enabled: bool = False
    operator_stop: bool = False
    claude_model: str = DEFAULT_CLAUDE_MODEL
    grok_model: str = DEFAULT_GROK_MODEL
    max_debate_rounds: int = MAX_DEBATE_ROUNDS_CAP
    max_provider_calls_per_candidate: int = MAX_PROVIDER_CALLS_CAP
    cost_limit_usd: float = 0.0
    provider_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.shadow_only or self.real_trading_enabled:
            raise UnsafeConfigError("the multi-agent layer is shadow only; real trading cannot be enabled here")
        if not 0 <= self.max_debate_rounds <= MAX_DEBATE_ROUNDS_CAP:
            raise UnsafeConfigError("max_debate_rounds out of range")
        if not 0 <= self.max_provider_calls_per_candidate <= MAX_PROVIDER_CALLS_CAP:
            raise UnsafeConfigError("max_provider_calls_per_candidate out of range")
        if self.cost_limit_usd < 0:
            raise UnsafeConfigError("cost_limit_usd must be non-negative")

    @property
    def paid_calls_allowed(self) -> bool:
        """Paid provider calls need the layer on, no operator stop and a positive budget."""
        return self.enabled and not self.operator_stop and self.cost_limit_usd > 0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "MultiAgentConfig":
        env = os.environ if env is None else env
        if not _flag(env, "BRIAN_MULTI_AGENT_SHADOW_ONLY", True):
            raise UnsafeConfigError("BRIAN_MULTI_AGENT_SHADOW_ONLY=false is not supported")
        if _flag(env, "BRIAN_REAL_TRADING_ENABLED", False):
            raise UnsafeConfigError("BRIAN_REAL_TRADING_ENABLED=true is not supported by this layer")
        return cls(
            enabled=_flag(env, "BRIAN_MULTI_AGENT_ENABLED", False),
            operator_stop=_flag(env, "BRIAN_MULTI_AGENT_STOP", False),
            claude_model=(env.get("BRIAN_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip(),
            grok_model=(env.get("BRIAN_GROK_MODEL") or DEFAULT_GROK_MODEL).strip(),
            max_debate_rounds=_int(env, "BRIAN_MAX_DEBATE_ROUNDS", MAX_DEBATE_ROUNDS_CAP, 0, MAX_DEBATE_ROUNDS_CAP),
            max_provider_calls_per_candidate=_int(
                env, "BRIAN_MAX_PROVIDER_CALLS_PER_CANDIDATE", MAX_PROVIDER_CALLS_CAP, 0, MAX_PROVIDER_CALLS_CAP
            ),
            cost_limit_usd=_money(env, "BRIAN_AGENT_COST_LIMIT_USD"),
        )
