from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

from .evidence_ledger import content_hash
from .phase56_pretrade_risk_engine import (
    InstrumentRiskLimits,
    PreTradeRiskPolicy,
    TradingState,
)

PHASE68_SCHEMA_VERSION = "brian.phase68-operational-risk-governor.v1"
ExitReason = Literal[
    "TARGET",
    "STOP_LOSS",
    "TRAILING_STOP_LOSS",
    "STOPLOSS_ON_EXCHANGE",
    "LIQUIDATION",
    "MANUAL",
    "OTHER",
]
HealthKind = Literal[
    "EXECUTION_SUCCESS",
    "EXECUTION_FAILURE",
    "RECONCILIATION_SUCCESS",
    "RECONCILIATION_FAILURE",
    "UNKNOWN_ORDER_OUTCOME",
]

_STOPLOSS_REASONS = frozenset({
    "STOP_LOSS",
    "TRAILING_STOP_LOSS",
    "STOPLOSS_ON_EXCHANGE",
    "LIQUIDATION",
})


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: float
    equity_usd: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("equity timestamp must be finite")
        if not math.isfinite(self.equity_usd) or self.equity_usd <= 0:
            raise ValueError("equity_usd must be positive")


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    asset_id: str
    closed_at: float
    pnl_quote: float
    return_fraction: float
    exit_reason: ExitReason

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("closed trade asset_id is required")
        if not all(math.isfinite(value) for value in (
            self.closed_at,
            self.pnl_quote,
            self.return_fraction,
        )):
            raise ValueError("closed trade values must be finite")


@dataclass(frozen=True, slots=True)
class RuntimeHealthEvent:
    timestamp: float
    kind: HealthKind
    asset_id: str | None = None
    reference_id: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("runtime health timestamp must be finite")
        if self.asset_id is not None and not self.asset_id.strip():
            raise ValueError("health asset_id cannot be blank")


@dataclass(frozen=True, slots=True)
class OperationalRiskPolicy:
    drawdown_lookback_seconds: int = 86_400
    max_drawdown_fraction: float = 0.10
    daily_loss_lookback_seconds: int = 86_400
    max_daily_loss_fraction: float = 0.07
    stoploss_lookback_seconds: int = 3_600
    stoploss_limit: int = 4
    stoploss_required_profit: float = 0.0
    stoploss_lock_seconds: int = 1_800
    asset_cooldown_seconds: int = 300
    execution_failure_lookback_seconds: int = 900
    max_consecutive_execution_failures: int = 3
    reconciliation_failure_lookback_seconds: int = 900
    max_reconciliation_failures: int = 2
    unknown_outcome_lookback_seconds: int = 3_600
    max_unknown_order_outcomes: int = 1
    max_market_data_age_seconds: float = 30.0

    def __post_init__(self) -> None:
        integer_positive = (
            self.drawdown_lookback_seconds,
            self.daily_loss_lookback_seconds,
            self.stoploss_lookback_seconds,
            self.stoploss_limit,
            self.stoploss_lock_seconds,
            self.execution_failure_lookback_seconds,
            self.max_consecutive_execution_failures,
            self.reconciliation_failure_lookback_seconds,
            self.max_reconciliation_failures,
            self.unknown_outcome_lookback_seconds,
            self.max_unknown_order_outcomes,
        )
        if any(value <= 0 for value in integer_positive):
            raise ValueError("operational-risk windows/limits must be positive")
        if self.asset_cooldown_seconds < 0:
            raise ValueError("asset_cooldown_seconds must be non-negative")
        for label, value in (
            ("max_drawdown_fraction", self.max_drawdown_fraction),
            ("max_daily_loss_fraction", self.max_daily_loss_fraction),
        ):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{label} must be in (0,1)")
        if not math.isfinite(self.stoploss_required_profit):
            raise ValueError("stoploss_required_profit must be finite")
        if not math.isfinite(self.max_market_data_age_seconds) or self.max_market_data_age_seconds <= 0:
            raise ValueError("max_market_data_age_seconds must be positive")


@dataclass(frozen=True, slots=True)
class OperationalRiskReceipt:
    timestamp: float
    previous_state: TradingState
    trading_state: TradingState
    recommended_state: TradingState
    reasons: tuple[str, ...]
    max_drawdown_fraction: float
    window_loss_fraction: float
    qualifying_stoplosses: int
    stoploss_lock_until: float | None
    blocked_assets: tuple[str, ...]
    consecutive_execution_failures: int
    reconciliation_failures: int
    unknown_order_outcomes: int
    market_data_age_seconds: float
    manual_halt: bool
    manual_release_requested: bool
    halt_latched: bool
    receipt_id: str
    schema_version: str = PHASE68_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def asset_open_allowed(self, asset_id: str) -> bool:
        return self.trading_state == "ACTIVE" and asset_id not in self.blocked_assets

    def pretrade_policy(
        self,
        limits: InstrumentRiskLimits = InstrumentRiskLimits(),
    ) -> PreTradeRiskPolicy:
        return PreTradeRiskPolicy(
            trading_state=self.trading_state,
            limits=limits,
        )

    def pretrade_policy_for_asset(
        self,
        asset_id: str,
        limits: InstrumentRiskLimits = InstrumentRiskLimits(),
    ) -> PreTradeRiskPolicy:
        if not asset_id.strip():
            raise ValueError("asset_id is required")
        return PreTradeRiskPolicy(
            trading_state=self.trading_state,
            limits=limits,
            block_new_risk=asset_id in self.blocked_assets,
        )


def _severity(state: TradingState) -> int:
    return {"ACTIVE": 0, "REDUCING": 1, "HALTED": 2}[state]


def _recent[T](
    rows: Sequence[T],
    *,
    now: float,
    lookback_seconds: float,
    timestamp,
) -> tuple[T, ...]:
    lower = now - lookback_seconds
    return tuple(
        row
        for row in rows
        if lower <= float(timestamp(row)) <= now
    )


def _max_drawdown(points: Sequence[EquityPoint]) -> float:
    if len(points) < 2:
        return 0.0
    ordered = sorted(points, key=lambda row: row.timestamp)
    peak = ordered[0].equity_usd
    worst = 0.0
    for row in ordered:
        peak = max(peak, row.equity_usd)
        worst = max(worst, max(0.0, (peak - row.equity_usd) / peak))
    return float(worst)


def _window_loss(points: Sequence[EquityPoint]) -> float:
    if len(points) < 2:
        return 0.0
    ordered = sorted(points, key=lambda row: row.timestamp)
    start = ordered[0].equity_usd
    end = ordered[-1].equity_usd
    return float(max(0.0, (start - end) / start))


def _consecutive_execution_failures(events: Sequence[RuntimeHealthEvent]) -> int:
    count = 0
    for event in sorted(events, key=lambda row: row.timestamp, reverse=True):
        if event.kind == "EXECUTION_FAILURE":
            count += 1
        elif event.kind == "EXECUTION_SUCCESS":
            break
    return count


class OperationalRiskGovernor:
    """Stateful circuit breaker for the paper/shadow execution boundary.

    HALTED is latched and needs explicit manual release after the severe trigger
    clears. REDUCING is allowed to recover automatically once its time-window
    protections and health blockers clear. The governor never opens risk itself;
    it only emits a Phase 56-compatible trading state plus per-asset cooldowns.
    """

    def __init__(
        self,
        policy: OperationalRiskPolicy = OperationalRiskPolicy(),
        *,
        initial_state: TradingState = "ACTIVE",
    ) -> None:
        if initial_state not in ("ACTIVE", "REDUCING", "HALTED"):
            raise ValueError("invalid initial_state")
        self.policy = policy
        self._state: TradingState = initial_state
        self._halt_latched = initial_state == "HALTED"

    @property
    def state(self) -> TradingState:
        return self._state

    def evaluate(
        self,
        *,
        now: float,
        equity_points: Sequence[EquityPoint],
        closed_trades: Sequence[ClosedTrade],
        health_events: Sequence[RuntimeHealthEvent],
        market_data_timestamp: float,
        manual_halt: bool = False,
        manual_release: bool = False,
    ) -> OperationalRiskReceipt:
        if not math.isfinite(now) or not math.isfinite(market_data_timestamp):
            raise ValueError("governor timestamps must be finite")
        if market_data_timestamp > now:
            raise ValueError("market_data_timestamp cannot be in the future")

        policy = self.policy
        previous_state = self._state
        reasons: list[str] = []

        drawdown_points = _recent(
            equity_points,
            now=now,
            lookback_seconds=policy.drawdown_lookback_seconds,
            timestamp=lambda row: row.timestamp,
        )
        daily_points = _recent(
            equity_points,
            now=now,
            lookback_seconds=policy.daily_loss_lookback_seconds,
            timestamp=lambda row: row.timestamp,
        )
        max_drawdown = _max_drawdown(drawdown_points)
        window_loss = _window_loss(daily_points)

        stoploss_trades = tuple(
            trade
            for trade in _recent(
                closed_trades,
                now=now,
                lookback_seconds=policy.stoploss_lookback_seconds,
                timestamp=lambda row: row.closed_at,
            )
            if trade.exit_reason in _STOPLOSS_REASONS
            and trade.return_fraction < policy.stoploss_required_profit
        )
        stoploss_lock_until = (
            max(trade.closed_at for trade in stoploss_trades) + policy.stoploss_lock_seconds
            if len(stoploss_trades) >= policy.stoploss_limit
            else None
        )

        blocked_assets: set[str] = set()
        if policy.asset_cooldown_seconds > 0:
            for trade in closed_trades:
                if trade.closed_at > now:
                    continue
                if now < trade.closed_at + policy.asset_cooldown_seconds:
                    blocked_assets.add(trade.asset_id)

        execution_events = _recent(
            health_events,
            now=now,
            lookback_seconds=policy.execution_failure_lookback_seconds,
            timestamp=lambda row: row.timestamp,
        )
        consecutive_execution_failures = _consecutive_execution_failures(execution_events)

        reconciliation_events = _recent(
            health_events,
            now=now,
            lookback_seconds=policy.reconciliation_failure_lookback_seconds,
            timestamp=lambda row: row.timestamp,
        )
        reconciliation_failures = sum(
            event.kind == "RECONCILIATION_FAILURE"
            for event in reconciliation_events
        )

        unknown_events = _recent(
            health_events,
            now=now,
            lookback_seconds=policy.unknown_outcome_lookback_seconds,
            timestamp=lambda row: row.timestamp,
        )
        unknown_outcomes = sum(
            event.kind == "UNKNOWN_ORDER_OUTCOME"
            for event in unknown_events
        )

        market_data_age = now - market_data_timestamp
        recommended: TradingState = "ACTIVE"

        def escalate(state: TradingState, reason: str) -> None:
            nonlocal recommended
            if _severity(state) > _severity(recommended):
                recommended = state
            reasons.append(reason)

        if manual_halt:
            escalate("HALTED", "manual_halt")
        if max_drawdown > policy.max_drawdown_fraction:
            escalate(
                "HALTED",
                f"max_drawdown:{max_drawdown:.8f}>{policy.max_drawdown_fraction:.8f}",
            )
        if window_loss > policy.max_daily_loss_fraction:
            escalate(
                "HALTED",
                f"window_loss:{window_loss:.8f}>{policy.max_daily_loss_fraction:.8f}",
            )
        if market_data_age > policy.max_market_data_age_seconds:
            escalate(
                "HALTED",
                f"market_data_stale:{market_data_age:.3f}s>{policy.max_market_data_age_seconds:.3f}s",
            )
        if unknown_outcomes >= policy.max_unknown_order_outcomes:
            escalate(
                "HALTED",
                f"unknown_order_outcomes:{unknown_outcomes}",
            )

        if (
            stoploss_lock_until is not None
            and now < stoploss_lock_until
        ):
            escalate(
                "REDUCING",
                f"stoploss_guard:{len(stoploss_trades)}_until_{stoploss_lock_until:.6f}",
            )
        if consecutive_execution_failures >= policy.max_consecutive_execution_failures:
            escalate(
                "REDUCING",
                f"execution_failures:{consecutive_execution_failures}",
            )
        if reconciliation_failures >= policy.max_reconciliation_failures:
            escalate(
                "REDUCING",
                f"reconciliation_failures:{reconciliation_failures}",
            )

        if recommended == "HALTED":
            self._halt_latched = True
            next_state: TradingState = "HALTED"
        elif self._halt_latched:
            if manual_release:
                self._halt_latched = False
                next_state = recommended
                reasons.append("manual_halt_release")
            else:
                next_state = "HALTED"
                reasons.append("halt_latched_manual_release_required")
        else:
            next_state = recommended

        self._state = next_state
        payload = {
            "schema_version": PHASE68_SCHEMA_VERSION,
            "timestamp": float(now),
            "previous_state": previous_state,
            "trading_state": next_state,
            "recommended_state": recommended,
            "reasons": tuple(reasons),
            "max_drawdown_fraction": max_drawdown,
            "window_loss_fraction": window_loss,
            "qualifying_stoplosses": len(stoploss_trades),
            "stoploss_lock_until": stoploss_lock_until,
            "blocked_assets": tuple(sorted(blocked_assets)),
            "consecutive_execution_failures": consecutive_execution_failures,
            "reconciliation_failures": reconciliation_failures,
            "unknown_order_outcomes": unknown_outcomes,
            "market_data_age_seconds": market_data_age,
            "manual_halt": manual_halt,
            "manual_release_requested": manual_release,
            "halt_latched": self._halt_latched,
        }
        return OperationalRiskReceipt(
            timestamp=float(now),
            previous_state=previous_state,
            trading_state=next_state,
            recommended_state=recommended,
            reasons=tuple(reasons),
            max_drawdown_fraction=max_drawdown,
            window_loss_fraction=window_loss,
            qualifying_stoplosses=len(stoploss_trades),
            stoploss_lock_until=stoploss_lock_until,
            blocked_assets=tuple(sorted(blocked_assets)),
            consecutive_execution_failures=consecutive_execution_failures,
            reconciliation_failures=reconciliation_failures,
            unknown_order_outcomes=unknown_outcomes,
            market_data_age_seconds=market_data_age,
            manual_halt=manual_halt,
            manual_release_requested=manual_release,
            halt_latched=self._halt_latched,
            receipt_id=content_hash(payload),
        )
