from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal
import math

from .phase45_execution_contract import TradeIntent
from .phase55_rebalance_execution_intents import RiskReductionIntent

PHASE56_SCHEMA_VERSION = "brian.phase56-pretrade-risk-engine.v1"
TradingState = Literal["ACTIVE", "REDUCING", "HALTED"]
RiskAction = Literal["ALLOW", "DENY"]


@dataclass(frozen=True, slots=True)
class InstrumentRiskLimits:
    min_notional: float = 0.0
    max_notional: float | None = None
    max_notional_per_order: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_notional) or self.min_notional < 0:
            raise ValueError("min_notional must be finite and non-negative")
        for label, value in (
            ("max_notional", self.max_notional),
            ("max_notional_per_order", self.max_notional_per_order),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{label} must be positive when set")
        if self.max_notional is not None and self.max_notional < self.min_notional:
            raise ValueError("max_notional cannot be below min_notional")


@dataclass(frozen=True, slots=True)
class PreTradeAccountState:
    equity_usd: float
    available_cash_usd: float
    open_position_weight: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.equity_usd,
            self.available_cash_usd,
            self.open_position_weight,
        )):
            raise ValueError("account risk state must be finite")
        if self.equity_usd <= 0 or self.available_cash_usd < 0:
            raise ValueError("equity must be positive and cash non-negative")


@dataclass(frozen=True, slots=True)
class PreTradeRiskReceipt:
    action: RiskAction
    trading_state: TradingState
    asset_id: str
    requested_notional_usd: float
    reduce_only: bool
    reasons: tuple[str, ...]
    projected_position_weight: float | None
    checks: tuple[tuple[str, bool], ...]
    schema_version: str = PHASE56_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    @property
    def allowed(self) -> bool:
        return self.action == "ALLOW"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allowed"] = self.allowed
        return payload


def _notional_from_weight(weight: float, equity: float) -> float:
    return abs(float(weight)) * float(equity)


def _notional_checks(
    requested_notional: float,
    limits: InstrumentRiskLimits,
) -> tuple[tuple[str, bool], ...]:
    return (
        ("positive_notional", requested_notional > 0.0),
        ("instrument_min_notional", requested_notional + 1e-12 >= limits.min_notional),
        (
            "instrument_max_notional",
            limits.max_notional is None or requested_notional <= limits.max_notional + 1e-12,
        ),
        (
            "configured_max_notional_per_order",
            limits.max_notional_per_order is None
            or requested_notional <= limits.max_notional_per_order + 1e-12,
        ),
    )


def review_new_risk_intent(
    intent: TradeIntent,
    *,
    trading_state: TradingState,
    account: PreTradeAccountState,
    limits: InstrumentRiskLimits,
) -> PreTradeRiskReceipt:
    """Independent hard gate for a new/increasing-risk TradeIntent.

    Adapted from Nautilus' RiskEngine semantics: HALTED rejects submissions,
    REDUCING rejects new risk, and ACTIVE still enforces instrument/configured
    notional and cash constraints. The intelligence layer cannot override this
    receipt.
    """
    if trading_state not in ("ACTIVE", "REDUCING", "HALTED"):
        raise ValueError("invalid trading_state")
    requested_notional = _notional_from_weight(intent.target_weight, account.equity_usd)
    checks = list(_notional_checks(requested_notional, limits))
    checks.extend((
        ("trading_state_active", trading_state == "ACTIVE"),
        ("cash_available", requested_notional <= account.available_cash_usd + 1e-12),
        ("intent_is_shadow_only", intent.shadow_only and not intent.live_execution),
    ))
    reasons = tuple(name for name, passed in checks if not passed)
    projected = account.open_position_weight + intent.target_weight
    return PreTradeRiskReceipt(
        action="ALLOW" if not reasons else "DENY",
        trading_state=trading_state,
        asset_id=intent.asset_id,
        requested_notional_usd=requested_notional,
        reduce_only=False,
        reasons=reasons,
        projected_position_weight=projected,
        checks=tuple(checks),
    )


def review_reduce_only_intent(
    intent: RiskReductionIntent,
    *,
    trading_state: TradingState,
    account: PreTradeAccountState,
    limits: InstrumentRiskLimits,
) -> PreTradeRiskReceipt:
    """Validate exposure reduction under ACTIVE or REDUCING trading state.

    Mirrors Nautilus' reducing-submission invariants at portfolio-weight level:
    the identified position must exist, order side must oppose it, reduction
    cannot exceed current exposure, and resulting exposure cannot flip.
    HALTED denies submissions entirely.
    """
    if trading_state not in ("ACTIVE", "REDUCING", "HALTED"):
        raise ValueError("invalid trading_state")
    current = account.open_position_weight
    current_direction = 1 if current > 1e-12 else -1 if current < -1e-12 else 0
    requested_notional = _notional_from_weight(intent.reduce_weight, account.equity_usd)
    checks = list(_notional_checks(requested_notional, limits))
    checks.extend((
        ("trading_state_allows_reduction", trading_state in ("ACTIVE", "REDUCING")),
        ("identified_open_position", current_direction != 0),
        ("current_direction_matches_intent", current_direction == intent.current_direction),
        ("order_side_opposes_position", intent.order_direction == -current_direction if current_direction else False),
        ("reduction_not_larger_than_position", intent.reduce_weight <= abs(current) + 1e-12),
        (
            "result_does_not_flip",
            abs(intent.resulting_weight) <= 1e-12
            or (
                current_direction != 0
                and (1 if intent.resulting_weight > 0 else -1) == current_direction
            ),
        ),
        (
            "result_matches_declared_account_path",
            math.isclose(
                intent.resulting_weight,
                current + intent.order_direction * intent.reduce_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        ),
        ("intent_is_reduce_only", intent.reduce_only),
        ("intent_is_shadow_only", intent.shadow_only and not intent.live_execution),
    ))
    reasons = tuple(name for name, passed in checks if not passed)
    return PreTradeRiskReceipt(
        action="ALLOW" if not reasons else "DENY",
        trading_state=trading_state,
        asset_id=intent.asset_id,
        requested_notional_usd=requested_notional,
        reduce_only=True,
        reasons=reasons,
        projected_position_weight=intent.resulting_weight if not reasons else None,
        checks=tuple(checks),
    )


@dataclass(frozen=True, slots=True)
class PreTradeRiskPolicy:
    trading_state: TradingState = "ACTIVE"
    limits: InstrumentRiskLimits = InstrumentRiskLimits()


class PreTradeRiskEngine:
    """Small fail-closed shadow RiskEngine boundary.

    There is intentionally no bypass option. Intelligence, portfolio construction
    and execution orchestration receive only ALLOW/DENY receipts.
    """

    def __init__(self, policy: PreTradeRiskPolicy = PreTradeRiskPolicy()) -> None:
        self.policy = policy

    def review(
        self,
        intent: TradeIntent | RiskReductionIntent,
        account: PreTradeAccountState,
    ) -> PreTradeRiskReceipt:
        if isinstance(intent, TradeIntent):
            return review_new_risk_intent(
                intent,
                trading_state=self.policy.trading_state,
                account=account,
                limits=self.policy.limits,
            )
        if isinstance(intent, RiskReductionIntent):
            return review_reduce_only_intent(
                intent,
                trading_state=self.policy.trading_state,
                account=account,
                limits=self.policy.limits,
            )
        raise TypeError("unsupported pre-trade intent")
