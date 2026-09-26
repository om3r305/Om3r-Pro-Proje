from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Sequence
import math

from .phase44_portfolio_brain import PortfolioBookPlan

PHASE45_SCHEMA_VERSION = "brian.phase45-execution-contract.v1"
ExecutorStatus = Literal["NOT_STARTED", "RUNNING", "SHUTTING_DOWN", "TERMINATED"]
CloseType = Literal[
    "TIME_LIMIT",
    "STOP_LOSS",
    "TAKE_PROFIT",
    "EXPIRED",
    "EARLY_STOP",
    "TRAILING_STOP",
    "INSUFFICIENT_BALANCE",
    "FAILED",
    "COMPLETED",
    "POSITION_HOLD",
]
Side = Literal["BUY", "SELL"]


@dataclass(frozen=True, slots=True)
class TradeIntent:
    intent_id: str
    asset_id: str
    direction: int
    target_weight: float
    expected_edge_bps: float
    confidence: float
    max_slippage_bps: float
    created_at: float
    ttl_seconds: int
    evidence_ids: tuple[str, ...]
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = PHASE45_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.intent_id.strip() or not self.asset_id.strip():
            raise ValueError("intent identity is required")
        if self.direction not in (-1, 1):
            raise ValueError("trade intent direction must be -1 or 1")
        if not math.isfinite(self.target_weight) or self.target_weight == 0:
            raise ValueError("target_weight must be finite and non-zero")
        if (1 if self.target_weight > 0 else -1) != self.direction:
            raise ValueError("target_weight sign must match direction")
        if not math.isfinite(self.expected_edge_bps):
            raise ValueError("expected_edge_bps must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        if not math.isfinite(self.max_slippage_bps) or self.max_slippage_bps < 0:
            raise ValueError("max_slippage_bps must be non-negative")
        if not math.isfinite(self.created_at) or self.ttl_seconds <= 0:
            raise ValueError("intent time contract is invalid")
        if not self.evidence_ids:
            raise ValueError("trade intents require evidence lineage")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase 45 intents are shadow-only")

    def expired(self, now: float) -> bool:
        if now < self.created_at:
            raise ValueError("execution clock cannot precede intent creation")
        return now - self.created_at > self.ttl_seconds


@dataclass(frozen=True, slots=True)
class TripleBarrierPolicy:
    stop_loss_fraction: float | None = None
    take_profit_fraction: float | None = None
    time_limit_seconds: int | None = None
    open_order_type: str = "LIMIT"
    take_profit_order_type: str = "MARKET"
    stop_loss_order_type: str = "MARKET"
    time_limit_order_type: str = "MARKET"

    def __post_init__(self) -> None:
        for label, value in (
            ("stop_loss_fraction", self.stop_loss_fraction),
            ("take_profit_fraction", self.take_profit_fraction),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{label} must be positive when set")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive when set")
        if self.stop_loss_order_type != "MARKET":
            raise ValueError("stop-loss must close at market in Phase 45")
        if self.time_limit_order_type != "MARKET":
            raise ValueError("time-limit must close at market in Phase 45")


@dataclass(frozen=True, slots=True)
class PositionExecutorConfig:
    executor_id: str
    intent_id: str
    connector_name: str
    trading_pair: str
    side: Side
    amount_quote: float
    entry_price: float | None
    barrier: TripleBarrierPolicy
    leverage: int = 1
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.executor_id.strip() or not self.intent_id.strip():
            raise ValueError("executor identity is required")
        if self.connector_name != "shadow":
            raise ValueError("Phase 45 connector must remain the shadow adapter")
        if not self.trading_pair.strip():
            raise ValueError("trading_pair is required")
        if not math.isfinite(self.amount_quote) or self.amount_quote <= 0:
            raise ValueError("amount_quote must be positive")
        if self.entry_price is not None and (not math.isfinite(self.entry_price) or self.entry_price <= 0):
            raise ValueError("entry_price must be positive when set")
        if self.leverage < 1:
            raise ValueError("leverage must be at least 1")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase 45 executors cannot be live")


@dataclass(frozen=True, slots=True)
class CreateExecutorAction:
    controller_id: str
    executor_config: PositionExecutorConfig
    action: str = "CREATE"


@dataclass(frozen=True, slots=True)
class StopExecutorAction:
    controller_id: str
    executor_id: str
    keep_position: bool = False
    action: str = "STOP"


@dataclass(frozen=True, slots=True)
class StoreExecutorAction:
    controller_id: str
    executor_id: str
    action: str = "STORE"


def create_position_executor_action(
    intent: TradeIntent,
    *,
    equity_usd: float,
    reference_price: float,
    barrier: TripleBarrierPolicy,
    controller_id: str = "brian",
) -> CreateExecutorAction:
    if intent.expired(intent.created_at):
        raise ValueError("newly-created intent cannot already be expired")
    if not math.isfinite(equity_usd) or equity_usd <= 0:
        raise ValueError("equity_usd must be positive")
    if not math.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("reference_price must be positive")
    amount_quote = equity_usd * abs(intent.target_weight)
    config = PositionExecutorConfig(
        executor_id=f"position:{intent.intent_id}",
        intent_id=intent.intent_id,
        connector_name="shadow",
        trading_pair=intent.asset_id,
        side="BUY" if intent.direction > 0 else "SELL",
        amount_quote=amount_quote,
        entry_price=reference_price,
        barrier=barrier,
    )
    return CreateExecutorAction(controller_id=controller_id, executor_config=config)


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    fill_id: str
    order_id: str
    is_buy: bool
    amount_base: float
    amount_quote: float
    fee_quote: float
    observed_at: float

    def __post_init__(self) -> None:
        if not self.fill_id.strip() or not self.order_id.strip():
            raise ValueError("fill and order identity are required")
        if min(self.amount_base, self.amount_quote, self.fee_quote) < 0:
            raise ValueError("fill amounts and fees cannot be negative")
        if self.amount_base == 0 and self.amount_quote != 0:
            raise ValueError("zero-base fill cannot carry quote amount")
        if not all(math.isfinite(value) for value in (
            self.amount_base, self.amount_quote, self.fee_quote, self.observed_at
        )):
            raise ValueError("fill values must be finite")


@dataclass(slots=True)
class ShadowPositionHold:
    """Incremental net-position accounting adapted from Hummingbot PositionHold.

    Fills reduce existing exposure first; any excess flips the position and
    establishes a new average entry. Duplicate fill IDs are idempotent.
    """
    trading_pair: str
    processed_fill_ids: set[str] = field(default_factory=set)
    order_ids: set[str] = field(default_factory=set)
    net_amount_base: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl_quote: float = 0.0
    volume_traded_quote: float = 0.0
    cumulative_fees_quote: float = 0.0
    buy_amount_base: float = 0.0
    buy_amount_quote: float = 0.0
    sell_amount_base: float = 0.0
    sell_amount_quote: float = 0.0

    def apply_fill(self, fill: ExecutionFill) -> bool:
        if fill.fill_id in self.processed_fill_ids:
            return False
        self.processed_fill_ids.add(fill.fill_id)
        self.order_ids.add(fill.order_id)
        self.volume_traded_quote += fill.amount_quote
        self.cumulative_fees_quote += fill.fee_quote
        if fill.is_buy:
            self.buy_amount_base += fill.amount_base
            self.buy_amount_quote += fill.amount_quote
        else:
            self.sell_amount_base += fill.amount_base
            self.sell_amount_quote += fill.amount_quote

        if fill.amount_base == 0:
            return True
        order_price = fill.amount_quote / fill.amount_base
        is_reducing = (
            (self.net_amount_base > 0 and not fill.is_buy)
            or (self.net_amount_base < 0 and fill.is_buy)
        )
        if is_reducing:
            absolute_net = abs(self.net_amount_base)
            matched = min(fill.amount_base, absolute_net)
            if self.net_amount_base > 0:
                self.realized_pnl_quote += (order_price - self.avg_entry_price) * matched
            else:
                self.realized_pnl_quote += (self.avg_entry_price - order_price) * matched

            excess = fill.amount_base - matched
            if excess > 0:
                self.net_amount_base = excess if fill.is_buy else -excess
                self.avg_entry_price = order_price
            elif math.isclose(matched, absolute_net, rel_tol=0.0, abs_tol=1e-12):
                self.net_amount_base = 0.0
                self.avg_entry_price = 0.0
            elif self.net_amount_base > 0:
                self.net_amount_base -= matched
            else:
                self.net_amount_base += matched
        else:
            if self.net_amount_base == 0:
                self.net_amount_base = fill.amount_base if fill.is_buy else -fill.amount_base
                self.avg_entry_price = order_price
            else:
                absolute_net = abs(self.net_amount_base)
                total_cost = self.avg_entry_price * absolute_net + fill.amount_quote
                new_absolute = absolute_net + fill.amount_base
                self.avg_entry_price = total_cost / new_absolute
                self.net_amount_base = new_absolute if fill.is_buy else -new_absolute
        return True

    def unrealized_pnl(self, mid_price: float) -> float:
        if not math.isfinite(mid_price) or mid_price <= 0:
            raise ValueError("mid_price must be positive")
        if self.net_amount_base == 0:
            return 0.0
        if self.net_amount_base > 0:
            return (mid_price - self.avg_entry_price) * self.net_amount_base
        return (self.avg_entry_price - mid_price) * abs(self.net_amount_base)


@dataclass(slots=True)
class ShadowExecutorState:
    config: PositionExecutorConfig
    status: ExecutorStatus = "NOT_STARTED"
    close_type: CloseType | None = None
    hold: ShadowPositionHold | None = None

    def start(self) -> None:
        if self.status != "NOT_STARTED":
            raise ValueError("executor may start exactly once")
        self.hold = ShadowPositionHold(self.config.trading_pair)
        self.status = "RUNNING"

    def request_stop(self, *, keep_position: bool = False) -> None:
        if self.status not in ("RUNNING", "SHUTTING_DOWN"):
            raise ValueError("only active executors may stop")
        self.close_type = "POSITION_HOLD" if keep_position else "EARLY_STOP"
        self.status = "SHUTTING_DOWN"

    def terminate(self, close_type: CloseType | None = None) -> None:
        if self.status not in ("RUNNING", "SHUTTING_DOWN"):
            raise ValueError("executor must be active before termination")
        if close_type is not None:
            self.close_type = close_type
        if self.close_type is None:
            self.close_type = "COMPLETED"
        self.status = "TERMINATED"


class ShadowExecutorOrchestrator:
    """Create/stop/store orchestrator with no exchange/order transport."""

    def __init__(self) -> None:
        self.active: dict[str, ShadowExecutorState] = {}
        self.stored: dict[str, ShadowExecutorState] = {}

    def apply(self, action: CreateExecutorAction | StopExecutorAction | StoreExecutorAction) -> ShadowExecutorState:
        if isinstance(action, CreateExecutorAction):
            config = action.executor_config
            if config.executor_id in self.active or config.executor_id in self.stored:
                raise ValueError("duplicate executor id")
            state = ShadowExecutorState(config)
            state.start()
            self.active[config.executor_id] = state
            return state

        if isinstance(action, StopExecutorAction):
            state = self.active.get(action.executor_id)
            if state is None:
                raise KeyError(f"unknown active executor: {action.executor_id}")
            state.request_stop(keep_position=action.keep_position)
            return state

        if isinstance(action, StoreExecutorAction):
            state = self.active.get(action.executor_id)
            if state is None:
                raise KeyError(f"unknown active executor: {action.executor_id}")
            if state.status != "TERMINATED":
                raise ValueError("only terminated executors may be stored")
            self.stored[action.executor_id] = state
            del self.active[action.executor_id]
            return state

        raise TypeError("unsupported executor action")


def intents_from_portfolio_book(
    plan: PortfolioBookPlan,
    *,
    created_at: float,
    expected_edge_bps_by_asset: Mapping[str, float],
    confidence_by_asset: Mapping[str, float],
    max_slippage_bps: float,
    ttl_seconds: int,
) -> tuple[TradeIntent, ...]:
    intents: list[TradeIntent] = []
    for asset_id, weight in sorted(plan.risk.weights.items()):
        if abs(weight) <= 1e-12:
            continue
        edge = expected_edge_bps_by_asset.get(asset_id)
        confidence = confidence_by_asset.get(asset_id)
        if edge is None or confidence is None:
            continue
        direction = 1 if weight > 0 else -1
        intent_id = f"{asset_id}:{created_at:.6f}:{direction}:{abs(weight):.12f}"
        intents.append(TradeIntent(
            intent_id=intent_id,
            asset_id=asset_id,
            direction=direction,
            target_weight=float(weight),
            expected_edge_bps=float(edge),
            confidence=float(confidence),
            max_slippage_bps=float(max_slippage_bps),
            created_at=float(created_at),
            ttl_seconds=int(ttl_seconds),
            evidence_ids=plan.source_evidence_ids,
        ))
    return tuple(intents)
