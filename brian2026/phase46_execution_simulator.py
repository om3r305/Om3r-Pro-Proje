from __future__ import annotations

from dataclasses import asdict, dataclass, field
from random import Random
from typing import Literal, Sequence
import math

from .phase45_execution_contract import CreateExecutorAction, TradeIntent

PHASE46_SCHEMA_VERSION = "brian.phase46-execution-simulator.v1"
OrderType = Literal["MARKET", "LIMIT"]
Side = Literal["BUY", "SELL"]
FillStatus = Literal["FILLED", "PARTIAL_FILL", "NO_FILL", "VETO_SLIPPAGE"]


@dataclass(frozen=True, slots=True)
class StaticLatencyModel:
    base_latency_ms: float = 0.0
    insert_latency_ms: float = 0.0
    update_latency_ms: float = 0.0
    delete_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.base_latency_ms,
            self.insert_latency_ms,
            self.update_latency_ms,
            self.delete_latency_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("latencies must be finite and non-negative")

    @property
    def effective_insert_ms(self) -> float:
        return self.base_latency_ms + self.insert_latency_ms

    @property
    def effective_update_ms(self) -> float:
        return self.base_latency_ms + self.update_latency_ms

    @property
    def effective_delete_ms(self) -> float:
        return self.base_latency_ms + self.delete_latency_ms


@dataclass(slots=True)
class ProbabilisticFillModel:
    prob_fill_on_limit: float = 1.0
    prob_slippage: float = 0.0
    seed: int = 4601
    allow_inside_spread_fill: bool = False
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("prob_fill_on_limit", self.prob_fill_on_limit),
            ("prob_slippage", self.prob_slippage),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} must be in [0,1]")
        self._rng = Random(self.seed)

    def _event(self, probability: float) -> bool:
        if probability <= 0:
            return False
        if probability >= 1:
            return True
        return self._rng.random() < probability

    def is_limit_filled(self) -> bool:
        return self._event(self.prob_fill_on_limit)

    def is_slipped(self) -> bool:
        return self._event(self.prob_slippage)


@dataclass(frozen=True, slots=True)
class LiquidityLevel:
    price: float
    quantity: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("liquidity level price must be positive")
        if not math.isfinite(self.quantity) or self.quantity < 0:
            raise ValueError("liquidity level quantity must be non-negative")


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    timestamp: float
    bids: tuple[LiquidityLevel, ...]
    asks: tuple[LiquidityLevel, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("order book timestamp must be finite")
        if not self.bids or not self.asks:
            raise ValueError("order book requires bids and asks")
        if any(left.price < right.price for left, right in zip(self.bids, self.bids[1:])):
            raise ValueError("bids must be descending")
        if any(left.price > right.price for left, right in zip(self.asks, self.asks[1:])):
            raise ValueError("asks must be ascending")
        if self.bids[0].price >= self.asks[0].price:
            raise ValueError("crossed/locked order book is invalid")


class InflightOrderRegistry:
    """Tracks submits that have not reached the simulated venue yet.

    Duplicate submits do not require duplicate receipts: the first venue receipt
    releases the client order id, matching the behavior verified in Nautilus'
    inflight-order tests.
    """

    def __init__(self) -> None:
        self._ids: set[str] = set()

    def submit(self, client_order_ids: Sequence[str]) -> None:
        for value in client_order_ids:
            if not str(value).strip():
                raise ValueError("client order ids must be non-empty")
            self._ids.add(str(value))

    def receipt(self, client_order_ids: Sequence[str]) -> None:
        for value in client_order_ids:
            self._ids.discard(str(value))

    def contains(self, client_order_id: str) -> bool:
        return client_order_id in self._ids

    def clear(self) -> None:
        self._ids.clear()


@dataclass(frozen=True, slots=True)
class SimulatedExecutionReceipt:
    status: FillStatus
    side: Side
    order_type: OrderType
    submit_timestamp: float
    venue_timestamp: float | None
    snapshot_timestamp: float | None
    requested_base: float
    filled_base: float
    fill_fraction: float
    average_fill_price: float | None
    best_reference_price: float | None
    adverse_slippage_bps: float | None
    levels_consumed: int
    slipped_one_tick: bool
    reason: str
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = PHASE46_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _first_snapshot_after(
    snapshots: Sequence[OrderBookSnapshot],
    timestamp: float,
) -> OrderBookSnapshot | None:
    eligible = [row for row in snapshots if row.timestamp >= timestamp]
    return min(eligible, key=lambda row: row.timestamp) if eligible else None


def _adverse_bps(side: Side, best: float, average: float) -> float:
    if side == "BUY":
        return max(0.0, (average / best - 1.0) * 10_000.0)
    return max(0.0, (best / average - 1.0) * 10_000.0)


def simulate_order(
    *,
    side: Side,
    order_type: OrderType,
    requested_base: float,
    submit_timestamp: float,
    snapshots: Sequence[OrderBookSnapshot],
    latency: StaticLatencyModel = StaticLatencyModel(),
    fill_model: ProbabilisticFillModel | None = None,
    limit_price: float | None = None,
    tick_size: float,
    max_slippage_bps: float | None = None,
) -> SimulatedExecutionReceipt:
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    if order_type not in ("MARKET", "LIMIT"):
        raise ValueError("order_type must be MARKET or LIMIT")
    if not math.isfinite(requested_base) or requested_base <= 0:
        raise ValueError("requested_base must be positive")
    if not math.isfinite(submit_timestamp):
        raise ValueError("submit_timestamp must be finite")
    if not math.isfinite(tick_size) or tick_size <= 0:
        raise ValueError("tick_size must be positive")
    if order_type == "LIMIT" and (limit_price is None or not math.isfinite(limit_price) or limit_price <= 0):
        raise ValueError("limit orders require a positive limit_price")
    if max_slippage_bps is not None and (
        not math.isfinite(max_slippage_bps) or max_slippage_bps < 0
    ):
        raise ValueError("max_slippage_bps must be non-negative")

    model = fill_model or ProbabilisticFillModel()
    venue_timestamp = submit_timestamp + latency.effective_insert_ms / 1000.0
    book = _first_snapshot_after(snapshots, venue_timestamp)
    if book is None:
        return SimulatedExecutionReceipt(
            "NO_FILL", side, order_type, submit_timestamp, venue_timestamp, None,
            requested_base, 0.0, 0.0, None, None, None, 0, False,
            "no order-book snapshot available after venue arrival",
        )

    best_bid = book.bids[0].price
    best_ask = book.asks[0].price
    best_reference = best_ask if side == "BUY" else best_bid

    if order_type == "LIMIT":
        assert limit_price is not None
        marketable = limit_price >= best_ask if side == "BUY" else limit_price <= best_bid
        inside_spread = (
            best_bid <= limit_price < best_ask
            if side == "BUY"
            else best_bid < limit_price <= best_ask
        )
        if not marketable and not (model.allow_inside_spread_fill and inside_spread):
            return SimulatedExecutionReceipt(
                "NO_FILL", side, order_type, submit_timestamp, venue_timestamp, book.timestamp,
                requested_base, 0.0, 0.0, None, best_reference, None, 0, False,
                "limit price is not fillable against the simulated book",
            )
        if not model.is_limit_filled():
            return SimulatedExecutionReceipt(
                "NO_FILL", side, order_type, submit_timestamp, venue_timestamp, book.timestamp,
                requested_base, 0.0, 0.0, None, best_reference, None, 0, False,
                "probabilistic limit-fill model rejected the fill",
            )

    slipped = model.is_slipped()
    levels = book.asks if side == "BUY" else book.bids
    remaining = requested_base
    filled = 0.0
    quote = 0.0
    consumed = 0

    for level in levels:
        price = level.price + tick_size if side == "BUY" and slipped else (
            level.price - tick_size if side == "SELL" and slipped else level.price
        )
        if price <= 0:
            continue
        if order_type == "LIMIT":
            assert limit_price is not None
            if side == "BUY" and price > limit_price:
                break
            if side == "SELL" and price < limit_price:
                break
        quantity = min(remaining, level.quantity)
        if quantity <= 0:
            continue
        filled += quantity
        quote += quantity * price
        remaining -= quantity
        consumed += 1
        if remaining <= 1e-12:
            break

    if filled <= 0:
        return SimulatedExecutionReceipt(
            "NO_FILL", side, order_type, submit_timestamp, venue_timestamp, book.timestamp,
            requested_base, 0.0, 0.0, None, best_reference, None, 0, slipped,
            "visible simulated liquidity produced no fill",
        )

    average = quote / filled
    adverse = _adverse_bps(side, best_reference, average)
    if max_slippage_bps is not None and adverse > max_slippage_bps + 1e-12:
        return SimulatedExecutionReceipt(
            "VETO_SLIPPAGE", side, order_type, submit_timestamp, venue_timestamp, book.timestamp,
            requested_base, 0.0, 0.0, average, best_reference, adverse, consumed, slipped,
            "projected execution exceeds intent max_slippage_bps",
        )

    fraction = min(1.0, filled / requested_base)
    status: FillStatus = "FILLED" if fraction >= 1.0 - 1e-12 else "PARTIAL_FILL"
    return SimulatedExecutionReceipt(
        status, side, order_type, submit_timestamp, venue_timestamp, book.timestamp,
        requested_base, filled, fraction, average, best_reference, adverse, consumed, slipped,
        "tiered simulated liquidity consumed at venue-arrival snapshot",
    )


def simulate_create_action(
    action: CreateExecutorAction,
    intent: TradeIntent,
    *,
    snapshots: Sequence[OrderBookSnapshot],
    latency: StaticLatencyModel = StaticLatencyModel(),
    fill_model: ProbabilisticFillModel | None = None,
    tick_size: float,
) -> SimulatedExecutionReceipt:
    config = action.executor_config
    if config.intent_id != intent.intent_id:
        raise ValueError("executor action does not belong to trade intent")
    if intent.expired(max(intent.created_at, snapshots[0].timestamp if snapshots else intent.created_at)):
        return SimulatedExecutionReceipt(
            "NO_FILL", config.side, config.barrier.open_order_type, intent.created_at,
            None, None, 0.0, 0.0, 0.0, None, None, None, 0, False,
            "trade intent expired before execution simulation",
        )
    reference = config.entry_price
    if reference is None or reference <= 0:
        raise ValueError("Phase 46 requires a reference entry price")
    requested_base = config.amount_quote / reference
    return simulate_order(
        side=config.side,
        order_type=config.barrier.open_order_type,
        requested_base=requested_base,
        submit_timestamp=intent.created_at,
        snapshots=snapshots,
        latency=latency,
        fill_model=fill_model,
        limit_price=reference if config.barrier.open_order_type == "LIMIT" else None,
        tick_size=tick_size,
        max_slippage_bps=intent.max_slippage_bps,
    )
