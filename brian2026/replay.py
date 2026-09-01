from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol, Sequence

ReplayAction = Literal["LONG", "SHORT", "WAIT"]


@dataclass(frozen=True, slots=True)
class ReplayPoint:
    timestamp: float
    bid: float
    ask: float
    high: float
    low: float
    close: float
    bid_liquidity: float | None = None
    ask_liquidity: float | None = None

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask < self.bid or self.low > self.high:
            raise ValueError("invalid replay point")
        if not self.low <= self.close <= self.high:
            raise ValueError("close outside candle range")


class PartialFillModel(Protocol):
    def fraction(self, action: ReplayAction, requested_quantity: float,
                 point: ReplayPoint) -> float: ...


class FundingModel(Protocol):
    def cost(self, action: ReplayAction, notional: float,
             entry_timestamp: float, exit_timestamp: float) -> float: ...


class MarketImpactModel(Protocol):
    def impact_bps(self, action: ReplayAction, notional: float,
                   point: ReplayPoint) -> float: ...


@dataclass(frozen=True, slots=True)
class ConservativePartialFill:
    max_participation: float = 0.10

    def fraction(self, action: ReplayAction, requested_quantity: float,
                 point: ReplayPoint) -> float:
        available = point.ask_liquidity if action == "LONG" else point.bid_liquidity
        if available is None:
            return 1.0
        capacity = max(0.0, available * self.max_participation)
        return min(1.0, capacity / requested_quantity) if requested_quantity > 0 else 0.0


@dataclass(frozen=True, slots=True)
class FixedFunding:
    rate_bps_per_8h: float = 0.0

    def cost(self, action: ReplayAction, notional: float,
             entry_timestamp: float, exit_timestamp: float) -> float:
        periods = max(0.0, exit_timestamp - entry_timestamp) / (8.0 * 3600.0)
        signed = 1.0 if action == "LONG" else -1.0
        return notional * self.rate_bps_per_8h / 10_000.0 * periods * signed


@dataclass(frozen=True, slots=True)
class LinearMarketImpact:
    coefficient_bps: float = 0.0

    def impact_bps(self, action: ReplayAction, notional: float,
                   point: ReplayPoint) -> float:
        liquidity_qty = point.ask_liquidity if action == "LONG" else point.bid_liquidity
        if not liquidity_qty or liquidity_qty <= 0:
            return max(0.0, self.coefficient_bps)
        liquidity_notional = liquidity_qty * ((point.bid + point.ask) / 2.0)
        return max(0.0, self.coefficient_bps * notional / max(liquidity_notional, 1e-12))


@dataclass(frozen=True, slots=True)
class ReplaySettings:
    position_size: float = 1.0
    tp_pct: float = 1.0
    sl_pct: float = 0.5
    latency_ms: int = 0
    slippage_bps: float = 0.0
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 10.0
    entry_maker: bool = False
    exit_maker: bool = False
    stop_first_same_bar: bool = True


@dataclass(frozen=True, slots=True)
class ReplayResult:
    action: ReplayAction
    status: str
    entry_timestamp: float | None
    exit_timestamp: float | None
    entry_price: float | None
    exit_price: float | None
    requested_quantity: float
    filled_quantity: float
    fill_fraction: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    return_pct: float
    exposure_seconds: float
    exit_reason: str

    def to_dict(self):
        return asdict(self)


def _empty(action: ReplayAction, status: str) -> ReplayResult:
    return ReplayResult(action, status, None, None, None, None, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status)


def replay(
    path: Sequence[ReplayPoint], action: ReplayAction, settings: ReplaySettings,
    *, decision_timestamp: float | None = None,
    partial_fill: PartialFillModel | None = None,
    funding_model: FundingModel | None = None,
    impact_model: MarketImpactModel | None = None,
) -> ReplayResult:
    """Replay one decision against one immutable subsequent market path."""
    if action == "WAIT":
        return _empty(action, "WAIT")
    if not path or settings.position_size <= 0:
        return _empty(action, "NO_FILL")
    decision_ts = path[0].timestamp if decision_timestamp is None else decision_timestamp
    eligible = [point for point in path if point.timestamp >= decision_ts + settings.latency_ms / 1000.0]
    if not eligible:
        return _empty(action, "NO_FILL")
    entry_point = eligible[0]
    impact = (impact_model or LinearMarketImpact()).impact_bps(action, settings.position_size, entry_point)
    adverse_bps = max(0.0, settings.slippage_bps + impact)
    if action == "LONG":
        entry_price = entry_point.ask * (1.0 + adverse_bps / 10_000.0)
    else:
        entry_price = entry_point.bid * (1.0 - adverse_bps / 10_000.0)
    requested_qty = settings.position_size / entry_price
    fill_fraction = max(0.0, min(1.0, (partial_fill or ConservativePartialFill()).fraction(
        action, requested_qty, entry_point)))
    qty = requested_qty * fill_fraction
    if qty <= 0:
        return _empty(action, "NO_FILL")

    if action == "LONG":
        target, stop = entry_price * (1 + settings.tp_pct / 100.0), entry_price * (1 - settings.sl_pct / 100.0)
    else:
        target, stop = entry_price * (1 - settings.tp_pct / 100.0), entry_price * (1 + settings.sl_pct / 100.0)
    exit_point, exit_base, reason = eligible[-1], None, "PATH_END"
    for point in eligible[1:]:
        stop_hit = point.low <= stop if action == "LONG" else point.high >= stop
        target_hit = point.high >= target if action == "LONG" else point.low <= target
        if stop_hit and target_hit:
            reason = "STOP" if settings.stop_first_same_bar else "TARGET"
        elif stop_hit:
            reason = "STOP"
        elif target_hit:
            reason = "TARGET"
        else:
            continue
        exit_point = point
        exit_base = stop if reason == "STOP" else target
        break
    if exit_base is None:
        exit_base = exit_point.bid if action == "LONG" else exit_point.ask
    exit_adverse = max(0.0, settings.slippage_bps) / 10_000.0
    exit_price = exit_base * (1.0 - exit_adverse if action == "LONG" else 1.0 + exit_adverse)
    direction = 1.0 if action == "LONG" else -1.0
    gross = (exit_price - entry_price) * qty * direction
    entry_fee_rate = settings.maker_fee_bps if settings.entry_maker else settings.taker_fee_bps
    exit_fee_rate = settings.maker_fee_bps if settings.exit_maker else settings.taker_fee_bps
    fees = entry_price * qty * entry_fee_rate / 10_000.0 + exit_price * qty * exit_fee_rate / 10_000.0
    funding = (funding_model or FixedFunding()).cost(
        action, entry_price * qty, entry_point.timestamp, exit_point.timestamp)
    net = gross - fees - funding
    filled_notional = entry_price * qty
    return ReplayResult(
        action, "FILLED", entry_point.timestamp, exit_point.timestamp,
        entry_price, exit_price, requested_qty, qty, fill_fraction, gross, fees,
        funding, net, net / filled_notional * 100.0 if filled_notional else 0.0,
        max(0.0, exit_point.timestamp - entry_point.timestamp), reason,
    )
