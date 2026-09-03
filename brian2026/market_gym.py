from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Mapping, Sequence
import json
import math

GYM_SCHEMA_VERSION = "brian.market-gym.v1"
TRAINING_EVIDENCE_CLASS = "TRAINING_ONLY"


def _canonical_hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GymBar:
    asset_id: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    source_timestamp: float | None = None
    tradable: bool = True

    def __post_init__(self) -> None:
        if not self.asset_id.strip() or not math.isfinite(float(self.timestamp)):
            raise ValueError("asset_id and finite timestamp are required")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("prices must be positive")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("invalid OHLC bar")
        if self.volume is not None and (not math.isfinite(float(self.volume)) or self.volume < 0):
            raise ValueError("volume must be finite and non-negative")
        if self.source_timestamp is not None and not math.isfinite(float(self.source_timestamp)):
            raise ValueError("source_timestamp must be finite")


@dataclass(frozen=True, slots=True)
class GymFrame:
    timestamp: float
    bars: tuple[GymBar, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.timestamp)) or not self.bars:
            raise ValueError("frame requires finite timestamp and bars")
        ids = [bar.asset_id for bar in self.bars]
        if len(ids) != len(set(ids)):
            raise ValueError("frame cannot contain duplicate assets")
        if any(abs(float(bar.timestamp) - float(self.timestamp)) > 1e-9 for bar in self.bars):
            raise ValueError("all bars must use the frame timestamp")

    def by_asset(self) -> dict[str, GymBar]:
        return {bar.asset_id: bar for bar in self.bars}


@dataclass(frozen=True, slots=True)
class TargetAllocation:
    """Signed target portfolio weights. Gross exposure is checked by MarketGym."""

    weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        ids = [asset for asset, _ in self.weights]
        if len(ids) != len(set(ids)):
            raise ValueError("allocation cannot contain duplicate assets")
        for asset, weight in self.weights:
            if not asset.strip() or not math.isfinite(float(weight)):
                raise ValueError("allocation weights must be finite")

    @classmethod
    def from_mapping(cls, weights: Mapping[str, float]) -> "TargetAllocation":
        cleaned = tuple(sorted((str(asset), float(weight)) for asset, weight in weights.items() if abs(float(weight)) > 1e-15))
        return cls(cleaned)

    def as_dict(self) -> dict[str, float]:
        return dict(self.weights)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(float(weight)) for _, weight in self.weights)


@dataclass(frozen=True, slots=True)
class MarketGymConfig:
    starting_equity: float = 500.0
    fee_bps: float = 10.0
    assumed_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    max_gross_exposure: float = 1.0
    max_asset_weight: float = 0.35
    ruin_fraction: float = 0.01
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if min(self.fee_bps, self.assumed_spread_bps, self.slippage_bps) < 0:
            raise ValueError("cost parameters must be non-negative")
        if not 0 < self.max_gross_exposure <= 1:
            raise ValueError("gym is unlevered; max_gross_exposure must be in (0,1]")
        if not 0 < self.max_asset_weight <= self.max_gross_exposure:
            raise ValueError("invalid max_asset_weight")
        if not 0 <= self.ruin_fraction < 1:
            raise ValueError("ruin_fraction must be in [0,1)")

    @property
    def one_way_cost_rate(self) -> float:
        return (self.fee_bps + self.assumed_spread_bps / 2.0 + self.slippage_bps) / 10_000.0


@dataclass(frozen=True, slots=True)
class GymStep:
    observation_timestamp: float
    execution_timestamp: float
    equity_before: float
    equity_after_overnight: float
    equity_after_costs: float
    equity_after: float
    target_weights: tuple[tuple[str, float], ...]
    ending_weights: tuple[tuple[str, float], ...]
    turnover_notional: float
    trading_cost: float
    overnight_pnl: float
    intrabar_pnl: float
    drawdown_pct: float
    terminal: bool
    terminal_reason: str | None
    shadow_only: bool = True
    evidence_class: str = TRAINING_EVIDENCE_CLASS
    schema_version: str = GYM_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class GymEpisodeResult:
    episode_id: str
    starting_equity: float
    ending_equity: float
    return_pct: float
    max_drawdown_pct: float
    total_turnover: float
    total_costs: float
    rebalance_count: int
    steps: int
    ruined: bool
    terminal_reason: str
    final_weights: tuple[tuple[str, float], ...]
    trace: tuple[GymStep, ...]
    shadow_only: bool = True
    evidence_class: str = TRAINING_EVIDENCE_CLASS
    schema_version: str = GYM_SCHEMA_VERSION

    def compact_manifest(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("trace", None)
        return payload


class MarketGym:
    """Deterministic multi-asset training environment with next-frame-open execution.

    This class deliberately has no broker/exchange execution methods. Actions are target
    weights inside an unlevered virtual portfolio.
    """

    def __init__(self, frames: Sequence[GymFrame], config: MarketGymConfig = MarketGymConfig()) -> None:
        if len(frames) < 2:
            raise ValueError("market gym requires at least two frames")
        self.frames = tuple(frames)
        if any(right.timestamp <= left.timestamp for left, right in zip(self.frames, self.frames[1:])):
            raise ValueError("gym frames must be strictly chronological")
        self.config = config
        self.reset()

    def reset(self) -> GymFrame:
        self.index = 0
        self.equity = float(self.config.starting_equity)
        self.weights: dict[str, float] = {}
        self.peak_equity = float(self.config.starting_equity)
        self.max_drawdown_pct = 0.0
        self.total_turnover = 0.0
        self.total_costs = 0.0
        self.rebalance_count = 0
        self.trace: list[GymStep] = []
        self.terminated = False
        self.terminal_reason: str | None = None
        return self.frames[0]

    @property
    def observation(self) -> GymFrame:
        return self.frames[self.index]

    def _validate_allocation(self, allocation: TargetAllocation, observation: GymFrame) -> dict[str, float]:
        weights = allocation.as_dict()
        available = {asset for asset, bar in observation.by_asset().items() if bar.tradable}
        if any(asset not in available for asset in weights):
            raise ValueError("allocation may use only assets tradable in the observed frame")
        if not self.config.allow_short and any(weight < -1e-15 for weight in weights.values()):
            raise ValueError("short allocations are disabled")
        if any(abs(weight) > self.config.max_asset_weight + 1e-12 for weight in weights.values()):
            raise ValueError("target asset weight exceeds concentration limit")
        if sum(abs(weight) for weight in weights.values()) > self.config.max_gross_exposure + 1e-12:
            raise ValueError("target gross exposure exceeds unlevered limit")
        return weights

    @staticmethod
    def _portfolio_return(weights: Mapping[str, float], before: Mapping[str, GymBar], after: Mapping[str, GymBar], *, leg: str) -> float:
        total = 0.0
        for asset, weight in weights.items():
            if abs(weight) <= 1e-15:
                continue
            left = before.get(asset)
            right = after.get(asset)
            if left is None or right is None or not left.tradable or not right.tradable:
                raise KeyError(asset)
            if leg == "overnight":
                change = right.open / left.close - 1.0
            elif leg == "intrabar":
                change = right.close / right.open - 1.0
            else:
                raise ValueError("unknown return leg")
            total += float(weight) * change
        return total

    @staticmethod
    def _drift_weights(target: Mapping[str, float], next_bars: Mapping[str, GymBar], equity_before_leg: float, equity_after: float) -> dict[str, float]:
        if equity_after <= 0:
            return {}
        out: dict[str, float] = {}
        for asset, weight in target.items():
            if abs(weight) <= 1e-15:
                continue
            bar = next_bars[asset]
            ending_signed_exposure = float(weight) * equity_before_leg * (bar.close / bar.open)
            out[asset] = ending_signed_exposure / equity_after
        return out

    def _terminate_gap(self, current: GymFrame, nxt: GymFrame, asset: str) -> GymStep:
        self.terminated = True
        self.terminal_reason = f"DATA_GAP:{asset}"
        step = GymStep(
            current.timestamp, nxt.timestamp, self.equity, self.equity, self.equity, self.equity,
            tuple(sorted(self.weights.items())), tuple(sorted(self.weights.items())),
            0.0, 0.0, 0.0, 0.0, self.max_drawdown_pct, True, self.terminal_reason,
        )
        self.trace.append(step)
        self.index += 1
        return step

    def step(self, allocation: TargetAllocation) -> GymStep:
        if self.terminated:
            raise RuntimeError("episode is terminated; call reset for a new $500 life")
        if self.index + 1 >= len(self.frames):
            raise RuntimeError("episode has no future frame")
        current = self.frames[self.index]
        nxt = self.frames[self.index + 1]
        target = self._validate_allocation(allocation, current)
        current_bars = current.by_asset()
        next_bars = nxt.by_asset()

        required = set(self.weights) | set(target)
        for asset in sorted(required):
            if asset not in current_bars or asset not in next_bars or not current_bars[asset].tradable or not next_bars[asset].tradable:
                return self._terminate_gap(current, nxt, asset)

        equity_before = self.equity
        try:
            overnight_return = self._portfolio_return(self.weights, current_bars, next_bars, leg="overnight")
        except KeyError as exc:
            return self._terminate_gap(current, nxt, str(exc.args[0]))
        overnight_pnl = equity_before * overnight_return
        after_overnight = equity_before + overnight_pnl
        if after_overnight <= 0:
            after_overnight = 0.0

        turnover_fraction = sum(abs(target.get(asset, 0.0) - self.weights.get(asset, 0.0)) for asset in set(target) | set(self.weights))
        turnover_notional = max(0.0, after_overnight) * turnover_fraction
        trading_cost = turnover_notional * self.config.one_way_cost_rate
        after_costs = max(0.0, after_overnight - trading_cost)
        if turnover_notional > 1e-12:
            self.rebalance_count += 1
        self.total_turnover += turnover_notional
        self.total_costs += trading_cost

        try:
            intrabar_return = self._portfolio_return(target, current_bars, next_bars, leg="intrabar")
        except KeyError as exc:
            return self._terminate_gap(current, nxt, str(exc.args[0]))
        intrabar_pnl = after_costs * intrabar_return
        equity_after = max(0.0, after_costs + intrabar_pnl)
        ending_weights = self._drift_weights(target, next_bars, after_costs, equity_after)

        self.equity = equity_after
        self.weights = ending_weights
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown_pct = 100.0 * max(0.0, self.peak_equity - self.equity) / max(self.peak_equity, 1e-12)
        self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown_pct)
        self.index += 1

        ruin_level = self.config.starting_equity * self.config.ruin_fraction
        terminal_reason: str | None = None
        if self.equity <= ruin_level + 1e-12:
            self.terminated = True
            terminal_reason = "RUIN"
        elif self.index + 1 >= len(self.frames):
            self.terminated = True
            terminal_reason = "PATH_END"
        self.terminal_reason = terminal_reason

        result = GymStep(
            current.timestamp, nxt.timestamp, equity_before, after_overnight, after_costs,
            self.equity, tuple(sorted(target.items())), tuple(sorted(self.weights.items())),
            turnover_notional, trading_cost, overnight_pnl, intrabar_pnl, drawdown_pct,
            self.terminated, terminal_reason,
        )
        self.trace.append(result)
        return result

    def finish(self) -> GymEpisodeResult:
        if not self.trace:
            raise RuntimeError("episode has no steps")
        reason = self.terminal_reason or "MANUAL_END"
        payload = {
            "schema_version": GYM_SCHEMA_VERSION,
            "starting_equity": self.config.starting_equity,
            "ending_equity": self.equity,
            "steps": len(self.trace),
            "reason": reason,
            "trace_hash": _canonical_hash([asdict(step) for step in self.trace]),
        }
        episode_id = _canonical_hash(payload)
        return GymEpisodeResult(
            episode_id=episode_id,
            starting_equity=self.config.starting_equity,
            ending_equity=self.equity,
            return_pct=(self.equity / self.config.starting_equity - 1.0) * 100.0,
            max_drawdown_pct=self.max_drawdown_pct,
            total_turnover=self.total_turnover,
            total_costs=self.total_costs,
            rebalance_count=self.rebalance_count,
            steps=len(self.trace),
            ruined=reason == "RUIN",
            terminal_reason=reason,
            final_weights=tuple(sorted(self.weights.items())),
            trace=tuple(self.trace),
        )


def run_allocations(frames: Sequence[GymFrame], allocations: Sequence[TargetAllocation],
                    config: MarketGymConfig = MarketGymConfig()) -> GymEpisodeResult:
    gym = MarketGym(frames, config)
    for allocation in allocations:
        if gym.terminated:
            break
        gym.step(allocation)
    return gym.finish()
