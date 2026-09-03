from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import math


def _finite(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    quote_asset: str = "USDT"
    min_quote_volume: float = 5_000_000.0
    min_trades_24h: int = 1_000
    min_price: float = 1e-8
    top_n: int = 50
    max_abs_change_pct: float = 200.0
    excluded_base_assets: tuple[str, ...] = (
        "USDT", "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "EUR", "TRY",
    )

    def __post_init__(self) -> None:
        if not self.quote_asset.strip() or self.min_quote_volume < 0 or self.min_trades_24h < 0:
            raise ValueError("invalid universe configuration")
        if self.min_price <= 0 or self.top_n <= 0 or self.max_abs_change_pct <= 0:
            raise ValueError("invalid universe thresholds")


@dataclass(frozen=True, slots=True)
class MarketUniverseRow:
    symbol: str
    base_asset: str
    quote_asset: str
    last_price: float
    quote_volume: float
    trades_24h: int
    price_change_pct: float
    high_price: float
    low_price: float
    bid_price: float | None = None
    ask_price: float | None = None
    spot_trading_allowed: bool = True
    schema_version: str = "brian.market-universe-row.v1"

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.base_asset.strip() or not self.quote_asset.strip():
            raise ValueError("symbol/base/quote are required")
        if self.last_price <= 0 or self.quote_volume < 0 or self.trades_24h < 0:
            raise ValueError("invalid universe market values")
        if self.high_price < self.low_price or self.low_price < 0:
            raise ValueError("invalid 24h range")
        if self.bid_price is not None and self.bid_price < 0:
            raise ValueError("bid cannot be negative")
        if self.ask_price is not None and self.ask_price < 0:
            raise ValueError("ask cannot be negative")

    @property
    def range_pct(self) -> float:
        return 100.0 * max(0.0, self.high_price - self.low_price) / max(self.last_price, 1e-12)

    @property
    def spread_bps(self) -> float | None:
        if self.bid_price is None or self.ask_price is None or self.bid_price <= 0 or self.ask_price <= 0:
            return None
        mid = (self.bid_price + self.ask_price) / 2.0
        return 10_000.0 * max(0.0, self.ask_price - self.bid_price) / max(mid, 1e-12)


@dataclass(frozen=True, slots=True)
class UniverseCandidate:
    symbol: str
    base_asset: str
    liquidity_score: float
    activity_score: float
    volatility_score: float
    momentum_score: float
    spread_quality: float
    radar_score: float
    quote_volume: float
    trades_24h: int
    price_change_pct: float
    range_pct: float
    spread_bps: float | None
    reasons: tuple[str, ...]
    schema_version: str = "brian.universe-candidate.v1"


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    observed_at: float
    candidates: tuple[UniverseCandidate, ...]
    eligible_symbols: tuple[str, ...]
    rejected_count: int
    config: Mapping[str, object]
    source: str = "binance_public"
    schema_version: str = "brian.universe-snapshot.v1"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.observed_at)):
            raise ValueError("observed_at must be finite")


def _rank_percentiles(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    out = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    for rank, (index, _) in enumerate(indexed):
        out[index] = rank / denominator
    return out


def build_universe_snapshot(
    rows: Sequence[MarketUniverseRow],
    *,
    observed_at: float,
    config: UniverseConfig = UniverseConfig(),
) -> UniverseSnapshot:
    if not math.isfinite(float(observed_at)):
        raise ValueError("observed_at must be finite")
    excluded = {x.upper() for x in config.excluded_base_assets}
    eligible: list[MarketUniverseRow] = []
    for row in rows:
        if not row.spot_trading_allowed:
            continue
        if row.quote_asset.upper() != config.quote_asset.upper():
            continue
        if row.base_asset.upper() in excluded:
            continue
        if row.quote_volume < config.min_quote_volume or row.trades_24h < config.min_trades_24h:
            continue
        if row.last_price < config.min_price:
            continue
        eligible.append(row)

    if not eligible:
        return UniverseSnapshot(float(observed_at), (), (), len(rows), asdict(config))

    liquidity = _rank_percentiles([math.log1p(row.quote_volume) for row in eligible])
    activity = _rank_percentiles([math.log1p(row.trades_24h) for row in eligible])
    volatility = _rank_percentiles([row.range_pct for row in eligible])
    momentum = _rank_percentiles([
        min(config.max_abs_change_pct, abs(row.price_change_pct)) for row in eligible
    ])

    candidates: list[UniverseCandidate] = []
    for index, row in enumerate(eligible):
        spread = row.spread_bps
        # Missing spread is explicitly neutral rather than magically perfect.
        spread_quality = 0.50 if spread is None else 1.0 / (1.0 + max(0.0, spread) / 10.0)
        score = (
            0.34 * liquidity[index] + 0.20 * activity[index] +
            0.20 * volatility[index] + 0.16 * momentum[index] +
            0.10 * spread_quality
        )
        reasons: list[str] = []
        if liquidity[index] >= 0.80:
            reasons.append("high relative liquidity")
        if activity[index] >= 0.80:
            reasons.append("high trade activity")
        if volatility[index] >= 0.80:
            reasons.append("elevated 24h range")
        if momentum[index] >= 0.80:
            reasons.append("large absolute 24h move")
        if spread is not None and spread <= 5.0:
            reasons.append("tight top-of-book spread")
        candidates.append(UniverseCandidate(
            row.symbol, row.base_asset,
            _clip01(liquidity[index]), _clip01(activity[index]), _clip01(volatility[index]),
            _clip01(momentum[index]), _clip01(spread_quality), _clip01(score),
            row.quote_volume, row.trades_24h, row.price_change_pct, row.range_pct,
            spread, tuple(reasons),
        ))

    candidates.sort(key=lambda row: (-row.radar_score, -row.quote_volume, row.symbol))
    selected = tuple(candidates[: config.top_n])
    return UniverseSnapshot(
        float(observed_at), selected, tuple(sorted(row.symbol for row in eligible)),
        len(rows) - len(eligible), asdict(config),
    )


@dataclass(frozen=True, slots=True)
class UniverseDelta:
    observed_at: float
    newly_observed_symbols: tuple[str, ...]
    disappeared_symbols: tuple[str, ...]
    comparable: bool
    schema_version: str = "brian.universe-delta.v1"


def compare_universe(previous: UniverseSnapshot | None, current: UniverseSnapshot) -> UniverseDelta:
    if previous is None:
        # First snapshot is a baseline, not a listing alert.
        return UniverseDelta(current.observed_at, (), (), False)
    if current.observed_at <= previous.observed_at:
        raise ValueError("universe snapshots must be compared chronologically")
    before = set(previous.eligible_symbols)
    after = set(current.eligible_symbols)
    return UniverseDelta(
        current.observed_at,
        tuple(sorted(after - before)),
        tuple(sorted(before - after)),
        True,
    )
