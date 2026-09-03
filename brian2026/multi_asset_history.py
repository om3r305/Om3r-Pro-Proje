from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Literal, Mapping, Sequence
import json
import math
import re

from .archive import OFFICIAL_ARCHIVE_ROOT
from .portfolio import DEVELOPMENT_CUTOFF

MULTI_ASSET_HISTORY_SCHEMA_VERSION = "brian.multi-asset-history.v1"
AccessMode = Literal["PUBLIC_NO_KEY", "API_KEY_REQUIRED", "LICENSED_REQUIRED"]
RevisionModel = Literal["IMMUTABLE_MARKET", "REVISION_AWARE", "LATEST_ONLY", "LICENSED_VENDOR"]


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CurriculumArchiveSpec:
    """Official Binance Vision monthly spot archive spec for the broad curriculum.

    This deliberately does not change the older Phase 2.3 allow-list. Existing locked
    experiments keep their historical contract while Phase 3.3 can expand the universe.
    """

    symbol: str
    year: int
    month: int
    timeframe: str = "1m"
    frequency: Literal["monthly"] = "monthly"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z0-9]{2,20}USDT", self.symbol):
            raise ValueError("curriculum Binance symbol must be an explicit USDT spot symbol")
        if self.timeframe != "1m" or not 1 <= self.month <= 12:
            raise ValueError("curriculum archive supports monthly canonical 1m data")
        if datetime(self.year, self.month, 1, tzinfo=timezone.utc).timestamp() >= DEVELOPMENT_CUTOFF:
            raise ValueError("2026 historical development archives are INVALID_CONTAMINATED and forbidden")

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.timeframe}-{self.year:04d}-{self.month:02d}.zip"

    @property
    def url(self) -> str:
        return f"{OFFICIAL_ARCHIVE_ROOT}/spot/monthly/klines/{self.symbol}/{self.timeframe}/{self.filename}"

    @property
    def checksum_url(self) -> str:
        return self.url + ".CHECKSUM"

    @property
    def start(self) -> float:
        return datetime(self.year, self.month, 1, tzinfo=timezone.utc).timestamp()

    @property
    def end(self) -> float:
        year, month = (self.year + 1, 1) if self.month == 12 else (self.year, self.month + 1)
        return datetime(year, month, 1, tzinfo=timezone.utc).timestamp()


@dataclass(frozen=True, slots=True)
class HistoricalSeriesSpec:
    canonical_id: str
    asset_class: str
    provider: str
    provider_series_id: str
    native_frequency: str
    timezone: str
    trading_calendar: str
    access_mode: AccessMode
    revision_model: RevisionModel
    provenance_uri: str
    earliest_known_date: str | None = None
    is_proxy: bool = False
    proxy_for: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.canonical_id, self.asset_class, self.provider, self.provider_series_id,
            self.native_frequency, self.timezone, self.trading_calendar, self.provenance_uri,
        )
        if any(not str(value).strip() for value in required):
            raise ValueError("historical series metadata must be explicit")
        if self.is_proxy != bool(self.proxy_for):
            raise ValueError("proxy status and proxy_for must agree")
        if self.revision_model == "REVISION_AWARE" and self.provider not in {"fred_alfred", "ecb_sdmx"}:
            raise ValueError("revision-aware series needs a provider with historical revisions")


# Initial broad curriculum seed. This is a research universe, not a recommendation list.
CRYPTO_CURRICULUM_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
    "DOGEUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT",
    "BCHUSDT", "ETCUSDT", "XLMUSDT", "UNIUSDT", "AAVEUSDT", "NEARUSDT",
    "ATOMUSDT", "FILUSDT", "ALGOUSDT", "ARBUSDT", "SUIUSDT",
    # Fan-token curriculum: source availability is discovered month-by-month; missing
    # periods remain PRE_LISTING/SOURCE_MISSING rather than fabricated history.
    "BARUSDT", "PSGUSDT", "CITYUSDT", "PORTOUSDT", "LAZIOUSDT", "SANTOSUSDT", "ALPINEUSDT",
)


OFFICIAL_CROSS_MARKET_SERIES: tuple[HistoricalSeriesSpec, ...] = (
    HistoricalSeriesSpec(
        "fx-usd-eur", "fx", "ecb_sdmx", "EXR:D.USD.EUR.SP00.A", "1d", "Europe/Frankfurt",
        "ECB_WORKING_DAYS", "PUBLIC_NO_KEY", "REVISION_AWARE",
        "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A", "1999-01-04",
    ),
    HistoricalSeriesSpec(
        "fx-gbp-eur", "fx", "ecb_sdmx", "EXR:D.GBP.EUR.SP00.A", "1d", "Europe/Frankfurt",
        "ECB_WORKING_DAYS", "PUBLIC_NO_KEY", "REVISION_AWARE",
        "https://data-api.ecb.europa.eu/service/data/EXR/D.GBP.EUR.SP00.A", "1999-01-04",
    ),
    HistoricalSeriesSpec(
        "fx-jpy-eur", "fx", "ecb_sdmx", "EXR:D.JPY.EUR.SP00.A", "1d", "Europe/Frankfurt",
        "ECB_WORKING_DAYS", "PUBLIC_NO_KEY", "REVISION_AWARE",
        "https://data-api.ecb.europa.eu/service/data/EXR/D.JPY.EUR.SP00.A", "1999-01-04",
    ),
    HistoricalSeriesSpec(
        "fx-chf-eur", "fx", "ecb_sdmx", "EXR:D.CHF.EUR.SP00.A", "1d", "Europe/Frankfurt",
        "ECB_WORKING_DAYS", "PUBLIC_NO_KEY", "REVISION_AWARE",
        "https://data-api.ecb.europa.eu/service/data/EXR/D.CHF.EUR.SP00.A", "1999-01-04",
    ),
    HistoricalSeriesSpec(
        "oil-wti-spot", "commodity", "eia", "PET.RWTC.D", "1d", "America/New_York",
        "EIA_PETROLEUM", "API_KEY_REQUIRED", "LATEST_ONLY",
        "https://www.eia.gov/opendata/browser/petroleum/pri/spt", "1986-01-02",
    ),
    HistoricalSeriesSpec(
        "oil-brent-spot", "commodity", "eia", "PET.RBRTE.D", "1d", "Europe/London",
        "EIA_PETROLEUM", "API_KEY_REQUIRED", "LATEST_ONLY",
        "https://www.eia.gov/opendata/browser/petroleum/pri/spt", "1987-05-20",
    ),
    HistoricalSeriesSpec(
        "macro-fed-funds-effective", "rates", "fred_alfred", "DFF", "1d", "America/New_York",
        "US_FEDERAL", "API_KEY_REQUIRED", "REVISION_AWARE",
        "https://fred.stlouisfed.org/series/DFF",
    ),
    HistoricalSeriesSpec(
        "macro-us-10y-yield", "rates", "fred_alfred", "DGS10", "1d", "America/New_York",
        "US_FEDERAL", "API_KEY_REQUIRED", "REVISION_AWARE",
        "https://fred.stlouisfed.org/series/DGS10",
    ),
    HistoricalSeriesSpec(
        "macro-us-cpi", "macro", "fred_alfred", "CPIAUCSL", "1mo", "America/New_York",
        "US_MACRO_RELEASE", "API_KEY_REQUIRED", "REVISION_AWARE",
        "https://fred.stlouisfed.org/series/CPIAUCSL",
    ),
    HistoricalSeriesSpec(
        "macro-us-unemployment", "macro", "fred_alfred", "UNRATE", "1mo", "America/New_York",
        "US_MACRO_RELEASE", "API_KEY_REQUIRED", "REVISION_AWARE",
        "https://fred.stlouisfed.org/series/UNRATE",
    ),
    # Broad equity history is intentionally not filled from an unverified scraper.
    HistoricalSeriesSpec(
        "equity-us-broad-universe", "equity", "licensed_equity_vendor", "TBD_LICENSED_VENDOR",
        "1d", "America/New_York", "XNYS_XNAS", "LICENSED_REQUIRED", "LICENSED_VENDOR",
        "licensed://equity-provider-required",
    ),
    HistoricalSeriesSpec(
        "gold-tradable-history", "commodity", "licensed_commodity_vendor", "TBD_GOLD_SERIES",
        "1d", "UTC", "COMMODITY_VENDOR_CALENDAR", "LICENSED_REQUIRED", "LICENSED_VENDOR",
        "licensed://gold-provider-required",
    ),
)


@dataclass(frozen=True, slots=True)
class PointInTimeValue:
    series_id: str
    observation_time: float
    available_at: float
    value: float
    vintage_id: str | None = None

    def __post_init__(self) -> None:
        if not self.series_id.strip() or not all(math.isfinite(float(x)) for x in (self.observation_time, self.available_at, self.value)):
            raise ValueError("invalid point-in-time observation")
        if self.available_at < self.observation_time:
            raise ValueError("value cannot become available before its observation timestamp")


@dataclass(frozen=True, slots=True)
class AsOfValue:
    series_id: str
    decision_time: float
    source_observation_time: float
    source_available_at: float
    value: float
    age_seconds: float
    vintage_id: str | None


def asof_value(values: Sequence[PointInTimeValue], *, decision_time: float) -> AsOfValue | None:
    """Return only information that was actually available by decision_time.

    For revision-aware series, if several vintages describe the same/older observation,
    the latest *already available* vintage wins. Future revisions are invisible.
    """
    if not math.isfinite(float(decision_time)):
        raise ValueError("decision_time must be finite")
    eligible = [row for row in values if row.available_at <= decision_time]
    if not eligible:
        return None
    row = max(eligible, key=lambda item: (item.observation_time, item.available_at, item.vintage_id or ""))
    return AsOfValue(
        row.series_id, decision_time, row.observation_time, row.available_at, row.value,
        max(0.0, decision_time - row.available_at), row.vintage_id,
    )


@dataclass(frozen=True, slots=True)
class CurriculumHistoryManifest:
    historical_cutoff_exclusive: float
    crypto_symbols: tuple[str, ...]
    cross_market_series: tuple[HistoricalSeriesSpec, ...]
    allow_fake_intraday_upsampling: bool = False
    final_holdout_evaluated: bool = False
    training_only: bool = True
    schema_version: str = MULTI_ASSET_HISTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.historical_cutoff_exclusive > DEVELOPMENT_CUTOFF:
            raise ValueError("historical curriculum cannot cross the contaminated 2026 cutoff")
        if self.allow_fake_intraday_upsampling:
            raise ValueError("daily/monthly data must never be fabricated into intraday bars")
        if self.final_holdout_evaluated:
            raise ValueError("curriculum history cannot be a pristine final holdout")

    @property
    def manifest_id(self) -> str:
        return _hash({
            "historical_cutoff_exclusive": self.historical_cutoff_exclusive,
            "crypto_symbols": self.crypto_symbols,
            "cross_market_series": [asdict(row) for row in self.cross_market_series],
            "allow_fake_intraday_upsampling": self.allow_fake_intraday_upsampling,
            "final_holdout_evaluated": self.final_holdout_evaluated,
            "training_only": self.training_only,
            "schema_version": self.schema_version,
        })


def default_curriculum_history_manifest() -> CurriculumHistoryManifest:
    return CurriculumHistoryManifest(
        historical_cutoff_exclusive=DEVELOPMENT_CUTOFF,
        crypto_symbols=CRYPTO_CURRICULUM_SYMBOLS,
        cross_market_series=OFFICIAL_CROSS_MARKET_SERIES,
    )


def monthly_crypto_specs(symbol: str, *, start: datetime, end: datetime) -> tuple[CurriculumArchiveSpec, ...]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("history planning timestamps must be timezone-aware")
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    finish = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    items: list[CurriculumArchiveSpec] = []
    while cursor < finish:
        items.append(CurriculumArchiveSpec(symbol, cursor.year, cursor.month))
        cursor = datetime(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1, tzinfo=timezone.utc)
    return tuple(items)


def provider_requirements(series: Sequence[HistoricalSeriesSpec] = OFFICIAL_CROSS_MARKET_SERIES) -> Mapping[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {"PUBLIC_NO_KEY": [], "API_KEY_REQUIRED": [], "LICENSED_REQUIRED": []}
    for row in series:
        groups[row.access_mode].append(row.canonical_id)
    return {key: tuple(sorted(value)) for key, value in groups.items()}
