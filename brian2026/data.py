from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Sequence
import argparse
import json
import time

import requests

from .dataset import ClosedCandle, MarketDataset, MarketEvent, _timeframe_seconds
from .features import FeatureSnapshot, from_closed_candles

RAW_SCHEMA_VERSION = "brian.raw-kline.v1"
NORMALIZED_SCHEMA_VERSION = "brian.normalized-kline.v1"
QUALITY_SCHEMA_VERSION = "brian.data-quality.v1"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_TIMEFRAMES = ("1m", "5m", "15m", "1h")


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def utc_timestamp(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include UTC offset")
    return parsed.astimezone(timezone.utc).timestamp()


@dataclass(frozen=True, slots=True)
class Instrument:
    exchange: str
    market_type: Literal["spot", "perpetual"]
    symbol: str
    base_asset: str
    quote_asset: str

    def __post_init__(self) -> None:
        if self.exchange != "binance" or not self.symbol or not self.base_asset or not self.quote_asset:
            raise ValueError("invalid instrument metadata")


@dataclass(frozen=True, slots=True)
class RawKline:
    values: tuple[str, ...]
    source_endpoint: str
    retrieval_timestamp: float
    timestamp_unit: Literal["ms", "us"] = "ms"
    schema_version: str = RAW_SCHEMA_VERSION

    @property
    def raw_id(self) -> str:
        payload = asdict(self)
        payload.pop("retrieval_timestamp", None)
        return canonical_hash(payload)


@dataclass(frozen=True, slots=True)
class RawKlineBatch:
    instrument: Instrument
    timeframe: str
    requested_start: float
    requested_end: float
    records: tuple[RawKline, ...]
    schema_version: str = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_start", float(self.requested_start))
        object.__setattr__(self, "requested_end", float(self.requested_end))

    @property
    def raw_hash(self) -> str:
        payload = asdict(self)
        # Retrieval time describes this import, not source content identity.
        for record in payload["records"]:
            record.pop("retrieval_timestamp", None)
        return canonical_hash(payload)

    def write(self, root: str | Path) -> Path:
        path = Path(root) / "raw" / self.raw_hash[:2] / f"{self.raw_hash}.json"
        payload = asdict(self)
        for record in payload["records"]:
            record.pop("retrieval_timestamp", None)
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        if path.exists():
            if path.read_bytes() != content:
                raise FileExistsError(f"immutable raw object collision: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        import_record = {"raw_hash": self.raw_hash,
                         "retrieval_timestamps": [record.retrieval_timestamp for record in self.records],
                         "source_endpoints": sorted({record.source_endpoint for record in self.records}),
                         "provenance_type": "offline_public_api_retrieval"}
        import_id = canonical_hash(import_record)
        manifest = Path(root) / "imports" / self.raw_hash / f"{import_id}.json"
        manifest_content = json.dumps(import_record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if manifest.exists() and manifest.read_bytes() != manifest_content:
            raise FileExistsError(f"immutable import manifest collision: {manifest}")
        if not manifest.exists():
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_bytes(manifest_content)
        return path


@dataclass(frozen=True, slots=True)
class NormalizedKline:
    instrument: Instrument
    timeframe: str
    exchange_open_timestamp: float
    exchange_close_timestamp: float
    source_event_timestamp: float
    observed_timestamp: float
    ingestion_timestamp_type: Literal["offline_close_bound", "live_observed"]
    open: float
    high: float
    low: float
    close: float
    volume: float
    raw_id: str
    source_endpoint: str
    bid: float | None = None
    ask: float | None = None
    order_book_available: bool = False
    funding_rate: float | None = None
    schema_version: str = NORMALIZED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.exchange_close_timestamp > self.source_event_timestamp:
            raise ValueError("source event precedes candle close")
        if self.observed_timestamp < self.exchange_close_timestamp:
            raise ValueError("observation cannot precede candle close")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close) or self.high < self.low:
            raise ValueError("invalid OHLC")
        if self.volume < 0 or (self.bid is None) != (self.ask is None):
            raise ValueError("invalid market values")


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    observations: int
    first_timestamp: float | None
    last_timestamp: float | None
    duplicate_count: int
    missing_interval_count: int
    largest_gap_seconds: float
    incomplete_rejected: int
    future_rejected: int
    invalid_ohlc_rejected: int
    timestamp_monotonic: bool
    missing_bid_ask_pct: float
    missing_order_book_pct: float
    missing_funding_pct: float
    raw_hash: str
    dataset_hash: str | None
    status: Literal["READY", "WARNING", "REJECTED"]
    schema_version: str = QUALITY_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    records: tuple[NormalizedKline, ...]
    report: DataQualityReport


class HttpResponse(Protocol):
    status_code: int
    headers: dict[str, str]
    def json(self) -> Any: ...
    def raise_for_status(self) -> None: ...


class BinancePublicKlineAdapter:
    """Unauthenticated spot OHLCV adapter. It contains no order methods."""
    endpoint = "https://data-api.binance.vision/api/v3/klines"

    def __init__(self, session: Any | None = None, retries: int = 4,
                 backoff_seconds: float = 0.5, timeout: float = 20.0) -> None:
        self.session = session or requests.Session()
        self.retries, self.backoff_seconds, self.timeout = retries, backoff_seconds, timeout

    def _get(self, params: dict[str, Any]) -> HttpResponse:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(self.endpoint, params=params, timeout=self.timeout)
                if response.status_code in (418, 429) or response.status_code >= 500:
                    retry_after = float(response.headers.get("Retry-After", 0) or 0)
                    if attempt == self.retries:
                        response.raise_for_status()
                    time.sleep(max(retry_after, self.backoff_seconds * (2 ** attempt)))
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last = exc
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff_seconds * (2 ** attempt))
        raise RuntimeError("public market data request failed") from last

    def fetch(self, instrument: Instrument, timeframe: str, start: float, end: float,
              *, now: float | None = None, limit: int = 1000) -> RawKlineBatch:
        if instrument.market_type != "spot":
            raise ValueError("spot adapter cannot fetch non-spot instruments")
        if timeframe not in DEFAULT_TIMEFRAMES or start >= end or limit < 1 or limit > 1000:
            raise ValueError("invalid fetch range/timeframe/limit")
        now = time.time() if now is None else now
        cursor_ms, end_ms = int(start * 1000), int(end * 1000)
        duration_ms = _timeframe_seconds(timeframe) * 1000
        raw: list[RawKline] = []
        while cursor_ms < end_ms:
            response = self._get({"symbol": instrument.symbol, "interval": timeframe,
                                  "startTime": cursor_ms, "endTime": end_ms - 1, "limit": limit})
            page = response.json()
            if not isinstance(page, list):
                raise ValueError("unexpected Binance response")
            retrieval = time.time()
            accepted = [row for row in page if len(row) >= 7 and int(row[0]) < end_ms]
            raw.extend(RawKline(tuple(str(value) for value in row), self.endpoint, retrieval, "ms") for row in accepted)
            if not accepted:
                break
            next_cursor = int(accepted[-1][0]) + duration_ms
            if next_cursor <= cursor_ms:
                raise ValueError("pagination did not advance")
            cursor_ms = next_cursor
            if len(page) < limit:
                break
        return RawKlineBatch(instrument, timeframe, start, end, tuple(raw))


def normalize(batch: RawKlineBatch, *, now: float | None = None) -> NormalizationResult:
    now = time.time() if now is None else now
    scale = 1_000.0
    candidates: list[NormalizedKline] = []
    incomplete = future = invalid = 0
    opens_seen: set[float] = set()
    duplicates = 0
    for raw in batch.records:
        try:
            unit_scale = 1_000_000.0 if raw.timestamp_unit == "us" else scale
            open_ts = int(raw.values[0]) / unit_scale
            close_ts = int(raw.values[6]) / unit_scale
            if open_ts in opens_seen:
                duplicates += 1
                continue
            opens_seen.add(open_ts)
            if open_ts > now or close_ts > now:
                future += 1
                continue
            if close_ts >= now or close_ts >= batch.requested_end:
                incomplete += 1
                continue
            record = NormalizedKline(batch.instrument, batch.timeframe, open_ts, close_ts,
                                     close_ts, close_ts, "offline_close_bound",
                                     float(raw.values[1]), float(raw.values[2]), float(raw.values[3]),
                                     float(raw.values[4]), float(raw.values[5]), raw.raw_id, raw.source_endpoint)
            candidates.append(record)
        except (ValueError, IndexError):
            invalid += 1
    ordered = tuple(sorted(candidates, key=lambda row: row.exchange_open_timestamp))
    duration = _timeframe_seconds(batch.timeframe)
    gaps = [max(0.0, ordered[i].exchange_open_timestamp - ordered[i-1].exchange_open_timestamp)
            for i in range(1, len(ordered))]
    missing = sum(max(0, round(gap / duration) - 1) for gap in gaps)
    monotonic = all(ordered[i].exchange_open_timestamp > ordered[i-1].exchange_open_timestamp
                    for i in range(1, len(ordered)))
    serious = duplicates > 0 or invalid > 0 or not monotonic
    status: Literal["READY", "WARNING", "REJECTED"] = "REJECTED" if serious else ("WARNING" if missing or incomplete or future else "READY")
    count = len(ordered)
    report = DataQualityReport(count, ordered[0].exchange_open_timestamp if count else None,
                               ordered[-1].exchange_close_timestamp if count else None, duplicates, missing,
                               max(gaps, default=0.0), incomplete, future, invalid, monotonic,
                               100.0 if count else 0.0, 100.0 if count else 0.0,
                               100.0 if count else 0.0,
                               batch.raw_hash, None, status)
    return NormalizationResult(ordered, report)


def build_market_dataset(batch: RawKlineBatch, normalized: NormalizationResult,
                         *, allow_quality_warning: bool = False) -> tuple[MarketDataset, DataQualityReport]:
    if normalized.report.status == "REJECTED" or (normalized.report.status == "WARNING" and not allow_quality_warning):
        raise ValueError(f"dataset is not training-ready: {normalized.report.status}")
    events = [MarketEvent(r.instrument.symbol, r.timeframe, r.source_event_timestamp,
                          r.observed_timestamp, ClosedCandle(r.exchange_open_timestamp,
                          r.exchange_close_timestamp, r.open, r.high, r.low, r.close, r.volume),
                          bid=r.bid, ask=r.ask,
                          sources=(("candles", r.source_endpoint), ("raw_hash", batch.raw_hash)))
              for r in normalized.records]
    metadata = tuple(sorted({"exchange": batch.instrument.exchange,
                             "market_type": batch.instrument.market_type,
                             "symbol": batch.instrument.symbol,
                             "base_asset": batch.instrument.base_asset,
                             "quote_asset": batch.instrument.quote_asset,
                             "timeframe": batch.timeframe,
                             "source": BinancePublicKlineAdapter.endpoint,
                             "requested_start": str(batch.requested_start),
                             "requested_end": str(batch.requested_end),
                             "actual_start": str(normalized.report.first_timestamp),
                             "actual_end": str(normalized.report.last_timestamp),
                             "ingestion_timestamp_type": "offline_close_bound",
                             "raw_hash": batch.raw_hash,
                             "quality_status": normalized.report.status}.items()))
    dataset = MarketDataset.from_events(events, metadata=metadata)
    return dataset, DataQualityReport(**{**asdict(normalized.report), "dataset_hash": dataset.dataset_id})


def build_feature_snapshots(dataset: MarketDataset) -> tuple[FeatureSnapshot, ...]:
    snapshots: list[FeatureSnapshot] = []
    history: list[tuple[float, float, float, float, float, float]] = []
    for event in dataset.events:
        candle = event.candle
        history.append((candle.open_timestamp * 1000, candle.open, candle.high, candle.low, candle.close, candle.volume))
        snapshots.append(from_closed_candles(symbol=event.symbol, price=candle.close,
                         regime=event.regime, candles=history, timeframe=event.timeframe,
                         timestamp=event.ingestion_timestamp, dataset_id=dataset.dataset_id,
                         source_timestamps={"closed_candle": candle.close_timestamp}))
    return tuple(snapshots)

@dataclass(frozen=True, slots=True)
class SimulationSpread:
    spread_bps: float
    provenance: str = "simulation_assumption"

    def __post_init__(self) -> None:
        if self.spread_bps < 0 or self.provenance != "simulation_assumption":
            raise ValueError("spread assumption must be explicit simulation metadata")


def replay_points(records: Sequence[NormalizedKline], spread: SimulationSpread):
    """Create replay quotes from an explicit assumption, never observed truth."""
    from .replay import ReplayPoint
    points = []
    for row in records:
        half = row.close * spread.spread_bps / 20_000.0
        points.append(ReplayPoint(row.source_event_timestamp, row.close - half, row.close + half,
                                  row.high, row.low, row.close))
    return tuple(points)
def read_raw(path: str | Path) -> RawKlineBatch:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    instrument = Instrument(**payload["instrument"])
    records = tuple(RawKline(tuple(record["values"]), record["source_endpoint"],
                             float(record.get("retrieval_timestamp", 0.0)),
                             record.get("timestamp_unit", "ms"),
                             record.get("schema_version", RAW_SCHEMA_VERSION))
                    for record in payload["records"])
    batch = RawKlineBatch(instrument, payload["timeframe"], payload["requested_start"],
                          payload["requested_end"], records,
                          payload.get("schema_version", RAW_SCHEMA_VERSION))
    expected = Path(path).stem
    if len(expected) == 64 and batch.raw_hash != expected:
        raise ValueError("raw content hash does not match filename")
    return batch

def _instrument(symbol: str, market_type: str) -> Instrument:
    quote = "USDT" if symbol.endswith("USDT") else "UNKNOWN"
    return Instrument("binance", market_type, symbol, symbol[:-len(quote)] if quote != "UNKNOWN" else symbol, quote)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brian 2026 public research data pipeline (never trading)")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--symbol", choices=DEFAULT_SYMBOLS, required=True)
    fetch.add_argument("--timeframe", choices=DEFAULT_TIMEFRAMES, required=True)
    fetch.add_argument("--start", required=True); fetch.add_argument("--end", required=True)
    fetch.add_argument("--output", default="research_data")
    inspect = sub.add_parser("inspect"); inspect.add_argument("path")
    validate = sub.add_parser("validate"); validate.add_argument("path")
    build = sub.add_parser("build-dataset"); build.add_argument("path"); build.add_argument("--output", default="research_data")
    build.add_argument("--allow-quality-warning", action="store_true")
    archive_fetch = sub.add_parser("archive-fetch")
    archive_fetch.add_argument("--symbol", choices=DEFAULT_SYMBOLS, required=True)
    archive_fetch.add_argument("--year", type=int, required=True); archive_fetch.add_argument("--month", type=int, required=True)
    archive_fetch.add_argument("--output", default="research_data"); archive_fetch.add_argument("--force", action="store_true")
    archive_plan = sub.add_parser("archive-plan"); archive_plan.add_argument("--symbol", choices=DEFAULT_SYMBOLS, required=True); archive_plan.add_argument("--start", required=True); archive_plan.add_argument("--end", required=True); archive_plan.add_argument("--root", default="research_data")
    archive_verify = sub.add_parser("archive-verify"); archive_verify.add_argument("manifest")
    archive_import = sub.add_parser("archive-import"); archive_import.add_argument("manifest")
    parquet = sub.add_parser("build-parquet"); parquet.add_argument("manifest"); parquet.add_argument("--output", default="research_data/parquet"); parquet.add_argument("--catalog-root", default="research_data")
    derive = sub.add_parser("derive-timeframes"); derive.add_argument("parquet"); derive.add_argument("--output", default="research_data/parquet"); derive.add_argument("--catalog-root", default="research_data")
    catalog_cmd = sub.add_parser("catalog"); catalog_cmd.add_argument("--root", default="research_data"); catalog_cmd.add_argument("--symbol"); catalog_cmd.add_argument("--timeframe")
    args = parser.parse_args(argv)
    if args.command.startswith("archive-") or args.command in ("build-parquet", "derive-timeframes", "catalog"):
        from .archive import (ArchiveSpec, BinanceArchiveAdapter, derive_timeframe, discover_first_available, plan_months,
                              load_manifest, parse_archive, spec_from_manifest)
        from .parquet_store import ParquetResearchStore
        if args.command == "archive-plan":
            from datetime import datetime, timezone
            adapter = BinanceArchiveAdapter()
            start = datetime.fromtimestamp(utc_timestamp(args.start), timezone.utc); end = datetime.fromtimestamp(utc_timestamp(args.end), timezone.utc)
            first = discover_first_available(args.symbol, start, end, adapter.exists)
            verified = set()
            manifest_root = Path(args.root) / "archive_manifests"
            if manifest_root.exists():
                for path in manifest_root.glob("*.json"):
                    item = json.loads(path.read_text(encoding="utf-8"))
                    if item.get("symbol") == args.symbol and item.get("checksum_verified"):
                        verified.add(Path(item["source_url"]).name)
            failed = set()
            failure_root = Path(args.root) / "archive_failures"
            if failure_root.exists():
                for path in failure_root.glob("*.json"):
                    item = json.loads(path.read_text(encoding="utf-8"))
                    failed.add(item["filename"])
            plan = plan_months(args.symbol, start, end, verified, adapter.exists, first, failed)
            print(json.dumps({"first_available": first, "items": [{"filename": p.spec.filename, "status": p.status} for p in plan]}, sort_keys=True)); return 0
        if args.command == "archive-fetch":
            manifest = BinanceArchiveAdapter().fetch(ArchiveSpec(args.symbol, "1m", args.year, args.month), args.output, force=args.force)
            print(json.dumps({"manifest_id": manifest.manifest_id, "content_hash": manifest.content_hash,
                              "verified": manifest.checksum_verified, "archive_path": manifest.archive_path}, sort_keys=True)); return 0
        if args.command == "catalog":
            from .research_catalog import ResearchCatalog
            print(json.dumps([asdict(r) for r in ResearchCatalog(args.root).search(symbol=args.symbol, timeframe=args.timeframe)], sort_keys=True)); return 0
        if args.command == "derive-timeframes":
            store = ParquetResearchStore(args.output); records, provenance, parent = store.read(args.parquet)
            output = {}
            for timeframe in ("5m", "15m", "1h"):
                derived = derive_timeframe(records, timeframe)
                if derived:
                    descriptor = store.write(derived, {**provenance, "derived_from": parent, "aggregation": timeframe})
                    from .research_catalog import ResearchCatalog
                    catalog_record = ResearchCatalog(args.catalog_root).add(
                        descriptor, exchange="binance", market_type="spot",
                        symbol=derived[0].instrument.symbol, raw_source_hashes=(parent,), quality_state="READY")
                    output[timeframe] = {"partition": asdict(descriptor), "catalog_record": asdict(catalog_record)}
            print(json.dumps(output, sort_keys=True)); return 0
        manifest = load_manifest(args.manifest); spec = spec_from_manifest(manifest)
        result = parse_archive(spec, manifest.archive_path, manifest)
        if args.command == "archive-verify":
            print(json.dumps({"manifest_id": manifest.manifest_id, "verified": True,
                              "rows": result.report.observations, "quality": asdict(result.report)}, sort_keys=True)); return 0
        if args.command == "archive-import":
            print(json.dumps({"rows": len(result.records), "quality": asdict(result.report)}, sort_keys=True)); return 0
        descriptor = ParquetResearchStore(args.output).write(result.records,
                     {"archive_provenance_id": manifest.provenance_id, "content_hash": manifest.content_hash})
        from .research_catalog import ResearchCatalog
        catalog_record = ResearchCatalog(args.catalog_root).add(descriptor, exchange="binance", market_type="spot",
                         symbol=manifest.symbol, raw_source_hashes=(manifest.content_hash,), quality_state=result.report.status)
        print(json.dumps({"partition": asdict(descriptor), "catalog_record": asdict(catalog_record)}, sort_keys=True)); return 0
    if args.command == "inspect":
        batch = read_raw(args.path)
        print(json.dumps({"raw_hash": batch.raw_hash, "records": len(batch.records),
                          "instrument": asdict(batch.instrument), "timeframe": batch.timeframe,
                          "requested_range": [batch.requested_start, batch.requested_end]}, sort_keys=True))
        return 0
    if args.command == "validate":
        result = normalize(read_raw(args.path))
        print(json.dumps(asdict(result.report), sort_keys=True)); return 0
    if args.command == "build-dataset":
        batch = read_raw(args.path); result = normalize(batch)
        dataset, report = build_market_dataset(batch, result, allow_quality_warning=args.allow_quality_warning)
        target = dataset.write(Path(args.output) / "datasets" / f"{dataset.dataset_id}.json")
        print(json.dumps({"dataset_path": str(target), "dataset_id": dataset.dataset_id,
                          "quality": asdict(report)}, sort_keys=True)); return 0
    start, end = utc_timestamp(args.start), utc_timestamp(args.end)
    batch = BinancePublicKlineAdapter().fetch(_instrument(args.symbol, "spot"), args.timeframe, start, end)
    raw_path = batch.write(args.output)
    normalized = normalize(batch)
    dataset, report = build_market_dataset(batch, normalized)
    dataset_path = dataset.write(Path(args.output) / "datasets" / f"{dataset.dataset_id}.json")
    print(json.dumps({"raw_path": str(raw_path), "dataset_path": str(dataset_path),
                      "dataset_id": dataset.dataset_id, "quality": asdict(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())