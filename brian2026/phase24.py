from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import time

from .archive import ArchiveSpec, BinanceArchiveAdapter, derive_timeframe, parse_archive
from .data import BinancePublicKlineAdapter, DataQualityReport, Instrument, NormalizationResult, canonical_hash, normalize
from .parquet_store import ParquetResearchStore, ParquetPartition
from .research_catalog import ResearchCatalog

PHASE24_DATASET_SCHEMA = "brian.btc-research-dataset.v1"
@dataclass(frozen=True, slots=True)
class MonthlyBuild:
    period: str
    archive_hash: str
    manifest_id: str
    quality_status: str
    rows_1m: int
    missing_intervals: int
    duplicates: int
    invalid_ohlc: int
    anomaly_classification: str
    anomaly_evidence_id: str | None
    partitions: tuple[tuple[str, str], ...]

def _gap_signatures(records):
    gaps=[]
    for left,right in zip(records,records[1:]):
        delta=right.exchange_open_timestamp-left.exchange_open_timestamp
        if delta>60:gaps.append((left.exchange_open_timestamp,right.exchange_open_timestamp,round(delta/60)-1))
    return tuple(gaps)


def _confirm_source_gaps(root: Path, spec: ArchiveSpec, archive_hash: str, records) -> str | None:
    gaps=_gap_signatures(records)
    if not gaps:return None
    receipt_payload={"monthly_archive_hash":archive_hash,"period":f"{spec.year:04d}-{spec.month:02d}",
                     "gaps":gaps,"evidence":"official_monthly+official_daily+public_rest",
                     "schema_version":"brian.known-source-gaps.v1"}
    receipt_id=canonical_hash(receipt_payload);target=root/"source_gap_evidence"/f"{receipt_id}.json"
    if target.exists():return receipt_id
    adapter=BinanceArchiveAdapter();daily_confirmed=[];rest_confirmed=[];monthly_discrepancies=[]
    for left,right,missing in gaps:
        day=datetime.fromtimestamp(left,timezone.utc)
        daily_spec=ArchiveSpec("BTCUSDT","1m",day.year,day.month,"daily",day.day)
        daily_manifest=adapter.fetch(daily_spec,root)
        daily=parse_archive(daily_spec,daily_manifest.archive_path,daily_manifest)
        daily_has_gap=(left,right,missing) in _gap_signatures(daily.records)
        daily_confirmed.append(daily_manifest.content_hash)
        instrument=Instrument("binance","spot","BTCUSDT","BTC","USDT")
        raw=BinancePublicKlineAdapter().fetch(instrument,"1m",left,right+60,now=2_000_000_000)
        rest=normalize(raw,now=2_000_000_000)
        rest_has_gap=(left,right,missing) in _gap_signatures(rest.records)
        if daily_has_gap != rest_has_gap:return None
        if not daily_has_gap:monthly_discrepancies.append((left,right,missing))
        rest_confirmed.append(raw.raw_hash)
    receipt_payload.update({"daily_archive_hashes":daily_confirmed,"rest_query_hashes":rest_confirmed})
    receipt_id=canonical_hash(receipt_payload);target=root/"source_gap_evidence"/f"{receipt_id}.json"
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(json.dumps(receipt_payload,sort_keys=True,separators=(",",":")).encode()+b"\n")
    return receipt_id

@dataclass(frozen=True, slots=True)
class MultiMonthDatasetManifest:
    symbol: str
    exchange: str
    market_type: str
    requested_start: str
    requested_end: str
    actual_start: float
    actual_end: float
    monthly_builds: tuple[MonthlyBuild, ...]
    row_counts: tuple[tuple[str, int], ...]
    quality_status: str
    source: str
    creation_timestamp: float
    schema_version: str = PHASE24_DATASET_SCHEMA

    @property
    def dataset_id(self) -> str:
        payload = asdict(self); payload.pop("creation_timestamp")
        return canonical_hash(payload)

    def write(self, root: str | Path) -> Path:
        payload = {"dataset_id": self.dataset_id, **asdict(self)}
        target = Path(root) / "dataset_manifests" / f"{self.dataset_id}.json"
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError("immutable multi-month manifest conflict")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
        return target


def _months(start: datetime, end: datetime):
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cursor < end:
        yield cursor.year, cursor.month
        cursor = datetime(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1, tzinfo=timezone.utc)


def build_btc_history(root: str | Path, start: datetime, end: datetime) -> MultiMonthDatasetManifest:
    root = Path(root); adapter = BinanceArchiveAdapter(); store = ParquetResearchStore(root / "parquet")
    catalog = ResearchCatalog(root); builds: list[MonthlyBuild] = []; total = {k: 0 for k in ("1m","5m","15m","1h")}
    actual_start = float("inf"); actual_end = 0.0
    for year, month in _months(start, end):
        spec = ArchiveSpec("BTCUSDT", "1m", year, month)
        manifest = adapter.fetch(spec, root)
        result = parse_archive(spec, manifest.archive_path, manifest)
        report = result.report
        period = f"{year:04d}-{month:02d}"
        evidence_id = None
        if report.status == "WARNING" and report.missing_interval_count and not report.duplicate_count and not report.invalid_ohlc_rejected:
            evidence_id = _confirm_source_gaps(root, spec, manifest.content_hash, result.records)
        known_source_gap = evidence_id is not None
        if report.status != "READY" and not known_source_gap:
            raise RuntimeError(f"quality gate failed for {spec.filename}: {asdict(report)}")
        anomaly = "KNOWN_SOURCE_GAP" if known_source_gap else "NONE"
        provenance = {"archive_provenance_id": manifest.provenance_id, "content_hash": manifest.content_hash}
        descriptors: dict[str, ParquetPartition] = {}
        descriptors["1m"] = store.write(result.records, provenance)
        for timeframe in ("5m","15m","1h"):
            derived = derive_timeframe(result.records, timeframe)
            expected = len(result.records) // {"5m":5,"15m":15,"1h":60}[timeframe]
            if not known_source_gap and len(derived) != expected:
                raise RuntimeError(f"unsafe/incomplete {timeframe} aggregation for {spec.filename}")
            if any(row.observed_timestamp != row.exchange_close_timestamp for row in derived):
                raise RuntimeError(f"early availability in {timeframe} aggregation for {spec.filename}")
            descriptors[timeframe] = store.write(derived, {**provenance,
                    "derived_from": descriptors["1m"].dataset_id, "aggregation": timeframe})
        for timeframe, descriptor in descriptors.items():
            catalog.add(descriptor, exchange="binance", market_type="spot", symbol="BTCUSDT",
                        raw_source_hashes=(manifest.content_hash,), quality_state="READY")
            total[timeframe] += descriptor.rows
        actual_start = min(actual_start, report.first_timestamp or actual_start)
        actual_end = max(actual_end, report.last_timestamp or actual_end)
        builds.append(MonthlyBuild(f"{year:04d}-{month:02d}", manifest.content_hash, manifest.manifest_id,
                    report.status, len(result.records), report.missing_interval_count,
                    report.duplicate_count, report.invalid_ohlc_rejected, anomaly, evidence_id,
                    tuple(sorted((timeframe, descriptor.dataset_id) for timeframe, descriptor in descriptors.items()))))
        print(json.dumps({"period": f"{year:04d}-{month:02d}", "status": "VERIFIED_BUILT",
                          "rows": len(result.records), "archive_hash": manifest.content_hash, "anomaly": anomaly}), flush=True)
        time.sleep(0.05)
    dataset = MultiMonthDatasetManifest("BTCUSDT","binance","spot",start.isoformat(),end.isoformat(),
              actual_start,actual_end,tuple(builds),tuple(sorted(total.items())),
              "READY_WITH_KNOWN_SOURCE_GAPS" if any(b.anomaly_classification != "NONE" for b in builds) else "READY",
              "https://data.binance.vision/data/spot/monthly/klines",time.time())
    path = dataset.write(root)
    print(json.dumps({"dataset_id":dataset.dataset_id,"manifest":str(path),"rows":total},sort_keys=True),flush=True)
    return dataset


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--root",default="research_data")
    parser.add_argument("--start",default="2020-01-01T00:00:00+00:00");parser.add_argument("--end",default="2026-08-01T00:00:00+00:00")
    args=parser.parse_args(argv);start=datetime.fromisoformat(args.start.replace("Z","+00:00"));end=datetime.fromisoformat(args.end.replace("Z","+00:00"))
    build_btc_history(args.root,start,end);return 0

if __name__=="__main__":raise SystemExit(main())
