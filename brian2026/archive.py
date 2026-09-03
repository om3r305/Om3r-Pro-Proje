from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any, Callable, Iterable, Literal
from zipfile import ZipFile, BadZipFile
import csv
import json
import time

import requests

from .data import (Instrument, NormalizationResult, NormalizedKline, RawKline,
                   RawKlineBatch, canonical_hash, normalize)
from .dataset import _timeframe_seconds

ARCHIVE_SCHEMA_VERSION = "brian.binance-archive.v1"
OFFICIAL_ARCHIVE_ROOT = "https://data.binance.vision/data"
SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    symbol: str
    timeframe: str
    year: int
    month: int
    frequency: Literal["monthly", "daily"] = "monthly"
    day: int | None = None
    market_type: Literal["spot"] = "spot"

    def __post_init__(self) -> None:
        if self.symbol not in SUPPORTED_SYMBOLS or self.timeframe != "1m":
            raise ValueError("archive foundation supports configured spot symbols at 1m")
        if not 1 <= self.month <= 12 or (self.frequency == "daily" and self.day is None):
            raise ValueError("invalid archive period")

    @property
    def filename(self) -> str:
        period = f"{self.year:04d}-{self.month:02d}" + (f"-{self.day:02d}" if self.frequency == "daily" else "")
        return f"{self.symbol}-{self.timeframe}-{period}.zip"

    @property
    def url(self) -> str:
        return f"{OFFICIAL_ARCHIVE_ROOT}/spot/{self.frequency}/klines/{self.symbol}/{self.timeframe}/{self.filename}"

    @property
    def checksum_url(self) -> str:
        return self.url + ".CHECKSUM"

    @property
    def start(self) -> float:
        return datetime(self.year, self.month, self.day or 1, tzinfo=timezone.utc).timestamp()

    @property
    def end(self) -> float:
        if self.frequency == "daily":
            return self.start + 86400
        year, month = (self.year + 1, 1) if self.month == 12 else (self.year, self.month + 1)
        return datetime(year, month, 1, tzinfo=timezone.utc).timestamp()


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    archive_period: str
    source_url: str
    checksum_algorithm: str
    checksum_value: str
    checksum_verified: bool
    content_hash: str
    download_timestamp: float
    byte_size: int
    archive_path: str
    schema_version: str = ARCHIVE_SCHEMA_VERSION

    @property
    def manifest_id(self) -> str:
        return canonical_hash(asdict(self))

    @property
    def provenance_id(self) -> str:
        payload = asdict(self); payload.pop("download_timestamp"); payload.pop("archive_path")
        return canonical_hash(payload)

    def write(self, root: str | Path) -> Path:
        target = Path(root) / "archive_manifests" / f"{self.manifest_id}.json"
        content = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError("immutable manifest conflict")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
        return target


@dataclass(frozen=True, slots=True)
class ArchiveFailure:
    filename: str
    source_url: str
    expected_checksum: str
    actual_checksum: str
    detected_timestamp: float
    reason: str = "CHECKSUM_FAILED"
    schema_version: str = ARCHIVE_SCHEMA_VERSION

    @property
    def failure_id(self) -> str:
        return canonical_hash(asdict(self))

    def write(self, root: str | Path) -> Path:
        target = Path(root) / "archive_failures" / f"{self.failure_id}.json"
        content = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
        elif target.read_bytes() != content: raise FileExistsError("immutable failure receipt conflict")
        return target


class ChecksumMismatch(ValueError): pass


class BinanceArchiveAdapter:
    """Official public archive reader; no credentials and no execution surface."""
    def __init__(self, session: Any | None = None, timeout: float = 60.0) -> None:
        self.session, self.timeout = session or requests.Session(), timeout

    def exists(self, spec: ArchiveSpec) -> bool:
        response = self.session.head(spec.checksum_url, timeout=self.timeout)
        return response.status_code == 200

    @staticmethod
    def expected_checksum(text: str, filename: str) -> str:
        parts = text.strip().split()
        if len(parts) < 1 or len(parts[0]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in parts[0]):
            raise ValueError("invalid official checksum document")
        if len(parts) > 1 and parts[-1].lstrip("*") != filename:
            raise ValueError("checksum filename mismatch")
        return parts[0].lower()

    def fetch(self, spec: ArchiveSpec, root: str | Path, *, force: bool = False) -> ArchiveManifest:
        manifest_root = Path(root) / "archive_manifests"
        if not force and manifest_root.exists():
            for path in sorted(manifest_root.glob("*.json")):
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("source_url") == spec.url and item.get("checksum_verified"):
                    archive_path = Path(item["archive_path"])
                    if archive_path.exists() and sha256(archive_path.read_bytes()).hexdigest() == item["content_hash"]:
                        return ArchiveManifest(**item)
        checksum_response = self.session.get(spec.checksum_url, timeout=self.timeout)
        checksum_response.raise_for_status()
        expected = self.expected_checksum(checksum_response.text, spec.filename)
        archive_response = self.session.get(spec.url, timeout=self.timeout)
        archive_response.raise_for_status(); content = bytes(archive_response.content)
        actual = sha256(content).hexdigest()
        if actual != expected:
            ArchiveFailure(spec.filename, spec.url, expected, actual, time.time()).write(root)
            raise ChecksumMismatch(f"archive checksum mismatch: expected {expected}, got {actual}")
        archive_path = Path(root) / "archives" / "sha256" / actual[:2] / f"{actual}.zip"
        if archive_path.exists() and archive_path.read_bytes() != content:
            raise FileExistsError("immutable archive conflict")
        if not archive_path.exists():
            archive_path.parent.mkdir(parents=True, exist_ok=True); archive_path.write_bytes(content)
        manifest = ArchiveManifest("binance", "spot", spec.symbol, spec.timeframe,
                                   spec.filename.removesuffix(".zip"), spec.url, "sha256",
                                   expected, True, actual, time.time(), len(content), str(archive_path),
                                   ARCHIVE_SCHEMA_VERSION)
        manifest.write(root); return manifest


def parse_archive(spec: ArchiveSpec, archive_path: str | Path, manifest: ArchiveManifest) -> NormalizationResult:
    content = Path(archive_path).read_bytes()
    if sha256(content).hexdigest() != manifest.checksum_value or not manifest.checksum_verified:
        raise ChecksumMismatch("archive must be verified before import")
    try:
        with ZipFile(BytesIO(content)) as zipped:
            names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1: raise ValueError("archive must contain exactly one CSV")
            with zipped.open(names[0]) as raw_file:
                reader = csv.reader(TextIOWrapper(raw_file, encoding="utf-8", newline=""))
                rows = [row for row in reader if row]
    except BadZipFile as exc:
        raise ValueError("corrupt ZIP archive") from exc
    if rows and not rows[0][0].isdigit(): rows = rows[1:]
    retrieval = manifest.download_timestamp
    raw_rows = tuple(RawKline(tuple(row), spec.url, retrieval,
                              "us" if int(row[0]) > 100_000_000_000_000 else "ms") for row in rows)
    instrument = Instrument("binance", "spot", spec.symbol, spec.symbol[:-4], "USDT")
    batch = RawKlineBatch(instrument, spec.timeframe, spec.start, spec.end, raw_rows)
    return normalize(batch, now=max(spec.end + 1, time.time()))


def derive_timeframe(records: Iterable[NormalizedKline], timeframe: str) -> tuple[NormalizedKline, ...]:
    target = _timeframe_seconds(timeframe)
    if target not in (300, 900, 3600): raise ValueError("derived timeframe must be 5m, 15m, or 1h")
    rows = tuple(sorted(records, key=lambda r: r.exchange_open_timestamp))
    if any(row.timeframe != "1m" for row in rows): raise ValueError("only canonical 1m rows may be aggregated")
    needed = target // 60; groups: dict[float, list[NormalizedKline]] = {}
    for row in rows:
        bucket = row.exchange_open_timestamp - (row.exchange_open_timestamp % target)
        groups.setdefault(bucket, []).append(row)
    derived: list[NormalizedKline] = []
    for bucket, group in sorted(groups.items()):
        expected = [bucket + i * 60 for i in range(needed)]
        if [r.exchange_open_timestamp for r in group] != expected: continue
        last_close = bucket + target - 0.001
        if abs(group[-1].exchange_close_timestamp - last_close) > 0.002: continue
        source_hash = canonical_hash({"timeframe": timeframe, "raw_ids": [r.raw_id for r in group]})
        first, last = group[0], group[-1]
        derived.append(NormalizedKline(first.instrument, timeframe, bucket, last_close,
                        last_close, last_close, "offline_close_bound", first.open,
                        max(r.high for r in group), min(r.low for r in group), last.close,
                        sum(r.volume for r in group), source_hash, "derived:binance-spot-1m"))
    return tuple(derived)


@dataclass(frozen=True, slots=True)
class ArchivePlanItem:
    spec: ArchiveSpec
    status: Literal["VERIFIED_LOCAL", "DOWNLOAD", "PRE_LISTING", "SOURCE_MISSING", "CHECKSUM_FAILED"]


def plan_months(symbol: str, start: datetime, end: datetime, local_verified: set[str],
                source_exists: Callable[[ArchiveSpec], bool], first_available: tuple[int, int] | None = None,
                checksum_failed: set[str] | frozenset[str] = frozenset()) -> tuple[ArchivePlanItem, ...]:
    items: list[ArchivePlanItem] = []; cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    finish = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while cursor < finish:
        spec = ArchiveSpec(symbol, "1m", cursor.year, cursor.month)
        if first_available and (cursor.year, cursor.month) < first_available: status = "PRE_LISTING"
        elif spec.filename in checksum_failed: status = "CHECKSUM_FAILED"
        elif spec.filename in local_verified: status = "VERIFIED_LOCAL"
        else: status = "DOWNLOAD" if source_exists(spec) else "SOURCE_MISSING"
        items.append(ArchivePlanItem(spec, status))
        cursor = datetime(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1, tzinfo=timezone.utc)
    return tuple(items)
def load_manifest(path: str | Path) -> ArchiveManifest:
    item = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = ArchiveManifest(**item)
    if Path(path).stem != manifest.manifest_id:
        raise ValueError("manifest content identity mismatch")
    return manifest


def spec_from_manifest(manifest: ArchiveManifest) -> ArchiveSpec:
    period = manifest.archive_period.rsplit("-", 2)
    year, month = int(period[-2]), int(period[-1])
    return ArchiveSpec(manifest.symbol, manifest.timeframe, year, month)
def discover_first_available(symbol: str, start: datetime, end: datetime,
                             source_exists: Callable[[ArchiveSpec], bool]) -> tuple[int, int] | None:
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    finish = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while cursor < finish:
        spec = ArchiveSpec(symbol, "1m", cursor.year, cursor.month)
        if source_exists(spec): return cursor.year, cursor.month
        cursor = datetime(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1, tzinfo=timezone.utc)
    return None