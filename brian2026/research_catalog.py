from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json
import time

from .data import canonical_hash
from .parquet_store import ParquetPartition

RESEARCH_CATALOG_SCHEMA_VERSION = "brian.research-catalog.v1"


@dataclass(frozen=True, slots=True)
class ResearchCatalogRecord:
    dataset_id: str
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    start_timestamp: float
    end_timestamp: float
    raw_source_hashes: tuple[str, ...]
    quality_state: str
    rows: int
    storage_location: str
    creation_timestamp: float
    schema_version: str = RESEARCH_CATALOG_SCHEMA_VERSION

    @property
    def record_id(self) -> str:
        payload = asdict(self); payload.pop("creation_timestamp")
        return canonical_hash(payload)


class ResearchCatalog:
    """Mutable index of immutable content-addressed dataset records."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "catalog"

    def add(self, partition: ParquetPartition, *, exchange: str, market_type: str,
            symbol: str, raw_source_hashes: Iterable[str], quality_state: str) -> ResearchCatalogRecord:
        record = ResearchCatalogRecord(partition.dataset_id, exchange, market_type, symbol,
                    partition.timeframe, partition.first_timestamp, partition.last_timestamp,
                    tuple(sorted(set(raw_source_hashes))), quality_state, partition.rows,
                    partition.path, time.time())
        target = self.root / f"{record.record_id}.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            return ResearchCatalogRecord(**{**existing, "raw_source_hashes": tuple(existing["raw_source_hashes"])})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return record

    def search(self, *, symbol: str | None = None, timeframe: str | None = None) -> tuple[ResearchCatalogRecord, ...]:
        records: list[ResearchCatalogRecord] = []
        if not self.root.exists(): return ()
        for path in sorted(self.root.glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            item["raw_source_hashes"] = tuple(item["raw_source_hashes"])
            record = ResearchCatalogRecord(**item)
            if (symbol is None or record.symbol == symbol) and (timeframe is None or record.timeframe == timeframe):
                records.append(record)
        return tuple(records)