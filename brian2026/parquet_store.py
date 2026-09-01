from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import os
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .data import Instrument, NormalizedKline, canonical_hash

PARQUET_SCHEMA_VERSION = "brian.parquet-market.v1"


def logical_row(row: NormalizedKline) -> dict[str, Any]:
    return {"exchange": row.instrument.exchange, "market_type": row.instrument.market_type,
            "symbol": row.instrument.symbol, "base_asset": row.instrument.base_asset,
            "quote_asset": row.instrument.quote_asset, "timeframe": row.timeframe,
            "open_timestamp": row.exchange_open_timestamp,
            "close_timestamp": row.exchange_close_timestamp,
            "source_event_timestamp": row.source_event_timestamp,
            "observed_timestamp": row.observed_timestamp,
            "ingestion_timestamp_type": row.ingestion_timestamp_type,
            "open": row.open, "high": row.high, "low": row.low, "close": row.close,
            "volume": row.volume, "raw_id": row.raw_id,
            "source_endpoint": row.source_endpoint,
            "bid": row.bid, "ask": row.ask,
            "order_book_available": row.order_book_available,
            "funding_rate": row.funding_rate,
            "row_schema_version": row.schema_version}


def logical_dataset_id(records: Iterable[NormalizedKline], provenance: Mapping[str, Any]) -> str:
    rows = [logical_row(row) for row in sorted(records, key=lambda r: (r.exchange_open_timestamp, r.instrument.symbol, r.timeframe))]
    return canonical_hash({"schema_version": PARQUET_SCHEMA_VERSION,
                           "provenance": dict(sorted(provenance.items())), "rows": rows})


@dataclass(frozen=True, slots=True)
class ParquetPartition:
    dataset_id: str
    path: str
    rows: int
    timeframe: str
    first_timestamp: float
    last_timestamp: float
    provenance: tuple[tuple[str, str], ...]
    physical_bytes: int
    schema_version: str = PARQUET_SCHEMA_VERSION


class ParquetResearchStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, records: Iterable[NormalizedKline], provenance: Mapping[str, Any]) -> ParquetPartition:
        rows = tuple(sorted(records, key=lambda r: r.exchange_open_timestamp))
        if not rows: raise ValueError("cannot persist empty partition")
        identities = {(r.instrument.exchange, r.instrument.market_type, r.instrument.symbol, r.timeframe) for r in rows}
        if len(identities) != 1: raise ValueError("partition cannot mix instruments or timeframes")
        exchange, market, symbol, timeframe = next(iter(identities))
        month_keys = {(datetime.fromtimestamp(r.exchange_open_timestamp, timezone.utc).year,
                       datetime.fromtimestamp(r.exchange_open_timestamp, timezone.utc).month) for r in rows}
        if len(month_keys) != 1: raise ValueError("partition must contain exactly one UTC month")
        year, month = next(iter(month_keys)); dataset_id = logical_dataset_id(rows, provenance)
        target = (self.root / f"exchange={exchange}" / f"market={market}" / f"symbol={symbol}" /
                  f"timeframe={timeframe}" / f"year={year:04d}" / f"month={month:02d}" / "part.parquet")
        if target.exists():
            existing = pq.read_metadata(target).metadata or {}
            if existing.get(b"brian_dataset_id", b"").decode() == dataset_id:
                return self._descriptor(target, rows, provenance, dataset_id)
            raise FileExistsError(f"conflicting immutable partition: {target}")
        dictionaries = [logical_row(row) for row in rows]
        table = pa.Table.from_pylist(dictionaries)
        metadata = dict(table.schema.metadata or {})
        metadata.update({b"brian_schema_version": PARQUET_SCHEMA_VERSION.encode(),
                         b"brian_dataset_id": dataset_id.encode(),
                         b"brian_provenance": json.dumps(dict(sorted(provenance.items())), sort_keys=True,
                                                          separators=(",", ":")).encode()})
        table = table.replace_schema_metadata(metadata); target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".part-", suffix=".parquet", dir=target.parent)
        os.close(fd)
        try:
            pq.write_table(table, temporary, compression="zstd", use_dictionary=True,
                           write_statistics=True)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        return self._descriptor(target, rows, provenance, dataset_id)

    @staticmethod
    def _descriptor(path: Path, rows: tuple[NormalizedKline, ...], provenance: Mapping[str, Any], dataset_id: str) -> ParquetPartition:
        return ParquetPartition(dataset_id, str(path), len(rows), rows[0].timeframe,
                                rows[0].exchange_open_timestamp, rows[-1].exchange_close_timestamp,
                                tuple(sorted((str(k), str(v)) for k, v in provenance.items())), path.stat().st_size)

    def read(self, path: str | Path) -> tuple[tuple[NormalizedKline, ...], dict[str, str], str]:
        table = pq.ParquetFile(path).read(); metadata = table.schema.metadata or {}
        dataset_id = metadata[b"brian_dataset_id"].decode()
        provenance = json.loads(metadata[b"brian_provenance"].decode())
        records: list[NormalizedKline] = []
        for item in table.to_pylist():
            instrument = Instrument(item["exchange"], item["market_type"], item["symbol"],
                                    item["base_asset"], item["quote_asset"])
            records.append(NormalizedKline(instrument, item["timeframe"], item["open_timestamp"],
                           item["close_timestamp"], item["source_event_timestamp"], item["observed_timestamp"],
                           item["ingestion_timestamp_type"], item["open"], item["high"], item["low"],
                           item["close"], item["volume"], item["raw_id"], item["source_endpoint"],
                           item["bid"], item["ask"], item["order_book_available"], item["funding_rate"],
                           item["row_schema_version"]))
        result = tuple(records)
        if logical_dataset_id(result, provenance) != dataset_id:
            raise ValueError("Parquet logical identity mismatch")
        return result, provenance, dataset_id