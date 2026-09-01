import hashlib
import inspect
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from datetime import datetime, timezone

from brian2026.archive import (ArchiveManifest, ArchiveSpec, BinanceArchiveAdapter,
                               ChecksumMismatch, derive_timeframe, discover_first_available, parse_archive, plan_months)
from brian2026.data import Instrument, NormalizedKline, main as data_main
from brian2026.engine import BrianEngine
from brian2026.parquet_store import ParquetResearchStore, logical_dataset_id
from brian2026.research_catalog import ResearchCatalog


def csv_zip(rows):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSDT-1m.csv", "\n".join(",".join(map(str, row)) for row in rows) + "\n")
    return stream.getvalue()


def csv_row(open_s, price=100.0):
    return (int(open_s * 1000), price, price + 1, price - 1, price + .5, 10,
            int((open_s + 59.999) * 1000), 0, 1, 0, 0, 0)


class Response:
    def __init__(self, *, content=b"", text="", status=200):
        self.content, self.text, self.status_code = content, text, status
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)


class Session:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def get(self, url, timeout): self.calls.append(url); return self.responses.pop(0)


def minute(index, price=None):
    start = 1_609_459_200 + index * 60
    value = 100 + index if price is None else price
    instrument = Instrument("binance", "spot", "BTCUSDT", "BTC", "USDT")
    return NormalizedKline(instrument, "1m", start, start + 59.999, start + 59.999,
                           start + 59.999, "offline_close_bound", value, value + 1,
                           value - 1, value + .5, 10, f"raw-{index}", "official")


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.spec = ArchiveSpec("BTCUSDT", "1m", 2021, 1)
        self.rows = [csv_row(self.spec.start + i * 60) for i in range(60)]
        self.content = csv_zip(self.rows); self.digest = hashlib.sha256(self.content).hexdigest()

    def test_checksum_success_parser_and_raw_immutable(self):
        session = Session([Response(text=f"{self.digest}  {self.spec.filename}"), Response(content=self.content)])
        with tempfile.TemporaryDirectory() as td:
            manifest = BinanceArchiveAdapter(session).fetch(self.spec, td)
            path = pathlib.Path(manifest.archive_path); before = path.read_bytes()
            result = parse_archive(self.spec, path, manifest)
            self.assertTrue(manifest.checksum_verified)
            self.assertEqual(len(result.records), 60)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(result.records[0].open, 100)

    def test_checksum_mismatch_rejected_before_import(self):
        session = Session([Response(text=f"{'0'*64}  {self.spec.filename}"), Response(content=self.content)])
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ChecksumMismatch): BinanceArchiveAdapter(session).fetch(self.spec, td)
            self.assertFalse((pathlib.Path(td) / "archives").exists())
            receipts=list((pathlib.Path(td)/"archive_failures").glob("*.json"))
            self.assertEqual(len(receipts),1)
            self.assertEqual(json.loads(receipts[0].read_text())["reason"],"CHECKSUM_FAILED")

    def test_verified_local_is_not_redownloaded(self):
        session = Session([Response(text=f"{self.digest}  {self.spec.filename}"), Response(content=self.content)])
        with tempfile.TemporaryDirectory() as td:
            adapter = BinanceArchiveAdapter(session); first = adapter.fetch(self.spec, td)
            second = adapter.fetch(self.spec, td)
            self.assertEqual(first.content_hash, second.content_hash)
            self.assertEqual(len(session.calls), 2)


class LogicalParquetTests(unittest.TestCase):
    def test_logical_hash_stability_and_sensitivity(self):
        rows = (minute(0), minute(1)); provenance = {"raw": "abc"}
        self.assertEqual(logical_dataset_id(rows, provenance), logical_dataset_id(tuple(rows), dict(provenance)))
        self.assertNotEqual(logical_dataset_id(rows, provenance), logical_dataset_id((minute(0), minute(1, 999)), provenance))
        self.assertNotEqual(logical_dataset_id(rows, provenance), logical_dataset_id(rows, {"raw": "changed"}))

    def test_parquet_roundtrip_and_conflicting_overwrite(self):
        rows = tuple(minute(i) for i in range(5)); provenance = {"raw": "abc"}
        with tempfile.TemporaryDirectory() as td:
            store = ParquetResearchStore(td); first = store.write(rows, provenance)
            loaded, loaded_provenance, identity = store.read(first.path)
            self.assertEqual(identity, first.dataset_id)
            self.assertEqual(loaded, rows); self.assertEqual(loaded_provenance, provenance)
            self.assertEqual(store.write(rows, provenance).dataset_id, first.dataset_id)
            with self.assertRaises(FileExistsError): store.write(tuple(list(rows[:-1]) + [minute(4, 999)]), provenance)

    def test_catalog_points_to_dataset_identity(self):
        with tempfile.TemporaryDirectory() as td:
            partition = ParquetResearchStore(pathlib.Path(td)/"parquet").write(tuple(minute(i) for i in range(5)), {"raw":"abc"})
            catalog = ResearchCatalog(td); record = catalog.add(partition, exchange="binance", market_type="spot",
                         symbol="BTCUSDT", raw_source_hashes=("abc",), quality_state="READY")
            found = catalog.search(symbol="BTCUSDT", timeframe="1m")
            self.assertEqual(found[0].dataset_id, partition.dataset_id)
            self.assertEqual(record.record_id, found[0].record_id)


class AggregationAndPlanningTests(unittest.TestCase):
    def test_utc_aggregation_and_availability(self):
        rows = tuple(minute(i) for i in range(60))
        five, fifteen, hour = derive_timeframe(rows,"5m"), derive_timeframe(rows,"15m"), derive_timeframe(rows,"1h")
        self.assertEqual((len(five),len(fifteen),len(hour)),(12,4,1))
        first = five[0]
        self.assertEqual(first.open,rows[0].open); self.assertEqual(first.high,max(r.high for r in rows[:5]))
        self.assertEqual(first.low,min(r.low for r in rows[:5])); self.assertEqual(first.close,rows[4].close)
        self.assertEqual(first.volume,sum(r.volume for r in rows[:5]))
        self.assertEqual(first.observed_timestamp,first.exchange_close_timestamp)
        self.assertEqual(hour[0].exchange_open_timestamp % 3600,0)

    def test_missing_constituent_prevents_unsafe_bar(self):
        rows = tuple(minute(i) for i in range(5) if i != 2)
        self.assertEqual(derive_timeframe(rows,"5m"),())

    def test_prelisting_and_resume_plan(self):
        start=datetime(2020,1,1,tzinfo=timezone.utc); end=datetime(2020,4,1,tzinfo=timezone.utc)
        verified={"SOLUSDT-1m-2020-03.zip"}
        plan=plan_months("SOLUSDT",start,end,verified,lambda spec: True,first_available=(2020,3))
        self.assertEqual([p.status for p in plan],["PRE_LISTING","PRE_LISTING","VERIFIED_LOCAL"])
        failed=plan_months("BTCUSDT",start,end,set(),lambda spec: spec.month != 2,None,{"BTCUSDT-1m-2020-01.zip"})
        self.assertEqual([p.status for p in failed],["CHECKSUM_FAILED","SOURCE_MISSING","DOWNLOAD"])
        self.assertEqual(discover_first_available("BTCUSDT",start,end,lambda spec: spec.month == 2),(2020,2))


class CliWiringTests(unittest.TestCase):
    def test_archive_plan_command_is_wired_without_download(self):
        with tempfile.TemporaryDirectory() as td, patch("brian2026.archive.BinanceArchiveAdapter.exists", return_value=True):
            self.assertEqual(data_main(["archive-plan", "--symbol", "BTCUSDT", "--start", "2024-01-01T00:00:00Z",
                                        "--end", "2024-02-01T00:00:00Z", "--root", td]), 0)

class SafetyTests(unittest.TestCase):
    def test_hard_shadow_and_no_execution_surface(self):
        with tempfile.TemporaryDirectory() as td: self.assertTrue(BrianEngine({"shadow_only":False},runtime_root=td).shadow_only)
        source=(inspect.getsource(BinanceArchiveAdapter)+inspect.getsource(ParquetResearchStore)).lower()
        for forbidden in ("create_order","place_order","cancel_order","api_key","secret"):
            self.assertNotIn(forbidden,source)


if __name__ == "__main__": unittest.main()