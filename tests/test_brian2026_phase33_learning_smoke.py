from __future__ import annotations

from pathlib import Path

from brian2026.archive import ArchiveManifest
from brian2026.data import Instrument, NormalizedKline, canonical_hash
from brian2026.phase33_learning_smoke import (
    SMOKE_PLAN,
    SMOKE_SYMBOLS,
    history_from_derived,
    run_smoke,
)
from brian2026.world_model import HistoricalBar, MultiAssetHistory
import brian2026.phase33_learning_smoke as smoke_module


def _normalized_fixture(rows: int = 180):
    output = {}
    provenance = {}
    for symbol, base_price in (("BTCUSDT", 40_000.0), ("ETHUSDT", 2_000.0), ("SOLUSDT", 100.0)):
        base = symbol.removesuffix("USDT")
        instrument = Instrument("binance", "spot", symbol, base, "USDT")
        items = []
        price = base_price
        for index in range(rows):
            open_ts = 1_700_000_000.0 + index * 300.0
            close_ts = open_ts + 299.999
            move = 1.0015 if (index + len(symbol)) % 5 < 3 else 0.9987
            close = price * move
            items.append(NormalizedKline(
                instrument=instrument,
                timeframe="5m",
                exchange_open_timestamp=open_ts,
                exchange_close_timestamp=close_ts,
                source_event_timestamp=close_ts,
                observed_timestamp=close_ts,
                ingestion_timestamp_type="offline_close_bound",
                open=price,
                high=max(price, close) * 1.0005,
                low=min(price, close) * 0.9995,
                close=close,
                volume=1000.0 + index,
                raw_id=canonical_hash({"symbol": symbol, "index": index}),
                source_endpoint="fixture:verified-archive",
            ))
            price = close
        output[symbol] = tuple(items)
        provenance[symbol] = canonical_hash({"verified": symbol})
    return output, provenance


def _history_fixture(rows: int = 220) -> MultiAssetHistory:
    series = {}
    for symbol, base_price in (("BTCUSDT", 40_000.0), ("ETHUSDT", 2_000.0), ("SOLUSDT", 100.0)):
        values = []
        price = base_price
        for index in range(rows):
            ts = 1_700_000_299.999 + index * 300.0
            phase = (index + len(symbol)) % 8
            move = 1.002 if phase < 4 else 0.9982
            close = price * move
            values.append(HistoricalBar(
                symbol,
                ts,
                price,
                max(price, close) * 1.0007,
                min(price, close) * 0.9993,
                close,
                1000.0 + index,
            ))
            price = close
        series[symbol] = tuple(values)
    return MultiAssetHistory.from_mapping("verified-smoke-fixture-v1", series)


def _manifest(symbol: str) -> ArchiveManifest:
    digest = canonical_hash({"archive": symbol})
    return ArchiveManifest(
        exchange="binance",
        market_type="spot",
        symbol=symbol,
        timeframe="1m",
        archive_period=f"{symbol}-1m-2024-01",
        source_url=f"https://data.binance.vision/{symbol}",
        checksum_algorithm="sha256",
        checksum_value=digest,
        checksum_verified=True,
        content_hash=digest,
        download_timestamp=1_700_000_000.0,
        byte_size=123,
        archive_path=f"/tmp/{digest}.zip",
    )


def test_history_from_derived_uses_completed_candle_timestamp_and_exact_universe() -> None:
    records, provenance = _normalized_fixture()
    history = history_from_derived(records, provenance)
    assert history.asset_ids == tuple(sorted(SMOKE_SYMBOLS))
    source = records["BTCUSDT"][0]
    first = history.as_mapping()["BTCUSDT"][0]
    assert first.timestamp == source.exchange_close_timestamp
    assert first.timestamp != source.exchange_open_timestamp
    assert len(history.dataset_id) == 64


def test_phase33_smoke_plan_is_fixed_100_lives_in_two_checkpointed_shards() -> None:
    assert SMOKE_PLAN.total_episodes == 100
    assert SMOKE_PLAN.shard_count == 2
    assert SMOKE_PLAN.shard_bounds(0) == (0, 50)
    assert SMOKE_PLAN.shard_bounds(1) == (50, 100)
    assert [SMOKE_PLAN.mode_for_episode(i) for i in (0, 24, 25, 74, 75, 99)] == [
        "REAL_REPLAY", "REAL_REPLAY", "BLOCK_BOOTSTRAP", "BLOCK_BOOTSTRAP",
        "STRESS_BOOTSTRAP", "STRESS_BOOTSTRAP",
    ]


def test_smoke_runner_produces_training_only_artifacts_and_continuous_policy_state(tmp_path: Path, monkeypatch) -> None:
    history = _history_fixture()
    manifests = tuple(_manifest(symbol) for symbol in SMOKE_SYMBOLS)
    row_counts = {symbol: len(history.as_mapping()[symbol]) for symbol in SMOKE_SYMBOLS}
    monkeypatch.setattr(
        smoke_module,
        "fetch_verified_smoke_history",
        lambda root, adapter=None: (history, manifests, row_counts),
    )
    output_path = tmp_path / "summary.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    summary = run_smoke(tmp_path / "research", output_path, checkpoint_path)

    assert output_path.exists()
    assert checkpoint_path.exists()
    assert summary["declaration"] == "TRAINING_ONLY_SHADOW_SMOKE"
    assert summary["scientific_interpretation"] == "INFRASTRUCTURE_AND_LEARNING_DIAGNOSTIC_NOT_PROFITABILITY_EVIDENCE"
    assert summary["episode_diagnostics"]["episode_count"] == 100
    assert summary["holdout"]["evaluation_allowed"] is False
    assert summary["shadow_only"] is True
    assert summary["automatic_promotion"] is False
    receipts = summary["curriculum"]["shard_receipts"]
    assert len(receipts) == 2
    assert receipts[1]["policy_state_in"] == receipts[0]["policy_state_out"]
    assert receipts[-1]["policy_state_out"] == summary["learner"]["training_state_id"]
    assert all(row["checksum_verified"] is True for row in summary["dataset"]["archives"])
