from __future__ import annotations

import math
import pytest

from brian2026.experience_memory import ExperienceMemory, ExperienceMemoryConfig, compact_jsonl_rows
from brian2026.market_gym import GymBar, GymFrame, MarketGym, MarketGymConfig, TargetAllocation
from brian2026.multi_asset_catalog import AssetSpec, MultiAssetCatalog
from brian2026.portfolio import DEVELOPMENT_CUTOFF
from brian2026.world_model import (
    BrianWorldModel,
    HistoricalBar,
    MultiAssetHistory,
    WorldModelConfig,
    align_history_intersection,
)


def frame(timestamp: float, rows: dict[str, tuple[float, float, float, float]]) -> GymFrame:
    bars = tuple(
        GymBar(asset, timestamp, o, h, l, c, source_timestamp=timestamp)
        for asset, (o, h, l, c) in sorted(rows.items())
    )
    return GymFrame(timestamp, bars)


def make_history(length: int = 50) -> MultiAssetHistory:
    a = []
    b = []
    price_a = 100.0
    price_b = 80.0
    for i in range(length):
        timestamp = 1_600_000_000.0 + i * 300.0
        move_a = 1.0 + (0.002 if i % 5 in (0, 1, 2) else -0.0015)
        move_b = 1.0 + (0.0015 if i % 5 in (0, 1, 2) else -0.001)
        open_a = price_a * (1.0003 if i % 7 == 0 else 1.0)
        close_a = open_a * move_a
        open_b = price_b * (1.0002 if i % 7 == 0 else 1.0)
        close_b = open_b * move_b
        a.append(HistoricalBar("A", timestamp, open_a, max(open_a, close_a) * 1.002,
                               min(open_a, close_a) * 0.998, close_a, 1000 + i))
        b.append(HistoricalBar("B", timestamp, open_b, max(open_b, close_b) * 1.0015,
                               min(open_b, close_b) * 0.9985, close_b, 900 + i))
        price_a = close_a
        price_b = close_b
    return MultiAssetHistory.from_mapping("fixture-multi-asset-v1", {"A": a, "B": b})


def test_market_gym_executes_only_at_next_frame_open() -> None:
    frames = (
        frame(1.0, {"A": (100, 100, 100, 100)}),
        frame(2.0, {"A": (110, 121, 110, 121)}),
    )
    gym = MarketGym(frames, MarketGymConfig(
        starting_equity=500, fee_bps=0, assumed_spread_bps=0, slippage_bps=0,
        max_asset_weight=1.0, ruin_fraction=0.0,
    ))
    step = gym.step(TargetAllocation.from_mapping({"A": 1.0}))
    assert step.observation_timestamp == 1.0
    assert step.execution_timestamp == 2.0
    # It did NOT capture the unobservable 100->110 gap. It entered at 110, then earned 110->121.
    assert step.overnight_pnl == pytest.approx(0.0)
    assert step.intrabar_pnl == pytest.approx(50.0)
    assert step.equity_after == pytest.approx(550.0)


def test_existing_position_experiences_real_close_to_next_open_gap_before_rebalance() -> None:
    frames = (
        frame(1.0, {"A": (100, 100, 100, 100)}),
        frame(2.0, {"A": (100, 100, 100, 100)}),
        frame(3.0, {"A": (110, 110, 110, 110)}),
    )
    gym = MarketGym(frames, MarketGymConfig(
        fee_bps=0, assumed_spread_bps=0, slippage_bps=0,
        max_asset_weight=1.0, ruin_fraction=0.0,
    ))
    gym.step(TargetAllocation.from_mapping({"A": 1.0}))
    step = gym.step(TargetAllocation.from_mapping({"A": 1.0}))
    assert step.overnight_pnl == pytest.approx(50.0)
    assert step.equity_after == pytest.approx(550.0)


def test_ruin_terminates_life_and_reset_restores_exactly_500() -> None:
    frames = (
        frame(1.0, {"A": (100, 100, 100, 100)}),
        frame(2.0, {"A": (100, 300, 100, 300)}),
        frame(3.0, {"A": (300, 300, 300, 300)}),
    )
    gym = MarketGym(frames, MarketGymConfig(
        starting_equity=500, fee_bps=0, assumed_spread_bps=0, slippage_bps=0,
        max_asset_weight=1.0, ruin_fraction=0.01, allow_short=True,
    ))
    step = gym.step(TargetAllocation.from_mapping({"A": -1.0}))
    assert step.terminal is True
    assert step.terminal_reason == "RUIN"
    assert gym.equity == pytest.approx(0.0)
    with pytest.raises(RuntimeError):
        gym.step(TargetAllocation())
    gym.reset()
    assert gym.equity == pytest.approx(500.0)
    assert gym.weights == {}
    assert gym.trace == []
    assert gym.terminated is False


def test_market_gym_rejects_leverage_and_concentration() -> None:
    frames = (
        frame(1.0, {"A": (100, 100, 100, 100), "B": (100, 100, 100, 100)}),
        frame(2.0, {"A": (100, 100, 100, 100), "B": (100, 100, 100, 100)}),
    )
    gym = MarketGym(frames, MarketGymConfig(max_asset_weight=0.6, max_gross_exposure=1.0))
    with pytest.raises(ValueError, match="gross exposure"):
        gym.step(TargetAllocation.from_mapping({"A": 0.6, "B": 0.6}))
    with pytest.raises(ValueError, match="concentration"):
        gym.step(TargetAllocation.from_mapping({"A": 0.8}))


def test_missing_market_terminates_instead_of_forward_filling() -> None:
    frames = (
        frame(1.0, {"A": (100, 100, 100, 100)}),
        frame(2.0, {"B": (50, 50, 50, 50)}),
    )
    gym = MarketGym(frames, MarketGymConfig(max_asset_weight=1.0))
    step = gym.step(TargetAllocation.from_mapping({"A": 1.0}))
    assert step.terminal is True
    assert step.terminal_reason == "DATA_GAP:A"
    assert step.equity_after == pytest.approx(500.0)


def test_historical_world_model_rejects_contaminated_2026_source() -> None:
    bad = HistoricalBar("A", DEVELOPMENT_CUTOFF, 100, 101, 99, 100)
    with pytest.raises(ValueError, match="INVALID_CONTAMINATED"):
        MultiAssetHistory.from_mapping("bad", {"A": [HistoricalBar("A", DEVELOPMENT_CUTOFF - 300, 100, 101, 99, 100), bad]})


def test_alignment_uses_intersection_and_never_forward_fills_closed_market() -> None:
    a = [HistoricalBar("A", ts, 100, 101, 99, 100) for ts in (100.0, 200.0, 300.0)]
    b = [HistoricalBar("B", ts, 50, 51, 49, 50) for ts in (100.0, 300.0)]
    history = MultiAssetHistory.from_mapping("calendar-fixture", {"A": a, "B": b})
    aligned = align_history_intersection(history)
    assert [row.timestamp for row in aligned] == [100.0, 300.0]
    assert all({bar.asset_id for bar in row.bars} == {"A", "B"} for row in aligned)


def test_block_bootstrap_is_deterministic_and_training_only() -> None:
    model = BrianWorldModel(make_history(), WorldModelConfig(horizon_steps=12, block_length=4, seed=77))
    left = model.generate("BLOCK_BOOTSTRAP", episode_index=9)
    right = model.generate("BLOCK_BOOTSTRAP", episode_index=9)
    assert left == right
    assert left.receipt.world_id == right.receipt.world_id
    assert left.receipt.synthetic is True
    assert left.receipt.training_only is True
    assert left.receipt.evidence_class == "TRAINING_ONLY"


def test_episode_index_changes_world_identity_without_outcome_based_sampling() -> None:
    model = BrianWorldModel(make_history(), WorldModelConfig(horizon_steps=12, block_length=4, seed=77))
    first = model.generate("BLOCK_BOOTSTRAP", episode_index=1)
    second = model.generate("BLOCK_BOOTSTRAP", episode_index=2)
    assert first.receipt.world_id != second.receipt.world_id
    assert first.receipt.episode_index == 1
    assert second.receipt.episode_index == 2


def test_synchronized_bootstrap_uses_same_source_time_for_all_assets_each_step() -> None:
    model = BrianWorldModel(make_history(), WorldModelConfig(horizon_steps=15, block_length=3, seed=12))
    world = model.generate("BLOCK_BOOTSTRAP", episode_index=4)
    for generated_frame in world.frames:
        source_times = {bar.source_timestamp for bar in generated_frame.bars}
        assert len(source_times) == 1


def test_stress_world_changes_path_but_keeps_same_source_recipe() -> None:
    config = WorldModelConfig(horizon_steps=10, block_length=5, seed=4, stress_return_scale=2.0)
    model = BrianWorldModel(make_history(), config)
    base = model.generate("BLOCK_BOOTSTRAP", episode_index=3)
    stress = model.generate("STRESS_BOOTSTRAP", episode_index=3)
    assert base.receipt.source_blocks == stress.receipt.source_blocks
    assert base.frames != stress.frames
    assert stress.receipt.evidence_class == "TRAINING_ONLY"


def test_compact_experience_memory_does_not_store_every_trace() -> None:
    model = BrianWorldModel(make_history(), WorldModelConfig(horizon_steps=6, block_length=3, seed=2))
    memory = ExperienceMemory(ExperienceMemoryConfig(
        max_summaries=200, max_audit_traces=3, deterministic_trace_sample_mod=2,
        high_drawdown_trace_pct=100,
    ))
    for episode_index in range(20):
        world = model.generate("BLOCK_BOOTSTRAP", episode_index=episode_index)
        gym = MarketGym(world.frames, MarketGymConfig(
            fee_bps=0, assumed_spread_bps=0, slippage_bps=0, max_asset_weight=0.5,
        ))
        while not gym.terminated:
            gym.step(TargetAllocation.from_mapping({"A": 0.5, "B": 0.5}))
        memory.record(gym.finish(), world.receipt, policy_version="fixture-policy-v1")
    assert len(memory.summaries) == 20
    assert len(memory.audit_traces) <= 3
    assert len(compact_jsonl_rows(memory)) == 20
    manifest = memory.compact_manifest()
    assert manifest["stores_full_trace_for_every_episode"] is False
    assert manifest["summary_count"] == 20


def test_training_experience_cannot_be_promoted_to_final_evidence() -> None:
    world = BrianWorldModel(make_history(), WorldModelConfig(horizon_steps=3, block_length=2)).generate(
        "BLOCK_BOOTSTRAP", episode_index=0
    )
    gym = MarketGym(world.frames, MarketGymConfig(
        fee_bps=0, assumed_spread_bps=0, slippage_bps=0, max_asset_weight=0.5,
    ))
    while not gym.terminated:
        gym.step(TargetAllocation.from_mapping({"A": 0.5, "B": 0.5}))
    summary = ExperienceMemory().record(gym.finish(), world.receipt, policy_version="p1")
    assert summary.training_only is True
    assert summary.evidence_class == "TRAINING_ONLY"
    with pytest.raises(ValueError, match="cannot become final"):
        summary.to_final_evidence()


def test_asset_catalog_requires_proxy_disclosure() -> None:
    with pytest.raises(ValueError, match="proxy assets"):
        AssetSpec(
            canonical_id="gold-proxy", symbol="X", asset_class="etf", venue="fixture",
            quote_currency="USD", timezone="UTC", trading_calendar="fixture",
            native_frequency="1d", source_name="fixture", provenance_uri="fixture://x",
            is_proxy=True,
        )
    real = AssetSpec(
        canonical_id="btc-usdt", symbol="BTCUSDT", asset_class="crypto_spot", venue="binance",
        quote_currency="USDT", timezone="UTC", trading_calendar="24x7",
        native_frequency="5m", source_name="binance-archive", provenance_uri="fixture://btc",
    )
    proxy = AssetSpec(
        canonical_id="gold-etf-proxy", symbol="GLD", asset_class="etf", venue="fixture",
        quote_currency="USD", timezone="America/New_York", trading_calendar="XNYS",
        native_frequency="1d", source_name="fixture", provenance_uri="fixture://gld",
        is_proxy=True, proxy_for="gold exposure",
    )
    catalog = MultiAssetCatalog.from_sequence([proxy, real])
    assert catalog.get("gold-etf-proxy").is_proxy is True
    assert len(catalog.assets) == 2
