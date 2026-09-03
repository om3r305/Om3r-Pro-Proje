from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from brian2026.counterfactual_learner import CausalCounterfactualLearner, FEATURE_NAMES, _AssetState
from brian2026.curriculum_runner import CurriculumPlan
from brian2026.phase35_training import (
    PHASE35_PLAN,
    PHASE35_RESERVED_EXAM_MONTHS,
    PHASE35_SYMBOLS,
    PHASE35_TRAINING_MONTHS,
    run_training,
)
from brian2026.state_fingerprint import portable_state_fingerprint
from brian2026.world_model import HistoricalBar, MultiAssetHistory, WorldModelConfig
import brian2026.phase35_training as phase35_module


def _history_fixture(rows: int = 96) -> MultiAssetHistory:
    prices = {
        "BTCUSDT": 23_000.0,
        "ETHUSDT": 1_600.0,
        "SOLUSDT": 24.0,
        "BNBUSDT": 300.0,
        "XRPUSDT": 0.40,
    }
    series = {}
    for symbol in PHASE35_SYMBOLS:
        price = prices[symbol]
        values = []
        for index in range(rows):
            timestamp = 1_675_000_299.999 + index * 300.0
            phase = (index + len(symbol)) % 11
            move = 1.0018 if phase < 6 else 0.9984
            close = price * move
            values.append(HistoricalBar(
                symbol,
                timestamp,
                price,
                max(price, close) * 1.0007,
                min(price, close) * 0.9993,
                close,
                1000.0 + index,
            ))
            price = close
        series[symbol] = tuple(values)
    return MultiAssetHistory.from_mapping("phase35-five-asset-fixture", series)


def test_phase35_plan_is_locked_to_exactly_1000_lives() -> None:
    assert PHASE35_PLAN.total_episodes == 1000
    assert PHASE35_PLAN.real_replay_episodes == 250
    assert PHASE35_PLAN.block_bootstrap_episodes == 500
    assert PHASE35_PLAN.stress_bootstrap_episodes == 250
    assert PHASE35_PLAN.shard_size == 100
    assert PHASE35_PLAN.shard_count == 10
    assert [PHASE35_PLAN.mode_for_episode(i) for i in (0, 249, 250, 749, 750, 999)] == [
        "REAL_REPLAY", "REAL_REPLAY", "BLOCK_BOOTSTRAP", "BLOCK_BOOTSTRAP",
        "STRESS_BOOTSTRAP", "STRESS_BOOTSTRAP",
    ]


def test_phase35_training_window_preregisters_march_and_april_as_unseen() -> None:
    assert PHASE35_TRAINING_MONTHS[0] == (2023, 1)
    assert PHASE35_TRAINING_MONTHS[-1] == (2024, 2)
    assert len(PHASE35_TRAINING_MONTHS) == 14
    assert PHASE35_RESERVED_EXAM_MONTHS == ((2024, 3), (2024, 4))
    assert not set(PHASE35_TRAINING_MONTHS).intersection(PHASE35_RESERVED_EXAM_MONTHS)
    assert all(year < 2026 for year, _ in PHASE35_TRAINING_MONTHS + PHASE35_RESERVED_EXAM_MONTHS)


def test_portable_fingerprint_ignores_only_last_bit_float_noise() -> None:
    learner = CausalCounterfactualLearner()
    learner._models["BTCUSDT"] = _AssetState(
        weights=[0.0012345678901234] + [0.0] * (len(FEATURE_NAMES) - 1),
        weighted_samples=100.0,
        updates=100,
        error_ewma=0.0012345678901234,
    )
    baseline = learner.state_dict()
    tiny_noise = deepcopy(baseline)
    tiny_noise["models"]["BTCUSDT"]["weights"][0] += 1e-18
    tiny_noise["models"]["BTCUSDT"]["error_ewma"] += 1e-18
    meaningful_change = deepcopy(baseline)
    meaningful_change["models"]["BTCUSDT"]["weights"][0] += 1e-7

    assert portable_state_fingerprint(baseline) == portable_state_fingerprint(tiny_noise)
    assert portable_state_fingerprint(baseline) != portable_state_fingerprint(meaningful_change)
    assert baseline["models"]["BTCUSDT"]["weights"][0] != meaningful_change["models"]["BTCUSDT"]["weights"][0]


def test_phase35_runner_keeps_checkpoint_chain_and_training_only_contract(tmp_path: Path, monkeypatch) -> None:
    history = _history_fixture()
    small_plan = CurriculumPlan(
        real_replay_episodes=2,
        block_bootstrap_episodes=2,
        stress_bootstrap_episodes=1,
        shard_size=2,
    )
    small_world = WorldModelConfig(
        horizon_steps=16,
        block_length=8,
        seed=3501,
        stress_return_scale=1.75,
    )
    monkeypatch.setattr(phase35_module, "PHASE35_PLAN", small_plan)
    monkeypatch.setattr(phase35_module, "PHASE35_WORLD_CONFIG", small_world)
    monkeypatch.setattr(
        phase35_module,
        "fetch_verified_training_history",
        lambda root, adapter=None: (
            history,
            (),
            {symbol: len(history.as_mapping()[symbol]) for symbol in PHASE35_SYMBOLS},
        ),
    )

    summary = run_training(
        tmp_path / "research",
        tmp_path / "summary.json",
        tmp_path / "checkpoint.json",
        tmp_path / "progress.json",
    )

    assert summary["declaration"] == "PHASE35_1000_LIFE_TRAINING_ONLY_SHADOW"
    assert summary["scientific_interpretation"] == "TRAINING_DIAGNOSTIC_NOT_PROFITABILITY_EVIDENCE"
    assert summary["episode_diagnostics"]["episode_count"] == 5
    assert summary["learner"]["manifest"]["episodes_learned"] == 5
    assert summary["preregistered_next_development_exam"]["status"] == "RESERVED_UNSEEN"
    assert summary["preregistered_next_development_exam"]["learning_allowed"] is False
    assert summary["automatic_promotion"] is False
    assert summary["live_execution"] is False
    assert summary["shadow_only"] is True
    receipts = summary["curriculum"]["shard_receipts"]
    assert len(receipts) == 3
    for left, right in zip(receipts, receipts[1:]):
        assert right["policy_state_in"] == left["policy_state_out"]
    assert receipts[-1]["policy_state_out"] == summary["learner"]["raw_training_state_id"]
    assert len(summary["learner"]["portable_training_state_fingerprint"]) == 64
    assert (tmp_path / "checkpoint.json").exists()
    assert (tmp_path / "progress.json").exists()
