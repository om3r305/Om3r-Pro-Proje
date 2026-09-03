from __future__ import annotations

from brian2026.counterfactual_learner import (
    CausalCounterfactualLearner,
    CounterfactualLearnerConfig,
)
from brian2026.curriculum_runner import CurriculumPlan, CurriculumRunner, PolicyObservation
from brian2026.market_gym import GymBar, GymFrame, MarketGymConfig
from brian2026.portfolio import DEVELOPMENT_CUTOFF
from brian2026.world_model import BrianWorldModel, HistoricalBar, MultiAssetHistory, WorldModelConfig
import pytest


def trend_history(length: int = 48) -> MultiAssetHistory:
    rows = {"A": [], "B": []}
    prices = {"A": 100.0, "B": 100.0}
    for i in range(length):
        ts = 1_600_000_000.0 + i * 300.0
        for asset, move in (("A", 1.012), ("B", 0.988)):
            open_price = prices[asset]
            close = open_price * move
            rows[asset].append(HistoricalBar(
                asset,
                ts,
                open_price,
                max(open_price, close) * 1.001,
                min(open_price, close) * 0.999,
                close,
                1000.0 + i,
            ))
            prices[asset] = close
    return MultiAssetHistory.from_mapping("trend-history-v1", rows)


def learner_config() -> CounterfactualLearnerConfig:
    return CounterfactualLearnerConfig(
        min_weighted_samples_per_asset=2.0,
        min_abs_edge=0.0001,
        risk_aversion=0.10,
        min_uncertainty=0.0001,
        max_positions=2,
        max_asset_weight=0.25,
        max_gross_exposure=0.50,
    )


def runner_for(plan: CurriculumPlan) -> CurriculumRunner:
    model = BrianWorldModel(
        trend_history(),
        WorldModelConfig(horizon_steps=10, block_length=4, seed=73, stress_return_scale=1.5),
    )
    return CurriculumRunner(
        model,
        plan=plan,
        gym_config=MarketGymConfig(
            fee_bps=0.0,
            assumed_spread_bps=0.0,
            slippage_bps=0.0,
            max_asset_weight=0.50,
            max_gross_exposure=1.0,
        ),
    )


def test_fresh_learner_waits_until_counterfactual_warmup_is_available() -> None:
    learner = CausalCounterfactualLearner(learner_config())
    frame = GymFrame(1.0, (
        GymBar("A", 1.0, 100, 101, 99, 100, source_timestamp=1_600_000_000.0),
        GymBar("B", 1.0, 100, 101, 99, 100, source_timestamp=1_600_000_000.0),
    ))
    observation = PolicyObservation((frame,), 500.0, (), 0, 500.0)
    assert learner.act(observation).weights == ()


def test_learner_hard_rejects_contaminated_2026_source_timestamp() -> None:
    learner = CausalCounterfactualLearner(learner_config())
    frame = GymFrame(1.0, (
        GymBar("A", 1.0, 100, 101, 99, 100, source_timestamp=DEVELOPMENT_CUTOFF),
    ))
    with pytest.raises(ValueError, match="INVALID_CONTAMINATED"):
        learner.act(PolicyObservation((frame,), 500.0, (), 0, 500.0))


def test_resolved_curriculum_teaches_long_uptrend_and_short_downtrend() -> None:
    plan = CurriculumPlan(real_replay_episodes=8, block_bootstrap_episodes=0, stress_bootstrap_episodes=0, shard_size=8)
    learner = CausalCounterfactualLearner(learner_config())
    output = runner_for(plan).run_shard(0, learner)
    manifest = learner.training_manifest()
    assert manifest["episodes_learned"] == 8
    assert manifest["transitions_learned"] > 0
    assert output.receipt.policy_state_in != output.receipt.policy_state_out

    first = runner_for(CurriculumPlan(1, 0, 0, 1)).world_model.real_replay(episode_index=0).frames[0]
    allocation = learner.act(PolicyObservation((first,), 500.0, (), 0, 500.0)).as_dict()
    assert allocation["A"] > 0
    assert allocation["B"] < 0


def test_same_curriculum_and_initial_state_are_bitwise_reproducible() -> None:
    plan = CurriculumPlan(real_replay_episodes=3, block_bootstrap_episodes=2, stress_bootstrap_episodes=1, shard_size=6)
    left = CausalCounterfactualLearner(learner_config())
    right = CausalCounterfactualLearner(learner_config())
    left_result = runner_for(plan).run_shard(0, left)
    right_result = runner_for(plan).run_shard(0, right)
    assert left_result.receipt.receipt_id == right_result.receipt.receipt_id
    assert left.training_state_id == right.training_state_id
    assert left.state_dict() == right.state_dict()


def test_checkpoint_roundtrip_preserves_exact_training_state() -> None:
    plan = CurriculumPlan(real_replay_episodes=2, block_bootstrap_episodes=1, stress_bootstrap_episodes=0, shard_size=3)
    learner = CausalCounterfactualLearner(learner_config())
    runner_for(plan).run_shard(0, learner)
    checkpoint = learner.state_dict()
    restored = CausalCounterfactualLearner.from_state(checkpoint)
    assert restored.state_dict() == checkpoint
    assert restored.training_state_id == learner.training_state_id


def test_world_mode_weights_prevent_synthetic_lives_from_counting_like_real_replay() -> None:
    ratios = {}
    for name, plan in (
        ("real", CurriculumPlan(1, 0, 0, 1)),
        ("block", CurriculumPlan(0, 1, 0, 1)),
        ("stress", CurriculumPlan(0, 0, 1, 1)),
    ):
        learner = CausalCounterfactualLearner(learner_config())
        runner_for(plan).run_shard(0, learner)
        manifest = learner.training_manifest()
        ratios[name] = manifest["weighted_transitions"] / manifest["transitions_learned"]
    assert ratios["real"] == pytest.approx(1.0)
    assert ratios["block"] == pytest.approx(0.5)
    assert ratios["stress"] == pytest.approx(0.25)


def test_drawdown_throttle_can_force_wait_without_changing_model_state() -> None:
    plan = CurriculumPlan(real_replay_episodes=4, block_bootstrap_episodes=0, stress_bootstrap_episodes=0, shard_size=4)
    learner = CausalCounterfactualLearner(learner_config())
    runner_for(plan).run_shard(0, learner)
    checkpoint = learner.training_state_id
    frame = runner_for(CurriculumPlan(1, 0, 0, 1)).world_model.real_replay(episode_index=0).frames[0]
    allocation = learner.act(PolicyObservation((frame,), 200.0, (), 0, 500.0))
    assert allocation.weights == ()
    # act() only records the causal decision; no learning occurs before episode resolution.
    assert learner.training_manifest()["episodes_learned"] == 4
    assert checkpoint != ""
