from __future__ import annotations

from dataclasses import dataclass, field

from brian2026.curriculum_runner import (
    CurriculumPlan,
    CurriculumRunner,
    FlatAuditPolicy,
    PolicyObservation,
    deterministic_shard_schedule,
)
from brian2026.market_gym import MarketGymConfig, TargetAllocation
from brian2026.world_model import BrianWorldModel, HistoricalBar, MultiAssetHistory, WorldModelConfig


def history(length: int = 24) -> MultiAssetHistory:
    rows = {"A": [], "B": []}
    prices = {"A": 100.0, "B": 75.0}
    for i in range(length):
        ts = 1_600_000_000.0 + i * 300.0
        for asset, phase in (("A", 0), ("B", 1)):
            open_price = prices[asset]
            move = 1.002 if (i + phase) % 4 < 2 else 0.9985
            close = open_price * move
            rows[asset].append(HistoricalBar(
                asset, ts, open_price, max(open_price, close) * 1.001,
                min(open_price, close) * 0.999, close,
            ))
            prices[asset] = close
    return MultiAssetHistory.from_mapping("curriculum-fixture-v1", rows)


@dataclass
class SpyPolicy:
    seen_lengths: list[int] = field(default_factory=list)
    learned_experiences: list[str] = field(default_factory=list)

    @property
    def policy_version(self) -> str:
        return "spy-policy-v1"

    def act(self, observation: PolicyObservation) -> TargetAllocation:
        self.seen_lengths.append(len(observation.visible_frames))
        assert observation.step_index == len(observation.visible_frames) - 1
        assert not hasattr(observation, "world_receipt")
        assert not hasattr(observation, "seed")
        assert not hasattr(observation, "source_blocks")
        return TargetAllocation()

    def learn_after_episode(self, experience) -> None:
        assert experience.training_only is True
        assert experience.evidence_class == "TRAINING_ONLY"
        self.learned_experiences.append(experience.experience_id)


def small_runner(plan: CurriculumPlan) -> CurriculumRunner:
    model = BrianWorldModel(history(), WorldModelConfig(horizon_steps=4, block_length=2, seed=91))
    return CurriculumRunner(
        model,
        plan=plan,
        gym_config=MarketGymConfig(
            fee_bps=0, assumed_spread_bps=0, slippage_bps=0,
            max_asset_weight=0.5,
        ),
    )


def test_default_100k_curriculum_is_20_deterministic_5000_life_shards() -> None:
    plan = CurriculumPlan()
    assert plan.total_episodes == 100_000
    assert plan.shard_count == 20
    schedule = deterministic_shard_schedule(plan)
    assert len(schedule) == 20
    assert schedule[0] == (0, 0, 5_000)
    assert schedule[-1] == (19, 95_000, 100_000)
    assert sum(end - start for _, start, end in schedule) == 100_000


def test_curriculum_modes_are_preregistered_by_global_episode_index() -> None:
    plan = CurriculumPlan(real_replay_episodes=2, block_bootstrap_episodes=3, stress_bootstrap_episodes=2, shard_size=3)
    assert [plan.mode_for_episode(i) for i in range(7)] == [
        "REAL_REPLAY", "REAL_REPLAY",
        "BLOCK_BOOTSTRAP", "BLOCK_BOOTSTRAP", "BLOCK_BOOTSTRAP",
        "STRESS_BOOTSTRAP", "STRESS_BOOTSTRAP",
    ]


def test_policy_sees_only_causal_prefix_and_learns_after_episode_resolution() -> None:
    plan = CurriculumPlan(real_replay_episodes=1, block_bootstrap_episodes=0, stress_bootstrap_episodes=0, shard_size=1)
    policy = SpyPolicy()
    output = small_runner(plan).run_shard(0, policy)
    assert policy.seen_lengths == [1, 2, 3, 4]
    assert len(policy.learned_experiences) == 1
    assert output.receipt.episode_count == 1
    assert output.receipt.training_only is True
    assert output.receipt.memory_manifest["summary_count"] == 1


def test_flat_audit_policy_proves_curriculum_itself_does_not_create_pnl() -> None:
    plan = CurriculumPlan(real_replay_episodes=1, block_bootstrap_episodes=1, stress_bootstrap_episodes=1, shard_size=3)
    output = small_runner(plan).run_shard(0, FlatAuditPolicy())
    assert output.receipt.mode_counts == (
        ("BLOCK_BOOTSTRAP", 1), ("REAL_REPLAY", 1), ("STRESS_BOOTSTRAP", 1)
    )
    for summary in output.memory.summaries:
        assert summary.starting_equity == 500.0
        assert summary.ending_equity == 500.0
        assert summary.return_pct == 0.0
        assert summary.total_costs == 0.0


def test_same_shard_same_policy_is_exactly_reproducible() -> None:
    plan = CurriculumPlan(real_replay_episodes=1, block_bootstrap_episodes=2, stress_bootstrap_episodes=1, shard_size=4)
    runner = small_runner(plan)
    left = runner.run_shard(0, FlatAuditPolicy())
    right = runner.run_shard(0, FlatAuditPolicy())
    assert left.receipt.receipt_id == right.receipt.receipt_id
    assert left.receipt.memory_manifest["summary_hash"] == right.receipt.memory_manifest["summary_hash"]
    assert left.memory.summaries == right.memory.summaries


def test_shard_can_cross_curriculum_mode_boundary_without_reindexing_worlds() -> None:
    plan = CurriculumPlan(real_replay_episodes=2, block_bootstrap_episodes=2, stress_bootstrap_episodes=2, shard_size=4)
    runner = small_runner(plan)
    first = runner.run_shard(0, FlatAuditPolicy())
    second = runner.run_shard(1, FlatAuditPolicy())
    assert first.receipt.first_episode_index == 0
    assert first.receipt.last_episode_index_exclusive == 4
    assert first.receipt.mode_counts == (("BLOCK_BOOTSTRAP", 2), ("REAL_REPLAY", 2))
    assert second.receipt.first_episode_index == 4
    assert second.receipt.last_episode_index_exclusive == 6
    assert second.receipt.mode_counts == (("STRESS_BOOTSTRAP", 2),)
