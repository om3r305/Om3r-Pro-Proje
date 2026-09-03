from __future__ import annotations

from dataclasses import dataclass, field
import pytest

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

    @property
    def training_state_id(self) -> str:
        return f"spy-state-after-{len(self.learned_experiences)}-episodes"

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


@dataclass
class ResolvedSpyPolicy(SpyPolicy):
    callback_order: list[str] = field(default_factory=list)
    resolved_lengths: list[int] = field(default_factory=list)

    @property
    def policy_version(self) -> str:
        return "resolved-spy-policy-v1"

    @property
    def training_state_id(self) -> str:
        return f"resolved-spy-after-{len(self.resolved_lengths)}-episodes"

    def learn_after_episode(self, experience) -> None:
        super().learn_after_episode(experience)
        self.callback_order.append("summary")

    def learn_after_resolved_episode(self, experience, result, resolved_frames) -> None:
        assert self.callback_order[-1] == "summary"
        assert self.learned_experiences[-1] == experience.experience_id
        assert result.trace[-1].terminal is True
        assert result.terminal_reason in {"PATH_END", "RUIN"} or result.terminal_reason.startswith("DATA_GAP:")
        assert len(resolved_frames) == 5
        self.resolved_lengths.append(len(resolved_frames))
        self.callback_order.append("resolved")


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
    assert output.receipt.policy_state_in == "spy-state-after-0-episodes"
    assert output.receipt.policy_state_out == "spy-state-after-1-episodes"
    assert output.receipt.memory_manifest["summary_count"] == 1


def test_resolved_episode_callback_runs_only_after_terminal_summary_release() -> None:
    plan = CurriculumPlan(real_replay_episodes=1, block_bootstrap_episodes=0, stress_bootstrap_episodes=0, shard_size=1)
    policy = ResolvedSpyPolicy()
    output = small_runner(plan).run_shard(0, policy)
    assert policy.callback_order == ["summary", "resolved"]
    assert policy.resolved_lengths == [5]
    assert output.receipt.policy_state_in == "resolved-spy-after-0-episodes"
    assert output.receipt.policy_state_out == "resolved-spy-after-1-episodes"


def test_flat_audit_policy_proves_curriculum_itself_does_not_create_pnl() -> None:
    plan = CurriculumPlan(real_replay_episodes=1, block_bootstrap_episodes=1, stress_bootstrap_episodes=1, shard_size=3)
    output = small_runner(plan).run_shard(0, FlatAuditPolicy())
    assert output.receipt.mode_counts == (
        ("BLOCK_BOOTSTRAP", 1), ("REAL_REPLAY", 1), ("STRESS_BOOTSTRAP", 1)
    )
    assert output.receipt.policy_state_in == output.receipt.policy_state_out
    for summary in output.memory.summaries:
        assert summary.starting_equity == 500.0
        assert summary.ending_equity == 500.0
        assert summary.return_pct == 0.0
        assert summary.total_costs == 0.0


def test_same_shard_same_initial_policy_state_is_exactly_reproducible() -> None:
    plan = CurriculumPlan(real_replay_episodes=1, block_bootstrap_episodes=2, stress_bootstrap_episodes=1, shard_size=4)
    runner = small_runner(plan)
    left = runner.run_shard(0, FlatAuditPolicy())
    right = runner.run_shard(0, FlatAuditPolicy())
    assert left.receipt.receipt_id == right.receipt.receipt_id
    assert left.receipt.memory_manifest["summary_hash"] == right.receipt.memory_manifest["summary_hash"]
    assert left.memory.summaries == right.memory.summaries


def test_runner_rejects_wrong_previous_shard_policy_checkpoint() -> None:
    plan = CurriculumPlan(real_replay_episodes=1, block_bootstrap_episodes=1, stress_bootstrap_episodes=0, shard_size=1)
    runner = small_runner(plan)
    policy = SpyPolicy()
    first = runner.run_shard(0, policy, expected_policy_state_in="spy-state-after-0-episodes")
    assert first.receipt.policy_state_out == "spy-state-after-1-episodes"
    with pytest.raises(ValueError, match="previous-shard checkpoint"):
        runner.run_shard(1, policy, expected_policy_state_in="wrong-state")
    second = runner.run_shard(1, policy, expected_policy_state_in=first.receipt.policy_state_out)
    assert second.receipt.policy_state_in == first.receipt.policy_state_out
    assert second.receipt.policy_state_out == "spy-state-after-2-episodes"


def test_shard_can_cross_curriculum_mode_boundary_without_reindexing_worlds() -> None:
    plan = CurriculumPlan(real_replay_episodes=2, block_bootstrap_episodes=2, stress_bootstrap_episodes=2, shard_size=4)
    runner = small_runner(plan)
    first = runner.run_shard(0, FlatAuditPolicy())
    second = runner.run_shard(1, FlatAuditPolicy(), expected_policy_state_in=first.receipt.policy_state_out)
    assert first.receipt.first_episode_index == 0
    assert first.receipt.last_episode_index_exclusive == 4
    assert first.receipt.mode_counts == (("BLOCK_BOOTSTRAP", 2), ("REAL_REPLAY", 2))
    assert second.receipt.first_episode_index == 4
    assert second.receipt.last_episode_index_exclusive == 6
    assert second.receipt.mode_counts == (("STRESS_BOOTSTRAP", 2),)
