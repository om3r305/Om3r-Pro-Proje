from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Protocol
import json

from .experience_memory import EpisodeExperience, ExperienceMemory, ExperienceMemoryConfig
from .market_gym import GymEpisodeResult, GymFrame, MarketGym, MarketGymConfig, TargetAllocation
from .world_model import BrianWorldModel, WorldMode

CURRICULUM_SCHEMA_VERSION = "brian.curriculum-runner.v1"


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CurriculumPlan:
    real_replay_episodes: int = 10_000
    block_bootstrap_episodes: int = 60_000
    stress_bootstrap_episodes: int = 30_000
    shard_size: int = 5_000
    schema_version: str = CURRICULUM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if min(self.real_replay_episodes, self.block_bootstrap_episodes, self.stress_bootstrap_episodes) < 0:
            raise ValueError("curriculum episode counts must be non-negative")
        if self.total_episodes < 1 or self.shard_size < 1:
            raise ValueError("curriculum requires episodes and positive shard size")

    @property
    def total_episodes(self) -> int:
        return self.real_replay_episodes + self.block_bootstrap_episodes + self.stress_bootstrap_episodes

    @property
    def shard_count(self) -> int:
        return (self.total_episodes + self.shard_size - 1) // self.shard_size

    @property
    def plan_id(self) -> str:
        return _hash(asdict(self))

    def mode_for_episode(self, episode_index: int) -> WorldMode:
        if not 0 <= episode_index < self.total_episodes:
            raise IndexError("episode index outside curriculum")
        if episode_index < self.real_replay_episodes:
            return "REAL_REPLAY"
        if episode_index < self.real_replay_episodes + self.block_bootstrap_episodes:
            return "BLOCK_BOOTSTRAP"
        return "STRESS_BOOTSTRAP"

    def shard_bounds(self, shard_index: int) -> tuple[int, int]:
        if not 0 <= shard_index < self.shard_count:
            raise IndexError("shard index outside curriculum")
        start = shard_index * self.shard_size
        return start, min(self.total_episodes, start + self.shard_size)


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """The only information a policy receives while an episode is alive."""

    visible_frames: tuple[GymFrame, ...]
    equity: float
    current_weights: tuple[tuple[str, float], ...]
    step_index: int
    starting_equity: float
    shadow_only: bool = True
    schema_version: str = CURRICULUM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.visible_frames:
            raise ValueError("policy observation needs at least one visible frame")
        if self.step_index != len(self.visible_frames) - 1:
            raise ValueError("step_index must match the visible causal history")
        if self.equity < 0 or self.starting_equity <= 0:
            raise ValueError("invalid policy account state")


class CausalCurriculumPolicy(Protocol):
    @property
    def policy_version(self) -> str: ...

    @property
    def training_state_id(self) -> str: ...

    def act(self, observation: PolicyObservation) -> TargetAllocation: ...

    def learn_after_episode(self, experience: EpisodeExperience) -> None: ...


class ResolvedEpisodeLearner(Protocol):
    """Optional post-episode training hook.

    The callback is invoked only after the episode has terminated. It receives the fully
    resolved market path and gym trace, but deliberately receives no world seed/source-block
    recipe. This is a training-only credit-assignment surface, never a live decision surface.
    """

    def learn_after_resolved_episode(
        self,
        experience: EpisodeExperience,
        result: GymEpisodeResult,
        resolved_frames: tuple[GymFrame, ...],
    ) -> None: ...


class FlatAuditPolicy:
    """Non-learning audit baseline. Useful to verify that the curriculum itself creates no PnL."""

    @property
    def policy_version(self) -> str:
        return "flat-audit-policy-v1"

    @property
    def training_state_id(self) -> str:
        return "flat-audit-policy-state-v1"

    def act(self, observation: PolicyObservation) -> TargetAllocation:
        return TargetAllocation()

    def learn_after_episode(self, experience: EpisodeExperience) -> None:
        return None


@dataclass(frozen=True, slots=True)
class CurriculumShardReceipt:
    plan_id: str
    shard_index: int
    first_episode_index: int
    last_episode_index_exclusive: int
    episode_count: int
    mode_counts: tuple[tuple[str, int], ...]
    policy_version: str
    policy_state_in: str
    policy_state_out: str
    memory_manifest: dict[str, object]
    receipt_id: str
    training_only: bool = True
    shadow_only: bool = True
    schema_version: str = CURRICULUM_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class CurriculumShardResult:
    receipt: CurriculumShardReceipt
    memory: ExperienceMemory


class CurriculumRunner:
    """Runs deterministic, shardable training lives without exposing future world recipes to policy code."""

    def __init__(self, world_model: BrianWorldModel, *, plan: CurriculumPlan = CurriculumPlan(),
                 gym_config: MarketGymConfig = MarketGymConfig(),
                 memory_config: ExperienceMemoryConfig | None = None) -> None:
        self.world_model = world_model
        self.plan = plan
        self.gym_config = gym_config
        self.memory_config = memory_config

    def run_shard(self, shard_index: int, policy: CausalCurriculumPolicy, *,
                  expected_policy_state_in: str | None = None) -> CurriculumShardResult:
        policy_version = str(policy.policy_version)
        policy_state_in = str(policy.training_state_id)
        if not policy_version.strip() or not policy_state_in.strip():
            raise ValueError("policy_version and training_state_id are required")
        if expected_policy_state_in is not None and policy_state_in != expected_policy_state_in:
            raise ValueError("policy training state does not match the required previous-shard checkpoint")

        start, end = self.plan.shard_bounds(shard_index)
        requested = end - start
        memory_cfg = self.memory_config or ExperienceMemoryConfig(max_summaries=max(requested, 1))
        if memory_cfg.max_summaries < requested:
            raise ValueError("experience memory capacity is smaller than requested curriculum shard")
        memory = ExperienceMemory(memory_cfg)
        mode_counts: dict[str, int] = {}

        for episode_index in range(start, end):
            mode = self.plan.mode_for_episode(episode_index)
            world = self.world_model.generate(mode, episode_index=episode_index)
            gym = MarketGym(world.frames, self.gym_config)

            while not gym.terminated:
                # Crucial causal boundary: the policy sees frames [0..current], never the
                # future frames, world seed, sampled source blocks or future outcomes.
                visible = tuple(world.frames[: gym.index + 1])
                observation = PolicyObservation(
                    visible_frames=visible,
                    equity=gym.equity,
                    current_weights=tuple(sorted(gym.weights.items())),
                    step_index=gym.index,
                    starting_equity=self.gym_config.starting_equity,
                )
                allocation = policy.act(observation)
                if not isinstance(allocation, TargetAllocation):
                    raise TypeError("curriculum policy must return TargetAllocation")
                gym.step(allocation)

            result = gym.finish()
            experience = memory.record(result, world.receipt, policy_version=policy_version)
            # Summary learning is released only after the whole episode has resolved.
            policy.learn_after_episode(experience)
            # Advanced credit assignment may inspect the now-complete path, but only after
            # termination. No seed/source-block recipe is exposed through this callback.
            resolved_callback = getattr(policy, "learn_after_resolved_episode", None)
            if callable(resolved_callback):
                resolved_callback(experience, result, tuple(world.frames))
            if str(policy.policy_version) != policy_version:
                raise ValueError("policy_version must remain stable inside one curriculum shard")
            if not str(policy.training_state_id).strip():
                raise ValueError("policy training_state_id became empty after learning")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        policy_state_out = str(policy.training_state_id)
        manifest = memory.compact_manifest()
        identity = {
            "schema_version": CURRICULUM_SCHEMA_VERSION,
            "plan_id": self.plan.plan_id,
            "shard_index": shard_index,
            "first_episode_index": start,
            "last_episode_index_exclusive": end,
            "policy_version": policy_version,
            "policy_state_in": policy_state_in,
            "policy_state_out": policy_state_out,
            "mode_counts": sorted(mode_counts.items()),
            "memory_summary_hash": manifest["summary_hash"],
        }
        receipt = CurriculumShardReceipt(
            plan_id=self.plan.plan_id,
            shard_index=shard_index,
            first_episode_index=start,
            last_episode_index_exclusive=end,
            episode_count=requested,
            mode_counts=tuple(sorted(mode_counts.items())),
            policy_version=policy_version,
            policy_state_in=policy_state_in,
            policy_state_out=policy_state_out,
            memory_manifest=manifest,
            receipt_id=_hash(identity),
        )
        return CurriculumShardResult(receipt, memory)


def deterministic_shard_schedule(plan: CurriculumPlan = CurriculumPlan()) -> tuple[tuple[int, int, int], ...]:
    """Returns (shard_index, start, end) without generating or storing any worlds."""
    return tuple((index, *plan.shard_bounds(index)) for index in range(plan.shard_count))
