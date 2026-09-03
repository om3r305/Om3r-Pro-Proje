from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Sequence
import json
import math

from .market_gym import GymEpisodeResult, GymStep, TRAINING_EVIDENCE_CLASS
from .world_model import WorldReceipt

EXPERIENCE_SCHEMA_VERSION = "brian.experience-memory.v1"


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExperienceMemoryConfig:
    max_summaries: int = 250_000
    max_audit_traces: int = 512
    deterministic_trace_sample_mod: int = 997
    high_drawdown_trace_pct: float = 35.0

    def __post_init__(self) -> None:
        if self.max_summaries < 1 or self.max_audit_traces < 0 or self.deterministic_trace_sample_mod < 1:
            raise ValueError("invalid experience memory limits")
        if not 0 <= self.high_drawdown_trace_pct <= 100:
            raise ValueError("high_drawdown_trace_pct must be in [0,100]")


@dataclass(frozen=True, slots=True)
class EpisodeExperience:
    experience_id: str
    world_id: str
    world_mode: str
    source_dataset_id: str
    policy_version: str
    starting_equity: float
    ending_equity: float
    return_pct: float
    max_drawdown_pct: float
    total_turnover: float
    total_costs: float
    rebalance_count: int
    steps: int
    ruined: bool
    terminal_reason: str
    lesson_tags: tuple[str, ...]
    synthetic_world: bool
    training_only: bool = True
    evidence_class: str = TRAINING_EVIDENCE_CLASS
    shadow_only: bool = True
    schema_version: str = EXPERIENCE_SCHEMA_VERSION

    def to_final_evidence(self) -> None:
        raise ValueError("synthetic/replay experience is TRAINING_ONLY and cannot become final scientific evidence")


@dataclass(frozen=True, slots=True)
class AuditTrace:
    experience_id: str
    reason: str
    trace: tuple[GymStep, ...]


class ExperienceMemory:
    """Compact training memory: summaries are retained; full step traces are tightly bounded."""

    def __init__(self, config: ExperienceMemoryConfig = ExperienceMemoryConfig()) -> None:
        self.config = config
        self._summaries: list[EpisodeExperience] = []
        self._audit_traces: list[AuditTrace] = []
        self._ids: set[str] = set()

    @staticmethod
    def _lesson_tags(result: GymEpisodeResult) -> tuple[str, ...]:
        tags: list[str] = []
        if result.ruined:
            tags.append("RUIN")
        if result.return_pct > 0:
            tags.append("POSITIVE_RETURN")
        elif result.return_pct < 0:
            tags.append("NEGATIVE_RETURN")
        else:
            tags.append("FLAT_RETURN")
        if result.max_drawdown_pct >= 35:
            tags.append("SEVERE_DRAWDOWN")
        elif result.max_drawdown_pct >= 15:
            tags.append("MATERIAL_DRAWDOWN")
        if result.rebalance_count > max(5, result.steps // 2):
            tags.append("HIGH_TURNOVER")
        if result.total_costs > abs(result.ending_equity - result.starting_equity) and result.total_costs > 0:
            tags.append("COST_DOMINATED")
        if not result.ruined and result.steps > 0:
            tags.append("SURVIVED")
        return tuple(tags)

    def record(self, result: GymEpisodeResult, receipt: WorldReceipt, *, policy_version: str) -> EpisodeExperience:
        if not policy_version.strip():
            raise ValueError("policy_version is required")
        if not result.shadow_only or result.evidence_class != TRAINING_EVIDENCE_CLASS:
            raise ValueError("market gym result must remain TRAINING_ONLY shadow research")
        if not receipt.training_only or receipt.evidence_class != TRAINING_EVIDENCE_CLASS:
            raise ValueError("world receipt must remain TRAINING_ONLY")
        if len(self._summaries) >= self.config.max_summaries:
            raise OverflowError("experience summary capacity reached; persist/roll the compact shard before continuing")

        identity = {
            "schema_version": EXPERIENCE_SCHEMA_VERSION,
            "world_id": receipt.world_id,
            "policy_version": policy_version,
            "gym_episode_id": result.episode_id,
        }
        experience_id = _hash(identity)
        if experience_id in self._ids:
            return next(row for row in self._summaries if row.experience_id == experience_id)

        summary = EpisodeExperience(
            experience_id=experience_id,
            world_id=receipt.world_id,
            world_mode=receipt.mode,
            source_dataset_id=receipt.source_dataset_id,
            policy_version=policy_version,
            starting_equity=result.starting_equity,
            ending_equity=result.ending_equity,
            return_pct=result.return_pct,
            max_drawdown_pct=result.max_drawdown_pct,
            total_turnover=result.total_turnover,
            total_costs=result.total_costs,
            rebalance_count=result.rebalance_count,
            steps=result.steps,
            ruined=result.ruined,
            terminal_reason=result.terminal_reason,
            lesson_tags=self._lesson_tags(result),
            synthetic_world=receipt.synthetic,
        )
        self._summaries.append(summary)
        self._ids.add(experience_id)

        if len(self._audit_traces) < self.config.max_audit_traces:
            digest_bucket = int(experience_id[:16], 16) % self.config.deterministic_trace_sample_mod
            reason: str | None = None
            if result.ruined:
                reason = "RUIN"
            elif result.max_drawdown_pct >= self.config.high_drawdown_trace_pct:
                reason = "HIGH_DRAWDOWN"
            elif digest_bucket == 0:
                reason = "DETERMINISTIC_AUDIT_SAMPLE"
            if reason is not None:
                self._audit_traces.append(AuditTrace(experience_id, reason, result.trace))
        return summary

    @property
    def summaries(self) -> tuple[EpisodeExperience, ...]:
        return tuple(self._summaries)

    @property
    def audit_traces(self) -> tuple[AuditTrace, ...]:
        return tuple(self._audit_traces)

    def compact_manifest(self) -> dict[str, object]:
        returns = [row.return_pct for row in self._summaries]
        drawdowns = [row.max_drawdown_pct for row in self._summaries]
        ruins = sum(row.ruined for row in self._summaries)
        modes: dict[str, int] = {}
        for row in self._summaries:
            modes[row.world_mode] = modes.get(row.world_mode, 0) + 1
        payload = {
            "schema_version": EXPERIENCE_SCHEMA_VERSION,
            "training_only": True,
            "evidence_class": TRAINING_EVIDENCE_CLASS,
            "summary_count": len(self._summaries),
            "audit_trace_count": len(self._audit_traces),
            "ruin_count": ruins,
            "mean_return_pct": sum(returns) / len(returns) if returns else 0.0,
            "mean_max_drawdown_pct": sum(drawdowns) / len(drawdowns) if drawdowns else 0.0,
            "world_modes": dict(sorted(modes.items())),
            "summary_hash": _hash([asdict(row) for row in self._summaries]),
            "stores_full_trace_for_every_episode": False,
        }
        if not all(math.isfinite(float(value)) for key, value in payload.items() if key in {"mean_return_pct", "mean_max_drawdown_pct"}):
            raise ValueError("experience aggregate contains non-finite metrics")
        return payload


def compact_jsonl_rows(memory: ExperienceMemory) -> tuple[str, ...]:
    """Stable small rows suitable for sharded persistence; excludes step traces by design."""
    return tuple(json.dumps(asdict(row), sort_keys=True, separators=(",", ":")) for row in memory.summaries)
