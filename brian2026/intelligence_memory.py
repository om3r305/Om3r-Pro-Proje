from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import math

from .intelligence_fabric import EventKind


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _sign(value: float, deadband: float = 0.0) -> int:
    return 1 if value > deadband else -1 if value < -deadband else 0


@dataclass(frozen=True, slots=True)
class SourceOutcome:
    source_id: str
    event_kind: EventKind
    observed_at: float
    resolution_at: float
    predicted_direction: float
    realized_return: float
    truth_confirmed: bool
    manipulation_confirmed: bool = False
    neutral_return_band: float = 0.001
    schema_version: str = "brian.source-outcome.v1"

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id is required")
        observed = _finite(self.observed_at, "observed_at")
        resolution = _finite(self.resolution_at, "resolution_at")
        if resolution <= observed:
            raise ValueError("source outcome must resolve after observation")
        if not -1 <= _finite(self.predicted_direction, "predicted_direction") <= 1:
            raise ValueError("predicted_direction must be in [-1,1]")
        _finite(self.realized_return, "realized_return")
        if self.neutral_return_band < 0:
            raise ValueError("neutral_return_band cannot be negative")

    @property
    def direction_correct(self) -> bool:
        prediction = _sign(self.predicted_direction)
        realized = _sign(self.realized_return, self.neutral_return_band)
        return prediction != 0 and realized != 0 and prediction == realized

    @property
    def directional_sample(self) -> bool:
        return _sign(self.predicted_direction) != 0 and _sign(self.realized_return, self.neutral_return_band) != 0


@dataclass(frozen=True, slots=True)
class SourceReputation:
    source_id: str
    event_kind: EventKind | None
    resolved_samples: int
    directional_samples: int
    directional_accuracy: float
    truth_rate: float
    manipulation_rate: float
    reliability: float
    confidence: float
    schema_version: str = "brian.source-reputation.v1"


class SourceReputationMemory:
    """Causal source memory: outcomes become learnable only after resolution time."""

    def __init__(self, outcomes: Sequence[SourceOutcome] = ()) -> None:
        self._rows: list[SourceOutcome] = []
        self._last_learning_time = float("-inf")
        for row in sorted(outcomes, key=lambda x: x.resolution_at):
            self.learn(row, current_timestamp=row.resolution_at)

    @property
    def outcomes(self) -> tuple[SourceOutcome, ...]:
        return tuple(self._rows)

    def learn(self, outcome: SourceOutcome, *, current_timestamp: float) -> None:
        now = _finite(current_timestamp, "current_timestamp")
        if now < outcome.resolution_at:
            raise ValueError("cannot learn source outcome before its resolution timestamp")
        if now < self._last_learning_time:
            raise ValueError("source reputation learning must be chronological")
        self._rows.append(outcome)
        self._last_learning_time = now

    def reputation(self, source_id: str, *, event_kind: EventKind | None = None,
                   as_of: float | None = None) -> SourceReputation:
        if not source_id.strip():
            raise ValueError("source_id is required")
        cutoff = float("inf") if as_of is None else _finite(as_of, "as_of")
        rows = [
            row for row in self._rows
            if row.source_id == source_id and row.resolution_at <= cutoff and
            (event_kind is None or row.event_kind == event_kind)
        ]
        directional = [row for row in rows if row.directional_sample]
        accuracy = sum(row.direction_correct for row in directional) / len(directional) if directional else 0.5
        truth_rate = sum(row.truth_confirmed for row in rows) / len(rows) if rows else 0.5
        manipulation = sum(row.manipulation_confirmed for row in rows) / len(rows) if rows else 0.0
        # Conservative empirical blend. Sample confidence prevents a lucky source from dominating early.
        raw = 0.50 * accuracy + 0.35 * truth_rate + 0.15 * (1.0 - manipulation)
        confidence = len(rows) / (len(rows) + 20.0)
        reliability = 0.50 + confidence * (raw - 0.50)
        return SourceReputation(
            source_id, event_kind, len(rows), len(directional), accuracy, truth_rate,
            manipulation, max(0.0, min(1.0, reliability)), confidence,
        )


@dataclass(frozen=True, slots=True)
class OpportunityOutcome:
    context_key: str
    observed_at: float
    resolution_at: float
    net_return: float
    max_adverse_excursion: float
    event_truth_score: float
    schema_version: str = "brian.opportunity-outcome.v1"

    def __post_init__(self) -> None:
        if not self.context_key.strip():
            raise ValueError("context_key is required")
        if _finite(self.resolution_at, "resolution_at") <= _finite(self.observed_at, "observed_at"):
            raise ValueError("opportunity outcome must resolve after observation")
        _finite(self.net_return, "net_return")
        if _finite(self.max_adverse_excursion, "max_adverse_excursion") < 0:
            raise ValueError("max_adverse_excursion cannot be negative")
        if not 0 <= _finite(self.event_truth_score, "event_truth_score") <= 1:
            raise ValueError("event_truth_score must be in [0,1]")


@dataclass(frozen=True, slots=True)
class OpportunityExperience:
    context_key: str
    samples: int
    mean_net_return: float
    win_rate: float
    mean_adverse_excursion: float
    conservative_edge: float
    confidence: float
    schema_version: str = "brian.opportunity-experience.v1"


class OpportunityMemory:
    """Empirical context memory using only outcomes resolved before query time."""

    def __init__(self) -> None:
        self._rows: list[OpportunityOutcome] = []
        self._last_learning_time = float("-inf")

    def learn(self, outcome: OpportunityOutcome, *, current_timestamp: float) -> None:
        now = _finite(current_timestamp, "current_timestamp")
        if now < outcome.resolution_at:
            raise ValueError("cannot learn opportunity outcome before resolution")
        if now < self._last_learning_time:
            raise ValueError("opportunity learning must be chronological")
        self._rows.append(outcome)
        self._last_learning_time = now

    def experience(self, context_key: str, *, as_of: float) -> OpportunityExperience:
        cutoff = _finite(as_of, "as_of")
        rows = [row for row in self._rows if row.context_key == context_key and row.resolution_at <= cutoff]
        if not rows:
            return OpportunityExperience(context_key, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        returns = [row.net_return for row in rows]
        mean = sum(returns) / len(returns)
        variance = sum((x - mean) ** 2 for x in returns) / max(1, len(returns) - 1)
        stderr = math.sqrt(variance / len(returns)) if len(returns) > 1 else abs(mean)
        conservative = mean - 1.28 * stderr
        win_rate = sum(x > 0 for x in returns) / len(returns)
        adverse = sum(row.max_adverse_excursion for row in rows) / len(rows)
        confidence = len(rows) / (len(rows) + 30.0)
        return OpportunityExperience(context_key, len(rows), mean, win_rate, adverse,
                                     conservative, confidence)

    def manifest(self, *, as_of: float) -> Mapping[str, dict]:
        keys = sorted({row.context_key for row in self._rows if row.resolution_at <= as_of})
        return {key: asdict(self.experience(key, as_of=as_of)) for key in keys}
