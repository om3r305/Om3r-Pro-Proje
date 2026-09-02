from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

import numpy as np

from .portfolio import DEVELOPMENT_CUTOFF

Action = Literal["BUY", "SELL", "WAIT"]

FAMILIARITY_FEATURES: tuple[str, ...] = (
    "return_1",
    "return_5",
    "ema_slope",
    "rsi",
    "atr_pct",
    "zscore",
    "bb_position",
    "structure_state",
    "dip_score",
    "rally_score",
    "relative_volume",
    "structure_15m",
    "structure_1h",
)


@dataclass(frozen=True, slots=True)
class FamiliarityConfig:
    quantile: float = 0.995
    min_observed_fraction: float = 0.60
    max_standardized_value: float = 12.0

    def __post_init__(self) -> None:
        if not 0.5 < self.quantile < 1.0:
            raise ValueError("familiarity quantile must be in (0.5, 1)")
        if not 0.0 < self.min_observed_fraction <= 1.0:
            raise ValueError("min_observed_fraction must be in (0, 1]")
        if self.max_standardized_value <= 0:
            raise ValueError("max_standardized_value must be positive")


@dataclass(frozen=True, slots=True)
class FamiliarityAssessment:
    score: float
    threshold: float
    familiarity: float
    observed_fraction: float
    out_of_distribution: bool


class MarketFamiliarityModel:
    """Train-reference-only robust market familiarity / OOD gate.

    This is intentionally small and deterministic. It never fits on validation or
    test data and does not infer unavailable values as zero. Distance is measured
    in robust standardized feature space using only values available at the
    current timestamp.
    """

    def __init__(self, feature_names: Sequence[str] = FAMILIARITY_FEATURES,
                 config: FamiliarityConfig = FamiliarityConfig()) -> None:
        self.feature_names = tuple(feature_names)
        self.config = config
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.expected_observed_fraction: float | None = None
        self.threshold: float | None = None

    def _matrix(self, features: Mapping[str, np.ndarray], indices: Sequence[int]) -> np.ndarray:
        if not self.feature_names:
            raise ValueError("familiarity feature set must not be empty")
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise KeyError(f"missing familiarity features: {missing}")
        return np.asarray([
            [float(features[name][int(index)]) for name in self.feature_names]
            for index in indices
        ], dtype=float)

    def fit(self, features: Mapping[str, np.ndarray], indices: Sequence[int],
            timestamps: Sequence[float]) -> "MarketFamiliarityModel":
        rows = np.asarray(indices, dtype=int)
        if not len(rows):
            raise ValueError("familiarity fit requires train observations")
        if any(float(timestamps[int(i)]) >= DEVELOPMENT_CUTOFF for i in rows):
            raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
        matrix = self._matrix(features, rows)
        with np.errstate(all="ignore"):
            center = np.nanmedian(matrix, axis=0)
            mad = np.nanmedian(np.abs(matrix - center), axis=0)
            robust = 1.4826 * mad
            std = np.nanstd(matrix, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.where(np.isfinite(robust) & (robust > 1e-12), robust, std)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        self.center = center.astype(float)
        self.scale = scale.astype(float)
        self.expected_observed_fraction = float(np.mean(np.isfinite(matrix)))
        scores = np.asarray([self._score_vector(row)[0] for row in matrix], dtype=float)
        finite = scores[np.isfinite(scores)]
        if not len(finite):
            raise ValueError("familiarity train rows have insufficient observed features")
        self.threshold = max(1e-6, float(np.quantile(finite, self.config.quantile)))
        return self

    def _score_vector(self, vector: np.ndarray) -> tuple[float, float]:
        if self.center is None or self.scale is None:
            raise RuntimeError("familiarity model not fitted")
        finite = np.isfinite(vector)
        observed = float(np.mean(finite))
        if observed < self.config.min_observed_fraction:
            return float("inf"), observed
        standardized = np.abs((vector[finite] - self.center[finite]) / self.scale[finite])
        standardized = np.minimum(standardized, self.config.max_standardized_value)
        distance = float(np.sqrt(np.mean(np.square(standardized))))
        expected = self.expected_observed_fraction if self.expected_observed_fraction is not None else 1.0
        missing_shift = abs(observed - expected)
        return distance + 0.50 * missing_shift, observed

    def assess_snapshot(self, snapshot: Mapping[str, object]) -> FamiliarityAssessment:
        if self.threshold is None:
            raise RuntimeError("familiarity model not fitted")
        vector = np.asarray([
            _finite(snapshot.get(name)) for name in self.feature_names
        ], dtype=float)
        score, observed = self._score_vector(vector)
        if not math.isfinite(score):
            return FamiliarityAssessment(score, self.threshold, 0.0, observed, True)
        ratio = score / max(self.threshold, 1e-9)
        familiarity = float(math.exp(-max(0.0, ratio - 0.25)))
        return FamiliarityAssessment(
            score, self.threshold, max(0.0, min(1.0, familiarity)), observed,
            bool(score > self.threshold),
        )

    def manifest(self) -> dict:
        if self.center is None or self.scale is None or self.threshold is None:
            raise RuntimeError("familiarity model not fitted")
        return {
            "schema": "brian.market-familiarity.v1",
            "feature_names": self.feature_names,
            "config": asdict(self.config),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "expected_observed_fraction": self.expected_observed_fraction,
            "threshold": self.threshold,
            "fit_partition": "train_only",
        }


@dataclass(frozen=True, slots=True)
class DriftConfig:
    window: int = 96
    warmup: int = 24
    validation_quantile: float = 0.995
    hard_multiplier: float = 1.50

    def __post_init__(self) -> None:
        if self.window < 2 or self.warmup < 2 or self.warmup > self.window:
            raise ValueError("invalid drift window/warmup")
        if not 0.5 < self.validation_quantile < 1.0:
            raise ValueError("validation_quantile must be in (0.5, 1)")
        if self.hard_multiplier <= 1.0:
            raise ValueError("hard_multiplier must exceed 1")


@dataclass(frozen=True, slots=True)
class DriftAssessment:
    score: float
    threshold: float
    drifted: bool
    hard_drift: bool
    observations: int


class CausalDriftMonitor:
    """Past-only rolling distribution-shift monitor.

    Reference center/scale come only from the train partition. At time t the
    score contains observations no later than t. Validation is used only to lock
    the alert threshold before any test decision is evaluated.
    """

    def __init__(self, familiarity: MarketFamiliarityModel,
                 config: DriftConfig = DriftConfig()) -> None:
        if familiarity.center is None or familiarity.scale is None:
            raise RuntimeError("drift monitor requires fitted train reference")
        self.feature_names = familiarity.feature_names
        self.center = familiarity.center.copy()
        self.scale = familiarity.scale.copy()
        self.config = config
        self.buffer: deque[np.ndarray] = deque(maxlen=config.window)
        self.threshold: float | None = None

    def _vector(self, snapshot: Mapping[str, object]) -> np.ndarray:
        values = np.asarray([_finite(snapshot.get(name)) for name in self.feature_names], dtype=float)
        finite = np.isfinite(values)
        normalized = np.full_like(values, np.nan, dtype=float)
        normalized[finite] = (values[finite] - self.center[finite]) / self.scale[finite]
        return normalized

    def observe_score(self, snapshot: Mapping[str, object]) -> float:
        self.buffer.append(self._vector(snapshot))
        if len(self.buffer) < self.config.warmup:
            return 0.0
        matrix = np.asarray(tuple(self.buffer), dtype=float)
        with np.errstate(all="ignore"):
            means = np.nanmean(matrix, axis=0)
        finite = np.isfinite(means)
        if not np.any(finite):
            return float("inf")
        mean_shift = float(np.sqrt(np.mean(np.square(np.clip(means[finite], -8.0, 8.0)))))
        observed = float(np.mean(np.isfinite(matrix)))
        return mean_shift + 0.50 * abs(1.0 - observed)

    def calibrate_validation(self, snapshots: Sequence[Mapping[str, object]]) -> tuple[float, ...]:
        if self.threshold is not None:
            raise RuntimeError("drift threshold already calibrated")
        scores = tuple(self.observe_score(row) for row in snapshots)
        usable = np.asarray(scores[self.config.warmup - 1:], dtype=float)
        usable = usable[np.isfinite(usable)]
        self.threshold = max(1e-6, float(np.quantile(usable, self.config.validation_quantile))) if len(usable) else 1.0
        return scores

    def assess(self, snapshot: Mapping[str, object]) -> DriftAssessment:
        if self.threshold is None:
            raise RuntimeError("drift threshold must be calibrated on validation before test")
        score = self.observe_score(snapshot)
        drifted = bool(not math.isfinite(score) or score > self.threshold)
        hard = bool(not math.isfinite(score) or score > self.threshold * self.config.hard_multiplier)
        return DriftAssessment(score, self.threshold, drifted, hard, len(self.buffer))


@dataclass(frozen=True, slots=True)
class ChallengerVote:
    name: str
    action: Action
    edge: float
    confidence: float
    validation_weight: float
    reason: str

    def __post_init__(self) -> None:
        if self.action not in ("BUY", "SELL", "WAIT"):
            raise ValueError("invalid challenger action")
        if not -1.0 <= self.edge <= 1.0:
            raise ValueError("challenger edge must be in [-1,1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("challenger confidence must be in [0,1]")
        if self.validation_weight < 0:
            raise ValueError("validation weight must be non-negative")


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    entry_edge: float = 0.18
    min_confidence: float = 0.52
    min_agreement: float = 0.58
    min_challengers: int = 2
    drift_penalty: float = 0.45
    familiarity_floor: float = 0.35

    def __post_init__(self) -> None:
        if not 0 < self.entry_edge <= 1:
            raise ValueError("entry_edge must be in (0,1]")
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.min_agreement <= 1:
            raise ValueError("confidence/agreement thresholds must be in [0,1]")
        if self.min_challengers < 1:
            raise ValueError("min_challengers must be positive")
        if not 0 < self.drift_penalty <= 1 or not 0 < self.familiarity_floor <= 1:
            raise ValueError("penalties must be in (0,1]")


@dataclass(frozen=True, slots=True)
class LeagueDecision:
    timestamp: float
    action: Action
    edge: float
    confidence: float
    agreement: float
    familiarity: float
    out_of_distribution: bool
    drift_score: float
    drift_threshold: float
    drifted: bool
    hard_drift: bool
    dominant_challenger: str | None
    votes: tuple[ChallengerVote, ...]
    veto_reasons: tuple[str, ...]
    shadow_only: bool = True
    schema_version: str = "brian.adaptive-quant-league.v1"


def validation_weight(*, directional_accuracy: float, coverage: float,
                      directional_samples: int) -> float:
    """Fixed preregistered validation-only challenger weighting rule."""
    if directional_samples <= 0 or not math.isfinite(directional_accuracy):
        return 0.05 * math.sqrt(max(0.01, min(1.0, coverage)))
    skill = max(0.05, directional_accuracy - 0.33 + 0.10)
    return float(skill * math.sqrt(max(0.01, min(1.0, coverage))))


def combine_challengers(timestamp: float, votes: Sequence[ChallengerVote],
                        familiarity: FamiliarityAssessment,
                        drift: DriftAssessment,
                        config: LeagueConfig = LeagueConfig()) -> LeagueDecision:
    if timestamp >= DEVELOPMENT_CUTOFF:
        raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
    active = tuple(vote for vote in votes if vote.validation_weight > 0)
    veto: list[str] = []
    if len(active) < config.min_challengers:
        veto.append("insufficient validation-qualified challengers")
    total_weight = sum(vote.validation_weight for vote in active)
    if total_weight <= 0:
        raw_edge = raw_confidence = agreement = 0.0
    else:
        raw_edge = sum(v.edge * v.validation_weight for v in active) / total_weight
        raw_confidence = sum(v.confidence * v.validation_weight for v in active) / total_weight
        direction = 1 if raw_edge > 0 else -1 if raw_edge < 0 else 0
        directional_weight = sum(
            v.validation_weight for v in active
            if v.action != "WAIT" and (1 if v.edge > 0 else -1 if v.edge < 0 else 0) == direction
        )
        agreement = directional_weight / total_weight if direction else 0.0

    if familiarity.out_of_distribution:
        veto.append("market familiarity gate: out-of-distribution")
    if drift.hard_drift:
        veto.append("concept drift gate: hard distribution shift")

    familiarity_scale = max(config.familiarity_floor, familiarity.familiarity)
    edge = raw_edge * familiarity_scale
    confidence = raw_confidence * familiarity_scale
    if drift.drifted and not drift.hard_drift:
        edge *= config.drift_penalty
        confidence *= config.drift_penalty

    action: Action = "WAIT"
    if not veto and abs(edge) >= config.entry_edge and confidence >= config.min_confidence and agreement >= config.min_agreement:
        action = "BUY" if edge > 0 else "SELL"

    dominant = None
    if active:
        dominant = max(active, key=lambda vote: (vote.validation_weight * max(vote.confidence, 1e-9), vote.name)).name

    return LeagueDecision(
        float(timestamp), action, float(max(-1.0, min(1.0, edge))),
        float(max(0.0, min(1.0, confidence))), float(max(0.0, min(1.0, agreement))),
        familiarity.familiarity, familiarity.out_of_distribution,
        drift.score, drift.threshold, drift.drifted, drift.hard_drift,
        dominant, active, tuple(veto), True,
    )


def probability_vote(name: str, prediction: object, weight: float,
                     *, edge_threshold: float = 0.12) -> ChallengerVote:
    down = float(getattr(prediction, "down"))
    neutral = float(getattr(prediction, "neutral"))
    up = float(getattr(prediction, "up"))
    edge = max(-1.0, min(1.0, up - down))
    confidence = max(up, down) * (1.0 - 0.25 * neutral)
    action: Action = "BUY" if edge >= edge_threshold else "SELL" if edge <= -edge_threshold else "WAIT"
    return ChallengerVote(name, action, edge, max(0.0, min(1.0, confidence)), weight,
                          "calibrated supervised challenger; validation-only weight")


def expert_reasoner_vote(decision: object, weight: float) -> ChallengerVote:
    action = str(getattr(decision, "action"))
    if action not in ("BUY", "SELL", "WAIT"):
        raise ValueError("invalid expert reasoner action")
    return ChallengerVote(
        "expert_reasoner", action, float(getattr(decision, "edge")),
        float(getattr(decision, "confidence")), weight,
        "Phase 2.8 expert reasoner challenger; validation-only weight",
    )


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")
