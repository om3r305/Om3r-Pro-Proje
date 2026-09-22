from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

from .adaptive_quant import DriftAssessment, FamiliarityAssessment

PHASE47_SCHEMA_VERSION = "brian.phase47-adaptive-lifecycle.v1"
LifecycleState = Literal[
    "NO_MODEL",
    "ACTIVE",
    "SOFT_DRIFT",
    "RETRAIN_REQUESTED",
    "CHALLENGER_PENDING",
    "EXPIRED",
    "HARD_DRIFT",
]
PredictionPermission = Literal["ALLOW", "REDUCE_RISK", "BLOCK"]


@dataclass(frozen=True, slots=True)
class AdaptiveLifecycleConfig:
    min_retrain_interval_seconds: float = 30 * 60.0
    expiration_seconds: float = 4 * 60 * 60.0
    soft_drift_risk_scale: float = 0.50
    hard_drift_blocks_predictions: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_retrain_interval_seconds) or self.min_retrain_interval_seconds < 0:
            raise ValueError("min_retrain_interval_seconds must be non-negative")
        if not math.isfinite(self.expiration_seconds) or self.expiration_seconds <= 0:
            raise ValueError("expiration_seconds must be positive")
        if not 0 < self.soft_drift_risk_scale <= 1:
            raise ValueError("soft_drift_risk_scale must be in (0,1]")


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    model_id: str
    asset_id: str
    trained_at: float
    training_dataset_id: str
    code_fingerprint: str
    research_gate_passed: bool
    robustness_gate_passed: bool
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = PHASE47_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (
            self.model_id,
            self.asset_id,
            self.training_dataset_id,
            self.code_fingerprint,
        )):
            raise ValueError("model artifact identity fields are required")
        if not math.isfinite(self.trained_at):
            raise ValueError("trained_at must be finite")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase 47 model artifacts are shadow-only")


@dataclass(frozen=True, slots=True)
class RetrainRequest:
    asset_id: str
    requested_at: float
    reason: str
    parent_model_id: str | None
    train_from_scratch: bool
    shadow_only: bool = True
    automatic_promotion: bool = False

    def __post_init__(self) -> None:
        if not self.asset_id.strip() or not self.reason.strip():
            raise ValueError("retrain request requires asset and reason")
        if not math.isfinite(self.requested_at):
            raise ValueError("requested_at must be finite")
        if self.automatic_promotion:
            raise ValueError("retrain requests cannot auto-promote")


@dataclass(frozen=True, slots=True)
class LifecycleAssessment:
    asset_id: str
    state: LifecycleState
    permission: PredictionPermission
    risk_scale: float
    active_model_id: str | None
    model_age_seconds: float | None
    retrain_due: bool
    retrain_request: RetrainRequest | None
    reasons: tuple[str, ...]
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


class AdaptiveModelLifecycle:
    """Age/drift-aware shadow model registry.

    Behavior mirrors FreqAI's operational lifecycle without reusing GPL code:
    the newest accepted model is active, model age can expire prediction
    eligibility, retraining is rate-limited, and drift can request a new model.
    Brian-specific safety is stricter: retraining only creates a request;
    challenger acceptance still requires Phase 42 + Phase 41 gates and never
    replaces the active model automatically.
    """

    def __init__(self, config: AdaptiveLifecycleConfig = AdaptiveLifecycleConfig()) -> None:
        self.config = config
        self._active: dict[str, ModelArtifact] = {}
        self._last_retrain_request_at: dict[str, float] = {}
        self._pending: dict[str, RetrainRequest] = {}

    def active_model(self, asset_id: str) -> ModelArtifact | None:
        return self._active.get(asset_id)

    def pending_request(self, asset_id: str) -> RetrainRequest | None:
        return self._pending.get(asset_id)

    def install_initial_model(self, artifact: ModelArtifact) -> None:
        if not artifact.research_gate_passed or not artifact.robustness_gate_passed:
            raise ValueError("initial model requires research and robustness gates")
        prior = self._active.get(artifact.asset_id)
        if prior is not None and artifact.trained_at < prior.trained_at:
            raise ValueError("cannot replace active model with an older artifact")
        self._active[artifact.asset_id] = artifact

    def submit_challenger(self, artifact: ModelArtifact) -> None:
        """Accept only a previously gated challenger; never from a retrain request alone."""
        if not artifact.research_gate_passed or not artifact.robustness_gate_passed:
            raise ValueError("challenger must pass Phase 42 and Phase 41 gates")
        request = self._pending.get(artifact.asset_id)
        if request is None:
            raise ValueError("no pending retrain request for challenger")
        if artifact.trained_at < request.requested_at:
            raise ValueError("challenger predates its retrain request")
        active = self._active.get(artifact.asset_id)
        if active is not None and artifact.trained_at <= active.trained_at:
            raise ValueError("challenger must be newer than active model")
        self._active[artifact.asset_id] = artifact
        del self._pending[artifact.asset_id]

    def _can_request(self, asset_id: str, now: float) -> bool:
        last = self._last_retrain_request_at.get(asset_id)
        return last is None or now - last >= self.config.min_retrain_interval_seconds

    def _request(
        self,
        asset_id: str,
        now: float,
        reason: str,
        *,
        train_from_scratch: bool,
    ) -> RetrainRequest | None:
        existing = self._pending.get(asset_id)
        if existing is not None:
            return existing
        if not self._can_request(asset_id, now):
            return None
        active = self._active.get(asset_id)
        request = RetrainRequest(
            asset_id=asset_id,
            requested_at=now,
            reason=reason,
            parent_model_id=active.model_id if active else None,
            train_from_scratch=train_from_scratch,
        )
        self._pending[asset_id] = request
        self._last_retrain_request_at[asset_id] = now
        return request

    def assess(
        self,
        asset_id: str,
        *,
        now: float,
        familiarity: FamiliarityAssessment,
        drift: DriftAssessment,
    ) -> LifecycleAssessment:
        if not asset_id.strip() or not math.isfinite(now):
            raise ValueError("asset_id and finite now are required")
        active = self._active.get(asset_id)
        reasons: list[str] = []
        request: RetrainRequest | None = self._pending.get(asset_id)

        if active is None:
            reasons.append("no accepted model available")
            request = request or self._request(
                asset_id,
                now,
                "NO_MODEL",
                train_from_scratch=True,
            )
            return LifecycleAssessment(
                asset_id, "NO_MODEL", "BLOCK", 0.0, None, None,
                request is not None, request, tuple(reasons),
            )

        if now < active.trained_at:
            raise ValueError("runtime clock precedes active model training timestamp")
        age = now - active.trained_at

        if drift.hard_drift or familiarity.out_of_distribution:
            reason = "HARD_DRIFT" if drift.hard_drift else "OUT_OF_DISTRIBUTION"
            reasons.append(reason)
            request = request or self._request(
                asset_id,
                now,
                reason,
                train_from_scratch=True,
            )
            return LifecycleAssessment(
                asset_id,
                "HARD_DRIFT",
                "BLOCK" if self.config.hard_drift_blocks_predictions else "REDUCE_RISK",
                0.0 if self.config.hard_drift_blocks_predictions else self.config.soft_drift_risk_scale,
                active.model_id,
                age,
                request is not None,
                request,
                tuple(reasons),
            )

        if age > self.config.expiration_seconds:
            reasons.append("MODEL_EXPIRED")
            request = request or self._request(
                asset_id,
                now,
                "MODEL_EXPIRED",
                train_from_scratch=True,
            )
            return LifecycleAssessment(
                asset_id, "EXPIRED", "BLOCK", 0.0, active.model_id, age,
                request is not None, request, tuple(reasons),
            )

        periodic_due = age >= self.config.min_retrain_interval_seconds
        if drift.drifted:
            reasons.append("SOFT_DRIFT")
            request = request or self._request(
                asset_id,
                now,
                "SOFT_DRIFT",
                train_from_scratch=True,
            )
            return LifecycleAssessment(
                asset_id,
                "RETRAIN_REQUESTED" if request is not None else "SOFT_DRIFT",
                "REDUCE_RISK",
                self.config.soft_drift_risk_scale,
                active.model_id,
                age,
                request is not None,
                request,
                tuple(reasons),
            )

        if periodic_due:
            reasons.append("PERIODIC_RETRAIN_DUE")
            request = request or self._request(
                asset_id,
                now,
                "PERIODIC_RETRAIN_DUE",
                train_from_scratch=True,
            )
            if request is not None:
                return LifecycleAssessment(
                    asset_id, "RETRAIN_REQUESTED", "ALLOW", 1.0,
                    active.model_id, age, True, request, tuple(reasons),
                )

        if request is not None:
            reasons.append("CHALLENGER_PENDING")
            return LifecycleAssessment(
                asset_id, "CHALLENGER_PENDING", "ALLOW", 1.0,
                active.model_id, age, True, request, tuple(reasons),
            )

        return LifecycleAssessment(
            asset_id, "ACTIVE", "ALLOW", 1.0, active.model_id, age,
            False, None, ("MODEL_CURRENT",),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": PHASE47_SCHEMA_VERSION,
            "active_models": {
                asset: asdict(model) for asset, model in sorted(self._active.items())
            },
            "pending_retrain_requests": {
                asset: asdict(request) for asset, request in sorted(self._pending.items())
            },
            "config": asdict(self.config),
            "newest_accepted_model_wins": True,
            "expired_models_block_predictions": True,
            "retrain_does_not_imply_promotion": True,
            "shadow_only": True,
            "live_execution": False,
            "automatic_promotion": False,
        }
