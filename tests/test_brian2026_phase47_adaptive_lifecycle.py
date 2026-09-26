from __future__ import annotations

import pytest

from brian2026.adaptive_quant import DriftAssessment, FamiliarityAssessment
from brian2026.phase47_adaptive_lifecycle import (
    AdaptiveLifecycleConfig,
    AdaptiveModelLifecycle,
    ModelArtifact,
)


def _familiarity(*, ood: bool = False) -> FamiliarityAssessment:
    return FamiliarityAssessment(
        score=2.0 if ood else 0.2,
        threshold=1.0,
        familiarity=0.1 if ood else 0.9,
        observed_fraction=1.0,
        out_of_distribution=ood,
    )


def _drift(*, drifted: bool = False, hard: bool = False) -> DriftAssessment:
    return DriftAssessment(
        score=2.0 if hard else 1.1 if drifted else 0.2,
        threshold=1.0,
        drifted=drifted or hard,
        hard_drift=hard,
        observations=96,
    )


def _model(
    model_id: str = "m1",
    *,
    trained_at: float = 1000.0,
    research: bool = True,
    robust: bool = True,
) -> ModelArtifact:
    return ModelArtifact(
        model_id=model_id,
        asset_id="BTCUSDT",
        trained_at=trained_at,
        training_dataset_id=f"dataset-{model_id}",
        code_fingerprint=f"code-{model_id}",
        research_gate_passed=research,
        robustness_gate_passed=robust,
    )


def test_initial_model_requires_both_scientific_gates() -> None:
    lifecycle = AdaptiveModelLifecycle()
    with pytest.raises(ValueError, match="research and robustness"):
        lifecycle.install_initial_model(_model(research=False))
    with pytest.raises(ValueError, match="research and robustness"):
        lifecycle.install_initial_model(_model(robust=False))


def test_current_model_is_allowed_before_retrain_interval() -> None:
    lifecycle = AdaptiveModelLifecycle(
        AdaptiveLifecycleConfig(
            min_retrain_interval_seconds=1800.0,
            expiration_seconds=7200.0,
        )
    )
    lifecycle.install_initial_model(_model(trained_at=1000.0))
    result = lifecycle.assess(
        "BTCUSDT",
        now=1600.0,
        familiarity=_familiarity(),
        drift=_drift(),
    )
    assert result.state == "ACTIVE"
    assert result.permission == "ALLOW"
    assert result.risk_scale == 1.0
    assert result.retrain_request is None


def test_periodic_retrain_requests_new_challenger_but_keeps_current_model_active() -> None:
    lifecycle = AdaptiveModelLifecycle(
        AdaptiveLifecycleConfig(
            min_retrain_interval_seconds=1800.0,
            expiration_seconds=7200.0,
        )
    )
    lifecycle.install_initial_model(_model(trained_at=1000.0))
    result = lifecycle.assess(
        "BTCUSDT",
        now=3000.0,
        familiarity=_familiarity(),
        drift=_drift(),
    )
    assert result.state == "RETRAIN_REQUESTED"
    assert result.permission == "ALLOW"
    assert result.active_model_id == "m1"
    assert result.retrain_request is not None
    assert result.retrain_request.automatic_promotion is False
    assert lifecycle.active_model("BTCUSDT").model_id == "m1"


def test_soft_drift_reduces_risk_and_hard_drift_blocks_predictions() -> None:
    lifecycle = AdaptiveModelLifecycle(
        AdaptiveLifecycleConfig(
            min_retrain_interval_seconds=600.0,
            expiration_seconds=7200.0,
            soft_drift_risk_scale=0.4,
        )
    )
    lifecycle.install_initial_model(_model(trained_at=1000.0))

    soft = lifecycle.assess(
        "BTCUSDT",
        now=1700.0,
        familiarity=_familiarity(),
        drift=_drift(drifted=True),
    )
    assert soft.permission == "REDUCE_RISK"
    assert soft.risk_scale == pytest.approx(0.4)
    assert soft.retrain_request is not None
    assert lifecycle.active_model("BTCUSDT").model_id == "m1"

    hard = lifecycle.assess(
        "BTCUSDT",
        now=1800.0,
        familiarity=_familiarity(),
        drift=_drift(hard=True),
    )
    assert hard.state == "HARD_DRIFT"
    assert hard.permission == "BLOCK"
    assert hard.risk_scale == 0.0
    assert lifecycle.active_model("BTCUSDT").model_id == "m1"


def test_ood_and_expired_model_fail_closed() -> None:
    ood_lifecycle = AdaptiveModelLifecycle(
        AdaptiveLifecycleConfig(
            min_retrain_interval_seconds=600.0,
            expiration_seconds=7200.0,
        )
    )
    ood_lifecycle.install_initial_model(_model(trained_at=1000.0))
    ood = ood_lifecycle.assess(
        "BTCUSDT",
        now=1700.0,
        familiarity=_familiarity(ood=True),
        drift=_drift(),
    )
    assert ood.permission == "BLOCK"
    assert "OUT_OF_DISTRIBUTION" in ood.reasons

    expired_lifecycle = AdaptiveModelLifecycle(
        AdaptiveLifecycleConfig(
            min_retrain_interval_seconds=600.0,
            expiration_seconds=1000.0,
        )
    )
    expired_lifecycle.install_initial_model(_model(trained_at=1000.0))
    expired = expired_lifecycle.assess(
        "BTCUSDT",
        now=2501.0,
        familiarity=_familiarity(),
        drift=_drift(),
    )
    assert expired.state == "EXPIRED"
    assert expired.permission == "BLOCK"
    assert expired.retrain_request is not None


def test_retrain_request_does_not_promote_and_challenger_requires_gates() -> None:
    lifecycle = AdaptiveModelLifecycle(
        AdaptiveLifecycleConfig(
            min_retrain_interval_seconds=600.0,
            expiration_seconds=7200.0,
        )
    )
    lifecycle.install_initial_model(_model(trained_at=1000.0))
    lifecycle.assess(
        "BTCUSDT",
        now=1700.0,
        familiarity=_familiarity(),
        drift=_drift(drifted=True),
    )

    with pytest.raises(ValueError, match="Phase 42 and Phase 41"):
        lifecycle.submit_challenger(
            _model("m2-bad", trained_at=1800.0, robust=False)
        )
    assert lifecycle.active_model("BTCUSDT").model_id == "m1"

    lifecycle.submit_challenger(_model("m2", trained_at=1800.0))
    assert lifecycle.active_model("BTCUSDT").model_id == "m2"
    assert lifecycle.pending_request("BTCUSDT") is None
    manifest = lifecycle.manifest()
    assert manifest["retrain_does_not_imply_promotion"] is True
    assert manifest["automatic_promotion"] is False
    assert manifest["live_execution"] is False
