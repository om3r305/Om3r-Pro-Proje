from __future__ import annotations

import pytest

from brian2026.phase49_promotion_gate import PromotionReceipt
from brian2026.phase50_execution_reconciliation import ReconciliationBatchReport
from brian2026.phase58_advanced_overfit_audit import (
    AdvancedOverfitReport,
    CPCVPlan,
    DeflatedSharpeResult,
    PBOResult,
)
from brian2026.phase59_hardened_promotion_gate import (
    FinalValidationReceipt,
    evaluate_hardened_promotion,
)


def _preliminary(*, passed: bool = True) -> PromotionReceipt:
    checks = (
        ("research_challenger_passed", passed),
        ("robustness_passed", passed),
        ("causality_passed", passed),
        ("paper_parity_passed", passed),
    )
    return PromotionReceipt(
        status="MICRO_LIVE_ELIGIBLE" if passed else "RESEARCH_BLOCKED",
        candidate_id="candidate-59",
        checks=checks,
        reasons=() if passed else ("research_challenger_passed",),
        research_experiment_id="exp-59",
        paper_observations=40,
    )


def _advanced(*, passed: bool = True) -> AdvancedOverfitReport:
    cpcv = CPCVPlan(
        n_splits=6,
        n_test_splits=2,
        pct_embargo=0.01,
        combinations_count=15,
        backtest_paths=5,
        splits=(),
    )
    pbo = PBOResult(
        pbo=0.05 if passed else 0.60,
        logits=(),
        in_sample_best_sharpes=(),
        oos_sharpes_of_in_sample_winner=(),
        n_splits=15,
        n_blocks=6,
        n_strategies=10,
    )
    dsr = DeflatedSharpeResult(
        best_strategy_index=0,
        observed_sharpe=0.25,
        naive_psr=0.99,
        deflated_sharpe_probability=0.98 if passed else 0.40,
        selection_bias_benchmark_sharpe=0.10,
        n_observations=500,
        n_trials=10,
        sharpe_variance_across_trials=0.01,
        skew=0.0,
        raw_kurtosis=3.0,
    )
    checks = (
        ("cpcv_has_enough_paths", passed),
        ("pbo_within_limit", passed),
        ("deflated_sharpe_significant", passed),
    )
    return AdvancedOverfitReport(
        cpcv=cpcv,
        pbo=pbo,
        deflated_sharpe=dsr,
        checks=checks,
        status="ADVANCED_ROBUSTNESS_CANDIDATE" if passed else "OVERFIT_RISK",
    )


def _reconciliation(*, ready: bool = True) -> ReconciliationBatchReport:
    checks = (
        ("authoritative_reports_present", ready),
        ("positions_reconciled", ready),
        ("history_contract_acceptable", ready),
        ("no_unknown_command_outcomes", ready),
        ("no_duplicate_fill_ids", ready),
    )
    return ReconciliationBatchReport(
        results=(),
        tracked_assets=("BTCUSDT",),
        reports_complete=True,
        unresolved_command_ids=() if ready else ("unknown-cmd",),
        duplicate_fill_ids=(),
        checks=checks,
        ready=ready,
    )


def _final(*, passed: bool = True) -> FinalValidationReceipt:
    return FinalValidationReceipt(
        dataset_id="future-pristine-holdout-v1",
        dataset_seal_sha256="a" * 64,
        code_commit="abc123",
        sealed_at=100.0,
        evaluation_started_at=200.0,
        evaluation_finished_at=300.0,
        evaluation_count=1,
        used_for_tuning=False,
        contaminated=False,
        passed=passed,
        evidence_ref="final-evidence-59",
    )


def test_all_preconditions_yield_only_human_review_ready_not_live_activation() -> None:
    receipt = evaluate_hardened_promotion(
        _preliminary(),
        _advanced(),
        _reconciliation(),
        _final(),
    )
    assert receipt.status == "HUMAN_REVIEW_READY"
    assert all(dict(receipt.checks).values())
    assert receipt.blockers == ()
    assert receipt.human_authorization_required is True
    assert receipt.exchange_adapter_enabled is False
    assert receipt.capital_authorized is False
    assert receipt.automatic_activation is False
    assert receipt.live_execution is False


def test_missing_final_validation_is_an_explicit_hard_blocker() -> None:
    receipt = evaluate_hardened_promotion(
        _preliminary(),
        _advanced(),
        _reconciliation(),
        None,
    )
    assert receipt.status == "FINAL_VALIDATION_REQUIRED"
    assert "pristine_final_validation_passed" in receipt.blockers
    assert receipt.final_validation_evidence_ref is None
    assert receipt.live_execution is False


def test_preliminary_or_overfit_failure_blocks_before_execution_readiness() -> None:
    preliminary_bad = evaluate_hardened_promotion(
        _preliminary(passed=False),
        _advanced(),
        _reconciliation(),
        _final(),
    )
    assert preliminary_bad.status == "PRELIVE_RESEARCH_BLOCKED"

    overfit_bad = evaluate_hardened_promotion(
        _preliminary(),
        _advanced(passed=False),
        _reconciliation(),
        _final(),
    )
    assert overfit_bad.status == "PRELIVE_RESEARCH_BLOCKED"
    assert "phase58_advanced_overfit_passed" in overfit_bad.blockers


def test_unreconciled_execution_state_blocks_even_after_research_passes() -> None:
    receipt = evaluate_hardened_promotion(
        _preliminary(),
        _advanced(),
        _reconciliation(ready=False),
        _final(),
    )
    assert receipt.status == "EXECUTION_RECONCILIATION_BLOCKED"
    assert "phase50_execution_state_reconciled" in receipt.blockers
    assert receipt.reconciliation_ready is False


def test_final_validation_receipt_is_one_shot_pristine_and_presealed() -> None:
    with pytest.raises(ValueError, match="sealed before"):
        FinalValidationReceipt(
            dataset_id="x",
            dataset_seal_sha256="b" * 64,
            code_commit="abc",
            sealed_at=300.0,
            evaluation_started_at=200.0,
            evaluation_finished_at=400.0,
            evaluation_count=1,
            used_for_tuning=False,
            contaminated=False,
            passed=True,
            evidence_ref="ev",
        )

    with pytest.raises(ValueError, match="one-shot"):
        FinalValidationReceipt(
            dataset_id="x",
            dataset_seal_sha256="b" * 64,
            code_commit="abc",
            sealed_at=100.0,
            evaluation_started_at=200.0,
            evaluation_finished_at=400.0,
            evaluation_count=2,
            used_for_tuning=False,
            contaminated=False,
            passed=True,
            evidence_ref="ev",
        )

    with pytest.raises(ValueError, match="cannot be used for tuning"):
        FinalValidationReceipt(
            dataset_id="x",
            dataset_seal_sha256="b" * 64,
            code_commit="abc",
            sealed_at=100.0,
            evaluation_started_at=200.0,
            evaluation_finished_at=400.0,
            evaluation_count=1,
            used_for_tuning=True,
            contaminated=False,
            passed=True,
            evidence_ref="ev",
        )

    with pytest.raises(ValueError, match="contaminated"):
        FinalValidationReceipt(
            dataset_id="x",
            dataset_seal_sha256="b" * 64,
            code_commit="abc",
            sealed_at=100.0,
            evaluation_started_at=200.0,
            evaluation_finished_at=400.0,
            evaluation_count=1,
            used_for_tuning=False,
            contaminated=True,
            passed=True,
            evidence_ref="ev",
        )


def test_bad_final_dataset_hash_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        FinalValidationReceipt(
            dataset_id="x",
            dataset_seal_sha256="not-a-sha",
            code_commit="abc",
            sealed_at=100.0,
            evaluation_started_at=200.0,
            evaluation_finished_at=400.0,
            evaluation_count=1,
            used_for_tuning=False,
            contaminated=False,
            passed=True,
            evidence_ref="ev",
        )
