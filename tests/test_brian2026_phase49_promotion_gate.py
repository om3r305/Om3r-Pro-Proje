from __future__ import annotations

import pytest

from brian2026.phase41_robustness_lab import (
    MonteCarloSummary,
    RobustnessReport,
    RuleSignificanceResult,
)
from brian2026.phase42_autonomous_alpha_lab import AlphaExperimentOutcome
from brian2026.phase48_causality_auditor import (
    CausalityPromotionGate,
    LookaheadAuditReport,
    TemporalAuditResult,
)
from brian2026.phase49_promotion_gate import (
    PaperParityPolicy,
    ShadowPaperObservation,
    evaluate_micro_live_eligibility,
    evaluate_shadow_paper_parity,
)


def _research(*, passed: bool = True) -> AlphaExperimentOutcome:
    checks = (
        ("enough_locked_folds", passed),
        ("mean_accuracy_improved", passed),
        ("mean_brier_not_worse", passed),
        ("no_material_fold_regression", passed),
    )
    return AlphaExperimentOutcome(
        experiment_id="exp-49",
        candidate_id="candidate-49",
        comparisons=(),
        mean_balanced_accuracy_delta=0.02 if passed else -0.01,
        mean_brier_delta=-0.01 if passed else 0.02,
        checks=checks,
        decision="RESEARCH_CHALLENGER_CANDIDATE" if passed else "REJECTED_EXPERIMENT",
    )


def _mc(method: str) -> MonteCarloSummary:
    return MonteCarloSummary(
        method=method,
        trials=1000,
        seed=1,
        median_return_pct=5.0,
        p05_return_pct=1.0,
        p95_return_pct=9.0,
        median_max_drawdown_pct=3.0,
        p95_max_drawdown_pct=6.0,
        loss_probability=0.02,
        ruin_probability=0.0,
    )


def _robust(*, passed: bool = True) -> RobustnessReport:
    checks = (
        ("block_p05_return_meets_floor", passed),
        ("trade_order_drawdown_within_limit", passed),
        ("block_loss_probability_within_limit", passed),
        ("trade_order_ruin_within_limit", passed),
        ("block_ruin_within_limit", passed),
        ("rule_significance_passes", passed),
    )
    return RobustnessReport(
        trade_order=_mc("trade_order_shuffle"),
        block_bootstrap=_mc("contiguous_block_bootstrap"),
        rule_significance=RuleSignificanceResult(
            trials=2000,
            seed=2,
            active_count=20,
            opportunity_count=100,
            observed_mean_return=0.01,
            random_mean_return=0.0,
            lift=0.01,
            p_value=0.01 if passed else 0.5,
        ),
        checks=checks,
        status="ROBUSTNESS_CANDIDATE" if passed else "INSUFFICIENT_ROBUSTNESS",
    )


def _causal(*, passed: bool = True) -> CausalityPromotionGate:
    return CausalityPromotionGate(
        temporal=TemporalAuditResult(passed=passed, future_dependencies=() if passed else ("future",), checked=10),
        lookahead=LookaheadAuditReport(
            status="PASS_CAUSALITY" if passed else "FAIL_LOOKAHEAD",
            probes_requested=10,
            probes_completed=10,
            minimum_probes=5,
            biased_signal_fields=() if passed else ("entry",),
            biased_indicator_fields=(),
            differences=(),
        ),
        partition_contract_passed=passed,
        eligible=passed,
        reasons=() if passed else ("future source availability detected",),
    )


def _parity_rows(count: int = 40, *, adverse_bps: float = 4.0) -> tuple[ShadowPaperObservation, ...]:
    rows = []
    for index in range(count):
        reference = 100.0 + index
        paper = reference * (1.0 + adverse_bps / 10_000.0)
        rows.append(ShadowPaperObservation(
            intent_id=f"intent-{index}",
            shadow_direction=1,
            paper_direction=1,
            reference_price=reference,
            paper_fill_price=paper,
            shadow_fill_fraction=1.0,
            paper_fill_fraction=0.98,
            paper_acknowledged=True,
            reconciliation_complete=True,
            ambiguous_outcome=False,
            observed_at=1_760_000_000.0 + index,
        ))
    return tuple(rows)


def test_clean_shadow_paper_parity_passes_preregistered_execution_checks() -> None:
    report = evaluate_shadow_paper_parity(
        _parity_rows(),
        policy=PaperParityPolicy(
            min_observations=30,
            min_direction_match_rate=0.98,
            min_acknowledgement_rate=0.99,
            min_reconciliation_rate=1.0,
            min_mean_fill_fraction_ratio=0.90,
            max_p95_adverse_execution_drift_bps=8.0,
            max_ambiguous_outcomes=0,
        ),
    )
    assert report.status == "PASS_PAPER_PARITY"
    assert report.direction_match_rate == pytest.approx(1.0)
    assert report.acknowledgement_rate == pytest.approx(1.0)
    assert report.reconciliation_rate == pytest.approx(1.0)
    assert report.mean_fill_fraction_ratio == pytest.approx(0.98)
    assert report.p95_adverse_execution_drift_bps == pytest.approx(4.0, abs=1e-9)
    assert all(dict(report.checks).values())
    assert report.shadow_only is True
    assert report.live_execution is False


def test_ambiguous_or_unreconciled_outcomes_block_paper_parity() -> None:
    rows = list(_parity_rows())
    bad = rows[0]
    rows[0] = ShadowPaperObservation(
        intent_id=bad.intent_id,
        shadow_direction=bad.shadow_direction,
        paper_direction=-1,
        reference_price=bad.reference_price,
        paper_fill_price=bad.paper_fill_price,
        shadow_fill_fraction=1.0,
        paper_fill_fraction=0.5,
        paper_acknowledged=False,
        reconciliation_complete=False,
        ambiguous_outcome=True,
        observed_at=bad.observed_at,
    )
    report = evaluate_shadow_paper_parity(tuple(rows))
    assert report.status == "FAIL_PAPER_PARITY"
    assert report.ambiguous_outcomes == 1
    checks = dict(report.checks)
    assert checks["no_unresolved_ambiguous_outcomes"] is False
    assert checks["reconciliation_complete_enough"] is False


def test_promotion_gate_requires_research_robustness_causality_and_paper_parity() -> None:
    parity = evaluate_shadow_paper_parity(_parity_rows())
    receipt = evaluate_micro_live_eligibility(
        _research(),
        _robust(),
        _causal(),
        parity,
    )
    assert receipt.status == "MICRO_LIVE_ELIGIBLE"
    assert all(dict(receipt.checks).values())
    assert receipt.human_authorization_required is True
    assert receipt.exchange_adapter_enabled is False
    assert receipt.automatic_activation is False
    assert receipt.shadow_only is True
    assert receipt.live_execution is False


def test_missing_paper_parity_stays_pending_and_never_enables_execution() -> None:
    receipt = evaluate_micro_live_eligibility(
        _research(),
        _robust(),
        _causal(),
        None,
    )
    assert receipt.status == "PAPER_PARITY_PENDING"
    assert "paper_parity_passed" in receipt.reasons
    assert receipt.exchange_adapter_enabled is False
    assert receipt.live_execution is False


@pytest.mark.parametrize(
    "research_ok,robust_ok,causal_ok",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_any_scientific_gate_failure_blocks_before_paper(
    research_ok: bool,
    robust_ok: bool,
    causal_ok: bool,
) -> None:
    receipt = evaluate_micro_live_eligibility(
        _research(passed=research_ok),
        _robust(passed=robust_ok),
        _causal(passed=causal_ok),
        evaluate_shadow_paper_parity(_parity_rows()),
    )
    assert receipt.status == "RESEARCH_BLOCKED"
    assert receipt.live_execution is False
    assert receipt.automatic_activation is False


def test_duplicate_intent_ids_fail_closed() -> None:
    rows = list(_parity_rows(5))
    rows[1] = ShadowPaperObservation(
        intent_id=rows[0].intent_id,
        shadow_direction=1,
        paper_direction=1,
        reference_price=101.0,
        paper_fill_price=101.01,
        shadow_fill_fraction=1.0,
        paper_fill_fraction=1.0,
        paper_acknowledged=True,
        reconciliation_complete=True,
        ambiguous_outcome=False,
        observed_at=1_760_000_100.0,
    )
    with pytest.raises(ValueError, match="unique intent ids"):
        evaluate_shadow_paper_parity(tuple(rows))
