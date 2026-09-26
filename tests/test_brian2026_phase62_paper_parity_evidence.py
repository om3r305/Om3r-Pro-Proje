from __future__ import annotations

import pytest

from brian2026.phase46_execution_simulator import SimulatedExecutionReceipt
from brian2026.phase49_promotion_gate import PaperParityPolicy
from brian2026.phase50_execution_reconciliation import (
    PositionReconciliationResult,
    ReconciliationBatchReport,
)
from brian2026.phase56_pretrade_risk_engine import PreTradeRiskReceipt
from brian2026.phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from brian2026.phase61_stateful_paper_venue import (
    PaperCycleReceipt,
    PaperOrderOutcome,
)
from brian2026.phase62_paper_parity_evidence import (
    PaperParityEvidenceConflictError,
    PaperParityEvidenceLedger,
    build_cycle_parity_observations,
)


TS = 1_760_000_000.0


def _risk(asset: str, *, allowed: bool = True) -> PreTradeRiskReceipt:
    return PreTradeRiskReceipt(
        action="ALLOW" if allowed else "DENY",
        trading_state="ACTIVE",
        asset_id=asset,
        requested_notional_usd=100.0,
        reduce_only=False,
        reasons=() if allowed else ("denied",),
        projected_position_weight=0.1,
        checks=(("fixture", allowed),),
    )


def _execution(
    *,
    side: str = "BUY",
    status: str = "FILLED",
    fill_fraction: float = 1.0,
    price: float = 100.0,
) -> SimulatedExecutionReceipt:
    requested = 1.0
    filled = requested * fill_fraction
    return SimulatedExecutionReceipt(
        status=status,
        side=side,
        order_type="MARKET",
        submit_timestamp=TS,
        venue_timestamp=TS + 0.1,
        snapshot_timestamp=TS + 0.1,
        requested_base=requested,
        filled_base=filled,
        fill_fraction=fill_fraction,
        average_fill_price=price if status != "NO_FILL" else None,
        best_reference_price=100.0,
        adverse_slippage_bps=0.0,
        levels_consumed=1 if filled else 0,
        slipped_one_tick=False,
        reason="fixture",
    )


def _cycle(
    cycle_id: str,
    *,
    asset: str = "BTCUSDT",
    allowed: bool = True,
    execution: SimulatedExecutionReceipt | None = None,
) -> ShadowExecutionCycle:
    item = ShadowExecutionCycleItem(
        instruction_kind="OPEN",
        asset_id=asset,
        risk_receipt=_risk(asset, allowed=allowed),
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=100.0 if allowed else 0.0,
        status="fixture",
    )
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=(item,),
        initial_available_cash_usd=500.0,
        reserved_new_risk_cash_usd=100.0 if allowed else 0.0,
        remaining_unreserved_cash_usd=400.0 if allowed else 500.0,
        denied_assets=() if allowed else (asset,),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def _paper(
    cycle: ShadowExecutionCycle,
    *,
    status: str = "FILLED",
    acknowledged: bool = True,
    fill_fraction: float = 1.0,
    price: float | None = 100.0,
) -> PaperCycleReceipt:
    item = cycle.items[0]
    paper_order_id = f"paper-{cycle.cycle_id}"
    outcome = PaperOrderOutcome(
        paper_order_id=paper_order_id,
        cycle_id=cycle.cycle_id,
        asset_id=item.asset_id,
        instruction_kind=item.instruction_kind,
        status=status,
        acknowledged=acknowledged,
        requested_notional_usd=item.risk_receipt.requested_notional_usd,
        requested_base=0.0 if item.execution_receipt is None else item.execution_receipt.requested_base,
        filled_base=(
            0.0
            if item.execution_receipt is None
            else item.execution_receipt.requested_base * fill_fraction
        ),
        fill_fraction=fill_fraction,
        average_fill_price=price,
        fill_ids=() if fill_fraction <= 0 else (f"fill-{cycle.cycle_id}",),
        reason="fixture",
    )
    from brian2026.evidence_ledger import content_hash
    cycle_hash = content_hash(cycle.to_dict())
    return PaperCycleReceipt(
        cycle_id=cycle.cycle_id,
        cycle_hash=cycle_hash,
        outcomes=(outcome,),
        fill_ids=outcome.fill_ids,
        cash_before_usd=500.0,
        cash_after_usd=400.0,
        state_version_before=0,
        state_version_after=1,
        receipt_id=f"receipt-{cycle.cycle_id}",
    )


def _reconciliation(
    asset: str = "BTCUSDT",
    *,
    ready: bool = True,
    resolved: bool = True,
) -> ReconciliationBatchReport:
    result = PositionReconciliationResult(
        account_id="paper-acct",
        asset_id=asset,
        status="MATCHED" if resolved else "UNRESOLVED",
        local_quantity=1.0 if resolved else 0.5,
        venue_quantity=1.0,
        quantity_difference=0.0 if resolved else 0.5,
        local_avg_entry_price=100.0,
        venue_avg_entry_price=100.0,
        quantity_within_tolerance=resolved,
        entry_price_within_tolerance=True,
        recovery_events=(),
        reasons=() if resolved else ("mismatch",),
        authoritative_report_present=True,
    )
    checks = (
        ("authoritative_reports_present", ready),
        ("positions_reconciled", ready),
        ("history_contract_acceptable", ready),
        ("no_unknown_command_outcomes", ready),
        ("no_duplicate_fill_ids", ready),
    )
    return ReconciliationBatchReport(
        results=(result,),
        tracked_assets=(asset,),
        reports_complete=True,
        unresolved_command_ids=(),
        duplicate_fill_ids=(),
        checks=checks,
        ready=ready,
    )


def test_filled_shadow_and_paper_cycle_builds_reconciled_parity_observation() -> None:
    cycle = _cycle("c1", execution=_execution())
    paper = _paper(cycle)
    observations = build_cycle_parity_observations(
        cycle,
        paper,
        _reconciliation(),
        reference_prices={"BTCUSDT": 100.0},
        observed_at=TS + 1,
    )
    assert len(observations) == 1
    row = observations[0]
    assert row.intent_id == "paper-c1"
    assert row.shadow_direction == 1
    assert row.paper_direction == 1
    assert row.shadow_fill_fraction == pytest.approx(1.0)
    assert row.paper_fill_fraction == pytest.approx(1.0)
    assert row.paper_acknowledged is True
    assert row.reconciliation_complete is True
    assert row.ambiguous_outcome is False


def test_definitive_paper_rejection_surfaces_direction_and_ack_mismatch() -> None:
    cycle = _cycle("reject", execution=_execution())
    paper = _paper(
        cycle,
        status="VENUE_REJECTED_BALANCE",
        acknowledged=False,
        fill_fraction=0.0,
        price=None,
    )
    row = build_cycle_parity_observations(
        cycle,
        paper,
        _reconciliation(),
        reference_prices={"BTCUSDT": 100.0},
        observed_at=TS + 1,
    )[0]
    assert row.shadow_direction == 1
    assert row.paper_direction == 0
    assert row.direction_matches is False
    assert row.paper_acknowledged is False
    assert row.paper_fill_fraction == pytest.approx(0.0)
    assert row.ambiguous_outcome is False


def test_unreconciled_asset_marks_observation_ambiguous() -> None:
    cycle = _cycle("ambiguous", execution=_execution())
    paper = _paper(cycle)
    row = build_cycle_parity_observations(
        cycle,
        paper,
        _reconciliation(ready=False, resolved=False),
        reference_prices={"BTCUSDT": 100.0},
        observed_at=TS + 1,
    )[0]
    assert row.reconciliation_complete is False
    assert row.ambiguous_outcome is True


def test_risk_denied_and_local_veto_are_not_submitted_parity_observations() -> None:
    denied = _cycle("denied", allowed=False, execution=None)
    denied_paper = _paper(
        denied,
        status="RISK_DENIED",
        acknowledged=False,
        fill_fraction=0.0,
        price=None,
    )
    assert build_cycle_parity_observations(
        denied,
        denied_paper,
        _reconciliation(),
        reference_prices={"BTCUSDT": 100.0},
        observed_at=TS + 1,
    ) == ()

    veto = _cycle(
        "veto",
        execution=_execution(status="VETO_SLIPPAGE", fill_fraction=0.0),
    )
    veto_paper = _paper(
        veto,
        status="LOCAL_VETO",
        acknowledged=False,
        fill_fraction=0.0,
        price=100.0,
    )
    assert build_cycle_parity_observations(
        veto,
        veto_paper,
        _reconciliation(),
        reference_prices={"BTCUSDT": 100.0},
        observed_at=TS + 1,
    ) == ()


def test_ledger_aggregates_real_cycle_evidence_and_evaluates_phase49_policy() -> None:
    ledger = PaperParityEvidenceLedger()
    for index in range(5):
        cycle = _cycle(f"c{index}", execution=_execution(price=100.02))
        paper = _paper(cycle, price=100.02)
        receipt = ledger.append_cycle(
            cycle,
            paper,
            _reconciliation(),
            reference_prices={"BTCUSDT": 100.0},
            observed_at=TS + index + 1,
        )
        assert receipt.observations_added == 1

    report = ledger.evaluate(
        policy=PaperParityPolicy(
            min_observations=5,
            min_direction_match_rate=1.0,
            min_acknowledgement_rate=1.0,
            min_reconciliation_rate=1.0,
            min_mean_fill_fraction_ratio=1.0,
            max_p95_adverse_execution_drift_bps=5.0,
            max_ambiguous_outcomes=0,
        )
    )
    assert report.status == "PASS_PAPER_PARITY"
    assert report.observations == 5
    assert all(dict(report.checks).values())


def test_duplicate_cycle_is_idempotent_but_changed_evidence_conflicts() -> None:
    ledger = PaperParityEvidenceLedger()
    cycle = _cycle("idem", execution=_execution())
    paper = _paper(cycle)
    kwargs = dict(
        cycle=cycle,
        paper=paper,
        reconciliation=_reconciliation(),
        reference_prices={"BTCUSDT": 100.0},
        observed_at=TS + 1,
    )
    first = ledger.append_cycle(**kwargs)
    duplicate = ledger.append_cycle(**kwargs)
    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.total_observations == 1

    changed = _paper(cycle, price=100.10)
    with pytest.raises(PaperParityEvidenceConflictError, match="different parity evidence"):
        ledger.append_cycle(
            cycle,
            changed,
            _reconciliation(),
            reference_prices={"BTCUSDT": 100.0},
            observed_at=TS + 1,
        )


def test_cycle_hash_mismatch_and_missing_reference_fail_closed() -> None:
    cycle = _cycle("bad", execution=_execution())
    paper = _paper(cycle)
    broken = PaperCycleReceipt(
        cycle_id=paper.cycle_id,
        cycle_hash="0" * 64,
        outcomes=paper.outcomes,
        fill_ids=paper.fill_ids,
        cash_before_usd=paper.cash_before_usd,
        cash_after_usd=paper.cash_after_usd,
        state_version_before=paper.state_version_before,
        state_version_after=paper.state_version_after,
        receipt_id=paper.receipt_id,
    )
    with pytest.raises(PaperParityEvidenceConflictError, match="cycle hash"):
        build_cycle_parity_observations(
            cycle,
            broken,
            _reconciliation(),
            reference_prices={"BTCUSDT": 100.0},
            observed_at=TS + 1,
        )

    with pytest.raises(KeyError, match="missing parity reference price"):
        build_cycle_parity_observations(
            cycle,
            paper,
            _reconciliation(),
            reference_prices={},
            observed_at=TS + 1,
        )


def test_manifest_is_append_only_and_shadow_only() -> None:
    ledger = PaperParityEvidenceLedger()
    cycle = _cycle("manifest", execution=_execution())
    ledger.append_cycle(
        cycle,
        _paper(cycle),
        _reconciliation(),
        reference_prices={"BTCUSDT": 100.0},
        observed_at=TS + 1,
    )
    manifest = ledger.manifest()
    assert manifest["append_only"] is True
    assert manifest["cycle_count"] == 1
    assert manifest["observation_count"] == 1
    assert manifest["shadow_only"] is True
    assert manifest["live_execution"] is False
    assert isinstance(manifest["ledger_hash"], str)
