from __future__ import annotations

import pytest

from brian2026.phase50_execution_reconciliation import (
    LocalPositionState,
    ReconciliationBatchPolicy,
    VenuePositionReport,
    reconcile_execution_state,
    reconcile_position,
)


def _local(quantity: float = 1.0, avg: float | None = 100.0) -> LocalPositionState:
    return LocalPositionState("acct", "BTCUSDT", quantity, avg, ("fill-1",))


def _venue(quantity: float = 1.0, avg: float | None = 100.0) -> VenuePositionReport:
    return VenuePositionReport("acct", "BTCUSDT", quantity, avg, True, "venue-pos-1")


def test_explicit_matching_position_is_resolved() -> None:
    result = reconcile_position(
        _local(),
        _venue(),
        account_id="acct",
        asset_id="BTCUSDT",
    )
    assert result.status == "MATCHED"
    assert result.resolved is True
    assert result.quantity_within_tolerance is True
    assert result.entry_price_within_tolerance is True
    assert result.recovery_events == ()


def test_missing_report_is_not_treated_as_flat() -> None:
    result = reconcile_position(
        _local(),
        None,
        account_id="acct",
        asset_id="BTCUSDT",
    )
    assert result.status == "NO_AUTHORITATIVE_POSITION_REPORT"
    assert result.resolved is False
    assert result.authoritative_report_present is False
    assert "does not mean flat" in result.reasons[0]


def test_open_report_without_entry_average_fails_closed() -> None:
    result = reconcile_position(
        None,
        _venue(quantity=2.0, avg=None),
        account_id="acct",
        asset_id="BTCUSDT",
    )
    assert result.status == "UNRESOLVED"
    assert result.recovery_events == ()
    assert "missing avg_entry_price" in result.reasons[0]


def test_quantity_match_with_price_mismatch_remains_unresolved() -> None:
    result = reconcile_position(
        _local(quantity=1.0, avg=100.0),
        _venue(quantity=1.0, avg=102.0),
        account_id="acct",
        asset_id="BTCUSDT",
        entry_price_relative_tolerance=1e-4,
    )
    assert result.quantity_within_tolerance is True
    assert result.entry_price_within_tolerance is False
    assert result.status == "UNRESOLVED"


def test_flat_authoritative_report_proposes_close_recovery_then_requires_rerun() -> None:
    result = reconcile_position(
        _local(quantity=1.25, avg=100.0),
        _venue(quantity=0.0, avg=None),
        account_id="acct",
        asset_id="BTCUSDT",
    )
    assert result.status == "RECOVERY_REQUIRED"
    assert len(result.recovery_events) == 1
    event = result.recovery_events[0]
    assert event.quantity_delta == pytest.approx(-1.25)
    assert event.target_quantity == pytest.approx(0.0)
    assert event.synthetic_price == pytest.approx(100.0)
    assert event.establishes_realized_pnl is False


def test_direction_reversal_is_two_step_close_then_open() -> None:
    result = reconcile_position(
        _local(quantity=1.0, avg=100.0),
        _venue(quantity=-0.5, avg=90.0),
        account_id="acct",
        asset_id="BTCUSDT",
    )
    assert result.status == "RECOVERY_REQUIRED"
    assert len(result.recovery_events) == 2
    close, open_ = result.recovery_events
    assert close.quantity_delta == pytest.approx(-1.0)
    assert close.target_quantity == pytest.approx(0.0)
    assert close.synthetic_price == pytest.approx(100.0)
    assert open_.quantity_delta == pytest.approx(-0.5)
    assert open_.target_quantity == pytest.approx(-0.5)
    assert open_.synthetic_price == pytest.approx(90.0)


def test_reduction_preserves_local_entry_average_for_synthetic_recovery() -> None:
    result = reconcile_position(
        _local(quantity=2.0, avg=105.0),
        _venue(quantity=1.0, avg=105.0),
        account_id="acct",
        asset_id="BTCUSDT",
    )
    assert result.status == "RECOVERY_REQUIRED"
    assert result.recovery_events[0].quantity_delta == pytest.approx(-1.0)
    assert result.recovery_events[0].synthetic_price == pytest.approx(105.0)


def test_batch_blocks_unknown_outcomes_and_duplicate_fill_ids() -> None:
    report = reconcile_execution_state(
        {"BTCUSDT": _local()},
        {"BTCUSDT": _venue()},
        account_id="acct",
        tracked_assets=("BTCUSDT",),
        unresolved_command_ids=("cmd-unknown",),
        fill_ids=("fill-a", "fill-a"),
        reports_complete=True,
    )
    assert report.ready is False
    checks = dict(report.checks)
    assert checks["positions_reconciled"] is True
    assert checks["no_unknown_command_outcomes"] is False
    assert checks["no_duplicate_fill_ids"] is False


def test_incomplete_bounded_history_can_be_acceptable_with_explicit_matching_position() -> None:
    report = reconcile_execution_state(
        {"BTCUSDT": _local()},
        {"BTCUSDT": _venue()},
        account_id="acct",
        tracked_assets=("BTCUSDT",),
        reports_complete=False,
        policy=ReconciliationBatchPolicy(
            require_authoritative_report_for_tracked_assets=True,
            allow_incomplete_history_with_explicit_positions=True,
        ),
    )
    assert report.ready is True
    assert dict(report.checks)["history_contract_acceptable"] is True


def test_missing_explicit_position_report_blocks_even_if_local_state_is_flat() -> None:
    report = reconcile_execution_state(
        {},
        {},
        account_id="acct",
        tracked_assets=("BTCUSDT",),
        reports_complete=True,
    )
    assert report.ready is False
    assert dict(report.checks)["authoritative_reports_present"] is False
