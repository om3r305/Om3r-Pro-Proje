from __future__ import annotations

from dataclasses import replace

import pytest

from brian2026.phase46_execution_simulator import SimulatedExecutionReceipt
from brian2026.phase50_execution_reconciliation import LocalPositionState
from brian2026.phase56_pretrade_risk_engine import PreTradeRiskReceipt
from brian2026.phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from brian2026.phase60_shadow_state_ledger import ShadowAccountState, ShadowStateLedger
from brian2026.phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from brian2026.phase63_crash_recovery import (
    PaperVenueCheckpoint,
    RuntimeRecoveryError,
    create_paper_venue_checkpoint,
    create_runtime_checkpoint,
    restore_paper_venue,
    restore_runtime_checkpoint,
    restore_shadow_state_ledger,
)


TS = 1_760_000_000.0


def _cycle(cycle_id: str, *, price: float = 100.0) -> ShadowExecutionCycle:
    risk = PreTradeRiskReceipt(
        action="ALLOW",
        trading_state="ACTIVE",
        asset_id="BTCUSDT",
        requested_notional_usd=100.0,
        reduce_only=False,
        reasons=(),
        projected_position_weight=0.1,
        checks=(("fixture", True),),
    )
    execution = SimulatedExecutionReceipt(
        status="FILLED",
        side="BUY",
        order_type="MARKET",
        submit_timestamp=TS,
        venue_timestamp=TS + 0.1,
        snapshot_timestamp=TS + 0.1,
        requested_base=1.0,
        filled_base=1.0,
        fill_fraction=1.0,
        average_fill_price=price,
        best_reference_price=price,
        adverse_slippage_bps=0.0,
        levels_consumed=1,
        slipped_one_tick=False,
        reason="phase63 fixture",
    )
    item = ShadowExecutionCycleItem(
        instruction_kind="OPEN",
        asset_id="BTCUSDT",
        risk_receipt=risk,
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=100.0,
        status="fixture",
    )
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=(item,),
        initial_available_cash_usd=1000.0,
        reserved_new_risk_cash_usd=100.0,
        remaining_unreserved_cash_usd=900.0,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def _genesis(account_id: str = "paper-acct") -> ShadowAccountState:
    return ShadowAccountState(
        account_id=account_id,
        observed_at=TS,
        equity_usd=1000.0,
        available_cash_usd=1000.0,
        position_weights=(),
        covered_assets=("BTCUSDT", "ETHUSDT"),
        source_kind="GENESIS",
        source_ref="phase63-genesis",
    )


def _local_from_venue(venue: PaperVenue) -> dict[str, LocalPositionState]:
    position = venue.position("BTCUSDT")
    return {
        "BTCUSDT": LocalPositionState(
            account_id=venue.config.account_id,
            asset_id="BTCUSDT",
            quantity=position.quantity,
            avg_entry_price=position.avg_entry_price,
            source_fill_ids=position.source_fill_ids,
        )
    }


def test_pending_cycle_survives_crash_between_paper_fill_and_reconciliation() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
            fee_bps=0.0,
        )
    )
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("pending-crash")

    ledger.append_cycle(cycle)
    venue.apply_cycle(cycle)
    assert ledger.pending_cycle_id == cycle.cycle_id
    assert venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert ledger.head_state.position_weights == ()

    checkpoint = create_runtime_checkpoint(ledger, venue)
    restored_ledger, restored_venue = restore_runtime_checkpoint(checkpoint)

    assert restored_ledger.pending_cycle_id == cycle.cycle_id
    assert restored_ledger.head_state.state_id == ledger.head_state.state_id
    assert restored_venue.cash_usd == pytest.approx(900.0)
    assert restored_venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert restored_venue.state_version == 1
    assert restored_ledger.verify_integrity() is True

    reconciliation = restored_venue.reconcile_against_local(
        _local_from_venue(restored_venue),
        tracked_assets=("BTCUSDT", "ETHUSDT"),
    )
    assert reconciliation.ready is True
    state = restored_venue.build_reconciled_state(
        reconciliation,
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 10,
        source_ref="recovered-paper-report",
    )
    restored_ledger.commit_reconciled_state(
        cycle.cycle_id,
        reconciliation,
        state,
    )
    assert restored_ledger.pending_cycle_id is None
    assert restored_ledger.head_state.state_id == state.state_id


def test_committed_runtime_roundtrip_preserves_manifest_and_paper_idempotency() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
            fee_bps=0.0,
        )
    )
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("committed")
    ledger.append_cycle(cycle)
    original_receipt = venue.apply_cycle(cycle)
    reconciliation = venue.reconcile_against_local(
        _local_from_venue(venue),
        tracked_assets=("BTCUSDT", "ETHUSDT"),
    )
    state = venue.build_reconciled_state(
        reconciliation,
        marks={"BTCUSDT": 105.0},
        observed_at=TS + 10,
        source_ref="commit-before-checkpoint",
    )
    ledger.commit_reconciled_state(cycle.cycle_id, reconciliation, state)

    checkpoint = create_runtime_checkpoint(ledger, venue)
    restored_ledger, restored_venue = restore_runtime_checkpoint(checkpoint)

    assert restored_ledger.manifest() == ledger.manifest()
    assert create_paper_venue_checkpoint(restored_venue).to_dict() == checkpoint.paper.to_dict()
    duplicate_receipt = restored_venue.apply_cycle(cycle)
    assert duplicate_receipt == original_receipt
    assert restored_venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert restored_venue.cash_usd == pytest.approx(900.0)
    assert restored_venue.state_version == 1


def test_paper_checkpoint_replays_multiple_cycles_and_fill_order() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
            fee_bps=0.0,
        )
    )
    venue.apply_cycle(_cycle("c1", price=100.0))
    venue.apply_cycle(_cycle("c2", price=120.0))
    checkpoint = create_paper_venue_checkpoint(venue)
    restored = restore_paper_venue(checkpoint)

    assert [fill.fill_id for fill in restored.fills] == [
        fill.fill_id for fill in venue.fills
    ]
    assert restored.cash_usd == pytest.approx(780.0)
    assert restored.position("BTCUSDT").quantity == pytest.approx(2.0)
    assert restored.position("BTCUSDT").avg_entry_price == pytest.approx(110.0)
    assert restored.state_version == 2


def test_forged_fill_content_is_detected_even_when_checkpoint_hash_is_recomputed() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
            fee_bps=0.0,
        )
    )
    venue.apply_cycle(_cycle("forged-fill"))
    checkpoint = create_paper_venue_checkpoint(venue)
    original_fill = checkpoint.fill_sequence[0]
    forged_fill = replace(original_fill, price=101.0)
    forged_checkpoint = PaperVenueCheckpoint(
        config=checkpoint.config,
        cash_usd=checkpoint.cash_usd,
        state_version=checkpoint.state_version,
        fill_sequence=(forged_fill,),
        cycle_receipts=checkpoint.cycle_receipts,
        final_positions=checkpoint.final_positions,
    )
    with pytest.raises(RuntimeRecoveryError, match="fill content hash mismatch"):
        restore_paper_venue(forged_checkpoint)


def test_forged_cash_checkpoint_fails_replay_validation() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
            fee_bps=0.0,
        )
    )
    venue.apply_cycle(_cycle("forged-cash"))
    checkpoint = create_paper_venue_checkpoint(venue)
    forged = PaperVenueCheckpoint(
        config=checkpoint.config,
        cash_usd=checkpoint.cash_usd + 5.0,
        state_version=checkpoint.state_version,
        fill_sequence=checkpoint.fill_sequence,
        cycle_receipts=checkpoint.cycle_receipts,
        final_positions=checkpoint.final_positions,
    )
    with pytest.raises(RuntimeRecoveryError, match="restored paper cash mismatch"):
        restore_paper_venue(forged)


def test_orphan_fill_in_checkpoint_is_rejected() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
            fee_bps=0.0,
        )
    )
    venue.apply_cycle(_cycle("orphan-base"))
    checkpoint = create_paper_venue_checkpoint(venue)
    extra = replace(
        checkpoint.fill_sequence[0],
        fill_id="f" * 64,
        cycle_id="orphan-cycle",
    )
    forged = PaperVenueCheckpoint(
        config=checkpoint.config,
        cash_usd=checkpoint.cash_usd,
        state_version=checkpoint.state_version,
        fill_sequence=checkpoint.fill_sequence + (extra,),
        cycle_receipts=checkpoint.cycle_receipts,
        final_positions=checkpoint.final_positions,
    )
    with pytest.raises(RuntimeRecoveryError, match="orphan or reordered fills"):
        restore_paper_venue(forged)


def test_shadow_ledger_manifest_roundtrip_and_tamper_detection() -> None:
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("manifest")
    ledger.append_cycle(cycle)
    manifest = ledger.manifest()
    restored = restore_shadow_state_ledger(manifest)
    assert restored.manifest() == manifest
    assert restored.pending_cycle_id == cycle.cycle_id

    tampered = dict(manifest)
    tampered["ledger_hash"] = "0" * 64
    with pytest.raises(RuntimeRecoveryError, match="manifest hash mismatch"):
        restore_shadow_state_ledger(tampered)


def test_checkpoint_dict_roundtrip_accepts_jsonb_integral_numeric_rendering() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="BRIAN-PAPER-RUNTIME",
            starting_cash_usd=5000.0,
            fee_bps=10.0,
        )
    )
    checkpoint = create_paper_venue_checkpoint(venue)
    payload = checkpoint.to_dict()

    # PostgreSQL jsonb may serialize semantically integral numerics without a
    # decimal point on the read path (5000.0 -> 5000, 10.0 -> 10).
    payload["cash_usd"] = 5000
    payload["config"]["starting_cash_usd"] = 5000
    payload["config"]["fee_bps"] = 10

    restored = PaperVenueCheckpoint.from_dict(payload)

    assert restored.checkpoint_id == checkpoint.checkpoint_id
    assert restored.config.starting_cash_usd == 5000.0
    assert restored.config.fee_bps == 10.0


def test_runtime_checkpoint_rejects_account_identity_split_brain() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
        )
    )
    ledger = ShadowStateLedger(_genesis(account_id="different-account"))
    with pytest.raises(RuntimeRecoveryError, match="account ids differ"):
        create_runtime_checkpoint(ledger, venue)


def test_checkpoint_dict_roundtrip_validates_content_hash() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
            fee_bps=0.0,
        )
    )
    venue.apply_cycle(_cycle("dict-roundtrip"))
    checkpoint = create_paper_venue_checkpoint(venue)
    restored_checkpoint = PaperVenueCheckpoint.from_dict(checkpoint.to_dict())
    assert restored_checkpoint == checkpoint

    payload = checkpoint.to_dict()
    payload["cash_usd"] = float(payload["cash_usd"]) + 1.0
    with pytest.raises(RuntimeRecoveryError, match="content hash mismatch"):
        PaperVenueCheckpoint.from_dict(payload)
