from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase46_execution_simulator import LiquidityLevel, OrderBookSnapshot
from brian2026.phase56_pretrade_risk_engine import InstrumentRiskLimits
from brian2026.phase57_shadow_execution_cycle import ExecutionMarketInput
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase81_cancel_recovery_directive import CancelRecoveryLeg
from brian2026.phase82_recovery_claim_fencing import (
    RecoveryClaimError,
    RecoveryClaimRenewal,
)
from brian2026.phase83_atomic_recovery_start import AtomicRecoveryStartReceipt
from brian2026.phase84_recovery_execution_checkpoint import (
    PersistedRecoveryExecutionSupervisor,
    RecoveryCheckpointReceipt,
    RecoveryExecutionError,
    build_recovery_cycle,
    compile_recovery_plan,
)


TS = 1_760_000_000.0


def _leg() -> CancelRecoveryLeg:
    return CancelRecoveryLeg(
        asset_id="BTCUSDT",
        before_weight=0.10,
        current_weight=0.25,
        target_weight=0.10,
        reduce_weight=0.15,
        current_direction=1,
        order_direction=-1,
        reduce_only=True,
    )


def _start() -> AtomicRecoveryStartReceipt:
    return AtomicRecoveryStartReceipt(
        runtime_id="runtime-84",
        cycle_id="o" * 64,
        dispatch_id="d" * 64,
        cancel_risk_receipt_id="r" * 64,
        runtime_version=5,
        head_state_id="h" * 64,
        fencing_token=1,
        recovery_claim_fencing_token=3,
        status="STARTED",
        started=True,
        duplicate=False,
        resume_only=False,
        risk_version=9,
        risk_receipt_id="s" * 64,
        risk_state="REDUCING",
        recovery_legs=(_leg(),),
    )


def _market(*, bid_qty=10.0) -> ExecutionMarketInput:
    return ExecutionMarketInput(
        reference_price=100.0,
        tick_size=0.01,
        snapshots=(
            OrderBookSnapshot(
                timestamp=TS + 0.01,
                bids=(
                    LiquidityLevel(100.0, bid_qty),
                    LiquidityLevel(99.9, bid_qty),
                ),
                asks=(LiquidityLevel(100.1, 10.0),),
            ),
        ),
    )


def test_recovery_plan_contains_only_phase55_reduce_only_intents() -> None:
    plan = compile_recovery_plan(
        _start(),
        created_at=TS,
        ttl_seconds=60,
    )
    assert len(plan.instructions) == 1
    instruction = plan.instructions[0]
    assert instruction.kind == "REDUCE"
    assert instruction.trade_intent is None
    assert instruction.pending_reversal is None
    assert instruction.reduction_intent is not None
    assert instruction.reduction_intent.reduce_only is True
    assert instruction.reduction_intent.current_weight == pytest.approx(0.25)
    assert instruction.reduction_intent.resulting_weight == pytest.approx(0.10)
    assert plan.shadow_only is True
    assert plan.live_execution is False


def test_recovery_cycle_is_independently_risk_allowed_and_fully_filled() -> None:
    cycle = build_recovery_cycle(
        _start(),
        equity_usd=1000.0,
        available_cash_usd=100.0,
        current_weights={"BTCUSDT": 0.25},
        markets={"BTCUSDT": _market()},
        risk_limits_by_asset={
            "BTCUSDT": InstrumentRiskLimits(max_notional_per_order=500.0)
        },
        trading_state="REDUCING",
        created_at=TS,
        ttl_seconds=60,
    )
    assert len(cycle.items) == 1
    item = cycle.items[0]
    assert item.risk_receipt.allowed is True
    assert item.risk_receipt.reduce_only is True
    assert item.execution_receipt is not None
    assert item.execution_receipt.status == "FILLED"
    assert item.execution_receipt.side == "SELL"
    assert cycle.reserved_new_risk_cash_usd == pytest.approx(0.0)
    assert cycle.denied_assets == ()
    assert cycle.pending_reversal_assets == ()


def test_recovery_cycle_refuses_partial_fill_before_any_paper_side_effect() -> None:
    with pytest.raises(RecoveryExecutionError, match="not fully filled"):
        build_recovery_cycle(
            _start(),
            equity_usd=1000.0,
            available_cash_usd=100.0,
            current_weights={"BTCUSDT": 0.25},
            markets={"BTCUSDT": _market(bid_qty=0.2)},
            risk_limits_by_asset={},
            trading_state="REDUCING",
            created_at=TS,
            ttl_seconds=60,
        )


def test_recovery_cycle_refuses_halted_state_even_for_reduce_only_leg() -> None:
    with pytest.raises(RecoveryExecutionError, match="ACTIVE/REDUCING"):
        build_recovery_cycle(
            _start(),
            equity_usd=1000.0,
            available_cash_usd=100.0,
            current_weights={"BTCUSDT": 0.25},
            markets={"BTCUSDT": _market()},
            risk_limits_by_asset={},
            trading_state="HALTED",
            created_at=TS,
            ttl_seconds=60,
        )


def test_terminal_checkpoint_contract_requires_committed_recovery_stage() -> None:
    receipt = RecoveryCheckpointReceipt(
        runtime_id="runtime-84",
        original_cycle_id="o" * 64,
        recovery_cycle_id="c" * 64,
        dispatch_id="d" * 64,
        checkpoint_id="k" * 64,
        journal_stage="COMMITTED",
        version=7,
        current_version=7,
        fencing_token=1,
        recovery_claim_fencing_token=3,
        status="RECOVERY_COMMITTED_PENDING_AUDIT",
        committed=True,
        duplicate=False,
        terminal=True,
        head_state_id="z" * 64,
    )
    assert receipt.terminal is True

    with pytest.raises(ValueError, match="terminal recovery checkpoint"):
        RecoveryCheckpointReceipt(
            runtime_id="runtime-84",
            original_cycle_id="o" * 64,
            recovery_cycle_id="c" * 64,
            dispatch_id="d" * 64,
            checkpoint_id="k" * 64,
            journal_stage="PAPER_APPLIED",
            version=6,
            current_version=6,
            fencing_token=1,
            recovery_claim_fencing_token=3,
            status="COMMITTED",
            committed=True,
            duplicate=False,
            terminal=True,
        )


class _Supervisor:
    def __init__(self):
        self._valid = True


def _renewal(status: str, *, renewed: bool, state: str | None = "REDUCING"):
    return RecoveryClaimRenewal(
        runtime_id="runtime-84",
        cycle_id="o" * 64,
        dispatch_id="d" * 64,
        runtime_version=5,
        head_state_id="h" * 64,
        fencing_token=1,
        claim_fencing_token=3,
        status=status,
        renewed=renewed,
        risk_version=10 if state is not None else None,
        risk_receipt_id=("t" * 64) if state is not None else None,
        risk_state=state,
    )


def test_phase84_renewal_gate_distinguishes_halt_claim_loss_and_stale_risk() -> None:
    supervisor = _Supervisor()

    PersistedRecoveryExecutionSupervisor._handle_renewal(
        supervisor,
        _renewal("RENEWED", renewed=True),
    )
    assert supervisor._valid is True

    PersistedRecoveryExecutionSupervisor._handle_renewal(
        supervisor,
        _renewal("RENEWAL_BLOCKED_RISK", renewed=False, state="HALTED"),
    )
    assert supervisor._valid is True

    with pytest.raises(RecoveryClaimError):
        PersistedRecoveryExecutionSupervisor._handle_renewal(
            supervisor,
            _renewal("RENEWAL_LOST", renewed=False, state=None),
        )
    assert supervisor._valid is True

    with pytest.raises(PersistedRuntimeStaleError):
        PersistedRecoveryExecutionSupervisor._handle_renewal(
            supervisor,
            _renewal("RISK_STATE_UNAVAILABLE", renewed=False, state=None),
        )
    assert supervisor._valid is False


def test_phase84_lease_loss_invalidates_runtime() -> None:
    supervisor = _Supervisor()
    with pytest.raises(PersistedRuntimeLeaseError):
        PersistedRecoveryExecutionSupervisor._handle_renewal(
            supervisor,
            _renewal("LEASE_LOST", renewed=False, state=None),
        )
    assert supervisor._valid is False
