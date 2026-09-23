from __future__ import annotations

from types import SimpleNamespace

import pytest

import brian2026.phase95_auto_binance_recovery_worker as phase95
from brian2026.phase46_execution_simulator import LiquidityLevel, OrderBookSnapshot
from brian2026.phase56_pretrade_risk_engine import InstrumentRiskLimits
from brian2026.phase57_shadow_execution_cycle import ExecutionMarketInput
from brian2026.phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from brian2026.phase81_cancel_recovery_directive import (
    CancelRecoveryDirectiveReceipt,
    CancelRecoveryLeg,
)
from brian2026.phase86_recovery_admission_interlock import RecoveryAdmissionState
from brian2026.phase87_recovery_restart_resume import RecoveryRestartWorkItem
from brian2026.phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from brian2026.phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryAssetEvidence,
    BinanceSpotRecoveryEvidenceBundle,
)
from brian2026.phase95_auto_binance_recovery_worker import (
    AutoRecoveryEvidenceError,
    run_one_auto_binance_recovery,
    run_one_auto_binance_recovery_from_env,
)


RUNTIME = "runtime-95"
ORIGINAL = "o" * 64
DISPATCH = "d" * 64
CANCEL = "r" * 64
CHECKPOINT = "k" * 64
HEAD = "h" * 64


def _work(state="NEEDS_CLAIM", **overrides):
    values = dict(
        runtime_id=RUNTIME,
        has_work=True,
        status="WORK",
        work_state=state,
        original_cycle_id=ORIGINAL,
        dispatch_id=DISPATCH,
        cancel_risk_version=7,
        cancel_risk_receipt_id=CANCEL,
        cancel_reason="REDUCING_NEW_RISK",
        requested_at="2026-09-23T12:00:00Z",
        runtime_version=10,
        runtime_checkpoint_id=CHECKPOINT,
        runtime_head_state_id=HEAD,
        directive_exists=True,
        recovery_status="READY_REDUCE_ONLY",
        directive_runtime_version=10,
        directive_state_id=HEAD,
        directive_prepared_at="2026-09-23T12:00:01Z",
        claim_status=None,
        claim_worker_token=None,
        claim_fencing_token=None,
        claim_until=None,
        recovery_cycle_id=None,
        progress_runtime_version=None,
        progress_head_state_id=None,
        progress_checkpoint_id=None,
        started=False,
        started_at=None,
        recovery_journal_stage=None,
        phase84_terminal_event=False,
    )
    if state == "NEEDS_DIRECTIVE":
        values.update(
            directive_exists=False,
            recovery_status=None,
            directive_runtime_version=None,
            directive_state_id=None,
            directive_prepared_at=None,
        )
    elif state == "NEEDS_AUDIT":
        values.update(
            claim_status="CLAIMED",
            claim_worker_token="worker-a",
            claim_fencing_token=3,
            claim_until="2026-09-23T13:00:00Z",
            started=True,
            started_at="2026-09-23T12:01:00Z",
            recovery_cycle_id="y" * 64,
            progress_runtime_version=10,
            progress_head_state_id=HEAD,
            progress_checkpoint_id="p" * 64,
            recovery_journal_stage="COMMITTED",
            phase84_terminal_event=True,
        )
    values.update(overrides)
    return RecoveryRestartWorkItem(**values)


def _idle():
    return RecoveryRestartWorkItem(
        runtime_id=RUNTIME,
        has_work=False,
        status="IDLE",
        work_state="IDLE",
    )


def _leg(asset="BTCUSDT"):
    return CancelRecoveryLeg(
        asset_id=asset,
        before_weight=0.10,
        current_weight=0.25,
        target_weight=0.10,
        reduce_weight=0.15,
        current_direction=1,
        order_direction=-1,
        reduce_only=True,
    )


def _directive(*, status="DUPLICATE", recovery_status="READY_REDUCE_ONLY", assets=("BTCUSDT",)):
    prepared = status in {"PREPARED", "DUPLICATE"}
    return CancelRecoveryDirectiveReceipt(
        runtime_id=RUNTIME,
        cycle_id=ORIGINAL,
        dispatch_id=DISPATCH,
        runtime_version=10,
        fencing_token=71,
        status=status,
        prepared=prepared,
        duplicate=status == "DUPLICATE",
        cancel_risk_version=7 if prepared else None,
        cancel_risk_receipt_id=CANCEL if prepared else None,
        cancel_reason="REDUCING_NEW_RISK" if prepared else None,
        pre_state_id="a" * 64 if prepared else None,
        current_state_id=HEAD if prepared else None,
        current_risk_version=8 if prepared else None,
        current_risk_receipt_id="s" * 64 if prepared else None,
        current_risk_state=(
            "HALTED" if recovery_status == "WAIT_RISK_RELEASE" else "REDUCING"
        ) if prepared else None,
        recovery_status=recovery_status if prepared else None,
        recovery_legs=(
            tuple(_leg(asset) for asset in assets)
            if prepared and recovery_status not in {"NO_RECOVERY_REQUIRED", "MANUAL_REVIEW"}
            else ()
        ),
        unsafe_assets=(
            ({"asset_id": "BTCUSDT", "reason": "ROLLBACK_NOT_REDUCE_ONLY"},)
            if prepared and recovery_status == "MANUAL_REVIEW"
            else ()
        ),
        journal_stage=(
            "PAPER_APPLIED"
            if status == "WAIT_ORIGINAL_COMMIT"
            else None
        ),
    )


def _bundle(assets=("BTCUSDT",), *, observed=101.0):
    rows = []
    for index, asset in enumerate(sorted(assets)):
        bid = 99.9 + index
        ask = 100.1 + index
        mark = (bid + ask) / 2.0
        snapshot = OrderBookSnapshot(
            timestamp=observed + index,
            bids=(LiquidityLevel(bid, 2.0),),
            asks=(LiquidityLevel(ask, 2.5),),
        )
        rows.append(BinanceSpotRecoveryAssetEvidence(
            asset_id=asset,
            market=ExecutionMarketInput(
                reference_price=mark,
                tick_size=0.1,
                snapshots=(snapshot,),
            ),
            risk_limits=InstrumentRiskLimits(min_notional=10.0),
            mark=mark,
            observed_at=observed + index,
            depth_last_update_id=str(100 + index),
            source_host="https://data-api.binance.vision",
            exchange_status="TRADING",
        ))
    return BinanceSpotRecoveryEvidenceBundle(tuple(rows))


class _Backlog:
    def __init__(self, *items):
        self.items = list(items)
        self.calls = 0

    def read_next(self, *, runtime_id):
        assert runtime_id == RUNTIME
        self.calls += 1
        if len(self.items) > 1:
            return self.items.pop(0)
        return self.items[0]


class _Directives:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def prepare(self, lease, *, cycle_id, expected_runtime_version):
        assert cycle_id == ORIGINAL
        assert expected_runtime_version == 10
        self.calls += 1
        return self.receipt


class _Provider:
    def __init__(self, bundle=None, error=None):
        self.bundle = bundle or _bundle()
        self.error = error
        self.calls = []
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited += 1

    def collect(self, assets):
        self.calls.append(tuple(assets))
        if self.error is not None:
            raise self.error
        return self.bundle


class _Supervisor:
    def __init__(self):
        self.persisted_version = 10
        self.lease = SimpleNamespace(
            acquired=True,
            runtime_id=RUNTIME,
            owner_token="owner-a",
            fencing_token=71,
        )
        self._valid = True
        self.runtime = SimpleNamespace(
            checkpoint=lambda: SimpleNamespace(checkpoint_id=CHECKPOINT),
            ledger=SimpleNamespace(
                head_state=SimpleNamespace(state_id=HEAD)
            ),
        )

    @property
    def valid(self):
        return self._valid


class _Session:
    def __init__(self, backlog, directives, gate_result=None):
        self.runtime_id = RUNTIME
        self.runtime_supervisor = _Supervisor()
        self.stack = SimpleNamespace(
            backlog=backlog,
            directives=directives,
        )
        self.closed = False
        self.gate_calls = []
        self.gate_result = gate_result or RecoveryStartupGateReceipt(
            runtime_id=RUNTIME,
            steps=(),
            admission=RecoveryAdmissionState(
                runtime_id=RUNTIME,
                status="OPEN",
                blocked=False,
            ),
            status="READY_FOR_NORMAL_WORK",
            ready_for_normal_work=True,
            processed_items=0,
            max_items=1,
        )

    def run_startup_gate(self, **kwargs):
        self.gate_calls.append(dict(kwargs))
        return self.gate_result


def test_idle_never_constructs_binance_provider_and_runs_empty_one_item_gate() -> None:
    provider_calls = 0

    def provider_factory():
        nonlocal provider_calls
        provider_calls += 1
        return _Provider()

    session = _Session(_Backlog(_idle()), _Directives(_directive()))
    receipt = run_one_auto_binance_recovery(
        session,
        recovery_worker_token="worker-a",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase95:test",
        provider_factory=provider_factory,
        clock=lambda: 100.0,
    )
    assert receipt.preflight_work_state == "IDLE"
    assert receipt.evidence_assets == ()
    assert provider_calls == 0
    assert session.gate_calls[0]["max_items"] == 1
    assert session.gate_calls[0]["recovery_markets"] == {}
    assert session.gate_calls[0]["marks"] == {}


def test_needs_audit_does_not_fetch_market_evidence() -> None:
    provider_calls = 0

    def provider_factory():
        nonlocal provider_calls
        provider_calls += 1
        return _Provider()

    session = _Session(
        _Backlog(_work("NEEDS_AUDIT")),
        _Directives(_directive()),
    )
    receipt = run_one_auto_binance_recovery(
        session,
        recovery_worker_token="worker-a",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase95:audit",
        provider_factory=provider_factory,
        clock=lambda: 100.0,
    )
    assert receipt.preflight_work_state == "NEEDS_AUDIT"
    assert provider_calls == 0
    assert session.gate_calls[0]["recovery_markets"] == {}


def test_ready_directive_fetches_exact_assets_after_decision_and_feeds_gate() -> None:
    ready = _directive(assets=("ETHUSDT", "BTCUSDT"))
    provider = _Provider(_bundle(("BTCUSDT", "ETHUSDT"), observed=101.0))
    work = _work()
    session = _Session(
        _Backlog(work, work),
        _Directives(ready),
    )
    receipt = run_one_auto_binance_recovery(
        session,
        recovery_worker_token="worker-a",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase95:ready",
        provider_factory=lambda: provider,
        clock=lambda: 100.0,
    )
    assert provider.calls == [("BTCUSDT", "ETHUSDT")]
    assert provider.entered == provider.exited == 1
    assert receipt.evidence_assets == ("BTCUSDT", "ETHUSDT")
    assert receipt.decision_at == 100.0
    gate = session.gate_calls[0]
    assert gate["observed_at"] == 100.0
    assert set(gate["recovery_markets"]) == {"BTCUSDT", "ETHUSDT"}
    assert set(gate["recovery_risk_limits_by_asset"]) == {"BTCUSDT", "ETHUSDT"}
    assert set(gate["marks"]) == {"BTCUSDT", "ETHUSDT"}
    assert gate["max_items"] == 1


def test_needs_directive_may_advance_to_needs_claim_without_identity_drift() -> None:
    before = _work("NEEDS_DIRECTIVE")
    after = _work("NEEDS_CLAIM")
    provider = _Provider(_bundle(observed=101.0))
    session = _Session(
        _Backlog(before, after),
        _Directives(_directive(status="PREPARED")),
    )
    receipt = run_one_auto_binance_recovery(
        session,
        recovery_worker_token="worker-a",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase95:prepare",
        provider_factory=lambda: provider,
        clock=lambda: 100.0,
    )
    assert receipt.directive_status == "PREPARED"
    assert provider.calls == [("BTCUSDT",)]
    assert len(session.gate_calls) == 1


def test_no_recovery_required_never_fetches_binance() -> None:
    calls = 0

    def provider_factory():
        nonlocal calls
        calls += 1
        return _Provider()

    session = _Session(
        _Backlog(_work()),
        _Directives(_directive(recovery_status="NO_RECOVERY_REQUIRED", assets=())),
    )
    receipt = run_one_auto_binance_recovery(
        session,
        recovery_worker_token="worker-a",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase95:no-recovery",
        provider_factory=provider_factory,
        clock=lambda: 100.0,
    )
    assert calls == 0
    assert receipt.evidence_assets == ()
    assert session.gate_calls[0]["recovery_markets"] == {}


def test_market_evidence_before_decision_is_rejected_before_gate() -> None:
    work = _work()
    session = _Session(
        _Backlog(work, work),
        _Directives(_directive()),
    )
    provider = _Provider(_bundle(observed=99.0))
    with pytest.raises(AutoRecoveryEvidenceError, match="predates"):
        run_one_auto_binance_recovery(
            session,
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase95:causal",
            provider_factory=lambda: provider,
            clock=lambda: 100.0,
        )
    assert session.gate_calls == []


def test_evidence_asset_mismatch_is_rejected_before_gate() -> None:
    work = _work()
    session = _Session(
        _Backlog(work, work),
        _Directives(_directive()),
    )
    provider = _Provider(_bundle(("ETHUSDT",), observed=101.0))
    with pytest.raises(AutoRecoveryEvidenceError, match="asset set"):
        run_one_auto_binance_recovery(
            session,
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase95:mismatch",
            provider_factory=lambda: provider,
            clock=lambda: 100.0,
        )
    assert session.gate_calls == []


def test_backlog_identity_change_during_external_fetch_invalidates_runtime() -> None:
    before = _work()
    after = _work(
        original_cycle_id="x" * 64,
        dispatch_id="z" * 64,
        cancel_risk_receipt_id="q" * 64,
    )
    session = _Session(
        _Backlog(before, after),
        _Directives(_directive()),
    )
    with pytest.raises(PersistedRuntimeStaleError, match="identity changed"):
        run_one_auto_binance_recovery(
            session,
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase95:race",
            provider_factory=lambda: _Provider(_bundle(observed=101.0)),
            clock=lambda: 100.0,
        )
    assert session.runtime_supervisor.valid is False
    assert session.gate_calls == []


def test_runtime_head_change_during_external_fetch_invalidates_runtime() -> None:
    before = _work()
    after = _work(runtime_version=11)
    session = _Session(
        _Backlog(before, after),
        _Directives(_directive()),
    )
    with pytest.raises(PersistedRuntimeStaleError, match="runtime/head changed"):
        run_one_auto_binance_recovery(
            session,
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase95:head-race",
            provider_factory=lambda: _Provider(_bundle(observed=101.0)),
            clock=lambda: 100.0,
        )
    assert session.runtime_supervisor.valid is False


def test_provider_failure_closes_provider_and_never_enters_gate() -> None:
    work = _work()
    provider = _Provider(error=RuntimeError("market down"))
    session = _Session(
        _Backlog(work),
        _Directives(_directive()),
    )
    with pytest.raises(RuntimeError, match="market down"):
        run_one_auto_binance_recovery(
            session,
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase95:provider-fail",
            provider_factory=lambda: provider,
            clock=lambda: 100.0,
        )
    assert provider.entered == provider.exited == 1
    assert session.gate_calls == []


def test_nonfinite_clock_is_rejected_before_provider() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _Provider()

    session = _Session(
        _Backlog(_work()),
        _Directives(_directive()),
    )
    with pytest.raises(AutoRecoveryEvidenceError, match="non-finite"):
        run_one_auto_binance_recovery(
            session,
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase95:clock",
            provider_factory=factory,
            clock=lambda: float("nan"),
        )
    assert calls == 0


def test_env_helper_always_closes_phase92_session_on_provider_failure(monkeypatch) -> None:
    work = _work()
    session = _Session(
        _Backlog(work),
        _Directives(_directive()),
    )
    entered = 0
    exited = 0

    class ContextSession(_Session):
        def __enter__(self):
            nonlocal entered
            entered += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            nonlocal exited
            exited += 1
            self.closed = True

    context = ContextSession(session.stack.backlog, session.stack.directives)

    monkeypatch.setattr(
        phase95.RecoveryWorkerSession,
        "from_env",
        classmethod(lambda cls, **kwargs: context),
    )
    provider = _Provider(error=RuntimeError("binance down"))
    with pytest.raises(RuntimeError, match="binance down"):
        run_one_auto_binance_recovery_from_env(
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase95:env",
            env={"BRIAN_RUNTIME_ID": RUNTIME},
            provider_factory=lambda: provider,
            clock=lambda: 100.0,
        )
    assert entered == exited == 1
    assert context.closed is True
