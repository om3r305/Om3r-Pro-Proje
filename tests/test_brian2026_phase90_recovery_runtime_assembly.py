from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from brian2026.phase90_recovery_runtime_assembly import (
    RecoveryRuntimeStack,
    build_recovery_runtime_stack,
)


class _RuntimeSupervisor:
    def __init__(self, *, valid=True):
        self.runtime_id = "runtime-90"
        self._valid = valid

    @property
    def valid(self):
        return self._valid


class _Aborter:
    def abort_authorized_cycle(self, *, cycle_id, reason):
        raise AssertionError("assembly test must not execute aborter")


def _rpc(name, params):
    raise AssertionError(f"assembly test must not call RPC {name}: {params}")


def test_factory_wires_one_rpc_transport_and_one_runtime_authority() -> None:
    runtime = _RuntimeSupervisor()
    stack = build_recovery_runtime_stack(
        rpc=_rpc,
        runtime_supervisor=runtime,
    )
    assert isinstance(stack, RecoveryRuntimeStack)
    assert stack.runtime_supervisor is runtime

    stores = (
        stack.directives,
        stack.claims,
        stack.starts,
        stack.checkpoints,
        stack.audits,
        stack.admission,
        stack.backlog,
    )
    assert all(store._rpc is _rpc for store in stores)

    assert stack.execution._runtime_supervisor() is runtime
    assert stack.execution._claims() is stack.claims
    assert stack.restart.runtime_supervisor is runtime
    assert stack.restart.work is stack.backlog
    assert stack.restart.directives is stack.directives
    assert stack.restart.claims is stack.claims
    assert stack.restart.starts is stack.starts
    assert stack.restart.recovery_execution is stack.execution
    assert stack.restart.audits is stack.audits
    assert stack.startup_gate.recovery is stack.restart
    assert stack.startup_gate.admission is stack.admission
    assert stack.shadow_only is True
    assert stack.live_execution is False


def test_optional_foreign_cycle_aborter_is_propagated_only_to_restart_orchestrator() -> None:
    aborter = _Aborter()
    stack = build_recovery_runtime_stack(
        rpc=_rpc,
        runtime_supervisor=_RuntimeSupervisor(),
        foreign_cycle_aborter=aborter,
    )
    assert stack.restart.foreign_cycle_aborter is aborter


def test_noncallable_rpc_is_rejected_before_partial_stack_is_built() -> None:
    with pytest.raises(TypeError, match="rpc transport"):
        build_recovery_runtime_stack(
            rpc=None,
            runtime_supervisor=_RuntimeSupervisor(),
        )


def test_stale_runtime_is_rejected_at_assembly_boundary() -> None:
    with pytest.raises(PersistedRuntimeStaleError, match="stale"):
        build_recovery_runtime_stack(
            rpc=_rpc,
            runtime_supervisor=_RuntimeSupervisor(valid=False),
        )


def test_runtime_must_expose_nonempty_runtime_id() -> None:
    runtime = _RuntimeSupervisor()
    runtime.runtime_id = ""
    with pytest.raises(ValueError, match="runtime_id"):
        build_recovery_runtime_stack(
            rpc=_rpc,
            runtime_supervisor=runtime,
        )


def test_stack_dataclass_rejects_cross_wired_execution_authority() -> None:
    stack = build_recovery_runtime_stack(
        rpc=_rpc,
        runtime_supervisor=_RuntimeSupervisor(),
    )
    wrong_runtime = _RuntimeSupervisor()
    wrong_execution = SimpleNamespace(
        _runtime_supervisor=lambda: wrong_runtime,
        _claims=lambda: stack.claims,
    )
    with pytest.raises(ValueError, match="Phase84 runtime supervisor"):
        RecoveryRuntimeStack(
            runtime_supervisor=stack.runtime_supervisor,
            directives=stack.directives,
            claims=stack.claims,
            starts=stack.starts,
            checkpoints=stack.checkpoints,
            execution=wrong_execution,
            audits=stack.audits,
            admission=stack.admission,
            backlog=stack.backlog,
            restart=stack.restart,
            startup_gate=stack.startup_gate,
        )
