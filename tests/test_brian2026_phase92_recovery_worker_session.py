from __future__ import annotations

from types import SimpleNamespace

import pytest

import brian2026.phase92_recovery_worker_session as phase92
from brian2026.phase92_recovery_worker_session import (
    RecoveryWorkerSession,
    RecoveryWorkerSessionError,
    run_recovery_startup_once_from_env,
)


class FakeRpc:
    def __init__(self):
        self.closed = 0

    def __call__(self, name, params):
        raise AssertionError(f"unexpected RPC {name}: {params}")

    def close(self):
        self.closed += 1


class FakeSupervisor:
    def __init__(self, store, *, release_result=True, release_error=None):
        self.store = store
        self.runtime_id = "runtime-92"
        self.release_result = release_result
        self.release_error = release_error
        self.release_calls = 0
        self.valid = True

    def release(self):
        self.release_calls += 1
        if self.release_error is not None:
            raise self.release_error
        self.valid = False
        return self.release_result


class FakeGate:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class FakeStack:
    def __init__(self, supervisor, gate=None):
        self.runtime_supervisor = supervisor
        self.startup_gate = gate or FakeGate(result="gate-result")


def _session(*, owns_rpc=True, gate=None, release_error=None):
    rpc = FakeRpc()
    store = SimpleNamespace()
    supervisor = FakeSupervisor(store, release_error=release_error)
    stack = FakeStack(supervisor, gate)
    return RecoveryWorkerSession(
        rpc=rpc,
        runtime_store=store,
        runtime_supervisor=supervisor,
        stack=stack,
        owns_rpc=owns_rpc,
    ), rpc, supervisor, stack


def test_session_delegates_startup_gate_and_close_releases_lease_and_transport() -> None:
    session, rpc, supervisor, stack = _session()
    result = session.run_startup_gate(
        max_items=3,
        recovery_worker_token="worker-a",
        recovery_claim_seconds=30,
        recovery_markets={},
        recovery_risk_limits_by_asset={},
        recovery_ttl_seconds=60,
        marks={},
        observed_at=1.0,
        source_ref="phase92-test",
    )
    assert result == "gate-result"
    assert stack.startup_gate.calls[0]["max_items"] == 3
    assert session.close() is True
    assert supervisor.release_calls == 1
    assert rpc.closed == 1
    assert session.closed is True

    # Repeated close never performs another release or HTTP close.
    assert session.close() is True
    assert supervisor.release_calls == 1
    assert rpc.closed == 1


def test_closed_session_cannot_run_gate() -> None:
    session, _, _, _ = _session()
    session.close()
    with pytest.raises(RecoveryWorkerSessionError, match="closed"):
        session.run_startup_gate(
            max_items=1,
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_markets={},
            recovery_risk_limits_by_asset={},
            recovery_ttl_seconds=60,
            marks={},
            observed_at=1.0,
            source_ref="phase92-test",
        )


def test_context_manager_closes_session_when_gate_raises() -> None:
    gate = FakeGate(error=RuntimeError("gate exploded"))
    session, rpc, supervisor, _ = _session(gate=gate)

    with pytest.raises(RuntimeError, match="gate exploded"):
        with session:
            session.run_startup_gate(
                max_items=1,
                recovery_worker_token="worker-a",
                recovery_claim_seconds=30,
                recovery_markets={},
                recovery_risk_limits_by_asset={},
                recovery_ttl_seconds=60,
                marks={},
                observed_at=1.0,
                source_ref="phase92-test",
            )

    assert supervisor.release_calls == 1
    assert rpc.closed == 1
    assert session.closed is True


def test_release_error_still_closes_owned_transport() -> None:
    session, rpc, supervisor, _ = _session(
        release_error=RuntimeError("release exploded")
    )
    with pytest.raises(RuntimeError, match="release exploded"):
        session.close()
    assert supervisor.release_calls == 1
    assert rpc.closed == 1
    assert session.closed is True


def test_open_releases_supervisor_and_owned_rpc_when_stack_assembly_fails(monkeypatch) -> None:
    rpc = FakeRpc()
    captured = {}
    store_holder = {}

    class FakeStore:
        def __init__(self, incoming_rpc):
            assert incoming_rpc is rpc
            store_holder["store"] = self

    def acquire(**kwargs):
        captured.update(kwargs)
        return FakeSupervisor(kwargs["store"])

    monkeypatch.setattr(phase92, "DurableRuntimeStore", FakeStore)
    monkeypatch.setattr(
        phase92,
        "PersistedDurableRuntimeSupervisor",
        SimpleNamespace(acquire=acquire),
    )
    monkeypatch.setattr(
        phase92,
        "build_recovery_runtime_stack",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("assembly exploded")),
    )

    with pytest.raises(RuntimeError, match="assembly exploded"):
        RecoveryWorkerSession.open(
            rpc=rpc,
            runtime_id="runtime-92",
            owner_token="owner-a",
            lease_seconds=60,
            owns_rpc=True,
        )

    supervisor = captured
    # The fake supervisor created inside acquire is not directly exposed; the
    # release behavior is asserted by capturing it on construction below.


def test_open_cleanup_calls_release_on_assembled_supervisor_failure(monkeypatch) -> None:
    rpc = FakeRpc()
    holder = {}

    class FakeStore:
        def __init__(self, incoming_rpc):
            assert incoming_rpc is rpc

    def acquire(**kwargs):
        supervisor = FakeSupervisor(kwargs["store"])
        holder["supervisor"] = supervisor
        return supervisor

    monkeypatch.setattr(phase92, "DurableRuntimeStore", FakeStore)
    monkeypatch.setattr(
        phase92,
        "PersistedDurableRuntimeSupervisor",
        SimpleNamespace(acquire=acquire),
    )
    monkeypatch.setattr(
        phase92,
        "build_recovery_runtime_stack",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("assembly exploded")),
    )

    with pytest.raises(RuntimeError, match="assembly exploded"):
        RecoveryWorkerSession.open(
            rpc=rpc,
            runtime_id="runtime-92",
            owner_token="owner-a",
            lease_seconds=60,
            owns_rpc=True,
        )

    assert holder["supervisor"].release_calls == 1
    assert rpc.closed == 1


@pytest.mark.parametrize("lease_seconds", [0, 9, 301, "bad"])
def test_open_rejects_unsafe_lease_bounds_before_store_construction(lease_seconds) -> None:
    rpc = FakeRpc()
    with pytest.raises(RecoveryWorkerSessionError):
        RecoveryWorkerSession.open(
            rpc=rpc,
            runtime_id="runtime-92",
            owner_token="owner-a",
            lease_seconds=lease_seconds,
        )
    assert rpc.closed == 0


def test_from_env_requires_runtime_id_before_constructing_supabase_transport(monkeypatch) -> None:
    called = False

    def forbidden_from_env(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("transport must not be created")

    monkeypatch.setattr(
        phase92.SupabaseRecoveryRpcTransport,
        "from_env",
        forbidden_from_env,
    )
    with pytest.raises(RecoveryWorkerSessionError, match="BRIAN_RUNTIME_ID"):
        RecoveryWorkerSession.from_env(
            env={
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_SECRET_KEY": "sb_secret_test",
            }
        )
    assert called is False


def test_from_env_generates_process_unique_owner_when_not_configured(monkeypatch) -> None:
    rpc = FakeRpc()
    captured = {}

    monkeypatch.setattr(
        phase92.SupabaseRecoveryRpcTransport,
        "from_env",
        lambda **kwargs: rpc,
    )

    def fake_open(cls, **kwargs):
        captured.update(kwargs)
        return "session"

    monkeypatch.setattr(
        RecoveryWorkerSession,
        "open",
        classmethod(fake_open),
    )

    result = RecoveryWorkerSession.from_env(
        env={
            "BRIAN_RUNTIME_ID": "runtime-92",
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_secret_test",
        }
    )
    assert result == "session"
    assert captured["runtime_id"] == "runtime-92"
    assert captured["lease_seconds"] == 60
    assert captured["owner_token"].startswith("phase92-")
    assert captured["owns_rpc"] is True


def test_one_shot_helper_always_closes_on_gate_failure(monkeypatch) -> None:
    session, rpc, supervisor, _ = _session(
        gate=FakeGate(error=RuntimeError("worker failed"))
    )
    monkeypatch.setattr(
        RecoveryWorkerSession,
        "from_env",
        classmethod(lambda cls, **kwargs: session),
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        run_recovery_startup_once_from_env(
            max_items=1,
            recovery_worker_token="worker-a",
            recovery_claim_seconds=30,
            recovery_markets={},
            recovery_risk_limits_by_asset={},
            recovery_ttl_seconds=60,
            marks={},
            observed_at=1.0,
            source_ref="phase92-one-shot",
            env={},
        )

    assert supervisor.release_calls == 1
    assert rpc.closed == 1
