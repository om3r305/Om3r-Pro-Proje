"""Real Postgres tests for Phase76 durable shadow execution outbox."""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip(
    "psycopg2",
    reason="psycopg2 is only installed in the dedicated Postgres CI job",
)
from psycopg2.extras import Json

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = (
    ROOT / "supabase" / "migrations" / "202609230720_brian_phase70_durable_runtime_store.sql",
    ROOT / "supabase" / "migrations" / "202609230745_brian_phase73_operational_risk_store.sql",
    ROOT / "supabase" / "migrations" / "202609230815_brian_phase74_governed_cycle_binding.sql",
    ROOT / "supabase" / "migrations" / "202609230845_brian_phase75_atomic_governed_writeahead.sql",
    ROOT / "supabase" / "migrations" / "202609230915_brian_phase76_shadow_execution_outbox.sql",
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase76 Postgres tests require CI database",
)
TS = 1_760_000_000.0

_BOOTSTRAP = """
do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin; end if;
end
$$;
"""


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


@pytest.fixture(scope="module", autouse=True)
def _apply_migrations():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP)
            for migration in MIGRATIONS:
                cur.execute(migration.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _rid(label: str) -> str:
    return f"pytest-phase76-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _cycle(cycle_id: str):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": f"plan-{cycle_id[:8]}",
        "items": [],
        "initial_available_cash_usd": 1000.0,
        "reserved_new_risk_cash_usd": 0.0,
        "remaining_unreserved_cash_usd": 1000.0,
        "denied_assets": [],
        "pending_reversal_assets": [],
        "cycle_id": cycle_id,
        "account_state_mutated": False,
        "shadow_only": True,
        "live_execution": False,
    }


def _runtime_checkpoint(
    checkpoint_char: str,
    *,
    cycle_id: str | None = None,
    head_char: str = "a",
    terminal_stage: str | None = None,
):
    cycles = {}
    entries = []
    if cycle_id is not None:
        cycles[cycle_id] = _cycle(cycle_id)
        entries = [{
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "sequence": 0,
            "stage": "CYCLE_CREATED",
            "cycle_id": cycle_id,
            "cycle_hash": _h("c"),
            "previous_entry_id": None,
            "artifact_hash": _h("d"),
            "artifact_ref": f"cycle:{cycle_id}",
            "entry_id": _h("e"),
        }]
        if terminal_stage is not None:
            entries.append({
                "schema_version": "brian.phase66-durable-cycle-journal.v1",
                "sequence": 1,
                "stage": terminal_stage,
                "cycle_id": cycle_id,
                "cycle_hash": _h("c"),
                "previous_entry_id": _h("e"),
                "artifact_hash": _h("f"),
                "artifact_ref": f"terminal:{terminal_stage}",
                "entry_id": _h("0"),
            })
    return {
        "schema_version": "brian.phase67-durable-runtime-orchestrator.v1",
        "runtime_checkpoint": {
            "schema_version": "brian.phase63-crash-recovery.v1",
            "paper": {
                "schema_version": "brian.phase63-crash-recovery.v1",
                "config": {
                    "account_id": "paper-acct",
                    "starting_cash_usd": 1000.0,
                    "fee_bps": 0.0,
                },
                "cash_usd": 1000.0,
                "state_version": 0,
                "fill_sequence": [],
                "cycle_receipts": [],
                "final_positions": [],
                "paper_only": True,
                "live_execution": False,
                "checkpoint_id": _h("p"),
            },
            "shadow_ledger_manifest": {
                "schema_version": "brian.phase60-shadow-state-ledger.v1",
                "append_only": True,
                "head_state_id": _h(head_char),
                "pending_cycle_id": None,
                "states": {
                    _h(head_char): {
                        "state_id": _h(head_char),
                        "account_id": "paper-acct",
                    }
                },
                "transitions": [],
                "ledger_hash": _h("l"),
                "shadow_only": True,
                "live_execution": False,
            },
            "pending_cycle_id": None,
            "live_execution": False,
            "checkpoint_id": _h("r"),
        },
        "journal_manifest": {
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "append_only": True,
            "cycles": cycles,
            "entries": entries,
            "journal_hash": _h(
                "j" if cycle_id is None else ("m" if terminal_stage is not None else "k")
            ),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h(checkpoint_char),
    }


def _risk_entry(seq, entry_char, receipt_char, prev, ts):
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "sequence": seq,
        "previous_entry_id": None if prev is None else _h(prev),
        "policy_hash": _h("q"),
        "receipt": {
            "schema_version": "brian.phase68-operational-risk-governor.v1",
            "timestamp": ts,
            "previous_state": "ACTIVE",
            "trading_state": "ACTIVE",
            "recommended_state": "ACTIVE",
            "reasons": [],
            "max_drawdown_fraction": 0.0,
            "window_loss_fraction": 0.0,
            "qualifying_stoplosses": 0,
            "stoploss_lock_until": None,
            "blocked_assets": [],
            "consecutive_execution_failures": 0,
            "reconciliation_failures": 0,
            "unknown_order_outcomes": 0,
            "market_data_age_seconds": 0.0,
            "manual_halt": False,
            "manual_release_requested": False,
            "halt_latched": False,
            "receipt_id": _h(receipt_char),
            "execution_failure_lock_until": None,
            "reconciliation_failure_lock_until": None,
            "asset_cooldown_until": [],
            "shadow_only": True,
            "live_execution": False,
        },
        "entry_id": _h(entry_char),
    }


def _risk_manifest(char="1", *, extended=False):
    entries = [_risk_entry(0, "a", "b", None, TS)]
    if extended:
        entries.append(_risk_entry(1, "f", "0", "a", TS + 1))
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "append_only": True,
        "policy": {"max_drawdown_fraction": 0.5, "max_daily_loss_fraction": 0.5},
        "policy_hash": _h("q"),
        "initial_state": "ACTIVE",
        "entries": entries,
        "entry_count": len(entries),
        "head_entry_id": entries[-1]["entry_id"],
        "current_state": "ACTIVE",
        "halt_latched": False,
        "ledger_hash": _h(char),
        "shadow_only": True,
        "live_execution": False,
    }


def _acquire(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_acquire_shadow_runtime_lease(%s,%s,%s)",
            (runtime_id, "owner-a", 30),
        )
        return cur.fetchone()[0]


def _commit_runtime(conn, runtime_id, fence, version, checkpoint):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_shadow_runtime_checkpoint(%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", fence, version, Json(checkpoint)),
        )
        return cur.fetchone()[0]


def _commit_risk(conn, runtime_id, fence, version, manifest):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_operational_risk_ledger(%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", fence, version, Json(manifest)),
        )
        return cur.fetchone()[0]


def _authorize(conn, runtime_id, fence, runtime_version, risk_version, manifest, cycle_id):
    checkpoint = _runtime_checkpoint("4", cycle_id=cycle_id)
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_authorize_and_persist_governed_cycle("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                fence,
                runtime_version,
                risk_version,
                manifest["ledger_hash"],
                manifest["entries"][-1]["receipt"]["receipt_id"],
                cycle_id,
                _h("g"),
                _h("p"),
                Json(checkpoint),
            ),
        )
        return cur.fetchone()[0]


def _submit(conn, runtime_id, fence, cycle_id, dispatch_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_submit_shadow_execution_dispatch(%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", fence, cycle_id, dispatch_id),
        )
        return cur.fetchone()[0]


def _bootstrap_authorized(conn, runtime_id, cycle_id):
    lease = _acquire(conn, runtime_id)
    fence = lease["fencing_token"]
    runtime = _commit_runtime(
        conn, runtime_id, fence, 0, _runtime_checkpoint("1")
    )
    manifest = _risk_manifest("2")
    risk = _commit_risk(conn, runtime_id, fence, 0, manifest)
    auth = _authorize(
        conn, runtime_id, fence, runtime["version"], risk["version"], manifest, cycle_id
    )
    assert auth["status"] == "AUTHORIZED_AND_PERSISTED"
    return lease, runtime, risk, manifest, auth


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(f"select count(*) from public.{table} where runtime_id=%s", (runtime_id,))
        return int(cur.fetchone()[0])


def test_submit_current_authorization_creates_immutable_dispatch():
    runtime_id = _rid("submit")
    cycle_id = _h("1")
    conn = _connect()
    try:
        lease, _, _, manifest, auth = _bootstrap_authorized(conn, runtime_id, cycle_id)
        dispatch_id = _h("2")
        row = _submit(conn, runtime_id, lease["fencing_token"], cycle_id, dispatch_id)

        assert row["submitted"] is True
        assert row["status"] == "SUBMITTED"
        assert row["dispatch_id"] == dispatch_id
        assert row["authorization_checkpoint_id"] == auth["checkpoint_id"]
        assert row["risk_ledger_hash"] == manifest["ledger_hash"]
        assert _count(conn, "brian_shadow_execution_dispatches", runtime_id) == 1
    finally:
        conn.close()


def test_exact_dispatch_retry_is_duplicate_current():
    runtime_id = _rid("duplicate")
    cycle_id = _h("3")
    conn = _connect()
    try:
        lease, *_ = _bootstrap_authorized(conn, runtime_id, cycle_id)
        dispatch_id = _h("4")
        first = _submit(conn, runtime_id, lease["fencing_token"], cycle_id, dispatch_id)
        duplicate = _submit(conn, runtime_id, lease["fencing_token"], cycle_id, dispatch_id)
        assert first["status"] == "SUBMITTED"
        assert duplicate["status"] == "DUPLICATE_CURRENT"
        assert duplicate["duplicate"] is True
        assert _count(conn, "brian_shadow_execution_dispatches", runtime_id) == 1
    finally:
        conn.close()


def test_concurrent_exact_dispatch_has_one_submit_and_one_duplicate():
    runtime_id = _rid("race")
    cycle_id = _h("5")
    setup = _connect()
    try:
        lease, *_ = _bootstrap_authorized(setup, runtime_id, cycle_id)
        fence = lease["fencing_token"]
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(_):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _submit(conn, runtime_id, fence, cycle_id, _h("6"))
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "SUBMITTED" for row in rows) == 1
    assert sum(row["status"] == "DUPLICATE_CURRENT" for row in rows) == 1


def test_missing_authorization_never_creates_dispatch():
    runtime_id = _rid("missing")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id)
        fence = lease["fencing_token"]
        _commit_runtime(conn, runtime_id, fence, 0, _runtime_checkpoint("7"))
        row = _submit(conn, runtime_id, fence, _h("8"), _h("9"))
        assert row["submitted"] is False
        assert row["status"] == "AUTHORIZATION_MISSING"
        assert _count(conn, "brian_shadow_execution_dispatches", runtime_id) == 0
    finally:
        conn.close()


def test_risk_change_after_authorization_blocks_dispatch():
    runtime_id = _rid("risk-stale")
    cycle_id = _h("a")
    conn = _connect()
    try:
        lease, _, risk, _, _ = _bootstrap_authorized(conn, runtime_id, cycle_id)
        newer = _risk_manifest("3", extended=True)
        advanced = _commit_risk(
            conn, runtime_id, lease["fencing_token"], risk["version"], newer
        )
        assert advanced["version"] == 2

        row = _submit(conn, runtime_id, lease["fencing_token"], cycle_id, _h("b"))
        assert row["submitted"] is False
        assert row["status"] == "RISK_VERSION_CONFLICT"
        assert row["risk_version"] == 2
        assert _count(conn, "brian_shadow_execution_dispatches", runtime_id) == 0
    finally:
        conn.close()


def test_runtime_change_after_authorization_blocks_dispatch():
    runtime_id = _rid("runtime-stale")
    cycle_id = _h("c")
    conn = _connect()
    try:
        lease, _, _, _, auth = _bootstrap_authorized(conn, runtime_id, cycle_id)
        newer = _commit_runtime(
            conn,
            runtime_id,
            lease["fencing_token"],
            auth["runtime_version_after"],
            _runtime_checkpoint(
                "d",
                cycle_id=cycle_id,
                terminal_stage="ABORTED",
            ),
        )
        assert newer["version"] == auth["runtime_version_after"] + 1

        row = _submit(conn, runtime_id, lease["fencing_token"], cycle_id, _h("e"))
        assert row["submitted"] is False
        assert row["status"] == "RUNTIME_VERSION_CONFLICT"
        assert _count(conn, "brian_shadow_execution_dispatches", runtime_id) == 0
    finally:
        conn.close()


def test_same_cycle_cannot_be_submitted_under_different_dispatch_id():
    runtime_id = _rid("conflict")
    cycle_id = _h("f")
    conn = _connect()
    try:
        lease, *_ = _bootstrap_authorized(conn, runtime_id, cycle_id)
        assert _submit(
            conn, runtime_id, lease["fencing_token"], cycle_id, _h("0")
        )["status"] == "SUBMITTED"
        with pytest.raises(psycopg2.Error, match="PHASE76_DISPATCH_CONFLICT"):
            _submit(conn, runtime_id, lease["fencing_token"], cycle_id, _h("1"))
    finally:
        conn.close()


def test_service_role_cannot_mutate_dispatch_history():
    runtime_id = _rid("priv")
    cycle_id = _h("2")
    conn = _connect()
    try:
        lease, *_ = _bootstrap_authorized(conn, runtime_id, cycle_id)
        dispatch_id = _h("3")
        assert _submit(
            conn, runtime_id, lease["fencing_token"], cycle_id, dispatch_id
        )["status"] == "SUBMITTED"

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_shadow_execution_dispatches "
                    "set policy_fingerprint=%s where runtime_id=%s and dispatch_id=%s",
                    (_h("9"), runtime_id, dispatch_id),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
