"""Real Postgres tests for Phase75 atomic risk authorization + write-ahead."""

from __future__ import annotations

import copy
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
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase75 Postgres tests require CI database",
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
    return f"pytest-phase75-{label}-{uuid.uuid4().hex[:10]}"


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
    head_char: str = "a",
    cycle_id: str | None = None,
    entry_sequence: int = 0,
    entry_stage: str = "CYCLE_CREATED",
):
    cycles = {}
    entries = []
    if cycle_id is not None:
        cycles[cycle_id] = _cycle(cycle_id)
        entries.append({
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "sequence": entry_sequence,
            "stage": entry_stage,
            "cycle_id": cycle_id,
            "cycle_hash": _h("c"),
            "previous_entry_id": None,
            "artifact_hash": _h("d"),
            "artifact_ref": f"cycle:{cycle_id}",
            "entry_id": _h("e"),
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
            "journal_hash": _h("j" if cycle_id is None else "k"),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h(checkpoint_char),
    }


def _risk_entry(sequence: int, *, entry_char: str, receipt_char: str, prev: str | None, ts: float):
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "sequence": sequence,
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


def _risk_manifest(char: str = "1", *, extended: bool = False):
    entries = [_risk_entry(0, entry_char="a", receipt_char="b", prev=None, ts=TS)]
    if extended:
        entries.append(
            _risk_entry(1, entry_char="f", receipt_char="0", prev="a", ts=TS + 1)
        )
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


def _acquire(conn, runtime_id: str, owner: str = "owner-a"):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_acquire_shadow_runtime_lease(%s,%s,%s)",
            (runtime_id, owner, 30),
        )
        return cur.fetchone()[0]


def _commit_runtime(conn, runtime_id, owner, fence, version, checkpoint):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_shadow_runtime_checkpoint(%s,%s,%s,%s,%s)",
            (runtime_id, owner, fence, version, Json(checkpoint)),
        )
        return cur.fetchone()[0]


def _commit_risk(conn, runtime_id, owner, fence, version, manifest):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_operational_risk_ledger(%s,%s,%s,%s,%s)",
            (runtime_id, owner, fence, version, Json(manifest)),
        )
        return cur.fetchone()[0]


def _authorize(
    conn,
    runtime_id,
    owner,
    fence,
    runtime_version,
    risk_version,
    risk_hash,
    receipt_id,
    cycle_id,
    governed_id,
    policy_id,
    checkpoint,
):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_authorize_and_persist_governed_cycle("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                runtime_id, owner, fence, runtime_version,
                risk_version, risk_hash, receipt_id,
                cycle_id, governed_id, policy_id, Json(checkpoint),
            ),
        )
        return cur.fetchone()[0]


def _read_runtime(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_read_shadow_runtime_checkpoint(%s)",
            (runtime_id,),
        )
        return cur.fetchone()[0]


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(f"select count(*) from public.{table} where runtime_id=%s", (runtime_id,))
        return int(cur.fetchone()[0])


def _bootstrap(conn, runtime_id):
    lease = _acquire(conn, runtime_id)
    fence = lease["fencing_token"]
    runtime = _commit_runtime(
        conn, runtime_id, "owner-a", fence, 0, _runtime_checkpoint("1")
    )
    manifest = _risk_manifest("2")
    risk = _commit_risk(conn, runtime_id, "owner-a", fence, 0, manifest)
    return lease, runtime, risk, manifest


def test_atomic_authorization_advances_runtime_and_persists_binding_together():
    runtime_id = _rid("success")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cycle_id = _h("3")
        checkpoint = _runtime_checkpoint("4", cycle_id=cycle_id)
        row = _authorize(
            conn, runtime_id, "owner-a", lease["fencing_token"],
            runtime["version"], risk["version"], manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            cycle_id, _h("5"), _h("6"), checkpoint,
        )

        assert row["authorized"] is True
        assert row["status"] == "AUTHORIZED_AND_PERSISTED"
        assert row["runtime_version_before"] == 1
        assert row["runtime_version_after"] == 2
        assert row["checkpoint_id"] == checkpoint["checkpoint_id"]

        head = _read_runtime(conn, runtime_id)
        assert head["version"] == 2
        assert head["checkpoint_id"] == checkpoint["checkpoint_id"]
        assert _count(conn, "brian_governed_cycle_authorizations", runtime_id) == 1
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 2
    finally:
        conn.close()


def test_exact_lost_response_retry_is_duplicate_current_without_new_checkpoint():
    runtime_id = _rid("duplicate")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cycle_id = _h("7")
        checkpoint = _runtime_checkpoint("8", cycle_id=cycle_id)
        args = (
            runtime_id, "owner-a", lease["fencing_token"],
            runtime["version"], risk["version"], manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            cycle_id, _h("9"), _h("0"), checkpoint,
        )
        first = _authorize(conn, *args)
        duplicate = _authorize(conn, *args)

        assert first["status"] == "AUTHORIZED_AND_PERSISTED"
        assert duplicate["status"] == "DUPLICATE_CURRENT"
        assert duplicate["duplicate"] is True
        assert duplicate["runtime_version_after"] == 2
        assert duplicate["current_runtime_version"] == 2
        assert _count(conn, "brian_governed_cycle_authorizations", runtime_id) == 1
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 2
    finally:
        conn.close()


def test_concurrent_exact_authorization_has_one_commit_and_one_duplicate():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(setup, runtime_id)
        params = (
            runtime_id, "owner-a", lease["fencing_token"],
            runtime["version"], risk["version"], manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            _h("a"), _h("b"), _h("c"),
            _runtime_checkpoint("d", cycle_id=_h("a")),
        )
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(_):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _authorize(conn, *params)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "AUTHORIZED_AND_PERSISTED" for row in rows) == 1
    assert sum(row["status"] == "DUPLICATE_CURRENT" for row in rows) == 1


def test_risk_advance_before_atomic_call_prevents_writeahead_commit():
    runtime_id = _rid("risk-stale")
    conn = _connect()
    try:
        lease, runtime, risk, old = _bootstrap(conn, runtime_id)
        newer = _risk_manifest("3", extended=True)
        advanced = _commit_risk(
            conn, runtime_id, "owner-a", lease["fencing_token"], risk["version"], newer
        )
        assert advanced["version"] == 2

        cycle_id = _h("e")
        checkpoint = _runtime_checkpoint("f", cycle_id=cycle_id)
        rejected = _authorize(
            conn, runtime_id, "owner-a", lease["fencing_token"],
            runtime["version"], risk["version"], old["ledger_hash"],
            old["entries"][-1]["receipt"]["receipt_id"],
            cycle_id, _h("1"), _h("2"), checkpoint,
        )
        assert rejected["status"] == "RISK_VERSION_CONFLICT"
        assert rejected["authorized"] is False
        assert _read_runtime(conn, runtime_id)["version"] == 1
        assert _count(conn, "brian_governed_cycle_authorizations", runtime_id) == 0
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
    finally:
        conn.close()


def test_runtime_advance_before_atomic_call_prevents_authorization():
    runtime_id = _rid("runtime-stale")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        newer = _commit_runtime(
            conn, runtime_id, "owner-a", lease["fencing_token"],
            runtime["version"], _runtime_checkpoint("3", head_char="3"),
        )
        assert newer["version"] == 2

        cycle_id = _h("4")
        rejected = _authorize(
            conn, runtime_id, "owner-a", lease["fencing_token"],
            runtime["version"], risk["version"], manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            cycle_id, _h("5"), _h("6"),
            _runtime_checkpoint("7", cycle_id=cycle_id),
        )
        assert rejected["status"] == "RUNTIME_VERSION_CONFLICT"
        assert rejected["authorized"] is False
        assert _count(conn, "brian_governed_cycle_authorizations", runtime_id) == 0
    finally:
        conn.close()


def test_inner_phase70_failure_rolls_back_authorization_atomically():
    runtime_id = _rid("rollback")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cycle_id = _h("8")
        malformed = _runtime_checkpoint(
            "9",
            cycle_id=cycle_id,
            entry_sequence=1,  # Phase70 requires global sequence to begin at zero.
        )
        with pytest.raises(psycopg2.Error, match="PHASE70_JOURNAL_SEQUENCE"):
            _authorize(
                conn, runtime_id, "owner-a", lease["fencing_token"],
                runtime["version"], risk["version"], manifest["ledger_hash"],
                manifest["entries"][-1]["receipt"]["receipt_id"],
                cycle_id, _h("a"), _h("b"), malformed,
            )

        assert _read_runtime(conn, runtime_id)["version"] == 1
        assert _count(conn, "brian_governed_cycle_authorizations", runtime_id) == 0
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
    finally:
        conn.close()


def test_same_cycle_cannot_be_reauthorized_with_different_governed_evidence():
    runtime_id = _rid("conflict")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cycle_id = _h("c")
        checkpoint = _runtime_checkpoint("d", cycle_id=cycle_id)
        assert _authorize(
            conn, runtime_id, "owner-a", lease["fencing_token"],
            runtime["version"], risk["version"], manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            cycle_id, _h("e"), _h("f"), checkpoint,
        )["status"] == "AUTHORIZED_AND_PERSISTED"

        with pytest.raises(psycopg2.Error, match="PHASE75_AUTHORIZATION_CONFLICT"):
            _authorize(
                conn, runtime_id, "owner-a", lease["fencing_token"],
                runtime["version"], risk["version"], manifest["ledger_hash"],
                manifest["entries"][-1]["receipt"]["receipt_id"],
                cycle_id, _h("0"), _h("f"), checkpoint,
            )
    finally:
        conn.close()


def test_service_role_cannot_mutate_authorization_history():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cycle_id = _h("1")
        assert _authorize(
            conn, runtime_id, "owner-a", lease["fencing_token"],
            runtime["version"], risk["version"], manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            cycle_id, _h("2"), _h("3"),
            _runtime_checkpoint("4", cycle_id=cycle_id),
        )["authorized"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_governed_cycle_authorizations "
                    "set policy_fingerprint=%s where runtime_id=%s and cycle_id=%s",
                    (_h("9"), runtime_id, cycle_id),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
