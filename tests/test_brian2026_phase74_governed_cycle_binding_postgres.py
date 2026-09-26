"""Real Postgres integration tests for Phase74 governed-cycle binding.

Only stdlib + psycopg2 are imported so the dedicated database CI job can prove
the SQL authorization/fencing boundary without importing Brian runtime modules.
"""

from __future__ import annotations

import os
import threading
import time
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
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase74 Postgres tests require CI database",
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


def _runtime_id(label: str) -> str:
    return f"pytest-phase74-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _runtime_checkpoint(checkpoint_char: str, *, head_char: str = "a"):
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
                "states": {_h(head_char): {"state_id": _h(head_char), "account_id": "paper-acct"}},
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
            "cycles": {},
            "entries": [],
            "journal_hash": _h("j"),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h(checkpoint_char),
    }


def _risk_entry(
    sequence: int,
    *,
    entry_char: str,
    receipt_char: str,
    previous_entry_char: str | None,
    timestamp: float,
):
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "sequence": sequence,
        "previous_entry_id": None if previous_entry_char is None else _h(previous_entry_char),
        "policy_hash": _h("e"),
        "receipt": {
            "schema_version": "brian.phase68-operational-risk-governor.v1",
            "timestamp": timestamp,
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


def _risk_manifest(ledger_char: str = "1", *, extended: bool = False):
    entries = [
        _risk_entry(
            0,
            entry_char="a",
            receipt_char="b",
            previous_entry_char=None,
            timestamp=TS,
        )
    ]
    if extended:
        entries.append(
            _risk_entry(
                1,
                entry_char="c",
                receipt_char="d",
                previous_entry_char="a",
                timestamp=TS + 1,
            )
        )
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "append_only": True,
        "policy": {"max_drawdown_fraction": 0.5, "max_daily_loss_fraction": 0.5},
        "policy_hash": _h("e"),
        "initial_state": "ACTIVE",
        "entries": entries,
        "entry_count": len(entries),
        "head_entry_id": entries[-1]["entry_id"],
        "current_state": "ACTIVE",
        "halt_latched": False,
        "ledger_hash": _h(ledger_char),
        "shadow_only": True,
        "live_execution": False,
    }


def _acquire(conn, runtime_id: str, owner: str, seconds: int = 30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_acquire_shadow_runtime_lease(%s,%s,%s)",
            (runtime_id, owner, seconds),
        )
        return cur.fetchone()[0]


def _commit_runtime(conn, runtime_id: str, owner: str, fence: int, version: int, checkpoint: dict):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_shadow_runtime_checkpoint(%s,%s,%s,%s,%s)",
            (runtime_id, owner, fence, version, Json(checkpoint)),
        )
        return cur.fetchone()[0]


def _commit_risk(conn, runtime_id: str, owner: str, fence: int, version: int, manifest: dict):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_operational_risk_ledger(%s,%s,%s,%s,%s)",
            (runtime_id, owner, fence, version, Json(manifest)),
        )
        return cur.fetchone()[0]


def _bind(
    conn,
    runtime_id: str,
    owner: str,
    fence: int,
    runtime_version: int,
    risk_version: int,
    risk_hash: str,
    receipt_id: str,
    cycle_id: str,
    governed_id: str,
    policy_id: str,
):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_bind_governed_shadow_cycle("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
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
            ),
        )
        return cur.fetchone()[0]


def _bootstrap_heads(conn, runtime_id: str, owner: str = "owner-a", seconds: int = 30):
    lease = _acquire(conn, runtime_id, owner, seconds)
    fence = lease["fencing_token"]
    runtime = _commit_runtime(
        conn,
        runtime_id,
        owner,
        fence,
        0,
        _runtime_checkpoint("1"),
    )
    risk_manifest = _risk_manifest("2")
    risk = _commit_risk(conn, runtime_id, owner, fence, 0, risk_manifest)
    return lease, runtime, risk, risk_manifest


def test_binding_succeeds_only_against_current_runtime_and_risk_heads():
    runtime_id = _runtime_id("bind")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap_heads(conn, runtime_id)
        row = _bind(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            _h("3"),
            _h("4"),
            _h("5"),
        )
        assert row["bound"] is True
        assert row["duplicate"] is False
        assert row["status"] == "BOUND"
        assert row["runtime_version"] == 1
        assert row["risk_version"] == 1
        assert row["risk_ledger_hash"] == manifest["ledger_hash"]
        assert row["risk_receipt_id"] == manifest["entries"][-1]["receipt"]["receipt_id"]
    finally:
        conn.close()


def test_exact_duplicate_binding_is_idempotent_and_echoes_same_evidence():
    runtime_id = _runtime_id("duplicate")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap_heads(conn, runtime_id)
        args = (
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            _h("6"),
            _h("7"),
            _h("8"),
        )
        first = _bind(conn, *args)
        duplicate = _bind(conn, *args)
        assert first["status"] == "BOUND"
        assert duplicate["status"] == "DUPLICATE"
        assert duplicate["bound"] is True
        assert duplicate["duplicate"] is True
        for key in (
            "risk_ledger_hash",
            "risk_receipt_id",
            "governed_result_id",
            "policy_fingerprint",
        ):
            assert duplicate[key] == first[key]
    finally:
        conn.close()


def test_concurrent_exact_bind_has_one_bound_and_one_duplicate():
    runtime_id = _runtime_id("race")
    setup = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap_heads(setup, runtime_id)
        fence = lease["fencing_token"]
        runtime_version = runtime["version"]
        risk_version = risk["version"]
        risk_hash = manifest["ledger_hash"]
        receipt_id = manifest["entries"][-1]["receipt"]["receipt_id"]
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(_):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _bind(
                conn,
                runtime_id,
                "owner-a",
                fence,
                runtime_version,
                risk_version,
                risk_hash,
                receipt_id,
                _h("9"),
                _h("0"),
                _h("1"),
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "BOUND" for row in results) == 1
    assert sum(row["status"] == "DUPLICATE" for row in results) == 1


def test_risk_head_advance_invalidates_old_governed_authorization():
    runtime_id = _runtime_id("risk-stale")
    conn = _connect()
    try:
        lease, runtime, risk, first = _bootstrap_heads(conn, runtime_id)
        second = _risk_manifest("3", extended=True)
        advanced = _commit_risk(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            risk["version"],
            second,
        )
        assert advanced["version"] == 2

        stale = _bind(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            first["ledger_hash"],
            first["entries"][-1]["receipt"]["receipt_id"],
            _h("2"),
            _h("3"),
            _h("4"),
        )
        assert stale["bound"] is False
        assert stale["status"] == "RISK_VERSION_CONFLICT"
        assert stale["risk_version"] == 2
    finally:
        conn.close()


def test_runtime_head_advance_invalidates_precomputed_cycle_binding():
    runtime_id = _runtime_id("runtime-stale")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap_heads(conn, runtime_id)
        newer = _commit_runtime(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            runtime["version"],
            _runtime_checkpoint("5", head_char="5"),
        )
        assert newer["version"] == 2

        stale = _bind(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            _h("6"),
            _h("7"),
            _h("8"),
        )
        assert stale["bound"] is False
        assert stale["status"] == "RUNTIME_VERSION_CONFLICT"
        assert stale["runtime_version"] == 2
    finally:
        conn.close()


def test_expired_takeover_fence_invalidates_old_worker_binding():
    runtime_id = _runtime_id("lease-stale")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap_heads(
            conn,
            runtime_id,
            seconds=1,
        )
        old_fence = lease["fencing_token"]
        time.sleep(1.15)
        takeover = _acquire(conn, runtime_id, "owner-b", 30)
        assert takeover["status"] == "EXPIRED_RECOVERY"
        assert takeover["fencing_token"] == old_fence + 1

        stale = _bind(
            conn,
            runtime_id,
            "owner-a",
            old_fence,
            runtime["version"],
            risk["version"],
            manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            _h("a"),
            _h("b"),
            _h("c"),
        )
        assert stale["bound"] is False
        assert stale["status"] == "LEASE_LOST"
    finally:
        conn.close()


def test_same_cycle_id_cannot_be_rebound_to_different_governed_evidence():
    runtime_id = _runtime_id("conflict")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap_heads(conn, runtime_id)
        cycle_id = _h("d")
        first = _bind(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            cycle_id,
            _h("e"),
            _h("f"),
        )
        assert first["status"] == "BOUND"

        with pytest.raises(psycopg2.Error, match="PHASE74_BIND_CONFLICT"):
            _bind(
                conn,
                runtime_id,
                "owner-a",
                lease["fencing_token"],
                runtime["version"],
                risk["version"],
                manifest["ledger_hash"],
                manifest["entries"][-1]["receipt"]["receipt_id"],
                cycle_id,
                _h("0"),
                _h("f"),
            )
    finally:
        conn.close()


def test_service_role_cannot_modify_binding_history_directly():
    runtime_id = _runtime_id("priv")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap_heads(conn, runtime_id)
        cycle_id = _h("1")
        assert _bind(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest["ledger_hash"],
            manifest["entries"][-1]["receipt"]["receipt_id"],
            cycle_id,
            _h("2"),
            _h("3"),
        )["status"] == "BOUND"

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_governed_cycle_bindings "
                    "set policy_fingerprint=%s where runtime_id=%s and cycle_id=%s",
                    (_h("9"), runtime_id, cycle_id),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
