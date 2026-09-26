"""Real Postgres integration tests for Phase73 operational-risk persistence.

This file deliberately imports only stdlib + psycopg2. The dedicated Postgres
CI job validates the SQL concurrency boundary without installing Brian's full
runtime dependency graph. Phase72/73 Python content-hash reconstruction is
tested separately in the normal full Python suite.
"""

from __future__ import annotations

import copy
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
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase73 Postgres tests require CI database",
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
    return f"pytest-phase73-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _entry(
    sequence: int,
    *,
    entry_char: str,
    receipt_char: str,
    previous_entry_char: str | None,
    previous_state: str,
    trading_state: str,
    timestamp: float,
):
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "sequence": sequence,
        "previous_entry_id": (
            None if previous_entry_char is None else _h(previous_entry_char)
        ),
        "policy_hash": _h("e"),
        "receipt": {
            "schema_version": "brian.phase68-operational-risk-governor.v1",
            "timestamp": timestamp,
            "previous_state": previous_state,
            "trading_state": trading_state,
            "recommended_state": trading_state,
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
            "halt_latched": trading_state == "HALTED",
            "receipt_id": _h(receipt_char),
            "execution_failure_lock_until": None,
            "reconciliation_failure_lock_until": None,
            "asset_cooldown_until": [],
            "shadow_only": True,
            "live_execution": False,
        },
        "entry_id": _h(entry_char),
    }


def _manifest(
    *,
    ledger_char: str,
    entries: list[dict],
    current_state: str,
):
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "append_only": True,
        "policy": {
            "max_drawdown_fraction": 0.5,
            "max_daily_loss_fraction": 0.5,
        },
        "policy_hash": _h("e"),
        "initial_state": "ACTIVE",
        "entries": entries,
        "entry_count": len(entries),
        "head_entry_id": None if not entries else entries[-1]["entry_id"],
        "current_state": current_state,
        "halt_latched": current_state == "HALTED",
        "ledger_hash": _h(ledger_char),
        "shadow_only": True,
        "live_execution": False,
    }


def _active_manifest(ledger_char: str = "1"):
    return _manifest(
        ledger_char=ledger_char,
        entries=[
            _entry(
                0,
                entry_char="a",
                receipt_char="b",
                previous_entry_char=None,
                previous_state="ACTIVE",
                trading_state="ACTIVE",
                timestamp=TS,
            )
        ],
        current_state="ACTIVE",
    )


def _halted_manifest(ledger_char: str = "2"):
    return _manifest(
        ledger_char=ledger_char,
        entries=[
            _entry(
                0,
                entry_char="c",
                receipt_char="d",
                previous_entry_char=None,
                previous_state="ACTIVE",
                trading_state="HALTED",
                timestamp=TS,
            )
        ],
        current_state="HALTED",
    )


def _extended_active_manifest(ledger_char: str = "3"):
    first = _active_manifest()["entries"][0]
    return _manifest(
        ledger_char=ledger_char,
        entries=[
            first,
            _entry(
                1,
                entry_char="f",
                receipt_char="0",
                previous_entry_char="a",
                previous_state="ACTIVE",
                trading_state="ACTIVE",
                timestamp=TS + 1,
            ),
        ],
        current_state="ACTIVE",
    )


def _acquire(conn, runtime_id: str, owner: str, seconds: int = 30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_acquire_shadow_runtime_lease(%s,%s,%s)",
            (runtime_id, owner, seconds),
        )
        return cur.fetchone()[0]


def _commit(conn, runtime_id: str, owner: str, fence: int, version: int, manifest: dict):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_operational_risk_ledger(%s,%s,%s,%s,%s)",
            (runtime_id, owner, fence, version, Json(manifest)),
        )
        return cur.fetchone()[0]


def _read(conn, runtime_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_read_operational_risk_ledger(%s)",
            (runtime_id,),
        )
        return cur.fetchone()[0]


def _count(conn, table: str, runtime_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_first_risk_commit_persists_manifest_and_exact_retry_is_idempotent():
    runtime_id = _runtime_id("commit")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a")
        manifest = _active_manifest()
        first = _commit(conn, runtime_id, "owner-a", lease["fencing_token"], 0, manifest)
        duplicate = _commit(conn, runtime_id, "owner-a", lease["fencing_token"], 0, manifest)
        head = _read(conn, runtime_id)

        assert first["status"] == "COMMITTED"
        assert first["version"] == 1
        assert duplicate["status"] == "DUPLICATE_CURRENT"
        assert duplicate["version"] == 1
        assert head["version"] == 1
        assert head["ledger_hash"] == manifest["ledger_hash"]
        assert head["manifest"] == manifest
        assert _count(conn, "brian_operational_risk_snapshots", runtime_id) == 1
        assert _count(conn, "brian_operational_risk_entries", runtime_id) == 1
    finally:
        conn.close()


def test_two_concurrent_same_version_risk_commits_have_one_winner():
    runtime_id = _runtime_id("race")
    setup = _connect()
    try:
        fence = _acquire(setup, runtime_id, "owner-a")["fencing_token"]
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(manifest):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _commit(conn, runtime_id, "owner-a", fence, 0, manifest)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(contender, (_active_manifest(), _halted_manifest())))

    assert sum(row["status"] == "COMMITTED" for row in results) == 1
    assert sum(row["status"] == "CAS_CONFLICT" for row in results) == 1

    conn = _connect()
    try:
        assert _read(conn, runtime_id)["version"] == 1
        assert _count(conn, "brian_operational_risk_snapshots", runtime_id) == 1
        assert _count(conn, "brian_operational_risk_entries", runtime_id) == 1
    finally:
        conn.close()


def test_expired_runtime_fence_prevents_stale_risk_commit():
    runtime_id = _runtime_id("fence")
    conn = _connect()
    try:
        assert _acquire(conn, runtime_id, "owner-a", 1)["fencing_token"] == 1
        time.sleep(1.15)
        takeover = _acquire(conn, runtime_id, "owner-b", 30)
        assert takeover["status"] == "EXPIRED_RECOVERY"
        assert takeover["fencing_token"] == 2

        stale = _commit(conn, runtime_id, "owner-a", 1, 0, _active_manifest())
        assert stale["status"] == "LEASE_LOST"
        assert stale["committed"] is False

        good = _commit(conn, runtime_id, "owner-b", 2, 0, _active_manifest())
        assert good["status"] == "COMMITTED"
        assert good["version"] == 1
    finally:
        conn.close()


def test_risk_ledger_prefix_extends_append_only():
    runtime_id = _runtime_id("prefix")
    conn = _connect()
    try:
        fence = _acquire(conn, runtime_id, "owner-a")["fencing_token"]
        first = _active_manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 0, first)["version"] == 1

        second = _extended_active_manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 1, second)["version"] == 2
        assert _count(conn, "brian_operational_risk_entries", runtime_id) == 2
        assert _count(conn, "brian_operational_risk_snapshots", runtime_id) == 2
        assert _read(conn, runtime_id)["ledger_hash"] == second["ledger_hash"]
    finally:
        conn.close()


def test_changed_persisted_prefix_is_rejected_without_advancing_head():
    runtime_id = _runtime_id("rewrite")
    conn = _connect()
    try:
        fence = _acquire(conn, runtime_id, "owner-a")["fencing_token"]
        first = _active_manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 0, first)["version"] == 1

        forged = copy.deepcopy(first)
        forged["entries"][0]["receipt"]["reasons"] = ["forged-history-rewrite"]
        forged["ledger_hash"] = _h("4")
        with pytest.raises(psycopg2.Error, match="PHASE73_LEDGER_PREFIX_CONFLICT"):
            _commit(conn, runtime_id, "owner-a", fence, 1, forged)

        assert _read(conn, runtime_id)["version"] == 1
        assert _count(conn, "brian_operational_risk_snapshots", runtime_id) == 1
    finally:
        conn.close()


def test_state_chain_and_timestamp_regression_are_rejected_in_database():
    runtime_id = _runtime_id("continuity")
    conn = _connect()
    try:
        fence = _acquire(conn, runtime_id, "owner-a")["fencing_token"]

        bad_state = _manifest(
            ledger_char="5",
            entries=[
                _entry(
                    0,
                    entry_char="1",
                    receipt_char="2",
                    previous_entry_char=None,
                    previous_state="REDUCING",
                    trading_state="ACTIVE",
                    timestamp=TS,
                )
            ],
            current_state="ACTIVE",
        )
        with pytest.raises(psycopg2.Error, match="PHASE73_LEDGER_STATE_CHAIN"):
            _commit(conn, runtime_id, "owner-a", fence, 0, bad_state)

        bad_time = _manifest(
            ledger_char="6",
            entries=[
                _entry(
                    0,
                    entry_char="3",
                    receipt_char="4",
                    previous_entry_char=None,
                    previous_state="ACTIVE",
                    trading_state="ACTIVE",
                    timestamp=TS,
                ),
                _entry(
                    1,
                    entry_char="5",
                    receipt_char="6",
                    previous_entry_char="3",
                    previous_state="ACTIVE",
                    trading_state="ACTIVE",
                    timestamp=TS,
                ),
            ],
            current_state="ACTIVE",
        )
        with pytest.raises(psycopg2.Error, match="PHASE73_LEDGER_TIME"):
            _commit(conn, runtime_id, "owner-a", fence, 0, bad_time)
    finally:
        conn.close()


def test_historical_retry_does_not_roll_back_newer_risk_head():
    runtime_id = _runtime_id("historical")
    conn = _connect()
    try:
        fence = _acquire(conn, runtime_id, "owner-a")["fencing_token"]
        first = _active_manifest()
        second = _extended_active_manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 0, first)["version"] == 1
        assert _commit(conn, runtime_id, "owner-a", fence, 1, second)["version"] == 2

        retry = _commit(conn, runtime_id, "owner-a", fence, 0, first)
        assert retry["status"] == "DUPLICATE_HISTORICAL"
        assert retry["version"] == 1
        assert retry["current_version"] == 2
        assert _read(conn, runtime_id)["ledger_hash"] == second["ledger_hash"]
        assert _read(conn, runtime_id)["version"] == 2
    finally:
        conn.close()


def test_same_ledger_hash_with_changed_manifest_is_integrity_error():
    runtime_id = _runtime_id("hash-conflict")
    conn = _connect()
    try:
        fence = _acquire(conn, runtime_id, "owner-a")["fencing_token"]
        manifest = _active_manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 0, manifest)["version"] == 1

        forged = copy.deepcopy(manifest)
        forged["entries"][0]["receipt"]["reasons"] = ["changed-under-same-hash"]
        with pytest.raises(psycopg2.Error, match="PHASE73_LEDGER_HASH_CONFLICT"):
            _commit(conn, runtime_id, "owner-a", fence, 1, forged)
    finally:
        conn.close()


def test_service_role_cannot_mutate_risk_history_directly():
    runtime_id = _runtime_id("priv")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a")
        assert _commit(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            0,
            _active_manifest(),
        )["status"] == "COMMITTED"

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_operational_risk_snapshots "
                    "set current_state='REDUCING' where runtime_id=%s",
                    (runtime_id,),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
