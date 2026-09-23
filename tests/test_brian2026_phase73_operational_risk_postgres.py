"""Real Postgres integration tests for Phase73 operational-risk persistence."""

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

from brian2026.phase68_operational_risk_governor import (
    EquityPoint,
    OperationalRiskPolicy,
    RuntimeHealthEvent,
)
from brian2026.phase72_operational_risk_ledger import OperationalRiskLedger


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
            for path in MIGRATIONS:
                cur.execute(path.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _runtime_id(label: str) -> str:
    return f"pytest-phase73-{label}-{uuid.uuid4().hex[:10]}"


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


def _policy() -> OperationalRiskPolicy:
    return OperationalRiskPolicy(
        max_drawdown_fraction=0.50,
        max_daily_loss_fraction=0.50,
        max_unknown_order_outcomes=1,
        max_market_data_age_seconds=10.0,
    )


def _ledger(*, now: float = TS, halted: bool = False) -> OperationalRiskLedger:
    ledger = OperationalRiskLedger(_policy())
    governor = ledger.governor()
    receipt = governor.evaluate(
        now=now,
        equity_points=(
            EquityPoint(now - 60, 1000.0),
            EquityPoint(now, 1000.0),
        ),
        closed_trades=(),
        health_events=(),
        market_data_timestamp=now - 20 if halted else now,
    )
    ledger.append(receipt)
    return ledger


def _append_success(ledger: OperationalRiskLedger, *, now: float):
    governor = ledger.governor()
    receipt = governor.evaluate(
        now=now,
        equity_points=(
            EquityPoint(now - 60, 1000.0),
            EquityPoint(now, 1000.0),
        ),
        closed_trades=(),
        health_events=(RuntimeHealthEvent(now, "EXECUTION_SUCCESS"),),
        market_data_timestamp=now,
    )
    ledger.append(receipt)
    return ledger


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
        ledger = _ledger()
        manifest = ledger.manifest()

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
        lease = _acquire(setup, runtime_id, "owner-a")
        fence = lease["fencing_token"]
    finally:
        setup.close()

    healthy = _ledger().manifest()
    halted = _ledger(halted=True).manifest()
    barrier = threading.Barrier(2)

    def contender(manifest):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _commit(conn, runtime_id, "owner-a", fence, 0, manifest)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(contender, (healthy, halted)))

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
        first = _acquire(conn, runtime_id, "owner-a", 1)
        assert first["fencing_token"] == 1
        time.sleep(1.15)
        second = _acquire(conn, runtime_id, "owner-b", 30)
        assert second["status"] == "EXPIRED_RECOVERY"
        assert second["fencing_token"] == 2

        stale = _commit(conn, runtime_id, "owner-a", 1, 0, _ledger().manifest())
        assert stale["status"] == "LEASE_LOST"
        assert stale["committed"] is False

        good = _commit(conn, runtime_id, "owner-b", 2, 0, _ledger().manifest())
        assert good["status"] == "COMMITTED"
        assert good["version"] == 1
    finally:
        conn.close()


def test_risk_ledger_prefix_extends_append_only():
    runtime_id = _runtime_id("prefix")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a")
        fence = lease["fencing_token"]
        ledger = _ledger()
        first_manifest = ledger.manifest()
        first = _commit(conn, runtime_id, "owner-a", fence, 0, first_manifest)
        assert first["version"] == 1

        _append_success(ledger, now=TS + 1)
        second_manifest = ledger.manifest()
        second = _commit(conn, runtime_id, "owner-a", fence, 1, second_manifest)
        assert second["version"] == 2
        assert _count(conn, "brian_operational_risk_entries", runtime_id) == 2
        assert _count(conn, "brian_operational_risk_snapshots", runtime_id) == 2
        assert _read(conn, runtime_id)["ledger_hash"] == second_manifest["ledger_hash"]
    finally:
        conn.close()


def test_changed_persisted_prefix_is_rejected_without_advancing_head():
    runtime_id = _runtime_id("rewrite")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a")
        fence = lease["fencing_token"]
        ledger = _ledger()
        first_manifest = ledger.manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 0, first_manifest)["version"] == 1

        forged = copy.deepcopy(first_manifest)
        forged["entries"][0]["receipt"]["reasons"] = ["forged-history-rewrite"]
        forged["ledger_hash"] = "9" * 64

        with pytest.raises(psycopg2.Error, match="PHASE73_LEDGER_PREFIX_CONFLICT"):
            _commit(conn, runtime_id, "owner-a", fence, 1, forged)

        assert _read(conn, runtime_id)["version"] == 1
        assert _count(conn, "brian_operational_risk_snapshots", runtime_id) == 1
    finally:
        conn.close()


def test_historical_retry_does_not_roll_back_newer_risk_head():
    runtime_id = _runtime_id("historical")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a")
        fence = lease["fencing_token"]
        ledger = _ledger()
        first_manifest = ledger.manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 0, first_manifest)["version"] == 1

        _append_success(ledger, now=TS + 1)
        second_manifest = ledger.manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 1, second_manifest)["version"] == 2

        retry = _commit(conn, runtime_id, "owner-a", fence, 0, first_manifest)
        assert retry["status"] == "DUPLICATE_HISTORICAL"
        assert retry["version"] == 1
        assert retry["current_version"] == 2
        assert _read(conn, runtime_id)["ledger_hash"] == second_manifest["ledger_hash"]
        assert _read(conn, runtime_id)["version"] == 2
    finally:
        conn.close()


def test_same_ledger_hash_with_changed_manifest_is_integrity_error():
    runtime_id = _runtime_id("hash-conflict")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a")
        fence = lease["fencing_token"]
        manifest = _ledger().manifest()
        assert _commit(conn, runtime_id, "owner-a", fence, 0, manifest)["version"] == 1

        forged = copy.deepcopy(manifest)
        forged["current_state"] = "REDUCING"
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
            _ledger().manifest(),
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
