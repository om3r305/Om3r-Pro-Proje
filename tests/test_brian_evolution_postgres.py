"""Real-Postgres invariants for Brian Evolution Treasury/Ocean persistence."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 only installed in Postgres CI")

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = [
    ROOT / "supabase" / "migrations" / "202609111670_brian_treasury_layer5.sql",
    ROOT / "supabase" / "migrations" / "202609111675_brian_treasury_atomic_cas_guard.sql",
    ROOT / "supabase" / "migrations" / "202609111690_brian_ocean_layer6.sql",
    ROOT / "supabase" / "migrations" / "202609111695_brian_ocean_atomic_control_guard.sql",
]
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")


def _connect():
    connection = psycopg2.connect(DATABASE_URL)
    connection.autocommit = True
    return connection


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _snapshot(snapshot_id: str, cycle_id: str, observed_at: str, previous_snapshot_id: str | None, *, equity: float = 10_000, cash: float = 10_000, deployment: float = 0, deployment_pct: float = 0):
    return {
        "snapshot_id": snapshot_id,
        "cycle_id": cycle_id,
        "observed_at": observed_at,
        "previous_snapshot_id": previous_snapshot_id,
        "starting_equity_usd": 10_000,
        "cash_usd": cash,
        "equity_usd": equity,
        "realized_pnl_usd": 0,
        "cumulative_costs_usd": 0,
        "deployment_usd": deployment,
        "deployment_pct": deployment_pct,
        "cash_reserve_pct": 0 if equity <= 0 else cash / equity,
        "positions": [],
        "promotion_gate_open": False,
        "promotion_gate_ref": None,
        "promotion_gate_reason": "test",
        "treasury_version": "brian.treasury-shadow.v3",
        "gate_version": "brian.treasury-promotion-gate.v2",
        "blocked_reasons": [],
        "metadata": {"test": True},
    }


@pytest.fixture(scope="module", autouse=True)
def _setup():
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                do $$ begin
                  if not exists(select 1 from pg_roles where rolname='anon') then create role anon nologin; end if;
                  if not exists(select 1 from pg_roles where rolname='authenticated') then create role authenticated nologin; end if;
                  if not exists(select 1 from pg_roles where rolname='service_role') then create role service_role nologin; end if;
                end $$;
                alter role service_role bypassrls;
                create schema if not exists brian_private;
                create or replace function public.brian_reject_mutation()
                returns trigger language plpgsql as $$begin raise exception 'append only'; end$$;
                """
            )
            for migration in MIGRATIONS:
                cursor.execute(migration.read_text(encoding="utf-8"))
    finally:
        connection.close()


def _commit(snapshot: dict):
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("set role service_role")
            cursor.execute(
                "select public.brian_commit_treasury_shadow_cycle(%s::jsonb,%s::jsonb)",
                (json.dumps(snapshot), "[]"),
            )
            return cursor.fetchone()[0]
    finally:
        connection.close()


def test_treasury_cas_rejects_stale_parent_and_keeps_exact_retry_idempotent():
    first = _snapshot("snap-1", "cycle-1", "2026-09-12T00:00:00Z", None)
    second = _snapshot("snap-2", "cycle-2", "2026-09-12T00:01:00Z", "snap-1")
    assert _commit(first) == "cycle-1"
    assert _commit(second) == "cycle-2"
    assert _commit(second) == "cycle-2"

    stale = _snapshot("snap-stale", "cycle-stale", "2026-09-12T00:02:00Z", "snap-1")
    with pytest.raises(psycopg2.Error, match="TREASURY_COMMIT_STALE_PARENT"):
        _commit(stale)


def test_full_conviction_drawdown_snapshot_remains_persistable():
    drawdown = _snapshot(
        "snap-3",
        "cycle-3",
        "2026-09-12T00:03:00Z",
        "snap-2",
        equity=9_980,
        cash=0,
        deployment=9_990,
        deployment_pct=1.0,
    )
    assert _commit(drawdown) == "cycle-3"


def test_treasury_concurrent_children_cannot_fork_parent_chain():
    left = _snapshot("snap-left", "cycle-left", "2026-09-12T00:04:00Z", "snap-3")
    right = _snapshot("snap-right", "cycle-right", "2026-09-12T00:04:01Z", "snap-3")

    def attempt(snapshot):
        try:
            return ("ok", _commit(snapshot))
        except psycopg2.Error as error:
            return ("error", str(error))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [left, right]))

    assert sum(kind == "ok" for kind, _ in results) == 1, results
    assert sum("TREASURY_COMMIT_STALE_PARENT" in payload for kind, payload in results if kind == "error") == 1, results


def _ocean_start(command_id: str, run_id: str, requested_at: str):
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("set role service_role")
            cursor.execute(
                "select public.brian_ocean_start_run(%s,%s,%s::timestamptz,24,%s,%s,%s::jsonb)",
                (command_id, run_id, requested_at, "test", "ci", "{}"),
            )
            return cursor.fetchone()[0]
    finally:
        connection.close()


def test_ocean_rejects_backdated_privileged_start():
    old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    with pytest.raises(psycopg2.Error, match="OCEAN_CONTROL_TIMESTAMP_SKEW"):
        _ocean_start("backdated-command", "backdated-run", old_time)


def test_ocean_concurrent_start_serializes_to_one_active_run_and_direct_insert_is_denied():
    requested_at = _now_iso()

    def attempt(args):
        try:
            return ("ok", _ocean_start(*args))
        except psycopg2.Error as error:
            return ("error", str(error))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [
            ("cmd-a", "run-a", requested_at),
            ("cmd-b", "run-b", requested_at),
        ]))

    assert sum(kind == "ok" for kind, _ in results) == 1, results
    assert sum("OCEAN_RUN_ALREADY_ACTIVE" in payload for kind, payload in results if kind == "error") == 1, results
    active_run = next(payload for kind, payload in results if kind == "ok")

    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cursor.execute(
                    """
                    insert into public.brian_ocean_run_commands(
                      command_id,run_id,command,requested_at,duration_hours,requested_by
                    ) values('direct-write','direct-run','START',clock_timestamp(),24,'ci')
                    """
                )
            with pytest.raises(psycopg2.Error, match="OCEAN_RUN_NOT_ACTIVE"):
                cursor.execute(
                    "select public.brian_ocean_stop_run('wrong-stop','not-the-active-run',clock_timestamp(),'test','ci','{}'::jsonb)"
                )
            cursor.execute(
                "select public.brian_ocean_stop_run('good-stop',%s,clock_timestamp(),'test','ci','{}'::jsonb)",
                (active_run,),
            )
            assert cursor.fetchone()[0] == active_run
    finally:
        connection.close()
