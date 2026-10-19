"""Real-Postgres tests for Prediction Intelligence Phase 1c (bounded storage).

Applies the Phase 1b migration and then 20261019200517_brian_prediction_retention.sql, and proves:
the storage cap blocks new evidence without deleting anything, health rows still record why, the
v1 batch contract keeps working under the cap, and the archive policies are registered without
enabling anything. Skipped unless psycopg2 and BRIAN_TEST_DATABASE_URL are available.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 is only installed in the Postgres CI job")

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "supabase" / "migrations"
BASE = MIGRATIONS / "20261019195019_brian_prediction_intelligence.sql"
RETENTION = MIGRATIONS / "20261019200517_brian_prediction_retention.sql"
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")

_BOOTSTRAP = """
do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin; end if;
end $$;
-- Minimal stand-in for the production archive policy table (same columns, PK and check).
create table if not exists public.brian_archive_policies (
  table_name text primary key,
  time_column text not null,
  pk_column text not null,
  hot_retention_days integer not null,
  batch_size integer not null default 500 check (batch_size between 50 and 2000),
  archive_enabled boolean not null default true,
  purge_enabled boolean not null default false,
  priority integer not null default 100,
  notes text,
  updated_at timestamptz not null default now()
);
"""


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


@pytest.fixture(scope="module", autouse=True)
def _apply():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP)
            cur.execute(BASE.read_text(encoding="utf-8"))
            cur.execute(RETENTION.read_text(encoding="utf-8"))
    finally:
        conn.close()


@pytest.fixture()
def cur():
    conn = _connect()
    c = conn.cursor()
    c.execute("update public.brian_prediction_control set module_enabled = true, wallet_tracking_enabled = false, max_storage_mb = 256")
    yield c
    c.execute("reset role")
    c.execute("update public.brian_prediction_control set module_enabled = false, wallet_tracking_enabled = false, max_storage_mb = 256")
    conn.close()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _market(uid):
    return {
        "market_key": f"polymarket:{uid}", "source": "polymarket-gamma", "market_id": uid, "condition_id": None,
        "event_id": None, "event_title": None, "question": f"Q {uid}?", "slug": None, "end_date": None,
        "outcomes": [{"label": "Yes", "clob_token_id": None}], "first_seen_at": "2026-10-19T19:30:00Z",
    }


def _health(uid):
    return {
        "health_id": f"h:{uid}", "source": "polymarket-gamma", "status": "OK", "http_status": 200, "attempts": 1,
        "latency_ms": 10, "records_seen": 1, "records_stored": 0, "detail": None,
        "observed_at": "2026-10-19T19:30:00Z", "metadata": {},
    }


def _record(cur, markets=(), snapshots=(), trades=(), health=()):
    cur.execute(
        "select public.brian_record_prediction_batch_v1(%s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)",
        tuple(json.dumps(list(x)) for x in (markets, snapshots, trades, health)),
    )
    return cur.fetchone()[0]


def test_retention_migration_is_idempotent_and_cap_defaults_to_256mb():
    conn = _connect()
    try:
        with conn.cursor() as c:
            c.execute(RETENTION.read_text(encoding="utf-8"))
            c.execute("select max_storage_mb, module_enabled from public.brian_prediction_control")
            assert c.fetchone()[0] == 256
    finally:
        conn.close()


def test_cap_bounds_are_enforced(cur):
    for bad in (0, 4096):
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute("update public.brian_prediction_control set max_storage_mb = %s", (bad,))


def test_under_cap_records_normally_and_reports_storage(cur):
    uid = _uid()
    r = _record(cur, [_market(uid)], [], [], [_health(uid)])
    assert r["status"] == "SUCCESS"
    assert r["markets"] == {"received": 1, "stored": 1}
    assert r["cap_bytes"] == 256 * 1024 * 1024
    assert 0 < r["storage_bytes"] < r["cap_bytes"]


def test_at_cap_no_new_evidence_is_stored_and_nothing_is_deleted(cur):
    # Fill the module's own health table past a 1 MB cap with padded rows written as postgres.
    cur.execute(
        """
        insert into public.brian_prediction_source_health (health_id, source, status, observed_at, metadata)
        select 'fill:' || g || ':' || %s, 'fill', 'OK', now(), jsonb_build_object('pad', repeat('x', 900))
        from generate_series(1, 2500) g
        """,
        (_uid(),),
    )
    cur.execute("select public.brian_prediction_storage_bytes()")
    assert cur.fetchone()[0] >= 1024 * 1024
    cur.execute("select count(*) from public.brian_prediction_markets")
    markets_before = cur.fetchone()[0]
    cur.execute("select count(*) from public.brian_prediction_source_health")
    health_before = cur.fetchone()[0]

    cur.execute("update public.brian_prediction_control set max_storage_mb = 1")
    uid = _uid()
    r = _record(cur, [_market(uid)], [], [], [_health(uid)])
    assert r["status"] == "STORAGE_CAP_REACHED"
    assert r["storage_bytes"] >= r["cap_bytes"] == 1024 * 1024
    assert r["health"] == {"received": 1, "stored": 1}, "the reason stays visible"

    cur.execute("select count(*) from public.brian_prediction_markets")
    assert cur.fetchone()[0] == markets_before, "no new market stored at the cap"
    cur.execute("select count(*) from public.brian_prediction_source_health")
    assert cur.fetchone()[0] == health_before + 1, "nothing deleted to make room"
    cur.execute("select metadata->>'storage_cap_reached', records_stored from public.brian_prediction_source_health where health_id = %s", (f"h:{uid}",))
    assert cur.fetchone() == ("true", 0)

    cur.execute("update public.brian_prediction_control set max_storage_mb = 256")
    again = _record(cur, [_market(uid)], [], [], [])
    assert again["status"] == "SUCCESS" and again["markets"]["stored"] == 1, "raising the cap resumes storage"


def test_disabled_module_still_wins_over_cap(cur):
    cur.execute("update public.brian_prediction_control set module_enabled = false, max_storage_mb = 1")
    assert _record(cur, [_market(_uid())]) == {"status": "OPERATOR_STOPPED"}


def test_storage_function_grants(cur):
    for role in ("anon", "authenticated"):
        cur.execute(f"set role {role}")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select public.brian_prediction_storage_bytes()")
        cur.execute("reset role")
    cur.execute("set role service_role")
    cur.execute("select public.brian_prediction_storage_bytes()")
    assert cur.fetchone()[0] > 0
    cur.execute("reset role")


def test_archive_policies_registered_conservatively(cur):
    cur.execute(
        "select table_name, time_column, pk_column, hot_retention_days, archive_enabled, purge_enabled "
        "from public.brian_archive_policies where table_name in ('brian_prediction_market_snapshots', "
        "'brian_prediction_markets', 'brian_prediction_source_health', 'brian_prediction_wallet_trades')"
    )
    # Sorted in Python: ORDER BY on text follows the database collation, which differs between a
    # C-locale cluster and the en_US CI image.
    rows = sorted(cur.fetchall())
    assert rows == [
        ("brian_prediction_market_snapshots", "received_at", "snapshot_id", 90, True, True),
        ("brian_prediction_markets", "first_seen_at", "market_key", 365, True, False),
        ("brian_prediction_source_health", "observed_at", "health_id", 14, True, True),
        ("brian_prediction_wallet_trades", "traded_at", "idempotency_key", 30, True, True),
    ]
    # Every policy column actually exists on its table (the archiver builds queries from them).
    for table, time_col, pk_col, *_ in rows:
        for col in (time_col, pk_col):
            cur.execute(
                "select 1 from information_schema.columns where table_schema='public' and table_name=%s and column_name=%s",
                (table, col),
            )
            assert cur.fetchone() is not None, f"{table}.{col} missing"
