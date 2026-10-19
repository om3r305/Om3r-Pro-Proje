"""Real-Postgres test for Prediction Intelligence Phase 1d (schedule + snapshots-only switch).

Applies the 1b, 1c and 1d migrations in order. Proves the switch is snapshots-only (module on,
wallet tracking explicitly off even if it had been on), that the migration is idempotent, and
that it applies cleanly on a database without pg_cron. Skipped unless psycopg2 and
BRIAN_TEST_DATABASE_URL are available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 is only installed in the Postgres CI job")

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "supabase" / "migrations"
ORDER = [
    MIGRATIONS / "20261019195019_brian_prediction_intelligence.sql",
    MIGRATIONS / "20261019200517_brian_prediction_retention.sql",
    MIGRATIONS / "20261019220000_brian_prediction_schedule.sql",
]
SCHEDULE = ORDER[-1]
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")

_BOOTSTRAP = """
do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin; end if;
end $$;
"""


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


@pytest.fixture()
def cur():
    conn = _connect()
    c = conn.cursor()
    c.execute(_BOOTSTRAP)
    for path in ORDER[:-1]:
        c.execute(path.read_text(encoding="utf-8"))
    yield c
    c.execute("update public.brian_prediction_control set module_enabled = false, wallet_tracking_enabled = false")
    conn.close()


def _control(cur):
    cur.execute("select module_enabled, wallet_tracking_enabled, shadow_only, live_execution from public.brian_prediction_control")
    return cur.fetchone()


def test_schedule_migration_enables_snapshots_only(cur):
    cur.execute("update public.brian_prediction_control set module_enabled = false, wallet_tracking_enabled = true")
    cur.execute(SCHEDULE.read_text(encoding="utf-8"))
    assert _control(cur) == (True, False, True, False), "module on, wallet tracking forced off, still shadow-only"


def test_schedule_migration_is_idempotent_and_skips_without_pg_cron(cur):
    cur.execute(SCHEDULE.read_text(encoding="utf-8"))
    cur.execute(SCHEDULE.read_text(encoding="utf-8"))
    assert _control(cur) == (True, False, True, False)
    cur.execute("select to_regnamespace('cron') is null")
    if cur.fetchone()[0]:
        return  # vanilla CI Postgres: the schedule step must be a no-op notice, which it was
    cur.execute("select count(*) from cron.job where jobname = 'brian-prediction-eye-15m'")
    assert cur.fetchone()[0] == 1, "re-applying never duplicates the job"


def test_enabled_module_records_through_the_capped_rpc(cur):
    cur.execute(SCHEDULE.read_text(encoding="utf-8"))
    cur.execute("select public.brian_record_prediction_batch_v1('[]','[]','[]','[]')")
    r = cur.fetchone()[0]
    assert r["status"] == "SUCCESS" and r["trades"]["tracking_enabled"] is False
    assert r["cap_bytes"] == 256 * 1024 * 1024
