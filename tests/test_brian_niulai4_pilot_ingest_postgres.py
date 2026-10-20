"""Real-Postgres tests for the NIULAI4 pilot sink and the read-only UI status functions.

Proves: the migration creates no key and refuses nothing silently (status NOT_CONFIGURED), only
shadow-only payloads with the run's own fingerprint are stored, one live run at a time, heartbeats
and reports are rate-limited, a silent run is reported STALE (never RUNNING), FINAL is stored once,
the tables are not readable by anon/authenticated, and the prediction status reflects facts
(DISABLED / NO_DATA / STALE / OK) without writing anything.
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
ORDER = [
    MIGRATIONS / "20261019195019_brian_prediction_intelligence.sql",
    MIGRATIONS / "20261019200517_brian_prediction_retention.sql",
    MIGRATIONS / "20261019202909_brian_prediction_probability_changes.sql",
    MIGRATIONS / "20261019210906_brian_prediction_resolutions.sql",
    MIGRATIONS / "20261020080000_brian_niulai4_pilot_ingest.sql",
]
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")

FP = "a" * 64
_BOOTSTRAP = """
do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin; end if;
end $$;
"""


@pytest.fixture(scope="module")
def conn():
    c = psycopg2.connect(DATABASE_URL)
    c.autocommit = True
    with c.cursor() as cur:
        cur.execute(_BOOTSTRAP)
        for path in ORDER:
            cur.execute(path.read_text(encoding="utf-8"))
    yield c
    c.close()


@pytest.fixture()
def cur(conn):
    c = conn.cursor()
    c.execute("delete from public.brian_niulai4_pilot_reports; delete from public.brian_niulai4_pilot_heartbeats; delete from public.brian_niulai4_pilot_runs")
    c.execute("update public.brian_niulai4_pilot_auth set enabled = false, ingest_key_sha256 = null, configured_at = null")
    yield c
    c.execute("reset role")


def rec(cur, payload):
    cur.execute("select public.brian_record_niulai4_pilot_v1(%s::jsonb)", (json.dumps(payload),))
    return cur.fetchone()[0]


def status(cur, as_of="now()"):
    cur.execute(f"select public.brian_niulai4_pilot_status_v1({as_of})")
    return cur.fetchone()[0]


def base(run, kind, **extra):
    p = {"kind": kind, "run_id": run, "observed_at": "now", "shadow_only": True, "trade_signal": False,
         "rules_version": "niulai4-pilot.v2", "rules_fingerprint": FP}
    p.update(extra)
    return p


def now_iso(cur, offset="0 seconds"):
    cur.execute(f"select to_char((now() + interval '{offset}') at time zone 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"')")
    return cur.fetchone()[0]


def start(cur, run=None):
    run = run or f"run-{uuid.uuid4().hex[:12]}"
    r = rec(cur, base(run, "START", observed_at=now_iso(cur), host_label="pc"))
    assert r["status"] == "STORED", r
    return run


def test_fresh_install_has_no_key_and_reports_not_configured(cur):
    cur.execute("select ingest_key_sha256, enabled from public.brian_niulai4_pilot_auth")
    assert cur.fetchall() == [(None, False)]
    s = status(cur)
    assert s["status"] == "NOT_CONFIGURED" and s["ingest_configured"] is False
    assert s["run"] is None and s["trade_signal"] is False


def test_enabled_without_hash_is_impossible(cur):
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("update public.brian_niulai4_pilot_auth set enabled = true")


def test_configured_key_without_run_is_not_started(cur):
    cur.execute("update public.brian_niulai4_pilot_auth set ingest_key_sha256 = %s, enabled = true, configured_at = now()", ("b" * 64,))
    assert status(cur)["status"] == "NOT_STARTED"


def test_rejects_non_shadow_bad_version_and_clock(cur):
    run = f"run-{uuid.uuid4().hex[:12]}"
    assert rec(cur, base(run, "START", observed_at=now_iso(cur), trade_signal=True))["reason"] == "NOT_SHADOW_ONLY"
    assert rec(cur, base(run, "START", observed_at=now_iso(cur), shadow_only=False))["reason"] == "NOT_SHADOW_ONLY"
    assert rec(cur, base(run, "START", observed_at=now_iso(cur), rules_version="niulai4-pilot.v1"))["reason"] == "BAD_RUN_FIELDS"
    assert rec(cur, base(run, "START", observed_at=now_iso(cur), rules_fingerprint="xyz"))["reason"] == "BAD_RUN_FIELDS"
    assert rec(cur, base(run, "START", observed_at=now_iso(cur, "10 minutes")))["reason"] == "OBSERVED_AT_OUT_OF_RANGE"
    assert rec(cur, base(run, "START", observed_at=now_iso(cur, "-2 hours")))["reason"] == "OBSERVED_AT_OUT_OF_RANGE"
    assert rec(cur, base(run, "START", observed_at="not a time"))["reason"] == "BAD_OBSERVED_AT"
    assert rec(cur, base("x", "START", observed_at=now_iso(cur)))["reason"] == "BAD_RUN_ID"
    assert rec(cur, base(run, "DELETE", observed_at=now_iso(cur)))["reason"] == "UNKNOWN_KIND"
    cur.execute("select count(*) from public.brian_niulai4_pilot_runs")
    assert cur.fetchone()[0] == 0


def test_heartbeat_flow_running_then_stale(cur):
    run = start(cur)
    r = rec(cur, base(run, "HEARTBEAT", observed_at=now_iso(cur), seq=1, health={"wsMessages": 10}))
    assert r["status"] == "STORED"
    s = status(cur)
    assert s["status"] == "RUNNING" and s["last_heartbeat"]["health"] == {"wsMessages": 10}
    # Second heartbeat within 30 s is rate-limited, not stored.
    assert rec(cur, base(run, "HEARTBEAT", observed_at=now_iso(cur), seq=2, health={}))["status"] == "RATE_LIMITED"
    # Seen from 10 minutes later, the same run is STALE -- never shown as running.
    assert status(cur, "now() + interval '10 minutes'")["status"] == "STALE"


def test_wrong_fingerprint_unknown_run_and_bad_health_rejected(cur):
    run = start(cur)
    assert rec(cur, base(run, "HEARTBEAT", observed_at=now_iso(cur), seq=1, health={}, rules_fingerprint="c" * 64))["reason"] == "FINGERPRINT_MISMATCH"
    assert rec(cur, base("run-unknown-1", "HEARTBEAT", observed_at=now_iso(cur), seq=1, health={}))["reason"] == "UNKNOWN_RUN"
    assert rec(cur, base(run, "HEARTBEAT", observed_at=now_iso(cur), seq=1, health=[1, 2]))["reason"] == "BAD_HEALTH"
    assert rec(cur, base(run, "HEARTBEAT", observed_at=now_iso(cur), seq=-1, health={}))["reason"] == "BAD_SEQ"
    assert rec(cur, base(run, "HEARTBEAT", observed_at=now_iso(cur), seq=1, health={"x": "y" * 20000}))["reason"] == "BAD_HEALTH"


def test_one_live_run_at_a_time_and_silent_run_is_closed(cur):
    first = start(cur)
    second = f"run-{uuid.uuid4().hex[:12]}"
    r = rec(cur, base(second, "START", observed_at=now_iso(cur)))
    assert r["reason"] == "ANOTHER_RUN_ACTIVE" and r["active_run_id"] == first
    # The first run goes silent for 11 minutes -> a new start closes it as HEARTBEAT_LOST.
    cur.execute("update public.brian_niulai4_pilot_runs set started_at = now() - interval '11 minutes', last_heartbeat_at = now() - interval '11 minutes' where run_id = %s", (first,))
    assert rec(cur, base(second, "START", observed_at=now_iso(cur)))["status"] == "STORED"
    cur.execute("select state, stop_reason from public.brian_niulai4_pilot_runs where run_id = %s", (first,))
    assert cur.fetchone() == ("STOPPED", "HEARTBEAT_LOST")
    assert rec(cur, base(second, "START", observed_at=now_iso(cur)))["status"] == "DUPLICATE"


def test_reports_rate_limited_final_once_and_trade_signal_enforced(cur):
    run = start(cur)
    summary = {"tradeSignal": False, "verdict": "INTERIM_NO_VERDICT"}
    assert rec(cur, base(run, "REPORT", observed_at=now_iso(cur), seq=1, report_kind="INTERIM", summary=summary))["status"] == "STORED"
    assert rec(cur, base(run, "REPORT", observed_at=now_iso(cur), seq=2, report_kind="INTERIM", summary=summary))["status"] == "RATE_LIMITED"
    cur.execute("update public.brian_niulai4_pilot_reports set received_at = now() - interval '6 minutes'")
    bad = rec(cur, base(run, "REPORT", observed_at=now_iso(cur), seq=3, report_kind="INTERIM", summary={"tradeSignal": True}))
    assert bad["reason"] == "BAD_SUMMARY"
    assert rec(cur, base(run, "REPORT", observed_at=now_iso(cur), seq=4, report_kind="FINAL", summary=summary))["status"] == "STORED"
    cur.execute("update public.brian_niulai4_pilot_reports set received_at = now() - interval '6 minutes'")
    assert rec(cur, base(run, "REPORT", observed_at=now_iso(cur), seq=5, report_kind="FINAL", summary=summary))["reason"] == "FINAL_ALREADY_STORED"
    assert rec(cur, base(run, "REPORT", observed_at=now_iso(cur), seq=6, report_kind="OTHER", summary=summary))["reason"] == "BAD_REPORT_KIND"
    s = status(cur)
    assert s["latest_report"]["kind"] == "FINAL" and s["counts"]["reports"] == 2


def test_stop_marks_completed_or_stopped_and_blocks_further_writes(cur):
    run = start(cur)
    assert rec(cur, base(run, "STOP", observed_at=now_iso(cur), stop_reason="DURATION_ELAPSED"))["status"] == "STORED"
    s = status(cur)
    assert s["status"] == "COMPLETED" and s["run"]["stop_reason"] == "DURATION_ELAPSED"
    assert rec(cur, base(run, "HEARTBEAT", observed_at=now_iso(cur), seq=9, health={}))["reason"] == "RUN_NOT_RUNNING"
    run2 = start(cur)
    rec(cur, base(run2, "STOP", observed_at=now_iso(cur), stop_reason="ACCESS_DENIED"))
    assert status(cur)["status"] == "STOPPED"


def test_tables_and_functions_closed_to_client_roles(cur):
    for role in ("anon", "authenticated"):
        cur.execute(f"set role {role}")
        for table in ("brian_niulai4_pilot_auth", "brian_niulai4_pilot_runs", "brian_niulai4_pilot_heartbeats", "brian_niulai4_pilot_reports"):
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(f"select 1 from public.{table} limit 1")
        for fn in ("brian_niulai4_pilot_status_v1(now())", "brian_prediction_status_v1(now())",
                   "brian_record_niulai4_pilot_v1('{}'::jsonb)"):
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(f"select public.{fn}")
        cur.execute("reset role")
    # service_role may read but never write the tables directly.
    cur.execute("set role service_role")
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        cur.execute("insert into public.brian_niulai4_pilot_runs (run_id, rules_version, rules_fingerprint, started_at) values ('run-direct-1', 'niulai4-pilot.v2', %s, now())", (FP,))
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        cur.execute("update public.brian_niulai4_pilot_auth set enabled = false")
    cur.execute("reset role")


def test_prediction_status_reflects_facts_and_writes_nothing(cur):
    cur.execute("update public.brian_prediction_control set module_enabled = false")
    s = json.loads(json.dumps(_pstatus(cur)))
    assert s["status"] == "DISABLED" and s["control"]["wallet_tracking_enabled"] is False
    cur.execute("update public.brian_prediction_control set module_enabled = true")
    cur.execute("delete from public.brian_prediction_source_health where source like 'test-status:%'")
    cur.execute("select count(*) from public.brian_prediction_source_health")
    if cur.fetchone()[0] == 0:
        assert _pstatus(cur)["status"] == "NO_DATA"
    hid = uuid.uuid4().hex
    cur.execute("insert into public.brian_prediction_source_health (health_id, source, status, observed_at) values (%s, 'test-status:a', 'OK', now() + interval '1 day')", (hid,))
    assert _pstatus(cur, "now() + interval '1 day 1 minute'")["status"] == "OK"
    assert _pstatus(cur, "now() + interval '1 day 41 minutes'")["status"] == "STALE"
    cur.execute("insert into public.brian_prediction_source_health (health_id, source, status, observed_at) values (%s, 'test-status:b', 'RATE_LIMITED', now() + interval '1 day')", (hid + "b",))
    s = _pstatus(cur, "now() + interval '1 day 1 minute'")
    assert s["status"] == "DEGRADED" and s["trade_signal"] is False
    assert {"source": "test-status:b", "status": "RATE_LIMITED"} in s["last_cycle_sources"]
    cur.execute("delete from public.brian_prediction_source_health where source like 'test-status:%'")
    cur.execute("update public.brian_prediction_control set module_enabled = false")


def _pstatus(cur, as_of="now()"):
    cur.execute(f"select public.brian_prediction_status_v1({as_of})")
    return cur.fetchone()[0]
