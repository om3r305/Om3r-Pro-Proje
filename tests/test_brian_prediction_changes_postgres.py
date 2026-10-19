"""Real-Postgres tests for Prediction Intelligence Phase 2a (point-in-time changes + data health).

Proves the two read-only functions never use information received after p_as_of, report a missing
history as NULL/INSUFFICIENT_HISTORY rather than 0, flag stale markets, and summarize source
health and cycle coverage. Skipped unless psycopg2 and BRIAN_TEST_DATABASE_URL are available.
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
    MIGRATIONS / "20261019204328_brian_prediction_probability_changes_elapsed.sql",
]
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")

AS_OF = "2031-01-01T12:00:00Z"

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
    yield c
    c.execute("reset role")


def _market(cur, outcomes=("Yes", "No")):
    key = f"polymarket:t-{uuid.uuid4().hex[:10]}"
    cur.execute(
        "insert into public.brian_prediction_markets (market_key, source, market_id, question, outcomes, first_seen_at) "
        "values (%s, 'polymarket-gamma', %s, %s, %s::jsonb, '2031-01-01T00:00:00Z')",
        (key, key, f"Q {key}?", json.dumps([{"label": o, "clob_token_id": None} for o in outcomes])),
    )
    return key


def _snap(cur, key, received_at, p_yes, labels=("Yes", "No"), volume=1000):
    outcomes = [{"label": labels[0], "implied_probability": p_yes, "clob_token_id": None},
                {"label": labels[1], "implied_probability": None if p_yes is None else round(1 - p_yes, 6), "clob_token_id": None}]
    cur.execute(
        "insert into public.brian_prediction_market_snapshots (snapshot_id, market_key, source_updated_at, received_at, outcomes, "
        "active, closed, model_version, volume_24h) values (%s, %s, null, %s, %s::jsonb, true, false, 'test', %s)",
        (f"{key}|{received_at}", key, received_at, json.dumps(outcomes), volume),
    )


def _changes(cur, as_of=AS_OF):
    cur.execute("select public.brian_prediction_probability_changes_v1(%s::timestamptz, 300)", (as_of,))
    out = cur.fetchone()[0]
    return out, {m["market_key"]: m for m in out["markets"]}


def test_change_uses_only_snapshots_received_by_as_of(cur):
    key = _market(cur)
    _snap(cur, key, "2031-01-01T10:30:00Z", 0.40)   # older than 1h -> baseline
    _snap(cur, key, "2031-01-01T11:50:00Z", 0.55)   # latest known at as_of
    _snap(cur, key, "2031-01-01T12:05:00Z", 0.99)   # arrives AFTER as_of: must be invisible
    out, rows = _changes(cur)
    m = rows[key]
    assert float(m["probability"]) == 0.55, "a later-received snapshot never leaks in"
    assert float(m["p_1h"]) == 0.40
    assert float(m["change_1h"]) == pytest.approx(0.15)
    assert m["change_6h"] is None and m["change_24h"] is None, "no history that old -> NULL, not 0"
    assert m["data_status"] == "FRESH"
    assert m["snapshot_count"] == 2
    assert out["trade_signal"] is False and out["method"] == "point_in_time_received_at_only"


def test_insufficient_history_is_a_state_not_a_zero_move(cur):
    key = _market(cur)
    _snap(cur, key, "2031-01-01T11:40:00Z", 0.30)
    _, rows = _changes(cur)
    m = rows[key]
    assert m["data_status"] == "INSUFFICIENT_HISTORY"
    assert m["change_1h"] is None and m["p_1h"] is None


def test_stale_market_is_flagged(cur):
    key = _market(cur)
    _snap(cur, key, "2031-01-01T05:00:00Z", 0.20)
    _snap(cur, key, "2031-01-01T10:00:00Z", 0.25)
    _, rows = _changes(cur)
    m = rows[key]
    assert m["data_status"] == "STALE"
    assert m["age_seconds"] == 7200
    # As of 12:00 the value known 1h earlier (11:00) is the same 10:00 snapshot, so what Brian
    # knew did not change over the last hour; the 6h window (06:00) reaches back to the 05:00 snapshot.
    assert float(m["change_1h"]) == 0.0
    assert float(m["change_6h"]) == pytest.approx(0.05), "stale is flagged, not hidden"


def test_missing_probability_stays_null(cur):
    key = _market(cur)
    _snap(cur, key, "2031-01-01T10:00:00Z", 0.5)
    _snap(cur, key, "2031-01-01T11:55:00Z", None)
    _, rows = _changes(cur)
    m = rows[key]
    assert m["probability"] is None and m["change_1h"] is None
    assert m["data_status"] == "NO_PROBABILITY"


def test_yes_label_is_preferred_over_position(cur):
    key = _market(cur, outcomes=("No", "Yes"))
    # First outcome is "No"=0.9, second "Yes"=0.1: the function must report 0.1.
    for at in ("2031-01-01T10:00:00Z", "2031-01-01T11:55:00Z"):
        cur.execute(
            "insert into public.brian_prediction_market_snapshots (snapshot_id, market_key, received_at, outcomes, active, closed, model_version) "
            "values (%s, %s, %s, %s::jsonb, true, false, 'test')",
            (f"{key}|{at}", key, at, json.dumps([{"label": "No", "implied_probability": 0.9}, {"label": "Yes", "implied_probability": 0.1}])),
        )
    _, rows = _changes(cur)
    assert float(rows[key]["probability"]) == 0.1


def test_earlier_as_of_sees_earlier_world(cur):
    key = _market(cur)
    _snap(cur, key, "2031-01-01T08:00:00Z", 0.10)
    _snap(cur, key, "2031-01-01T11:50:00Z", 0.60)
    _, rows = _changes(cur, "2031-01-01T09:00:00Z")
    m = rows[key]
    assert float(m["probability"]) == 0.10, "as of 09:00 only the 08:00 snapshot existed"
    assert m["snapshot_count"] == 1


def test_health_summary_counts_sources_cycles_and_storage(cur):
    tag = uuid.uuid4().hex[:6]
    for minute, status in ((0, "OK"), (15, "OK"), (30, "BLOCKED")):
        cur.execute(
            "insert into public.brian_prediction_source_health (health_id, source, status, http_status, latency_ms, observed_at) "
            "values (%s, %s, %s, 200, 100, %s::timestamptz)",
            (f"h-{tag}-{minute}", f"polymarket-gamma-events:t{tag}", status, f"2032-06-01T11:{minute:02d}:00Z"),
        )
    cur.execute("select public.brian_prediction_health_summary_v1('2032-06-01T11:59:00Z'::timestamptz, interval '1 hour')")
    out = cur.fetchone()[0]
    src = {s["source"]: s for s in out["sources"]}[f"polymarket-gamma-events:t{tag}"]
    assert (src["rows"], src["ok"], src["not_ok"], src["last_status"]) == (3, 2, 1, "BLOCKED")
    assert out["cycles"]["observed"] == 3
    assert out["cycles"]["expected"] >= 1
    assert out["storage"]["bytes"] > 0 and out["storage"]["cap_bytes"] > 0
    assert out["window_seconds"] == 3600


def test_functions_are_service_role_only(cur):
    for role in ("anon", "authenticated"):
        cur.execute(f"set role {role}")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select public.brian_prediction_probability_changes_v1()")
        cur.execute("reset role")
        cur.execute(f"set role {role}")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select public.brian_prediction_health_summary_v1()")
        cur.execute("reset role")
    cur.execute("set role service_role")
    cur.execute("select public.brian_prediction_probability_changes_v1()")
    assert "markets" in cur.fetchone()[0]
    cur.execute("reset role")


def test_point_and_relative_change_are_separate_and_elapsed_time_is_real(cur):
    key = _market(cur)
    _snap(cur, key, "2031-01-01T10:20:00Z", 0.40)   # baseline actually received 1h40m before latest
    _snap(cur, key, "2031-01-01T12:00:00Z", 0.50)
    out, rows = _changes(cur)
    m = rows[key]
    assert float(m["change_1h"]) == pytest.approx(0.10), "+10 percentage points"
    assert float(m["relative_change_1h"]) == pytest.approx(0.25), "+25% relative"
    assert m["elapsed_1h_seconds"] == 6000, "real gap between observations, not the nominal 1h"
    assert m["baseline_1h_at"].startswith("2031-01-01T10:20:00")
    assert "probability_points" in out["change_unit"]


def test_relative_change_is_null_when_baseline_is_zero(cur):
    key = _market(cur)
    _snap(cur, key, "2031-01-01T10:00:00Z", 0.0)
    _snap(cur, key, "2031-01-01T11:59:00Z", 0.02)
    _, rows = _changes(cur)
    m = rows[key]
    assert float(m["change_1h"]) == pytest.approx(0.02)
    assert m["relative_change_1h"] is None
