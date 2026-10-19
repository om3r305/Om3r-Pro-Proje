"""Real-Postgres tests for Prediction Intelligence Phase 2c (resolutions + calibration inputs).

Proves: only genuine resolutions can be stored (and only while the module is on), the candidate
list is bounded and excludes resolved markets, calibration inputs are point-in-time (forecast
received at or before end_date - lead; resolutions observed after as_of are invisible), the cluster
key is the event id, and exclusions are counted rather than back-filled.
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
    MIGRATIONS / "20261020000000_brian_prediction_resolutions.sql",
]
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")

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
    c.execute("update public.brian_prediction_control set module_enabled = true, max_storage_mb = 256")
    yield c
    c.execute("reset role")
    c.execute("update public.brian_prediction_control set module_enabled = false")


def _market(cur, end_date, event_id=None):
    mid = str(uuid.uuid4().int % 10**12)
    key = f"polymarket:{mid}"
    cur.execute(
        "insert into public.brian_prediction_markets (market_key, source, market_id, event_id, question, end_date, outcomes, first_seen_at) "
        "values (%s, 'polymarket-gamma', %s, %s, %s, %s, '[{\"label\":\"Yes\"},{\"label\":\"No\"}]'::jsonb, '2033-01-01T00:00:00Z')",
        (key, mid, event_id, f"Q {mid}?", end_date),
    )
    return mid, key


def _snap(cur, key, received_at, p):
    cur.execute(
        "insert into public.brian_prediction_market_snapshots (snapshot_id, market_key, received_at, outcomes, active, closed, model_version) "
        "values (%s, %s, %s, %s::jsonb, true, false, 'test')",
        (f"{key}|{received_at}", key, received_at, json.dumps([{"label": "Yes", "implied_probability": p}, {"label": "No", "implied_probability": None if p is None else 1 - p}])),
    )


def _resolution(key, yes, received_at="2033-02-01T00:00:00Z", prices=None, uma="resolved"):
    return {
        "resolution_id": f"{key}|resolved", "market_key": key, "winning_label": "Yes" if yes == 1 else "No",
        "yes_outcome": yes, "final_prices": prices or [{"label": "Yes", "price": yes}, {"label": "No", "price": 1 - yes}],
        "uma_resolution_status": uma, "closed_time": None, "received_at": received_at, "model_version": "test",
    }


def _record(cur, rows):
    cur.execute("select public.brian_record_prediction_resolutions_v1(%s::jsonb)", (json.dumps(rows),))
    return cur.fetchone()[0]


def _inputs(cur, as_of, lead="24 hours"):
    cur.execute("select public.brian_prediction_calibration_inputs_v1(%s::timestamptz, %s::interval)", (as_of, lead))
    return cur.fetchone()[0]


def test_record_is_gated_idempotent_and_requires_a_known_market(cur):
    _, key = _market(cur, "2033-01-10T00:00:00Z")
    cur.execute("update public.brian_prediction_control set module_enabled = false")
    assert _record(cur, [_resolution(key, 1)]) == {"status": "OPERATOR_STOPPED"}
    cur.execute("update public.brian_prediction_control set module_enabled = true")
    assert _record(cur, [_resolution(key, 1)]) == {"status": "SUCCESS", "received": 1, "stored": 1}
    assert _record(cur, [_resolution(key, 1)])["stored"] == 0, "re-recording is a no-op"
    assert _record(cur, [_resolution("polymarket:unknown", 1)])["stored"] == 0, "unknown market is not stored"


def test_only_final_resolutions_satisfy_the_constraints(cur):
    for bad in (
        {"prices": [{"label": "Yes", "price": 0.5}, {"label": "No", "price": 0.5}]},
        {"prices": [{"label": "Yes", "price": 1}, {"label": "No", "price": 1}]},
        {"prices": [{"label": "Yes", "price": 0.9995}, {"label": "No", "price": 0.0005}]},
        {"uma": "proposed"},
    ):
        _, key = _market(cur, "2033-01-10T00:00:00Z")
        with pytest.raises(psycopg2.errors.CheckViolation):
            _record(cur, [_resolution(key, 1, prices=bad.get("prices"), uma=bad.get("uma", "resolved"))])


def test_resolutions_are_append_only_and_private(cur):
    _, key = _market(cur, "2033-01-10T00:00:00Z")
    _record(cur, [_resolution(key, 0)])
    cur.execute("set role service_role")
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        cur.execute("update public.brian_prediction_resolutions set winning_label = 'Yes'")
    cur.execute("reset role")
    for role in ("anon", "authenticated"):
        cur.execute(f"set role {role}")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select * from public.brian_prediction_resolutions")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select public.brian_prediction_calibration_inputs_v1()")
        cur.execute("reset role")


def test_candidates_are_bounded_oldest_first_and_skip_resolved_or_not_yet_due(cur):
    # Year 2001 keeps these markets older than every other test's markets, so the bounded
    # list is not filled by unrelated rows sharing the database.
    as_of = "2001-03-01T12:00:00Z"
    due = [_market(cur, f"2001-03-01T0{h}:00:00Z") for h in range(1, 5)]
    late = _market(cur, "2001-03-01T11:30:00Z")  # inside the 1h grace: not yet due
    _record(cur, [_resolution(due[0][1], 1)])  # already resolved
    cur.execute("select market_id, market_key from public.brian_prediction_resolution_candidates_v1(%s::timestamptz, 50)", (as_of,))
    got = [r[1] for r in cur.fetchall()]
    mine = [k for k in got if k in {d[1] for d in due}]
    assert mine == [due[1][1], due[2][1], due[3][1]], "oldest first, resolved excluded"
    assert late[1] not in got, "a market inside the grace period is not due yet"
    assert len(got) <= 10, "bounded to 10 even when more are requested"


def test_calibration_inputs_are_point_in_time_and_clustered_by_event(cur):
    ev = f"ev-{uuid.uuid4().hex[:6]}"
    end = "2035-06-10T00:00:00Z"
    _, a = _market(cur, end, ev)
    _, b = _market(cur, end, ev)
    _, c = _market(cur, end, None)       # no event id -> clusters on its own market key
    _, d = _market(cur, end, ev)         # only a LATE snapshot -> excluded, not back-filled
    _snap(cur, a, "2035-06-08T00:00:00Z", 0.30)
    _snap(cur, a, "2035-06-09T12:00:00Z", 0.95)   # after end - 24h: must not be used
    _snap(cur, b, "2035-06-08T10:00:00Z", 0.60)
    _snap(cur, c, "2035-06-07T00:00:00Z", 0.10)
    _snap(cur, d, "2035-06-09T20:00:00Z", 0.50)
    _record(cur, [_resolution(a, 1, "2035-06-10T02:00:00Z"), _resolution(b, 0, "2035-06-10T02:00:00Z"),
                  _resolution(c, 0, "2035-06-10T02:00:00Z"), _resolution(d, 1, "2035-06-10T02:00:00Z")])

    before = _inputs(cur, "2035-06-10T01:00:00Z")
    assert not [r for r in before["rows"] if r["marketKey"] in (a, b, c, d)], "resolutions observed after as_of are invisible"

    out = _inputs(cur, "2035-06-11T00:00:00Z")
    rows = {r["marketKey"]: r for r in out["rows"] if r["marketKey"] in (a, b, c, d)}
    assert set(rows) == {a, b, c}
    assert float(rows[a]["forecast"]) == 0.30, "the later 0.95 snapshot is after end - lead and must not leak"
    assert rows[a]["outcome"] == 1 and rows[b]["outcome"] == 0
    assert rows[a]["eventId"] == ev == rows[b]["eventId"], "markets of one event share a cluster"
    assert rows[c]["eventId"] == c, "no event id -> its own cluster"
    assert out["excluded"]["no_forecast_before_lead"] >= 1, "the late-only market is counted as excluded"
    assert out["trade_signal"] is False

    shorter = _inputs(cur, "2035-06-11T00:00:00Z", "1 hour")
    srows = {r["marketKey"]: r for r in shorter["rows"] if r["marketKey"] in (a, d)}
    assert float(srows[a]["forecast"]) == 0.95 and float(srows[d]["forecast"]) == 0.50, "lead is a real parameter"


def test_non_binary_resolutions_are_counted_not_guessed(cur):
    _, key = _market(cur, "2036-01-01T00:00:00Z")
    row = _resolution(key, 1)
    row["yes_outcome"] = None
    row["winning_label"] = "Candidate A"
    row["final_prices"] = [{"label": "Candidate A", "price": 1}, {"label": "Candidate B", "price": 0}]
    assert _record(cur, [row])["stored"] == 1
    out = _inputs(cur, "2036-02-01T00:00:00Z")
    assert key not in {r["marketKey"] for r in out["rows"]}
    assert out["excluded"]["non_binary_resolutions"] >= 1
