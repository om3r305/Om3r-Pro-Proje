"""Real disposable Postgres tests. The DSN guard rejects every non-local server."""
import concurrent.futures
import copy
import datetime as dt
import os
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg2")
from psycopg2.extras import Json
from psycopg2.extensions import parse_dsn

ROOT = Path(__file__).resolve().parents[1]
POLICY = "dip-v8-integrity-20260908.1"
DSN = os.environ.get("BRIAN_V8_TEST_DATABASE_URL", "")
CONFIG = {"engine_version": "brian-dip-chart-reader-v8", "policy_version": POLICY, "symbols": ["ETHUSDT"], "server_authoritative": True, "browser_execution": False, "shadow_only": True, "live_execution": False, "allow_shadow_short": False, "execution_mode": "SHADOW_PAPER"}


@pytest.fixture(scope="module", autouse=True)
def schema():
    if not DSN:
        pytest.skip("dedicated local Postgres DSN required")
    assert parse_dsn(DSN).get("host") in {"127.0.0.1", "localhost"}, "Production DB is forbidden in this suite"
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
          do $$ begin
            if not exists(select 1 from pg_roles where rolname='anon') then create role anon; end if;
            if not exists(select 1 from pg_roles where rolname='authenticated') then create role authenticated; end if;
            if not exists(select 1 from pg_roles where rolname='service_role') then create role service_role bypassrls; end if;
          end $$;
          create schema if not exists cron;
          create table cron.job(jobid bigint,jobname text);
          create table public.brian_dip_session_events(event_id text primary key,session_id text not null,event_kind text not null,requested_at timestamptz not null default clock_timestamp(),starting_equity numeric,trade_notional numeric,config jsonb,engine_token_sha256 text);
          create table public.brian_dip_events(event_id text primary key,session_id text not null,observed_at timestamptz,event_kind text,symbol text,price numeric,entry_price numeric,exit_price numeric,quantity numeric,notional numeric,fees numeric,realized_pnl numeric,cash_after numeric,equity_after numeric,metadata jsonb,shadow_only boolean,live_execution boolean);
          create table public.brian_dip_snapshots(snapshot_id text primary key,session_id text not null,observed_at timestamptz,cash numeric,equity numeric,realized_pnl numeric,unrealized_pnl numeric,trade_count integer,win_count integer,loss_count integer,state jsonb,shadow_only boolean,live_execution boolean);
          create table public.brian_collector_leases(collector_id text primary key,owner_token text,lease_until timestamptz);
          create function public.brian_reject_mutation() returns trigger language plpgsql as $$ begin raise exception 'APPEND_ONLY'; end $$;
          grant select,insert on public.brian_dip_session_events,public.brian_dip_events,public.brian_dip_snapshots to service_role;
          grant select,update on public.brian_collector_leases to service_role;
        """)
        cur.execute((ROOT / "supabase/migrations/20260908140703_brian_dip_v8_integrity.sql").read_text())


@pytest.fixture(autouse=True)
def fresh(schema):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("truncate public.brian_dip_v8_runtime,public.brian_dip_v8_ledger,public.brian_dip_v8_decisions,public.brian_dip_session_events,public.brian_dip_events,public.brian_dip_snapshots,public.brian_collector_leases")
        cur.execute("insert into public.brian_dip_session_events(event_id,session_id,event_kind,starting_equity,config) values('start','session','START',1000,%s)", (Json(CONFIG),))
        cur.execute("insert into public.brian_collector_leases values('brian-dip-shadow-worker-v8','owner',clock_timestamp()+interval '5 minutes')")


def rows(sql, args=()):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall() if cur.description else []


def buy_packet():
    rt = rows("select runtime from public.brian_dip_v8_runtime where session_id='session'")[0][0]
    now = dt.datetime.now(dt.timezone.utc)
    pos = {"side": "LONG", "venue": "SPOT", "position_id": "position", "thesis_id": "occ", "episode_id": "episode", "entry": 100, "stop": 99, "target": 103, "qty": .8, "notional": 80, "fees_open": .08, "checked_until": int(now.timestamp()*1000), "market_price": 100}
    rt.update(cash=919.92, pos=pos, lastOccurrence="occ", lastSnapshotHour=now.isoformat()[:13])
    snapshot = {"snapshot_id": "snap", "session_id": "session", "observed_at": now.isoformat(), "cash": 919.92, "equity": 999.84, "realized_pnl": 0, "unrealized_pnl": -.16, "trade_count": 0, "win_count": 0, "loss_count": 0, "state": {"serverRuntime": {"policy_version": POLICY, "shadow_only": True, "live_execution": False, "authoritative": True}}}
    decision = {"occurrence_id": "occ", "session_id": "session", "episode_id": "episode", "symbol": "ETHUSDT", "decision_at": now.isoformat(), "due_at": (now+dt.timedelta(minutes=90)).isoformat(), "setup": "SWEEP_RECLAIM", "direction": "UP", "regime": "RANGE", "venue": "SPOT", "entry_price": 100, "target_price": 103, "invalidation_price": 99, "raw_conviction": .7, "calibrated_probability": None, "calibration_samples": 0, "policy_version": POLICY, "metric_version": "target-before-invalidation-v8.1", "evidence": {"veto": ["CALIBRATING"]}}
    event = {"transition_id": "v8-buy-occ", "event_kind": "BUY", "occurrence_id": "occ", "episode_id": "episode", "price": 100, "entry_price": 100, "quantity": .8, "notional": 80, "fees": .08, "realized_pnl": 0, "cash_after": 919.92, "equity_after": 999.84, "metadata": {"server_v8": True}}
    snapshot["state"]["v8"] = rt
    return ["session", 0, "owner", "commit-0", rt, snapshot, decision, [event]]


def commit(packet):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("set local role service_role")
        cur.execute("select public.brian_dip_v8_commit(%s,%s,%s,%s,%s,%s,%s,%s)", tuple(Json(x) if i>=4 else x for i,x in enumerate(packet)))
        return cur.fetchone()[0]


def test_session_start_and_runtime_are_atomic_and_hard_shadow():
    assert rows("select state_version,runtime->>'cash' from public.brian_dip_v8_runtime") == [(0, "1000")]
    assert rows("select count(*) from public.brian_dip_v8_ledger") == [(1,)]
    with pytest.raises(psycopg.Error, match="INVALID_V8_SHADOW_SESSION"):
        rows("insert into public.brian_dip_session_events(event_id,session_id,event_kind,starting_equity,config) values('bad','bad','START',1000,%s)", (Json({**CONFIG,"live_execution":True}),))


def test_duplicate_retry_commits_exactly_one_buy_and_one_decision():
    p=buy_packet()
    assert commit(p)["status"] == "COMMITTED"
    assert commit(p)["status"] == "ALREADY_COMMITTED"
    assert rows("select count(*) from public.brian_dip_v8_ledger where event_kind='BUY'") == [(1,)]
    assert rows("select count(*) from public.brian_dip_v8_decisions") == [(1,)]
    p[4]["cash"]=1000
    with pytest.raises(psycopg.Error, match="IDEMPOTENCY_PAYLOAD_CONFLICT"): commit(p)


def test_two_overlapping_workers_cannot_both_change_the_state():
    a=buy_packet();b=copy.deepcopy(a);b[3]="other-commit"
    def attempt(p):
        try:return commit(p)["status"]
        except psycopg.Error as e:return str(e)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(attempt,[a,b]))
    assert results.count("COMMITTED")==1
    assert sum("STATE_VERSION_CONFLICT" in r for r in results)==1
    assert rows("select state_version from public.brian_dip_v8_runtime") == [(1,)]


def test_snapshot_failure_rolls_back_event_decision_and_cash_then_retry_succeeds():
    p=buy_packet();p[5]["snapshot_id"]=None
    with pytest.raises(psycopg.Error): commit(p)
    assert rows("select runtime->>'cash',state_version from public.brian_dip_v8_runtime") == [("1000",0)]
    assert rows("select count(*) from public.brian_dip_v8_decisions") == [(0,)]
    assert rows("select count(*) from public.brian_dip_events") == [(0,)]
    assert rows("select count(*) from public.brian_dip_v8_ledger where event_kind='BUY'") == [(0,)]
    p[5]["snapshot_id"]="fixed"
    assert commit(p)["status"]=="COMMITTED"


def test_pause_and_lease_loss_are_checked_at_commit():
    p=buy_packet();p[2]="stale-owner"
    with pytest.raises(psycopg.Error, match="LEASE_LOST"): commit(p)
    p[2]="owner"
    rows("insert into public.brian_dip_session_events(event_id,session_id,event_kind,starting_equity,config) values('pause','session','PAUSE',1000,%s)",(Json(CONFIG),))
    with pytest.raises(psycopg.Error, match="ENTRY_DISABLED"): commit(p)
    assert rows("select state_version from public.brian_dip_v8_runtime") == [(0,)]


def test_restart_cannot_orphan_open_position_and_plan_cannot_drift():
    p=buy_packet();commit(p)
    with pytest.raises(psycopg.Error, match="OPEN_V8_POSITION_BLOCKS_RESTART"):
        rows("insert into public.brian_dip_session_events(event_id,session_id,event_kind,starting_equity,config) values('restart','new-session','START',1000,%s)",(Json(CONFIG),))
    p[1]=1;p[3]="next";p[6]=None;p[7]=[];p[4]["pos"]["target"]=104
    with pytest.raises(psycopg.Error, match="OPEN_PLAN_IMMUTABLE"):commit(p)


def test_calibration_inputs_and_resolved_outcome_are_immutable():
    commit(buy_packet())
    with pytest.raises(psycopg.Error, match="DECISION_INPUT_IMMUTABLE"):
        rows("update public.brian_dip_v8_decisions set target_price=104")
    rows("update public.brian_dip_v8_decisions set resolved_at=clock_timestamp(),hit=null,resolution='{\"reason\":\"AMBIGUOUS\"}'::jsonb")
    assert rows("select hit from public.brian_dip_v8_calibration('SWEEP_RECLAIM','UP','RANGE')") == [(None,)]
    with pytest.raises(psycopg.Error, match="RESOLUTION_IMMUTABLE"):
        rows("update public.brian_dip_v8_decisions set hit=true")


def test_runtime_ticks_do_not_append_full_snapshots_or_ledger_rows():
    p=buy_packet();commit(p)
    p[1]=1;p[3]="tick";p[6]=None;p[7]=[]
    assert commit(p)["status"]=="COMMITTED"
    assert rows("select count(*) from public.brian_dip_snapshots") == [(1,)]
    assert rows("select count(*) from public.brian_dip_v8_ledger") == [(2,)]
    assert rows("select state_version from public.brian_dip_v8_runtime") == [(2,)]


def test_browser_roles_cannot_mutate_v8_accounting():
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("select has_function_privilege('anon','public.brian_dip_v8_commit(text,bigint,text,text,jsonb,jsonb,jsonb,jsonb)','execute'),has_table_privilege('authenticated','public.brian_dip_v8_ledger','insert')")
        assert cur.fetchone()==(False,False)
