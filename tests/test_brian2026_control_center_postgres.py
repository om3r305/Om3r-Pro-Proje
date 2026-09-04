"""Real-Postgres invariants for the SHADOW-only Brian Control Center."""
from __future__ import annotations
import json, os, uuid
from pathlib import Path
import pytest
psycopg2=pytest.importorskip("psycopg2",reason="psycopg2 only installed in Postgres CI")
ROOT=Path(__file__).resolve().parents[1]
MIGRATION=ROOT/"supabase"/"migrations"/"202609040013_brian_control_center.sql"
DATABASE_URL=os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DATABASE_URL,reason="BRIAN_TEST_DATABASE_URL not set")
SOURCE="phase37-prospective-live-20260903"

def _connect():
    c=psycopg2.connect(DATABASE_URL);c.autocommit=True;return c

@pytest.fixture(scope="module",autouse=True)
def _setup():
    c=_connect()
    try:
        with c.cursor() as cur:
            cur.execute("""
            do $$ begin
              if not exists(select 1 from pg_roles where rolname='anon') then create role anon nologin; end if;
              if not exists(select 1 from pg_roles where rolname='authenticated') then create role authenticated nologin; end if;
              if not exists(select 1 from pg_roles where rolname='service_role') then create role service_role nologin; end if;
            end $$;
            alter role service_role bypassrls;
            create schema if not exists brian_private;
            create or replace function public.brian_reject_mutation() returns trigger language plpgsql as $$begin raise exception 'append only'; end$$;
            create table if not exists public.brian_live_shadow_experiments(experiment_id text primary key);
            insert into public.brian_live_shadow_experiments(experiment_id) values ('phase37-prospective-live-20260903') on conflict do nothing;
            """)
            cur.execute(MIGRATION.read_text(encoding="utf-8"))
    finally:c.close()

def _start(c,equity=500,policy="BOTH"):
    session=f"pytest-{uuid.uuid4().hex}"
    with c.cursor() as cur:
        cur.execute("set role service_role")
        cur.execute("select (public.brian_dashboard_start_session(%s,%s,%s,%s,%s)).session_id",(f"evt-{uuid.uuid4().hex}",session,equity,policy,SOURCE))
        assert cur.fetchone()==(session,)
        cur.execute("reset role")
    return session

def _pause(c,session):
    with c.cursor() as cur:
        cur.execute("set role service_role")
        cur.execute("select (public.brian_dashboard_pause_session(%s,%s)).event_kind",(f"evt-{uuid.uuid4().hex}",session))
        assert cur.fetchone()==("PAUSE",)
        cur.execute("reset role")

def test_custom_shadow_balances_and_hard_live_lock():
    c=_connect()
    try:
        for equity in (500,1000,2750.50):
            s=_start(c,equity)
            with c.cursor() as cur:
                cur.execute("select starting_equity,shadow_only,live_execution from public.brian_dashboard_session_events where session_id=%s and event_kind='START'",(s,))
                row=cur.fetchone();assert float(row[0])==equity;assert row[1:] == (True,False)
            _pause(c,s)
        with pytest.raises(psycopg2.Error): _start(c,0)
        with c.cursor() as cur:
            with pytest.raises(psycopg2.Error):
                cur.execute("insert into public.brian_dashboard_session_events(event_id,session_id,event_kind,starting_equity,policy_scope,source_experiment_id,shadow_only,live_execution) values(%s,%s,'START',500,'BOTH',%s,true,true)",(f"evt-{uuid.uuid4().hex}",f"bad-{uuid.uuid4().hex}",SOURCE))
    finally:c.close()

def test_double_start_rejected_and_restart_is_atomic_history():
    c=_connect()
    try:
        old=_start(c,500,"PROFIT")
        with pytest.raises(psycopg2.Error): _start(c,1000)
        new=f"pytest-restart-{uuid.uuid4().hex}"
        with c.cursor() as cur:
            cur.execute("set role service_role")
            cur.execute("select (public.brian_dashboard_restart_session(%s,%s,%s,%s,%s,%s)).session_id",(f"evt-p-{uuid.uuid4().hex}",f"evt-s-{uuid.uuid4().hex}",new,1234.56,"BOTH",SOURCE))
            assert cur.fetchone()==(new,)
            cur.execute("reset role")
            cur.execute("select event_kind from public.brian_dashboard_session_events where session_id=%s order by requested_at,event_id",(old,));assert [r[0] for r in cur.fetchall()]==["START","PAUSE"]
            cur.execute("select starting_equity,policy_scope,shadow_only,live_execution from public.brian_dashboard_session_events where session_id=%s and event_kind='START'",(new,));row=cur.fetchone();assert float(row[0])==1234.56;assert row[1:]==("BOTH",True,False)
        _pause(c,new)
    finally:c.close()

def test_service_role_append_only_for_events_and_reports():
    c=_connect()
    try:
        s=_start(c,500);_pause(c,s)
        with c.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):cur.execute("update public.brian_dashboard_session_events set note='x' where session_id=%s",(s,))
            with pytest.raises(psycopg2.Error):cur.execute("delete from public.brian_dashboard_session_events where session_id=%s",(s,))
            report=f"report-{uuid.uuid4().hex}"
            cur.execute("insert into public.brian_dashboard_hourly_reports(report_id,session_id,policy_kind,window_start,window_end,payload,shadow_only,live_execution) values(%s,%s,'PROFIT','2026-09-04T10:00:00Z','2026-09-04T11:00:00Z',%s::jsonb,true,false)",(report,s,json.dumps({"shadow_only":True,"live_execution":False})))
            with pytest.raises(psycopg2.Error):cur.execute("delete from public.brian_dashboard_hourly_reports where report_id=%s",(report,))
            cur.execute("reset role")
    finally:c.close()
