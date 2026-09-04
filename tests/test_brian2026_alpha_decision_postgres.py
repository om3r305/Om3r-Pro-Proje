"""Real-Postgres invariants for Brian ALPHA v2 decision/cost/outcome persistence."""
from __future__ import annotations
import os, uuid
from pathlib import Path
import pytest
psycopg2=pytest.importorskip("psycopg2",reason="psycopg2 only installed in Postgres CI")
ROOT=Path(__file__).resolve().parents[1]
MIGRATION=ROOT/"supabase"/"migrations"/"202609040015_brian_alpha_decision_compiler.sql"
DATABASE_URL=os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark=pytest.mark.skipif(not DATABASE_URL,reason="BRIAN_TEST_DATABASE_URL not set")

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
            create or replace function public.brian_reject_mutation() returns trigger language plpgsql as $$begin raise exception 'append only'; end$$;
            """)
            cur.execute(MIGRATION.read_text(encoding="utf-8"))
    finally:c.close()

def _valid_cost(cur):
    q=f"q-{uuid.uuid4().hex}"
    cur.execute("""
      insert into public.brian_dynamic_cost_quotes(
        quote_id,compiler_version,asset_id,observed_at,side,requested_notional_usd,
        filled_notional_usd,fill_ratio,fillable,fee_bps,spread_bps,depth_slippage_bps,
        one_way_cost_bps,estimated_round_trip_cost_bps,quality,reason,shadow_only,live_execution)
      values(%s,'test-v2','crypto:BTCUSDT','2026-09-04T12:00:00Z','BUY',100,100,1,true,10,2,1,12,24,'L2_OBSERVED','fixture',true,false)
    """,(q,))
    return q

def _valid_decision(cur,q):
    d=f"d-{uuid.uuid4().hex}"
    cur.execute("""
      insert into public.brian_alpha_decisions(
        decision_id,compiler_version,observed_at,asset_id,action,direction,evidence_score,
        independent_group_count,support_groups,conflict_groups,source_observation_ids,
        source_intrabar_event_ids,source_cost_quote_id,requested_virtual_notional_usd,
        estimated_round_trip_cost_bps,reason,shadow_only,live_execution)
      values(%s,'test-v2','2026-09-04T12:00:00Z','crypto:BTCUSDT','OPEN_LONG',1,0.3,2,
        array['micro_velocity','derivatives_taker'],array[]::text[],array[]::text[],array[]::text[],%s,100,24,'fixture',true,false)
    """,(d,q))
    return d

def test_valid_shadow_chain_and_hard_live_lock():
    c=_connect()
    try:
        with c.cursor() as cur:
            cur.execute("set role service_role")
            q=_valid_cost(cur);d=_valid_decision(cur,q)
            o=f"o-{uuid.uuid4().hex}"
            cur.execute("""
              insert into public.brian_alpha_decision_outcomes(
                outcome_id,decision_id,asset_id,horizon_seconds,observed_at,resolved_at,
                reference_price,resolved_price,gross_return,direction_adjusted_return,
                classification,explanation,shadow_only,live_execution)
              values(%s,%s,'crypto:BTCUSDT',300,'2026-09-04T12:00:00Z','2026-09-04T12:05:00Z',100,101,0.01,0.01,'ACTION_FAVORABLE_AFTER_COST','fixture',true,false)
            """,(o,d))
            cur.execute("select shadow_only,live_execution from public.brian_alpha_decisions where decision_id=%s",(d,));assert cur.fetchone()==(True,False)
            with pytest.raises(psycopg2.Error):
                cur.execute("""insert into public.brian_dynamic_cost_quotes(quote_id,compiler_version,asset_id,observed_at,side,requested_notional_usd,filled_notional_usd,fill_ratio,fillable,fee_bps,spread_bps,depth_slippage_bps,one_way_cost_bps,estimated_round_trip_cost_bps,quality,reason,shadow_only,live_execution) values(%s,'x','crypto:XRPUSDT',now(),'BUY',10,10,1,true,1,1,1,3,6,'L2_OBSERVED','bad',true,true)""",(f"bad-{uuid.uuid4().hex}",))
            cur.execute("reset role")
    finally:c.close()

def test_invalid_action_direction_and_cost_quality_rejected():
    c=_connect()
    try:
        with c.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute("""insert into public.brian_alpha_decisions(decision_id,compiler_version,observed_at,asset_id,action,direction,evidence_score,independent_group_count,requested_virtual_notional_usd,reason,shadow_only,live_execution) values(%s,'x',now(),'crypto:ETHUSDT','OPEN_LONG',-1,0.3,2,10,'bad',true,false)""",(f"bad-{uuid.uuid4().hex}",))
            with pytest.raises(psycopg2.Error):
                cur.execute("""insert into public.brian_dynamic_cost_quotes(quote_id,compiler_version,asset_id,observed_at,side,requested_notional_usd,filled_notional_usd,fill_ratio,fillable,fee_bps,spread_bps,depth_slippage_bps,one_way_cost_bps,estimated_round_trip_cost_bps,quality,reason,shadow_only,live_execution) values(%s,'x','crypto:ETHUSDT',now(),'BUY',10,10,1,true,1,1,1,3,6,'MAGIC_COST','bad',true,false)""",(f"badq-{uuid.uuid4().hex}",))
            cur.execute("reset role")
    finally:c.close()

def test_outcome_unique_horizon_and_time_order():
    c=_connect()
    try:
        with c.cursor() as cur:
            cur.execute("set role service_role");q=_valid_cost(cur);d=_valid_decision(cur,q)
            cur.execute("""insert into public.brian_alpha_decision_outcomes(outcome_id,decision_id,asset_id,horizon_seconds,observed_at,resolved_at,reference_price,resolved_price,gross_return,direction_adjusted_return,classification,explanation,shadow_only,live_execution) values(%s,%s,'crypto:BTCUSDT',900,'2026-09-04T12:00:00Z','2026-09-04T12:15:00Z',100,99,-0.01,-0.01,'ACTION_UNFAVORABLE_AFTER_COST','fixture',true,false)""",(f"o-{uuid.uuid4().hex}",d))
            with pytest.raises(psycopg2.Error):
                cur.execute("""insert into public.brian_alpha_decision_outcomes(outcome_id,decision_id,asset_id,horizon_seconds,observed_at,resolved_at,reference_price,resolved_price,gross_return,direction_adjusted_return,classification,explanation,shadow_only,live_execution) values(%s,%s,'crypto:BTCUSDT',900,'2026-09-04T12:00:00Z','2026-09-04T12:15:00Z',100,99,-0.01,-0.01,'duplicate','bad',true,false)""",(f"dup-{uuid.uuid4().hex}",d))
            with pytest.raises(psycopg2.Error):
                cur.execute("""insert into public.brian_alpha_decision_outcomes(outcome_id,decision_id,asset_id,horizon_seconds,observed_at,resolved_at,reference_price,resolved_price,gross_return,direction_adjusted_return,classification,explanation,shadow_only,live_execution) values(%s,%s,'crypto:BTCUSDT',3600,'2026-09-04T13:00:00Z','2026-09-04T12:59:00Z',100,99,-0.01,-0.01,'bad','bad',true,false)""",(f"time-{uuid.uuid4().hex}",d))
            cur.execute("reset role")
    finally:c.close()

def test_service_role_tables_are_append_only():
    c=_connect()
    try:
        with c.cursor() as cur:
            cur.execute("set role service_role");q=_valid_cost(cur);d=_valid_decision(cur,q)
            with pytest.raises(psycopg2.Error):cur.execute("update public.brian_dynamic_cost_quotes set reason='mutated' where quote_id=%s",(q,))
            with pytest.raises(psycopg2.Error):cur.execute("delete from public.brian_alpha_decisions where decision_id=%s",(d,))
            cur.execute("reset role")
    finally:c.close()
