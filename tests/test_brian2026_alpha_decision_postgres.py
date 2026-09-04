"""Real-Postgres invariants for Brian ALPHA v2 decision/cost/outcome/shadow persistence."""
from __future__ import annotations
import os, uuid
from pathlib import Path
import pytest
psycopg2=pytest.importorskip("psycopg2",reason="psycopg2 only installed in Postgres CI")
ROOT=Path(__file__).resolve().parents[1]
MIGRATION=ROOT/"supabase"/"migrations"/"202609040015_brian_alpha_decision_compiler.sql"
STATE_MIGRATION=ROOT/"supabase"/"migrations"/"202609040016_brian_alpha_shadow_state.sql"
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
            create schema if not exists brian_private;
            create or replace function public.brian_reject_mutation() returns trigger language plpgsql as $$begin raise exception 'append only'; end$$;
            create table if not exists public.brian_live_shadow_ticks(
              tick_id text primary key,
              experiment_id text not null,
              policy_kind text not null,
              observed_at timestamptz not null,
              target_weights jsonb not null default '{}'::jsonb
            );
            """)
            cur.execute(MIGRATION.read_text(encoding="utf-8"))
            cur.execute(STATE_MIGRATION.read_text(encoding="utf-8"))
    finally:c.close()

def _valid_cost(cur, asset="crypto:BTCUSDT", observed_at="2026-09-04T12:00:00Z", side="BUY"):
    q=f"q-{uuid.uuid4().hex}"
    cur.execute("""
      insert into public.brian_dynamic_cost_quotes(
        quote_id,compiler_version,asset_id,observed_at,side,requested_notional_usd,
        filled_notional_usd,fill_ratio,fillable,fee_bps,spread_bps,depth_slippage_bps,
        one_way_cost_bps,estimated_round_trip_cost_bps,quality,reason,shadow_only,live_execution)
      values(%s,'test-v2',%s,%s,%s,100,100,1,true,10,2,1,12,24,'L2_OBSERVED','fixture',true,false)
    """,(q,asset,observed_at,side))
    return q

def _insert_decision(cur,q,asset="crypto:BTCUSDT",observed_at="2026-09-04T12:00:00Z",action="OPEN_LONG",direction=1,reference_price=100):
    d=f"d-{uuid.uuid4().hex}"
    cur.execute("""
      insert into public.brian_alpha_decisions(
        decision_id,compiler_version,observed_at,asset_id,observed_reference_price,action,direction,evidence_score,
        independent_group_count,support_groups,conflict_groups,source_observation_ids,
        source_intrabar_event_ids,source_cost_quote_id,requested_virtual_notional_usd,
        estimated_round_trip_cost_bps,reason,shadow_only,live_execution)
      values(%s,'test-v2',%s,%s,%s,%s,%s,0.3,2,
        array['micro_velocity','derivatives_taker'],array[]::text[],array[]::text[],array[]::text[],%s,100,24,'fixture',true,false)
    """,(d,observed_at,asset,reference_price,action,direction,q))
    return d

def _valid_decision(cur,q):
    return _insert_decision(cur,q)

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
            cur.execute("select shadow_only,live_execution,observed_reference_price from public.brian_alpha_decisions where decision_id=%s",(d,));assert cur.fetchone()==(True,False,100)
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
            with pytest.raises(psycopg2.Error):cur.execute("update public.brian_alpha_shadow_position_events set reference_price=1 where decision_id=%s",(d,))
            with pytest.raises(psycopg2.Error):cur.execute("delete from public.brian_alpha_phase37_comparisons where decision_id=%s",(d,))
            cur.execute("reset role")
    finally:c.close()

def test_shadow_position_book_reaffirms_then_flips_without_rebasing_entry():
    c=_connect()
    asset=f"crypto:STATE{uuid.uuid4().hex[:6].upper()}USDT"
    try:
        with c.cursor() as cur:
            cur.execute("set role service_role")
            q1=_valid_cost(cur,asset,"2026-09-04T14:00:00Z","BUY")
            d1=_insert_decision(cur,q1,asset,"2026-09-04T14:00:00Z","OPEN_LONG",1,100)
            q2=_valid_cost(cur,asset,"2026-09-04T14:01:00Z","BUY")
            d2=_insert_decision(cur,q2,asset,"2026-09-04T14:01:00Z","OPEN_LONG",1,102)
            q3=_valid_cost(cur,asset,"2026-09-04T14:02:00Z","SELL")
            d3=_insert_decision(cur,q3,asset,"2026-09-04T14:02:00Z","OPEN_SHORT",-1,99)
            cur.execute("""select decision_id,event_type,position_before,position_after,entry_price_after,realized_gross_bps
                           from public.brian_alpha_shadow_position_events where asset_id=%s order by event_ts""",(asset,))
            rows=cur.fetchall()
            assert rows[0][:5]==(d1,"OPEN",0,1,100)
            assert rows[1][:5]==(d2,"REAFFIRM",1,1,100)
            assert rows[1][5] is None
            assert rows[2][:5]==(d3,"FLIP",1,-1,99)
            assert abs(rows[2][5]-(-100.0)) < 1e-9
            cur.execute("select position,entry_price,last_event_type from public.brian_alpha_shadow_position_book where asset_id=%s",(asset,))
            assert cur.fetchone()==(-1,99,"FLIP")
            cur.execute("reset role")
    finally:c.close()

def test_phase37_comparison_is_at_or_before_only_and_tracks_both_policies():
    c=_connect()
    asset=f"crypto:CMP{uuid.uuid4().hex[:6].upper()}USDT"
    symbol=asset.split(":",1)[1]
    exp="phase37-prospective-live-20260903"
    try:
        with c.cursor() as cur:
            cur.execute("""insert into public.brian_live_shadow_ticks(tick_id,experiment_id,policy_kind,observed_at,target_weights)
                           values(%s,%s,'NATIVE','2026-09-04T15:59:00Z',%s::jsonb),
                                 (%s,%s,'NATIVE','2026-09-04T16:01:00Z',%s::jsonb),
                                 (%s,%s,'PROFIT','2026-09-04T15:58:00Z',%s::jsonb),
                                 (%s,%s,'PROFIT','2026-09-04T16:01:00Z',%s::jsonb)""",
                        (f"nprev-{uuid.uuid4().hex}",exp,f'{{"{symbol}":0.2}}',
                         f"nfut-{uuid.uuid4().hex}",exp,f'{{"{symbol}":-0.3}}',
                         f"pprev-{uuid.uuid4().hex}",exp,f'{{"{symbol}":-0.1}}',
                         f"pfut-{uuid.uuid4().hex}",exp,f'{{"{symbol}":0.3}}'))
            cur.execute("set role service_role")
            q=_valid_cost(cur,asset,"2026-09-04T16:00:00Z","BUY")
            d=_insert_decision(cur,q,asset,"2026-09-04T16:00:00Z","OPEN_LONG",1,100)
            cur.execute("""select phase37_policy_kind,phase37_observed_at,phase37_direction,relationship
                           from public.brian_alpha_phase37_comparisons where decision_id=%s order by phase37_policy_kind""",(d,))
            rows=cur.fetchall()
            assert len(rows)==2
            native=next(r for r in rows if r[0]=="NATIVE")
            profit=next(r for r in rows if r[0]=="PROFIT")
            assert native[1].isoformat().startswith("2026-09-04T15:59:00")
            assert native[2:]==(1,"AGREES")
            assert profit[1].isoformat().startswith("2026-09-04T15:58:00")
            assert profit[2:]==(-1,"DISAGREES")
            cur.execute("reset role")
    finally:c.close()
