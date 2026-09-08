"""V8.3 regression tests against a disposable, loopback-only PostgreSQL database."""
import copy
import datetime as dt
import pytest
import test_brian_dip_v8_postgres as base
from psycopg2.extras import Json
from psycopg2.extensions import parse_dsn

rows=base.rows
POLICY='dip-v8-dual-20260908.2'
REV='dip-v8-dual-integrity-20260908.5'
CONFIG={**base.CONFIG,'engine_version':'brian-dip-chart-reader-v8-dual','policy_version':POLICY,'allow_shadow_short':True,'max_shadow_leverage':2}

@pytest.fixture(scope='module', autouse=True)
def schema():
    if not base.DSN: pytest.skip('dedicated local Postgres DSN required')
    assert parse_dsn(base.DSN).get('host') in {'127.0.0.1','localhost'}
    assert parse_dsn(base.DSN).get('dbname')=='v8_test', 'Only disposable CI database permitted'
    rows('drop schema if exists public cascade; create schema public; drop schema if exists cron cascade')
    base.schema.__wrapped__()
    # Execute migration files without an empty parameter tuple. psycopg2 treats
    # literal percent signs inside SQL as interpolation markers when args=().
    dual=(base.ROOT/'supabase/migrations/20260908175500_brian_dip_v82_dual_direction_shadow.sql').read_text()
    repair=next((base.ROOT/'supabase/migrations').glob('*_brian_dip_v83_repair.sql')).read_text()
    with base.psycopg.connect(base.DSN) as conn,conn.cursor() as cur:
        cur.execute(dual)
        cur.execute(repair)
    rows('grant usage on schema public to service_role')

@pytest.fixture(autouse=True)
def fresh(schema):
    rows('truncate public.brian_dip_v8_runtime,public.brian_dip_v8_ledger,public.brian_dip_v8_decisions,public.brian_dip_session_events,public.brian_dip_events,public.brian_dip_snapshots,public.brian_collector_leases')
    rows("insert into public.brian_dip_session_events(event_id,session_id,event_kind,starting_equity,trade_notional,config) values('start','session','START',1000,1000,%s)",(Json(CONFIG),))
    rows("insert into public.brian_collector_leases values('brian-dip-dual-shadow-worker-v82','owner',clock_timestamp()+interval '5 minutes')")

def packet(occ='occ',episode='episode',qty=.8,side='LONG'):
    p=base.buy_packet();rt=rows("select runtime,state_version from public.brian_dip_v8_runtime where session_id='session'")[0]
    p[1]=rt[1];p[3]='commit-'+occ;p[4]=copy.deepcopy(rt[0]);cash=p[4]['cash'];now=dt.datetime.now(dt.timezone.utc)
    kind='BUY' if side=='LONG' else 'SHORT_OPEN'
    stop,target=(99,110) if side=='LONG' else (101,90)
    pos={'side':side,'venue':'SHADOW_PERP','position_id':'p-'+occ,'thesis_id':occ,'episode_id':episode,'entry':100,'stop':stop,'target':target,'qty':qty,'notional':qty*100,'margin':qty*100,'leverage':1,'fees_open':qty*.1,'fee_bps':10,'slippage_bps':1,'spread_bps':.1,'actual_fraction':qty*100/cash,'gross_fraction':qty*100/cash,'opened_at':(now-dt.timedelta(seconds=10)).isoformat(),'checked_until':int(now.timestamp()*1000),'market_price':100}
    p[4].update(pos=pos,cash=cash-pos['margin']-pos['fees_open'],lastOccurrence=occ,lastSnapshotHour=now.isoformat()[:13])
    p[6].update(occurrence_id=occ,episode_id=episode,venue='SHADOW_PERP',direction='UP' if side=='LONG' else 'DOWN',invalidation_price=stop,target_price=target,policy_version=POLICY,metric_version='target-before-invalidation-v8.2')
    p[6]['evidence']['decision_revision']=REV
    p[7]=[{'transition_id':'v82-'+kind.lower()+'-'+occ,'event_kind':kind,'occurrence_id':occ,'episode_id':episode,'price':100,'entry_price':100,'quantity':qty,'notional':pos['notional'],'fees':pos['fees_open'],'realized_pnl':0,'cash_after':p[4]['cash'],'equity_after':cash-pos['fees_open'],'metadata':{'server_v8':True}}]
    sync(p);return p

def sync(p):
    r=p[4];p[5].update(cash=r['cash'],realized_pnl=r['realized'],trade_count=r['trades'],win_count=r['wins'],loss_count=r['losses'],observed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    p[5]['state']={'serverRuntime':{'policy_version':POLICY,'shadow_only':True,'live_execution':False,'dual_direction':True},'v8':r}

def commit(p):
    with base.psycopg.connect(base.DSN) as conn,conn.cursor() as cur:
        cur.execute('set local role service_role')
        cur.execute('select public.brian_dip_v82_commit(%s,%s,%s,%s,%s,%s,%s,%s)',tuple(Json(x) if i>=4 else x for i,x in enumerate(p)))
        return cur.fetchone()[0]

def close(p,funding=0):
    p=copy.deepcopy(p);pos=p[4]['pos'];p[1]+=1;p[3]+='-close';p[6]=None
    fee=pos['fees_open'];net=-2*fee+funding;p[4]['cash']+=pos['margin']-fee+funding;p[4]['realized']+=net;p[4]['trades']+=1;p[4]['wins']+=int(net>0);p[4]['losses']+=int(net<0);p[4]['pos']=None
    p[4]['lastLock']={'episode_id':pos['episode_id'],'fingerprint':'opposite-overwrite','last5m_t':0,'reason':'TEST'}
    kind='SELL' if pos['side']=='LONG' else 'SHORT_CLOSE';now=int(dt.datetime.now(dt.timezone.utc).timestamp()*1000)
    sign=-1 if pos['side']=='LONG' else 1
    evidence=[] if not funding else [{'fundingTime':now-1000,'markPrice':100,'fundingRate':funding/(sign*pos['qty']*100)}]
    p[7]=[{'transition_id':'v82-'+kind.lower()+'-'+pos['thesis_id'],'event_kind':kind,'occurrence_id':pos['thesis_id'],'episode_id':pos['episode_id'],'price':100,'entry_price':100,'exit_price':100,'quantity':pos['qty'],'notional':pos['notional'],'fees':2*fee,'funding_cashflow':funding,'realized_pnl':net,'cash_after':p[4]['cash'],'equity_after':p[4]['cash'],'metadata':{'server_v8':True,'funding_settlements':evidence,'resolution':{'eventAt':now}}}]
    sync(p);return p

def test_atomic_retry_and_rollback():
    p=packet();bad=copy.deepcopy(p);bad[5]['snapshot_id']=None
    with pytest.raises(base.psycopg.Error):commit(bad)
    assert rows('select count(*) from public.brian_dip_v8_decisions')==[(0,)]
    assert commit(p)['status']=='COMMITTED'
    assert commit(p)['status']=='ALREADY_COMMITTED'

def test_opposite_trade_does_not_erase_episode_lock():
    a=packet('long1','long-episode');commit(a);commit(close(a))
    b=packet('short1','short-episode',qty=.7,side='SHORT');commit(b);commit(close(b))
    with pytest.raises(base.psycopg.Error,match='EPISODE_ALREADY_TRADED'):commit(packet('long2','long-episode',qty=.7))
    assert rows("select count(*) from public.brian_dip_v8_decisions where occurrence_id='long2'")==[(0,)]

@pytest.mark.parametrize('side',['LONG','SHORT'])
def test_cold_notional_cap_is_enforced_in_database(side):
    with pytest.raises(base.psycopg.Error,match='NOTIONAL_CAP'):commit(packet(qty=1,side=side))
    assert rows("select runtime->>'cash' from public.brian_dip_v8_runtime")==[('1000',)]

@pytest.mark.parametrize('side,funding',[('LONG',-.2),('SHORT',.2)])
def test_pause_allows_exit_and_signed_funding_balances(side,funding):
    p=packet(side=side);commit(p)
    rows("insert into public.brian_dip_session_events(event_id,session_id,event_kind,config) values('pause','session','PAUSE','{}')")
    assert commit(close(p,funding))['status']=='COMMITTED'
    cash,realized=rows("select (runtime->>'cash')::numeric,(runtime->>'realized')::numeric from public.brian_dip_v8_runtime")[0]
    assert abs(float(cash)-1000-float(realized))<1e-8
    with pytest.raises(base.psycopg.Error,match='ENTRY_DISABLED'):commit(packet('new','new',qty=.7))

def test_old_revision_cannot_enter_and_calibration_does_not_mix():
    p=packet();p[6]['evidence']['decision_revision']='old'
    with pytest.raises(base.psycopg.Error,match='REVISION_MISMATCH'):commit(p)
    p=packet();commit(p)
    rows("update public.brian_dip_v8_decisions set resolved_at=clock_timestamp(),hit=false,resolution=%s",(Json({'resolver_version':'old-resolver'}),))
    assert rows("select * from public.brian_dip_v82_calibration('SWEEP_RECLAIM','UP','RANGE')")==[]

def test_entry_target_must_match_frozen_decision():
    p=packet();p[4]['pos']['target']=115;sync(p)
    with pytest.raises(base.psycopg.Error,match='PLAN_OR_REVISION'):commit(p)

def test_lease_loss_blocks_commit_and_anon_cannot_execute():
    p=packet();p[2]='stale'
    with pytest.raises(base.psycopg.Error,match='LEASE_LOST'):commit(p)
    assert rows("select has_function_privilege('anon','public.brian_dip_v82_commit(text,bigint,text,text,jsonb,jsonb,jsonb,jsonb)','execute')")==[(False,)]
