"""Real PostgreSQL checks for new evidence storage, bounded inputs and net protocol."""
from pathlib import Path
import re
import pytest
from test_brian_outcome_coverage_postgres import db  # isolated fixture; always rolled back

ROOT=Path(__file__).resolve().parents[1]
MIGRATION=next((ROOT/'supabase/migrations').glob('*_world_evidence_reviews.sql')).read_text()
SQL=re.search(r'NET_VALIDATION_SQL=`([\s\S]*?)`;', (ROOT/'supabase/functions/brian-shadow-scorecard/advantage.ts').read_text()).group(1)
START='2026-10-19T16:00:00Z'
END='2026-11-18T16:00:00Z'

def result(db,at='2026-10-19T17:00:00Z',sample='prospective'):
    r=db(SQL.replace('$1','%s').replace('$2','%s').replace('$3','%s'),(at,START,END))
    return next(x for x in r['rows'] if x['sample']==sample)

def decision(db,id,at,action='OPEN_LONG',asset='crypto:BTCUSDT'):
    db("insert into public.brian_alpha_decisions(decision_id,asset_id,observed_at,action) values(%s,%s,%s,%s)",(id,asset,at,action))
    db("update public.brian_alpha_decisions set created_at=observed_at where decision_id=%s",(id,))

def outcome(db,id,at,cost='20',ret=.01):
    db("insert into public.brian_alpha_decision_outcomes(outcome_id,decision_id,horizon_seconds,resolved_at,created_at,metadata,direction_adjusted_return) values(%s,%s,900,%s,%s,%s::jsonb,%s)",(id,id,at,at,'{"estimated_round_trip_cost_bps":'+cost+'}',ret))

def test_fixed_first_decision_selection_and_cash_denominator(db):
    decision(db,'first',START,'WAIT')
    decision(db,'second','2026-10-19T16:00:30Z')
    decision(db,'outside','2026-10-19T16:01:00Z',asset='crypto:ETHUSDT')
    decision(db,'next','2026-10-19T16:18:00Z')
    outcome(db,'next','2026-10-19T16:33:00Z')
    r=result(db)
    assert r['selected']==2 and r['cash_decisions']==1 and r['directional']==1
    assert r['measured']==1 and r['mean_net_bps']==pytest.approx(80)
    assert len(r['day_blocks'])==1

@pytest.mark.parametrize('cost,missing',[('20',0),('null',1),('"NaN"',1),('"bad"',1),('-1',1),('"1e999"',1)])
def test_cost_unknown_is_never_zero(db,cost,missing):
    decision(db,'a',START)
    outcome(db,'a','2026-10-19T16:15:00Z',cost)
    r=result(db)
    assert r['missing_cost']==missing and r['measured']==1-missing
    assert (r['mean_net_bps'] is None)==bool(missing)

def test_terminal_tolerance_reference_and_point_in_time(db):
    for id,asset in [('early','BTC'),('late','ETH'),('ref','XRP'),('future','SOL'),('young','ADA')]:
        decision(db,id,'2026-10-19T16:54:00Z' if id=='young' else START,asset='crypto:'+asset+'USDT')
    outcome(db,'early','2026-10-19T16:14:59Z')
    outcome(db,'late','2026-10-19T16:17:01Z')
    outcome(db,'ref','2026-10-19T16:15:00Z')
    db("update public.brian_alpha_decision_outcomes set reference_price=99 where decision_id='ref'")
    outcome(db,'future','2026-10-19T16:15:00Z')
    db("update public.brian_alpha_decision_outcomes set created_at='2026-10-19T18:00:00Z' where decision_id='future'")
    r=result(db)
    assert r['invalid_outcome']==3 and r['unresolved']==1 and r['not_mature']==1
    assert r['measured']==0

def test_mode_and_future_decisions_excluded_and_empty_is_not_measured(db):
    decision(db,'live',START)
    db("update public.brian_alpha_decisions set live_execution=true where decision_id='live'")
    decision(db,'future',START)
    db("update public.brian_alpha_decisions set created_at='2026-10-19T18:00:00Z' where decision_id='future'")
    assert result(db)['selected']==0
    assert result(db)['mean_net_bps'] is None
    assert result(db,at='2026-10-19T15:00:00Z')['selected']==0

def install(db):
    db('''create table public.brian_intel_events(event_id text primary key,source_id text,claim text,provenance_uri text,published_at timestamptz,first_observed_at timestamptz,created_at timestamptz,trust_class text);
    create table public.brian_source_endpoints_v2(source_id text,canonical_domain text,tier text,lifecycle_state text,access_mode text);
    create function public.brian_reject_mutation() returns trigger language plpgsql as $$begin raise exception 'append only'; end$$;''')
    # Roles already exist in CI in many suites, create only if absent; transactional.
    db("do $$begin if not exists(select from pg_roles where rolname='anon') then create role anon; end if; if not exists(select from pg_roles where rolname='authenticated') then create role authenticated; end if; if not exists(select from pg_roles where rolname='service_role') then create role service_role; end if; end$$")
    # Preserve exact production SQL, replacing only fixture schema/search_path.
    schema=db('select current_schema()')
    # Fixture replaces qualified public. references automatically. Read the actual schema via table OID.
    ns=db("select n.nspname from pg_class c join pg_namespace n on n.oid=c.relnamespace where c.oid='public.brian_intel_events'::regclass")
    db(MIGRATION.replace('search_path=pg_catalog,public','search_path=pg_catalog,'+ns).replace('%','%%'))

def test_migration_inputs_append_only_modes_and_grants(db):
    install(db)
    db("insert into public.brian_source_endpoints_v2 values('official:test','example.org','T1_OFFICIAL_PRIMARY','ACTIVE','PUBLIC_NO_KEY')")
    db("insert into public.brian_intel_events values('e','official:test','Central bank interest rates increase','https://example.org/official',now()-interval '10 minutes',now()-interval '9 minutes',now()-interval '9 minutes','OFFICIAL_PRIMARY')")
    r=db('select public.brian_world_research_inputs_v1()')
    assert len(r['events'])==1 and r['events'][0]['event_id']=='e'
    assert len(r['events'][0]['probes'])==2
    db("insert into public.brian_world_evidence_reviews(review_id,event_id,assessed_at,version,status,review) values('r','e',now(),'v1','READ','{}')")
    assert db('select public.brian_world_research_inputs_v1()')['events']==[]
    assert db("select relrowsecurity from pg_class where oid='public.brian_world_evidence_reviews'::regclass") is True
    assert db("select has_table_privilege('anon','public.brian_world_evidence_reviews','SELECT')") is False
    assert db("select has_table_privilege('authenticated','public.brian_world_evidence_reviews','INSERT')") is False
    assert db("select has_table_privilege('service_role','public.brian_world_evidence_reviews','INSERT')") is True
    assert db("select has_function_privilege('anon','public.brian_world_research_inputs_v1()','EXECUTE')") is False
    db('savepoint bad_write')
    with pytest.raises(Exception,match='append only'):
        db("update public.brian_world_evidence_reviews set status='FAILED' where review_id='r'")
    db('rollback to savepoint bad_write')
    with pytest.raises(Exception):
        db("insert into public.brian_world_evidence_reviews(review_id,event_id,assessed_at,version,status,review,live_execution) values('bad','e',now(),'v1','READ','{}',true)")


def test_late_decisions_and_unknown_policy_are_explicit(db):
    decision(db,'late',START)
    outcome(db,'late','2026-10-19T16:15:00Z')
    db("update public.brian_alpha_decisions set created_at=observed_at+interval '61 seconds' where decision_id='late'")
    decision(db,'unknown',START,asset='crypto:ETHUSDT')
    db("update public.brian_alpha_decisions set compiler_version=null where decision_id='unknown'")
    r=result(db)
    assert r['late_decision']==1 and r['unknown_policy']==1 and r['measured']==0


def test_repaired_inputs_bound_scan_preserve_fairness_and_revisit_sources(db):
    install(db)
    repair=next((ROOT/'supabase/migrations').glob('*_repair_world_research_inputs.sql')).read_text()
    ns=db("select n.nspname from pg_class c join pg_namespace n on n.oid=c.relnamespace where c.oid='public.brian_intel_events'::regclass")
    db(repair.replace('search_path=pg_catalog,public','search_path=pg_catalog,'+ns).replace('%','%%'))
    db("insert into public.brian_source_endpoints_v2 values('official:a','example.org','T1_OFFICIAL_PRIMARY','ACTIVE','PUBLIC_NO_KEY'),('official:b','example.net','T1_OFFICIAL_PRIMARY','ACTIVE','PUBLIC_NO_KEY')")
    db("insert into public.brian_intel_events select 'e'||n,'official:a','Central bank rates increase','https://example.org/a',now()-interval '10 minutes',now()-interval '9 minutes',now()-interval '9 minutes','OFFICIAL_PRIMARY' from generate_series(1,2100) n")
    db("insert into public.brian_intel_events values('b','official:b','Central bank rates decrease','https://example.net/b',now()-interval '8 minutes',now()-interval '8 minutes',now()-interval '8 minutes','OFFICIAL_PRIMARY')")
    r=db('select public.brian_world_research_inputs_v1()')
    assert len(r['events'])==2
    assert {e['source_id'] for e in r['events']}=={'official:a','official:b'}
    for e in r['events']:
        assert e['counter_pool_scan_limit']==2000 and e['counter_pool_scan_truncated'] is True
        assert len(e['counter_pool'])<=100 and len(e['probes'])==2
    db("insert into public.brian_world_evidence_reviews(review_id,event_id,assessed_at,version,status,review) values('done','b',now(),'v1','READ','{}')")
    r=db('select public.brian_world_research_inputs_v1()')
    assert len(r['events'])==1 and r['events'][0]['source_id']=='official:a'


def install_domain_cooldown(db):
    install(db)
    migration=next((ROOT/'supabase/migrations').glob('*_world_research_domain_cooldown.sql')).read_text()
    ns=db("select n.nspname from pg_class c join pg_namespace n on n.oid=c.relnamespace where c.oid='public.brian_intel_events'::regclass")
    db(migration.replace('search_path=pg_catalog,public','search_path=pg_catalog,'+ns).replace('%','%%'))
    db("insert into public.brian_source_endpoints_v2 values('official:a','example.org','T1_OFFICIAL_PRIMARY','ACTIVE','PUBLIC_NO_KEY'),('official:b','example.org','T1_OFFICIAL_PRIMARY','ACTIVE','PUBLIC_NO_KEY'),('official:c','example.net','T1_OFFICIAL_PRIMARY','ACTIVE','PUBLIC_NO_KEY')")
    for ident,source,age in [('a','official:a',20),('b','official:b',19),('c','official:c',18)]:
        db("insert into public.brian_intel_events values(%s,%s,'Central bank rates increase','https://example.org/a',now()-make_interval(mins=>%s),now()-make_interval(mins=>%s),now()-make_interval(mins=>%s),'OFFICIAL_PRIMARY')",(ident,source,age,age,age))


def access_review(db,ident,event,status,error,age):
    import json
    db("insert into public.brian_world_evidence_reviews(review_id,event_id,assessed_at,captured_at,version,status,review) values(%s,%s,now()-make_interval(mins=>%s),now()-make_interval(mins=>%s),'test',%s,%s::jsonb)",(ident,event,age,age,status,json.dumps({'error':error})))


def test_domain_access_cooldown_preserves_other_sources_and_is_explicit(db):
    install_domain_cooldown(db)
    access_review(db,'blocked','a','FAILED','OFFICIAL_HTTP_403',1)
    r=db('select public.brian_world_research_inputs_v1()')
    assert {e['source_id'] for e in r['events']}=={'official:c'}
    health=r['source_access']
    assert health['status']=='PARTIAL_ACCESS'
    assert {s['source_id'] for s in health['cooldown_sources']}=={'official:a','official:b'}
    assert all(s['next_retry_at'] is not None for s in health['cooldown_sources'])
    assert db("select has_function_privilege('anon','public.brian_world_research_inputs_v1()','EXECUTE')") is False


@pytest.mark.parametrize('error',['OFFICIAL_HTTP_429','OFFICIAL_REDIRECT_REJECTED','OFFICIAL_DOCUMENT_NOT_READABLE'])
def test_access_errors_cool_down_the_official_domain(db,error):
    install_domain_cooldown(db)
    access_review(db,'blocked','a','FAILED',error,1)
    assert db('select public.brian_world_research_inputs_v1()')['source_access']['status']=='PARTIAL_ACCESS'


def test_cooldown_expires_without_manual_reset_and_transient_failure_does_not_block_host(db):
    install_domain_cooldown(db)
    access_review(db,'expired','a','FAILED','OFFICIAL_HTTP_403',16)
    r=db('select public.brian_world_research_inputs_v1()')
    assert r['source_access']['cooldown_sources']==[]
    assert 'official:b' in {e['source_id'] for e in r['events']}
    access_review(db,'transient','b','FAILED','OFFICIAL_HTTP_503',1)
    assert db('select public.brian_world_research_inputs_v1()')['source_access']['cooldown_sources']==[]


def test_readable_body_resets_domain_cooldown_and_future_receipt_is_not_used(db):
    install_domain_cooldown(db)
    access_review(db,'blocked','a','FAILED','OFFICIAL_HTTP_403',10)
    access_review(db,'recovered','b','READ',None,5)
    assert db('select public.brian_world_research_inputs_v1()')['source_access']['cooldown_sources']==[]
    access_review(db,'future','a','FAILED','OFFICIAL_HTTP_403',-5)
    assert db('select public.brian_world_research_inputs_v1()')['source_access']['cooldown_sources']==[]
