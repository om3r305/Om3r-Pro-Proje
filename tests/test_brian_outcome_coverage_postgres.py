"""Run the exact production SELECT against isolated PostgreSQL fixture tables."""
import os
from pathlib import Path
import re
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
SQL = re.search(r'OUTCOME_COVERAGE_SQL = `([\s\S]*?)`;', (ROOT / 'supabase/functions/brian-shadow-scorecard/coverage.ts').read_text()).group(1)
AT = '2026-10-19T14:40:00Z'
START, END = '2026-10-19T13:00:00Z', '2026-10-19T14:00:00Z'


@pytest.fixture()
def db():
    url = os.getenv('BRIAN_TEST_DATABASE_URL')
    if not url:
        pytest.skip('requires isolated PostgreSQL test database')
    psycopg2 = pytest.importorskip('psycopg2')
    conn = psycopg2.connect(url)
    schema = 'coverage_' + uuid.uuid4().hex
    cur = conn.cursor()
    cur.execute(f'create schema {schema}')
    cur.execute(f'''create table {schema}.brian_alpha_decisions (
      decision_id text primary key,asset_id text,compiler_version text default 'v1',observed_at timestamptz,
      created_at timestamptz default '2026-10-19T14:20:00Z',observed_reference_price numeric default 100,
      action text default 'OPEN_LONG',shadow_only boolean default true,live_execution boolean default false);
      create table {schema}.brian_alpha_decision_outcomes (
      outcome_id text,decision_id text,horizon_seconds int,resolved_at timestamptz,
      created_at timestamptz default '2026-10-19T14:30:00Z',reference_price numeric default 100,
      resolved_price numeric default 101,direction_adjusted_return float8 default .01,
      metadata jsonb default '{{"estimated_round_trip_cost_bps":20}}',shadow_only boolean default true,live_execution boolean default false);
      create table {schema}.brian_intrabar_reaction_events (
      asset_id text,observed_at timestamptz,created_at timestamptz,observed_mid_price numeric default 101,
      shadow_only boolean default true,live_execution boolean default false);
      create table {schema}.brian_audit_price_samples (
      asset_id text,observed_at timestamptz,captured_at timestamptz,observed_mid_price numeric default 101,
      shadow_only boolean default true,live_execution boolean default false)''')
    def execute(statement, args=()):
        cur.execute(statement.replace('public.', schema + '.'), args)
        return cur.fetchone()[0] if cur.description else None
    yield execute
    conn.rollback()
    cur.close()
    conn.close()


def run(db):
    return db(SQL.replace('$1', '%s').replace('$2', '%s').replace('$3', '%s'), (AT, START, END))


def decision(db, ident, at='2026-10-19T14:20:00Z'):
    db('insert into public.brian_alpha_decisions(decision_id,asset_id,observed_at) values(%s,%s,%s)', (ident, ident, at))


def row(result, asset, seconds=300, sample='rolling'):
    return next(r for r in result['rows'] if r['asset_id'] == asset and r['seconds'] == seconds and r['sample'] == sample)


def test_each_missing_lane_and_all_three_price_paths(db):
    for ident in ['alpha', 'intrabar', 'audit', 'absent', 'late', 'futurecapture']:
        decision(db, ident)
    db("insert into public.brian_alpha_decisions(decision_id,asset_id,observed_at) values('p','alpha','2026-10-19T14:25:00Z')")
    db("insert into public.brian_intrabar_reaction_events(asset_id,observed_at,created_at) values('intrabar','2026-10-19T14:25:00Z','2026-10-19T14:26:00Z')")
    db("insert into public.brian_audit_price_samples(asset_id,observed_at,captured_at) values('audit','2026-10-19T14:25:00Z','2026-10-19T14:26:00Z'),('late','2026-10-19T14:27:01Z','2026-10-19T14:28:00Z'),('futurecapture','2026-10-19T14:25:00Z','2026-10-19T14:41:00Z')")
    result = run(db)
    for asset in ['alpha', 'intrabar', 'audit']:
        assert row(result, asset)['awaiting_auditor'] == 1
    for asset in ['absent', 'late', 'futurecapture']:
        assert row(result, asset)['missing_terminal'] == 1


def test_not_mature_open_window_entry_and_invalid_receipt(db):
    decision(db, 'young', '2026-10-19T14:39:00Z')
    decision(db, 'open', '2026-10-19T14:34:00Z')
    decision(db, 'entry')
    db("update public.brian_alpha_decisions set observed_reference_price=0 where decision_id='entry'")
    decision(db, 'invalid')
    db("insert into public.brian_alpha_decision_outcomes(outcome_id,decision_id,horizon_seconds,resolved_at) values('i','invalid',300,'2026-10-19T14:24:59Z')")
    result = run(db)
    assert row(result, 'young')['not_mature'] == 1
    assert row(result, 'open')['window_open'] == 1
    assert row(result, 'entry')['missing_entry'] == 1
    assert row(result, 'invalid')['invalid_outcome'] == 1
    for r in result['rows']:
        assert r['sampled'] == r['mature'] + r['not_mature']
        assert r['mature'] == sum(r[k] for k in ['resolved', 'missing_entry', 'invalid_outcome', 'awaiting_auditor', 'window_open', 'missing_terminal'])


@pytest.mark.parametrize('cost,missing', [('20', 0), ('null', 1), ('"bad"', 1), ('-1', 1), ('"NaN"', 1), ('"1e999"', 1)])
def test_unknown_or_invalid_cost_is_never_zero(db, cost, missing):
    decision(db, 'resolved')
    db("insert into public.brian_alpha_decision_outcomes(outcome_id,decision_id,horizon_seconds,resolved_at,metadata) values('r','resolved',300,'2026-10-19T14:25:00Z',%s::jsonb)", ('{"estimated_round_trip_cost_bps":' + cost + '}',))
    r = row(run(db), 'resolved')
    assert r['resolved'] == 1
    assert r['missing_cost'] == missing
    assert r['cost_known_directional'] == 1 - missing


def test_point_in_time_and_mode_exclusions(db):
    for ident in ['futurecreated', 'nonshadow', 'live', 'outcomefuture', 'refmismatch']:
        decision(db, ident)
    decision(db, 'futureobserved', '2026-10-19T14:41:00Z')
    db("update public.brian_alpha_decisions set created_at='2026-10-19T14:41:00Z' where decision_id='futurecreated'")
    db("update public.brian_alpha_decisions set shadow_only=false where decision_id='nonshadow'")
    db("update public.brian_alpha_decisions set live_execution=true where decision_id='live'")
    db("insert into public.brian_alpha_decision_outcomes(outcome_id,decision_id,horizon_seconds,resolved_at,created_at) values('f','outcomefuture',300,'2026-10-19T14:25:00Z','2026-10-19T14:41:00Z')")
    db("insert into public.brian_alpha_decision_outcomes(outcome_id,decision_id,horizon_seconds,resolved_at,reference_price) values('m','refmismatch',300,'2026-10-19T14:25:00Z',99)")
    result = run(db)
    assert {r['asset_id'] for r in result['rows']} == {'outcomefuture', 'refmismatch'}
    assert row(result, 'outcomefuture')['missing_terminal'] == 1
    assert row(result, 'refmismatch')['invalid_outcome'] == 1


def test_frozen_window_is_half_open_includes_wait_and_veto(db):
    for ident, at in [('before', '2026-10-19T12:59:59Z'), ('start', START), ('inside', '2026-10-19T13:59:59Z'), ('end', END)]:
        decision(db, ident, at)
    db("update public.brian_alpha_decisions set action='WAIT' where decision_id='start'")
    db("update public.brian_alpha_decisions set action='VETO' where decision_id='inside'")
    result = run(db)
    assert {r['asset_id'] for r in result['rows'] if r['sample'] == 'prospective'} == {'start', 'inside'}
    assert result['prospective_truncated'] is False


def test_caps_are_explicit_and_empty_is_not_full_coverage(db):
    empty = run(db)
    assert empty['rows'] == []
    db("insert into public.brian_alpha_decisions(decision_id,asset_id,observed_at) select 'd'||n,'asset','2026-10-19T13:30:00Z'::timestamptz from generate_series(1,1001) n")
    result = run(db)
    assert result['rolling_truncated'] is True
    assert result['prospective_truncated'] is True
    assert row(result, 'asset')['sampled'] == 300
    assert row(result, 'asset', sample='prospective')['sampled'] == 1000
