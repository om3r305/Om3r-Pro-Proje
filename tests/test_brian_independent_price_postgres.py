"""Actual migration + RPCs in a rollback-only isolated schema, never production fixtures."""
import hashlib
import json
import os
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / 'supabase/migrations/20261019132120_independent_outcome_price_capture.sql').read_text()


@pytest.fixture()
def db():
    url = os.getenv('BRIAN_TEST_DATABASE_URL')
    if not url:
        pytest.skip('requires isolated PostgreSQL test database')
    psycopg2 = pytest.importorskip('psycopg2')
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    schema = 'price_' + uuid.uuid4().hex
    cur.execute("do $$begin if not exists(select 1 from pg_roles where rolname='anon') then create role anon; end if; if not exists(select 1 from pg_roles where rolname='authenticated') then create role authenticated; end if; if not exists(select 1 from pg_roles where rolname='service_role') then create role service_role; end if; end$$")
    cur.execute(f'''create schema {schema};
      create table {schema}.brian_evolution_runtime_config(config_key text primary key,config_value text);
      insert into {schema}.brian_evolution_runtime_config values('brian_system_enabled','true');
      create table {schema}.brian_alpha_decisions(decision_id text,asset_id text,observed_at timestamptz,
        created_at timestamptz default clock_timestamp(),observed_reference_price numeric default 100,
        shadow_only boolean default true,live_execution boolean default false);
      create table {schema}.brian_alpha_decision_outcomes(decision_id text,horizon_seconds int,
        resolved_at timestamptz,created_at timestamptz default clock_timestamp(),shadow_only boolean default true,live_execution boolean default false);
      create table {schema}.brian_audit_price_samples(source_event_id text primary key,asset_id text,observed_at timestamptz,
        observed_mid_price numeric not null check(observed_mid_price>0 and observed_mid_price::text not in ('NaN','Infinity','-Infinity')),
        estimated_round_trip_cost_bps float8,source_project text constraint brian_audit_price_samples_source_project_check check(source_project='REALTIME'),
        captured_at timestamptz not null default clock_timestamp(),shadow_only boolean default true check(shadow_only),live_execution boolean default false check(not live_execution));
      create function {schema}.reject_mutation() returns trigger language plpgsql as $$begin raise exception 'APPEND_ONLY'; end$$;
      create trigger immutable before update or delete on {schema}.brian_audit_price_samples for each row execute function {schema}.reject_mutation();
      grant usage on schema {schema} to anon,authenticated,service_role;
      grant select,insert on {schema}.brian_audit_price_samples to service_role;''')
    def execute(statement, args=()):
        statement = statement.replace('public.', schema + '.').replace('pg_catalog,public', 'pg_catalog,' + schema)
        cur.execute(statement, args)
        return cur.fetchall() if cur.description else []
    execute(MIGRATION)
    yield execute
    conn.rollback()
    cur.close()
    conn.close()


def seed(db, ident='d', asset='crypto:BTCUSDT', age='1 minute'):
    db("insert into public.brian_alpha_decisions(decision_id,asset_id,observed_at) values(%s,%s,clock_timestamp()-%s::interval)", (ident, asset, age))


def receipt(db, asset='crypto:BTCUSDT'):
    observed = db("select clock_timestamp()-interval '2 seconds'")[0][0]
    minute = observed.strftime('%Y-%m-%dT%H:%M')
    return {'source_event_id': hashlib.sha256(f'brian.outcome-price-eye.v1|{asset}|{minute}'.encode()).hexdigest(),
            'asset_id': asset, 'observed_at': observed.isoformat(), 'request_started_at': (observed - timedelta(seconds=1)).isoformat(),
            'bid_price': '.1', 'ask_price': '.2', 'source_metadata': {'quote_time_semantics': 'HTTP_RECEIPT_TIME_EXCHANGE_TIME_UNKNOWN', 'provider_response_sha256': 'a' * 64}}


def persist(db, rows):
    return db('select public.brian_record_outcome_prices_v1(%s::jsonb)', (json.dumps(rows),))[0][0]


def test_watch_is_independent_of_radar_and_preserves_all_modes_and_horizons(db):
    seed(db, 'rotated', 'crypto:ROTATEDUSDT')
    seed(db, 'old', 'crypto:OLDUSDT', '63 minutes')
    seed(db, 'future', 'crypto:FUTUREUSDT', '-1 minute')
    seed(db, 'null', 'crypto:NULLUSDT')
    db("update public.brian_alpha_decisions set observed_reference_price=null where decision_id='null'")
    seed(db, 'mode', 'crypto:MODEUSDT')
    db("update public.brian_alpha_decisions set live_execution=true where decision_id='mode'")
    seed(db, 'done', 'crypto:DONEUSDT', '61 minutes')
    db("insert into public.brian_alpha_decision_outcomes(decision_id,horizon_seconds,resolved_at) select 'done',seconds,observed_at+make_interval(secs=>seconds) from public.brian_alpha_decisions cross join (values(300),(900),(3600)) h(seconds) where decision_id='done'")
    assert db('select * from public.brian_outcome_watch_assets_v1()') == [('crypto:ROTATEDUSDT',)]


def test_exact_decimal_midpoint_unknown_cost_and_append_only_identity(db):
    seed(db)
    p = receipt(db)
    assert persist(db, [p])['stored'] == 1
    assert persist(db, [p])['stored'] == 0
    row = db('select source_project,observed_mid_price,estimated_round_trip_cost_bps,observed_at<=captured_at from public.brian_audit_price_samples')[0]
    assert row == ('CORE', Decimal('.15'), None, True)
    db('savepoint immutable')
    with pytest.raises(Exception, match='APPEND_ONLY'):
        db('update public.brian_audit_price_samples set observed_mid_price=200')
    db('rollback to savepoint immutable')


def test_stop_is_authoritative_at_persist_time(db):
    seed(db)
    db("update public.brian_evolution_runtime_config set config_value='false'")
    assert persist(db, [receipt(db)]) == {'status': 'OPERATOR_STOPPED', 'stored': 0}
    assert db('select count(*) from public.brian_audit_price_samples')[0][0] == 0


@pytest.mark.parametrize('change', ['crossed', 'nan', 'future', 'old', 'duration', 'identity', 'metadata'])
def test_invalid_receipts_fail_without_synthetic_rows(db, change):
    seed(db)
    p = receipt(db)
    if change == 'crossed':
        p['ask_price'] = '.01'
    elif change == 'nan':
        p['bid_price'] = 'NaN'
    elif change == 'future':
        p['observed_at'] = '2200-01-01T00:00:00Z'
    elif change == 'old':
        p['observed_at'] = '2000-01-01T00:00:00Z'
    elif change == 'duration':
        p['request_started_at'] = '2000-01-01T00:00:00Z'
    elif change == 'identity':
        p['source_event_id'] = 'x' * 64
    elif change == 'metadata':
        p['source_metadata'] = {}
    db('savepoint invalid')
    with pytest.raises(Exception):
        persist(db, [p])
    db('rollback to savepoint invalid')
    assert db('select count(*) from public.brian_audit_price_samples')[0][0] == 0


def test_existing_realtime_inserts_and_private_permissions_remain_compatible(db):
    db("insert into public.brian_audit_price_samples(source_event_id,asset_id,observed_at,observed_mid_price,source_project) values('old','crypto:BTCUSDT',clock_timestamp(),100,'REALTIME')")
    assert db('select bid_price,source_metadata from public.brian_audit_price_samples')[0] == (None, None)
    rights = db("select has_function_privilege('anon','public.brian_record_outcome_prices_v1(jsonb)','execute'),has_function_privilege('authenticated','public.brian_outcome_watch_assets_v1()','execute'),has_function_privilege('service_role','public.brian_record_outcome_prices_v1(jsonb)','execute'),has_table_privilege('anon','public.brian_audit_price_samples','select')")[0]
    assert rights == (False, False, True, False)
    assert db("select bool_and(not prosecdef) from pg_proc where oid in ('public.brian_record_outcome_prices_v1(jsonb)'::regprocedure,'public.brian_outcome_watch_assets_v1()'::regprocedure)")[0][0]


def test_clock_observation_is_available_in_existing_terminal_price_path(db):
    seed(db, age='5 minutes 2 seconds')
    p = receipt(db)
    assert persist(db, [p])['stored'] == 1
    assert db("select exists(select 1 from public.brian_audit_price_samples p join public.brian_alpha_decisions d on d.asset_id=p.asset_id where p.observed_at between d.observed_at+interval '5 minutes' and d.observed_at+interval '7 minutes' and p.captured_at<=clock_timestamp())")[0][0]
