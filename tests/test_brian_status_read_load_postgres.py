"""Real-Postgres test: the skip-scan "latest" views return exactly what the DISTINCT ON views did.

Both migrations (the original 202609151247 views and the 20261020090000 rewrite) are applied to an
isolated schema with minimal base tables, filled with random rows that repeat source ids and
timestamps. Columns, types, row sets and output order must match; the hypothesis index must exist.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 is only installed in the Postgres CI job")

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "supabase" / "migrations"
OLD = MIGRATIONS / "202609151247_brian_world_source_latest_views_v1.sql"
NEW = MIGRATIONS / "20261020090000_brian_status_read_load.sql"
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")

SCHEMA = "t_status_read_load"
TABLES = f"""
drop schema if exists {SCHEMA} cascade;
create schema {SCHEMA};
create table {SCHEMA}.brian_world_source_candidates (
  candidate_id text primary key, source_id text not null, discovered_at timestamptz not null,
  canonical_uri text not null, provider text not null, authority_class text not null,
  access_mode text not null, stage text not null);
create index on {SCHEMA}.brian_world_source_candidates (source_id, discovered_at desc);
create table {SCHEMA}.brian_world_source_assessments (
  assessment_id text primary key, source_id text not null, assessed_at timestamptz not null,
  trust_score double precision not null, eligible_for_research boolean not null);
create index on {SCHEMA}.brian_world_source_assessments (source_id, assessed_at desc);
create table {SCHEMA}.brian_evolution_hypothesis_snapshots (
  snapshot_id text primary key, hypothesis_id text not null, observed_at timestamptz not null);
"""
_BOOTSTRAP = """
do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin; end if;
end $$;
"""


def scoped(path: Path, suffix: str) -> str:
    sql = path.read_text(encoding="utf-8").replace("public.", f"{SCHEMA}.")
    # Keep both versions side by side: rename the views of this copy.
    for view in ("brian_world_source_latest", "brian_world_source_assessment_latest"):
        sql = sql.replace(f"{SCHEMA}.{view} ", f"{SCHEMA}.{view}_{suffix} ").replace(f"{SCHEMA}.{view}\n", f"{SCHEMA}.{view}_{suffix}\n")
        sql = sql.replace(f"{SCHEMA}.{view};", f"{SCHEMA}.{view}_{suffix};")
    return sql


@pytest.fixture(scope="module")
def cur():
    c = psycopg2.connect(DATABASE_URL)
    c.autocommit = True
    k = c.cursor()
    k.execute(_BOOTSTRAP)
    k.execute(TABLES)
    rnd = random.Random(20261020)
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    cands, assess = [], []
    for i in range(3000):
        src = f"src-{rnd.randrange(180):03d}"
        # Coarse timestamps so equal discovered_at values occur within one source.
        at = t0 + timedelta(minutes=rnd.randrange(400) * 5)
        cands.append((f"cand-{i:05d}", src, at, f"https://x/{i}", "p", "UNKNOWN", "PUBLIC_NO_KEY", rnd.choice(["DISCOVERED", "ACTIVE"])))
    for i in range(4000):
        src = f"src-{rnd.randrange(220):03d}"
        # Unique per (source, time) so the old view's arbitrary tie-break cannot differ.
        assess.append((f"as-{i:05d}", src, t0 + timedelta(seconds=i * 7 + rnd.randrange(5)), rnd.random(), rnd.random() < 0.5))
    k.executemany(f"insert into {SCHEMA}.brian_world_source_candidates values (%s,%s,%s,%s,%s,%s,%s,%s)", cands)
    k.executemany(f"insert into {SCHEMA}.brian_world_source_assessments values (%s,%s,%s,%s,%s)", assess)
    k.execute(f"analyze {SCHEMA}.brian_world_source_candidates; analyze {SCHEMA}.brian_world_source_assessments")
    k.execute(scoped(OLD, "old"))
    k.execute(scoped(NEW, "new"))
    yield k
    k.execute(f"drop schema if exists {SCHEMA} cascade")
    c.close()


def rows(cur, sql):
    cur.execute(sql)
    return cur.fetchall()


def test_old_and_new_definitions_really_differ(cur):
    for v in ("brian_world_source_latest", "brian_world_source_assessment_latest"):
        cur.execute(f"select pg_get_viewdef('{SCHEMA}.{v}_old'::regclass), pg_get_viewdef('{SCHEMA}.{v}_new'::regclass)")
        old, new = cur.fetchone()
        assert "DISTINCT ON" in old and "DISTINCT ON" not in new and "RECURSIVE" in new.upper()


def test_candidate_view_identical_rows_order_and_columns(cur):
    old = rows(cur, f"select * from {SCHEMA}.brian_world_source_latest_old")
    new = rows(cur, f"select * from {SCHEMA}.brian_world_source_latest_new")
    assert len(old) == 180 and old == new
    # The UI's actual query shape: newest 250 by discovered_at.
    q = "select * from {} order by discovered_at desc, source_id limit 250"
    assert rows(cur, q.format(f"{SCHEMA}.brian_world_source_latest_old")) == rows(cur, q.format(f"{SCHEMA}.brian_world_source_latest_new"))
    cols = "select column_name, data_type from information_schema.columns where table_schema = %s and table_name = %s order by ordinal_position"
    cur.execute(cols, (SCHEMA, "brian_world_source_latest_old"))
    a = cur.fetchall()
    cur.execute(cols, (SCHEMA, "brian_world_source_latest_new"))
    assert a == cur.fetchall()


def test_assessment_view_identical_rows_and_columns(cur):
    old = rows(cur, f"select * from {SCHEMA}.brian_world_source_assessment_latest_old")
    new = rows(cur, f"select * from {SCHEMA}.brian_world_source_assessment_latest_new")
    assert len(old) == 220 and old == new
    cols = "select column_name, data_type from information_schema.columns where table_schema = %s and table_name = %s order by ordinal_position"
    cur.execute(cols, (SCHEMA, "brian_world_source_assessment_latest_old"))
    a = cur.fetchall()
    cur.execute(cols, (SCHEMA, "brian_world_source_assessment_latest_new"))
    assert a == cur.fetchall()


def test_assessment_ties_are_deterministic(cur):
    at = "2027-01-01T00:00:00Z"
    cur.execute(f"insert into {SCHEMA}.brian_world_source_assessments values ('tie-a','src-tie',%s,0.1,false),('tie-b','src-tie',%s,0.9,true)", (at, at))
    r = rows(cur, f"select trust_score from {SCHEMA}.brian_world_source_assessment_latest_new where source_id = 'src-tie'")
    assert r == [(0.9,)]  # assessment_id desc: 'tie-b' wins, every time
    cur.execute(f"delete from {SCHEMA}.brian_world_source_assessments where source_id = 'src-tie'")


def test_empty_tables_and_security_invoker(cur):
    cur.execute(f"create table {SCHEMA}.empty_probe as select 1")
    cur.execute(f"select reloptions from pg_class where oid = '{SCHEMA}.brian_world_source_latest_new'::regclass")
    assert cur.fetchone()[0] == ["security_invoker=true"]
    cur.execute(f"select has_table_privilege('anon', '{SCHEMA}.brian_world_source_latest_new', 'select')")
    assert cur.fetchone()[0] is False
    cur.execute(f"select indexdef from pg_indexes where schemaname = %s and indexname = 'brian_evolution_hypothesis_snapshots_observed_idx'", (SCHEMA,))
    assert "observed_at DESC" in cur.fetchone()[0]


def test_skip_scan_on_empty_base_returns_nothing(cur):
    cur.execute(f"create table {SCHEMA}.bk as select * from {SCHEMA}.brian_world_source_candidates")
    cur.execute(f"delete from {SCHEMA}.brian_world_source_candidates")
    assert rows(cur, f"select * from {SCHEMA}.brian_world_source_latest_new") == []
    cur.execute(f"insert into {SCHEMA}.brian_world_source_candidates select * from {SCHEMA}.bk")
