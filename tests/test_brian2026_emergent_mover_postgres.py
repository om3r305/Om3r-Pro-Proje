"""Real-Postgres checks for the Emergent Mover append-only SHADOW store.

Skipped outside the dedicated CI Postgres job. The test applies the existing intelligence-memory
base migration first because migration 011 references brian_universe_snapshots and the shared
brian_reject_mutation() trigger function.

The vanilla postgres:16 service container does not ship Supabase's platform roles. The bootstrap
below mirrors the hosted role semantics that matter here: `service_role` has BYPASSRLS in Supabase.
Table grants are still tested separately, so BYPASSRLS does not grant UPDATE/DELETE privileges that
the migration deliberately revokes.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip(
    "psycopg2",
    reason="psycopg2 is only installed in the dedicated real-Postgres CI job",
)

ROOT = Path(__file__).resolve().parents[1]
BASE_MIGRATION = ROOT / "supabase" / "migrations" / "202609030001_brian_intelligence_memory.sql"
EMERGENT_MIGRATION = ROOT / "supabase" / "migrations" / "202609040011_brian_emergent_mover_frames.sql"
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; real-Postgres tests only run in dedicated CI",
)

_BOOTSTRAP_SUPABASE_ROLES_SQL = """
do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then
    create role anon nologin;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then
    create role service_role nologin;
  end if;
end
$$;
-- Hosted Supabase's service_role is an elevated server-side role with BYPASSRLS. This ALTER is
-- intentionally unconditional because the collector-lease test module may have created the
-- vanilla-CI placeholder role earlier in the same Postgres service container.
alter role service_role bypassrls;
"""


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


@pytest.fixture(scope="module", autouse=True)
def _apply_migrations():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP_SUPABASE_ROLES_SQL)
            cur.execute(BASE_MIGRATION.read_text(encoding="utf-8"))
            cur.execute(EMERGENT_MIGRATION.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _snapshot(conn) -> str:
    snapshot_id = f"pytest-universe-{uuid.uuid4().hex}"
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_universe_snapshots
              (snapshot_id, provider, observed_at, eligible_count, candidates, raw_capture_ids)
            values (%s, 'binance_public', '2026-09-04T08:00:00Z', 1, %s::jsonb, '{}')
            """,
            (snapshot_id, json.dumps({"eligible_symbols": ["BTCUSDT"]})),
        )
    return snapshot_id


def _insert_frame(
    conn,
    snapshot_id: str,
    *,
    frame_id: str | None = None,
    observed_at: str = "2026-09-04T08:05:00Z",
    baseline_observed_at: str | None = "2026-09-04T08:00:00Z",
    comparison_age_ms: int | None = 300_000,
    comparable: bool = True,
    shadow_only: bool = True,
    live_execution: bool = False,
):
    frame_id = frame_id or f"pytest-emergent-{uuid.uuid4().hex}"
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_emergent_mover_frames (
              frame_id, universe_snapshot_id, provider, observed_at,
              baseline_observed_at, comparison_age_ms, comparable, eligible_count,
              dropped_symbol_count, degraded_sources, raw_capture_ids,
              state, report, evidence_class, shadow_only, live_execution
            ) values (
              %s, %s, 'binance_public', %s,
              %s, %s, %s, 1,
              0, '{}', '{}',
              %s::jsonb, %s::jsonb, 'PROSPECTIVE_DEVELOPMENT_SHADOW', %s, %s
            )
            """,
            (
                frame_id,
                snapshot_id,
                observed_at,
                baseline_observed_at,
                comparison_age_ms,
                comparable,
                json.dumps({"schema_version": "brian.emergent-mover-frame.v1", "shadow_only": True}),
                json.dumps({"schema_version": "brian.emergent-mover-report.v1", "shadow_only": True}),
                shadow_only,
                live_execution,
            ),
        )
    return frame_id


def test_ci_service_role_matches_supabase_bypassrls_semantics():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("select rolbypassrls from pg_roles where rolname = 'service_role'")
            assert cur.fetchone() == (True,)
    finally:
        conn.close()


def test_valid_shadow_frame_inserts_and_service_role_can_read_it():
    conn = _connect()
    try:
        snapshot_id = _snapshot(conn)
        frame_id = _insert_frame(conn, snapshot_id)
        with conn.cursor() as cur:
            cur.execute("set role service_role")
            cur.execute(
                "select shadow_only, live_execution, evidence_class from public.brian_emergent_mover_frames where frame_id=%s",
                (frame_id,),
            )
            assert cur.fetchone() == (True, False, "PROSPECTIVE_DEVELOPMENT_SHADOW")
            cur.execute("reset role")
    finally:
        conn.close()


def test_service_role_can_insert_but_cannot_mutate_or_delete_a_frame():
    conn = _connect()
    try:
        snapshot_id = _snapshot(conn)
        frame_id = f"pytest-service-role-{uuid.uuid4().hex}"
        with conn.cursor() as cur:
            cur.execute("set role service_role")
            _insert_frame(conn, snapshot_id, frame_id=frame_id)
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_emergent_mover_frames set comparable=false where frame_id=%s",
                    (frame_id,),
                )
            with pytest.raises(psycopg2.Error):
                cur.execute("delete from public.brian_emergent_mover_frames where frame_id=%s", (frame_id,))
            cur.execute("reset role")
            cur.execute("select count(*) from public.brian_emergent_mover_frames where frame_id=%s", (frame_id,))
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shadow_only": False},
        {"live_execution": True},
        {"baseline_observed_at": "2026-09-04T08:05:00Z"},
        {"baseline_observed_at": "2026-09-04T08:06:00Z"},
        {"comparison_age_ms": 0},
        {"comparison_age_ms": -1},
    ],
)
def test_shadow_and_causality_constraints_fail_closed(kwargs):
    conn = _connect()
    try:
        snapshot_id = _snapshot(conn)
        with pytest.raises(psycopg2.Error):
            _insert_frame(conn, snapshot_id, **kwargs)
    finally:
        conn.close()


def test_baseline_frame_allows_null_baseline_and_null_comparison_age():
    conn = _connect()
    try:
        snapshot_id = _snapshot(conn)
        frame_id = _insert_frame(
            conn,
            snapshot_id,
            baseline_observed_at=None,
            comparison_age_ms=None,
            comparable=False,
        )
        with conn.cursor() as cur:
            cur.execute(
                "select comparable, baseline_observed_at, comparison_age_ms from public.brian_emergent_mover_frames where frame_id=%s",
                (frame_id,),
            )
            assert cur.fetchone() == (False, None, None)
    finally:
        conn.close()
