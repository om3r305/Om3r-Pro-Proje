"""Real-Postgres integration tests for the Item 1 collector lease/mutex (brian-2026 issue #32).

These tests apply supabase/migrations/202609030009_brian_collector_lease.sql to a real Postgres
database and exercise brian_acquire_collector_lease / brian_renew_collector_lease /
brian_release_collector_lease directly via SQL, proving atomicity (and, for renewal, safety for a
slow-but-alive owner) under real concurrency -- something a mocked RPC client (see
supabase/functions/_shared/collector_lease.test.ts) cannot prove by itself.

This file is a no-op everywhere except the dedicated `verify-collector-lease` CI job:
  - it is skipped entirely if psycopg2 is not installed (it deliberately is not a
    requirements.txt dependency, so the existing `verify` job's full test suite is unaffected);
  - it is skipped if BRIAN_TEST_DATABASE_URL is not set, which is only exported by the
    verify-collector-lease job in .github/workflows/brian-ci.yml, pointed at that job's Postgres
    service container.

The migration file has no vault/pg_net/pg_cron dependency, but it is NOT directly applicable to a
bare vanilla Postgres instance as-is: its REVOKE/GRANT statements reference the
anon/authenticated/service_role roles that only exist on a real Supabase Postgres instance. This
file is testable on vanilla Postgres only after the minimal role bootstrap below, run once as the
`postgres` superuser before the migration is applied -- see _apply_migration. The production
grants in the migration itself are unchanged by this bootstrap.
"""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 is only installed in the verify-collector-lease CI job")

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609030009_brian_collector_lease.sql"

DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; real-Postgres lease tests only run in the dedicated CI job",
)

# Minimal, idempotent bootstrap of the three Supabase roles the migration's REVOKE/GRANT
# statements reference. A vanilla postgres:16 CI database has none of these; a real Supabase
# Postgres instance always already has them (created by Supabase itself), so running this again
# there would be a no-op -- but this file only ever runs against the dedicated CI database, never
# against a real Supabase project.
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
"""


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


@pytest.fixture(scope="module", autouse=True)
def _apply_migration():
    sql = MIGRATION.read_text(encoding="utf-8")
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP_SUPABASE_ROLES_SQL)
            cur.execute(sql)
    finally:
        conn.close()


def _collector_id() -> str:
    # Each test uses its own randomly generated collector_id, so tests never interfere with each
    # other and no teardown/cleanup step is required for correctness.
    return f"pytest-lease-{uuid.uuid4().hex[:12]}"


def _acquire(conn, collector_id: str, owner_token: str, lease_seconds: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_acquire_collector_lease(%s, %s, %s)",
            (collector_id, owner_token, lease_seconds),
        )
        return bool(cur.fetchone()[0])


def _renew(conn, collector_id: str, owner_token: str, lease_seconds: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_renew_collector_lease(%s, %s, %s)",
            (collector_id, owner_token, lease_seconds),
        )
        return bool(cur.fetchone()[0])


def _release(conn, collector_id: str, owner_token: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_release_collector_lease(%s, %s)",
            (collector_id, owner_token),
        )
        return bool(cur.fetchone()[0])


def _lease_row(conn, collector_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "select owner_token, lease_until from public.brian_collector_leases where collector_id = %s",
            (collector_id,),
        )
        return cur.fetchone()


def _events(conn, collector_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "select event, owner_token from public.brian_collector_lease_events "
            "where collector_id = %s order by observed_at asc, ctid asc",
            (collector_id,),
        )
        return cur.fetchall()


def test_acquire_success_on_a_fresh_collector():
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        row = _lease_row(conn, collector_id)
        assert row is not None
        assert row[0] == "owner-a"
        assert _events(conn, collector_id) == [("ACQUIRED", "owner-a")]
    finally:
        conn.close()


def test_second_owner_is_contended_while_lease_is_active():
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        assert _acquire(conn, collector_id, "owner-b", 30) is False
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-a", "the contended acquire must not have overwritten the active owner"
        assert [e[0] for e in _events(conn, collector_id)] == ["ACQUIRED", "BLOCKED_ACTIVE"]
    finally:
        conn.close()


def test_release_by_wrong_owner_is_rejected():
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        assert _release(conn, collector_id, "owner-b") is False
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-a", "a wrong-owner release must not affect the current lease"
        assert row[1] is not None
        assert [e[0] for e in _events(conn, collector_id)] == ["ACQUIRED"], "a rejected release must not be logged as RELEASED"
    finally:
        conn.close()


def test_release_by_owner_succeeds_and_allows_immediate_reacquire():
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        assert _release(conn, collector_id, "owner-a") is True
        assert _acquire(conn, collector_id, "owner-b", 30) is True
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-b"
        assert [e[0] for e in _events(conn, collector_id)] == ["ACQUIRED", "RELEASED", "EXPIRED_RECOVERY"]
    finally:
        conn.close()


def test_expired_lease_is_taken_over_by_a_new_owner():
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        with conn.cursor() as cur:
            cur.execute(
                "update public.brian_collector_leases set lease_until = now() - interval '1 second' where collector_id = %s",
                (collector_id,),
            )
        assert _acquire(conn, collector_id, "owner-b", 30) is True
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-b"
        assert [e[0] for e in _events(conn, collector_id)] == ["ACQUIRED", "EXPIRED_RECOVERY"]
    finally:
        conn.close()


def test_concurrent_acquisitions_for_the_same_collector_never_both_succeed():
    """The real integration proof brian-2026 issue #32 asked for: N independent Postgres
    connections race to acquire the same brand-new collector_id at once. Exactly one must
    succeed regardless of how the OS schedules the threads -- this is what proves the
    INSERT ... ON CONFLICT ... WHERE primitive is atomic rather than merely 'usually fine'.
    """
    collector_id = _collector_id()
    attempts = 20

    def attempt(i: int) -> bool:
        conn = _connect()
        try:
            return _acquire(conn, collector_id, f"owner-{i}", 30)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        futures = [pool.submit(attempt, i) for i in range(attempts)]
        results = [f.result() for f in as_completed(futures)]

    assert results.count(True) == 1, f"expected exactly one winner among {attempts} concurrent acquirers, got {results.count(True)}"
    assert results.count(False) == attempts - 1

    conn = _connect()
    try:
        events = _events(conn, collector_id)
        assert len(events) == attempts
        assert sum(1 for e in events if e[0] == "ACQUIRED") == 1
        assert sum(1 for e in events if e[0] == "BLOCKED_ACTIVE") == attempts - 1
    finally:
        conn.close()


def test_renew_by_owner_extends_the_lease_and_logs_renewed():
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        row_before = _lease_row(conn, collector_id)
        assert _renew(conn, collector_id, "owner-a", 60) is True
        row_after = _lease_row(conn, collector_id)
        assert row_after[0] == "owner-a"
        assert row_after[1] > row_before[1], "renewal must push lease_until further into the future"
        assert [e[0] for e in _events(conn, collector_id)] == ["ACQUIRED", "RENEWED"]
    finally:
        conn.close()


def test_renew_by_wrong_owner_is_rejected_and_does_not_change_the_lease():
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        row_before = _lease_row(conn, collector_id)
        assert _renew(conn, collector_id, "owner-b", 60) is False
        row_after = _lease_row(conn, collector_id)
        assert row_after == row_before, "a wrong-owner renewal must not change the lease row at all"
        assert [e[0] for e in _events(conn, collector_id)] == ["ACQUIRED", "RENEWAL_LOST"]
    finally:
        conn.close()


def test_renewal_keeps_a_slow_but_alive_owner_safe_from_overlap():
    """The exact acceptance scenario from GPT-5.6 Sol's review: without renewal, a lease shorter
    than a slow invocation's real runtime lets a second invocation take over while the first is
    still alive and writing, recreating the overlap Item 1 exists to close. This proves the fix:

      1. A remains the live owner beyond its *original* lease window, via renewal -> B cannot
         acquire.
      2. Only after A stops renewing and the (renewed) lease genuinely expires can B acquire.
      3. A's stale release, arriving after B already owns the lease, cannot disturb B.
    """
    collector_id = _collector_id()
    conn = _connect()
    try:
        lease_seconds = 5
        assert _acquire(conn, collector_id, "owner-a", lease_seconds) is True

        time.sleep(2)
        # A renews well before its original lease would expire, simulating a heartbeat from a
        # still-alive, slow-running worker. This pushes lease_until to roughly now + 5s (~t=7).
        assert _renew(conn, collector_id, "owner-a", lease_seconds) is True

        time.sleep(3.5)
        # ~t=5.5 -- past the ORIGINAL lease_until (t=5) but comfortably before the renewed one
        # (~t=7). Without renewal this acquire would have succeeded; with it, it must not.
        assert _acquire(conn, collector_id, "owner-b", 30) is False
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-a", "A's renewed lease must still be in force past the original TTL"

        time.sleep(3)
        # ~t=8.5 -- past the renewed lease_until (~t=7). A has stopped renewing (simulating a
        # crash or a clean exit), so B may now legitimately take over.
        assert _acquire(conn, collector_id, "owner-b", 30) is True
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-b"

        # A's stale release, arriving late (e.g. from a delayed `finally` block), must not be able
        # to disturb B's now-current lease.
        assert _release(conn, collector_id, "owner-a") is False
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-b", "a stale release from the deposed owner must not affect the new owner's lease"

        assert [e[0] for e in _events(conn, collector_id)] == ["ACQUIRED", "RENEWED", "BLOCKED_ACTIVE", "EXPIRED_RECOVERY"]
    finally:
        conn.close()
