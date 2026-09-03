"""Real-Postgres integration tests for the Item 1 collector lease/mutex (brian-2026 issue #32).

These tests apply supabase/migrations/202609030009_brian_collector_lease.sql to a real Postgres
database and exercise brian_acquire_collector_lease / brian_release_collector_lease directly via
SQL, proving atomicity under real concurrency -- something a mocked RPC client (see
supabase/functions/_shared/collector_lease.test.ts) cannot prove by itself.

This file is a no-op everywhere except the dedicated `verify-collector-lease` CI job:
  - it is skipped entirely if psycopg2 is not installed (it deliberately is not a
    requirements.txt dependency, so the existing `verify` job's full test suite is unaffected);
  - it is skipped if BRIAN_TEST_DATABASE_URL is not set, which is only exported by the
    verify-collector-lease job in .github/workflows/brian-ci.yml, pointed at that job's Postgres
    service container.

The migration file is standalone (no vault/pg_net/pg_cron dependency -- see the comment at the
top of the migration itself), so it can be applied directly to a vanilla Postgres instance.
"""

from __future__ import annotations

import os
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
