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
import threading
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


def test_renewal_after_release_cannot_resurrect_the_lease():
    """Closes the race from GPT-5.6 Sol's review on this PR: a heartbeat renewal RPC that was
    already in flight when work() finished -- and the lease was released -- must not be able to
    arrive at Postgres afterward and push lease_until back into the future, resurrecting a lease
    that was correctly released and blocking the next legitimate owner for a full TTL.

    This sequential case is doubly covered now: brian_renew_collector_lease's own
    lease_until > v_now guard rejects it, and (see
    test_release_invalidates_the_owner_token_so_a_later_renew_cannot_use_it and
    test_stale_inflight_renewal_cannot_resurrect_after_owner_token_rotation below)
    brian_release_collector_lease also atomically rotates owner_token on release, so the stale
    renewal's owner_token predicate can never match again regardless of timing.

    Sequence proved here: acquire A -> release A -> stale/late renew A returns False and cannot
    extend/reanimate the row -> B can acquire immediately.
    """
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        assert _release(conn, collector_id, "owner-a") is True
        row_after_release = _lease_row(conn, collector_id)

        # Simulates the stale, already-in-flight heartbeat renewal RPC landing *after* release:
        # same collector_id + owner_token as the just-released lease.
        assert _renew(conn, collector_id, "owner-a", 30) is False, "a renewal arriving after release must not resurrect the lease"

        row_after_stale_renew = _lease_row(conn, collector_id)
        assert row_after_stale_renew == row_after_release, "a rejected (fail-closed) renewal must not change the lease row at all"

        # B must be able to acquire immediately -- the stale renewal must not have blocked it by
        # pushing lease_until back into the future.
        assert _acquire(conn, collector_id, "owner-b", 30) is True
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-b"

        assert [e[0] for e in _events(conn, collector_id)] == ["ACQUIRED", "RELEASED", "RENEWAL_LOST", "EXPIRED_RECOVERY"]
    finally:
        conn.close()


def test_release_invalidates_the_owner_token_so_a_later_renew_cannot_use_it():
    """Regression for GPT-5.6 Sol's final review on PR #36: the lease_until > v_now guard on
    brian_renew_collector_lease is timestamp-based, and a renewal's v_now is captured (as the
    first statement in its function body) before it can be delayed/blocked -- so a release that
    commits *later* could, by coincidence of when each side's clock was read, still satisfy that
    guard. The real fix: brian_release_collector_lease now atomically rotates owner_token to a
    fresh, unguessable value on every successful release, so a renewal carrying the OLD owner
    token can never match the row again -- independent of any timestamp on either side.
    """
    collector_id = _collector_id()
    conn = _connect()
    try:
        assert _acquire(conn, collector_id, "owner-a", 30) is True
        row_before_release = _lease_row(conn, collector_id)
        assert row_before_release[0] == "owner-a"

        assert _release(conn, collector_id, "owner-a") is True
        row_after_release = _lease_row(conn, collector_id)
        assert row_after_release[0] != "owner-a", (
            "release must atomically invalidate/rotate owner_token, not merely backdate lease_until"
        )

        assert _renew(conn, collector_id, "owner-a", 30) is False, "the invalidated owner token must never renew again"
        assert _lease_row(conn, collector_id) == row_after_release, (
            "a renewal carrying the invalidated owner token must not change the row at all"
        )

        assert _acquire(conn, collector_id, "owner-b", 30) is True
        row = _lease_row(conn, collector_id)
        assert row[0] == "owner-b"
    finally:
        conn.close()


def test_stale_inflight_renewal_cannot_resurrect_after_owner_token_rotation():
    """Deterministic concurrent/locking regression for GPT-5.6 Sol's final review comment: proves
    that even in the exact adversarial ordering described there -- a renewal's clock_timestamp()
    is captured *before* release commits, and the renewal's UPDATE is still physically blocked, in
    flight, at the moment release commits -- the renewal cannot resurrect the lease, because
    release's owner_token rotation makes the outcome independent of timing entirely.

    Sequenced across three real connections using pg_stat_activity polling (never a fixed sleep,
    so the ordering is deterministic rather than timing-hopeful):
      1. conn_release runs brian_release_collector_lease inside an explicit, not-yet-committed
         transaction. Release's own body has already executed (owner_token rotated, lease_until
         backdated) and its UPDATE holds the row's exclusive lock, but none of that is durable or
         visible yet.
      2. A background thread calls brian_renew_collector_lease with the ORIGINAL owner token on a
         separate connection. Its v_now is captured immediately (the first statement in its
         function body runs before anything can block), then its own UPDATE blocks waiting for
         conn_release's still-open lock -- exactly "captures v_now, then is delayed/blocked before
         its UPDATE" from GPT's description.
      3. Only once pg_stat_activity confirms that renewal backend is genuinely waiting on a lock
         does the test commit conn_release -- the precise moment GPT's scenario says a
         timestamp-only guard could be fooled.
      4. The renewal's UPDATE then resumes and must still fail: owner_token no longer matches,
         regardless of what either side's clock read.
    """
    collector_id = _collector_id()
    conn_main = _connect()
    try:
        assert _acquire(conn_main, collector_id, "owner-a", 30) is True

        conn_release = psycopg2.connect(DATABASE_URL)
        conn_release.autocommit = False
        renew_thread = None
        try:
            with conn_release.cursor() as cur:
                cur.execute("select public.brian_release_collector_lease(%s, %s)", (collector_id, "owner-a"))
                assert cur.fetchone()[0] is True, "release must succeed before we hold its transaction open"

            renew_result: dict = {}

            def _do_renew() -> None:
                conn_renew = psycopg2.connect(DATABASE_URL)
                conn_renew.autocommit = True
                try:
                    renew_result["value"] = _renew(conn_renew, collector_id, "owner-a", 30)
                finally:
                    conn_renew.close()

            renew_thread = threading.Thread(target=_do_renew)
            renew_thread.start()

            conn_probe = _connect()
            try:
                deadline = time.monotonic() + 10.0
                blocked = False
                while time.monotonic() < deadline:
                    with conn_probe.cursor() as cur:
                        cur.execute(
                            "select count(*) from pg_stat_activity "
                            "where wait_event_type = 'Lock' and query ilike %s",
                            ("%brian_renew_collector_lease%",),
                        )
                        if cur.fetchone()[0] > 0:
                            blocked = True
                            break
                    time.sleep(0.05)
                assert blocked, (
                    "the renewal never reached a blocked-on-lock state within 10s; this test's "
                    "concurrency setup assumption did not hold, not a statement about the fix itself"
                )
            finally:
                conn_probe.close()

            # This is the exact moment GPT's scenario targets: the already-in-flight renewal (v_now
            # captured, now blocked) is about to see release's commit land.
            conn_release.commit()
        finally:
            conn_release.close()

        renew_thread.join(timeout=10.0)
        assert not renew_thread.is_alive(), "the renewal thread did not complete after release committed"
        assert renew_result.get("value") is False, (
            "a renewal already in flight when release commits must still fail closed -- release "
            "atomically invalidated the owner token it is renewing against, independent of timing"
        )

        row = _lease_row(conn_main, collector_id)
        assert row[0] != "owner-a", "release must have rotated owner_token away from the released owner"

        assert _acquire(conn_main, collector_id, "owner-b", 30) is True
        row = _lease_row(conn_main, collector_id)
        assert row[0] == "owner-b"
    finally:
        conn_main.close()
