"""Real-Postgres integration tests for Phase 70 durable runtime persistence.

The tests run only in the dedicated Postgres CI job. They prove the database
boundary itself serializes lease ownership, fencing, checkpoint CAS and
append-only journal persistence under actual concurrent connections.
"""

from __future__ import annotations

import copy
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip(
    "psycopg2",
    reason="psycopg2 is only installed in the dedicated Postgres CI job",
)
from psycopg2.extras import Json

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "202609230720_brian_phase70_durable_runtime_store.sql"
)

DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase70 Postgres tests require CI database",
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
"""


def _connect(*, autocommit: bool = True):
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = autocommit
    return conn


@pytest.fixture(scope="module", autouse=True)
def _apply_migration():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP_SUPABASE_ROLES_SQL)
            cur.execute(MIGRATION.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _runtime_id(label: str) -> str:
    return f"pytest-phase70-{label}-{uuid.uuid4().hex[:10]}"


def _hash(char: str) -> str:
    return char * 64


def _checkpoint(
    checkpoint_char: str,
    *,
    head_char: str = "a",
    journal_char: str = "b",
    cycles: dict | None = None,
    entries: list | None = None,
    pending_cycle_id: str | None = None,
):
    cycles = {} if cycles is None else cycles
    entries = [] if entries is None else entries
    return {
        "schema_version": "brian.phase67-durable-runtime-orchestrator.v1",
        "runtime_checkpoint": {
            "schema_version": "brian.phase63-crash-recovery.v1",
            "paper": {
                "schema_version": "brian.phase63-crash-recovery.v1",
                "config": {
                    "account_id": "paper-acct",
                    "starting_cash_usd": 1000.0,
                    "fee_bps": 0.0,
                },
                "cash_usd": 1000.0,
                "state_version": 0,
                "fill_sequence": [],
                "cycle_receipts": [],
                "final_positions": [],
                "paper_only": True,
                "live_execution": False,
                "checkpoint_id": _hash("p"),
            },
            "shadow_ledger_manifest": {
                "schema_version": "brian.phase60-shadow-state-ledger.v1",
                "append_only": True,
                "head_state_id": _hash(head_char),
                "pending_cycle_id": pending_cycle_id,
                "states": {
                    _hash(head_char): {
                        "state_id": _hash(head_char),
                        "account_id": "paper-acct",
                    }
                },
                "transitions": [],
                "ledger_hash": _hash("l"),
                "shadow_only": True,
                "live_execution": False,
            },
            "pending_cycle_id": pending_cycle_id,
            "live_execution": False,
            "checkpoint_id": _hash("r"),
        },
        "journal_manifest": {
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "append_only": True,
            "cycles": cycles,
            "entries": entries,
            "journal_hash": _hash(journal_char),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _hash(checkpoint_char),
    }


def _cycle(cycle_id: str):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": f"plan-{cycle_id}",
        "items": [],
        "initial_available_cash_usd": 1000.0,
        "reserved_new_risk_cash_usd": 0.0,
        "remaining_unreserved_cash_usd": 1000.0,
        "denied_assets": [],
        "pending_reversal_assets": [],
        "cycle_id": cycle_id,
        "account_state_mutated": False,
        "shadow_only": True,
        "live_execution": False,
    }


def _entry(
    sequence: int,
    *,
    entry_char: str,
    previous_char: str | None,
    cycle_id: str,
    stage: str,
    cycle_char: str = "c",
    artifact_char: str = "d",
):
    return {
        "schema_version": "brian.phase66-durable-cycle-journal.v1",
        "sequence": sequence,
        "stage": stage,
        "cycle_id": cycle_id,
        "cycle_hash": _hash(cycle_char),
        "previous_entry_id": None if previous_char is None else _hash(previous_char),
        "artifact_hash": _hash(artifact_char),
        "artifact_ref": f"artifact-{sequence}-{stage}",
        "entry_id": _hash(entry_char),
    }


def _acquire(conn, runtime_id: str, owner: str, seconds: int = 30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_acquire_shadow_runtime_lease(%s,%s,%s)",
            (runtime_id, owner, seconds),
        )
        return cur.fetchone()[0]


def _renew(conn, runtime_id: str, owner: str, fence: int, seconds: int = 30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_renew_shadow_runtime_lease(%s,%s,%s,%s)",
            (runtime_id, owner, fence, seconds),
        )
        return cur.fetchone()[0]


def _release(conn, runtime_id: str, owner: str, fence: int):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_release_shadow_runtime_lease(%s,%s,%s)",
            (runtime_id, owner, fence),
        )
        return cur.fetchone()[0]


def _commit(conn, runtime_id: str, owner: str, fence: int, version: int, payload: dict):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_shadow_runtime_checkpoint(%s,%s,%s,%s,%s)",
            (runtime_id, owner, fence, version, Json(payload)),
        )
        return cur.fetchone()[0]


def _read(conn, runtime_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_read_shadow_runtime_checkpoint(%s)",
            (runtime_id,),
        )
        return cur.fetchone()[0]


def _count(conn, table: str, runtime_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_fresh_acquire_is_idempotent_for_same_owner_and_blocks_other_owner():
    runtime_id = _runtime_id("acquire")
    conn = _connect()
    try:
        first = _acquire(conn, runtime_id, "owner-a", 30)
        again = _acquire(conn, runtime_id, "owner-a", 30)
        blocked = _acquire(conn, runtime_id, "owner-b", 30)

        assert first["acquired"] is True
        assert first["status"] == "ACQUIRED"
        assert first["fencing_token"] == 1
        assert first["version"] == 0

        assert again["acquired"] is True
        assert again["status"] == "ALREADY_OWNED"
        assert again["fencing_token"] == 1

        assert blocked["acquired"] is False
        assert blocked["status"] == "BLOCKED_ACTIVE"
        assert blocked["fencing_token"] == 1
    finally:
        conn.close()


def test_two_concurrent_fresh_acquires_have_exactly_one_owner():
    runtime_id = _runtime_id("acquire-race")
    barrier = threading.Barrier(2)

    def contender(owner: str):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return owner, _acquire(conn, runtime_id, owner, 30)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(contender, ("owner-a", "owner-b")))

    winners = [(owner, row) for owner, row in results if row["acquired"]]
    losers = [(owner, row) for owner, row in results if not row["acquired"]]
    assert len(winners) == 1
    assert len(losers) == 1
    assert winners[0][1]["status"] == "ACQUIRED"
    assert losers[0][1]["status"] == "BLOCKED_ACTIVE"


def test_commit_persists_checkpoint_head_and_exact_retry_is_idempotent():
    runtime_id = _runtime_id("commit")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a", 30)
        checkpoint = _checkpoint("1")

        committed = _commit(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            0,
            checkpoint,
        )
        duplicate = _commit(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            0,
            checkpoint,
        )
        head = _read(conn, runtime_id)

        assert committed["committed"] is True
        assert committed["duplicate"] is False
        assert committed["status"] == "COMMITTED"
        assert committed["version"] == 1

        assert duplicate["committed"] is True
        assert duplicate["duplicate"] is True
        assert duplicate["status"] == "DUPLICATE_CURRENT"
        assert duplicate["version"] == 1

        assert head["version"] == 1
        assert head["checkpoint_id"] == checkpoint["checkpoint_id"]
        assert head["checkpoint_payload"] == checkpoint
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
    finally:
        conn.close()


def test_concurrent_same_version_commits_are_serialized_by_cas():
    runtime_id = _runtime_id("cas-race")
    setup = _connect()
    try:
        lease = _acquire(setup, runtime_id, "owner-a", 30)
        fence = lease["fencing_token"]
    finally:
        setup.close()

    barrier = threading.Barrier(2)
    payloads = (_checkpoint("2", head_char="2"), _checkpoint("3", head_char="3"))

    def contender(payload: dict):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _commit(conn, runtime_id, "owner-a", fence, 0, payload)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(contender, payloads))

    committed = [row for row in results if row["status"] == "COMMITTED"]
    conflicted = [row for row in results if row["status"] == "CAS_CONFLICT"]
    assert len(committed) == 1
    assert len(conflicted) == 1
    assert committed[0]["version"] == 1
    assert conflicted[0]["version"] == 1

    conn = _connect()
    try:
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
        assert _read(conn, runtime_id)["version"] == 1
    finally:
        conn.close()


def test_expired_takeover_increments_fence_and_stale_owner_cannot_commit():
    runtime_id = _runtime_id("fence")
    conn = _connect()
    try:
        first = _acquire(conn, runtime_id, "owner-a", 1)
        assert first["fencing_token"] == 1
        time.sleep(1.15)

        takeover = _acquire(conn, runtime_id, "owner-b", 30)
        assert takeover["acquired"] is True
        assert takeover["status"] == "EXPIRED_RECOVERY"
        assert takeover["fencing_token"] == 2

        stale = _commit(
            conn,
            runtime_id,
            "owner-a",
            1,
            0,
            _checkpoint("4"),
        )
        assert stale["committed"] is False
        assert stale["status"] == "LEASE_LOST"

        good = _commit(
            conn,
            runtime_id,
            "owner-b",
            2,
            0,
            _checkpoint("5"),
        )
        assert good["status"] == "COMMITTED"
        assert good["version"] == 1
    finally:
        conn.close()


def test_release_rotates_ownership_and_inflight_old_renewal_cannot_resurrect():
    runtime_id = _runtime_id("release")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a", 30)
        fence = lease["fencing_token"]
        released = _release(conn, runtime_id, "owner-a", fence)
        assert released["released"] is True

        renewal = _renew(conn, runtime_id, "owner-a", fence, 30)
        assert renewal["renewed"] is False
        assert renewal["status"] == "RENEWAL_LOST"

        takeover = _acquire(conn, runtime_id, "owner-b", 30)
        assert takeover["acquired"] is True
        assert takeover["status"] == "EXPIRED_RECOVERY"
        assert takeover["fencing_token"] == fence + 1
    finally:
        conn.close()


def test_journal_prefix_extension_is_append_only_across_checkpoint_versions():
    runtime_id = _runtime_id("journal")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a", 30)
        fence = lease["fencing_token"]
        cycle_id = "cycle-journal"
        cycles = {cycle_id: _cycle(cycle_id)}
        created = _entry(
            0,
            entry_char="6",
            previous_char=None,
            cycle_id=cycle_id,
            stage="CYCLE_CREATED",
        )
        checkpoint1 = _checkpoint(
            "7",
            journal_char="7",
            cycles=cycles,
            entries=[created],
            pending_cycle_id=cycle_id,
        )
        first = _commit(conn, runtime_id, "owner-a", fence, 0, checkpoint1)
        assert first["version"] == 1

        paper = _entry(
            1,
            entry_char="8",
            previous_char="6",
            cycle_id=cycle_id,
            stage="PAPER_APPLIED",
            artifact_char="8",
        )
        checkpoint2 = _checkpoint(
            "9",
            head_char="a",
            journal_char="9",
            cycles=cycles,
            entries=[created, paper],
            pending_cycle_id=cycle_id,
        )
        second = _commit(conn, runtime_id, "owner-a", fence, 1, checkpoint2)
        assert second["version"] == 2
        assert _count(conn, "brian_shadow_runtime_cycles", runtime_id) == 1
        assert _count(conn, "brian_shadow_runtime_journal_entries", runtime_id) == 2
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 2
    finally:
        conn.close()


def test_changed_existing_journal_prefix_is_rejected_without_advancing_head():
    runtime_id = _runtime_id("prefix-conflict")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a", 30)
        fence = lease["fencing_token"]
        cycle_id = "cycle-prefix"
        cycles = {cycle_id: _cycle(cycle_id)}
        created = _entry(
            0,
            entry_char="a",
            previous_char=None,
            cycle_id=cycle_id,
            stage="CYCLE_CREATED",
        )
        checkpoint1 = _checkpoint(
            "b",
            journal_char="c",
            cycles=cycles,
            entries=[created],
            pending_cycle_id=cycle_id,
        )
        assert _commit(conn, runtime_id, "owner-a", fence, 0, checkpoint1)["version"] == 1

        forged_created = copy.deepcopy(created)
        forged_created["artifact_ref"] = "forged-history-rewrite"
        forged = _checkpoint(
            "c",
            journal_char="d",
            cycles=cycles,
            entries=[forged_created],
            pending_cycle_id=cycle_id,
        )

        with pytest.raises(psycopg2.Error, match="PHASE70_JOURNAL_PREFIX_CONFLICT"):
            _commit(conn, runtime_id, "owner-a", fence, 1, forged)

        assert _read(conn, runtime_id)["version"] == 1
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
    finally:
        conn.close()


def test_historical_checkpoint_retry_never_rolls_back_newer_head():
    runtime_id = _runtime_id("historical-retry")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a", 30)
        fence = lease["fencing_token"]
        first_payload = _checkpoint("d", head_char="d")
        second_payload = _checkpoint("e", head_char="e", journal_char="e")

        first = _commit(conn, runtime_id, "owner-a", fence, 0, first_payload)
        second = _commit(conn, runtime_id, "owner-a", fence, 1, second_payload)
        retry = _commit(conn, runtime_id, "owner-a", fence, 0, first_payload)

        assert first["version"] == 1
        assert second["version"] == 2
        assert retry["status"] == "DUPLICATE_HISTORICAL"
        assert retry["version"] == 1
        assert retry["current_version"] == 2
        assert _read(conn, runtime_id)["checkpoint_id"] == second_payload["checkpoint_id"]
        assert _read(conn, runtime_id)["version"] == 2
    finally:
        conn.close()


def test_checkpoint_id_collision_with_changed_payload_is_fail_closed():
    runtime_id = _runtime_id("id-conflict")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a", 30)
        fence = lease["fencing_token"]
        original = _checkpoint("f")
        assert _commit(conn, runtime_id, "owner-a", fence, 0, original)["version"] == 1

        forged = copy.deepcopy(original)
        forged["runtime_checkpoint"]["shadow_ledger_manifest"]["head_state_id"] = _hash("0")
        with pytest.raises(psycopg2.Error, match="PHASE70_CHECKPOINT_ID_CONFLICT"):
            _commit(conn, runtime_id, "owner-a", fence, 1, forged)

        assert _read(conn, runtime_id)["version"] == 1
    finally:
        conn.close()


def test_service_role_cannot_mutate_append_only_history_tables_directly():
    runtime_id = _runtime_id("privileges")
    conn = _connect()
    try:
        lease = _acquire(conn, runtime_id, "owner-a", 30)
        assert _commit(
            conn,
            runtime_id,
            "owner-a",
            lease["fencing_token"],
            0,
            _checkpoint("1"),
        )["status"] == "COMMITTED"

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_shadow_runtime_checkpoints "
                    "set head_state_id=%s where runtime_id=%s",
                    (_hash("9"), runtime_id),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
