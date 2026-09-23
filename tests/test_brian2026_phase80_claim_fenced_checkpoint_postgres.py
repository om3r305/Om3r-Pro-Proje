"""Real Postgres red-team tests for Phase80 claim-fenced checkpoint commits.

Phase80 SQL is intentionally a draft contract outside supabase/migrations.
CI loads the official Phase70/73/75/76/77/78 migrations, then the Phase79
STARTED draft and finally the Phase80 claim-fenced checkpoint draft.

The tests prove that local paper progress cannot become authoritative after a
worker loses its claim, while an exact lost-response retry remains idempotent.
"""

from __future__ import annotations

import os
import threading
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
MIGRATIONS = (
    ROOT / "supabase" / "migrations" / "202609230720_brian_phase70_durable_runtime_store.sql",
    ROOT / "supabase" / "migrations" / "202609230745_brian_phase73_operational_risk_store.sql",
    ROOT / "supabase" / "migrations" / "202609230845_brian_phase75_atomic_governed_writeahead.sql",
    ROOT / "supabase" / "migrations" / "202609230915_brian_phase76_shadow_execution_outbox.sql",
    ROOT / "supabase" / "migrations" / "202609230945_brian_phase77_execution_claim_lifecycle.sql",
    ROOT / "supabase" / "migrations" / "202609231015_brian_phase78_execution_kill_switch.sql",
)
PHASE79_SQL = ROOT / "brian2026" / "sql" / "phase79_atomic_execution_start.sql"
PHASE80_SQL = ROOT / "brian2026" / "sql" / "phase80_claim_fenced_checkpoint_commit.sql"
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase80 Postgres tests require CI database",
)

_BOOTSTRAP = """
do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin; end if;
end
$$;
"""


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


@pytest.fixture(scope="module", autouse=True)
def _apply_sql():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP)
            for migration in MIGRATIONS:
                cur.execute(migration.read_text(encoding="utf-8"))
            cur.execute(PHASE79_SQL.read_text(encoding="utf-8"))
            cur.execute(PHASE80_SQL.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _rid(label: str) -> str:
    return f"pytest-phase80-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _cycle(cycle_id: str):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": "phase80-plan",
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
    cycle_id: str,
    stage: str,
    entry_char: str,
    previous_char: str | None,
    artifact_char: str,
):
    return {
        "schema_version": "brian.phase66-durable-cycle-journal.v1",
        "sequence": sequence,
        "stage": stage,
        "cycle_id": cycle_id,
        "cycle_hash": _h("c"),
        "previous_entry_id": None if previous_char is None else _h(previous_char),
        "artifact_hash": _h(artifact_char),
        "artifact_ref": f"{stage.lower()}:{sequence}",
        "entry_id": _h(entry_char),
    }


def _checkpoint(
    cycle_id: str,
    *,
    stages: tuple[str, ...],
    checkpoint_char: str,
    journal_char: str,
    head_char: str = "s",
):
    entry_chars = ("e", "f", "1", "2", "3", "4", "5")
    artifact_chars = ("a", "b", "6", "7", "8", "9", "0")
    entries = []
    previous = None
    for sequence, stage in enumerate(stages):
        char = entry_chars[sequence]
        entries.append(
            _entry(
                sequence,
                cycle_id=cycle_id,
                stage=stage,
                entry_char=char,
                previous_char=previous,
                artifact_char=artifact_chars[sequence],
            )
        )
        previous = char

    return {
        "schema_version": "brian.phase67-durable-runtime-orchestrator.v1",
        "runtime_checkpoint": {
            "schema_version": "brian.phase63-crash-recovery.v1",
            "paper": {
                "paper_only": True,
                "live_execution": False,
            },
            "shadow_ledger_manifest": {
                "head_state_id": _h(head_char),
                "pending_cycle_id": None,
            },
            "pending_cycle_id": None,
            "live_execution": False,
        },
        "journal_manifest": {
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "append_only": True,
            "cycles": {cycle_id: _cycle(cycle_id)},
            "entries": entries,
            "journal_hash": _h(journal_char),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h(checkpoint_char),
    }


def _setup(conn, runtime_id: str):
    cycle_id = _h("d")
    dispatch_id = _h("x")
    initial = _checkpoint(
        cycle_id,
        stages=("CYCLE_CREATED",),
        checkpoint_char="k",
        journal_char="j",
    )
    created_entry = initial["journal_manifest"]["entries"][0]
    cycle_payload = initial["journal_manifest"]["cycles"][cycle_id]

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_runtime_heads(
              runtime_id,version,checkpoint_id,checkpoint_payload,
              journal_hash,head_state_id,pending_cycle_id,
              owner_token,fencing_token,acquired_at,lease_until,updated_at
            ) values (
              %s,1,%s,%s,%s,%s,null,
              'owner-a',1,now(),now()+interval '5 minutes',now()
            )
            """,
            (
                runtime_id,
                initial["checkpoint_id"],
                Json(initial),
                initial["journal_manifest"]["journal_hash"],
                initial["runtime_checkpoint"]["shadow_ledger_manifest"]["head_state_id"],
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_runtime_checkpoints(
              runtime_id,version,checkpoint_id,checkpoint_payload,
              journal_hash,head_state_id,pending_cycle_id,
              fencing_token,committed_at
            ) values (%s,1,%s,%s,%s,%s,null,1,now())
            """,
            (
                runtime_id,
                initial["checkpoint_id"],
                Json(initial),
                initial["journal_manifest"]["journal_hash"],
                initial["runtime_checkpoint"]["shadow_ledger_manifest"]["head_state_id"],
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_runtime_cycles(
              runtime_id,cycle_id,cycle_hash,cycle_payload,first_seen_version
            ) values (%s,%s,%s,%s,1)
            """,
            (
                runtime_id,
                cycle_id,
                created_entry["cycle_hash"],
                Json(cycle_payload),
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_runtime_journal_entries(
              runtime_id,sequence,entry_id,previous_entry_id,
              cycle_id,stage,cycle_hash,artifact_hash,artifact_ref,
              entry_payload,first_seen_version
            ) values (%s,0,%s,null,%s,'CYCLE_CREATED',%s,%s,%s,%s,1)
            """,
            (
                runtime_id,
                created_entry["entry_id"],
                cycle_id,
                created_entry["cycle_hash"],
                created_entry["artifact_hash"],
                created_entry["artifact_ref"],
                Json(created_entry),
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_execution_dispatches(
              runtime_id,dispatch_id,cycle_id,governed_result_id,
              policy_fingerprint,authorization_checkpoint_id,
              authorization_runtime_version,risk_version,
              risk_ledger_hash,risk_receipt_id,fencing_token
            ) values (%s,%s,%s,%s,%s,%s,1,1,%s,%s,1)
            """,
            (
                runtime_id,
                dispatch_id,
                cycle_id,
                _h("g"),
                _h("p"),
                initial["checkpoint_id"],
                _h("l"),
                _h("r"),
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_execution_claims(
              runtime_id,dispatch_id,cycle_id,status,
              worker_token,claim_fencing_token,claimed_at,claim_until,
              risk_version_at_decision,risk_receipt_id_at_decision,
              journal_stage_at_decision,resume_only,
              cancel_reason,completed_at,completion_checkpoint_id,updated_at
            ) values (
              %s,%s,%s,'CLAIMED',
              'worker-a',1,now(),now()+interval '5 minutes',
              1,%s,'CYCLE_CREATED',false,
              null,null,null,now()
            )
            """,
            (runtime_id, dispatch_id, cycle_id, _h("r")),
        )
        cur.execute(
            """
            insert into public.brian_shadow_execution_starts(
              runtime_id,dispatch_id,cycle_id,worker_token,
              claim_fencing_token,runtime_fencing_token,
              runtime_version_at_start,risk_version_at_start,
              risk_receipt_id_at_start,risk_state_at_start,
              journal_stage_at_start,phase78_status_at_start,
              cancel_requested_at_start,cancel_reason_at_start,
              resume_only,started_at
            ) values (
              %s,%s,%s,'worker-a',
              1,1,1,1,%s,'ACTIVE',
              'CYCLE_CREATED','PROCEED',
              false,null,false,now()
            )
            """,
            (runtime_id, dispatch_id, cycle_id, _h("r")),
        )

    return cycle_id, dispatch_id, initial


def _commit(
    conn,
    runtime_id: str,
    cycle_id: str,
    checkpoint: dict,
    *,
    worker="worker-a",
    claim_fence=1,
    expected_version=1,
):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_claimed_shadow_runtime_checkpoint("
            "%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                1,
                cycle_id,
                worker,
                claim_fence,
                expected_version,
                Json(checkpoint),
            ),
        )
        return cur.fetchone()[0]


def _read_head(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            "select version,checkpoint_id,checkpoint_payload "
            "from public.brian_shadow_runtime_heads where runtime_id=%s",
            (runtime_id,),
        )
        return cur.fetchone()


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def _paper_applied(cycle_id: str, char="m"):
    return _checkpoint(
        cycle_id,
        stages=("CYCLE_CREATED", "PAPER_APPLIED"),
        checkpoint_char=char,
        journal_char="n",
    )


def test_current_claim_can_make_post_started_checkpoint_authoritative():
    runtime_id = _rid("commit")
    conn = _connect()
    try:
        cycle_id, _, _ = _setup(conn, runtime_id)
        checkpoint = _paper_applied(cycle_id)
        row = _commit(conn, runtime_id, cycle_id, checkpoint)

        assert row["committed"] is True
        assert row["duplicate"] is False
        assert row["status"] == "COMMITTED"
        assert row["version"] == 2
        assert row["current_version"] == 2
        assert row["journal_stage"] == "PAPER_APPLIED"

        version, checkpoint_id, payload = _read_head(conn, runtime_id)
        assert version == 2
        assert checkpoint_id == checkpoint["checkpoint_id"]
        assert payload == checkpoint
        assert _count(conn, "brian_shadow_claim_commit_events", runtime_id) == 1
    finally:
        conn.close()


def test_exact_lost_response_retry_is_duplicate_current_for_same_live_claim():
    runtime_id = _rid("retry")
    conn = _connect()
    try:
        cycle_id, _, _ = _setup(conn, runtime_id)
        checkpoint = _paper_applied(cycle_id)
        first = _commit(conn, runtime_id, cycle_id, checkpoint)
        retry = _commit(conn, runtime_id, cycle_id, checkpoint)

        assert first["status"] == "COMMITTED"
        assert retry["committed"] is True
        assert retry["duplicate"] is True
        assert retry["status"] == "DUPLICATE_CURRENT"
        assert retry["version"] == 2
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 2
    finally:
        conn.close()


def test_concurrent_exact_commit_has_one_commit_and_one_duplicate_current():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        cycle_id, _, _ = _setup(setup, runtime_id)
        checkpoint = _paper_applied(cycle_id)
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(_):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _commit(conn, runtime_id, cycle_id, checkpoint)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "COMMITTED" for row in rows) == 1
    assert sum(row["status"] == "DUPLICATE_CURRENT" for row in rows) == 1

    conn = _connect()
    try:
        assert _read_head(conn, runtime_id)[0] == 2
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 2
    finally:
        conn.close()


def test_old_worker_cannot_commit_after_claim_takeover():
    runtime_id = _rid("takeover")
    conn = _connect()
    try:
        cycle_id, dispatch_id, _ = _setup(conn, runtime_id)
        checkpoint = _paper_applied(cycle_id)
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.brian_shadow_execution_claims
                set worker_token='worker-b',
                    claim_fencing_token=2,
                    claimed_at=now(),
                    claim_until=now()+interval '5 minutes',
                    updated_at=now()
                where runtime_id=%s and dispatch_id=%s
                """,
                (runtime_id, dispatch_id),
            )

        stale = _commit(
            conn,
            runtime_id,
            cycle_id,
            checkpoint,
            worker="worker-a",
            claim_fence=1,
        )
        assert stale["committed"] is False
        assert stale["status"] == "CLAIM_LOST"
        assert _read_head(conn, runtime_id)[0] == 1
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
    finally:
        conn.close()


def test_lost_response_retry_is_rejected_if_claim_changed_after_original_commit():
    runtime_id = _rid("retry-takeover")
    conn = _connect()
    try:
        cycle_id, dispatch_id, _ = _setup(conn, runtime_id)
        checkpoint = _paper_applied(cycle_id)
        assert _commit(conn, runtime_id, cycle_id, checkpoint)["status"] == "COMMITTED"

        with conn.cursor() as cur:
            cur.execute(
                """
                update public.brian_shadow_execution_claims
                set worker_token='worker-b',
                    claim_fencing_token=2,
                    claimed_at=now(),
                    claim_until=now()+interval '5 minutes',
                    updated_at=now()
                where runtime_id=%s and dispatch_id=%s
                """,
                (runtime_id, dispatch_id),
            )

        stale_retry = _commit(
            conn,
            runtime_id,
            cycle_id,
            checkpoint,
            worker="worker-a",
            claim_fence=1,
            expected_version=1,
        )
        assert stale_retry["committed"] is False
        assert stale_retry["status"] == "CLAIM_LOST"
        assert _read_head(conn, runtime_id)[0] == 2
    finally:
        conn.close()


def test_missing_started_boundary_blocks_authoritative_commit():
    runtime_id = _rid("start-missing")
    conn = _connect()
    try:
        cycle_id, dispatch_id, _ = _setup(conn, runtime_id)
        with conn.cursor() as cur:
            cur.execute(
                "delete from public.brian_shadow_execution_starts "
                "where runtime_id=%s and dispatch_id=%s",
                (runtime_id, dispatch_id),
            )

        checkpoint = _paper_applied(cycle_id)
        row = _commit(conn, runtime_id, cycle_id, checkpoint)
        assert row["committed"] is False
        assert row["status"] == "START_MISSING"
        assert _read_head(conn, runtime_id)[0] == 1
    finally:
        conn.close()


def test_different_checkpoint_from_stale_runtime_version_is_cas_conflict():
    runtime_id = _rid("cas")
    conn = _connect()
    try:
        cycle_id, _, _ = _setup(conn, runtime_id)
        first = _paper_applied(cycle_id, "m")
        second = _paper_applied(cycle_id, "q")
        assert _commit(conn, runtime_id, cycle_id, first)["status"] == "COMMITTED"

        conflict = _commit(
            conn,
            runtime_id,
            cycle_id,
            second,
            expected_version=1,
        )
        assert conflict["committed"] is False
        assert conflict["status"] == "RUNTIME_VERSION_CONFLICT"
        assert conflict["current_version"] == 2
        assert _read_head(conn, runtime_id)[1] == first["checkpoint_id"]
    finally:
        conn.close()


def test_service_role_cannot_write_claim_commit_event_history_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        cycle_id, _, _ = _setup(conn, runtime_id)
        assert _commit(
            conn,
            runtime_id,
            cycle_id,
            _paper_applied(cycle_id),
        )["status"] == "COMMITTED"

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    insert into public.brian_shadow_claim_commit_events(
                      runtime_id,dispatch_id,cycle_id,worker_token,
                      claim_fencing_token,runtime_fencing_token,event
                    ) values (%s,%s,%s,'attacker',1,1,'COMMITTED')
                    """,
                    (runtime_id, _h("x"), cycle_id),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
