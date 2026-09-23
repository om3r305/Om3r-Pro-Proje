"""Real Postgres tests for Phase87 durable recovery backlog / restart view."""

from __future__ import annotations

import os
import uuid
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
    ROOT / "supabase" / "migrations" / "202609230815_brian_phase74_governed_cycle_binding.sql",
    ROOT / "supabase" / "migrations" / "202609230845_brian_phase75_atomic_governed_writeahead.sql",
    ROOT / "supabase" / "migrations" / "202609230915_brian_phase76_shadow_execution_outbox.sql",
    ROOT / "supabase" / "migrations" / "202609230945_brian_phase77_execution_claim_lifecycle.sql",
    ROOT / "supabase" / "migrations" / "202609231015_brian_phase78_execution_kill_switch.sql",
)
DRAFTS = tuple(
    ROOT / "brian2026" / "sql" / name
    for name in (
        "phase79_atomic_execution_start.sql",
        "phase80_claim_fenced_checkpoint_commit.sql",
        "phase81_cancel_recovery_directive.sql",
        "phase82_recovery_claim_fencing.sql",
        "phase83_atomic_recovery_start.sql",
        "phase84_recovery_execution_checkpoint.sql",
        "phase85_recovery_completion_audit.sql",
        "phase86_recovery_admission_interlock.sql",
        "phase87_recovery_restart_resume.sql",
    )
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase87 Postgres tests require CI database",
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
            for draft in DRAFTS:
                cur.execute(draft.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _rid(label: str) -> str:
    return f"pytest-phase87-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _checkpoint(*, checkpoint_char="k", head_char="h", cycle_id=None, stage=None):
    cycles = {}
    entries = []
    if cycle_id is not None:
        cycles[cycle_id] = {
            "schema_version": "brian.phase57-shadow-execution-cycle.v1",
            "source_plan_id": _h("p"),
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
        entries.append({
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "sequence": 0,
            "stage": "CYCLE_CREATED",
            "cycle_id": cycle_id,
            "cycle_hash": _h("c"),
            "previous_entry_id": None,
            "artifact_hash": _h("a"),
            "artifact_ref": "phase87:cycle",
            "entry_id": _h("e"),
        })
        if stage and stage != "CYCLE_CREATED":
            entries.append({
                "schema_version": "brian.phase66-durable-cycle-journal.v1",
                "sequence": 1,
                "stage": stage,
                "cycle_id": cycle_id,
                "cycle_hash": _h("c"),
                "previous_entry_id": _h("e"),
                "artifact_hash": _h("b"),
                "artifact_ref": f"phase87:{stage}",
                "entry_id": _h("f"),
            })
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
                "checkpoint_id": _h("q"),
            },
            "shadow_ledger_manifest": {
                "schema_version": "brian.phase60-shadow-state-ledger.v1",
                "append_only": True,
                "head_state_id": _h(head_char),
                "pending_cycle_id": None,
                "states": {
                    _h(head_char): {
                        "state_id": _h(head_char),
                        "account_id": "paper-acct",
                    }
                },
                "transitions": [],
                "ledger_hash": _h("l"),
                "shadow_only": True,
                "live_execution": False,
            },
            "pending_cycle_id": None,
            "live_execution": False,
            "checkpoint_id": _h("r"),
        },
        "journal_manifest": {
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "append_only": True,
            "cycles": cycles,
            "entries": entries,
            "journal_hash": _h("j"),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h(checkpoint_char),
    }


def _runtime(conn, runtime_id, *, version=10, checkpoint=None):
    checkpoint = checkpoint or _checkpoint()
    head = checkpoint["runtime_checkpoint"]["shadow_ledger_manifest"]["head_state_id"]
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_runtime_heads(
              runtime_id,version,checkpoint_id,checkpoint_payload,
              journal_hash,head_state_id,pending_cycle_id,
              owner_token,fencing_token,acquired_at,lease_until,updated_at
            ) values (
              %s,%s,%s,%s,%s,%s,null,
              'owner-a',1,now(),now()+interval '5 minutes',now()
            )
            """,
            (
                runtime_id,
                version,
                checkpoint["checkpoint_id"],
                Json(checkpoint),
                checkpoint["journal_manifest"]["journal_hash"],
                head,
            ),
        )


def _cancel(
    conn,
    runtime_id,
    *,
    cycle_char="o",
    dispatch_char="d",
    receipt_char="r",
    requested_offset="0 seconds",
):
    row = {
        "cycle_id": _h(cycle_char),
        "dispatch_id": _h(dispatch_char),
        "receipt_id": _h(receipt_char),
    }
    with conn.cursor() as cur:
        cur.execute(
            f"""
            insert into public.brian_shadow_execution_cancel_requests(
              runtime_id,dispatch_id,cycle_id,
              risk_version,risk_receipt_id,journal_stage,
              phase,reason,requested_at,
              worker_token,claim_fencing_token
            ) values (
              %s,%s,%s,
              7,%s,'PAPER_APPLIED',
              'AFTER_START','REDUCING_NEW_RISK',
              now()+interval '{requested_offset}',
              'worker-original',1
            )
            """,
            (
                runtime_id,
                row["dispatch_id"],
                row["cycle_id"],
                row["receipt_id"],
            ),
        )
    return row


def _leg():
    return {
        "asset_id": "BTCUSDT",
        "before_weight": 0.10,
        "current_weight": 0.25,
        "target_weight": 0.10,
        "reduce_weight": 0.15,
        "current_direction": 1,
        "order_direction": -1,
        "reduce_only": True,
    }


def _directive(conn, runtime_id, cancel, *, status="READY_REDUCE_ONLY"):
    legs = [] if status == "NO_RECOVERY_REQUIRED" else [_leg()]
    unsafe = (
        [{"asset_id": "BTCUSDT", "reason": "ROLLBACK_NOT_REDUCE_ONLY"}]
        if status == "MANUAL_REVIEW"
        else []
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_cancel_recovery_directives(
              runtime_id,dispatch_id,cycle_id,
              cancel_risk_version,cancel_risk_receipt_id,cancel_reason,
              source_runtime_version,pre_state_id,current_state_id,
              current_risk_version,current_risk_receipt_id,current_risk_state,
              recovery_status,recovery_legs,unsafe_assets,prepared_at
            )
            select %s,%s,%s,
                   7,%s,'REDUCING_NEW_RISK',
                   version,%s,head_state_id,
                   8,%s,'REDUCING',
                   %s,%s,%s,now()
            from public.brian_shadow_runtime_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                cancel["dispatch_id"],
                cancel["cycle_id"],
                cancel["receipt_id"],
                _h("a"),
                _h("s"),
                status,
                Json(legs),
                Json(unsafe),
                runtime_id,
            ),
        )


def _claim(
    conn,
    runtime_id,
    cancel,
    *,
    expired=False,
    worker="recovery-a",
    fence=3,
    status="CLAIMED",
):
    until = "now()-interval '1 second'" if expired else "now()+interval '5 minutes'"
    with conn.cursor() as cur:
        cur.execute(
            f"""
            insert into public.brian_shadow_recovery_claims(
              runtime_id,dispatch_id,cycle_id,cancel_risk_receipt_id,
              status,worker_token,claim_fencing_token,
              claimed_at,claim_until,
              runtime_version_at_claim,head_state_id_at_claim,
              risk_version_at_claim,risk_receipt_id_at_claim,risk_state_at_claim,
              updated_at
            )
            select %s,%s,%s,%s,
                   %s,%s,%s,
                   now(),{until},
                   version,head_state_id,
                   8,%s,'REDUCING',
                   now()
            from public.brian_shadow_runtime_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                cancel["dispatch_id"],
                cancel["cycle_id"],
                cancel["receipt_id"],
                status,
                worker,
                fence,
                _h("s"),
                runtime_id,
            ),
        )


def _start(conn, runtime_id, cancel, *, fence=3):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_recovery_starts(
              runtime_id,dispatch_id,cycle_id,cancel_risk_receipt_id,
              worker_token,recovery_claim_fencing_token,
              runtime_fencing_token,runtime_version_at_start,
              head_state_id_at_start,
              risk_version_at_start,risk_receipt_id_at_start,risk_state_at_start,
              directive_source_runtime_version,directive_current_state_id,
              recovery_legs,started_at
            )
            select %s,%s,%s,%s,
                   'recovery-a',%s,
                   1,version,head_state_id,
                   8,%s,'REDUCING',
                   version,head_state_id,
                   %s,now()
            from public.brian_shadow_runtime_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                cancel["dispatch_id"],
                cancel["cycle_id"],
                cancel["receipt_id"],
                fence,
                _h("s"),
                Json([_leg()]),
                runtime_id,
            ),
        )


def _set_progress(
    conn,
    runtime_id,
    cancel,
    *,
    stage,
    recovery_cycle_char="y",
    checkpoint_char="p",
):
    recovery_cycle_id = _h(recovery_cycle_char)
    checkpoint = _checkpoint(
        checkpoint_char=checkpoint_char,
        head_char="z" if stage == "COMMITTED" else "h",
        cycle_id=recovery_cycle_id,
        stage=stage,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            update public.brian_shadow_runtime_heads
            set version=version+1,
                checkpoint_id=%s,
                checkpoint_payload=%s,
                journal_hash=%s,
                head_state_id=%s,
                updated_at=now()
            where runtime_id=%s
            returning version,head_state_id
            """,
            (
                checkpoint["checkpoint_id"],
                Json(checkpoint),
                checkpoint["journal_manifest"]["journal_hash"],
                checkpoint["runtime_checkpoint"]["shadow_ledger_manifest"]["head_state_id"],
                runtime_id,
            ),
        )
        version, head_state_id = cur.fetchone()
        cur.execute(
            """
            update public.brian_shadow_recovery_claims
            set recovery_cycle_id=%s,
                progress_runtime_version=%s,
                progress_head_state_id=%s,
                progress_checkpoint_id=%s,
                updated_at=now()
            where runtime_id=%s
              and dispatch_id=%s
              and cancel_risk_receipt_id=%s
            """,
            (
                recovery_cycle_id,
                version,
                head_state_id,
                checkpoint["checkpoint_id"],
                runtime_id,
                cancel["dispatch_id"],
                cancel["receipt_id"],
            ),
        )
    return recovery_cycle_id, checkpoint


def _terminal_event(conn, runtime_id, cancel, recovery_cycle_id, checkpoint):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_recovery_commit_events(
              runtime_id,dispatch_id,original_cycle_id,recovery_cycle_id,
              cancel_risk_receipt_id,worker_token,
              recovery_claim_fencing_token,runtime_fencing_token,
              expected_runtime_version,committed_runtime_version,
              checkpoint_id,recovery_journal_stage,
              event,observed_at
            )
            select %s,%s,%s,%s,
                   %s,'recovery-a',
                   3,1,
                   version-1,version,
                   %s,'COMMITTED',
                   'RECOVERY_COMMITTED_PENDING_AUDIT',now()
            from public.brian_shadow_runtime_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                cancel["dispatch_id"],
                cancel["cycle_id"],
                recovery_cycle_id,
                cancel["receipt_id"],
                checkpoint["checkpoint_id"],
                runtime_id,
            ),
        )


def _certificate(conn, runtime_id, cancel, recovery_cycle_id, checkpoint):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_recovery_completion_certificates(
              runtime_id,dispatch_id,original_cycle_id,recovery_cycle_id,
              cancel_risk_receipt_id,completion_runtime_version,
              completion_checkpoint_id,start_head_state_id,final_head_state_id,
              paper_checkpoint_id,recovery_claim_fencing_token,
              leg_audits,recovery_fill_count,certified_at
            )
            select %s,%s,%s,%s,
                   %s,version,
                   %s,%s,head_state_id,
                   %s,3,
                   %s,1,now()
            from public.brian_shadow_runtime_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                cancel["dispatch_id"],
                cancel["cycle_id"],
                recovery_cycle_id,
                cancel["receipt_id"],
                checkpoint["checkpoint_id"],
                _h("h"),
                _h("q"),
                Json([{"asset_id": "BTCUSDT", "safe": True}]),
                runtime_id,
            ),
        )


def _read(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_read_next_shadow_recovery_work(%s)",
            (runtime_id,),
        )
        return cur.fetchone()[0]


def test_idle_without_unresolved_after_start_cancel():
    runtime_id = _rid("idle")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        row = _read(conn, runtime_id)
        assert row == {
            "has_work": False,
            "status": "IDLE",
            "runtime_id": runtime_id,
        }
    finally:
        conn.close()


def test_backlog_moves_from_directive_to_claim_to_start():
    runtime_id = _rid("early")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)

        row = _read(conn, runtime_id)
        assert row["work_state"] == "NEEDS_DIRECTIVE"
        assert row["directive_exists"] is False

        _directive(conn, runtime_id, cancel)
        row = _read(conn, runtime_id)
        assert row["work_state"] == "NEEDS_CLAIM"
        assert row["directive_exists"] is True

        _claim(conn, runtime_id, cancel)
        row = _read(conn, runtime_id)
        assert row["work_state"] == "NEEDS_START"
        assert row["claim_worker_token"] == "recovery-a"
        assert row["claim_fencing_token"] == 3

        _start(conn, runtime_id, cancel)
        row = _read(conn, runtime_id)
        assert row["work_state"] == "STARTED_NEEDS_EXECUTION"
        assert row["started"] is True
    finally:
        conn.close()


def test_expired_claim_is_explicit_takeover_work():
    runtime_id = _rid("expired")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)
        _directive(conn, runtime_id, cancel)
        _claim(conn, runtime_id, cancel, expired=True)

        row = _read(conn, runtime_id)
        assert row["work_state"] == "CLAIM_EXPIRED"
        assert row["claim_status"] == "CLAIMED"
    finally:
        conn.close()


def test_manual_review_is_never_auto_hidden():
    runtime_id = _rid("manual")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)
        _directive(conn, runtime_id, cancel, status="MANUAL_REVIEW")
        row = _read(conn, runtime_id)
        assert row["work_state"] == "MANUAL_REVIEW"
        assert row["recovery_status"] == "MANUAL_REVIEW"
    finally:
        conn.close()


def test_no_recovery_required_removes_item_from_backlog():
    runtime_id = _rid("no-recovery")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)
        _directive(conn, runtime_id, cancel, status="NO_RECOVERY_REQUIRED")
        assert _read(conn, runtime_id)["work_state"] == "IDLE"
    finally:
        conn.close()


def test_progress_and_terminal_commit_are_distinguished_for_restart():
    runtime_id = _rid("progress")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)
        _directive(conn, runtime_id, cancel)
        _claim(conn, runtime_id, cancel)
        _start(conn, runtime_id, cancel)

        recovery_cycle_id, checkpoint = _set_progress(
            conn,
            runtime_id,
            cancel,
            stage="PAPER_APPLIED",
        )
        row = _read(conn, runtime_id)
        assert row["work_state"] == "RECOVERY_PROGRESS"
        assert row["recovery_cycle_id"] == recovery_cycle_id
        assert row["recovery_journal_stage"] == "PAPER_APPLIED"
        assert row["phase84_terminal_event"] is False

        recovery_cycle_id, checkpoint = _set_progress(
            conn,
            runtime_id,
            cancel,
            stage="COMMITTED",
            recovery_cycle_char="y",
            checkpoint_char="u",
        )
        _terminal_event(
            conn,
            runtime_id,
            cancel,
            recovery_cycle_id,
            checkpoint,
        )
        row = _read(conn, runtime_id)
        assert row["work_state"] == "NEEDS_AUDIT"
        assert row["recovery_journal_stage"] == "COMMITTED"
        assert row["phase84_terminal_event"] is True
        assert row["progress_checkpoint_id"] == checkpoint["checkpoint_id"]
    finally:
        conn.close()


def test_certificate_resolves_backlog_to_idle():
    runtime_id = _rid("cert")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)
        _directive(conn, runtime_id, cancel)
        _claim(conn, runtime_id, cancel)
        _start(conn, runtime_id, cancel)
        recovery_cycle_id, checkpoint = _set_progress(
            conn,
            runtime_id,
            cancel,
            stage="COMMITTED",
        )
        _terminal_event(conn, runtime_id, cancel, recovery_cycle_id, checkpoint)
        assert _read(conn, runtime_id)["work_state"] == "NEEDS_AUDIT"

        _certificate(conn, runtime_id, cancel, recovery_cycle_id, checkpoint)
        assert _read(conn, runtime_id)["work_state"] == "IDLE"
    finally:
        conn.close()


def test_completed_claim_without_certificate_is_fail_closed_visible_work():
    runtime_id = _rid("broken-complete")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)
        _directive(conn, runtime_id, cancel)
        _claim(conn, runtime_id, cancel)
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.brian_shadow_recovery_claims
                set status='COMPLETED',
                    claim_until=null,
                    completed_at=now(),
                    completion_ref=%s,
                    updated_at=now()
                where runtime_id=%s
                """,
                (_h("p"), runtime_id),
            )
        row = _read(conn, runtime_id)
        assert row["work_state"] == "COMPLETED_WITHOUT_CERTIFICATE"
        assert row["has_work"] is True
    finally:
        conn.close()


def test_oldest_unresolved_cancel_is_selected_deterministically():
    runtime_id = _rid("ordering")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        older = _cancel(
            conn,
            runtime_id,
            cycle_char="a",
            dispatch_char="b",
            receipt_char="c",
            requested_offset="-10 seconds",
        )
        _cancel(
            conn,
            runtime_id,
            cycle_char="x",
            dispatch_char="y",
            receipt_char="z",
            requested_offset="0 seconds",
        )
        row = _read(conn, runtime_id)
        assert row["original_cycle_id"] == older["cycle_id"]
        assert row["dispatch_id"] == older["dispatch_id"]
    finally:
        conn.close()


def test_service_role_can_read_backlog_but_anon_cannot_execute_reader():
    runtime_id = _rid("permissions")
    conn = _connect()
    try:
        _runtime(conn, runtime_id)
        _cancel(conn, runtime_id)

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            cur.execute(
                "select public.brian_read_next_shadow_recovery_work(%s)",
                (runtime_id,),
            )
            assert cur.fetchone()[0]["has_work"] is True
            cur.execute("reset role")

            cur.execute("set role anon")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "select public.brian_read_next_shadow_recovery_work(%s)",
                    (runtime_id,),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
