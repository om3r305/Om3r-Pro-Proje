"""Real Postgres tests for Phase86 unresolved-recovery admission interlock."""

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
    )
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase86 Postgres tests require CI database",
)
TS = 1_760_000_000.0

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
    return f"pytest-phase86-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _cycle(cycle_id: str):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": f"plan-{cycle_id[:8]}",
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


def _runtime_checkpoint(
    checkpoint_char: str,
    *,
    cycle_id: str | None = None,
    head_char: str = "a",
    terminal_stage: str | None = None,
):
    cycles = {}
    entries = []
    if cycle_id is not None:
        cycles[cycle_id] = _cycle(cycle_id)
        entries = [{
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "sequence": 0,
            "stage": "CYCLE_CREATED",
            "cycle_id": cycle_id,
            "cycle_hash": _h("c"),
            "previous_entry_id": None,
            "artifact_hash": _h("d"),
            "artifact_ref": f"cycle:{cycle_id}",
            "entry_id": _h("e"),
        }]
        if terminal_stage is not None:
            entries.append({
                "schema_version": "brian.phase66-durable-cycle-journal.v1",
                "sequence": 1,
                "stage": terminal_stage,
                "cycle_id": cycle_id,
                "cycle_hash": _h("c"),
                "previous_entry_id": _h("e"),
                "artifact_hash": _h("f"),
                "artifact_ref": f"terminal:{terminal_stage}",
                "entry_id": _h("0"),
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
                "checkpoint_id": _h("p"),
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
            "journal_hash": _h(
                "j" if cycle_id is None else ("m" if terminal_stage is not None else "k")
            ),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h(checkpoint_char),
    }


def _risk_entry(seq, entry_char, receipt_char, prev, ts):
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "sequence": seq,
        "previous_entry_id": None if prev is None else _h(prev),
        "policy_hash": _h("q"),
        "receipt": {
            "schema_version": "brian.phase68-operational-risk-governor.v1",
            "timestamp": ts,
            "previous_state": "ACTIVE",
            "trading_state": "ACTIVE",
            "recommended_state": "ACTIVE",
            "reasons": [],
            "max_drawdown_fraction": 0.0,
            "window_loss_fraction": 0.0,
            "qualifying_stoplosses": 0,
            "stoploss_lock_until": None,
            "blocked_assets": [],
            "consecutive_execution_failures": 0,
            "reconciliation_failures": 0,
            "unknown_order_outcomes": 0,
            "market_data_age_seconds": 0.0,
            "manual_halt": False,
            "manual_release_requested": False,
            "halt_latched": False,
            "receipt_id": _h(receipt_char),
            "execution_failure_lock_until": None,
            "reconciliation_failure_lock_until": None,
            "asset_cooldown_until": [],
            "shadow_only": True,
            "live_execution": False,
        },
        "entry_id": _h(entry_char),
    }


def _risk_manifest(char="1"):
    entries = [_risk_entry(0, "a", "b", None, TS)]
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "append_only": True,
        "policy": {"max_drawdown_fraction": 0.5, "max_daily_loss_fraction": 0.5},
        "policy_hash": _h("q"),
        "initial_state": "ACTIVE",
        "entries": entries,
        "entry_count": 1,
        "head_entry_id": entries[-1]["entry_id"],
        "current_state": "ACTIVE",
        "halt_latched": False,
        "ledger_hash": _h(char),
        "shadow_only": True,
        "live_execution": False,
    }


def _acquire(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_acquire_shadow_runtime_lease(%s,%s,%s)",
            (runtime_id, "owner-a", 30),
        )
        return cur.fetchone()[0]


def _commit_runtime(conn, runtime_id, fence, version, checkpoint):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_shadow_runtime_checkpoint(%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", fence, version, Json(checkpoint)),
        )
        return cur.fetchone()[0]


def _commit_risk(conn, runtime_id, fence, version, manifest):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_operational_risk_ledger(%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", fence, version, Json(manifest)),
        )
        return cur.fetchone()[0]


def _authorize(conn, runtime_id, fence, runtime_version, risk_version, manifest, cycle_id):
    checkpoint = _runtime_checkpoint("4", cycle_id=cycle_id)
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_authorize_and_persist_governed_cycle("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                fence,
                runtime_version,
                risk_version,
                manifest["ledger_hash"],
                manifest["entries"][-1]["receipt"]["receipt_id"],
                cycle_id,
                _h("g"),
                _h("p"),
                Json(checkpoint),
            ),
        )
        return cur.fetchone()[0]


def _submit(conn, runtime_id, fence, cycle_id, dispatch_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_submit_shadow_execution_dispatch(%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", fence, cycle_id, dispatch_id),
        )
        return cur.fetchone()[0]


def _bootstrap(conn, runtime_id):
    lease = _acquire(conn, runtime_id)
    fence = lease["fencing_token"]
    runtime = _commit_runtime(
        conn,
        runtime_id,
        fence,
        0,
        _runtime_checkpoint("1"),
    )
    manifest = _risk_manifest("2")
    risk = _commit_risk(conn, runtime_id, fence, 0, manifest)
    return lease, runtime, risk, manifest


def _cancel(
    conn,
    runtime_id,
    *,
    original_cycle_id="o" * 64,
    phase="AFTER_START",
    reason="HALTED",
    receipt_id="r" * 64,
    dispatch_id="x" * 64,
):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_execution_cancel_requests(
              runtime_id,dispatch_id,cycle_id,
              risk_version,risk_receipt_id,journal_stage,
              phase,reason,requested_at,
              worker_token,claim_fencing_token
            ) values (
              %s,%s,%s,
              1,%s,'PAPER_APPLIED',
              %s,%s,now(),
              'worker-original',1
            )
            """,
            (
                runtime_id,
                dispatch_id,
                original_cycle_id,
                receipt_id,
                phase,
                reason,
            ),
        )
    return {
        "original_cycle_id": original_cycle_id,
        "dispatch_id": dispatch_id,
        "receipt_id": receipt_id,
    }


def _directive(
    conn,
    runtime_id,
    cancel,
    *,
    recovery_status,
):
    unsafe = (
        [{"asset_id": "BTCUSDT", "reason": "ROLLBACK_NOT_REDUCE_ONLY"}]
        if recovery_status == "MANUAL_REVIEW"
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
                   1,%s,'HALTED',
                   version,%s,head_state_id,
                   1,%s,'ACTIVE',
                   %s,'[]'::jsonb,%s,now()
            from public.brian_shadow_runtime_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                cancel["dispatch_id"],
                cancel["original_cycle_id"],
                cancel["receipt_id"],
                _h("a"),
                _h("s"),
                recovery_status,
                Json(unsafe),
                runtime_id,
            ),
        )


def _certificate(conn, runtime_id, cancel):
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
                   checkpoint_id,%s,head_state_id,
                   %s,1,
                   '[]'::jsonb,1,now()
            from public.brian_shadow_runtime_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                cancel["dispatch_id"],
                cancel["original_cycle_id"],
                _h("y"),
                cancel["receipt_id"],
                _h("a"),
                _h("p"),
                runtime_id,
            ),
        )


def _admission(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_read_shadow_recovery_admission(%s)",
            (runtime_id,),
        )
        return cur.fetchone()[0]


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_reader_is_open_without_after_start_cancel_and_before_execution_cancel_does_not_block():
    runtime_id = _rid("reader-open")
    conn = _connect()
    try:
        _bootstrap(conn, runtime_id)
        assert _admission(conn, runtime_id)["status"] == "OPEN"
        _cancel(conn, runtime_id, phase="BEFORE_EXECUTION")
        row = _admission(conn, runtime_id)
        assert row["blocked"] is False
        assert row["status"] == "OPEN"
    finally:
        conn.close()


def test_after_start_cancel_blocks_until_no_recovery_or_certificate_resolves_it():
    for resolution in ("NO_RECOVERY_REQUIRED", "CERTIFICATE"):
        runtime_id = _rid(f"resolve-{resolution.lower()}")
        conn = _connect()
        try:
            _bootstrap(conn, runtime_id)
            cancel = _cancel(conn, runtime_id)
            blocked = _admission(conn, runtime_id)
            assert blocked["blocked"] is True
            assert blocked["status"] == "RECOVERY_BARRIER"
            assert blocked["original_cycle_id"] == cancel["original_cycle_id"]

            if resolution == "NO_RECOVERY_REQUIRED":
                _directive(
                    conn,
                    runtime_id,
                    cancel,
                    recovery_status="NO_RECOVERY_REQUIRED",
                )
            else:
                _certificate(conn, runtime_id, cancel)

            opened = _admission(conn, runtime_id)
            assert opened["blocked"] is False
            assert opened["status"] == "OPEN"
        finally:
            conn.close()


def test_manual_review_directive_keeps_barrier_closed():
    runtime_id = _rid("manual")
    conn = _connect()
    try:
        _bootstrap(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)
        _directive(conn, runtime_id, cancel, recovery_status="MANUAL_REVIEW")
        row = _admission(conn, runtime_id)
        assert row["blocked"] is True
        assert row["status"] == "RECOVERY_BARRIER"
    finally:
        conn.close()


def test_unresolved_recovery_blocks_new_phase75_authorization_without_runtime_advance():
    runtime_id = _rid("auth-block")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        _cancel(conn, runtime_id)
        cycle_id = _h("3")
        row = _authorize(
            conn,
            runtime_id,
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest,
            cycle_id,
        )
        assert row["authorized"] is False
        assert row["status"] == "RECOVERY_BARRIER"
        assert _count(conn, "brian_governed_cycle_authorizations", runtime_id) == 0
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
        assert _count(conn, "brian_shadow_recovery_admission_events", runtime_id) == 1
    finally:
        conn.close()


def test_phase75_exact_retry_remains_idempotent_after_barrier_appears():
    runtime_id = _rid("auth-duplicate")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cycle_id = _h("4")
        first = _authorize(
            conn,
            runtime_id,
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest,
            cycle_id,
        )
        assert first["status"] == "AUTHORIZED_AND_PERSISTED"
        _cancel(conn, runtime_id)

        retry = _authorize(
            conn,
            runtime_id,
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest,
            cycle_id,
        )
        assert retry["authorized"] is True
        assert retry["duplicate"] is True
        assert retry["status"] == "DUPLICATE_CURRENT"
    finally:
        conn.close()


def test_pre_authorized_cycle_is_blocked_at_first_dispatch_when_recovery_barrier_appears():
    runtime_id = _rid("dispatch-block")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cycle_id = _h("5")
        auth = _authorize(
            conn,
            runtime_id,
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest,
            cycle_id,
        )
        assert auth["authorized"] is True
        _cancel(conn, runtime_id)

        row = _submit(
            conn,
            runtime_id,
            lease["fencing_token"],
            cycle_id,
            _h("6"),
        )
        assert row["submitted"] is False
        assert row["status"] == "RECOVERY_BARRIER"
        assert _count(conn, "brian_shadow_execution_dispatches", runtime_id) == 0
        assert _count(conn, "brian_shadow_recovery_admission_events", runtime_id) == 1
    finally:
        conn.close()


def test_existing_dispatch_duplicate_remains_readable_after_barrier_appears():
    runtime_id = _rid("dispatch-duplicate")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cycle_id = _h("7")
        assert _authorize(
            conn,
            runtime_id,
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest,
            cycle_id,
        )["authorized"] is True
        dispatch_id = _h("8")
        assert _submit(
            conn,
            runtime_id,
            lease["fencing_token"],
            cycle_id,
            dispatch_id,
        )["status"] == "SUBMITTED"
        _cancel(conn, runtime_id)

        retry = _submit(
            conn,
            runtime_id,
            lease["fencing_token"],
            cycle_id,
            dispatch_id,
        )
        assert retry["submitted"] is True
        assert retry["duplicate"] is True
        assert retry["status"] == "DUPLICATE_CURRENT"
    finally:
        conn.close()


def test_resolved_barrier_reopens_phase75_authorization():
    runtime_id = _rid("reopen")
    conn = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(conn, runtime_id)
        cancel = _cancel(conn, runtime_id)
        _directive(
            conn,
            runtime_id,
            cancel,
            recovery_status="NO_RECOVERY_REQUIRED",
        )
        cycle_id = _h("9")
        row = _authorize(
            conn,
            runtime_id,
            lease["fencing_token"],
            runtime["version"],
            risk["version"],
            manifest,
            cycle_id,
        )
        assert row["authorized"] is True
        assert row["status"] == "AUTHORIZED_AND_PERSISTED"
    finally:
        conn.close()


def test_concurrent_new_authorizations_all_see_existing_recovery_barrier():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        lease, runtime, risk, manifest = _bootstrap(setup, runtime_id)
        _cancel(setup, runtime_id)
        fence = lease["fencing_token"]
        version = runtime["version"]
        risk_version = risk["version"]
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(char):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _authorize(
                conn,
                runtime_id,
                fence,
                version,
                risk_version,
                manifest,
                _h(char),
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, ("a", "b")))

    assert all(row["authorized"] is False for row in rows)
    assert all(row["status"] == "RECOVERY_BARRIER" for row in rows)
    conn = _connect()
    try:
        assert _count(conn, "brian_governed_cycle_authorizations", runtime_id) == 0
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
    finally:
        conn.close()


def test_phase81_refuses_to_freeze_directive_while_foreign_cycle_is_active():
    runtime_id = _rid("foreign")
    conn = _connect()
    try:
        original = _h("o")
        foreign = _h("f")
        dispatch = _h("d")
        receipt = _h("r")
        checkpoint = {
            "journal_manifest": {
                "entries": [
                    {"cycle_id": original, "stage": "COMMITTED"},
                    {"cycle_id": foreign, "stage": "CYCLE_CREATED"},
                ]
            }
        }
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.brian_shadow_runtime_heads(
                  runtime_id,version,checkpoint_id,checkpoint_payload,
                  journal_hash,head_state_id,pending_cycle_id,
                  owner_token,fencing_token,acquired_at,lease_until,updated_at
                ) values (
                  %s,5,%s,%s,%s,%s,null,
                  'owner-a',1,now(),now()+interval '5 minutes',now()
                )
                """,
                (
                    runtime_id,
                    _h("k"),
                    Json(checkpoint),
                    _h("j"),
                    _h("h"),
                ),
            )
            cur.execute(
                """
                insert into public.brian_shadow_execution_dispatches(
                  runtime_id,dispatch_id,cycle_id,governed_result_id,
                  policy_fingerprint,authorization_checkpoint_id,
                  authorization_runtime_version,risk_version,
                  risk_ledger_hash,risk_receipt_id,fencing_token
                ) values (%s,%s,%s,%s,%s,%s,4,1,%s,%s,1)
                """,
                (
                    runtime_id,
                    dispatch,
                    original,
                    _h("g"),
                    _h("p"),
                    _h("w"),
                    _h("l"),
                    _h("u"),
                ),
            )
            cur.execute(
                """
                insert into public.brian_shadow_execution_cancel_requests(
                  runtime_id,dispatch_id,cycle_id,
                  risk_version,risk_receipt_id,journal_stage,
                  phase,reason,requested_at,
                  worker_token,claim_fencing_token
                ) values (
                  %s,%s,%s,
                  1,%s,'PAPER_APPLIED',
                  'AFTER_START','HALTED',now(),
                  'worker-a',1
                )
                """,
                (runtime_id, dispatch, original, receipt),
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
                  1,1,
                  4,1,%s,'ACTIVE',
                  'CYCLE_CREATED','PROCEED',
                  false,null,false,now()
                )
                """,
                (runtime_id, dispatch, original, _h("u")),
            )

            cur.execute(
                "select public.brian_prepare_shadow_cancel_recovery(%s,%s,%s,%s,%s)",
                (runtime_id, "owner-a", 1, original, 5),
            )
            row = cur.fetchone()[0]

        assert row["prepared"] is False
        assert row["status"] == "FOREIGN_CYCLE_ACTIVE"
        assert row["foreign_cycle_id"] == foreign
        assert row["foreign_cycle_stage"] == "CYCLE_CREATED"
        assert _count(
            conn,
            "brian_shadow_cancel_recovery_directives",
            runtime_id,
        ) == 0
    finally:
        conn.close()


def test_service_role_cannot_write_admission_event_history_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        _bootstrap(conn, runtime_id)
        _cancel(conn, runtime_id)
        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    insert into public.brian_shadow_recovery_admission_events(
                      runtime_id,candidate_cycle_id,stage,event,
                      barrier_original_cycle_id,barrier_cancel_risk_receipt_id,
                      barrier_reason
                    ) values (%s,%s,'AUTHORIZATION','RECOVERY_BARRIER',%s,%s,'HALTED')
                    """,
                    (runtime_id, _h("c"), _h("o"), _h("r")),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
