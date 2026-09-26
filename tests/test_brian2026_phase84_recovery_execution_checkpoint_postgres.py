"""Real Postgres tests for Phase84 recovery execution checkpoint fencing."""

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
DRAFTS = (
    ROOT / "brian2026" / "sql" / "phase79_atomic_execution_start.sql",
    ROOT / "brian2026" / "sql" / "phase80_claim_fenced_checkpoint_commit.sql",
    ROOT / "brian2026" / "sql" / "phase81_cancel_recovery_directive.sql",
    ROOT / "brian2026" / "sql" / "phase82_recovery_claim_fencing.sql",
    ROOT / "brian2026" / "sql" / "phase83_atomic_recovery_start.sql",
    ROOT / "brian2026" / "sql" / "phase84_recovery_execution_checkpoint.sql",
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase84 Postgres tests require CI database",
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
    return f"pytest-phase84-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


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


def _original_cycle(cycle_id: str):
    return {
        "source_plan_id": "original-plan",
        "items": [],
        "cycle_id": cycle_id,
        "account_state_mutated": False,
        "shadow_only": True,
        "live_execution": False,
    }


def _recovery_cycle(cycle_id: str, *, valid=True):
    return {
        "source_plan_id": "phase84-recovery-plan",
        "items": [{
            "instruction_kind": "REDUCE",
            "asset_id": "BTCUSDT",
            "risk_receipt": {
                "allowed": True,
                "reduce_only": valid,
                "projected_position_weight": 0.10,
            },
            "execution_receipt": {
                "status": "FILLED",
                "side": "SELL",
            },
            "pending_reversal": None,
            "new_risk_cash_reserved_usd": 0.0,
            "status": "REDUCTION_FILLED",
        }],
        "cycle_id": cycle_id,
        "account_state_mutated": False,
        "shadow_only": True,
        "live_execution": False,
    }


def _entry(
    sequence: int,
    *,
    cycle_id: str,
    cycle_hash: str,
    stage: str,
    entry_id: str,
    previous_entry_id: str | None,
    artifact_hash: str,
):
    return {
        "schema_version": "brian.phase66-durable-cycle-journal.v1",
        "sequence": sequence,
        "stage": stage,
        "cycle_id": cycle_id,
        "cycle_hash": cycle_hash,
        "previous_entry_id": previous_entry_id,
        "artifact_hash": artifact_hash,
        "artifact_ref": f"{stage.lower()}:{sequence}",
        "entry_id": entry_id,
    }


def _base_entries(original_cycle_id: str):
    e0 = _entry(
        0,
        cycle_id=original_cycle_id,
        cycle_hash=_h("a"),
        stage="CYCLE_CREATED",
        entry_id=_h("0"),
        previous_entry_id=None,
        artifact_hash=_h("A"),
    )
    e1 = _entry(
        1,
        cycle_id=original_cycle_id,
        cycle_hash=_h("a"),
        stage="COMMITTED",
        entry_id=_h("1"),
        previous_entry_id=e0["entry_id"],
        artifact_hash=_h("B"),
    )
    return [e0, e1]


def _checkpoint(
    original_cycle_id: str,
    recovery_cycle_id: str | None,
    *,
    recovery_stages=(),
    checkpoint_char: str,
    journal_char: str,
    head_state_id: str,
    recovery_valid=True,
):
    cycles = {original_cycle_id: _original_cycle(original_cycle_id)}
    entries = _base_entries(original_cycle_id)
    previous = entries[-1]["entry_id"]

    if recovery_cycle_id is not None:
        cycles[recovery_cycle_id] = _recovery_cycle(
            recovery_cycle_id,
            valid=recovery_valid,
        )
        stage_chars = ("2", "3", "4", "5", "6", "7", "8")
        artifact_chars = ("C", "D", "E", "F", "G", "H", "I")
        for stage in recovery_stages:
            seq = len(entries)
            entry = _entry(
                seq,
                cycle_id=recovery_cycle_id,
                cycle_hash=_h("b"),
                stage=stage,
                entry_id=_h(stage_chars[seq - 2]),
                previous_entry_id=previous,
                artifact_hash=_h(artifact_chars[seq - 2]),
            )
            entries.append(entry)
            previous = entry["entry_id"]

    pending = None
    transitions = []
    if recovery_cycle_id is not None and recovery_stages:
        transitions.append({
            "sequence": 3,
            "kind": "CYCLE_PROPOSED",
            "cycle_id": recovery_cycle_id,
            "before_state_id": _h("h"),
            "after_state_id": _h("h"),
        })
    if recovery_cycle_id is not None and "COMMITTED" in recovery_stages:
        transitions.append({
            "sequence": 4,
            "kind": "RECONCILED_COMMIT",
            "cycle_id": recovery_cycle_id,
            "before_state_id": _h("h"),
            "after_state_id": head_state_id,
        })

    return {
        "schema_version": "brian.phase67-durable-runtime-orchestrator.v1",
        "runtime_checkpoint": {
            "schema_version": "brian.phase63-crash-recovery.v1",
            "paper": {
                "paper_only": True,
                "live_execution": False,
            },
            "shadow_ledger_manifest": {
                "head_state_id": head_state_id,
                "pending_cycle_id": pending,
                "transitions": transitions,
            },
            "pending_cycle_id": pending,
            "live_execution": False,
        },
        "journal_manifest": {
            "schema_version": "brian.phase66-durable-cycle-journal.v1",
            "append_only": True,
            "cycles": cycles,
            "entries": entries,
            "journal_hash": _h(journal_char),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h(checkpoint_char),
    }


def _risk_receipt(receipt_id: str, state="REDUCING"):
    return {
        "receipt_id": receipt_id,
        "trading_state": state,
        "blocked_assets": [],
    }


def _setup(conn, runtime_id: str):
    original_cycle_id = _h("o")
    recovery_cycle_id = _h("c")
    dispatch_id = _h("d")
    cancel_receipt_id = _h("r")
    risk_receipt_id = _h("s")
    risk_entry_id = _h("e")
    initial = _checkpoint(
        original_cycle_id,
        None,
        checkpoint_char="k",
        journal_char="j",
        head_state_id=_h("h"),
    )

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
                initial["checkpoint_id"],
                Json(initial),
                initial["journal_manifest"]["journal_hash"],
                _h("h"),
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_runtime_checkpoints(
              runtime_id,version,checkpoint_id,checkpoint_payload,
              journal_hash,head_state_id,pending_cycle_id,
              fencing_token,committed_at
            ) values (%s,5,%s,%s,%s,%s,null,1,now())
            """,
            (
                runtime_id,
                initial["checkpoint_id"],
                Json(initial),
                initial["journal_manifest"]["journal_hash"],
                _h("h"),
            ),
        )

        for cycle_id, payload in initial["journal_manifest"]["cycles"].items():
            cur.execute(
                """
                insert into public.brian_shadow_runtime_cycles(
                  runtime_id,cycle_id,cycle_hash,cycle_payload,first_seen_version
                ) values (%s,%s,%s,%s,5)
                """,
                (runtime_id, cycle_id, _h("a"), Json(payload)),
            )

        for entry in initial["journal_manifest"]["entries"]:
            cur.execute(
                """
                insert into public.brian_shadow_runtime_journal_entries(
                  runtime_id,sequence,entry_id,previous_entry_id,
                  cycle_id,stage,cycle_hash,artifact_hash,artifact_ref,
                  entry_payload,first_seen_version
                ) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,5)
                """,
                (
                    runtime_id,
                    entry["sequence"],
                    entry["entry_id"],
                    entry["previous_entry_id"],
                    entry["cycle_id"],
                    entry["stage"],
                    entry["cycle_hash"],
                    entry["artifact_hash"],
                    entry["artifact_ref"],
                    Json(entry),
                ),
            )

        cur.execute(
            """
            insert into public.brian_operational_risk_entries(
              runtime_id,sequence,entry_id,previous_entry_id,
              policy_hash,receipt_id,receipt_timestamp,
              previous_state,trading_state,halt_latched,
              entry_payload,first_seen_version
            ) values (
              %s,0,%s,null,%s,%s,1.0,
              'ACTIVE','REDUCING',false,%s,1
            )
            """,
            (
                runtime_id,
                risk_entry_id,
                _h("q"),
                risk_receipt_id,
                Json({"receipt": _risk_receipt(risk_receipt_id)}),
            ),
        )
        cur.execute(
            """
            insert into public.brian_operational_risk_heads(
              runtime_id,version,ledger_hash,manifest,policy_hash,
              head_entry_id,current_state,halt_latched,updated_at
            ) values (%s,1,%s,'{}'::jsonb,%s,%s,'REDUCING',false,now())
            """,
            (runtime_id, _h("l"), _h("q"), risk_entry_id),
        )

        cur.execute(
            """
            insert into public.brian_shadow_cancel_recovery_directives(
              runtime_id,dispatch_id,cycle_id,
              cancel_risk_version,cancel_risk_receipt_id,cancel_reason,
              source_runtime_version,pre_state_id,current_state_id,
              current_risk_version,current_risk_receipt_id,current_risk_state,
              recovery_status,recovery_legs,unsafe_assets,prepared_at
            ) values (
              %s,%s,%s,
              1,%s,'REDUCING_NEW_RISK',
              5,%s,%s,
              1,%s,'REDUCING',
              'READY_REDUCE_ONLY',%s,'[]'::jsonb,now()
            )
            """,
            (
                runtime_id,
                dispatch_id,
                original_cycle_id,
                cancel_receipt_id,
                _h("a"),
                _h("h"),
                risk_receipt_id,
                Json([_leg()]),
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_recovery_claims(
              runtime_id,dispatch_id,cycle_id,cancel_risk_receipt_id,
              status,worker_token,claim_fencing_token,
              claimed_at,claim_until,
              runtime_version_at_claim,head_state_id_at_claim,
              risk_version_at_claim,risk_receipt_id_at_claim,risk_state_at_claim,
              updated_at
            ) values (
              %s,%s,%s,%s,
              'CLAIMED','worker-a',3,
              now(),now()+interval '5 minutes',
              5,%s,1,%s,'REDUCING',now()
            )
            """,
            (
                runtime_id,
                dispatch_id,
                original_cycle_id,
                cancel_receipt_id,
                _h("h"),
                risk_receipt_id,
            ),
        )
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
            ) values (
              %s,%s,%s,%s,
              'worker-a',3,
              1,5,%s,
              1,%s,'REDUCING',
              5,%s,%s,now()
            )
            """,
            (
                runtime_id,
                dispatch_id,
                original_cycle_id,
                cancel_receipt_id,
                _h("h"),
                risk_receipt_id,
                _h("h"),
                Json([_leg()]),
            ),
        )

    return {
        "original_cycle_id": original_cycle_id,
        "recovery_cycle_id": recovery_cycle_id,
        "dispatch_id": dispatch_id,
        "cancel_receipt_id": cancel_receipt_id,
    }


def _commit(
    conn,
    runtime_id: str,
    ctx: dict,
    checkpoint: dict,
    *,
    expected_version: int,
    worker="worker-a",
    claim_fence=3,
):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_commit_shadow_recovery_checkpoint("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                1,
                ctx["original_cycle_id"],
                ctx["recovery_cycle_id"],
                worker,
                claim_fence,
                expected_version,
                Json(checkpoint),
            ),
        )
        return cur.fetchone()[0]


def _read_claim(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            select status,worker_token,claim_fencing_token,
                   recovery_cycle_id,progress_runtime_version,
                   progress_head_state_id,progress_checkpoint_id,
                   completion_ref
            from public.brian_shadow_recovery_claims
            where runtime_id=%s
            """,
            (runtime_id,),
        )
        return cur.fetchone()


def _read_head(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            "select version,checkpoint_id,head_state_id "
            "from public.brian_shadow_runtime_heads where runtime_id=%s",
            (runtime_id,),
        )
        return cur.fetchone()


def _renew(conn, runtime_id, original_cycle_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_renew_shadow_cancel_recovery_claim("
            "%s,%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                1,
                original_cycle_id,
                "worker-a",
                3,
                60,
            ),
        )
        return cur.fetchone()[0]


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_write_ahead_commit_updates_recovery_progress_anchor_and_keeps_claim_active():
    runtime_id = _rid("writeahead")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        checkpoint = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=("CYCLE_CREATED",),
            checkpoint_char="w",
            journal_char="n",
            head_state_id=_h("h"),
        )
        row = _commit(conn, runtime_id, ctx, checkpoint, expected_version=5)
        assert row["committed"] is True
        assert row["terminal"] is False
        assert row["status"] == "COMMITTED"
        assert row["journal_stage"] == "CYCLE_CREATED"
        assert row["version"] == 6

        claim = _read_claim(conn, runtime_id)
        assert claim[0] == "CLAIMED"
        assert claim[3] == ctx["recovery_cycle_id"]
        assert claim[4] == 6
        assert claim[5] == _h("h")
        assert claim[6] == checkpoint["checkpoint_id"]
        assert claim[7] is None
    finally:
        conn.close()


def test_phase82_renewal_accepts_phase84_authorized_progress_after_restart():
    runtime_id = _rid("resume-anchor")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        checkpoint = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=("CYCLE_CREATED",),
            checkpoint_char="w",
            journal_char="n",
            head_state_id=_h("h"),
        )
        assert _commit(
            conn,
            runtime_id,
            ctx,
            checkpoint,
            expected_version=5,
        )["version"] == 6

        renewed = _renew(conn, runtime_id, ctx["original_cycle_id"])
        assert renewed["renewed"] is True
        assert renewed["status"] == "RENEWED"
        assert renewed["runtime_version"] == 6
        assert renewed["head_state_id"] == _h("h")
    finally:
        conn.close()


def test_final_recovery_commit_atomically_completes_claim_and_advances_head():
    runtime_id = _rid("complete")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        writeahead = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=("CYCLE_CREATED",),
            checkpoint_char="w",
            journal_char="n",
            head_state_id=_h("h"),
        )
        assert _commit(conn, runtime_id, ctx, writeahead, expected_version=5)["version"] == 6

        final = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=(
                "CYCLE_CREATED",
                "PAPER_APPLIED",
                "LOCAL_PROJECTED",
                "RECONCILED",
                "COMMITTED",
            ),
            checkpoint_char="z",
            journal_char="y",
            head_state_id=_h("z"),
        )
        row = _commit(conn, runtime_id, ctx, final, expected_version=6)
        assert row["committed"] is True
        assert row["terminal"] is True
        assert row["status"] == "RECOVERY_COMMITTED_PENDING_AUDIT"
        assert row["version"] == 7
        assert row["head_state_id"] == _h("z")

        claim = _read_claim(conn, runtime_id)
        # Phase84 leaves the claim open for Phase85's authoritative
        # post-recovery invariant audit.
        assert claim[0] == "CLAIMED"
        assert claim[3] == ctx["recovery_cycle_id"]
        assert claim[4] == 7
        assert claim[5] == _h("z")
        assert claim[6] == final["checkpoint_id"]
        assert claim[7] is None

        head = _read_head(conn, runtime_id)
        assert head == (7, final["checkpoint_id"], _h("z"))
    finally:
        conn.close()


def test_exact_final_lost_response_retry_is_duplicate_current_after_claim_completed():
    runtime_id = _rid("final-retry")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        writeahead = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=("CYCLE_CREATED",),
            checkpoint_char="w",
            journal_char="n",
            head_state_id=_h("h"),
        )
        _commit(conn, runtime_id, ctx, writeahead, expected_version=5)

        final = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=(
                "CYCLE_CREATED",
                "PAPER_APPLIED",
                "LOCAL_PROJECTED",
                "RECONCILED",
                "COMMITTED",
            ),
            checkpoint_char="z",
            journal_char="y",
            head_state_id=_h("z"),
        )
        first = _commit(conn, runtime_id, ctx, final, expected_version=6)
        retry = _commit(conn, runtime_id, ctx, final, expected_version=6)
        assert first["status"] == "RECOVERY_COMMITTED_PENDING_AUDIT"
        assert retry["committed"] is True
        assert retry["duplicate"] is True
        assert retry["terminal"] is True
        assert retry["status"] == "DUPLICATE_CURRENT"
        assert _read_head(conn, runtime_id)[0] == 7
    finally:
        conn.close()


def test_invalid_non_reduce_only_recovery_cycle_never_reaches_phase70_commit():
    runtime_id = _rid("invalid")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        checkpoint = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=("CYCLE_CREATED",),
            checkpoint_char="w",
            journal_char="n",
            head_state_id=_h("h"),
            recovery_valid=False,
        )
        row = _commit(conn, runtime_id, ctx, checkpoint, expected_version=5)
        assert row["committed"] is False
        assert row["status"] == "RECOVERY_CYCLE_INVALID"
        assert _read_head(conn, runtime_id)[0] == 5
        assert _count(conn, "brian_shadow_runtime_checkpoints", runtime_id) == 1
    finally:
        conn.close()


def test_concurrent_write_ahead_commit_is_one_commit_and_one_duplicate_current():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        ctx = _setup(setup, runtime_id)
        checkpoint = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=("CYCLE_CREATED",),
            checkpoint_char="w",
            journal_char="n",
            head_state_id=_h("h"),
        )
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(_):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _commit(
                conn,
                runtime_id,
                ctx,
                checkpoint,
                expected_version=5,
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "COMMITTED" for row in rows) == 1
    assert sum(row["status"] == "DUPLICATE_CURRENT" for row in rows) == 1
    conn = _connect()
    try:
        assert _read_head(conn, runtime_id)[0] == 6
    finally:
        conn.close()


def test_stale_recovery_worker_cannot_commit_after_claim_takeover():
    runtime_id = _rid("takeover")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.brian_shadow_recovery_claims
                set worker_token='worker-b',
                    claim_fencing_token=4,
                    claimed_at=now(),
                    claim_until=now()+interval '5 minutes',
                    updated_at=now()
                where runtime_id=%s
                """,
                (runtime_id,),
            )

        checkpoint = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=("CYCLE_CREATED",),
            checkpoint_char="w",
            journal_char="n",
            head_state_id=_h("h"),
        )
        row = _commit(conn, runtime_id, ctx, checkpoint, expected_version=5)
        assert row["committed"] is False
        assert row["status"] == "CLAIM_LOST"
        assert _read_head(conn, runtime_id)[0] == 5
    finally:
        conn.close()


def test_service_role_cannot_write_recovery_commit_events_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        checkpoint = _checkpoint(
            ctx["original_cycle_id"],
            ctx["recovery_cycle_id"],
            recovery_stages=("CYCLE_CREATED",),
            checkpoint_char="w",
            journal_char="n",
            head_state_id=_h("h"),
        )
        assert _commit(
            conn,
            runtime_id,
            ctx,
            checkpoint,
            expected_version=5,
        )["committed"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    insert into public.brian_shadow_recovery_commit_events(
                      runtime_id,original_cycle_id,recovery_cycle_id,
                      worker_token,recovery_claim_fencing_token,
                      runtime_fencing_token,event
                    ) values (%s,%s,%s,'attacker',3,1,'WRITE_AHEAD_COMMITTED')
                    """,
                    (
                        runtime_id,
                        ctx["original_cycle_id"],
                        ctx["recovery_cycle_id"],
                    ),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
