"""Real Postgres tests for Phase83 atomic recovery STARTED fencing."""

from __future__ import annotations

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
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase83 Postgres tests require CI database",
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
    return f"pytest-phase83-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _leg(*, reduce_only=True, order_direction=-1):
    return {
        "asset_id": "BTCUSDT",
        "before_weight": 0.10,
        "current_weight": 0.25,
        "target_weight": 0.10,
        "reduce_weight": 0.15,
        "current_direction": 1,
        "order_direction": order_direction,
        "reduce_only": reduce_only,
    }


def _risk_receipt(receipt_id: str, state: str):
    return {
        "receipt_id": receipt_id,
        "trading_state": state,
        "blocked_assets": [],
    }


def _setup(
    conn,
    runtime_id: str,
    *,
    risk_state="REDUCING",
    claim_seconds=60,
    malformed_leg=False,
):
    cycle_id = _h("c")
    dispatch_id = _h("d")
    cancel_receipt_id = _h("r")
    current_receipt_id = _h("s")
    risk_entry_id = _h("e")
    head_state_id = _h("h")
    leg = _leg(
        reduce_only=not malformed_leg,
        order_direction=1 if malformed_leg else -1,
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
                _h("k"),
                Json({"checkpoint_id": _h("k")}),
                _h("j"),
                head_state_id,
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
              'ACTIVE',%s,%s,%s,1
            )
            """,
            (
                runtime_id,
                risk_entry_id,
                _h("q"),
                current_receipt_id,
                risk_state,
                risk_state == "HALTED",
                Json({"receipt": _risk_receipt(current_receipt_id, risk_state)}),
            ),
        )
        cur.execute(
            """
            insert into public.brian_operational_risk_heads(
              runtime_id,version,ledger_hash,manifest,policy_hash,
              head_entry_id,current_state,halt_latched,updated_at
            ) values (%s,1,%s,'{}'::jsonb,%s,%s,%s,%s,now())
            """,
            (
                runtime_id,
                _h("l"),
                _h("q"),
                risk_entry_id,
                risk_state,
                risk_state == "HALTED",
            ),
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
              1,%s,%s,
              %s,%s,'[]'::jsonb,now()
            )
            """,
            (
                runtime_id,
                dispatch_id,
                cycle_id,
                cancel_receipt_id,
                _h("a"),
                head_state_id,
                current_receipt_id,
                risk_state,
                "WAIT_RISK_RELEASE" if risk_state == "HALTED" else "READY_REDUCE_ONLY",
                Json([leg]),
            ),
        )

    claim = _claim(
        conn,
        runtime_id,
        cycle_id,
        "worker-a",
        claim_seconds,
    )
    return {
        "cycle_id": cycle_id,
        "dispatch_id": dispatch_id,
        "cancel_receipt_id": cancel_receipt_id,
        "head_state_id": head_state_id,
        "claim": claim,
    }


def _claim(conn, runtime_id, cycle_id, worker, seconds=60):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_claim_shadow_cancel_recovery(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, worker, seconds),
        )
        return cur.fetchone()[0]


def _start(conn, runtime_id, cycle_id, worker, claim_fence):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_mark_shadow_recovery_started(%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                1,
                cycle_id,
                worker,
                claim_fence,
            ),
        )
        return cur.fetchone()[0]


def _set_risk(conn, runtime_id, *, state, receipt_char="t", entry_char="f"):
    receipt_id = _h(receipt_char)
    entry_id = _h(entry_char)
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_operational_risk_entries(
              runtime_id,sequence,entry_id,previous_entry_id,
              policy_hash,receipt_id,receipt_timestamp,
              previous_state,trading_state,halt_latched,
              entry_payload,first_seen_version
            )
            select %s,1,%s,head_entry_id,%s,%s,2.0,
                   current_state,%s,%s,%s,2
            from public.brian_operational_risk_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                entry_id,
                _h("q"),
                receipt_id,
                state,
                state == "HALTED",
                Json({"receipt": _risk_receipt(receipt_id, state)}),
                runtime_id,
            ),
        )
        cur.execute(
            """
            update public.brian_operational_risk_heads
            set version=2,
                ledger_hash=%s,
                head_entry_id=%s,
                current_state=%s,
                halt_latched=%s,
                updated_at=now()
            where runtime_id=%s
            """,
            (_h("m"), entry_id, state, state == "HALTED", runtime_id),
        )


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_current_recovery_claim_crosses_started_once():
    runtime_id = _rid("start")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        claim = ctx["claim"]
        assert claim["claimed"] is True
        row = _start(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert row["started"] is True
        assert row["status"] == "STARTED"
        assert row["duplicate"] is False
        assert row["resume_only"] is False
        assert row["risk_state"] == "REDUCING"
        assert len(row["recovery_legs"]) == 1
        assert _count(conn, "brian_shadow_recovery_starts", runtime_id) == 1
    finally:
        conn.close()


def test_exact_start_retry_is_started_already_with_one_immutable_row():
    runtime_id = _rid("duplicate")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        fence = ctx["claim"]["claim_fencing_token"]
        first = _start(conn, runtime_id, ctx["cycle_id"], "worker-a", fence)
        duplicate = _start(conn, runtime_id, ctx["cycle_id"], "worker-a", fence)
        assert first["status"] == "STARTED"
        assert duplicate["status"] == "STARTED_ALREADY"
        assert duplicate["started"] is True
        assert duplicate["duplicate"] is True
        assert duplicate["resume_only"] is False
        assert _count(conn, "brian_shadow_recovery_starts", runtime_id) == 1
    finally:
        conn.close()


def test_concurrent_exact_start_has_one_started_and_one_duplicate():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        ctx = _setup(setup, runtime_id)
        fence = ctx["claim"]["claim_fencing_token"]
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(_):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _start(
                conn,
                runtime_id,
                ctx["cycle_id"],
                "worker-a",
                fence,
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "STARTED" for row in rows) == 1
    assert sum(row["status"] == "STARTED_ALREADY" for row in rows) == 1


def test_wrong_worker_or_claim_fence_cannot_cross_started():
    runtime_id = _rid("ownership")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        fence = ctx["claim"]["claim_fencing_token"]
        wrong_worker = _start(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-b",
            fence,
        )
        wrong_fence = _start(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
            fence + 1,
        )
        assert wrong_worker["started"] is False
        assert wrong_worker["status"] == "CLAIM_LOST"
        assert wrong_fence["started"] is False
        assert wrong_fence["status"] == "CLAIM_LOST"
        assert _count(conn, "brian_shadow_recovery_starts", runtime_id) == 0
    finally:
        conn.close()


def test_halt_after_claim_but_before_started_blocks_boundary():
    runtime_id = _rid("halt")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        fence = ctx["claim"]["claim_fencing_token"]
        _set_risk(conn, runtime_id, state="HALTED")
        row = _start(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
            fence,
        )
        assert row["started"] is False
        assert row["status"] == "WAIT_RISK_RELEASE"
        assert row["risk_state"] == "HALTED"
        assert _count(conn, "brian_shadow_recovery_starts", runtime_id) == 0
    finally:
        conn.close()


def test_runtime_or_head_movement_after_claim_blocks_first_started():
    for move in ("version", "head"):
        runtime_id = _rid(f"moved-{move}")
        conn = _connect()
        try:
            ctx = _setup(conn, runtime_id)
            fence = ctx["claim"]["claim_fencing_token"]
            with conn.cursor() as cur:
                if move == "version":
                    cur.execute(
                        "update public.brian_shadow_runtime_heads "
                        "set version=version+1 where runtime_id=%s",
                        (runtime_id,),
                    )
                else:
                    cur.execute(
                        "update public.brian_shadow_runtime_heads "
                        "set head_state_id=%s where runtime_id=%s",
                        (_h("z"), runtime_id),
                    )
            row = _start(
                conn,
                runtime_id,
                ctx["cycle_id"],
                "worker-a",
                fence,
            )
            assert row["started"] is False
            assert row["status"] == "HEAD_MOVED"
            assert _count(conn, "brian_shadow_recovery_starts", runtime_id) == 0
        finally:
            conn.close()


def test_invalid_reduce_only_directive_is_rejected_even_if_phase82_claim_exists():
    runtime_id = _rid("bad-leg")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id, malformed_leg=True)
        claim = ctx["claim"]
        assert claim["claimed"] is True
        row = _start(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert row["started"] is False
        assert row["status"] == "EVIDENCE_INVALID"
        assert _count(conn, "brian_shadow_recovery_starts", runtime_id) == 0
    finally:
        conn.close()


def test_expired_claim_takeover_resumes_existing_started_without_second_start_row():
    runtime_id = _rid("resume")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id, claim_seconds=1)
        first_claim = ctx["claim"]
        assert _start(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
            first_claim["claim_fencing_token"],
        )["status"] == "STARTED"

        time.sleep(1.15)
        takeover = _claim(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-b",
            30,
        )
        assert takeover["status"] == "EXPIRED_RECOVERY"
        assert takeover["claim_fencing_token"] == (
            first_claim["claim_fencing_token"] + 1
        )

        resumed = _start(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-b",
            takeover["claim_fencing_token"],
        )
        assert resumed["started"] is True
        assert resumed["duplicate"] is True
        assert resumed["resume_only"] is True
        assert resumed["status"] == "STARTED_RESUME"
        assert _count(conn, "brian_shadow_recovery_starts", runtime_id) == 1
    finally:
        conn.close()


def test_service_role_cannot_mutate_recovery_started_history_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        assert _start(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
            ctx["claim"]["claim_fencing_token"],
        )["started"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_shadow_recovery_starts "
                    "set worker_token='attacker' where runtime_id=%s",
                    (runtime_id,),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
