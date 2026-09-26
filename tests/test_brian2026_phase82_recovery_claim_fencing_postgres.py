"""Real Postgres tests for Phase82 recovery claim fencing."""

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
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase82 Postgres tests require CI database",
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
    return f"pytest-phase82-{label}-{uuid.uuid4().hex[:10]}"


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
    recovery_status="READY_REDUCE_ONLY",
    runtime_version=5,
    head_state_id=None,
):
    cycle_id = _h("c")
    dispatch_id = _h("d")
    cancel_receipt_id = _h("r")
    current_receipt_id = _h("s")
    current_entry_id = _h("e")
    head_state_id = head_state_id or _h("h")
    legs = [] if recovery_status == "NO_RECOVERY_REQUIRED" else [_leg()]
    unsafe = (
        [{"asset_id": "BTCUSDT", "reason": "ROLLBACK_NOT_REDUCE_ONLY"}]
        if recovery_status == "MANUAL_REVIEW"
        else []
    )

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
                runtime_version,
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
                current_entry_id,
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
                current_entry_id,
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
              %s,%s,%s,
              1,%s,%s,
              %s,%s,%s,now()
            )
            """,
            (
                runtime_id,
                dispatch_id,
                cycle_id,
                cancel_receipt_id,
                runtime_version,
                _h("a"),
                head_state_id,
                current_receipt_id,
                risk_state,
                recovery_status,
                Json(legs),
                Json(unsafe),
            ),
        )

    return {
        "cycle_id": cycle_id,
        "dispatch_id": dispatch_id,
        "cancel_receipt_id": cancel_receipt_id,
        "runtime_version": runtime_version,
        "head_state_id": head_state_id,
    }


def _claim(conn, runtime_id, cycle_id, worker, seconds=30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_claim_shadow_cancel_recovery(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, worker, seconds),
        )
        return cur.fetchone()[0]


def _renew(conn, runtime_id, cycle_id, worker, claim_fence, seconds=30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_renew_shadow_cancel_recovery_claim(%s,%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                1,
                cycle_id,
                worker,
                claim_fence,
                seconds,
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
    return receipt_id


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_ready_recovery_can_be_claimed_by_one_worker():
    runtime_id = _rid("claim")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        row = _claim(conn, runtime_id, ctx["cycle_id"], "worker-a")
        assert row["claimed"] is True
        assert row["status"] == "CLAIMED"
        assert row["claim_fencing_token"] == 1
        assert row["worker_token"] == "worker-a"
        assert row["risk_state"] == "REDUCING"
        assert len(row["recovery_legs"]) == 1
    finally:
        conn.close()


def test_two_concurrent_workers_have_exactly_one_active_recovery_claim():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        ctx = _setup(setup, runtime_id)
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(worker):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return worker, _claim(conn, runtime_id, ctx["cycle_id"], worker)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, ("worker-a", "worker-b")))

    assert sum(row["status"] == "CLAIMED" for _, row in rows) == 1
    assert sum(row["status"] == "BLOCKED_ACTIVE" for _, row in rows) == 1
    winner = next(row for _, row in rows if row["status"] == "CLAIMED")
    assert winner["claim_fencing_token"] == 1


def test_same_worker_retry_is_already_owned_with_same_fence():
    runtime_id = _rid("same-worker")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        first = _claim(conn, runtime_id, ctx["cycle_id"], "worker-a")
        duplicate = _claim(conn, runtime_id, ctx["cycle_id"], "worker-a")
        assert first["status"] == "CLAIMED"
        assert duplicate["claimed"] is True
        assert duplicate["status"] == "ALREADY_OWNED"
        assert duplicate["claim_fencing_token"] == first["claim_fencing_token"]
    finally:
        conn.close()


def test_expired_recovery_claim_takeover_increments_fence_and_old_worker_cannot_renew():
    runtime_id = _rid("takeover")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        first = _claim(conn, runtime_id, ctx["cycle_id"], "worker-a", 1)
        assert first["claim_fencing_token"] == 1
        time.sleep(1.15)

        takeover = _claim(conn, runtime_id, ctx["cycle_id"], "worker-b", 30)
        assert takeover["claimed"] is True
        assert takeover["status"] == "EXPIRED_RECOVERY"
        assert takeover["claim_fencing_token"] == 2

        stale = _renew(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
            first["claim_fencing_token"],
            30,
        )
        assert stale["renewed"] is False
        assert stale["status"] == "RENEWAL_LOST"

        renewed = _renew(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-b",
            takeover["claim_fencing_token"],
            30,
        )
        assert renewed["renewed"] is True
        assert renewed["status"] == "RENEWED"
    finally:
        conn.close()


def test_halted_risk_blocks_initial_recovery_claim_without_creating_claim_row():
    runtime_id = _rid("halted")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            risk_state="HALTED",
            recovery_status="WAIT_RISK_RELEASE",
        )
        row = _claim(conn, runtime_id, ctx["cycle_id"], "worker-a")
        assert row["claimed"] is False
        assert row["terminal"] is False
        assert row["status"] == "WAIT_RISK_RELEASE"
        assert row["risk_state"] == "HALTED"
        assert _count(conn, "brian_shadow_recovery_claims", runtime_id) == 0
    finally:
        conn.close()


def test_risk_halt_after_claim_blocks_renewal_instead_of_extending_authority():
    runtime_id = _rid("halt-renew")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        claim = _claim(conn, runtime_id, ctx["cycle_id"], "worker-a", 30)
        _set_risk(conn, runtime_id, state="HALTED")
        renewal = _renew(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
            claim["claim_fencing_token"],
            30,
        )
        assert renewal["renewed"] is False
        assert renewal["status"] == "RENEWAL_BLOCKED_RISK"
        assert renewal["risk_state"] == "HALTED"
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("version_delta", "new_head"),
    [
        (1, None),
        (0, "z"),
    ],
)
def test_runtime_or_head_movement_invalidates_recovery_claim(version_delta, new_head):
    runtime_id = _rid("head-moved")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        with conn.cursor() as cur:
            if version_delta:
                cur.execute(
                    "update public.brian_shadow_runtime_heads "
                    "set version=version+%s where runtime_id=%s",
                    (version_delta, runtime_id),
                )
            if new_head is not None:
                cur.execute(
                    "update public.brian_shadow_runtime_heads "
                    "set head_state_id=%s where runtime_id=%s",
                    (_h(new_head), runtime_id),
                )

        row = _claim(conn, runtime_id, ctx["cycle_id"], "worker-a")
        assert row["claimed"] is False
        assert row["status"] == "HEAD_MOVED"
        assert _count(conn, "brian_shadow_recovery_claims", runtime_id) == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("recovery_status", "expected_status"),
    [
        ("NO_RECOVERY_REQUIRED", "NO_RECOVERY_REQUIRED"),
        ("MANUAL_REVIEW", "MANUAL_REVIEW"),
    ],
)
def test_terminal_directives_are_never_auto_claimed(recovery_status, expected_status):
    runtime_id = _rid("terminal")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            recovery_status=recovery_status,
        )
        row = _claim(conn, runtime_id, ctx["cycle_id"], "worker-a")
        assert row["claimed"] is False
        assert row["terminal"] is True
        assert row["status"] == expected_status
        assert _count(conn, "brian_shadow_recovery_claims", runtime_id) == 0
    finally:
        conn.close()


def test_service_role_cannot_mutate_recovery_claim_or_event_tables_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        assert _claim(
            conn,
            runtime_id,
            ctx["cycle_id"],
            "worker-a",
        )["claimed"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_shadow_recovery_claims "
                    "set worker_token='attacker' where runtime_id=%s",
                    (runtime_id,),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
