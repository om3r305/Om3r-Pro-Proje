"""Real Postgres tests for Phase78 execution kill-switch decisions."""

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
    ROOT / "supabase" / "migrations" / "202609230845_brian_phase75_atomic_governed_writeahead.sql",
    ROOT / "supabase" / "migrations" / "202609230915_brian_phase76_shadow_execution_outbox.sql",
    ROOT / "supabase" / "migrations" / "202609230945_brian_phase77_execution_claim_lifecycle.sql",
    ROOT / "supabase" / "migrations" / "202609231015_brian_phase78_execution_kill_switch.sql",
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase78 Postgres tests require CI database",
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
def _apply_migrations():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP)
            for migration in MIGRATIONS:
                cur.execute(migration.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _rid(label: str) -> str:
    return f"pytest-phase78-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _cycle(cycle_id: str, *, reduce_only=False):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": "phase78-plan",
        "items": [{
            "instruction_kind": "REDUCE" if reduce_only else "OPEN",
            "asset_id": "BTCUSDT",
            "risk_receipt": {
                "allowed": True,
                "reduce_only": reduce_only,
            },
            "execution_receipt": None,
            "pending_reversal": None,
            "new_risk_cash_reserved_usd": 0.0 if reduce_only else 100.0,
            "status": "fixture",
        }],
        "initial_available_cash_usd": 1000.0,
        "reserved_new_risk_cash_usd": 0.0 if reduce_only else 100.0,
        "remaining_unreserved_cash_usd": 1000.0 if reduce_only else 900.0,
        "denied_assets": [],
        "pending_reversal_assets": [],
        "cycle_id": cycle_id,
        "account_state_mutated": False,
        "shadow_only": True,
        "live_execution": False,
    }


def _checkpoint(cycle_id: str, *, stage="CYCLE_CREATED", checkpoint_char="k"):
    entries = [{"sequence": 0, "stage": "CYCLE_CREATED", "cycle_id": cycle_id}]
    if stage != "CYCLE_CREATED":
        entries.append({"sequence": 1, "stage": stage, "cycle_id": cycle_id})
    return {
        "journal_manifest": {"entries": entries},
        "checkpoint_id": _h(checkpoint_char),
    }


def _receipt(receipt_id: str, *, state="ACTIVE", blocked=()):
    return {
        "receipt_id": receipt_id,
        "trading_state": state,
        "blocked_assets": list(blocked),
    }


def _setup(conn, runtime_id: str, *, reduce_only=False):
    cycle_id = _h("c")
    dispatch_id = _h("d")
    checkpoint_id = _h("k")
    cycle = _cycle(cycle_id, reduce_only=reduce_only)
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_runtime_heads(
              runtime_id,version,checkpoint_id,checkpoint_payload,
              journal_hash,head_state_id,pending_cycle_id,
              owner_token,fencing_token,acquired_at,lease_until,updated_at
            ) values (%s,2,%s,%s,%s,%s,null,'owner-a',1,now(),now()+interval '5 minutes',now())
            """,
            (runtime_id, checkpoint_id, Json(_checkpoint(cycle_id)), _h("j"), _h("s")),
        )
        cur.execute(
            """
            insert into public.brian_shadow_runtime_cycles(
              runtime_id,cycle_id,cycle_hash,cycle_payload,first_seen_version
            ) values (%s,%s,%s,%s,1)
            """,
            (runtime_id, cycle_id, _h("x"), Json(cycle)),
        )
        cur.execute(
            """
            insert into public.brian_shadow_execution_dispatches(
              runtime_id,dispatch_id,cycle_id,governed_result_id,
              policy_fingerprint,authorization_checkpoint_id,
              authorization_runtime_version,risk_version,
              risk_ledger_hash,risk_receipt_id,fencing_token
            ) values (%s,%s,%s,%s,%s,%s,2,1,%s,%s,1)
            """,
            (
                runtime_id, dispatch_id, cycle_id, _h("g"), _h("p"),
                checkpoint_id, _h("l"), _h("r"),
            ),
        )
        cur.execute(
            """
            insert into public.brian_operational_risk_heads(
              runtime_id,version,ledger_hash,manifest,policy_hash,
              head_entry_id,current_state,halt_latched,updated_at
            ) values (%s,1,%s,'{}'::jsonb,%s,%s,'ACTIVE',false,now())
            """,
            (runtime_id, _h("l"), _h("q"), _h("a")),
        )
        cur.execute(
            """
            insert into public.brian_operational_risk_entries(
              runtime_id,sequence,entry_id,previous_entry_id,
              policy_hash,receipt_id,receipt_timestamp,
              previous_state,trading_state,halt_latched,
              entry_payload,first_seen_version
            ) values (%s,0,%s,null,%s,%s,%s,'ACTIVE','ACTIVE',false,%s,1)
            """,
            (
                runtime_id, _h("a"), _h("q"), _h("r"), TS,
                Json({"receipt": _receipt(_h("r"))}),
            ),
        )
    claim = _claim(conn, runtime_id, cycle_id)
    assert claim["claimed"] is True
    return cycle_id, dispatch_id, claim


def _claim(conn, runtime_id, cycle_id):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_claim_shadow_execution_dispatch(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, "worker-a", 60),
        )
        return cur.fetchone()[0]


def _kill(conn, runtime_id, cycle_id, claim_fence, *, worker="worker-a"):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_check_shadow_execution_kill_switch(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, worker, claim_fence),
        )
        return cur.fetchone()[0]


def _set_risk(conn, runtime_id, *, state, blocked=()):
    receipt_id = _h("h" if state == "HALTED" else "u" if state == "REDUCING" else "b")
    entry_id = _h("i" if state == "HALTED" else "v" if state == "REDUCING" else "n")
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_operational_risk_entries(
              runtime_id,sequence,entry_id,previous_entry_id,
              policy_hash,receipt_id,receipt_timestamp,
              previous_state,trading_state,halt_latched,
              entry_payload,first_seen_version
            ) values (%s,1,%s,%s,%s,%s,%s,'ACTIVE',%s,%s,%s,2)
            """,
            (
                runtime_id, entry_id, _h("a"), _h("q"), receipt_id, TS + 1,
                state, state == "HALTED",
                Json({"receipt": _receipt(receipt_id, state=state, blocked=blocked)}),
            ),
        )
        cur.execute(
            """
            update public.brian_operational_risk_heads
            set version=2, ledger_hash=%s, head_entry_id=%s,
                current_state=%s, halt_latched=%s, updated_at=now()
            where runtime_id=%s
            """,
            (_h("2"), entry_id, state, state == "HALTED", runtime_id),
        )
    return receipt_id


def _set_stage(conn, runtime_id, cycle_id, stage, checkpoint_char):
    with conn.cursor() as cur:
        cur.execute(
            """
            update public.brian_shadow_runtime_heads
            set checkpoint_id=%s, checkpoint_payload=%s, updated_at=now()
            where runtime_id=%s
            """,
            (
                _h(checkpoint_char),
                Json(_checkpoint(cycle_id, stage=stage, checkpoint_char=checkpoint_char)),
                runtime_id,
            ),
        )


def _cancel_count(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            "select count(*) from public.brian_shadow_execution_cancel_requests where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_unchanged_active_risk_proceeds_before_execution():
    runtime_id = _rid("proceed")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
        row = _kill(conn, runtime_id, cycle_id, claim["claim_fencing_token"])
        assert row["status"] == "PROCEED"
        assert row["proceed"] is True
        assert row["cancel_requested"] is False
        assert _cancel_count(conn, runtime_id) == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("state", "blocked", "reason"),
    [
        ("HALTED", (), "HALTED"),
        ("REDUCING", (), "REDUCING_NEW_RISK"),
        ("ACTIVE", ("BTCUSDT",), "ASSET_COOLDOWN"),
    ],
)
def test_risk_change_after_claim_cancels_fresh_cycle(state, blocked, reason):
    runtime_id = _rid("fresh-cancel")
    conn = _connect()
    try:
        cycle_id, dispatch_id, claim = _setup(conn, runtime_id)
        _set_risk(conn, runtime_id, state=state, blocked=blocked)
        row = _kill(conn, runtime_id, cycle_id, claim["claim_fencing_token"])
        assert row["status"] == "CANCELLED_BEFORE_EXECUTION"
        assert row["proceed"] is False
        assert row["cancel_requested"] is True
        assert row["terminal"] is True
        assert row["reason"] == reason
        assert _cancel_count(conn, runtime_id) == 1
        with conn.cursor() as cur:
            cur.execute(
                "select status,cancel_reason from public.brian_shadow_execution_claims "
                "where runtime_id=%s and dispatch_id=%s",
                (runtime_id, dispatch_id),
            )
            status, cancel_reason = cur.fetchone()
        assert status == "CANCELLED_BEFORE_EXECUTION"
        assert cancel_reason == f"phase78:{reason}"
    finally:
        conn.close()


def test_reduce_only_claim_still_proceeds_under_reducing_and_cooldown():
    for state, blocked in (("REDUCING", ()), ("ACTIVE", ("BTCUSDT",))):
        runtime_id = _rid("reduce")
        conn = _connect()
        try:
            cycle_id, _, claim = _setup(conn, runtime_id, reduce_only=True)
            _set_risk(conn, runtime_id, state=state, blocked=blocked)
            row = _kill(conn, runtime_id, cycle_id, claim["claim_fencing_token"])
            assert row["status"] == "PROCEED"
            assert row["proceed"] is True
            assert _cancel_count(conn, runtime_id) == 0
        finally:
            conn.close()


def test_halt_after_paper_started_creates_cancel_request_but_allows_resume():
    runtime_id = _rid("after-start")
    conn = _connect()
    try:
        cycle_id, dispatch_id, claim = _setup(conn, runtime_id)
        _set_stage(conn, runtime_id, cycle_id, "PAPER_APPLIED", "m")
        receipt_id = _set_risk(conn, runtime_id, state="HALTED")

        row = _kill(conn, runtime_id, cycle_id, claim["claim_fencing_token"])
        assert row["status"] == "CANCEL_REQUESTED"
        assert row["proceed"] is True
        assert row["cancel_requested"] is True
        assert row["terminal"] is False
        assert row["journal_stage"] == "PAPER_APPLIED"
        assert row["risk_receipt_id"] == receipt_id
        assert _cancel_count(conn, runtime_id) == 1

        with conn.cursor() as cur:
            cur.execute(
                "select status from public.brian_shadow_execution_claims "
                "where runtime_id=%s and dispatch_id=%s",
                (runtime_id, dispatch_id),
            )
            assert cur.fetchone()[0] == "CLAIMED"
            cur.execute(
                "select phase,reason from public.brian_shadow_execution_cancel_requests "
                "where runtime_id=%s and dispatch_id=%s",
                (runtime_id, dispatch_id),
            )
            phase, reason = cur.fetchone()
        assert phase == "AFTER_START"
        assert reason == "HALTED"
    finally:
        conn.close()


def test_healthy_in_progress_cycle_is_resume_only():
    runtime_id = _rid("resume")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
        _set_stage(conn, runtime_id, cycle_id, "LOCAL_PROJECTED", "n")
        row = _kill(conn, runtime_id, cycle_id, claim["claim_fencing_token"])
        assert row["status"] == "RESUME_ONLY"
        assert row["proceed"] is True
        assert row["journal_stage"] == "LOCAL_PROJECTED"
    finally:
        conn.close()


def test_wrong_worker_or_claim_fence_fails_closed():
    runtime_id = _rid("lost")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
        wrong_worker = _kill(
            conn,
            runtime_id,
            cycle_id,
            claim["claim_fencing_token"],
            worker="worker-b",
        )
        wrong_fence = _kill(
            conn,
            runtime_id,
            cycle_id,
            claim["claim_fencing_token"] + 1,
        )
        assert wrong_worker["status"] == "CLAIM_LOST"
        assert wrong_fence["status"] == "CLAIM_LOST"
    finally:
        conn.close()


def test_committed_runtime_returns_terminal_completed_before_new_risk_decision():
    runtime_id = _rid("completed")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
        _set_stage(conn, runtime_id, cycle_id, "COMMITTED", "z")
        _set_risk(conn, runtime_id, state="HALTED")
        row = _kill(conn, runtime_id, cycle_id, claim["claim_fencing_token"])
        assert row["status"] == "COMPLETED"
        assert row["terminal"] is True
        assert row["proceed"] is False
        assert _cancel_count(conn, runtime_id) == 0
    finally:
        conn.close()


def test_service_role_cannot_mutate_cancel_request_history():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
        _set_risk(conn, runtime_id, state="HALTED")
        assert _kill(
            conn, runtime_id, cycle_id, claim["claim_fencing_token"]
        )["status"] == "CANCELLED_BEFORE_EXECUTION"

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_shadow_execution_cancel_requests "
                    "set reason='forged' where runtime_id=%s",
                    (runtime_id,),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
