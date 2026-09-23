"""Real Postgres tests for Phase78 execution-start and post-start kill semantics.

Phase78 SQL is still a draft contract, not an official Supabase migration. CI
loads it after the deployed-order Phase70/73/75/76/77 migration fixtures.
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
)
PHASE78_SQL = (
    ROOT / "brian2026" / "sql" / "phase78_execution_start_kill_switch.sql"
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
def _apply_sql():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP)
            for migration in MIGRATIONS:
                cur.execute(migration.read_text(encoding="utf-8"))
            cur.execute(PHASE78_SQL.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _rid(label: str) -> str:
    return f"pytest-phase78-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _cycle(cycle_id: str, *, asset="BTCUSDT", reduce_only=False):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": "phase78-plan",
        "items": [{
            "instruction_kind": "REDUCE" if reduce_only else "OPEN",
            "asset_id": asset,
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
    entries = [{
        "sequence": 0,
        "stage": "CYCLE_CREATED",
        "cycle_id": cycle_id,
        "entry_id": _h("e"),
    }]
    if stage != "CYCLE_CREATED":
        entries.append({
            "sequence": 1,
            "stage": stage,
            "cycle_id": cycle_id,
            "entry_id": _h("f"),
        })
    return {
        "journal_manifest": {"entries": entries},
        "checkpoint_id": _h(checkpoint_char),
    }


def _risk_receipt(receipt_id, state, blocked_assets=()):
    return {
        "receipt_id": receipt_id,
        "trading_state": state,
        "blocked_assets": list(blocked_assets),
    }


def _setup(
    conn,
    runtime_id: str,
    *,
    risk_state="ACTIVE",
    blocked_assets=(),
    reduce_only=False,
    stage="CYCLE_CREATED",
):
    cycle_id = _h("c")
    dispatch_id = _h("d")
    checkpoint_id = _h("k")
    risk_entry_id = _h("a")
    risk_receipt_id = _h("r")
    cycle_payload = _cycle(cycle_id, reduce_only=reduce_only)
    checkpoint_payload = _checkpoint(cycle_id, stage=stage)

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_runtime_heads(
              runtime_id, version, checkpoint_id, checkpoint_payload,
              journal_hash, head_state_id, pending_cycle_id,
              owner_token, fencing_token, acquired_at, lease_until, updated_at
            ) values (%s,2,%s,%s,%s,%s,null,'owner-a',1,now(),now()+interval '5 minutes',now())
            """,
            (
                runtime_id,
                checkpoint_id,
                Json(checkpoint_payload),
                _h("j"),
                _h("s"),
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_runtime_cycles(
              runtime_id, cycle_id, cycle_hash, cycle_payload, first_seen_version
            ) values (%s,%s,%s,%s,1)
            """,
            (runtime_id, cycle_id, _h("x"), Json(cycle_payload)),
        )
        cur.execute(
            """
            insert into public.brian_shadow_execution_dispatches(
              runtime_id, dispatch_id, cycle_id, governed_result_id,
              policy_fingerprint, authorization_checkpoint_id,
              authorization_runtime_version, risk_version,
              risk_ledger_hash, risk_receipt_id, fencing_token
            ) values (%s,%s,%s,%s,%s,%s,2,1,%s,%s,1)
            """,
            (
                runtime_id,
                dispatch_id,
                cycle_id,
                _h("g"),
                _h("p"),
                checkpoint_id,
                _h("l"),
                risk_receipt_id,
            ),
        )
        cur.execute(
            """
            insert into public.brian_operational_risk_heads(
              runtime_id, version, ledger_hash, manifest, policy_hash,
              head_entry_id, current_state, halt_latched, updated_at
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
            insert into public.brian_operational_risk_entries(
              runtime_id, sequence, entry_id, previous_entry_id,
              policy_hash, receipt_id, receipt_timestamp,
              previous_state, trading_state, halt_latched,
              entry_payload, first_seen_version
            ) values (%s,0,%s,null,%s,%s,%s,'ACTIVE',%s,%s,%s,1)
            """,
            (
                runtime_id,
                risk_entry_id,
                _h("q"),
                risk_receipt_id,
                TS,
                risk_state,
                risk_state == "HALTED",
                Json({
                    "receipt": _risk_receipt(
                        risk_receipt_id,
                        risk_state,
                        blocked_assets,
                    )
                }),
            ),
        )
    return cycle_id, dispatch_id


def _replace_risk(
    conn,
    runtime_id,
    *,
    state,
    blocked_assets=(),
    entry_char="b",
    receipt_char="t",
):
    entry_id = _h(entry_char)
    receipt_id = _h(receipt_char)
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_operational_risk_entries(
              runtime_id, sequence, entry_id, previous_entry_id,
              policy_hash, receipt_id, receipt_timestamp,
              previous_state, trading_state, halt_latched,
              entry_payload, first_seen_version
            )
            select %s,1,%s,head_entry_id,%s,%s,%s,current_state,%s,%s,%s,2
            from public.brian_operational_risk_heads
            where runtime_id=%s
            """,
            (
                runtime_id,
                entry_id,
                _h("q"),
                receipt_id,
                TS + 1,
                state,
                state == "HALTED",
                Json({
                    "receipt": _risk_receipt(
                        receipt_id,
                        state,
                        blocked_assets,
                    )
                }),
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


def _claim(conn, runtime_id, cycle_id, worker="worker-a", seconds=60):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_claim_shadow_execution_dispatch(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, worker, seconds),
        )
        return cur.fetchone()[0]


def _start(conn, runtime_id, cycle_id, worker, claim_fence):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_start_shadow_execution_claim(%s,%s,%s,%s,%s,%s)",
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


def _kill(conn, runtime_id, cycle_id, worker, claim_fence):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_check_shadow_execution_kill_switch(%s,%s,%s,%s,%s,%s)",
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


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_current_claim_crosses_start_once_and_exact_retry_is_idempotent():
    runtime_id = _rid("start")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id)
        claim = _claim(conn, runtime_id, cycle_id)
        first = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        duplicate = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )

        assert first["started"] is True
        assert first["status"] == "STARTED"
        assert duplicate["started"] is True
        assert duplicate["duplicate"] is True
        assert duplicate["status"] == "STARTED_ALREADY"
        assert _count(conn, "brian_shadow_execution_starts", runtime_id) == 1
    finally:
        conn.close()


def test_concurrent_exact_start_has_one_start_and_one_duplicate():
    runtime_id = _rid("start-race")
    setup = _connect()
    try:
        cycle_id, _ = _setup(setup, runtime_id)
        claim = _claim(setup, runtime_id, cycle_id)
        claim_fence = claim["claim_fencing_token"]
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
                cycle_id,
                "worker-a",
                claim_fence,
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "STARTED" for row in rows) == 1
    assert sum(row["status"] == "STARTED_ALREADY" for row in rows) == 1


@pytest.mark.parametrize(
    ("state", "blocked", "reason"),
    [
        ("HALTED", (), "HALTED"),
        ("REDUCING", (), "REDUCING_NEW_RISK"),
        ("ACTIVE", ("BTCUSDT",), "ASSET_COOLDOWN"),
    ],
)
def test_risk_change_after_claim_but_before_start_cancels_without_start(
    state,
    blocked,
    reason,
):
    runtime_id = _rid(f"prestart-{state.lower()}")
    conn = _connect()
    try:
        cycle_id, dispatch_id = _setup(conn, runtime_id)
        claim = _claim(conn, runtime_id, cycle_id)
        _replace_risk(conn, runtime_id, state=state, blocked_assets=blocked)

        start = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert start["started"] is False
        assert start["cancelled"] is True
        assert start["status"] == "CANCELLED_BEFORE_START"
        assert start["cancel_reason"] == reason
        assert _count(conn, "brian_shadow_execution_starts", runtime_id) == 0

        with conn.cursor() as cur:
            cur.execute(
                "select status,cancel_reason from public.brian_shadow_execution_claims "
                "where runtime_id=%s and dispatch_id=%s",
                (runtime_id, dispatch_id),
            )
            status, cancel_reason = cur.fetchone()
        assert status == "CANCELLED_BEFORE_EXECUTION"
        assert cancel_reason == reason
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("state", "blocked"),
    [
        ("REDUCING", ()),
        ("ACTIVE", ("BTCUSDT",)),
    ],
)
def test_reduce_only_claim_can_start_under_reducing_or_asset_cooldown(
    state,
    blocked,
):
    runtime_id = _rid("reduce-only")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id, reduce_only=True)
        claim = _claim(conn, runtime_id, cycle_id)
        _replace_risk(conn, runtime_id, state=state, blocked_assets=blocked)

        start = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert start["started"] is True
        assert start["cancelled"] is False
        assert start["status"] == "STARTED"
    finally:
        conn.close()


def test_already_started_paper_journal_is_recovered_as_resume_even_when_halted():
    runtime_id = _rid("resume")
    conn = _connect()
    try:
        cycle_id, _ = _setup(
            conn,
            runtime_id,
            risk_state="HALTED",
            stage="PAPER_APPLIED",
        )
        claim = _claim(conn, runtime_id, cycle_id, worker="worker-recover")
        assert claim["claimed"] is True
        assert claim["resume_only"] is True

        start = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-recover",
            claim["claim_fencing_token"],
        )
        assert start["started"] is True
        assert start["cancelled"] is False
        assert start["status"] == "STARTED_RESUME"
        assert start["resume_only"] is True
        assert start["journal_stage"] == "PAPER_APPLIED"
    finally:
        conn.close()


def test_halt_after_start_creates_idempotent_kill_request_not_rollback():
    runtime_id = _rid("kill")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id)
        claim = _claim(conn, runtime_id, cycle_id)
        start = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert start["status"] == "STARTED"

        _replace_risk(conn, runtime_id, state="HALTED")
        first = _kill(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        duplicate = _kill(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )

        assert first["kill_requested"] is True
        assert first["status"] == "KILL_REQUESTED"
        assert first["reason"] == "HALTED"
        assert duplicate["kill_requested"] is True
        assert duplicate["request_sequence"] == first["request_sequence"]
        assert _count(conn, "brian_shadow_execution_kill_requests", runtime_id) == 1

        # STARTED is immutable evidence: a later kill never deletes/relabels it.
        assert _count(conn, "brian_shadow_execution_starts", runtime_id) == 1
    finally:
        conn.close()


def test_reduce_only_started_cycle_does_not_request_kill_for_reducing_or_cooldown():
    runtime_id = _rid("reduce-kill")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id, reduce_only=True)
        claim = _claim(conn, runtime_id, cycle_id)
        assert _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )["started"] is True

        _replace_risk(
            conn,
            runtime_id,
            state="REDUCING",
            blocked_assets=("BTCUSDT",),
        )
        check = _kill(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert check["kill_requested"] is False
        assert check["status"] == "CONTINUE"
        assert _count(conn, "brian_shadow_execution_kill_requests", runtime_id) == 0
    finally:
        conn.close()


def test_committed_runtime_is_terminal_even_if_risk_halts_after_start():
    runtime_id = _rid("committed")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id)
        claim = _claim(conn, runtime_id, cycle_id)
        assert _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )["started"] is True

        with conn.cursor() as cur:
            cur.execute(
                """
                update public.brian_shadow_runtime_heads
                set checkpoint_id=%s,
                    checkpoint_payload=%s,
                    updated_at=now()
                where runtime_id=%s
                """,
                (
                    _h("z"),
                    Json(_checkpoint(
                        cycle_id,
                        stage="COMMITTED",
                        checkpoint_char="z",
                    )),
                    runtime_id,
                ),
            )
        _replace_risk(conn, runtime_id, state="HALTED")

        check = _kill(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert check["kill_requested"] is False
        assert check["status"] == "COMMITTED"
        assert _count(conn, "brian_shadow_execution_kill_requests", runtime_id) == 0
    finally:
        conn.close()


def test_service_role_cannot_mutate_start_or_kill_history_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        cycle_id, dispatch_id = _setup(conn, runtime_id)
        claim = _claim(conn, runtime_id, cycle_id)
        assert _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )["started"] is True
        _replace_risk(conn, runtime_id, state="HALTED")
        assert _kill(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )["kill_requested"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "delete from public.brian_shadow_execution_starts "
                    "where runtime_id=%s and dispatch_id=%s",
                    (runtime_id, dispatch_id),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
