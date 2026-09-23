"""Real Postgres tests for Phase79 atomic execution-start point-of-no-return.

Phase79 SQL is intentionally a draft contract outside supabase/migrations.
CI loads the official Phase70/73/75/76/77/78 migrations first and then applies
this draft SQL. At rollout freeze it must be converted with
`supabase migration new` before any live database deployment.
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
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase79 Postgres tests require CI database",
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
            cur.execute(PHASE79_SQL.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _rid(label: str) -> str:
    return f"pytest-phase79-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _cycle(cycle_id: str, *, reduce_only: bool = False):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": "phase79-plan",
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
    entries = [{
        "sequence": 0,
        "stage": "CYCLE_CREATED",
        "cycle_id": cycle_id,
    }]
    if stage != "CYCLE_CREATED":
        entries.append({
            "sequence": 1,
            "stage": stage,
            "cycle_id": cycle_id,
        })
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


def _setup(
    conn,
    runtime_id: str,
    *,
    stage="CYCLE_CREATED",
    reduce_only=False,
):
    cycle_id = _h("c")
    dispatch_id = _h("d")
    checkpoint_id = _h("k")
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_runtime_heads(
              runtime_id,version,checkpoint_id,checkpoint_payload,
              journal_hash,head_state_id,pending_cycle_id,
              owner_token,fencing_token,acquired_at,lease_until,updated_at
            ) values (%s,2,%s,%s,%s,%s,null,'owner-a',1,now(),now()+interval '5 minutes',now())
            """,
            (
                runtime_id,
                checkpoint_id,
                Json(_checkpoint(cycle_id, stage=stage)),
                _h("j"),
                _h("s"),
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_runtime_cycles(
              runtime_id,cycle_id,cycle_hash,cycle_payload,first_seen_version
            ) values (%s,%s,%s,%s,1)
            """,
            (runtime_id, cycle_id, _h("x"), Json(_cycle(cycle_id, reduce_only=reduce_only))),
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
                runtime_id,
                dispatch_id,
                cycle_id,
                _h("g"),
                _h("p"),
                checkpoint_id,
                _h("l"),
                _h("r"),
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
                runtime_id,
                _h("a"),
                _h("q"),
                _h("r"),
                TS,
                Json({"receipt": _receipt(_h("r"))}),
            ),
        )

    claim = _claim(conn, runtime_id, cycle_id, "worker-a")
    assert claim["claimed"] is True
    return cycle_id, dispatch_id, claim


def _claim(conn, runtime_id, cycle_id, worker, seconds=60):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_claim_shadow_execution_dispatch(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, worker, seconds),
        )
        return cur.fetchone()[0]


def _start(conn, runtime_id, cycle_id, worker, claim_fence):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_mark_shadow_execution_started(%s,%s,%s,%s,%s,%s)",
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


def _set_risk(conn, runtime_id, *, state, blocked=()):
    receipt_id = _h(
        "h" if state == "HALTED"
        else "u" if state == "REDUCING"
        else "b"
    )
    entry_id = _h(
        "i" if state == "HALTED"
        else "v" if state == "REDUCING"
        else "n"
    )
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
                runtime_id,
                entry_id,
                _h("a"),
                _h("q"),
                receipt_id,
                TS + 1,
                state,
                state == "HALTED",
                Json({
                    "receipt": _receipt(
                        receipt_id,
                        state=state,
                        blocked=blocked,
                    )
                }),
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


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_healthy_claim_atomically_marks_started_and_retry_is_idempotent():
    runtime_id = _rid("healthy")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
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
        assert first["phase78_status"] == "PROCEED"
        assert first["journal_stage"] == "CYCLE_CREATED"

        assert duplicate["started"] is True
        assert duplicate["duplicate"] is True
        assert duplicate["status"] == "STARTED_ALREADY"
        assert _count(conn, "brian_shadow_execution_starts", runtime_id) == 1
    finally:
        conn.close()


def test_concurrent_exact_start_has_one_started_and_one_duplicate():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        cycle_id, _, claim = _setup(setup, runtime_id)
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

    conn = _connect()
    try:
        assert _count(conn, "brian_shadow_execution_starts", runtime_id) == 1
    finally:
        conn.close()


def test_wrong_worker_or_claim_fence_cannot_exploit_existing_start_duplicate():
    runtime_id = _rid("ownership")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
        claim_fence = claim["claim_fencing_token"]
        assert _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim_fence,
        )["status"] == "STARTED"

        wrong_worker = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-b",
            claim_fence,
        )
        wrong_fence = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim_fence + 1,
        )
        assert wrong_worker["started"] is False
        assert wrong_worker["status"] == "CLAIM_LOST"
        assert wrong_fence["started"] is False
        assert wrong_fence["status"] == "CLAIM_LOST"
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
def test_risk_change_after_claim_before_start_vetoes_without_started_row(
    state,
    blocked,
    reason,
):
    runtime_id = _rid(f"veto-{state.lower()}")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
        _set_risk(conn, runtime_id, state=state, blocked=blocked)

        row = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert row["started"] is False
        assert row["terminal"] is True
        assert row["status"] == "CANCELLED_BEFORE_EXECUTION"
        assert row["reason"] == reason
        assert _count(conn, "brian_shadow_execution_starts", runtime_id) == 0
    finally:
        conn.close()


def test_reduce_only_cycle_still_starts_under_reducing_or_asset_cooldown():
    for state, blocked in (
        ("REDUCING", ()),
        ("ACTIVE", ("BTCUSDT",)),
    ):
        runtime_id = _rid("reduce-only")
        conn = _connect()
        try:
            cycle_id, _, claim = _setup(
                conn,
                runtime_id,
                reduce_only=True,
            )
            _set_risk(conn, runtime_id, state=state, blocked=blocked)
            row = _start(
                conn,
                runtime_id,
                cycle_id,
                "worker-a",
                claim["claim_fencing_token"],
            )
            assert row["started"] is True
            assert row["status"] == "STARTED"
            assert row["phase78_status"] == "PROCEED"
        finally:
            conn.close()


def test_in_progress_halted_cycle_marks_started_resume_with_cancel_request():
    runtime_id = _rid("resume")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(
            conn,
            runtime_id,
            stage="PAPER_APPLIED",
        )
        assert claim["resume_only"] is True
        receipt_id = _set_risk(conn, runtime_id, state="HALTED")

        row = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert row["started"] is True
        assert row["status"] == "STARTED_RESUME"
        assert row["resume_only"] is True
        assert row["cancel_requested"] is True
        assert row["phase78_status"] == "CANCEL_REQUESTED"
        assert row["risk_receipt_id"] == receipt_id

        with conn.cursor() as cur:
            cur.execute(
                """
                select cancel_requested_at_start,cancel_reason_at_start,
                       phase78_status_at_start,journal_stage_at_start
                from public.brian_shadow_execution_starts
                where runtime_id=%s and cycle_id=%s
                """,
                (runtime_id, cycle_id),
            )
            requested, reason, phase78_status, stage = cur.fetchone()
        assert requested is True
        assert reason == "HALTED"
        assert phase78_status == "CANCEL_REQUESTED"
        assert stage == "PAPER_APPLIED"
    finally:
        conn.close()


def test_later_halt_is_post_start_cancel_request_and_start_evidence_remains():
    runtime_id = _rid("post-start")
    conn = _connect()
    try:
        cycle_id, _, claim = _setup(conn, runtime_id)
        claim_fence = claim["claim_fencing_token"]
        assert _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim_fence,
        )["status"] == "STARTED"

        # Durable journal progress is what makes the later Phase78 decision
        # explicitly post-start rather than a pre-execution rollback.
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.brian_shadow_runtime_heads
                set checkpoint_payload=%s, updated_at=now()
                where runtime_id=%s
                """,
                (
                    Json(_checkpoint(
                        cycle_id,
                        stage="PAPER_APPLIED",
                        checkpoint_char="m",
                    )),
                    runtime_id,
                ),
            )

        _set_risk(conn, runtime_id, state="HALTED")
        decision = _kill(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim_fence,
        )
        assert decision["status"] == "CANCEL_REQUESTED"
        assert decision["proceed"] is True
        assert decision["terminal"] is False
        assert decision["reason"] == "HALTED"
        assert _count(conn, "brian_shadow_execution_starts", runtime_id) == 1
        assert _count(conn, "brian_shadow_execution_cancel_requests", runtime_id) == 1
    finally:
        conn.close()


def test_read_rpc_returns_exact_persisted_start_anchors():
    runtime_id = _rid("read")
    conn = _connect()
    try:
        cycle_id, dispatch_id, claim = _setup(conn, runtime_id)
        started = _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )
        assert started["started"] is True

        with conn.cursor() as cur:
            cur.execute(
                "select public.brian_read_shadow_execution_start(%s,%s)",
                (runtime_id, cycle_id),
            )
            row = cur.fetchone()[0]
        assert row["runtime_id"] == runtime_id
        assert row["dispatch_id"] == dispatch_id
        assert row["cycle_id"] == cycle_id
        assert row["worker_token"] == "worker-a"
        assert row["claim_fencing_token"] == claim["claim_fencing_token"]
        assert row["phase78_status_at_start"] == "PROCEED"
        assert row["shadow_only"] is True
        assert row["live_execution"] is False
    finally:
        conn.close()


def test_service_role_cannot_mutate_start_history_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        cycle_id, dispatch_id, claim = _setup(conn, runtime_id)
        assert _start(
            conn,
            runtime_id,
            cycle_id,
            "worker-a",
            claim["claim_fencing_token"],
        )["started"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    update public.brian_shadow_execution_starts
                    set worker_token='attacker'
                    where runtime_id=%s and dispatch_id=%s
                    """,
                    (runtime_id, dispatch_id),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
