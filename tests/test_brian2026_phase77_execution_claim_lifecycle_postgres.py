"""Real Postgres tests for Phase77 fenced execution claims and risk kill-switch."""

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
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase77 Postgres tests require CI database",
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
    return f"pytest-phase77-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _cycle(cycle_id: str, *, asset="BTCUSDT", reduce_only=False):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": "phase77-plan",
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
        "journal_manifest": {
            "entries": entries,
        },
        "checkpoint_id": _h(checkpoint_char),
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

    receipt = {
        "receipt_id": risk_receipt_id,
        "trading_state": risk_state,
        "blocked_assets": list(blocked_assets),
    }
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
                Json({"receipt": receipt}),
            ),
        )
    return cycle_id, dispatch_id


def _claim(conn, runtime_id, cycle_id, worker, seconds=30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_claim_shadow_execution_dispatch(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, worker, seconds),
        )
        return cur.fetchone()[0]


def _renew(conn, runtime_id, cycle_id, worker, claim_fence, seconds=30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_renew_shadow_execution_claim(%s,%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, worker, claim_fence, seconds),
        )
        return cur.fetchone()[0]


def _complete(conn, runtime_id, cycle_id, worker, claim_fence):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_complete_shadow_execution_claim(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, worker, claim_fence),
        )
        return cur.fetchone()[0]


def test_active_cycle_claims_once_and_blocks_second_worker():
    runtime_id = _rid("active")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id)
        first = _claim(conn, runtime_id, cycle_id, "worker-a")
        same = _claim(conn, runtime_id, cycle_id, "worker-a")
        other = _claim(conn, runtime_id, cycle_id, "worker-b")

        assert first["claimed"] is True
        assert first["status"] == "CLAIMED"
        assert first["claim_fencing_token"] == 1
        assert same["status"] == "ALREADY_CLAIMED"
        assert same["claim_fencing_token"] == 1
        assert other["claimed"] is False
        assert other["status"] == "BLOCKED_ACTIVE"
    finally:
        conn.close()


def test_concurrent_two_workers_have_one_claim_owner():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        cycle_id, _ = _setup(setup, runtime_id)
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(worker):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return worker, _claim(conn, runtime_id, cycle_id, worker)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, ("worker-a", "worker-b")))

    assert sum(row["claimed"] is True for _, row in rows) == 1
    assert sum(row["status"] == "BLOCKED_ACTIVE" for _, row in rows) == 1


def test_expired_claim_takeover_increments_claim_fence():
    runtime_id = _rid("takeover")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id)
        first = _claim(conn, runtime_id, cycle_id, "worker-a", 1)
        assert first["claim_fencing_token"] == 1
        time.sleep(1.15)
        takeover = _claim(conn, runtime_id, cycle_id, "worker-b", 30)
        assert takeover["claimed"] is True
        assert takeover["status"] == "EXPIRED_RECOVERY"
        assert takeover["claim_fencing_token"] == 2
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("risk_state", "blocked", "reduce_only", "reason"),
    [
        ("HALTED", (), False, "HALTED"),
        ("REDUCING", (), False, "REDUCING_NEW_RISK"),
        ("ACTIVE", ("BTCUSDT",), False, "ASSET_COOLDOWN"),
    ],
)
def test_fresh_claim_is_cancelled_when_current_risk_no_longer_allows_new_risk(
    risk_state, blocked, reduce_only, reason
):
    runtime_id = _rid("cancel")
    conn = _connect()
    try:
        cycle_id, _ = _setup(
            conn,
            runtime_id,
            risk_state=risk_state,
            blocked_assets=blocked,
            reduce_only=reduce_only,
        )
        row = _claim(conn, runtime_id, cycle_id, "worker-a")
        assert row["claimed"] is False
        assert row["cancelled"] is True
        assert row["terminal"] is True
        assert row["status"] == "CANCELLED_BEFORE_EXECUTION"
        assert row["cancel_reason"] == reason
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("risk_state", "blocked"),
    [
        ("REDUCING", ()),
        ("ACTIVE", ("BTCUSDT",)),
    ],
)
def test_reduce_only_cycle_can_claim_under_reducing_or_asset_cooldown(risk_state, blocked):
    runtime_id = _rid("reduce")
    conn = _connect()
    try:
        cycle_id, _ = _setup(
            conn,
            runtime_id,
            risk_state=risk_state,
            blocked_assets=blocked,
            reduce_only=True,
        )
        row = _claim(conn, runtime_id, cycle_id, "worker-a")
        assert row["claimed"] is True
        assert row["status"] == "CLAIMED"
    finally:
        conn.close()


def test_expired_in_progress_claim_is_resume_only_even_if_new_risk_now_halted():
    runtime_id = _rid("resume")
    conn = _connect()
    try:
        cycle_id, _ = _setup(
            conn,
            runtime_id,
            risk_state="HALTED",
            stage="PAPER_APPLIED",
        )
        # No prior claim row is required for crash recovery: persisted journal
        # already proves paper side effects started, so the claim is resume-only.
        row = _claim(conn, runtime_id, cycle_id, "worker-recover")
        assert row["claimed"] is True
        assert row["resume_only"] is True
        assert row["status"] == "CLAIMED_RESUME"
        assert row["journal_stage"] == "PAPER_APPLIED"
    finally:
        conn.close()


def test_claim_renewal_requires_exact_worker_and_claim_fence():
    runtime_id = _rid("renew")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id)
        claim = _claim(conn, runtime_id, cycle_id, "worker-a")
        bad_worker = _renew(
            conn, runtime_id, cycle_id, "worker-b", claim["claim_fencing_token"]
        )
        bad_fence = _renew(
            conn, runtime_id, cycle_id, "worker-a", claim["claim_fencing_token"] + 1
        )
        good = _renew(
            conn, runtime_id, cycle_id, "worker-a", claim["claim_fencing_token"], 60
        )
        assert bad_worker["renewed"] is False
        assert bad_fence["renewed"] is False
        assert good["renewed"] is True
        assert good["status"] == "RENEWED"
    finally:
        conn.close()


def test_completion_requires_committed_runtime_and_exact_claim_owner():
    runtime_id = _rid("complete")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id)
        claim = _claim(conn, runtime_id, cycle_id, "worker-a", 60)

        early = _complete(
            conn, runtime_id, cycle_id, "worker-a", claim["claim_fencing_token"]
        )
        assert early["completed"] is False
        assert early["status"] == "RUNTIME_NOT_COMMITTED"

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
                    Json(_checkpoint(cycle_id, stage="COMMITTED", checkpoint_char="z")),
                    runtime_id,
                ),
            )

        wrong = _complete(
            conn, runtime_id, cycle_id, "worker-b", claim["claim_fencing_token"]
        )
        assert wrong["completed"] is False
        assert wrong["status"] == "CLAIM_LOST"

        done = _complete(
            conn, runtime_id, cycle_id, "worker-a", claim["claim_fencing_token"]
        )
        assert done["completed"] is True
        assert done["status"] == "COMPLETED"
        assert done["completion_checkpoint_id"] == _h("z")

        duplicate = _complete(
            conn, runtime_id, cycle_id, "worker-a", claim["claim_fencing_token"]
        )
        assert duplicate["completed"] is True
        assert duplicate["duplicate"] is True
    finally:
        conn.close()


def test_claim_after_committed_runtime_recovers_completed_without_reexecution():
    runtime_id = _rid("recover-complete")
    conn = _connect()
    try:
        cycle_id, _ = _setup(conn, runtime_id, stage="COMMITTED")
        row = _claim(conn, runtime_id, cycle_id, "worker-recover")
        assert row["claimed"] is False
        assert row["terminal"] is True
        assert row["status"] == "COMPLETED"
        assert row["completion_checkpoint_id"] == _h("k")
    finally:
        conn.close()


def test_service_role_cannot_mutate_claim_or_claim_event_tables_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        cycle_id, dispatch_id = _setup(conn, runtime_id)
        assert _claim(conn, runtime_id, cycle_id, "worker-a")["claimed"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_shadow_execution_claims "
                    "set status='COMPLETED' where runtime_id=%s and dispatch_id=%s",
                    (runtime_id, dispatch_id),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
