"""Real Postgres red-team tests for Phase81 cancel-recovery directives.

Phase81 is intentionally a draft SQL contract outside supabase/migrations.
The tests prove that an AFTER_START cancel request is converted into a durable
reduce-only rollback obligation only after the original cycle has an
authoritative COMMITTED state.
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
PHASE81_SQL = ROOT / "brian2026" / "sql" / "phase81_cancel_recovery_directive.sql"
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase81 Postgres tests require CI database",
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
            cur.execute(PHASE81_SQL.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _rid(label: str) -> str:
    return f"pytest-phase81-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _state(state_id: str, positions: tuple[tuple[str, float], ...]):
    return {
        "state_id": state_id,
        "position_weights": [[asset, weight] for asset, weight in positions],
    }


def _cycle(cycle_id: str, assets=("BTCUSDT",)):
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": "phase81-plan",
        "items": [
            {
                "instruction_kind": "INCREASE",
                "asset_id": asset,
                "risk_receipt": {
                    "allowed": True,
                    "reduce_only": False,
                },
                "execution_receipt": None,
                "pending_reversal": None,
                "new_risk_cash_reserved_usd": 100.0,
                "status": "fixture",
            }
            for asset in assets
        ],
        "initial_available_cash_usd": 1000.0,
        "reserved_new_risk_cash_usd": 100.0 * len(assets),
        "remaining_unreserved_cash_usd": max(0.0, 1000.0 - 100.0 * len(assets)),
        "denied_assets": [],
        "pending_reversal_assets": [],
        "cycle_id": cycle_id,
        "account_state_mutated": False,
        "shadow_only": True,
        "live_execution": False,
    }


def _checkpoint(
    cycle_id: str,
    *,
    runtime_version: int,
    journal_stage: str,
    pre_positions: tuple[tuple[str, float], ...],
    current_positions: tuple[tuple[str, float], ...],
    head_moved: bool = False,
):
    pre_state_id = _h("a")
    commit_state_id = _h("b")
    head_state_id = _h("h") if head_moved else commit_state_id

    states = {
        pre_state_id: _state(pre_state_id, pre_positions),
        commit_state_id: _state(commit_state_id, current_positions),
    }
    if head_moved:
        states[head_state_id] = _state(head_state_id, current_positions)

    transitions = [
        {
            "sequence": 1,
            "kind": "CYCLE_PROPOSED",
            "cycle_id": cycle_id,
            "before_state_id": pre_state_id,
            "after_state_id": pre_state_id,
        },
        {
            "sequence": 2,
            "kind": "RECONCILED_COMMIT",
            "cycle_id": cycle_id,
            "before_state_id": pre_state_id,
            "after_state_id": commit_state_id,
        },
    ]

    return {
        "schema_version": "brian.phase67-durable-runtime-orchestrator.v1",
        "runtime_checkpoint": {
            "shadow_ledger_manifest": {
                "head_state_id": head_state_id,
                "states": states,
                "transitions": transitions,
            }
        },
        "journal_manifest": {
            "entries": [{
                "sequence": 0,
                "cycle_id": cycle_id,
                "stage": journal_stage,
            }]
        },
        "live_execution": False,
        "checkpoint_id": _h("k"),
        "_fixture_runtime_version": runtime_version,
    }


def _receipt(receipt_id: str, *, state: str, blocked=()):
    return {
        "receipt_id": receipt_id,
        "trading_state": state,
        "blocked_assets": list(blocked),
    }


def _setup(
    conn,
    runtime_id: str,
    *,
    cancel_reason="REDUCING_NEW_RISK",
    cancel_blocked=(),
    current_risk_state="REDUCING",
    pre_positions=(("BTCUSDT", 0.10),),
    current_positions=(("BTCUSDT", 0.25),),
    assets=("BTCUSDT",),
    journal_stage="COMMITTED",
    head_moved=False,
    include_cancel=True,
    runtime_version=5,
):
    cycle_id = _h("c")
    dispatch_id = _h("d")
    cancel_receipt_id = _h("r")
    current_receipt_id = _h("s")
    cancel_entry_id = _h("e")
    current_entry_id = _h("f")
    checkpoint = _checkpoint(
        cycle_id,
        runtime_version=runtime_version,
        journal_stage=journal_stage,
        pre_positions=tuple(pre_positions),
        current_positions=tuple(current_positions),
        head_moved=head_moved,
    )
    head_state_id = checkpoint["runtime_checkpoint"]["shadow_ledger_manifest"]["head_state_id"]

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
                checkpoint["checkpoint_id"],
                Json(checkpoint),
                _h("j"),
                head_state_id,
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_runtime_cycles(
              runtime_id,cycle_id,cycle_hash,cycle_payload,first_seen_version
            ) values (%s,%s,%s,%s,1)
            """,
            (runtime_id, cycle_id, _h("x"), Json(_cycle(cycle_id, assets))),
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
                _h("w"),
                _h("l"),
                _h("u"),
            ),
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
            (runtime_id, dispatch_id, cycle_id, _h("u")),
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
                cancel_entry_id,
                _h("q"),
                cancel_receipt_id,
                "HALTED" if cancel_reason == "HALTED" else "ACTIVE",
                cancel_reason == "HALTED",
                Json({
                    "receipt": _receipt(
                        cancel_receipt_id,
                        state="HALTED" if cancel_reason == "HALTED" else "ACTIVE",
                        blocked=cancel_blocked,
                    )
                }),
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
              %s,1,%s,%s,%s,%s,2.0,
              'ACTIVE',%s,%s,%s,2
            )
            """,
            (
                runtime_id,
                current_entry_id,
                cancel_entry_id,
                _h("q"),
                current_receipt_id,
                current_risk_state,
                current_risk_state == "HALTED",
                Json({
                    "receipt": _receipt(
                        current_receipt_id,
                        state=current_risk_state,
                        blocked=cancel_blocked,
                    )
                }),
            ),
        )
        cur.execute(
            """
            insert into public.brian_operational_risk_heads(
              runtime_id,version,ledger_hash,manifest,policy_hash,
              head_entry_id,current_state,halt_latched,updated_at
            ) values (%s,2,%s,'{}'::jsonb,%s,%s,%s,%s,now())
            """,
            (
                runtime_id,
                _h("m"),
                _h("q"),
                current_entry_id,
                current_risk_state,
                current_risk_state == "HALTED",
            ),
        )

        if include_cancel:
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
                  'AFTER_START',%s,now(),
                  'worker-a',1
                )
                """,
                (
                    runtime_id,
                    dispatch_id,
                    cycle_id,
                    cancel_receipt_id,
                    cancel_reason,
                ),
            )

    return {
        "cycle_id": cycle_id,
        "dispatch_id": dispatch_id,
        "cancel_receipt_id": cancel_receipt_id,
        "current_receipt_id": current_receipt_id,
        "runtime_version": runtime_version,
    }


def _prepare(conn, runtime_id, cycle_id, expected_version):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_prepare_shadow_cancel_recovery(%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", 1, cycle_id, expected_version),
        )
        return cur.fetchone()[0]


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


def test_preexisting_exposure_is_preserved_by_reduce_only_rollback_target():
    runtime_id = _rid("preexisting")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        row = _prepare(
            conn,
            runtime_id,
            ctx["cycle_id"],
            ctx["runtime_version"],
        )
        assert row["prepared"] is True
        assert row["status"] == "PREPARED"
        assert row["recovery_status"] == "READY_REDUCE_ONLY"
        assert len(row["recovery_legs"]) == 1
        leg = row["recovery_legs"][0]
        assert leg["asset_id"] == "BTCUSDT"
        assert float(leg["before_weight"]) == pytest.approx(0.10)
        assert float(leg["current_weight"]) == pytest.approx(0.25)
        assert float(leg["target_weight"]) == pytest.approx(0.10)
        assert float(leg["reduce_weight"]) == pytest.approx(0.15)
        assert leg["reduce_only"] is True
        assert leg["order_direction"] == -1
    finally:
        conn.close()


def test_halted_state_prepares_obligation_but_never_marks_it_ready_to_submit():
    runtime_id = _rid("halted")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            cancel_reason="HALTED",
            current_risk_state="HALTED",
        )
        row = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        assert row["prepared"] is True
        assert row["recovery_status"] == "WAIT_RISK_RELEASE"
        assert row["current_risk_state"] == "HALTED"
        assert row["recovery_legs"]
    finally:
        conn.close()


def test_asset_cooldown_rolls_back_only_asset_named_by_cancel_risk_evidence():
    runtime_id = _rid("cooldown")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            cancel_reason="ASSET_COOLDOWN",
            cancel_blocked=("BTCUSDT",),
            current_risk_state="ACTIVE",
            pre_positions=(("BTCUSDT", 0.10), ("ETHUSDT", 0.10)),
            current_positions=(("BTCUSDT", 0.25), ("ETHUSDT", 0.30)),
            assets=("BTCUSDT", "ETHUSDT"),
        )
        row = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        assert row["recovery_status"] == "READY_REDUCE_ONLY"
        assert [leg["asset_id"] for leg in row["recovery_legs"]] == ["BTCUSDT"]
    finally:
        conn.close()


def test_no_recovery_is_required_when_exposure_did_not_increase_vs_precycle_state():
    runtime_id = _rid("no-recovery")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            pre_positions=(("BTCUSDT", 0.10),),
            current_positions=(("BTCUSDT", 0.05),),
        )
        row = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        assert row["prepared"] is True
        assert row["recovery_status"] == "NO_RECOVERY_REQUIRED"
        assert row["recovery_legs"] == []
    finally:
        conn.close()


def test_position_sign_flip_is_never_auto_recovered_and_requires_manual_review():
    runtime_id = _rid("manual")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            pre_positions=(("BTCUSDT", 0.10),),
            current_positions=(("BTCUSDT", -0.25),),
        )
        row = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        assert row["prepared"] is True
        assert row["recovery_status"] == "MANUAL_REVIEW"
        assert row["unsafe_assets"][0]["asset_id"] == "BTCUSDT"
        assert row["unsafe_assets"][0]["reason"] == "ROLLBACK_NOT_REDUCE_ONLY"
    finally:
        conn.close()


def test_original_cycle_must_be_authoritatively_committed_before_recovery_is_prepared():
    runtime_id = _rid("wait")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            journal_stage="PAPER_APPLIED",
        )
        row = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        assert row["prepared"] is False
        assert row["status"] == "WAIT_ORIGINAL_COMMIT"
        assert _count(
            conn,
            "brian_shadow_cancel_recovery_directives",
            runtime_id,
        ) == 0
    finally:
        conn.close()


def test_head_moved_after_original_commit_refuses_automatic_rollback():
    runtime_id = _rid("head-moved")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id, head_moved=True)
        row = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        assert row["prepared"] is False
        assert row["status"] == "HEAD_MOVED"
        assert _count(
            conn,
            "brian_shadow_cancel_recovery_directives",
            runtime_id,
        ) == 0
    finally:
        conn.close()


def test_no_after_start_cancel_request_produces_no_recovery_directive():
    runtime_id = _rid("none")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id, include_cancel=False)
        row = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        assert row["prepared"] is False
        assert row["status"] == "NO_CANCEL_REQUEST"
    finally:
        conn.close()


def test_runtime_version_conflict_fails_closed_before_recovery_evidence_is_created():
    runtime_id = _rid("cas")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        row = _prepare(
            conn,
            runtime_id,
            ctx["cycle_id"],
            ctx["runtime_version"] - 1,
        )
        assert row["prepared"] is False
        assert row["status"] == "RUNTIME_VERSION_CONFLICT"
        assert _count(
            conn,
            "brian_shadow_cancel_recovery_directives",
            runtime_id,
        ) == 0
    finally:
        conn.close()


def test_exact_prepare_retry_is_idempotent_and_preserves_original_directive():
    runtime_id = _rid("duplicate")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        first = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        duplicate = _prepare(conn, runtime_id, ctx["cycle_id"], ctx["runtime_version"])
        assert first["status"] == "PREPARED"
        assert duplicate["prepared"] is True
        assert duplicate["duplicate"] is True
        assert duplicate["status"] == "DUPLICATE"
        assert duplicate["recovery_status"] == first["recovery_status"]
        assert duplicate["recovery_legs"] == first["recovery_legs"]
        assert _count(
            conn,
            "brian_shadow_cancel_recovery_directives",
            runtime_id,
        ) == 1
    finally:
        conn.close()


def test_concurrent_exact_prepare_has_one_prepared_and_one_duplicate():
    runtime_id = _rid("race")
    setup = _connect()
    try:
        ctx = _setup(setup, runtime_id)
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(_):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return _prepare(
                conn,
                runtime_id,
                ctx["cycle_id"],
                ctx["runtime_version"],
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "PREPARED" for row in rows) == 1
    assert sum(row["status"] == "DUPLICATE" for row in rows) == 1


def test_service_role_cannot_mutate_recovery_directive_history_directly():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        assert _prepare(
            conn,
            runtime_id,
            ctx["cycle_id"],
            ctx["runtime_version"],
        )["prepared"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    update public.brian_shadow_cancel_recovery_directives
                    set recovery_status='NO_RECOVERY_REQUIRED'
                    where runtime_id=%s
                    """,
                    (runtime_id,),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
