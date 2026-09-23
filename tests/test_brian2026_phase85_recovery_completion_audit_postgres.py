"""Real Postgres tests for Phase85 authoritative recovery completion audit."""

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
    ROOT / "brian2026" / "sql" / "phase85_recovery_completion_audit.sql",
)
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; Phase85 Postgres tests require CI database",
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
    return f"pytest-phase85-{label}-{uuid.uuid4().hex[:10]}"


def _h(char: str) -> str:
    return char * 64


def _leg(*, direction=1):
    if direction > 0:
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
    return {
        "asset_id": "BTCUSDT",
        "before_weight": -0.10,
        "current_weight": -0.25,
        "target_weight": -0.10,
        "reduce_weight": 0.15,
        "current_direction": -1,
        "order_direction": 1,
        "reduce_only": True,
    }


def _checkpoint(
    recovery_cycle_id: str,
    *,
    final_quantity: float,
    fill_side: str,
    fill_quantity: float,
    final_weight: float,
    include_fill: bool = True,
):
    fill_sequence = []
    if include_fill:
        fill_sequence.append({
            "fill_id": _h("f"),
            "paper_order_id": _h("q"),
            "cycle_id": recovery_cycle_id,
            "asset_id": "BTCUSDT",
            "side": fill_side,
            "quantity_base": fill_quantity,
            "price": 100.0,
            "fee_quote": 0.01,
            "timestamp": 1_760_000_000.0,
        })

    final_positions = []
    if abs(final_quantity) > 1e-12:
        final_positions.append({
            "asset_id": "BTCUSDT",
            "quantity": final_quantity,
            "avg_entry_price": 100.0,
            "realized_pnl_quote": 0.0,
            "source_fill_ids": [_h("f")] if include_fill else [],
        })

    final_state_id = _h("z")
    return {
        "schema_version": "brian.phase67-durable-runtime-orchestrator.v1",
        "runtime_checkpoint": {
            "schema_version": "brian.phase63-crash-recovery.v1",
            "paper": {
                "schema_version": "brian.phase63-crash-recovery.v1",
                "config": {
                    "account_id": "BRIAN-PAPER",
                    "starting_cash_usd": 500.0,
                    "fee_bps": 10.0,
                    "allow_short": True,
                },
                "cash_usd": 500.0,
                "state_version": 1,
                "fill_sequence": fill_sequence,
                "cycle_receipts": [],
                "final_positions": final_positions,
                "paper_only": True,
                "live_execution": False,
                "checkpoint_id": _h("p"),
            },
            "shadow_ledger_manifest": {
                "head_state_id": final_state_id,
                "pending_cycle_id": None,
                "states": {
                    final_state_id: {
                        "state_id": final_state_id,
                        "position_weights": (
                            [] if abs(final_weight) <= 1e-12
                            else [["BTCUSDT", final_weight]]
                        ),
                    }
                },
                "transitions": [{
                    "kind": "RECONCILED_COMMIT",
                    "cycle_id": recovery_cycle_id,
                    "after_state_id": final_state_id,
                }],
            },
            "pending_cycle_id": None,
            "live_execution": False,
        },
        "journal_manifest": {
            "append_only": True,
            "cycles": {recovery_cycle_id: {
                "cycle_id": recovery_cycle_id,
                "account_state_mutated": False,
                "shadow_only": True,
                "live_execution": False,
            }},
            "entries": [{
                "sequence": 0,
                "cycle_id": recovery_cycle_id,
                "stage": "COMMITTED",
            }],
            "journal_hash": _h("j"),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h("k"),
    }


def _setup(
    conn,
    runtime_id: str,
    *,
    direction=1,
    final_quantity=1.0,
    fill_side="SELL",
    fill_quantity=1.5,
    final_weight=0.10,
    include_fill=True,
):
    original_cycle_id = _h("o")
    recovery_cycle_id = _h("c")
    dispatch_id = _h("d")
    cancel_receipt_id = _h("r")
    checkpoint = _checkpoint(
        recovery_cycle_id,
        final_quantity=final_quantity,
        fill_side=fill_side,
        fill_quantity=fill_quantity,
        final_weight=final_weight,
        include_fill=include_fill,
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_shadow_runtime_heads(
              runtime_id,version,checkpoint_id,checkpoint_payload,
              journal_hash,head_state_id,pending_cycle_id,
              owner_token,fencing_token,acquired_at,lease_until,updated_at
            ) values (
              %s,7,%s,%s,%s,%s,null,
              'owner-a',1,now(),now()+interval '5 minutes',now()
            )
            """,
            (
                runtime_id,
                checkpoint["checkpoint_id"],
                Json(checkpoint),
                checkpoint["journal_manifest"]["journal_hash"],
                _h("z"),
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
                _h("s"),
                Json([_leg(direction=direction)]),
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
              recovery_cycle_id,progress_runtime_version,
              progress_head_state_id,progress_checkpoint_id,
              updated_at
            ) values (
              %s,%s,%s,%s,
              'CLAIMED','worker-a',4,
              now(),now()+interval '5 minutes',
              5,%s,1,%s,'REDUCING',
              %s,7,%s,%s,now()
            )
            """,
            (
                runtime_id,
                dispatch_id,
                original_cycle_id,
                cancel_receipt_id,
                _h("h"),
                _h("s"),
                recovery_cycle_id,
                _h("z"),
                checkpoint["checkpoint_id"],
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
              'worker-a',4,
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
                _h("s"),
                _h("h"),
                Json([_leg(direction=direction)]),
            ),
        )
        cur.execute(
            """
            insert into public.brian_shadow_recovery_commit_events(
              runtime_id,dispatch_id,original_cycle_id,recovery_cycle_id,
              cancel_risk_receipt_id,worker_token,
              recovery_claim_fencing_token,runtime_fencing_token,
              expected_runtime_version,committed_runtime_version,
              checkpoint_id,recovery_journal_stage,event,observed_at
            ) values (
              %s,%s,%s,%s,
              %s,'worker-a',
              4,1,
              6,7,
              %s,'COMMITTED','RECOVERY_COMMITTED_PENDING_AUDIT',now()
            )
            """,
            (
                runtime_id,
                dispatch_id,
                original_cycle_id,
                recovery_cycle_id,
                cancel_receipt_id,
                checkpoint["checkpoint_id"],
            ),
        )

    return {
        "original_cycle_id": original_cycle_id,
        "recovery_cycle_id": recovery_cycle_id,
        "dispatch_id": dispatch_id,
        "cancel_receipt_id": cancel_receipt_id,
        "checkpoint_id": checkpoint["checkpoint_id"],
    }


def _certify(conn, runtime_id, ctx, *, expected_checkpoint_id=None):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_certify_shadow_recovery_completion("
            "%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                1,
                ctx["original_cycle_id"],
                ctx["recovery_cycle_id"],
                expected_checkpoint_id or ctx["checkpoint_id"],
            ),
        )
        return cur.fetchone()[0]


def _claim(conn, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            select status,claim_until,completed_at,completion_ref
            from public.brian_shadow_recovery_claims
            where runtime_id=%s
            """,
            (runtime_id,),
        )
        return cur.fetchone()


def _count(conn, table, runtime_id):
    with conn.cursor() as cur:
        cur.execute(
            f"select count(*) from public.{table} where runtime_id=%s",
            (runtime_id,),
        )
        return int(cur.fetchone()[0])


@pytest.mark.parametrize(
    ("direction", "final_quantity", "fill_side", "fill_quantity", "final_weight"),
    [
        (1, 1.0, "SELL", 1.5, 0.10),
        (-1, -1.0, "BUY", 1.5, -0.10),
        (1, 0.0, "SELL", 2.5, 0.0),
    ],
)
def test_valid_long_short_and_flat_recovery_are_certified(
    direction,
    final_quantity,
    fill_side,
    fill_quantity,
    final_weight,
):
    runtime_id = _rid("valid")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            direction=direction,
            final_quantity=final_quantity,
            fill_side=fill_side,
            fill_quantity=fill_quantity,
            final_weight=final_weight,
        )
        row = _certify(conn, runtime_id, ctx)
        assert row["certified"] is True
        assert row["status"] == "CERTIFIED"
        assert row["recovery_fill_count"] == 1
        assert row["leg_audits"][0]["direction_preserved"] is True
        assert row["leg_audits"][0]["quantity_exposure_reduced"] is True

        claim = _claim(conn, runtime_id)
        assert claim[0] == "COMPLETED"
        assert claim[1] is None
        assert claim[2] is not None
        assert claim[3] == ctx["checkpoint_id"]
        assert _count(
            conn,
            "brian_shadow_recovery_completion_certificates",
            runtime_id,
        ) == 1
    finally:
        conn.close()


def test_overreduction_direction_flip_fails_audit_and_claim_stays_open():
    runtime_id = _rid("flip")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            direction=1,
            final_quantity=-0.5,
            fill_side="SELL",
            fill_quantity=3.0,
            final_weight=-0.05,
        )
        row = _certify(conn, runtime_id, ctx)
        assert row["certified"] is False
        assert row["status"] == "AUDIT_FAILED"
        reasons = {failure["reason"] for failure in row["failures"]}
        assert "RECOVERY_OVERREDUCED_AND_FLIPPED_QUANTITY" in reasons
        assert _claim(conn, runtime_id)[0] == "CLAIMED"
        assert _count(
            conn,
            "brian_shadow_recovery_completion_certificates",
            runtime_id,
        ) == 0
    finally:
        conn.close()


def test_missing_recovery_fill_fails_audit():
    runtime_id = _rid("missing-fill")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            include_fill=False,
            final_quantity=1.0,
            final_weight=0.10,
        )
        row = _certify(conn, runtime_id, ctx)
        assert row["certified"] is False
        assert row["status"] == "AUDIT_FAILED"
        assert row["failures"][0]["reason"] == "RECOVERY_FILL_MISSING"
        assert _claim(conn, runtime_id)[0] == "CLAIMED"
    finally:
        conn.close()


def test_phase60_direction_flip_fails_even_if_paper_quantity_reduced_safely():
    runtime_id = _rid("weight-flip")
    conn = _connect()
    try:
        ctx = _setup(
            conn,
            runtime_id,
            final_quantity=1.0,
            fill_side="SELL",
            fill_quantity=1.5,
            final_weight=-0.10,
        )
        row = _certify(conn, runtime_id, ctx)
        assert row["certified"] is False
        assert row["status"] == "AUDIT_FAILED"
        reasons = {failure["reason"] for failure in row["failures"]}
        assert "FINAL_AUTHORITATIVE_WEIGHT_FLIPPED" in reasons
    finally:
        conn.close()


def test_checkpoint_head_drift_is_rejected_before_audit():
    runtime_id = _rid("head")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        row = _certify(
            conn,
            runtime_id,
            ctx,
            expected_checkpoint_id=_h("x"),
        )
        assert row["certified"] is False
        assert row["status"] == "HEAD_MOVED"
        assert _claim(conn, runtime_id)[0] == "CLAIMED"
    finally:
        conn.close()


def test_claim_progress_anchor_drift_is_rejected():
    runtime_id = _rid("claim-drift")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.brian_shadow_recovery_claims
                set progress_runtime_version=6
                where runtime_id=%s
                """,
                (runtime_id,),
            )
        row = _certify(conn, runtime_id, ctx)
        assert row["certified"] is False
        assert row["status"] == "CLAIM_STATE_INVALID"
    finally:
        conn.close()


def test_exact_certificate_retry_is_duplicate_and_does_not_recomplete_claim():
    runtime_id = _rid("duplicate")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        first = _certify(conn, runtime_id, ctx)
        duplicate = _certify(conn, runtime_id, ctx)
        assert first["status"] == "CERTIFIED"
        assert duplicate["certified"] is True
        assert duplicate["duplicate"] is True
        assert duplicate["status"] == "DUPLICATE"
        assert _count(
            conn,
            "brian_shadow_recovery_completion_certificates",
            runtime_id,
        ) == 1
    finally:
        conn.close()


def test_concurrent_certificate_attempts_are_one_certified_and_one_duplicate():
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
            return _certify(conn, runtime_id, ctx)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, range(2)))

    assert sum(row["status"] == "CERTIFIED" for row in rows) == 1
    assert sum(row["status"] == "DUPLICATE" for row in rows) == 1


def test_service_role_cannot_mutate_completion_certificate_history():
    runtime_id = _rid("priv")
    conn = _connect()
    try:
        ctx = _setup(conn, runtime_id)
        assert _certify(conn, runtime_id, ctx)["certified"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    update public.brian_shadow_recovery_completion_certificates
                    set recovery_fill_count=999
                    where runtime_id=%s
                    """,
                    (runtime_id,),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
