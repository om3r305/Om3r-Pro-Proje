"""Real Postgres red-team tests for Phase77 execution claim lifecycle.

These tests exercise the actual Postgres 16 locking/fencing boundary:
- concurrent claim exclusivity,
- claim expiry takeover,
- pre-execution risk kill switches,
- resume-only recovery after paper work began,
- completion fencing/idempotency,
- direct service-role mutation denial.
"""

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
    ROOT / "supabase" / "migrations" / "202609230815_brian_phase74_governed_cycle_binding.sql",
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


def _cycle(cycle_id: str, *, new_risk_asset: str | None = "BTCUSDT"):
    items = []
    if new_risk_asset is not None:
        items.append({
            "instruction_kind": "OPEN",
            "asset_id": new_risk_asset,
            "risk_receipt": {
                "action": "ALLOW",
                "trading_state": "ACTIVE",
                "asset_id": new_risk_asset,
                "requested_notional_usd": 100.0,
                "reduce_only": False,
                "reasons": [],
                "projected_position_weight": 0.1,
                "checks": [["fixture", True]],
            },
            "execution_receipt": {
                "status": "FILLED",
                "side": "BUY",
                "order_type": "MARKET",
            },
            "pending_reversal": None,
            "new_risk_cash_reserved_usd": 100.0,
            "status": "fixture",
        })
    return {
        "schema_version": "brian.phase57-shadow-execution-cycle.v1",
        "source_plan_id": f"plan-{cycle_id[:8]}",
        "items": items,
        "initial_available_cash_usd": 1000.0,
        "reserved_new_risk_cash_usd": 100.0 if items else 0.0,
        "remaining_unreserved_cash_usd": 900.0 if items else 1000.0,
        "denied_assets": [],
        "pending_reversal_assets": [],
        "cycle_id": cycle_id,
        "account_state_mutated": False,
        "shadow_only": True,
        "live_execution": False,
    }


def _entry(
    sequence: int,
    *,
    entry_char: str,
    previous_char: str | None,
    cycle_id: str,
    stage: str,
    artifact_char: str,
):
    return {
        "schema_version": "brian.phase66-durable-cycle-journal.v1",
        "sequence": sequence,
        "stage": stage,
        "cycle_id": cycle_id,
        "cycle_hash": _h("c"),
        "previous_entry_id": None if previous_char is None else _h(previous_char),
        "artifact_hash": _h(artifact_char),
        "artifact_ref": f"{stage.lower()}:{sequence}",
        "entry_id": _h(entry_char),
    }


def _runtime_checkpoint(
    checkpoint_char: str,
    *,
    cycle_id: str | None = None,
    journal_stages: tuple[str, ...] = (),
    new_risk_asset: str | None = "BTCUSDT",
    head_char: str = "a",
):
    cycles = {}
    entries = []
    if cycle_id is not None:
        cycles[cycle_id] = _cycle(cycle_id, new_risk_asset=new_risk_asset)
        entry_chars = ("e", "f", "1", "2", "3", "4", "5")
        artifact_chars = ("d", "6", "7", "8", "9", "0", "a")
        previous = None
        for sequence, stage in enumerate(journal_stages):
            char = entry_chars[sequence]
            entries.append(
                _entry(
                    sequence,
                    entry_char=char,
                    previous_char=previous,
                    cycle_id=cycle_id,
                    stage=stage,
                    artifact_char=artifact_chars[sequence],
                )
            )
            previous = char

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
            "journal_hash": _h("j" if not entries else "k"),
            "shadow_only": True,
            "live_execution": False,
        },
        "live_execution": False,
        "checkpoint_id": _h(checkpoint_char),
    }


def _risk_entry(
    sequence: int,
    *,
    entry_char: str,
    receipt_char: str,
    previous_entry_char: str | None,
    previous_state: str,
    trading_state: str,
    timestamp: float,
    blocked_assets: tuple[str, ...] = (),
):
    cooldown = [[asset, timestamp + 600.0] for asset in blocked_assets]
    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "sequence": sequence,
        "previous_entry_id": (
            None if previous_entry_char is None else _h(previous_entry_char)
        ),
        "policy_hash": _h("q"),
        "receipt": {
            "schema_version": "brian.phase68-operational-risk-governor.v1",
            "timestamp": timestamp,
            "previous_state": previous_state,
            "trading_state": trading_state,
            "recommended_state": trading_state,
            "reasons": [],
            "max_drawdown_fraction": 0.0,
            "window_loss_fraction": 0.0,
            "qualifying_stoplosses": 0,
            "stoploss_lock_until": None,
            "blocked_assets": list(blocked_assets),
            "consecutive_execution_failures": 0,
            "reconciliation_failures": 0,
            "unknown_order_outcomes": 0,
            "market_data_age_seconds": 0.0,
            "manual_halt": False,
            "manual_release_requested": False,
            "halt_latched": trading_state == "HALTED",
            "receipt_id": _h(receipt_char),
            "execution_failure_lock_until": None,
            "reconciliation_failure_lock_until": None,
            "asset_cooldown_until": cooldown,
            "shadow_only": True,
            "live_execution": False,
        },
        "entry_id": _h(entry_char),
    }


def _risk_manifest(
    ledger_char: str,
    *,
    state: str = "ACTIVE",
    blocked_assets: tuple[str, ...] = (),
    include_initial: bool = False,
):
    if include_initial:
        first = _risk_entry(
            0,
            entry_char="a",
            receipt_char="b",
            previous_entry_char=None,
            previous_state="ACTIVE",
            trading_state="ACTIVE",
            timestamp=TS,
        )
        second = _risk_entry(
            1,
            entry_char="f",
            receipt_char="0",
            previous_entry_char="a",
            previous_state="ACTIVE",
            trading_state=state,
            timestamp=TS + 1,
            blocked_assets=blocked_assets,
        )
        entries = [first, second]
    else:
        entries = [
            _risk_entry(
                0,
                entry_char="a",
                receipt_char="b",
                previous_entry_char=None,
                previous_state="ACTIVE",
                trading_state=state,
                timestamp=TS,
                blocked_assets=blocked_assets,
            )
        ]

    return {
        "schema_version": "brian.phase72-operational-risk-ledger.v1",
        "append_only": True,
        "policy": {
            "max_drawdown_fraction": 0.5,
            "max_daily_loss_fraction": 0.5,
        },
        "policy_hash": _h("q"),
        "initial_state": "ACTIVE",
        "entries": entries,
        "entry_count": len(entries),
        "head_entry_id": entries[-1]["entry_id"],
        "current_state": state,
        "halt_latched": state == "HALTED",
        "ledger_hash": _h(ledger_char),
        "shadow_only": True,
        "live_execution": False,
    }


def _acquire(conn, runtime_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_acquire_shadow_runtime_lease(%s,%s,%s)",
            (runtime_id, "owner-a", 60),
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
    checkpoint = _runtime_checkpoint(
        "4",
        cycle_id=cycle_id,
        journal_stages=("CYCLE_CREATED",),
    )
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


def _claim(conn, runtime_id, fence, cycle_id, worker, seconds=30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_claim_shadow_execution_dispatch(%s,%s,%s,%s,%s,%s)",
            (runtime_id, "owner-a", fence, cycle_id, worker, seconds),
        )
        return cur.fetchone()[0]


def _renew(conn, runtime_id, fence, cycle_id, worker, claim_fence, seconds=30):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_renew_shadow_execution_claim(%s,%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                fence,
                cycle_id,
                worker,
                claim_fence,
                seconds,
            ),
        )
        return cur.fetchone()[0]


def _complete(conn, runtime_id, fence, cycle_id, worker, claim_fence):
    with conn.cursor() as cur:
        cur.execute(
            "select public.brian_complete_shadow_execution_claim(%s,%s,%s,%s,%s,%s)",
            (
                runtime_id,
                "owner-a",
                fence,
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


def _bootstrap_submitted(conn, runtime_id, cycle_id):
    lease = _acquire(conn, runtime_id)
    fence = lease["fencing_token"]
    runtime = _commit_runtime(
        conn,
        runtime_id,
        fence,
        0,
        _runtime_checkpoint("1"),
    )
    risk_manifest = _risk_manifest("2")
    risk = _commit_risk(conn, runtime_id, fence, 0, risk_manifest)
    auth = _authorize(
        conn,
        runtime_id,
        fence,
        runtime["version"],
        risk["version"],
        risk_manifest,
        cycle_id,
    )
    assert auth["status"] == "AUTHORIZED_AND_PERSISTED"
    dispatch = _submit(conn, runtime_id, fence, cycle_id, _h("d"))
    assert dispatch["status"] == "SUBMITTED"
    return lease, auth, risk, risk_manifest, dispatch


def test_two_concurrent_workers_have_exactly_one_active_claim():
    runtime_id = _rid("claim-race")
    cycle_id = _h("1")
    setup = _connect()
    try:
        lease, *_ = _bootstrap_submitted(setup, runtime_id, cycle_id)
        fence = lease["fencing_token"]
    finally:
        setup.close()

    barrier = threading.Barrier(2)

    def contender(worker):
        conn = _connect()
        try:
            barrier.wait(timeout=5)
            return worker, _claim(conn, runtime_id, fence, cycle_id, worker, 30)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(contender, ("worker-a", "worker-b")))

    assert sum(row["status"] == "CLAIMED" for _, row in rows) == 1
    assert sum(row["status"] == "BLOCKED_ACTIVE" for _, row in rows) == 1
    winner = next((worker, row) for worker, row in rows if row["status"] == "CLAIMED")
    loser = next((worker, row) for worker, row in rows if row["status"] == "BLOCKED_ACTIVE")
    assert winner[1]["claimed"] is True
    assert loser[1]["claimed"] is False
    assert winner[1]["claim_fencing_token"] == 1


def test_expired_claim_takeover_increments_claim_fence_and_old_worker_cannot_renew():
    runtime_id = _rid("expiry")
    cycle_id = _h("2")
    conn = _connect()
    try:
        lease, *_ = _bootstrap_submitted(conn, runtime_id, cycle_id)
        fence = lease["fencing_token"]
        first = _claim(conn, runtime_id, fence, cycle_id, "worker-a", 1)
        assert first["status"] == "CLAIMED"
        assert first["claim_fencing_token"] == 1

        time.sleep(1.15)
        takeover = _claim(conn, runtime_id, fence, cycle_id, "worker-b", 30)
        assert takeover["status"] == "EXPIRED_RECOVERY"
        assert takeover["claimed"] is True
        assert takeover["claim_fencing_token"] == 2

        stale = _renew(
            conn,
            runtime_id,
            fence,
            cycle_id,
            "worker-a",
            first["claim_fencing_token"],
            30,
        )
        assert stale["renewed"] is False
        assert stale["status"] == "RENEWAL_LOST"

        renewed = _renew(
            conn,
            runtime_id,
            fence,
            cycle_id,
            "worker-b",
            takeover["claim_fencing_token"],
            30,
        )
        assert renewed["renewed"] is True
        assert renewed["status"] == "RENEWED"
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("state", "blocked_assets", "expected_reason"),
    [
        ("HALTED", (), "HALTED"),
        ("REDUCING", (), "REDUCING_NEW_RISK"),
        ("ACTIVE", ("BTCUSDT",), "ASSET_COOLDOWN"),
    ],
)
def test_current_risk_can_cancel_submitted_cycle_before_paper_execution(
    state,
    blocked_assets,
    expected_reason,
):
    runtime_id = _rid(f"cancel-{state.lower()}")
    cycle_id = _h("3")
    conn = _connect()
    try:
        lease, _, risk, _, _ = _bootstrap_submitted(conn, runtime_id, cycle_id)
        newer = _risk_manifest(
            "5",
            state=state,
            blocked_assets=blocked_assets,
            include_initial=True,
        )
        committed = _commit_risk(
            conn,
            runtime_id,
            lease["fencing_token"],
            risk["version"],
            newer,
        )
        assert committed["version"] == 2

        claim = _claim(
            conn,
            runtime_id,
            lease["fencing_token"],
            cycle_id,
            "worker-risk",
            30,
        )
        assert claim["claimed"] is False
        assert claim["cancelled"] is True
        assert claim["terminal"] is True
        assert claim["status"] == "CANCELLED_BEFORE_EXECUTION"
        assert claim["cancel_reason"] == expected_reason
        assert claim["journal_stage"] == "CYCLE_CREATED"
        assert _count(conn, "brian_shadow_execution_claims", runtime_id) == 1
    finally:
        conn.close()


def test_risk_halt_after_paper_started_does_not_erase_side_effect_on_expired_takeover():
    runtime_id = _rid("resume-only")
    cycle_id = _h("4")
    conn = _connect()
    try:
        lease, auth, risk, _, _ = _bootstrap_submitted(conn, runtime_id, cycle_id)
        fence = lease["fencing_token"]
        first = _claim(conn, runtime_id, fence, cycle_id, "worker-a", 1)
        assert first["status"] == "CLAIMED"

        paper_checkpoint = _runtime_checkpoint(
            "6",
            cycle_id=cycle_id,
            journal_stages=("CYCLE_CREATED", "PAPER_APPLIED"),
        )
        advanced_runtime = _commit_runtime(
            conn,
            runtime_id,
            fence,
            auth["runtime_version_after"],
            paper_checkpoint,
        )
        assert advanced_runtime["version"] == auth["runtime_version_after"] + 1

        halted = _risk_manifest("7", state="HALTED", include_initial=True)
        advanced_risk = _commit_risk(
            conn,
            runtime_id,
            fence,
            risk["version"],
            halted,
        )
        assert advanced_risk["version"] == 2

        time.sleep(1.15)
        recovered = _claim(conn, runtime_id, fence, cycle_id, "worker-b", 30)
        assert recovered["claimed"] is True
        assert recovered["cancelled"] is False
        assert recovered["status"] == "EXPIRED_RECOVERY"
        assert recovered["resume_only"] is True
        assert recovered["journal_stage"] == "PAPER_APPLIED"
        assert recovered["claim_fencing_token"] == first["claim_fencing_token"] + 1
    finally:
        conn.close()


def test_completion_requires_committed_runtime_and_is_idempotent():
    runtime_id = _rid("complete")
    cycle_id = _h("5")
    conn = _connect()
    try:
        lease, auth, _, _, _ = _bootstrap_submitted(conn, runtime_id, cycle_id)
        fence = lease["fencing_token"]
        claim = _claim(conn, runtime_id, fence, cycle_id, "worker-a", 30)
        claim_fence = claim["claim_fencing_token"]

        too_early = _complete(
            conn,
            runtime_id,
            fence,
            cycle_id,
            "worker-a",
            claim_fence,
        )
        assert too_early["completed"] is False
        assert too_early["status"] == "RUNTIME_NOT_COMMITTED"

        committed_checkpoint = _runtime_checkpoint(
            "8",
            cycle_id=cycle_id,
            journal_stages=(
                "CYCLE_CREATED",
                "PAPER_APPLIED",
                "LOCAL_PROJECTED",
                "RECONCILED",
                "COMMITTED",
            ),
            head_char="8",
        )
        committed = _commit_runtime(
            conn,
            runtime_id,
            fence,
            auth["runtime_version_after"],
            committed_checkpoint,
        )
        assert committed["version"] == auth["runtime_version_after"] + 1

        completed = _complete(
            conn,
            runtime_id,
            fence,
            cycle_id,
            "worker-a",
            claim_fence,
        )
        duplicate = _complete(
            conn,
            runtime_id,
            fence,
            cycle_id,
            "worker-a",
            claim_fence,
        )
        assert completed["completed"] is True
        assert completed["duplicate"] is False
        assert completed["status"] == "COMPLETED"
        assert completed["completion_checkpoint_id"] == committed_checkpoint["checkpoint_id"]

        assert duplicate["completed"] is True
        assert duplicate["duplicate"] is True
        assert duplicate["status"] == "COMPLETED"
        assert duplicate["completion_checkpoint_id"] == committed_checkpoint["checkpoint_id"]
    finally:
        conn.close()


def test_claim_recovery_from_already_committed_runtime_never_reexecutes():
    runtime_id = _rid("recover-completed")
    cycle_id = _h("6")
    conn = _connect()
    try:
        lease, auth, _, _, _ = _bootstrap_submitted(conn, runtime_id, cycle_id)
        fence = lease["fencing_token"]

        committed_checkpoint = _runtime_checkpoint(
            "9",
            cycle_id=cycle_id,
            journal_stages=(
                "CYCLE_CREATED",
                "PAPER_APPLIED",
                "LOCAL_PROJECTED",
                "RECONCILED",
                "COMMITTED",
            ),
            head_char="9",
        )
        _commit_runtime(
            conn,
            runtime_id,
            fence,
            auth["runtime_version_after"],
            committed_checkpoint,
        )

        recovered = _claim(conn, runtime_id, fence, cycle_id, "worker-recover", 30)
        assert recovered["claimed"] is False
        assert recovered["terminal"] is True
        assert recovered["status"] == "COMPLETED"
        assert recovered["journal_stage"] == "COMMITTED"
        assert recovered["completion_checkpoint_id"] == committed_checkpoint["checkpoint_id"]
    finally:
        conn.close()


def test_service_role_cannot_mutate_claim_or_claim_event_tables_directly():
    runtime_id = _rid("priv")
    cycle_id = _h("7")
    conn = _connect()
    try:
        lease, *_ = _bootstrap_submitted(conn, runtime_id, cycle_id)
        claim = _claim(
            conn,
            runtime_id,
            lease["fencing_token"],
            cycle_id,
            "worker-a",
            30,
        )
        assert claim["claimed"] is True

        with conn.cursor() as cur:
            cur.execute("set role service_role")
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_shadow_execution_claims "
                    "set worker_token='attacker' where runtime_id=%s",
                    (runtime_id,),
                )
            conn.rollback()
            conn.autocommit = True
    finally:
        conn.close()
