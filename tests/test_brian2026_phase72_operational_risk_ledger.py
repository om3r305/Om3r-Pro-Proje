from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from brian2026.phase68_operational_risk_governor import (
    ClosedTrade,
    EquityPoint,
    OperationalRiskGovernor,
    OperationalRiskPolicy,
    RuntimeHealthEvent,
)
from brian2026.phase72_operational_risk_ledger import (
    OperationalRiskLedger,
    OperationalRiskLedgerError,
    restore_operational_risk_ledger,
)


TS = 1_760_000_000.0


def _policy(**overrides) -> OperationalRiskPolicy:
    values = dict(
        max_drawdown_fraction=0.50,
        max_daily_loss_fraction=0.50,
        stoploss_limit=3,
        stoploss_lock_seconds=600,
        asset_cooldown_seconds=300,
        execution_failure_lookback_seconds=900,
        max_consecutive_execution_failures=3,
        reconciliation_failure_lookback_seconds=900,
        max_reconciliation_failures=2,
        max_unknown_order_outcomes=1,
        max_market_data_age_seconds=30.0,
    )
    values.update(overrides)
    return OperationalRiskPolicy(**values)


def _equity(now: float):
    return (
        EquityPoint(now - 60, 1000.0),
        EquityPoint(now, 1000.0),
    )


def _evaluate(
    governor: OperationalRiskGovernor,
    *,
    now: float,
    trades=(),
    health=(),
    market_data_timestamp: float | None = None,
    manual_release: bool = False,
):
    return governor.evaluate(
        now=now,
        equity_points=_equity(now),
        closed_trades=trades,
        health_events=health,
        market_data_timestamp=(
            now if market_data_timestamp is None else market_data_timestamp
        ),
        manual_release=manual_release,
    )


def test_receipt_identity_now_covers_persisted_temporary_locks() -> None:
    governor = OperationalRiskGovernor(_policy())
    receipt = _evaluate(
        governor,
        now=TS,
        trades=(ClosedTrade("BTCUSDT", TS - 10, 1.0, 0.01, "TARGET"),),
    )
    assert receipt.verify_identity() is True
    assert receipt.asset_cooldown_until
    assert receipt.blocked_assets == ("BTCUSDT",)

    forged = replace(receipt, receipt_id="0" * 64)
    assert forged.verify_identity() is False


def test_halted_latch_survives_manifest_roundtrip_and_requires_manual_release() -> None:
    policy = _policy(max_market_data_age_seconds=10.0)
    ledger = OperationalRiskLedger(policy)
    governor = ledger.governor()

    halted = _evaluate(
        governor,
        now=TS,
        market_data_timestamp=TS - 20,
    )
    ledger.append(halted)
    assert halted.trading_state == "HALTED"
    assert halted.halt_latched is True

    restored = restore_operational_risk_ledger(ledger.manifest())
    recovered_governor = restored.governor()
    still_halted = _evaluate(
        recovered_governor,
        now=TS + 1,
        market_data_timestamp=TS + 1,
    )
    assert still_halted.recommended_state == "ACTIVE"
    assert still_halted.trading_state == "HALTED"
    assert still_halted.halt_latched is True
    assert "halt_latched_manual_release_required" in still_halted.reasons
    restored.append(still_halted)

    released = _evaluate(
        recovered_governor,
        now=TS + 2,
        market_data_timestamp=TS + 2,
        manual_release=True,
    )
    restored.append(released)
    assert released.trading_state == "ACTIVE"
    assert released.halt_latched is False


def test_asset_cooldown_survives_restart_without_replaying_closed_trade_history() -> None:
    policy = _policy(asset_cooldown_seconds=300)
    ledger = OperationalRiskLedger(policy)
    governor = ledger.governor()

    receipt = _evaluate(
        governor,
        now=TS,
        trades=(ClosedTrade("BTCUSDT", TS - 30, 1.0, 0.01, "TARGET"),),
    )
    ledger.append(receipt)
    assert receipt.blocked_assets == ("BTCUSDT",)
    cooldown_until = dict(receipt.asset_cooldown_until)["BTCUSDT"]
    assert cooldown_until == pytest.approx(TS + 270)

    restored = restore_operational_risk_ledger(ledger.manifest())
    recovered = restored.governor()
    after_restart = _evaluate(recovered, now=TS + 1, trades=())
    assert after_restart.blocked_assets == ("BTCUSDT",)
    assert dict(after_restart.asset_cooldown_until)["BTCUSDT"] == pytest.approx(
        cooldown_until
    )

    expired = _evaluate(recovered, now=TS + 271, trades=())
    assert expired.blocked_assets == ()
    assert expired.asset_cooldown_until == ()


def test_execution_failure_reducing_lock_survives_restart_without_health_history() -> None:
    policy = _policy(max_consecutive_execution_failures=3)
    ledger = OperationalRiskLedger(policy)
    governor = ledger.governor()
    failures = (
        RuntimeHealthEvent(TS - 3, "EXECUTION_FAILURE"),
        RuntimeHealthEvent(TS - 2, "EXECUTION_FAILURE"),
        RuntimeHealthEvent(TS - 1, "EXECUTION_FAILURE"),
    )

    receipt = _evaluate(governor, now=TS, health=failures)
    ledger.append(receipt)
    assert receipt.trading_state == "REDUCING"
    assert receipt.execution_failure_lock_until == pytest.approx(TS + 899)

    restored = restore_operational_risk_ledger(ledger.manifest())
    recovered = restored.governor()
    after_restart = _evaluate(recovered, now=TS + 1, health=())
    assert after_restart.trading_state == "REDUCING"
    assert after_restart.execution_failure_lock_until == pytest.approx(TS + 899)


def test_execution_success_can_clear_persisted_failure_lock_early() -> None:
    policy = _policy(max_consecutive_execution_failures=3)
    ledger = OperationalRiskLedger(policy)
    governor = ledger.governor()
    receipt = _evaluate(
        governor,
        now=TS,
        health=(
            RuntimeHealthEvent(TS - 3, "EXECUTION_FAILURE"),
            RuntimeHealthEvent(TS - 2, "EXECUTION_FAILURE"),
            RuntimeHealthEvent(TS - 1, "EXECUTION_FAILURE"),
        ),
    )
    ledger.append(receipt)

    recovered = restore_operational_risk_ledger(ledger.manifest()).governor()
    cleared = _evaluate(
        recovered,
        now=TS + 1,
        health=(RuntimeHealthEvent(TS + 1, "EXECUTION_SUCCESS"),),
    )
    assert cleared.trading_state == "ACTIVE"
    assert cleared.execution_failure_lock_until is None
    assert cleared.consecutive_execution_failures == 0


def test_reconciliation_failure_lock_survives_restart_and_success_clears_it() -> None:
    policy = _policy(max_reconciliation_failures=2)
    ledger = OperationalRiskLedger(policy)
    governor = ledger.governor()
    receipt = _evaluate(
        governor,
        now=TS,
        health=(
            RuntimeHealthEvent(TS - 2, "RECONCILIATION_FAILURE"),
            RuntimeHealthEvent(TS - 1, "RECONCILIATION_FAILURE"),
        ),
    )
    ledger.append(receipt)
    assert receipt.trading_state == "REDUCING"
    assert receipt.reconciliation_failure_lock_until == pytest.approx(TS + 899)

    recovered = restore_operational_risk_ledger(ledger.manifest()).governor()
    still_locked = _evaluate(recovered, now=TS + 1, health=())
    assert still_locked.trading_state == "REDUCING"

    cleared = _evaluate(
        recovered,
        now=TS + 2,
        health=(RuntimeHealthEvent(TS + 2, "RECONCILIATION_SUCCESS"),),
    )
    assert cleared.trading_state == "ACTIVE"
    assert cleared.reconciliation_failure_lock_until is None
    assert cleared.reconciliation_failures == 0


def test_stoploss_lock_survives_restart_without_trade_history_until_expiry() -> None:
    policy = _policy(stoploss_limit=3, stoploss_lock_seconds=600)
    ledger = OperationalRiskLedger(policy)
    governor = ledger.governor()
    trades = tuple(
        ClosedTrade(
            f"ASSET{index}",
            TS - 10 - index,
            -10.0,
            -0.01,
            "STOP_LOSS",
        )
        for index in range(3)
    )
    receipt = _evaluate(governor, now=TS, trades=trades)
    ledger.append(receipt)
    assert receipt.trading_state == "REDUCING"
    assert receipt.stoploss_lock_until == pytest.approx(TS + 590)

    recovered = restore_operational_risk_ledger(ledger.manifest()).governor()
    locked = _evaluate(recovered, now=TS + 1, trades=())
    assert locked.trading_state == "REDUCING"
    assert locked.stoploss_lock_until == pytest.approx(TS + 590)

    expired = _evaluate(recovered, now=TS + 591, trades=())
    assert expired.trading_state == "ACTIVE"
    assert expired.stoploss_lock_until is None


def test_ledger_rejects_state_branch_and_non_monotonic_distinct_receipt() -> None:
    policy = _policy()
    ledger = OperationalRiskLedger(policy)
    governor = ledger.governor()
    first = _evaluate(governor, now=TS)
    ledger.append(first)

    foreign_governor = OperationalRiskGovernor(policy, initial_state="REDUCING")
    wrong_previous = _evaluate(foreign_governor, now=TS + 1)
    with pytest.raises(OperationalRiskLedgerError, match="previous_state"):
        ledger.append(wrong_previous)

    same_time_governor = ledger.governor()
    distinct_same_time = _evaluate(
        same_time_governor,
        now=TS,
        trades=(ClosedTrade("BTCUSDT", TS - 1, 1.0, 0.01, "TARGET"),),
    )
    with pytest.raises(OperationalRiskLedgerError, match="advance timestamp"):
        ledger.append(distinct_same_time)


def test_duplicate_receipt_is_idempotent() -> None:
    ledger = OperationalRiskLedger(_policy())
    receipt = _evaluate(ledger.governor(), now=TS)
    first = ledger.append(receipt)
    duplicate = ledger.append(receipt)

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.entry_id == first.entry_id
    assert len(ledger.entries) == 1


def test_manifest_restore_accepts_jsonb_integral_policy_numerics() -> None:
    ledger = OperationalRiskLedger(
        _policy(
            stoploss_required_profit=0.0,
            max_market_data_age_seconds=30.0,
        )
    )
    ledger.append(_evaluate(ledger.governor(), now=TS))
    manifest = copy.deepcopy(ledger.manifest())

    manifest["policy"]["stoploss_required_profit"] = 0
    manifest["policy"]["max_market_data_age_seconds"] = 30

    restored = restore_operational_risk_ledger(manifest)

    assert restored.policy_hash == ledger.policy_hash
    assert restored.manifest()["ledger_hash"] == ledger.manifest()["ledger_hash"]
    assert restored.policy.stoploss_required_profit == 0.0
    assert restored.policy.max_market_data_age_seconds == 30.0


def test_manifest_tamper_policy_and_chain_are_detected() -> None:
    ledger = OperationalRiskLedger(_policy())
    ledger.append(_evaluate(ledger.governor(), now=TS))
    manifest = ledger.manifest()

    bad_policy = copy.deepcopy(manifest)
    bad_policy["policy"]["max_drawdown_fraction"] = 0.01
    with pytest.raises(OperationalRiskLedgerError, match="policy hash mismatch"):
        restore_operational_risk_ledger(bad_policy)

    bad_chain = copy.deepcopy(manifest)
    bad_chain["entries"][0]["entry_id"] = "0" * 64
    with pytest.raises(OperationalRiskLedgerError, match="entry content hash"):
        restore_operational_risk_ledger(bad_chain)


def test_tampered_receipt_hash_is_rejected_before_append() -> None:
    ledger = OperationalRiskLedger(_policy())
    receipt = _evaluate(ledger.governor(), now=TS)
    forged = replace(receipt, receipt_id="0" * 64)
    with pytest.raises(OperationalRiskLedgerError, match="receipt content hash"):
        ledger.append(forged)
