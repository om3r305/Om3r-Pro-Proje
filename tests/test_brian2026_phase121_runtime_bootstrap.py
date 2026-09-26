from __future__ import annotations

from dataclasses import replace

import pytest

from brian2026.phase68_operational_risk_governor import OperationalRiskPolicy
from brian2026.phase70_durable_runtime_store import (
    RuntimeCommitReceipt,
    RuntimeLease,
    StoredRuntimeCheckpoint,
)
from brian2026.phase73_operational_risk_store import (
    OperationalRiskCommitReceipt,
    StoredOperationalRiskLedger,
)
from brian2026.phase121_runtime_bootstrap import (
    RuntimeBootstrapConflictError,
    RuntimeBootstrapSpec,
    RuntimeBootstrapper,
    build_runtime_bootstrap_artifacts,
)


TS = 1_790_000_000.0


def _spec(
    *,
    observed_at=TS,
    account_id="BRIAN-PAPER-RUNTIME",
    policy=None,
):
    return RuntimeBootstrapSpec(
        runtime_id="brian-shadow-main",
        account_id=account_id,
        covered_assets=(
            "crypto:ETHUSDT",
            "crypto:BTCUSDT",
            "crypto:BTCUSDT",
        ),
        starting_cash_usd=1000.0,
        paper_fee_bps=10.0,
        allow_short=True,
        observed_at=observed_at,
        source_ref="phase121:genesis",
        risk_policy=policy or OperationalRiskPolicy(
            max_drawdown_fraction=0.10,
            max_daily_loss_fraction=0.07,
            max_market_data_age_seconds=30.0,
        ),
    )


class MemoryRuntimeStore:
    def __init__(self):
        self.version = 0
        self.checkpoint = None
        self.owner = None
        self.fence = 0

    def acquire(self, *, runtime_id, owner_token, lease_seconds):
        del lease_seconds
        if self.owner is not None and self.owner != owner_token:
            return RuntimeLease(
                runtime_id=runtime_id,
                owner_token=owner_token,
                fencing_token=max(self.fence, 1),
                version=self.version,
                status="BLOCKED_ACTIVE",
                acquired=False,
                lease_until=None,
            )
        if self.owner is None:
            self.fence += 1
            self.owner = owner_token
        return RuntimeLease(
            runtime_id=runtime_id,
            owner_token=owner_token,
            fencing_token=self.fence,
            version=self.version,
            status="ACQUIRED",
            acquired=True,
            lease_until=None,
        )

    def load(self, *, runtime_id):
        if self.checkpoint is None:
            return None
        return StoredRuntimeCheckpoint(
            runtime_id=runtime_id,
            version=self.version,
            checkpoint=self.checkpoint,
            journal_hash=str(self.checkpoint.journal_manifest["journal_hash"]),
            head_state_id=str(
                self.checkpoint.runtime_checkpoint.shadow_ledger_manifest[
                    "head_state_id"
                ]
            ),
            pending_cycle_id=self.checkpoint.runtime_checkpoint.pending_cycle_id,
            fencing_token=max(self.fence, 1),
            lease_until=None,
        )

    def commit(self, lease, *, expected_version, checkpoint):
        if lease.owner_token != self.owner or lease.fencing_token != self.fence:
            return RuntimeCommitReceipt(
                runtime_id=lease.runtime_id,
                checkpoint_id=checkpoint.checkpoint_id,
                fencing_token=max(self.fence, 1),
                version=self.version,
                current_version=self.version,
                status="LEASE_LOST",
                committed=False,
                duplicate=False,
            )
        if self.checkpoint is not None:
            if self.checkpoint.checkpoint_id == checkpoint.checkpoint_id:
                return RuntimeCommitReceipt(
                    runtime_id=lease.runtime_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    fencing_token=self.fence,
                    version=self.version,
                    current_version=self.version,
                    status="DUPLICATE_CURRENT",
                    committed=True,
                    duplicate=True,
                )
        if expected_version != self.version:
            return RuntimeCommitReceipt(
                runtime_id=lease.runtime_id,
                checkpoint_id=checkpoint.checkpoint_id,
                fencing_token=self.fence,
                version=self.version,
                current_version=self.version,
                status="CAS_CONFLICT",
                committed=False,
                duplicate=False,
            )
        self.version += 1
        self.checkpoint = checkpoint
        return RuntimeCommitReceipt(
            runtime_id=lease.runtime_id,
            checkpoint_id=checkpoint.checkpoint_id,
            fencing_token=self.fence,
            version=self.version,
            current_version=self.version,
            status="COMMITTED",
            committed=True,
            duplicate=False,
        )

    def renew(self, lease, *, lease_seconds):
        del lease_seconds
        return replace(lease, acquired=True, version=self.version)

    def release(self, lease):
        if (
            lease.owner_token != self.owner
            or lease.fencing_token != self.fence
        ):
            return False
        self.owner = None
        return True


class MemoryRiskStore:
    def __init__(self):
        self.version = 0
        self.ledger = None
        self.runtime_store = None

    def load(self, *, runtime_id):
        if self.ledger is None:
            return None
        manifest = self.ledger.manifest()
        return StoredOperationalRiskLedger(
            runtime_id=runtime_id,
            version=self.version,
            ledger=self.ledger,
            ledger_hash=str(manifest["ledger_hash"]),
            policy_hash=str(manifest["policy_hash"]),
            head_entry_id=manifest["head_entry_id"],
            current_state=str(manifest["current_state"]),
            halt_latched=bool(manifest["halt_latched"]),
        )

    def commit(self, lease, *, expected_version, ledger):
        if self.runtime_store is not None and (
            lease.owner_token != self.runtime_store.owner
            or lease.fencing_token != self.runtime_store.fence
        ):
            status = "LEASE_LOST"
            committed = False
        elif expected_version != self.version:
            status = "CAS_CONFLICT"
            committed = False
        else:
            self.version += 1
            self.ledger = ledger
            status = "COMMITTED"
            committed = True
        manifest = ledger.manifest()
        return OperationalRiskCommitReceipt(
            runtime_id=lease.runtime_id,
            ledger_hash=str(manifest["ledger_hash"]),
            fencing_token=lease.fencing_token,
            version=self.version,
            current_version=self.version,
            status=status,
            committed=committed,
            duplicate=False,
        )


def _bootstrapper(runtime=None, risk=None):
    bootstrapper = RuntimeBootstrapper(
        rpc=lambda name, params: pytest.fail(
            f"unexpected direct rpc {name} {params}"
        ),
        lease_seconds=60,
    )
    bootstrapper.runtime_store = runtime or MemoryRuntimeStore()
    bootstrapper.risk_store = risk or MemoryRiskStore()
    if isinstance(bootstrapper.risk_store, MemoryRiskStore):
        bootstrapper.risk_store.runtime_store = bootstrapper.runtime_store
    return bootstrapper


def test_bootstrap_artifacts_are_canonical_active_shadow_genesis() -> None:
    artifacts = build_runtime_bootstrap_artifacts(_spec())

    checkpoint = artifacts.runtime.checkpoint()
    head = artifacts.runtime.ledger.head_state
    manifest = artifacts.risk_ledger.manifest()

    assert head.account_id == "BRIAN-PAPER-RUNTIME"
    assert head.covered_assets == (
        "crypto:BTCUSDT",
        "crypto:ETHUSDT",
    )
    assert head.position_weights == ()
    assert head.equity_usd == pytest.approx(1000.0)
    assert head.available_cash_usd == pytest.approx(1000.0)
    assert checkpoint.checkpoint_id == artifacts.genesis_checkpoint_id
    assert manifest["current_state"] == "ACTIVE"
    assert manifest["halt_latched"] is False
    assert manifest["entry_count"] == 1
    assert manifest["ledger_hash"] == artifacts.risk_ledger_hash


def test_fresh_bootstrap_commits_runtime_then_risk_and_verifies_both() -> None:
    bootstrapper = _bootstrapper()

    receipt = bootstrapper.bootstrap(
        spec=_spec(),
        owner_token="bootstrap-owner",
    )

    assert receipt.status == "BOOTSTRAPPED"
    assert receipt.runtime_version == 1
    assert receipt.risk_version == 1
    assert receipt.completed_partial_bootstrap is False
    assert receipt.already_bootstrapped is False
    assert bootstrapper.runtime_store.owner is None
    assert len(receipt.bootstrap_id) == 64


def test_repeat_bootstrap_is_idempotent_after_both_heads_exist() -> None:
    bootstrapper = _bootstrapper()
    first = bootstrapper.bootstrap(
        spec=_spec(),
        owner_token="owner-a",
    )
    second = bootstrapper.bootstrap(
        spec=_spec(),
        owner_token="owner-b",
    )

    assert first.status == "BOOTSTRAPPED"
    assert second.status == "ALREADY_BOOTSTRAPPED"
    assert second.runtime_version == first.runtime_version
    assert second.risk_version == first.risk_version
    assert second.already_bootstrapped is True


def test_runtime_only_exact_genesis_is_safe_partial_bootstrap_recovery() -> None:
    runtime = MemoryRuntimeStore()
    risk = MemoryRiskStore()
    artifacts = build_runtime_bootstrap_artifacts(_spec())
    runtime.version = 1
    runtime.checkpoint = artifacts.runtime.checkpoint()
    bootstrapper = _bootstrapper(runtime, risk)

    receipt = bootstrapper.bootstrap(
        spec=_spec(),
        owner_token="partial-owner",
    )

    assert receipt.status == "COMPLETED_PARTIAL_BOOTSTRAP"
    assert receipt.runtime_version == 1
    assert receipt.risk_version == 1
    assert receipt.completed_partial_bootstrap is True


def test_runtime_only_non_genesis_state_fails_closed() -> None:
    runtime = MemoryRuntimeStore()
    risk = MemoryRiskStore()
    different = build_runtime_bootstrap_artifacts(
        _spec(observed_at=TS - 60)
    )
    runtime.version = 1
    runtime.checkpoint = different.runtime.checkpoint()
    bootstrapper = _bootstrapper(runtime, risk)

    with pytest.raises(
        RuntimeBootstrapConflictError,
        match="advanced beyond expected genesis",
    ):
        bootstrapper.bootstrap(
            spec=_spec(),
            owner_token="owner",
        )

    assert risk.ledger is None


def test_risk_without_runtime_is_inconsistent_and_never_auto_repaired() -> None:
    runtime = MemoryRuntimeStore()
    risk = MemoryRiskStore()
    risk.version = 1
    risk.ledger = build_runtime_bootstrap_artifacts(_spec()).risk_ledger
    bootstrapper = _bootstrapper(runtime, risk)

    with pytest.raises(
        RuntimeBootstrapConflictError,
        match="risk head exists without durable runtime head",
    ):
        bootstrapper.bootstrap(
            spec=_spec(),
            owner_token="owner",
        )


def test_existing_runtime_and_different_risk_policy_fails_closed() -> None:
    bootstrapper = _bootstrapper()
    bootstrapper.bootstrap(
        spec=_spec(),
        owner_token="owner-a",
    )
    different = OperationalRiskPolicy(
        max_drawdown_fraction=0.20,
        max_daily_loss_fraction=0.07,
        max_market_data_age_seconds=30.0,
    )

    with pytest.raises(
        RuntimeBootstrapConflictError,
        match="policy differs",
    ):
        bootstrapper.bootstrap(
            spec=_spec(policy=different),
            owner_token="owner-b",
        )


def test_invalid_assets_or_owner_are_rejected_before_mutation() -> None:
    with pytest.raises(ValueError, match=r"crypto:\*USDT"):
        RuntimeBootstrapSpec(
            runtime_id="runtime",
            account_id="account",
            covered_assets=("BTCUSDT",),
            starting_cash_usd=1000.0,
            paper_fee_bps=10.0,
            allow_short=True,
            observed_at=TS,
            source_ref="source",
            risk_policy=OperationalRiskPolicy(),
        )

    bootstrapper = _bootstrapper()
    with pytest.raises(ValueError, match="owner_token"):
        bootstrapper.bootstrap(
            spec=_spec(),
            owner_token="",
        )
