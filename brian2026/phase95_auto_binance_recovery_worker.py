from __future__ import annotations

from dataclasses import dataclass
import math
import time
from collections.abc import Callable, Mapping

from .phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from .phase81_cancel_recovery_directive import CancelRecoveryDirectiveReceipt
from .phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from .phase92_recovery_worker_session import RecoveryWorkerSession
from .phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryEvidenceBundle,
    BinanceSpotRecoveryEvidenceProvider,
)

PHASE95_SCHEMA_VERSION = "brian.phase95-auto-binance-recovery-worker.v1"

_EVIDENCE_CAPABLE_STATES = frozenset({
    "NEEDS_DIRECTIVE",
    "NEEDS_CLAIM",
    "CLAIM_EXPIRED",
    "NEEDS_START",
    "STARTED_NEEDS_EXECUTION",
    "RECOVERY_PROGRESS",
})


class AutoRecoveryEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AutoRecoveryStartupReceipt:
    gate: RecoveryStartupGateReceipt
    original_cycle_id: str | None
    cancel_risk_receipt_id: str | None
    preflight_work_state: str
    directive_status: str | None
    evidence_assets: tuple[str, ...]
    evidence_observed_at: tuple[tuple[str, float], ...]
    decision_at: float
    schema_version: str = PHASE95_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.decision_at):
            raise ValueError("decision_at must be finite")
        if self.evidence_assets != tuple(sorted(set(self.evidence_assets))):
            raise ValueError("evidence assets must be unique and sorted")
        observed_assets = tuple(asset for asset, _ in self.evidence_observed_at)
        if observed_assets != self.evidence_assets:
            raise ValueError("evidence timestamps must match evidence assets")
        for _, observed_at in self.evidence_observed_at:
            if not math.isfinite(observed_at) or observed_at < self.decision_at:
                raise ValueError(
                    "market evidence cannot precede the Phase95 recovery decision"
                )
        if self.original_cycle_id is not None and len(self.original_cycle_id) != 64:
            raise ValueError("original_cycle_id must be a content hash")
        if (
            self.cancel_risk_receipt_id is not None
            and len(self.cancel_risk_receipt_id) != 64
        ):
            raise ValueError("cancel_risk_receipt_id must be a content hash")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase95 auto-evidence worker must remain shadow-only")


ProviderFactory = Callable[[], BinanceSpotRecoveryEvidenceProvider]


def _same_backlog_identity(left, right) -> bool:
    if left.has_work != right.has_work:
        return False
    if not left.has_work:
        return True
    return (
        left.original_cycle_id == right.original_cycle_id
        and left.cancel_risk_receipt_id == right.cancel_risk_receipt_id
        and left.dispatch_id == right.dispatch_id
    )


def _directive_for_work(
    session: RecoveryWorkerSession,
    work,
) -> CancelRecoveryDirectiveReceipt | None:
    if (
        not work.has_work
        or work.work_state not in _EVIDENCE_CAPABLE_STATES
        or work.original_cycle_id is None
    ):
        return None
    return session.stack.directives.prepare(
        session.runtime_supervisor.lease,
        cycle_id=work.original_cycle_id,
        expected_runtime_version=session.runtime_supervisor.persisted_version,
    )


def _requires_market_evidence(
    directive: CancelRecoveryDirectiveReceipt | None,
) -> bool:
    return bool(
        directive is not None
        and directive.prepared
        and directive.recovery_status in {
            "READY_REDUCE_ONLY",
            "WAIT_RISK_RELEASE",
        }
        and directive.recovery_legs
    )


def run_one_auto_binance_recovery(
    session: RecoveryWorkerSession,
    *,
    recovery_worker_token: str,
    recovery_claim_seconds: int,
    recovery_ttl_seconds: int,
    source_ref: str,
    provider_factory: ProviderFactory = BinanceSpotRecoveryEvidenceProvider,
    clock: Callable[[], float] = time.time,
) -> AutoRecoveryStartupReceipt:
    """Process at most one Phase87 recovery item with causal Binance evidence.

    The decision timestamp is frozen before any external market-data request.
    Phase94 snapshots must arrive at/after that time, so the existing Phase46
    simulator never receives a hindsight/pre-decision order-book observation.

    Exactly one Phase87 item is passed to Phase89 (max_items=1). If more
    recovery work remains, a later invocation must fetch fresh evidence for that
    item's own immutable Phase81 legs.
    """
    if session.closed:
        raise AutoRecoveryEvidenceError("recovery worker session is closed")
    if not recovery_worker_token.strip():
        raise ValueError("recovery_worker_token is required")
    if recovery_claim_seconds <= 0 or recovery_ttl_seconds <= 0:
        raise ValueError("recovery claim/intent TTL must be positive")
    if not source_ref.strip():
        raise ValueError("source_ref is required")

    supervisor = session.runtime_supervisor
    work = session.stack.backlog.read_next(runtime_id=session.runtime_id)
    decision_at = float(clock())
    if not math.isfinite(decision_at):
        raise AutoRecoveryEvidenceError("Phase95 clock returned non-finite timestamp")

    directive = _directive_for_work(session, work)
    bundle = BinanceSpotRecoveryEvidenceBundle(())

    if _requires_market_evidence(directive):
        assert directive is not None
        assets = tuple(sorted(leg.asset_id for leg in directive.recovery_legs))
        if len(assets) != len(set(assets)):
            raise AutoRecoveryEvidenceError(
                "Phase81 directive contains duplicate recovery assets"
            )

        provider = provider_factory()
        if provider is None:
            raise AutoRecoveryEvidenceError("provider_factory returned no provider")
        try:
            with provider:
                bundle = provider.collect(assets)
        except AttributeError as exc:
            raise AutoRecoveryEvidenceError(
                "Phase94 provider must support context-manager cleanup"
            ) from exc

        if tuple(row.asset_id for row in bundle.assets) != assets:
            raise AutoRecoveryEvidenceError(
                "Phase94 evidence asset set differs from Phase81 recovery legs"
            )
        for row in bundle.assets:
            if row.observed_at < decision_at:
                raise AutoRecoveryEvidenceError(
                    f"{row.asset_id} market evidence predates recovery decision"
                )

        # External I/O happened while this runtime lease was held. Re-read the
        # DB-authoritative backlog before using the evidence. State may advance
        # from NEEDS_DIRECTIVE to NEEDS_CLAIM because Phase81 preflight itself
        # is durable, but the exact cancel/dispatch identity must not change.
        current = session.stack.backlog.read_next(runtime_id=session.runtime_id)
        if not _same_backlog_identity(work, current):
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "recovery backlog identity changed while Binance evidence was fetched"
            )
        if (
            current.has_work
            and (
                current.runtime_version != supervisor.persisted_version
                or current.runtime_checkpoint_id
                    != supervisor.runtime.checkpoint().checkpoint_id
                or current.runtime_head_state_id
                    != supervisor.runtime.ledger.head_state.state_id
            )
        ):
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "recovery runtime/head changed while Binance evidence was fetched"
            )

    gate = session.run_startup_gate(
        max_items=1,
        recovery_worker_token=recovery_worker_token,
        recovery_claim_seconds=recovery_claim_seconds,
        recovery_markets=bundle.markets,
        recovery_risk_limits_by_asset=bundle.risk_limits_by_asset,
        recovery_ttl_seconds=recovery_ttl_seconds,
        marks=bundle.marks,
        observed_at=decision_at,
        source_ref=source_ref,
    )

    return AutoRecoveryStartupReceipt(
        gate=gate,
        original_cycle_id=work.original_cycle_id,
        cancel_risk_receipt_id=work.cancel_risk_receipt_id,
        preflight_work_state=work.work_state,
        directive_status=None if directive is None else directive.status,
        evidence_assets=tuple(row.asset_id for row in bundle.assets),
        evidence_observed_at=tuple(
            (row.asset_id, row.observed_at)
            for row in bundle.assets
        ),
        decision_at=decision_at,
    )


def run_one_auto_binance_recovery_from_env(
    *,
    recovery_worker_token: str,
    recovery_claim_seconds: int,
    recovery_ttl_seconds: int,
    source_ref: str,
    env: Mapping[str, str] | None = None,
    initial_runtime=None,
    foreign_cycle_aborter=None,
    supabase_client=None,
    provider_factory: ProviderFactory = BinanceSpotRecoveryEvidenceProvider,
    clock: Callable[[], float] = time.time,
) -> AutoRecoveryStartupReceipt:
    """One-shot Phase95 helper; Phase92 owns all lease/HTTP cleanup."""
    with RecoveryWorkerSession.from_env(
        env=env,
        initial_runtime=initial_runtime,
        foreign_cycle_aborter=foreign_cycle_aborter,
        client=supabase_client,
    ) as session:
        return run_one_auto_binance_recovery(
            session,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            recovery_ttl_seconds=recovery_ttl_seconds,
            source_ref=source_ref,
            provider_factory=provider_factory,
            clock=clock,
        )
