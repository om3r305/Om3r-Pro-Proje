from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

from .evidence_ledger import content_hash
from .phase55_rebalance_execution_intents import (
    RebalanceExecutionInstruction,
    RebalanceExecutionPlan,
    RiskReductionIntent,
)
from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import (
    ExecutionMarketInput,
    ShadowExecutionCycle,
    run_shadow_execution_cycle,
)
from .phase67_durable_runtime_orchestrator import (
    DurableRuntimeCheckpoint,
    DurableRuntimeError,
)
from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase82_recovery_claim_fencing import (
    RecoveryClaimError,
    RecoveryClaimReceipt,
    RecoveryClaimRenewal,
)
from .phase83_atomic_recovery_start import (
    AtomicRecoveryStartExecutionStep,
    AtomicRecoveryStartReceipt,
    PersistedAtomicRecoveryStartSupervisor,
)

PHASE84_SCHEMA_VERSION = "brian.phase84-recovery-execution-checkpoint.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class RecoveryExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryCheckpointReceipt:
    runtime_id: str
    original_cycle_id: str
    recovery_cycle_id: str
    dispatch_id: str
    checkpoint_id: str
    journal_stage: str
    version: int
    current_version: int
    fencing_token: int
    recovery_claim_fencing_token: int
    status: str
    committed: bool
    duplicate: bool
    terminal: bool
    head_state_id: str | None = None
    schema_version: str = PHASE84_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        for label, value in (
            ("original_cycle_id", self.original_cycle_id),
            ("recovery_cycle_id", self.recovery_cycle_id),
            ("checkpoint_id", self.checkpoint_id),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash")
        if self.original_cycle_id == self.recovery_cycle_id:
            raise ValueError("recovery cycle must differ from original cycle")
        if self.journal_stage not in {
            "CYCLE_CREATED",
            "PAPER_APPLIED",
            "LOCAL_PROJECTED",
            "RECONCILIATION_REQUIRED",
            "RECONCILED",
            "COMMITTED",
            "ABORTED",
        }:
            raise ValueError("unsupported recovery journal stage")
        if self.version < 0 or self.current_version < 0:
            raise ValueError("runtime versions must be non-negative")
        if self.fencing_token <= 0 or self.recovery_claim_fencing_token <= 0:
            raise ValueError("runtime/recovery claim fences must be positive")
        if self.committed and self.status not in {
            "COMMITTED",
            "DUPLICATE_CURRENT",
            "RECOVERY_COMMITTED_PENDING_AUDIT",
        }:
            raise ValueError("unsupported committed recovery status")
        if self.terminal:
            if not self.committed or self.journal_stage != "COMMITTED":
                raise ValueError("terminal recovery checkpoint must be COMMITTED")
            if self.status not in {"RECOVERY_COMMITTED_PENDING_AUDIT", "DUPLICATE_CURRENT"}:
                raise ValueError("terminal recovery has invalid status")
        if self.status == "RECOVERY_COMMITTED_PENDING_AUDIT" and not self.terminal:
            raise ValueError("RECOVERY_COMMITTED_PENDING_AUDIT must be terminal")
        if self.head_state_id is not None and len(self.head_state_id) != 64:
            raise ValueError("head_state_id must be a content hash")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase84 checkpoint must remain shadow-only")


@dataclass(frozen=True, slots=True)
class RecoveryExecutionCoreStep:
    recovery_cycle_id: str | None
    write_ahead: RecoveryCheckpointReceipt | None
    progress: RecoveryCheckpointReceipt | None
    durable_status: str | None
    outcome: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE84_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryExecutionStep:
    start_step: AtomicRecoveryStartExecutionStep
    recovery_cycle_id: str | None
    write_ahead: RecoveryCheckpointReceipt | None
    progress: RecoveryCheckpointReceipt | None
    durable_status: str | None
    outcome: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE84_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RecoveryExecutionError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(
    value: object,
    label: str,
    *,
    default: int | None = None,
) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise RecoveryExecutionError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryExecutionError(f"{label} must be integer") from exc


def _optional_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64:
        raise RecoveryExecutionError(f"{label} must be a content hash")
    return result


class RecoveryCheckpointStore:
    """Commit recovery write-ahead/progress under the current recovery claim fence."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def commit(
        self,
        lease,
        claim: RecoveryClaimReceipt,
        *,
        recovery_cycle_id: str,
        worker_token: str,
        expected_version: int,
        checkpoint: DurableRuntimeCheckpoint,
    ) -> RecoveryCheckpointReceipt:
        if not lease.acquired:
            raise RecoveryExecutionError("runtime lease is not acquired")
        if not claim.claimed or claim.claim_fencing_token is None:
            raise RecoveryExecutionError("active recovery claim is required")
        if claim.runtime_id != lease.runtime_id:
            raise RecoveryExecutionError("recovery claim runtime does not match lease")
        if claim.fencing_token != lease.fencing_token:
            raise RecoveryExecutionError("recovery claim runtime fence does not match lease")
        if claim.worker_token != worker_token:
            raise RecoveryExecutionError("recovery claim worker does not match requester")
        if len(recovery_cycle_id) != 64:
            raise ValueError("recovery_cycle_id must be a content hash")
        if recovery_cycle_id == claim.cycle_id:
            raise ValueError("recovery cycle must differ from original cycle")
        if expected_version <= 0:
            raise ValueError("expected_version must be positive")

        try:
            canonical = DurableRuntimeCheckpoint.from_dict(checkpoint.to_dict())
        except (DurableRuntimeError, ValueError, TypeError, KeyError) as exc:
            raise RecoveryExecutionError(
                f"recovery checkpoint failed Phase67 validation: {exc}"
            ) from exc

        row = _mapping(
            self._rpc(
                "brian_commit_shadow_recovery_checkpoint",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_original_cycle_id": claim.cycle_id,
                    "p_recovery_cycle_id": recovery_cycle_id,
                    "p_worker_token": worker_token,
                    "p_recovery_claim_fencing_token": claim.claim_fencing_token,
                    "p_expected_version": int(expected_version),
                    "p_checkpoint": canonical.to_dict(),
                },
            ),
            "brian_commit_shadow_recovery_checkpoint",
        )

        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise RecoveryExecutionError("recovery commit runtime_id mismatch")
        if str(row.get("original_cycle_id", "")) != claim.cycle_id:
            raise RecoveryExecutionError("recovery commit original cycle mismatch")
        if str(row.get("recovery_cycle_id", "")) != recovery_cycle_id:
            raise RecoveryExecutionError("recovery commit recovery cycle mismatch")
        if str(row.get("checkpoint_id", "")) != canonical.checkpoint_id:
            raise RecoveryExecutionError("recovery commit checkpoint_id mismatch")

        committed_raw = row.get("committed")
        terminal_raw = row.get("terminal")
        if not isinstance(committed_raw, bool) or not isinstance(terminal_raw, bool):
            raise RecoveryExecutionError("committed/terminal must be boolean")

        receipt = RecoveryCheckpointReceipt(
            runtime_id=lease.runtime_id,
            original_cycle_id=claim.cycle_id,
            recovery_cycle_id=recovery_cycle_id,
            dispatch_id=str(row.get("dispatch_id", claim.dispatch_id)),
            checkpoint_id=canonical.checkpoint_id,
            journal_stage=str(row.get("journal_stage", "")),
            version=_integer(row.get("version"), "version"),
            current_version=_integer(
                row.get("current_version", row.get("version")),
                "current_version",
            ),
            fencing_token=_integer(
                row.get("fencing_token"),
                "fencing_token",
                default=lease.fencing_token,
            ),
            recovery_claim_fencing_token=_integer(
                row.get("recovery_claim_fencing_token"),
                "recovery_claim_fencing_token",
                default=claim.claim_fencing_token,
            ),
            status=str(row.get("status", "")),
            committed=committed_raw,
            duplicate=bool(row.get("duplicate", False)),
            terminal=terminal_raw,
            head_state_id=_optional_hash(
                row.get("head_state_id"),
                "head_state_id",
            ),
        )
        if receipt.dispatch_id != claim.dispatch_id:
            raise RecoveryExecutionError("recovery commit dispatch drift")
        if receipt.fencing_token != lease.fencing_token:
            raise RecoveryExecutionError("recovery commit runtime fence drift")
        if receipt.recovery_claim_fencing_token != claim.claim_fencing_token:
            raise RecoveryExecutionError("recovery commit claim fence drift")
        return receipt


def _recovery_plan_id(start: AtomicRecoveryStartReceipt) -> str:
    return content_hash({
        "schema_version": PHASE84_SCHEMA_VERSION,
        "kind": "CANCEL_RECOVERY_REDUCE_ONLY",
        "runtime_id": start.runtime_id,
        "original_cycle_id": start.cycle_id,
        "dispatch_id": start.dispatch_id,
        "cancel_risk_receipt_id": start.cancel_risk_receipt_id,
        "recovery_legs": [leg.to_dict() for leg in start.recovery_legs],
    })


def compile_recovery_plan(
    start: AtomicRecoveryStartReceipt,
    *,
    created_at: float,
    ttl_seconds: int,
) -> RebalanceExecutionPlan:
    if not start.started:
        raise RecoveryExecutionError("recovery must cross Phase83 STARTED first")
    if start.risk_state not in {"ACTIVE", "REDUCING"}:
        raise RecoveryExecutionError("recovery STARTED risk state is not executable")
    if not math.isfinite(created_at):
        raise ValueError("created_at must be finite")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    plan_id = _recovery_plan_id(start)
    instructions: list[RebalanceExecutionInstruction] = []
    for leg in start.recovery_legs:
        intent_id = content_hash({
            "schema_version": PHASE84_SCHEMA_VERSION,
            "plan_id": plan_id,
            "asset_id": leg.asset_id,
            "current_weight": leg.current_weight,
            "target_weight": leg.target_weight,
            "reduce_weight": leg.reduce_weight,
        })
        reduction = RiskReductionIntent(
            intent_id=intent_id,
            asset_id=leg.asset_id,
            current_direction=leg.current_direction,
            order_direction=leg.order_direction,
            reduce_weight=leg.reduce_weight,
            current_weight=leg.current_weight,
            resulting_weight=leg.target_weight,
            reason="PHASE84_CANCEL_RECOVERY",
            created_at=float(created_at),
            ttl_seconds=int(ttl_seconds),
        )
        instructions.append(RebalanceExecutionInstruction(
            kind="CLOSE" if abs(leg.target_weight) <= 1e-12 else "REDUCE",
            asset_id=leg.asset_id,
            current_weight=leg.current_weight,
            planned_weight=leg.target_weight,
            planned_delta=leg.target_weight - leg.current_weight,
            reduction_intent=reduction,
            reason="durable cancel recovery; restore only pre-cycle exposure",
        ))

    return RebalanceExecutionPlan(
        instructions=tuple(instructions),
        skipped_assets=(),
        plan_id=plan_id,
    )


def build_recovery_cycle(
    start: AtomicRecoveryStartReceipt,
    *,
    equity_usd: float,
    available_cash_usd: float,
    current_weights: Mapping[str, float],
    markets: Mapping[str, ExecutionMarketInput],
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
    trading_state: str,
    created_at: float,
    ttl_seconds: int,
) -> ShadowExecutionCycle:
    if trading_state not in {"ACTIVE", "REDUCING"}:
        raise RecoveryExecutionError("recovery execution requires ACTIVE/REDUCING risk")

    plan = compile_recovery_plan(
        start,
        created_at=created_at,
        ttl_seconds=ttl_seconds,
    )
    cycle = run_shadow_execution_cycle(
        plan,
        equity_usd=equity_usd,
        available_cash_usd=available_cash_usd,
        current_weights=current_weights,
        markets=markets,
        risk_limits_by_asset=risk_limits_by_asset,
        trading_state=trading_state,
    )

    expected = {leg.asset_id: leg for leg in start.recovery_legs}
    if len(cycle.items) != len(expected):
        raise RecoveryExecutionError("recovery cycle item count differs from STARTED legs")

    for item in cycle.items:
        leg = expected.get(item.asset_id)
        if leg is None:
            raise RecoveryExecutionError("recovery cycle contains unstarted asset")
        if not item.risk_receipt.allowed or not item.risk_receipt.reduce_only:
            raise RecoveryExecutionError(
                f"{item.asset_id} recovery was not independently risk-allowed reduce-only"
            )
        if item.execution_receipt is None or item.execution_receipt.status != "FILLED":
            status = None if item.execution_receipt is None else item.execution_receipt.status
            raise RecoveryExecutionError(
                f"{item.asset_id} recovery simulation is not fully filled: {status}"
            )
        if not math.isclose(
            float(item.risk_receipt.projected_position_weight),
            leg.target_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RecoveryExecutionError(
                f"{item.asset_id} recovery projected weight drifted from STARTED target"
            )
        expected_side = "BUY" if leg.order_direction > 0 else "SELL"
        if item.execution_receipt.side != expected_side:
            raise RecoveryExecutionError(
                f"{item.asset_id} recovery execution side drifted from STARTED leg"
            )

    if cycle.denied_assets or cycle.pending_reversal_assets:
        raise RecoveryExecutionError("recovery cycle cannot contain denied/pending assets")
    if cycle.reserved_new_risk_cash_usd > 1e-12:
        raise RecoveryExecutionError("recovery cycle unexpectedly reserved new-risk cash")
    return cycle


class PersistedRecoveryExecutionSupervisor:
    """Phase83 STARTED -> durable recovery write-ahead -> paper/reconcile -> completion."""

    def __init__(
        self,
        *,
        checkpoints: RecoveryCheckpointStore,
        start_supervisor: PersistedAtomicRecoveryStartSupervisor | None = None,
        runtime_supervisor=None,
        claims=None,
    ) -> None:
        if start_supervisor is None and (runtime_supervisor is None or claims is None):
            raise ValueError(
                "Phase84 requires either start_supervisor or direct runtime_supervisor+claims"
            )
        if start_supervisor is not None and (
            runtime_supervisor is not None or claims is not None
        ):
            raise ValueError(
                "Phase84 start_supervisor and direct dependencies are mutually exclusive"
            )
        self.start_supervisor = start_supervisor
        self.checkpoints = checkpoints
        self._direct_runtime_supervisor = runtime_supervisor
        self._direct_claims = claims

    def _runtime_supervisor(self):
        if self._direct_runtime_supervisor is not None:
            return self._direct_runtime_supervisor
        assert self.start_supervisor is not None
        return self.start_supervisor._runtime_supervisor()

    def _claims(self):
        if self._direct_claims is not None:
            return self._direct_claims
        assert self.start_supervisor is not None
        return self.start_supervisor.claim_supervisor.claims

    @staticmethod
    def _handle_renewal(supervisor, renewal: RecoveryClaimRenewal) -> None:
        if renewal.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost while renewing recovery execution claim"
            )
        if renewal.status == "RENEWAL_LOST":
            raise RecoveryClaimError(
                "recovery claim lost; acquire a fresh recovery claim before continuing"
            )
        if renewal.status == "RISK_STATE_UNAVAILABLE":
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "current risk state unavailable during recovery execution"
            )
        if renewal.status not in {"RENEWED", "RENEWAL_BLOCKED_RISK"}:
            raise RecoveryExecutionError(
                f"unsupported recovery renewal status {renewal.status}"
            )

    @staticmethod
    def _handle_commit(supervisor, receipt: RecoveryCheckpointReceipt) -> None:
        if receipt.committed:
            if receipt.current_version > supervisor.persisted_version:
                supervisor.accept_external_checkpoint_commit(
                    checkpoint_id=receipt.checkpoint_id,
                    version=receipt.current_version,
                )
            elif receipt.current_version != supervisor.persisted_version:
                supervisor._valid = False
                raise PersistedRuntimeStaleError(
                    "recovery checkpoint version moved backwards"
                )
            return

        if receipt.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost during recovery checkpoint commit"
            )
        if receipt.status == "CLAIM_LOST":
            raise RecoveryClaimError(
                "recovery claim lost before checkpoint commit"
            )
        supervisor._valid = False
        raise PersistedRuntimeStaleError(
            f"recovery checkpoint commit rejected with {receipt.status}"
        )

    @staticmethod
    def _existing_recovery_cycle(runtime, plan_id: str) -> ShadowExecutionCycle | None:
        matches: list[ShadowExecutionCycle] = []
        for cycle_id in runtime.journal.cycle_ids:
            cycle = runtime.journal.cycle(cycle_id)
            if cycle.source_plan_id == plan_id:
                matches.append(cycle)
        if len(matches) > 1:
            raise RecoveryExecutionError(
                "multiple durable recovery cycles share one recovery plan id"
            )
        return None if not matches else matches[0]

    def execute_started_recovery(
        self,
        *,
        start: AtomicRecoveryStartReceipt,
        claim: RecoveryClaimReceipt,
        recovery_worker_token: str,
        recovery_claim_seconds: int,
        recovery_markets: Mapping[str, ExecutionMarketInput],
        recovery_risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
        recovery_ttl_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
        base_outcome: str = "RECOVERY_RESUME",
    ) -> RecoveryExecutionCoreStep:
        """Resume Phase84 directly from durable Phase82/83 evidence.

        This method is intentionally independent of a fresh governed signal so a
        crash-restarted recovery worker can continue an already-durable recovery
        obligation. It still requires the current runtime lease, current recovery
        claim and immutable STARTED evidence.
        """
        supervisor = self._runtime_supervisor()
        if not start.started:
            raise RecoveryExecutionError("Phase84 direct resume requires STARTED evidence")
        if not claim.claimed or claim.claim_fencing_token is None:
            raise RecoveryExecutionError("Phase84 direct resume requires active recovery claim")
        if start.runtime_id != supervisor.runtime_id or claim.runtime_id != supervisor.runtime_id:
            raise RecoveryExecutionError("recovery evidence runtime does not match supervisor")
        if start.cycle_id != claim.cycle_id:
            raise RecoveryExecutionError("recovery STARTED/claim original cycle mismatch")
        if start.dispatch_id != claim.dispatch_id:
            raise RecoveryExecutionError("recovery STARTED/claim dispatch mismatch")
        if start.cancel_risk_receipt_id != claim.cancel_risk_receipt_id:
            raise RecoveryExecutionError("recovery STARTED/claim cancel receipt mismatch")
        if start.recovery_claim_fencing_token != claim.claim_fencing_token:
            raise RecoveryExecutionError("recovery STARTED/claim fence mismatch")
        if claim.worker_token != recovery_worker_token:
            raise RecoveryExecutionError("recovery claim worker does not match resume worker")
        if tuple(leg.to_dict() for leg in start.recovery_legs) != tuple(
            leg.to_dict() for leg in claim.recovery_legs
        ):
            raise RecoveryExecutionError("recovery STARTED legs differ from current claim")

        renewal = self._claims().renew(
            supervisor.lease,
            claim,
            worker_token=recovery_worker_token,
            claim_seconds=recovery_claim_seconds,
        )
        self._handle_renewal(supervisor, renewal)
        if renewal.status == "RENEWAL_BLOCKED_RISK":
            checkpoint = supervisor.runtime.checkpoint()
            return RecoveryExecutionCoreStep(
                recovery_cycle_id=None,
                write_ahead=None,
                progress=None,
                durable_status=None,
                outcome=f"{base_outcome}_EXECUTION_WAIT_RISK_RELEASE",
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        plan_id = _recovery_plan_id(start)
        cycle = self._existing_recovery_cycle(supervisor.runtime, plan_id)
        write_ahead: RecoveryCheckpointReceipt | None = None

        if cycle is None:
            head = supervisor.runtime.ledger.head_state
            if start.head_state_id is not None and head.state_id != start.head_state_id:
                supervisor._valid = False
                raise PersistedRuntimeStaleError(
                    "local Phase60 head no longer matches recovery STARTED anchor"
                )
            weights = dict(head.position_weights)
            for leg in start.recovery_legs:
                current = float(weights.get(leg.asset_id, 0.0))
                if not math.isclose(
                    current,
                    leg.current_weight,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    supervisor._valid = False
                    raise PersistedRuntimeStaleError(
                        f"{leg.asset_id} authoritative weight drifted before recovery"
                    )

            cycle = build_recovery_cycle(
                start,
                equity_usd=head.equity_usd,
                available_cash_usd=head.available_cash_usd,
                current_weights=weights,
                markets=recovery_markets,
                risk_limits_by_asset=recovery_risk_limits_by_asset,
                trading_state=str(renewal.risk_state),
                created_at=observed_at,
                ttl_seconds=recovery_ttl_seconds,
            )
            supervisor.runtime.journal_cycle(cycle)
            checkpoint = supervisor.runtime.checkpoint()
            write_ahead = self.checkpoints.commit(
                supervisor.lease,
                claim,
                recovery_cycle_id=cycle.cycle_id,
                worker_token=recovery_worker_token,
                expected_version=supervisor.persisted_version,
                checkpoint=checkpoint,
            )
            self._handle_commit(supervisor, write_ahead)

        stage = supervisor.runtime.journal.latest_stage(cycle.cycle_id)
        if stage == "COMMITTED":
            checkpoint = supervisor.runtime.checkpoint()
            progress = RecoveryCheckpointReceipt(
                runtime_id=supervisor.runtime_id,
                original_cycle_id=claim.cycle_id,
                recovery_cycle_id=cycle.cycle_id,
                dispatch_id=claim.dispatch_id,
                checkpoint_id=checkpoint.checkpoint_id,
                journal_stage="COMMITTED",
                version=supervisor.persisted_version,
                current_version=supervisor.persisted_version,
                fencing_token=supervisor.lease.fencing_token,
                recovery_claim_fencing_token=claim.claim_fencing_token,
                status="DUPLICATE_CURRENT",
                committed=True,
                duplicate=True,
                terminal=True,
                head_state_id=supervisor.runtime.ledger.head_state.state_id,
            )
            return RecoveryExecutionCoreStep(
                recovery_cycle_id=cycle.cycle_id,
                write_ahead=write_ahead,
                progress=progress,
                durable_status="COMMITTED",
                outcome=f"{base_outcome}_RECOVERY_COMMITTED_PENDING_AUDIT",
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )
        if stage == "ABORTED":
            raise RecoveryExecutionError(
                "durable recovery cycle is ABORTED and cannot be auto-resumed"
            )

        # Re-check risk after durable write-ahead and immediately before the
        # replay-safe paper side effect.
        renewal = self._claims().renew(
            supervisor.lease,
            claim,
            worker_token=recovery_worker_token,
            claim_seconds=recovery_claim_seconds,
        )
        self._handle_renewal(supervisor, renewal)
        if renewal.status == "RENEWAL_BLOCKED_RISK":
            checkpoint = supervisor.runtime.checkpoint()
            return RecoveryExecutionCoreStep(
                recovery_cycle_id=cycle.cycle_id,
                write_ahead=write_ahead,
                progress=None,
                durable_status=stage,
                outcome=f"{base_outcome}_WRITE_AHEAD_WAIT_RISK_RELEASE",
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        durable = supervisor.advance_pending_in_memory(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        checkpoint = supervisor.runtime.checkpoint()
        progress = self.checkpoints.commit(
            supervisor.lease,
            claim,
            recovery_cycle_id=cycle.cycle_id,
            worker_token=recovery_worker_token,
            expected_version=supervisor.persisted_version,
            checkpoint=checkpoint,
        )
        self._handle_commit(supervisor, progress)

        if progress.terminal:
            outcome = f"{base_outcome}_RECOVERY_COMMITTED_PENDING_AUDIT"
        elif durable.status == "RECONCILIATION_BLOCKED":
            outcome = f"{base_outcome}_RECOVERY_RECONCILIATION_BLOCKED"
        elif durable.status == "MARKS_REQUIRED":
            outcome = f"{base_outcome}_RECOVERY_MARKS_REQUIRED"
        else:
            outcome = f"{base_outcome}_RECOVERY_PROGRESS"

        return RecoveryExecutionCoreStep(
            recovery_cycle_id=cycle.cycle_id,
            write_ahead=write_ahead,
            progress=progress,
            durable_status=durable.status,
            outcome=outcome,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=supervisor.runtime.checkpoint().checkpoint_id,
        )

    def process_governed_cycle(
        self,
        governed,
        *,
        worker_token: str,
        claim_seconds: int,
        recovery_worker_token: str,
        recovery_claim_seconds: int,
        recovery_markets: Mapping[str, ExecutionMarketInput],
        recovery_risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
        recovery_ttl_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> RecoveryExecutionStep:
        if self.start_supervisor is None:
            raise RecoveryExecutionError(
                "process_governed_cycle requires Phase83 start_supervisor"
            )
        start_step = self.start_supervisor.process_governed_cycle(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        start = start_step.start
        claim = start_step.claim_step.claim

        if start is None or not start.started or claim is None or not claim.claimed:
            return RecoveryExecutionStep(
                start_step=start_step,
                recovery_cycle_id=None,
                write_ahead=None,
                progress=None,
                durable_status=None,
                outcome=start_step.outcome,
                persisted_version=start_step.persisted_version,
                checkpoint_id=start_step.checkpoint_id,
            )

        core = self.execute_started_recovery(
            start=start,
            claim=claim,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            recovery_markets=recovery_markets,
            recovery_risk_limits_by_asset=recovery_risk_limits_by_asset,
            recovery_ttl_seconds=recovery_ttl_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
            base_outcome=start_step.outcome,
        )
        return RecoveryExecutionStep(
            start_step=start_step,
            recovery_cycle_id=core.recovery_cycle_id,
            write_ahead=core.write_ahead,
            progress=core.progress,
            durable_status=core.durable_status,
            outcome=core.outcome,
            persisted_version=core.persisted_version,
            checkpoint_id=core.checkpoint_id,
        )
