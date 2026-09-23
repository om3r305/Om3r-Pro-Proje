from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase67_durable_runtime_orchestrator import (
    DurableRuntimeCheckpoint,
    DurableRuntimeError,
)
from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase77_execution_claim_lifecycle import (
    ClaimCompletionReceipt,
    ExecutionClaimError,
    ExecutionClaimReceipt,
    PersistedClaimedRuntimeSupervisor,
)
from .phase78_execution_kill_switch import (
    ExecutionKillSwitchDecision,
    ExecutionKillSwitchStore,
)
from .phase79_atomic_execution_start import (
    AtomicExecutionStartReceipt,
    AtomicExecutionStartStore,
)

PHASE80_SCHEMA_VERSION = "brian.phase80-claim-fenced-checkpoint.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class ClaimFencedCheckpointError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ClaimFencedCheckpointReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    checkpoint_id: str
    journal_stage: str
    version: int
    current_version: int
    fencing_token: int
    claim_fencing_token: int
    status: str
    committed: bool
    duplicate: bool
    schema_version: str = PHASE80_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        for label, value in (
            ("cycle_id", self.cycle_id),
            ("dispatch_id", self.dispatch_id),
            ("checkpoint_id", self.checkpoint_id),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if self.journal_stage not in {
            "PAPER_APPLIED",
            "LOCAL_PROJECTED",
            "RECONCILIATION_REQUIRED",
            "RECONCILED",
            "COMMITTED",
            "ABORTED",
        }:
            raise ValueError("invalid post-STARTED journal_stage")
        if self.version < 0 or self.current_version < 0:
            raise ValueError("runtime versions must be non-negative")
        if self.fencing_token <= 0 or self.claim_fencing_token <= 0:
            raise ValueError("runtime/claim fences must be positive")
        if self.status == "COMMITTED" and (not self.committed or self.duplicate):
            raise ValueError("COMMITTED flags are inconsistent")
        if self.status == "DUPLICATE_CURRENT" and not (
            self.committed and self.duplicate
        ):
            raise ValueError("DUPLICATE_CURRENT flags are inconsistent")
        if self.committed and self.status not in {
            "COMMITTED",
            "DUPLICATE_CURRENT",
        }:
            raise ValueError("unsupported committed status")
        if not self.shadow_only or self.live_execution:
            raise ValueError("claim-fenced commit must remain shadow-only")


@dataclass(frozen=True, slots=True)
class ClaimFencedExecutionStep:
    claim: ExecutionClaimReceipt
    start: AtomicExecutionStartReceipt | None
    checkpoint_commit: ClaimFencedCheckpointReceipt | None
    postcheck: ExecutionKillSwitchDecision | None
    outcome: str
    completion: ClaimCompletionReceipt | None
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE80_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ClaimFencedCheckpointError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise ClaimFencedCheckpointError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ClaimFencedCheckpointError(f"{label} must be integer") from exc


class ClaimFencedCheckpointStore:
    """Commit a post-STARTED runtime checkpoint only for the current claim owner."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def commit(
        self,
        lease,
        claim: ExecutionClaimReceipt,
        *,
        worker_token: str,
        expected_version: int,
        checkpoint: DurableRuntimeCheckpoint,
    ) -> ClaimFencedCheckpointReceipt:
        if not lease.acquired:
            raise ClaimFencedCheckpointError("runtime lease is not acquired")
        if claim.runtime_id != lease.runtime_id:
            raise ClaimFencedCheckpointError("claim runtime does not match lease")
        if claim.fencing_token != lease.fencing_token:
            raise ClaimFencedCheckpointError("claim runtime fence does not match lease")
        if not claim.claimed or claim.claim_fencing_token <= 0:
            raise ClaimFencedCheckpointError("active owned claim is required")
        if not worker_token.strip():
            raise ValueError("worker_token is required")
        if expected_version <= 0:
            raise ValueError("expected_version must be positive")

        try:
            canonical = DurableRuntimeCheckpoint.from_dict(checkpoint.to_dict())
        except (DurableRuntimeError, ValueError, TypeError, KeyError) as exc:
            raise ClaimFencedCheckpointError(
                f"checkpoint failed Phase67 validation before claim-fenced commit: {exc}"
            ) from exc

        row = _mapping(
            self._rpc(
                "brian_commit_claimed_shadow_runtime_checkpoint",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": claim.cycle_id,
                    "p_worker_token": worker_token,
                    "p_claim_fencing_token": claim.claim_fencing_token,
                    "p_expected_version": int(expected_version),
                    "p_checkpoint": canonical.to_dict(),
                },
            ),
            "brian_commit_claimed_shadow_runtime_checkpoint",
        )

        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise ClaimFencedCheckpointError("commit runtime_id mismatch")
        if str(row.get("cycle_id", "")) != claim.cycle_id:
            raise ClaimFencedCheckpointError("commit cycle_id mismatch")
        if str(row.get("dispatch_id", "")) != claim.dispatch_id:
            raise ClaimFencedCheckpointError("commit dispatch_id mismatch")
        if str(row.get("checkpoint_id", "")) != canonical.checkpoint_id:
            raise ClaimFencedCheckpointError("commit checkpoint_id mismatch")

        status = str(row.get("status", ""))
        committed_raw = row.get("committed")
        if not isinstance(committed_raw, bool):
            raise ClaimFencedCheckpointError("committed must be boolean")
        committed = committed_raw
        duplicate = bool(row.get("duplicate", False))

        receipt = ClaimFencedCheckpointReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=claim.dispatch_id,
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
            claim_fencing_token=_integer(
                row.get("claim_fencing_token"),
                "claim_fencing_token",
                default=claim.claim_fencing_token,
            ),
            status=status,
            committed=committed,
            duplicate=duplicate,
        )
        if receipt.fencing_token != lease.fencing_token:
            raise ClaimFencedCheckpointError(
                "database commit runtime fence does not match lease"
            )
        if receipt.claim_fencing_token != claim.claim_fencing_token:
            raise ClaimFencedCheckpointError(
                "database commit claim fence does not match owned claim"
            )
        return receipt


class PersistedClaimFencedRuntimeSupervisor:
    """Phase79 STARTED + current Phase77 claim fence + Phase70 authoritative commit."""

    def __init__(
        self,
        *,
        claimed_supervisor: PersistedClaimedRuntimeSupervisor,
        starts: AtomicExecutionStartStore,
        checkpoints: ClaimFencedCheckpointStore,
        kill_switch: ExecutionKillSwitchStore,
    ) -> None:
        self.claimed_supervisor = claimed_supervisor
        self.starts = starts
        self.checkpoints = checkpoints
        self.kill_switch = kill_switch

    def _runtime_supervisor(self):
        return (
            self.claimed_supervisor
            .dispatched_supervisor
            .governed_supervisor
            .runtime_supervisor
        )

    def process_governed_cycle(
        self,
        governed,
        *,
        worker_token: str,
        claim_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> ClaimFencedExecutionStep:
        phase77 = self.claimed_supervisor
        phase76 = phase77.dispatched_supervisor
        phase75 = phase76.governed_supervisor
        supervisor = self._runtime_supervisor()

        _, _, claim, terminal = phase77.authorize_submit_and_claim(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
        )

        if terminal is not None:
            checkpoint = supervisor.runtime.checkpoint()
            return ClaimFencedExecutionStep(
                claim=claim,
                start=None,
                checkpoint_commit=None,
                postcheck=None,
                outcome=terminal,
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        start = self.starts.mark_started(
            supervisor.lease,
            claim,
            worker_token=worker_token,
        )

        if start.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost at Phase80 execution-start boundary"
            )
        if start.status == "RISK_STATE_UNAVAILABLE":
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "risk state unavailable at Phase80 execution-start boundary"
            )
        if start.status == "CLAIM_LOST":
            raise ExecutionClaimError(
                "execution claim lost before Phase80 STARTED"
            )

        if start.status == "CANCELLED_BEFORE_EXECUTION":
            if supervisor.runtime.journal.latest_stage(claim.cycle_id) == "CYCLE_CREATED":
                phase75.abort_authorized_cycle(
                    cycle_id=claim.cycle_id,
                    reason=f"phase80:{start.reason or 'phase78_cancel'}",
                )
            checkpoint = supervisor.runtime.checkpoint()
            return ClaimFencedExecutionStep(
                claim=claim,
                start=start,
                checkpoint_commit=None,
                postcheck=None,
                outcome="CANCELLED_BEFORE_EXECUTION",
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if start.status in {"ABORTED", "COMPLETED"}:
            checkpoint = supervisor.runtime.checkpoint()
            return ClaimFencedExecutionStep(
                claim=claim,
                start=start,
                checkpoint_commit=None,
                postcheck=None,
                outcome=(
                    "ALREADY_COMPLETED"
                    if start.status == "COMPLETED"
                    else "ABORTED"
                ),
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if not start.started:
            raise ClaimFencedCheckpointError(
                f"unexpected non-started status {start.status}"
            )

        try:
            durable = supervisor.advance_pending_in_memory(
                marks=marks,
                observed_at=observed_at,
                source_ref=source_ref,
            )
            checkpoint = supervisor.runtime.checkpoint()
            commit = self.checkpoints.commit(
                supervisor.lease,
                claim,
                worker_token=worker_token,
                expected_version=supervisor.persisted_version,
                checkpoint=checkpoint,
            )
        except Exception:
            # Local paper/projector state may have advanced while DB authority did
            # not. Never continue from that in-memory copy.
            supervisor._valid = False
            raise

        if not commit.committed:
            supervisor._valid = False
            if commit.status == "LEASE_LOST":
                raise PersistedRuntimeLeaseError(
                    "runtime lease lost at claim-fenced checkpoint commit"
                )
            raise PersistedRuntimeStaleError(
                f"claim-fenced checkpoint rejected with {commit.status}; reload required"
            )

        if commit.status == "DUPLICATE_CURRENT":
            if commit.current_version != commit.version:
                supervisor._valid = False
                raise PersistedRuntimeStaleError(
                    "claim-fenced duplicate-current version mismatch"
                )
        elif commit.status != "COMMITTED":
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"unsupported successful claim-fenced commit {commit.status}"
            )

        supervisor.accept_external_checkpoint_commit(
            checkpoint_id=commit.checkpoint_id,
            version=commit.version,
        )

        outcome = "NONE" if durable is None else durable.status
        postcheck = self.kill_switch.check(
            supervisor.lease,
            claim,
            worker_token=worker_token,
        )

        if postcheck.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost during Phase80 post-commit risk check"
            )
        if postcheck.status in {"CLAIM_LOST", "RISK_STATE_UNAVAILABLE"}:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"post-commit risk check failed with {postcheck.status}"
            )
        if postcheck.status == "CANCEL_REQUESTED":
            outcome = f"{outcome}_CANCEL_REQUESTED"
        elif postcheck.status == "CANCELLED_BEFORE_EXECUTION":
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "post-STARTED evidence contradiction: pre-execution cancellation"
            )

        completion = None
        if outcome.startswith("COMMITTED") or postcheck.status == "COMPLETED":
            completion = phase77.claims.complete(
                supervisor.lease,
                claim,
                worker_token=worker_token,
            )
            if not completion.completed:
                outcome = f"COMMITTED_{completion.status}"
            elif postcheck.cancel_requested:
                outcome = "COMMITTED_CANCEL_REQUESTED"
            else:
                outcome = "COMMITTED"

        checkpoint = supervisor.runtime.checkpoint()
        return ClaimFencedExecutionStep(
            claim=claim,
            start=start,
            checkpoint_commit=commit,
            postcheck=postcheck,
            outcome=outcome,
            completion=completion,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=checkpoint.checkpoint_id,
        )
