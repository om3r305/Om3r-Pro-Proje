from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase70_durable_runtime_store import RuntimeLease
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

PHASE78_SCHEMA_VERSION = "brian.phase78-execution-kill-switch.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class ExecutionKillSwitchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionKillSwitchDecision:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    runtime_version: int
    status: str
    proceed: bool
    cancel_requested: bool
    terminal: bool
    risk_version: int | None
    risk_receipt_id: str | None
    journal_stage: str | None
    reason: str | None
    phase: str | None
    completion_checkpoint_id: str | None
    schema_version: str = PHASE78_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash")
        if self.runtime_version < 0:
            raise ValueError("runtime_version cannot be negative")
        if self.risk_version is not None and self.risk_version <= 0:
            raise ValueError("risk_version must be positive")
        if self.risk_receipt_id is not None and len(self.risk_receipt_id) != 64:
            raise ValueError("risk_receipt_id must be a content hash")
        if self.completion_checkpoint_id is not None and len(self.completion_checkpoint_id) != 64:
            raise ValueError("completion_checkpoint_id must be a content hash")
        if self.proceed and self.terminal:
            raise ValueError("terminal kill-switch decision cannot proceed")
        if not self.shadow_only or self.live_execution:
            raise ValueError("kill-switch decision must remain shadow-only")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ExecutionKillSwitchError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ExecutionKillSwitchError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionKillSwitchError(f"{label} must be integer") from exc


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ExecutionKillSwitchError(f"{label} must be boolean")
    return value


class ExecutionKillSwitchStore:
    """Evaluate current persisted risk against an owned Phase77 claim."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def check(
        self,
        lease: RuntimeLease,
        claim: ExecutionClaimReceipt,
        *,
        worker_token: str,
    ) -> ExecutionKillSwitchDecision:
        if not worker_token.strip():
            raise ValueError("worker_token is required")
        if claim.runtime_id != lease.runtime_id:
            raise ExecutionKillSwitchError("claim runtime does not match lease")
        if claim.claim_fencing_token <= 0:
            raise ExecutionKillSwitchError("claim has no positive fencing token")

        row = _mapping(
            self._rpc(
                "brian_check_shadow_execution_kill_switch",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": claim.cycle_id,
                    "p_worker_token": worker_token,
                    "p_claim_fencing_token": claim.claim_fencing_token,
                },
            ),
            "brian_check_shadow_execution_kill_switch",
        )
        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise ExecutionKillSwitchError("kill-switch runtime_id mismatch")
        if str(row.get("cycle_id", "")) != claim.cycle_id:
            raise ExecutionKillSwitchError("kill-switch cycle_id mismatch")

        status = str(row.get("status", ""))
        proceed = _boolean(row.get("proceed"), "proceed")
        cancel_requested = _boolean(
            row.get("cancel_requested"),
            "cancel_requested",
        )
        terminal = _boolean(row.get("terminal"), "terminal")

        if status in {"PROCEED", "RESUME_ONLY"} and not proceed:
            raise ExecutionKillSwitchError(f"{status} must return proceed=true")
        if status == "CANCELLED_BEFORE_EXECUTION" and not (
            cancel_requested and terminal and not proceed
        ):
            raise ExecutionKillSwitchError(
                "pre-execution cancellation flags are inconsistent"
            )
        if status == "CANCEL_REQUESTED" and not (
            cancel_requested and proceed and not terminal
        ):
            raise ExecutionKillSwitchError(
                "post-start cancel-request flags are inconsistent"
            )
        if status in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"} and proceed:
            raise ExecutionKillSwitchError(f"{status} cannot proceed")
        if status in {"COMPLETED", "ABORTED"} and not terminal:
            raise ExecutionKillSwitchError(f"{status} must be terminal")

        return ExecutionKillSwitchDecision(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=str(row.get("dispatch_id", claim.dispatch_id)),
            runtime_version=_integer(
                row.get("runtime_version"),
                "runtime_version",
                default=lease.version,
            ),
            status=status,
            proceed=proceed,
            cancel_requested=cancel_requested,
            terminal=terminal,
            risk_version=(
                None
                if row.get("risk_version") is None
                else _integer(row.get("risk_version"), "risk_version")
            ),
            risk_receipt_id=(
                None
                if row.get("risk_receipt_id") is None
                else str(row.get("risk_receipt_id"))
            ),
            journal_stage=(
                None if row.get("journal_stage") is None else str(row.get("journal_stage"))
            ),
            reason=None if row.get("reason") is None else str(row.get("reason")),
            phase=None if row.get("phase") is None else str(row.get("phase")),
            completion_checkpoint_id=(
                None
                if row.get("completion_checkpoint_id") is None
                else str(row.get("completion_checkpoint_id"))
            ),
        )


@dataclass(frozen=True, slots=True)
class KillSwitchedExecutionStep:
    precheck: ExecutionKillSwitchDecision | None
    postcheck: ExecutionKillSwitchDecision | None
    claim: ExecutionClaimReceipt
    outcome: str
    completion: ClaimCompletionReceipt | None
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE78_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


class PersistedKillSwitchRuntimeSupervisor:
    """Double-check persisted risk around the claimed paper execution boundary."""

    def __init__(
        self,
        *,
        claimed_supervisor: PersistedClaimedRuntimeSupervisor,
        kill_switch: ExecutionKillSwitchStore,
    ) -> None:
        self.claimed_supervisor = claimed_supervisor
        self.kill_switch = kill_switch

    def _fail_closed(
        self,
        decision: ExecutionKillSwitchDecision,
    ) -> None:
        phase77 = self.claimed_supervisor
        supervisor = (
            phase77.dispatched_supervisor
            .governed_supervisor
            .runtime_supervisor
        )
        supervisor._valid = False
        if decision.status == "LEASE_LOST":
            raise PersistedRuntimeLeaseError(
                "runtime lease lost during execution kill-switch check"
            )
        raise PersistedRuntimeStaleError(
            f"execution kill-switch failed with {decision.status}; reload required"
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
    ) -> KillSwitchedExecutionStep:
        phase77 = self.claimed_supervisor
        phase76 = phase77.dispatched_supervisor
        phase75 = phase76.governed_supervisor
        supervisor = phase75.runtime_supervisor

        _, _, claim, terminal = phase77.authorize_submit_and_claim(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
        )

        if terminal is not None:
            checkpoint = supervisor.runtime.checkpoint()
            return KillSwitchedExecutionStep(
                precheck=None,
                postcheck=None,
                claim=claim,
                outcome=terminal,
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        precheck = self.kill_switch.check(
            supervisor.lease,
            claim,
            worker_token=worker_token,
        )

        if precheck.status in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"}:
            self._fail_closed(precheck)

        if precheck.status == "CANCELLED_BEFORE_EXECUTION":
            if supervisor.runtime.journal.latest_stage(claim.cycle_id) == "CYCLE_CREATED":
                phase75.abort_authorized_cycle(
                    cycle_id=claim.cycle_id,
                    reason=f"phase78:{precheck.reason or 'kill_switch'}",
                )
            checkpoint = supervisor.runtime.checkpoint()
            return KillSwitchedExecutionStep(
                precheck=precheck,
                postcheck=None,
                claim=claim,
                outcome="CANCELLED_BEFORE_EXECUTION",
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if precheck.status == "ABORTED":
            checkpoint = supervisor.runtime.checkpoint()
            return KillSwitchedExecutionStep(
                precheck=precheck,
                postcheck=None,
                claim=claim,
                outcome="ABORTED",
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if precheck.status == "COMPLETED":
            checkpoint = supervisor.runtime.checkpoint()
            return KillSwitchedExecutionStep(
                precheck=precheck,
                postcheck=None,
                claim=claim,
                outcome="ALREADY_COMPLETED",
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if not precheck.proceed:
            raise ExecutionKillSwitchError(
                f"unexpected non-proceeding precheck status {precheck.status}"
            )

        _, outcome, _ = phase77.advance_claimed(
            claim,
            worker_token=worker_token,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
            complete_on_commit=False,
        )

        postcheck = self.kill_switch.check(
            supervisor.lease,
            claim,
            worker_token=worker_token,
        )

        if postcheck.status in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"}:
            self._fail_closed(postcheck)

        if postcheck.status == "CANCEL_REQUESTED":
            outcome = f"{outcome}_CANCEL_REQUESTED"
        elif postcheck.status == "CANCELLED_BEFORE_EXECUTION":
            # If this ever appears after advance, Phase67 did not move beyond
            # CYCLE_CREATED; keep the terminal cancellation explicit.
            if supervisor.runtime.journal.latest_stage(claim.cycle_id) == "CYCLE_CREATED":
                phase75.abort_authorized_cycle(
                    cycle_id=claim.cycle_id,
                    reason=f"phase78:{postcheck.reason or 'kill_switch'}",
                )
            outcome = "CANCELLED_BEFORE_EXECUTION"

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
        return KillSwitchedExecutionStep(
            precheck=precheck,
            postcheck=postcheck,
            claim=claim,
            outcome=outcome,
            completion=completion,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=checkpoint.checkpoint_id,
        )
