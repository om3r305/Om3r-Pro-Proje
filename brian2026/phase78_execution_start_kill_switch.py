from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

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

PHASE78_SCHEMA_VERSION = "brian.phase78-execution-start-kill-switch.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class ExecutionStartError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionStartReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    runtime_version: int
    fencing_token: int
    claim_fencing_token: int
    status: str
    started: bool
    cancelled: bool
    duplicate: bool
    risk_version: int | None
    risk_receipt_id: str | None
    risk_state: str | None
    journal_stage: str | None
    resume_only: bool
    cancel_reason: str | None
    schema_version: str = PHASE78_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash when present")
        if self.runtime_version < 0:
            raise ValueError("runtime_version must be non-negative")
        if self.fencing_token <= 0 or self.claim_fencing_token <= 0:
            raise ValueError("runtime/claim fencing tokens must be positive")
        if self.risk_version is not None and self.risk_version <= 0:
            raise ValueError("risk_version must be positive when present")
        if self.risk_receipt_id is not None and len(self.risk_receipt_id) != 64:
            raise ValueError("risk_receipt_id must be a content hash")
        if self.risk_state is not None and self.risk_state not in (
            "ACTIVE",
            "REDUCING",
            "HALTED",
        ):
            raise ValueError("invalid risk_state")
        if self.started and self.cancelled:
            raise ValueError("execution start cannot be both started and cancelled")
        if self.duplicate and not self.started:
            raise ValueError("duplicate start must also be started")
        if not self.shadow_only or self.live_execution:
            raise ValueError("execution-start receipt must remain shadow-only")


@dataclass(frozen=True, slots=True)
class ExecutionKillReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    runtime_version: int
    fencing_token: int
    claim_fencing_token: int
    status: str
    kill_requested: bool
    reason: str | None
    request_sequence: int | None
    risk_version: int | None
    risk_receipt_id: str | None
    risk_state: str | None
    journal_stage: str | None
    start_risk_version: int | None
    start_risk_receipt_id: str | None
    schema_version: str = PHASE78_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash when present")
        if self.runtime_version < 0:
            raise ValueError("runtime_version must be non-negative")
        if self.fencing_token <= 0 or self.claim_fencing_token <= 0:
            raise ValueError("runtime/claim fencing tokens must be positive")
        if self.request_sequence is not None and self.request_sequence <= 0:
            raise ValueError("request_sequence must be positive")
        if self.risk_version is not None and self.risk_version <= 0:
            raise ValueError("risk_version must be positive when present")
        if self.start_risk_version is not None and self.start_risk_version <= 0:
            raise ValueError("start_risk_version must be positive when present")
        for label, value in (
            ("risk_receipt_id", self.risk_receipt_id),
            ("start_risk_receipt_id", self.start_risk_receipt_id),
        ):
            if value is not None and len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if self.risk_state is not None and self.risk_state not in (
            "ACTIVE",
            "REDUCING",
            "HALTED",
        ):
            raise ValueError("invalid risk_state")
        if self.kill_requested and not self.reason:
            raise ValueError("kill request requires a reason")
        if not self.shadow_only or self.live_execution:
            raise ValueError("kill receipt must remain shadow-only")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ExecutionStartError(f"{label} returned non-object payload")
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
        raise ExecutionStartError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionStartError(f"{label} must be integer") from exc


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ExecutionStartError(f"{label} must be boolean")
    return value


def _optional_hash(value: object) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64:
        raise ExecutionStartError("database returned invalid content hash")
    return result


class ExecutionStartKillSwitchStore:
    """Phase78 transport adapter for the durable execution-start boundary."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def start(
        self,
        lease,
        claim: ExecutionClaimReceipt,
        *,
        worker_token: str,
    ) -> ExecutionStartReceipt:
        if not claim.claimed or claim.claim_fencing_token <= 0:
            raise ExecutionStartError("only an owned claim can cross execution start")
        if claim.runtime_id != lease.runtime_id:
            raise ExecutionStartError("claim runtime does not match lease")
        if claim.fencing_token != lease.fencing_token:
            raise ExecutionStartError("claim runtime fence does not match lease")
        if not worker_token.strip():
            raise ValueError("worker_token is required")

        row = _mapping(
            self._rpc(
                "brian_start_shadow_execution_claim",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": claim.cycle_id,
                    "p_worker_token": worker_token,
                    "p_claim_fencing_token": claim.claim_fencing_token,
                },
            ),
            "brian_start_shadow_execution_claim",
        )

        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise ExecutionStartError("start runtime_id mismatch")
        if str(row.get("cycle_id", "")) != claim.cycle_id:
            raise ExecutionStartError("start cycle_id mismatch")

        started = _boolean(row.get("started"), "started")
        cancelled = _boolean(row.get("cancelled"), "cancelled")
        duplicate = bool(row.get("duplicate", False))
        status = str(row.get("status", ""))

        if status in {"STARTED", "STARTED_RESUME", "STARTED_ALREADY"} and not started:
            raise ExecutionStartError(f"{status} must return started=true")
        if status == "STARTED_ALREADY" and not duplicate:
            raise ExecutionStartError("STARTED_ALREADY must be marked duplicate")
        if status == "CANCELLED_BEFORE_START" and not cancelled:
            raise ExecutionStartError("CANCELLED_BEFORE_START must be cancelled")
        if status in {
            "LEASE_LOST",
            "DISPATCH_MISSING",
            "CLAIM_LOST",
            "RISK_STATE_UNAVAILABLE",
        } and (started or cancelled):
            raise ExecutionStartError(f"{status} cannot be started/cancelled")

        return ExecutionStartReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=str(row.get("dispatch_id", "")),
            runtime_version=_integer(
                row.get("runtime_version"),
                "runtime_version",
                default=lease.version,
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
            started=started,
            cancelled=cancelled,
            duplicate=duplicate,
            risk_version=(
                None
                if row.get("risk_version") is None
                else _integer(row.get("risk_version"), "risk_version")
            ),
            risk_receipt_id=_optional_hash(row.get("risk_receipt_id")),
            risk_state=(
                None if row.get("risk_state") is None else str(row.get("risk_state"))
            ),
            journal_stage=(
                None
                if row.get("journal_stage") is None
                else str(row.get("journal_stage"))
            ),
            resume_only=bool(row.get("resume_only", False)),
            cancel_reason=(
                None
                if row.get("cancel_reason") is None
                else str(row.get("cancel_reason"))
            ),
        )

    def check_kill(
        self,
        lease,
        claim: ExecutionClaimReceipt,
        *,
        worker_token: str,
    ) -> ExecutionKillReceipt:
        if not claim.claimed or claim.claim_fencing_token <= 0:
            raise ExecutionStartError("only an owned claim can check kill state")
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
            raise ExecutionStartError("kill-check runtime_id mismatch")
        if str(row.get("cycle_id", "")) != claim.cycle_id:
            raise ExecutionStartError("kill-check cycle_id mismatch")

        requested = _boolean(row.get("kill_requested"), "kill_requested")
        status = str(row.get("status", ""))
        reason = None if row.get("reason") is None else str(row.get("reason"))

        if status == "KILL_REQUESTED" and not requested:
            raise ExecutionStartError("KILL_REQUESTED must set kill_requested=true")
        if requested and status not in {"KILL_REQUESTED", "RISK_STATE_UNAVAILABLE"}:
            raise ExecutionStartError(
                "kill_requested=true returned unsupported status"
            )
        if not requested and status == "KILL_REQUESTED":
            raise ExecutionStartError("kill status flags are inconsistent")

        return ExecutionKillReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=str(row.get("dispatch_id", "")),
            runtime_version=_integer(
                row.get("runtime_version"),
                "runtime_version",
                default=lease.version,
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
            kill_requested=requested,
            reason=reason,
            request_sequence=(
                None
                if row.get("request_sequence") is None
                else _integer(row.get("request_sequence"), "request_sequence")
            ),
            risk_version=(
                None
                if row.get("risk_version") is None
                else _integer(row.get("risk_version"), "risk_version")
            ),
            risk_receipt_id=_optional_hash(row.get("risk_receipt_id")),
            risk_state=(
                None if row.get("risk_state") is None else str(row.get("risk_state"))
            ),
            journal_stage=(
                None
                if row.get("journal_stage") is None
                else str(row.get("journal_stage"))
            ),
            start_risk_version=(
                None
                if row.get("start_risk_version") is None
                else _integer(
                    row.get("start_risk_version"),
                    "start_risk_version",
                )
            ),
            start_risk_receipt_id=_optional_hash(
                row.get("start_risk_receipt_id")
            ),
        )


@dataclass(frozen=True, slots=True)
class StartedExecutionStep:
    dispatch: object
    claim: ExecutionClaimReceipt
    start: ExecutionStartReceipt | None
    kill: ExecutionKillReceipt | None
    outcome: str
    completion: ClaimCompletionReceipt | None
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE78_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


class PersistedStartedRuntimeSupervisor:
    """Adds the durable STARTED point-of-no-return to Phase77 execution."""

    def __init__(
        self,
        *,
        claimed_supervisor: PersistedClaimedRuntimeSupervisor,
        starts: ExecutionStartKillSwitchStore,
    ) -> None:
        self.claimed_supervisor = claimed_supervisor
        self.starts = starts

    def process_governed_cycle(
        self,
        governed,
        *,
        worker_token: str,
        claim_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> StartedExecutionStep:
        phase77 = self.claimed_supervisor
        phase76 = phase77.dispatched_supervisor
        phase75 = phase76.governed_supervisor
        supervisor = phase75.runtime_supervisor

        authorization, dispatch, claim, terminal = phase77.authorize_submit_and_claim(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
        )
        del authorization

        if terminal is not None:
            checkpoint = supervisor.runtime.checkpoint()
            return StartedExecutionStep(
                dispatch=dispatch,
                claim=claim,
                start=None,
                kill=None,
                outcome=terminal,
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        start = self.starts.start(
            supervisor.lease,
            claim,
            worker_token=worker_token,
        )

        if start.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost at execution-start boundary"
            )
        if start.status in {
            "DISPATCH_MISSING",
            "RISK_STATE_UNAVAILABLE",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"execution start rejected with {start.status}; reload required"
            )
        if start.status == "CLAIM_LOST":
            raise ExecutionClaimError(
                "execution claim was lost before start; acquire a new claim"
            )

        if start.cancelled:
            if supervisor.runtime.journal.latest_stage(claim.cycle_id) == "CYCLE_CREATED":
                phase75.abort_authorized_cycle(
                    cycle_id=claim.cycle_id,
                    reason=f"phase78:{start.cancel_reason or 'risk_cancel'}",
                )
            checkpoint = supervisor.runtime.checkpoint()
            return StartedExecutionStep(
                dispatch=dispatch,
                claim=claim,
                start=start,
                kill=None,
                outcome="CANCELLED_BEFORE_START",
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if not start.started:
            raise ExecutionStartError(
                f"unexpected non-started status {start.status}"
            )
        if start.fencing_token != supervisor.lease.fencing_token:
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "execution-start runtime fence disagrees with supervisor"
            )
        if start.claim_fencing_token != claim.claim_fencing_token:
            raise ExecutionClaimError(
                "execution-start claim fence disagrees with owned claim"
            )

        advanced, outcome, _ = phase77.advance_claimed(
            claim,
            worker_token=worker_token,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
            complete_on_commit=False,
        )

        kill = self.starts.check_kill(
            supervisor.lease,
            claim,
            worker_token=worker_token,
        )
        if kill.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost during post-start kill check"
            )
        if kill.status in {"DISPATCH_MISSING", "START_MISSING"}:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"post-start kill check returned {kill.status}"
            )
        if kill.status == "CLAIM_LOST":
            raise ExecutionClaimError(
                "execution claim expired before post-start kill check"
            )

        completion = None
        if outcome == "COMMITTED":
            completion = phase77.claims.complete(
                supervisor.lease,
                claim,
                worker_token=worker_token,
            )
            if not completion.completed:
                outcome = f"COMMITTED_{completion.status}"
        elif kill.kill_requested:
            outcome = f"KILL_REQUESTED_{outcome}"

        checkpoint = supervisor.runtime.checkpoint()
        return StartedExecutionStep(
            dispatch=dispatch,
            claim=claim,
            start=start,
            kill=kill,
            outcome=outcome,
            completion=completion,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=checkpoint.checkpoint_id,
        )
