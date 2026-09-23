from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

PHASE86_SCHEMA_VERSION = "brian.phase86-recovery-admission-interlock.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class RecoveryAdmissionInterlockError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryAdmissionState:
    runtime_id: str
    status: str
    blocked: bool
    original_cycle_id: str | None = None
    cancel_risk_receipt_id: str | None = None
    reason: str | None = None
    requested_at: object | None = None
    schema_version: str = PHASE86_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.blocked:
            if self.status != "RECOVERY_BARRIER":
                raise ValueError("blocked admission must use RECOVERY_BARRIER")
            if self.original_cycle_id is None or len(self.original_cycle_id) != 64:
                raise ValueError("blocked admission requires original cycle id")
            if (
                self.cancel_risk_receipt_id is None
                or len(self.cancel_risk_receipt_id) != 64
            ):
                raise ValueError("blocked admission requires cancel risk receipt")
            if not self.reason:
                raise ValueError("blocked admission requires reason")
        elif self.status != "OPEN":
            raise ValueError("open admission must use OPEN")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase86 admission must remain shadow-only")


class RecoveryAdmissionInterlockStore:
    """Read the DB-authoritative unresolved-recovery admission barrier."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def read(self, *, runtime_id: str) -> RecoveryAdmissionState:
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        raw = self._rpc(
            "brian_read_shadow_recovery_admission",
            {"p_runtime_id": runtime_id},
        )
        if not isinstance(raw, Mapping):
            raise RecoveryAdmissionInterlockError(
                "brian_read_shadow_recovery_admission returned non-object payload"
            )
        row = {str(key): value for key, value in raw.items()}
        if str(row.get("runtime_id", "")) != runtime_id:
            raise RecoveryAdmissionInterlockError("admission runtime_id mismatch")
        blocked = row.get("blocked")
        if not isinstance(blocked, bool):
            raise RecoveryAdmissionInterlockError("admission blocked must be boolean")

        original = None if row.get("original_cycle_id") is None else str(row["original_cycle_id"])
        cancel = (
            None
            if row.get("cancel_risk_receipt_id") is None
            else str(row["cancel_risk_receipt_id"])
        )
        return RecoveryAdmissionState(
            runtime_id=runtime_id,
            status=str(row.get("status", "")),
            blocked=blocked,
            original_cycle_id=original,
            cancel_risk_receipt_id=cancel,
            reason=None if row.get("reason") is None else str(row["reason"]),
            requested_at=row.get("requested_at"),
        )
