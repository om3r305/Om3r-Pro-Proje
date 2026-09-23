from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

from .evidence_ledger import content_hash
from .phase50_execution_reconciliation import ReconciliationBatchReport
from .phase57_shadow_execution_cycle import ShadowExecutionCycle
from .phase60_shadow_state_ledger import (
    ShadowAccountState,
    ShadowLedgerReceipt,
    ShadowStateConflictError,
    ShadowStateLedger,
)
from .phase61_stateful_paper_venue import (
    PaperCycleReceipt,
    PaperVenue,
    PaperVenueConflictError,
)
from .phase63_crash_recovery import (
    ShadowRuntimeCheckpoint,
    create_runtime_checkpoint,
)
from .phase64_local_execution_projector import (
    LocalExecutionProjector,
    LocalExecutionProjectionError,
    ProjectionReceipt,
)
from .phase65_event_sourced_local_recovery import (
    RecoveredShadowPaperRuntime,
    restore_runtime_with_local_projection,
)

PHASE66_SCHEMA_VERSION = "brian.phase66-runtime-coordinator.v1"
CycleStatus = Literal[
    "COMMITTED",
    "RECONCILIATION_BLOCKED",
    "MARKS_REQUIRED",
]
ResumeStatus = Literal[
    "NO_PENDING_CYCLE",
    "COMMITTED",
    "RECONCILIATION_BLOCKED",
    "MARKS_REQUIRED",
]


class RuntimeCoordinatorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeCycleReceipt:
    cycle_id: str
    status: CycleStatus
    before_state_id: str
    after_state_id: str
    paper_receipt_id: str
    projection_hash: str
    reconciliation_hash: str
    tracked_assets: tuple[str, ...]
    missing_mark_assets: tuple[str, ...]
    commit_transition_id: str | None
    pending_cycle_id: str | None
    receipt_id: str
    schema_version: str = PHASE66_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PendingCycleResumeReceipt:
    cycle_id: str | None
    status: ResumeStatus
    before_state_id: str
    after_state_id: str
    reconciliation_hash: str | None
    tracked_assets: tuple[str, ...]
    missing_mark_assets: tuple[str, ...]
    commit_transition_id: str | None
    pending_cycle_id: str | None
    receipt_id: str
    schema_version: str = PHASE66_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _receipt_hash(payload: Mapping[str, object]) -> str:
    return content_hash(dict(payload))


class ShadowPaperRuntimeCoordinator:
    """Recovery-aware orchestration for the Phase 57 -> 66 paper path.

    This coordinator deliberately owns no exchange transport. It sequences the
    existing hard boundaries so that one cycle cannot overlap another and a
    paper fill cannot become the authoritative account state until an
    independently projected local cache reconciles through Phase 50.
    """

    def __init__(
        self,
        ledger: ShadowStateLedger,
        venue: PaperVenue,
        projector: LocalExecutionProjector,
    ) -> None:
        try:
            ledger.verify_integrity()
        except ShadowStateConflictError as exc:
            raise RuntimeCoordinatorError(str(exc)) from exc

        account_ids = {
            ledger.head_state.account_id,
            venue.config.account_id,
            projector.account_id,
        }
        if len(account_ids) != 1:
            raise RuntimeCoordinatorError(
                "ledger, paper venue and local projector account ids must match"
            )

        self.ledger = ledger
        self.venue = venue
        self.projector = projector

    @classmethod
    def restore(
        cls,
        checkpoint: ShadowRuntimeCheckpoint,
    ) -> "ShadowPaperRuntimeCoordinator":
        recovered = restore_runtime_with_local_projection(checkpoint)
        return cls(
            recovered.ledger,
            recovered.venue,
            recovered.projector,
        )

    @classmethod
    def from_recovered(
        cls,
        recovered: RecoveredShadowPaperRuntime,
    ) -> "ShadowPaperRuntimeCoordinator":
        if not recovered.replay.complete:
            raise RuntimeCoordinatorError(
                "runtime coordinator requires complete local event replay"
            )
        if not recovered.reconciliation.ready:
            raise RuntimeCoordinatorError(
                "recovered local execution state does not reconcile with paper venue"
            )
        return cls(recovered.ledger, recovered.venue, recovered.projector)

    def checkpoint(self) -> ShadowRuntimeCheckpoint:
        return create_runtime_checkpoint(self.ledger, self.venue)

    def _tracked_assets(
        self,
        extra_assets: Sequence[str] = (),
    ) -> tuple[str, ...]:
        assets = (
            set(self.ledger.head_state.covered_assets)
            | set(self.venue.positions)
            | set(self.projector.positions)
            | {str(asset) for asset in extra_assets if str(asset)}
        )
        if not assets:
            raise RuntimeCoordinatorError("runtime has no tracked assets")
        return tuple(sorted(assets))

    def _reconcile(
        self,
        *,
        extra_assets: Sequence[str] = (),
    ) -> tuple[tuple[str, ...], ReconciliationBatchReport]:
        tracked = self._tracked_assets(extra_assets)
        report = self.venue.reconcile_against_local(
            self.projector.local_positions(tracked_assets=tracked),
            tracked_assets=tracked,
        )
        return tracked, report

    def reconcile_current(
        self,
        *,
        extra_assets: Sequence[str] = (),
    ) -> tuple[tuple[str, ...], ReconciliationBatchReport]:
        """Expose the independent Phase 50 check without committing ledger state."""
        return self._reconcile(extra_assets=extra_assets)

    def _missing_marks(
        self,
        marks: Mapping[str, float],
    ) -> tuple[str, ...]:
        missing: list[str] = []
        for asset, position in self.venue.positions.items():
            if abs(position.quantity) <= 1e-12:
                continue
            value = marks.get(asset)
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                missing.append(asset)
        return tuple(sorted(missing))

    def _build_state_and_commit(
        self,
        *,
        cycle_id: str,
        reconciliation: ReconciliationBatchReport,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
        expected_state_id: str,
    ) -> tuple[ShadowAccountState | None, ShadowLedgerReceipt | None, tuple[str, ...], str]:
        missing_marks = self._missing_marks(marks)
        if missing_marks:
            return None, None, missing_marks, "MARKS_REQUIRED"

        try:
            state = self.venue.build_reconciled_state(
                reconciliation,
                marks=marks,
                observed_at=observed_at,
                source_ref=source_ref,
            )
            commit = self.ledger.commit_reconciled_state(
                cycle_id,
                reconciliation,
                state,
                expected_state_id=expected_state_id,
            )
        except (PaperVenueConflictError, ShadowStateConflictError) as exc:
            raise RuntimeCoordinatorError(str(exc)) from exc
        return state, commit, (), "COMMITTED"

    def process_cycle(
        self,
        cycle: ShadowExecutionCycle,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> RuntimeCycleReceipt:
        """Run one cycle through paper execution, local projection and commit.

        If reconciliation or mark-to-market data is incomplete, the cycle stays
        pending in Phase 60 and blocks any different cycle until it is resumed.
        Re-running the *same* pending cycle is safe because Phase 60, 61 and 64
        are all content-addressed/idempotent.
        """
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be finite")
        if not source_ref.strip():
            raise ValueError("source_ref is required")
        if self.ledger.pending_cycle_id not in (None, cycle.cycle_id):
            raise RuntimeCoordinatorError(
                f"pending cycle {self.ledger.pending_cycle_id} must resolve before {cycle.cycle_id}"
            )

        before_state_id = self.ledger.head_state.state_id
        try:
            self.ledger.append_cycle(
                cycle,
                expected_state_id=before_state_id,
            )
            paper = self.venue.apply_cycle(cycle)
            projection = self.projector.process_cycle(
                paper,
                {fill.fill_id: fill for fill in self.venue.fills},
            )
        except (
            ShadowStateConflictError,
            PaperVenueConflictError,
            LocalExecutionProjectionError,
        ) as exc:
            raise RuntimeCoordinatorError(str(exc)) from exc

        cycle_assets = tuple(item.asset_id for item in cycle.items)
        tracked, reconciliation = self._reconcile(extra_assets=cycle_assets)
        reconciliation_hash = content_hash(reconciliation.to_dict())

        if not reconciliation.ready or not all(
            value for _, value in reconciliation.checks
        ):
            status: CycleStatus = "RECONCILIATION_BLOCKED"
            missing_marks: tuple[str, ...] = ()
            commit_id = None
            after_state_id = before_state_id
        else:
            state, commit, missing_marks, commit_status = self._build_state_and_commit(
                cycle_id=cycle.cycle_id,
                reconciliation=reconciliation,
                marks=marks,
                observed_at=observed_at,
                source_ref=source_ref,
                expected_state_id=before_state_id,
            )
            status = commit_status  # type: ignore[assignment]
            commit_id = None if commit is None else commit.transition_id
            after_state_id = (
                before_state_id if state is None else state.state_id
            )

        payload = {
            "schema_version": PHASE66_SCHEMA_VERSION,
            "cycle_id": cycle.cycle_id,
            "status": status,
            "before_state_id": before_state_id,
            "after_state_id": after_state_id,
            "paper_receipt_id": paper.receipt_id,
            "projection_hash": projection.projection_hash,
            "reconciliation_hash": reconciliation_hash,
            "tracked_assets": tracked,
            "missing_mark_assets": missing_marks,
            "commit_transition_id": commit_id,
            "pending_cycle_id": self.ledger.pending_cycle_id,
        }
        return RuntimeCycleReceipt(
            cycle_id=cycle.cycle_id,
            status=status,
            before_state_id=before_state_id,
            after_state_id=after_state_id,
            paper_receipt_id=paper.receipt_id,
            projection_hash=projection.projection_hash,
            reconciliation_hash=reconciliation_hash,
            tracked_assets=tracked,
            missing_mark_assets=missing_marks,
            commit_transition_id=commit_id,
            pending_cycle_id=self.ledger.pending_cycle_id,
            receipt_id=_receipt_hash(payload),
        )

    def resume_pending(
        self,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> PendingCycleResumeReceipt:
        """Finish a pending cycle after recovery without needing the original cycle payload."""
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be finite")
        if not source_ref.strip():
            raise ValueError("source_ref is required")

        cycle_id = self.ledger.pending_cycle_id
        before_state_id = self.ledger.head_state.state_id
        if cycle_id is None:
            payload = {
                "schema_version": PHASE66_SCHEMA_VERSION,
                "cycle_id": None,
                "status": "NO_PENDING_CYCLE",
                "before_state_id": before_state_id,
                "after_state_id": before_state_id,
                "reconciliation_hash": None,
                "tracked_assets": self._tracked_assets(),
                "missing_mark_assets": (),
                "commit_transition_id": None,
                "pending_cycle_id": None,
            }
            return PendingCycleResumeReceipt(
                cycle_id=None,
                status="NO_PENDING_CYCLE",
                before_state_id=before_state_id,
                after_state_id=before_state_id,
                reconciliation_hash=None,
                tracked_assets=tuple(payload["tracked_assets"]),  # type: ignore[arg-type]
                missing_mark_assets=(),
                commit_transition_id=None,
                pending_cycle_id=None,
                receipt_id=_receipt_hash(payload),
            )

        tracked, reconciliation = self._reconcile()
        reconciliation_hash = content_hash(reconciliation.to_dict())

        if not reconciliation.ready or not all(
            value for _, value in reconciliation.checks
        ):
            status: ResumeStatus = "RECONCILIATION_BLOCKED"
            after_state_id = before_state_id
            missing_marks: tuple[str, ...] = ()
            commit_id = None
        else:
            state, commit, missing_marks, commit_status = self._build_state_and_commit(
                cycle_id=cycle_id,
                reconciliation=reconciliation,
                marks=marks,
                observed_at=observed_at,
                source_ref=source_ref,
                expected_state_id=before_state_id,
            )
            status = commit_status  # type: ignore[assignment]
            after_state_id = before_state_id if state is None else state.state_id
            commit_id = None if commit is None else commit.transition_id

        payload = {
            "schema_version": PHASE66_SCHEMA_VERSION,
            "cycle_id": cycle_id,
            "status": status,
            "before_state_id": before_state_id,
            "after_state_id": after_state_id,
            "reconciliation_hash": reconciliation_hash,
            "tracked_assets": tracked,
            "missing_mark_assets": missing_marks,
            "commit_transition_id": commit_id,
            "pending_cycle_id": self.ledger.pending_cycle_id,
        }
        return PendingCycleResumeReceipt(
            cycle_id=cycle_id,
            status=status,
            before_state_id=before_state_id,
            after_state_id=after_state_id,
            reconciliation_hash=reconciliation_hash,
            tracked_assets=tracked,
            missing_mark_assets=missing_marks,
            commit_transition_id=commit_id,
            pending_cycle_id=self.ledger.pending_cycle_id,
            receipt_id=_receipt_hash(payload),
        )
