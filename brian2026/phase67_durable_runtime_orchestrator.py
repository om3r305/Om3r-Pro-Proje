from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping
import math

from .evidence_ledger import content_hash
from .phase57_shadow_execution_cycle import ShadowExecutionCycle
from .phase60_shadow_state_ledger import ShadowStateConflictError
from .phase61_stateful_paper_venue import PaperVenueConflictError
from .phase63_crash_recovery import ShadowRuntimeCheckpoint
from .phase64_local_execution_projector import LocalExecutionProjectionError
from .phase66_durable_cycle_journal import (
    CycleJournalError,
    DurableCycleJournal,
    JournalAppendReceipt,
    JournalStage,
    restore_cycle_journal,
)
from .phase66_runtime_coordinator import (
    PendingCycleResumeReceipt,
    RuntimeCoordinatorError,
    ShadowPaperRuntimeCoordinator,
)

PHASE67_SCHEMA_VERSION = "brian.phase67-durable-runtime-orchestrator.v1"
DurableStatus = Literal[
    "NO_PENDING_CYCLE",
    "COMMITTED",
    "RECONCILIATION_BLOCKED",
    "MARKS_REQUIRED",
]

_TERMINAL_STAGES: frozenset[JournalStage] = frozenset({"COMMITTED", "ABORTED"})


class DurableRuntimeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DurableRuntimeCheckpoint:
    runtime_checkpoint: ShadowRuntimeCheckpoint
    journal_manifest: Mapping[str, object]
    schema_version: str = PHASE67_SCHEMA_VERSION
    live_execution: bool = False
    checkpoint_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.live_execution:
            raise ValueError("durable runtime checkpoint cannot contain live execution")
        object.__setattr__(self, "checkpoint_id", content_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_checkpoint": self.runtime_checkpoint.to_dict(),
            "journal_manifest": dict(self.journal_manifest),
            "live_execution": self.live_execution,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["checkpoint_id"] = self.checkpoint_id
        return payload


@dataclass(frozen=True, slots=True)
class DurableRuntimeReceipt:
    cycle_id: str | None
    status: DurableStatus
    journal_stage: JournalStage | None
    before_state_id: str
    after_state_id: str
    pending_cycle_id: str | None
    journal_entry_id: str | None
    missing_mark_assets: tuple[str, ...]
    receipt_id: str
    schema_version: str = PHASE67_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DurableShadowPaperRuntime:
    """Write-ahead, crash-resumable orchestration for the full paper runtime.

    The journal is authoritative for *what cycle should be processed*, while the
    Phase 60 ledger is authoritative for committed account state. Recovery may
    advance journal metadata only when an independently restored downstream
    artifact proves that side effect already happened.
    """

    def __init__(
        self,
        coordinator: ShadowPaperRuntimeCoordinator,
        journal: DurableCycleJournal | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.journal = journal or DurableCycleJournal()
        try:
            self.journal.verify_integrity()
        except CycleJournalError as exc:
            raise DurableRuntimeError(str(exc)) from exc
        self._validate_single_active_cycle()

    @property
    def ledger(self):
        return self.coordinator.ledger

    @property
    def venue(self):
        return self.coordinator.venue

    @property
    def projector(self):
        return self.coordinator.projector

    def _active_cycle_ids(self) -> tuple[str, ...]:
        return tuple(
            cycle_id
            for cycle_id in self.journal.cycle_ids
            if self.journal.latest_stage(cycle_id) not in _TERMINAL_STAGES
        )

    def _validate_single_active_cycle(self) -> None:
        active = self._active_cycle_ids()
        if len(active) > 1:
            raise DurableRuntimeError(
                f"multiple non-terminal journal cycles are forbidden: {active}"
            )
        if self.ledger.pending_cycle_id is not None:
            if not active:
                raise DurableRuntimeError(
                    "Phase 60 has a pending cycle which is missing from durable journal"
                )
            if active[0] != self.ledger.pending_cycle_id:
                raise DurableRuntimeError(
                    "durable journal active cycle does not match Phase 60 pending cycle"
                )

    def _committed_state_id(self, cycle_id: str) -> str | None:
        for transition in self.ledger.transitions:
            if transition.kind == "RECONCILED_COMMIT" and transition.cycle_id == cycle_id:
                return transition.after_state_id
        return None

    def _cycle_assets(self, cycle_id: str) -> tuple[str, ...]:
        cycle = self.journal.cycle(cycle_id)
        return tuple(sorted({item.asset_id for item in cycle.items}))

    def _latest_entry_id(self, cycle_id: str) -> str | None:
        rows = [
            entry
            for entry in self.journal.entries
            if entry.cycle_id == cycle_id
        ]
        return None if not rows else rows[-1].entry_id

    def _make_receipt(
        self,
        *,
        cycle_id: str | None,
        status: DurableStatus,
        before_state_id: str,
        after_state_id: str,
        missing_mark_assets: tuple[str, ...] = (),
    ) -> DurableRuntimeReceipt:
        stage = None if cycle_id is None else self.journal.latest_stage(cycle_id)
        entry_id = None if cycle_id is None else self._latest_entry_id(cycle_id)
        payload = {
            "schema_version": PHASE67_SCHEMA_VERSION,
            "cycle_id": cycle_id,
            "status": status,
            "journal_stage": stage,
            "before_state_id": before_state_id,
            "after_state_id": after_state_id,
            "pending_cycle_id": self.ledger.pending_cycle_id,
            "journal_entry_id": entry_id,
            "missing_mark_assets": missing_mark_assets,
        }
        return DurableRuntimeReceipt(
            cycle_id=cycle_id,
            status=status,
            journal_stage=stage,
            before_state_id=before_state_id,
            after_state_id=after_state_id,
            pending_cycle_id=self.ledger.pending_cycle_id,
            journal_entry_id=entry_id,
            missing_mark_assets=missing_mark_assets,
            receipt_id=content_hash(payload),
        )

    def journal_cycle(self, cycle: ShadowExecutionCycle) -> JournalAppendReceipt:
        """Persist the full cycle body before any account/execution side effect."""
        active = self._active_cycle_ids()
        if active and active != (cycle.cycle_id,):
            raise DurableRuntimeError(
                f"journal cycle {active[0]} must resolve before {cycle.cycle_id}"
            )
        if self.ledger.pending_cycle_id not in (None, cycle.cycle_id):
            raise DurableRuntimeError(
                f"Phase 60 pending cycle {self.ledger.pending_cycle_id} must resolve first"
            )
        try:
            return self.journal.record_cycle(cycle)
        except CycleJournalError as exc:
            raise DurableRuntimeError(str(exc)) from exc

    def checkpoint(self) -> DurableRuntimeCheckpoint:
        try:
            self.journal.verify_integrity()
        except CycleJournalError as exc:
            raise DurableRuntimeError(str(exc)) from exc
        return DurableRuntimeCheckpoint(
            runtime_checkpoint=self.coordinator.checkpoint(),
            journal_manifest=self.journal.manifest(),
        )

    @classmethod
    def restore(
        cls,
        checkpoint: DurableRuntimeCheckpoint,
    ) -> "DurableShadowPaperRuntime":
        if content_hash(checkpoint.identity_payload()) != checkpoint.checkpoint_id:
            raise DurableRuntimeError("durable runtime checkpoint content hash mismatch")
        try:
            coordinator = ShadowPaperRuntimeCoordinator.restore(
                checkpoint.runtime_checkpoint
            )
            journal = restore_cycle_journal(checkpoint.journal_manifest)
        except (
            RuntimeCoordinatorError,
            CycleJournalError,
        ) as exc:
            raise DurableRuntimeError(str(exc)) from exc
        runtime = cls(coordinator, journal)
        runtime._synchronize_recovered_artifacts()
        runtime._validate_single_active_cycle()
        return runtime

    def _synchronize_recovered_artifacts(self) -> None:
        """Move journal metadata forward only when restored artifacts prove it."""
        committed = {
            cycle_id: state_id
            for cycle_id in self.journal.cycle_ids
            if (state_id := self._committed_state_id(cycle_id)) is not None
        }

        for cycle_id in self.journal.cycle_ids:
            stage = self.journal.latest_stage(cycle_id)
            if stage in _TERMINAL_STAGES:
                continue

            if stage == "CYCLE_CREATED":
                try:
                    paper = self.venue.cycle_receipt(cycle_id)
                except KeyError:
                    paper = None
                if paper is not None:
                    try:
                        self.journal.mark_paper_applied(cycle_id, paper)
                    except CycleJournalError as exc:
                        raise DurableRuntimeError(str(exc)) from exc
                    stage = self.journal.latest_stage(cycle_id)

            if stage == "PAPER_APPLIED":
                try:
                    projection = self.projector.projection_receipt(cycle_id)
                except KeyError:
                    projection = None
                if projection is not None:
                    try:
                        self.journal.mark_local_projected(cycle_id, projection)
                    except CycleJournalError as exc:
                        raise DurableRuntimeError(str(exc)) from exc
                    stage = self.journal.latest_stage(cycle_id)

            if stage in {"LOCAL_PROJECTED", "RECONCILIATION_REQUIRED"}:
                if self.ledger.pending_cycle_id == cycle_id or cycle_id in committed:
                    tracked, report = self.coordinator.reconcile_current(
                        extra_assets=self._cycle_assets(cycle_id)
                    )
                    del tracked
                    try:
                        self.journal.mark_reconciliation(cycle_id, report)
                    except CycleJournalError as exc:
                        raise DurableRuntimeError(str(exc)) from exc
                    stage = self.journal.latest_stage(cycle_id)

            if stage == "RECONCILED" and cycle_id in committed:
                try:
                    self.journal.mark_committed(
                        cycle_id,
                        state_id=committed[cycle_id],
                    )
                except CycleJournalError as exc:
                    raise DurableRuntimeError(str(exc)) from exc
                stage = self.journal.latest_stage(cycle_id)

            if (
                stage not in {"CYCLE_CREATED", "COMMITTED", "ABORTED"}
                and self.ledger.pending_cycle_id != cycle_id
                and cycle_id not in committed
            ):
                raise DurableRuntimeError(
                    f"journal stage {stage} for {cycle_id} has side effects but no "
                    "matching Phase 60 pending/committed state"
                )

    def _ensure_ledger_pending(self, cycle_id: str) -> None:
        cycle = self.journal.cycle(cycle_id)
        if self.ledger.pending_cycle_id is None:
            try:
                self.ledger.append_cycle(
                    cycle,
                    expected_state_id=self.ledger.head_state.state_id,
                )
            except ShadowStateConflictError as exc:
                raise DurableRuntimeError(str(exc)) from exc
        elif self.ledger.pending_cycle_id == cycle_id:
            # Re-append validates the same content-addressed cycle body.
            try:
                self.ledger.append_cycle(
                    cycle,
                    expected_state_id=self.ledger.head_state.state_id,
                )
            except ShadowStateConflictError as exc:
                raise DurableRuntimeError(str(exc)) from exc
        else:
            raise DurableRuntimeError(
                f"Phase 60 pending cycle {self.ledger.pending_cycle_id} conflicts with {cycle_id}"
            )

    def _active_cycle_id(self) -> str | None:
        active = self._active_cycle_ids()
        if not active:
            return None
        if len(active) != 1:
            raise DurableRuntimeError("runtime contains multiple active journal cycles")
        return active[0]

    def advance_pending(
        self,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> DurableRuntimeReceipt:
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be finite")
        if not source_ref.strip():
            raise ValueError("source_ref is required")

        cycle_id = self._active_cycle_id()
        before_state_id = self.ledger.head_state.state_id
        if cycle_id is None:
            if self.ledger.pending_cycle_id is not None:
                raise DurableRuntimeError(
                    "Phase 60 has pending state without an active journal cycle"
                )
            return self._make_receipt(
                cycle_id=None,
                status="NO_PENDING_CYCLE",
                before_state_id=before_state_id,
                after_state_id=before_state_id,
            )

        stage = self.journal.latest_stage(cycle_id)
        assert stage is not None
        cycle = self.journal.cycle(cycle_id)

        if stage == "CYCLE_CREATED":
            self._ensure_ledger_pending(cycle_id)
            try:
                paper = self.venue.apply_cycle(cycle)
                self.journal.mark_paper_applied(cycle_id, paper)
            except (PaperVenueConflictError, CycleJournalError) as exc:
                raise DurableRuntimeError(str(exc)) from exc
            stage = self.journal.latest_stage(cycle_id)

        if stage == "PAPER_APPLIED":
            try:
                paper = self.venue.cycle_receipt(cycle_id)
                projection = self.projector.process_cycle(
                    paper,
                    {fill.fill_id: fill for fill in self.venue.fills},
                )
                self.journal.mark_local_projected(cycle_id, projection)
            except (
                KeyError,
                LocalExecutionProjectionError,
                CycleJournalError,
            ) as exc:
                raise DurableRuntimeError(str(exc)) from exc
            stage = self.journal.latest_stage(cycle_id)

        if stage in {"LOCAL_PROJECTED", "RECONCILIATION_REQUIRED"}:
            tracked, report = self.coordinator.reconcile_current(
                extra_assets=self._cycle_assets(cycle_id)
            )
            del tracked
            try:
                journal_receipt = self.journal.mark_reconciliation(
                    cycle_id,
                    report,
                )
            except CycleJournalError as exc:
                raise DurableRuntimeError(str(exc)) from exc
            stage = journal_receipt.stage
            if stage == "RECONCILIATION_REQUIRED":
                return self._make_receipt(
                    cycle_id=cycle_id,
                    status="RECONCILIATION_BLOCKED",
                    before_state_id=before_state_id,
                    after_state_id=before_state_id,
                )

        if stage == "RECONCILED":
            # Re-check the same independent state before authoritative commit.
            tracked, report = self.coordinator.reconcile_current(
                extra_assets=self._cycle_assets(cycle_id)
            )
            del tracked
            if not report.ready or not all(value for _, value in report.checks):
                raise DurableRuntimeError(
                    "reconciliation regressed after durable RECONCILED evidence"
                )
            try:
                self.journal.mark_reconciliation(cycle_id, report)
            except CycleJournalError as exc:
                raise DurableRuntimeError(str(exc)) from exc

            try:
                resumed: PendingCycleResumeReceipt = self.coordinator.resume_pending(
                    marks=marks,
                    observed_at=observed_at,
                    source_ref=source_ref,
                )
            except RuntimeCoordinatorError as exc:
                raise DurableRuntimeError(str(exc)) from exc

            if resumed.status == "MARKS_REQUIRED":
                return self._make_receipt(
                    cycle_id=cycle_id,
                    status="MARKS_REQUIRED",
                    before_state_id=before_state_id,
                    after_state_id=before_state_id,
                    missing_mark_assets=resumed.missing_mark_assets,
                )
            if resumed.status != "COMMITTED":
                raise DurableRuntimeError(
                    f"unexpected coordinator resume status after RECONCILED: {resumed.status}"
                )
            try:
                committed = self.journal.mark_committed(
                    cycle_id,
                    state_id=resumed.after_state_id,
                )
            except CycleJournalError as exc:
                raise DurableRuntimeError(str(exc)) from exc
            del committed
            return self._make_receipt(
                cycle_id=cycle_id,
                status="COMMITTED",
                before_state_id=before_state_id,
                after_state_id=resumed.after_state_id,
            )

        if stage == "COMMITTED":
            state_id = self._committed_state_id(cycle_id)
            if state_id is None:
                raise DurableRuntimeError(
                    "journal is COMMITTED but Phase 60 has no commit transition"
                )
            return self._make_receipt(
                cycle_id=cycle_id,
                status="COMMITTED",
                before_state_id=before_state_id,
                after_state_id=state_id,
            )

        if stage == "ABORTED":
            raise DurableRuntimeError("aborted journal cycle cannot be advanced")

        raise DurableRuntimeError(f"unsupported durable journal stage {stage}")

    def process_cycle(
        self,
        cycle: ShadowExecutionCycle,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> DurableRuntimeReceipt:
        self.journal_cycle(cycle)
        return self.advance_pending(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
