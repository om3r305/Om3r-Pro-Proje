from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Sequence
import math

from .evidence_ledger import content_hash
from .phase46_execution_simulator import SimulatedExecutionReceipt
from .phase50_execution_reconciliation import ReconciliationBatchReport
from .phase55_rebalance_execution_intents import PendingReversalOpen
from .phase56_pretrade_risk_engine import PreTradeRiskReceipt
from .phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from .phase61_stateful_paper_venue import PaperCycleReceipt
from .phase64_local_execution_projector import ProjectionReceipt

PHASE66_SCHEMA_VERSION = "brian.phase66-durable-cycle-journal.v1"
JournalStage = Literal[
    "CYCLE_CREATED",
    "PAPER_APPLIED",
    "LOCAL_PROJECTED",
    "RECONCILIATION_REQUIRED",
    "RECONCILED",
    "COMMITTED",
    "ABORTED",
]


class CycleJournalError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CycleJournalEntry:
    sequence: int
    stage: JournalStage
    cycle_id: str
    cycle_hash: str
    previous_entry_id: str | None
    artifact_hash: str
    artifact_ref: str
    schema_version: str = PHASE66_SCHEMA_VERSION
    entry_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("journal sequence must be non-negative")
        if not self.cycle_id.strip() or not self.artifact_ref.strip():
            raise ValueError("journal cycle_id and artifact_ref are required")
        if len(self.cycle_hash) != 64 or len(self.artifact_hash) != 64:
            raise ValueError("journal hashes must be SHA-256 content hashes")
        if self.sequence == 0 and self.previous_entry_id is not None:
            raise ValueError("first journal entry cannot have a previous entry")
        if self.sequence > 0 and not self.previous_entry_id:
            raise ValueError("non-first journal entry requires previous_entry_id")
        object.__setattr__(self, "entry_id", content_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "stage": self.stage,
            "cycle_id": self.cycle_id,
            "cycle_hash": self.cycle_hash,
            "previous_entry_id": self.previous_entry_id,
            "artifact_hash": self.artifact_hash,
            "artifact_ref": self.artifact_ref,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["entry_id"] = self.entry_id
        return payload


@dataclass(frozen=True, slots=True)
class JournalAppendReceipt:
    cycle_id: str
    stage: JournalStage
    entry_id: str
    sequence: int
    duplicate: bool


def _cycle_hash(cycle: ShadowExecutionCycle) -> str:
    return content_hash(cycle.to_dict())


def _parse_risk_receipt(payload: Mapping[str, object]) -> PreTradeRiskReceipt:
    return PreTradeRiskReceipt(
        action=str(payload["action"]),  # type: ignore[arg-type]
        trading_state=str(payload["trading_state"]),  # type: ignore[arg-type]
        asset_id=str(payload["asset_id"]),
        requested_notional_usd=float(payload["requested_notional_usd"]),
        reduce_only=bool(payload["reduce_only"]),
        reasons=tuple(str(value) for value in payload["reasons"]),  # type: ignore[union-attr]
        projected_position_weight=(
            None
            if payload.get("projected_position_weight") is None
            else float(payload["projected_position_weight"])
        ),
        checks=tuple(
            (str(name), bool(value))
            for name, value in payload["checks"]  # type: ignore[union-attr]
        ),
        schema_version=str(payload.get("schema_version", "brian.phase56-pretrade-risk-engine.v1")),
        shadow_only=bool(payload.get("shadow_only", True)),
        live_execution=bool(payload.get("live_execution", False)),
    )


def _parse_execution_receipt(
    payload: Mapping[str, object] | None,
) -> SimulatedExecutionReceipt | None:
    if payload is None:
        return None
    return SimulatedExecutionReceipt(
        status=str(payload["status"]),  # type: ignore[arg-type]
        side=str(payload["side"]),  # type: ignore[arg-type]
        order_type=str(payload["order_type"]),  # type: ignore[arg-type]
        submit_timestamp=float(payload["submit_timestamp"]),
        venue_timestamp=(
            None if payload.get("venue_timestamp") is None else float(payload["venue_timestamp"])
        ),
        snapshot_timestamp=(
            None if payload.get("snapshot_timestamp") is None else float(payload["snapshot_timestamp"])
        ),
        requested_base=float(payload["requested_base"]),
        filled_base=float(payload["filled_base"]),
        fill_fraction=float(payload["fill_fraction"]),
        average_fill_price=(
            None if payload.get("average_fill_price") is None else float(payload["average_fill_price"])
        ),
        best_reference_price=(
            None if payload.get("best_reference_price") is None else float(payload["best_reference_price"])
        ),
        adverse_slippage_bps=(
            None if payload.get("adverse_slippage_bps") is None else float(payload["adverse_slippage_bps"])
        ),
        levels_consumed=int(payload["levels_consumed"]),
        slipped_one_tick=bool(payload["slipped_one_tick"]),
        reason=str(payload["reason"]),
        shadow_only=bool(payload.get("shadow_only", True)),
        live_execution=bool(payload.get("live_execution", False)),
        schema_version=str(payload.get("schema_version", "brian.phase46-execution-simulator.v1")),
    )


def _parse_pending_reversal(
    payload: Mapping[str, object] | None,
) -> PendingReversalOpen | None:
    if payload is None:
        return None
    return PendingReversalOpen(
        pending_id=str(payload["pending_id"]),
        asset_id=str(payload["asset_id"]),
        required_flat_weight=float(payload["required_flat_weight"]),
        target_weight=float(payload["target_weight"]),
        expected_edge_bps=float(payload["expected_edge_bps"]),
        confidence=float(payload["confidence"]),
        max_slippage_bps=float(payload["max_slippage_bps"]),
        created_at=float(payload["created_at"]),
        ttl_seconds=int(payload["ttl_seconds"]),
        evidence_ids=tuple(str(value) for value in payload["evidence_ids"]),  # type: ignore[union-attr]
        parent_reduction_intent_id=str(payload["parent_reduction_intent_id"]),
        shadow_only=bool(payload.get("shadow_only", True)),
        live_execution=bool(payload.get("live_execution", False)),
        automatic_release=bool(payload.get("automatic_release", False)),
        schema_version=str(payload.get("schema_version", "brian.phase55-rebalance-execution-intents.v1")),
    )


def shadow_cycle_from_dict(payload: Mapping[str, object]) -> ShadowExecutionCycle:
    """Reconstruct and validate a journaled Phase 57 cycle body."""
    items: list[ShadowExecutionCycleItem] = []
    for raw_item in payload["items"]:  # type: ignore[union-attr]
        item = dict(raw_item)
        risk_payload = item["risk_receipt"]
        execution_payload = item.get("execution_receipt")
        pending_payload = item.get("pending_reversal")
        if not isinstance(risk_payload, Mapping):
            raise CycleJournalError("journaled cycle risk_receipt is invalid")
        if execution_payload is not None and not isinstance(execution_payload, Mapping):
            raise CycleJournalError("journaled cycle execution_receipt is invalid")
        if pending_payload is not None and not isinstance(pending_payload, Mapping):
            raise CycleJournalError("journaled cycle pending_reversal is invalid")
        items.append(ShadowExecutionCycleItem(
            instruction_kind=str(item["instruction_kind"]),
            asset_id=str(item["asset_id"]),
            risk_receipt=_parse_risk_receipt(risk_payload),
            execution_receipt=_parse_execution_receipt(execution_payload),
            pending_reversal=_parse_pending_reversal(pending_payload),
            new_risk_cash_reserved_usd=float(item["new_risk_cash_reserved_usd"]),
            status=str(item["status"]),
        ))

    cycle = ShadowExecutionCycle(
        source_plan_id=str(payload["source_plan_id"]),
        items=tuple(items),
        initial_available_cash_usd=float(payload["initial_available_cash_usd"]),
        reserved_new_risk_cash_usd=float(payload["reserved_new_risk_cash_usd"]),
        remaining_unreserved_cash_usd=float(payload["remaining_unreserved_cash_usd"]),
        denied_assets=tuple(str(value) for value in payload["denied_assets"]),  # type: ignore[union-attr]
        pending_reversal_assets=tuple(
            str(value) for value in payload["pending_reversal_assets"]  # type: ignore[union-attr]
        ),
        cycle_id=str(payload["cycle_id"]),
        schema_version=str(payload.get("schema_version", "brian.phase57-shadow-execution-cycle.v1")),
        account_state_mutated=bool(payload.get("account_state_mutated", False)),
        shadow_only=bool(payload.get("shadow_only", True)),
        live_execution=bool(payload.get("live_execution", False)),
    )
    if not cycle.shadow_only or cycle.live_execution or cycle.account_state_mutated:
        raise CycleJournalError("journaled cycle crossed the Phase 57 shadow boundary")
    return cycle


_ALLOWED_NEXT: dict[JournalStage, set[JournalStage]] = {
    "CYCLE_CREATED": {"PAPER_APPLIED", "ABORTED"},
    "PAPER_APPLIED": {"LOCAL_PROJECTED", "ABORTED"},
    "LOCAL_PROJECTED": {"RECONCILIATION_REQUIRED", "RECONCILED", "ABORTED"},
    "RECONCILIATION_REQUIRED": {"RECONCILED", "ABORTED"},
    "RECONCILED": {"COMMITTED", "ABORTED"},
    "COMMITTED": set(),
    "ABORTED": set(),
}


class DurableCycleJournal:
    """Append-only write-ahead journal for one or more Phase 57 paper cycles.

    Full cycle bodies are persisted before downstream paper/local/reconciliation
    side effects. The stage chain then records only content hashes/references to
    downstream artifacts, making crash position explicit and retryable.
    """

    def __init__(self) -> None:
        self._cycles: dict[str, ShadowExecutionCycle] = {}
        self._cycle_hashes: dict[str, str] = {}
        self._entries: list[CycleJournalEntry] = []
        self._latest_by_cycle: dict[str, CycleJournalEntry] = {}
        self._artifact_hash_by_stage: dict[tuple[str, JournalStage], str] = {}

    @property
    def entries(self) -> tuple[CycleJournalEntry, ...]:
        return tuple(self._entries)

    def cycle(self, cycle_id: str) -> ShadowExecutionCycle:
        try:
            return self._cycles[cycle_id]
        except KeyError as exc:
            raise KeyError(f"unknown journal cycle {cycle_id}") from exc

    def latest_stage(self, cycle_id: str) -> JournalStage | None:
        entry = self._latest_by_cycle.get(cycle_id)
        return None if entry is None else entry.stage

    def _append(
        self,
        *,
        cycle_id: str,
        stage: JournalStage,
        artifact_payload: object,
        artifact_ref: str,
    ) -> JournalAppendReceipt:
        if cycle_id not in self._cycles:
            raise CycleJournalError("cycle must be journaled before stage advancement")
        artifact_hash = content_hash(artifact_payload)
        key = (cycle_id, stage)
        previous_artifact = self._artifact_hash_by_stage.get(key)
        if previous_artifact is not None:
            if previous_artifact != artifact_hash:
                raise CycleJournalError(
                    f"{cycle_id} {stage} already exists with different artifact evidence"
                )
            entry = next(
                row
                for row in self._entries
                if row.cycle_id == cycle_id and row.stage == stage
            )
            return JournalAppendReceipt(
                cycle_id=cycle_id,
                stage=stage,
                entry_id=entry.entry_id,
                sequence=entry.sequence,
                duplicate=True,
            )

        previous_cycle_entry = self._latest_by_cycle.get(cycle_id)
        if stage == "CYCLE_CREATED":
            if previous_cycle_entry is not None:
                raise CycleJournalError("cycle already has journal history")
        else:
            if previous_cycle_entry is None:
                raise CycleJournalError("cycle has no CYCLE_CREATED journal entry")
            if stage not in _ALLOWED_NEXT[previous_cycle_entry.stage]:
                raise CycleJournalError(
                    f"illegal journal stage transition {previous_cycle_entry.stage} -> {stage}"
                )

        previous_global_id = (
            None if not self._entries else self._entries[-1].entry_id
        )
        entry = CycleJournalEntry(
            sequence=len(self._entries),
            stage=stage,
            cycle_id=cycle_id,
            cycle_hash=self._cycle_hashes[cycle_id],
            previous_entry_id=previous_global_id,
            artifact_hash=artifact_hash,
            artifact_ref=artifact_ref,
        )
        self._entries.append(entry)
        self._latest_by_cycle[cycle_id] = entry
        self._artifact_hash_by_stage[key] = artifact_hash
        return JournalAppendReceipt(
            cycle_id=cycle_id,
            stage=stage,
            entry_id=entry.entry_id,
            sequence=entry.sequence,
            duplicate=False,
        )

    def record_cycle(self, cycle: ShadowExecutionCycle) -> JournalAppendReceipt:
        if not cycle.shadow_only or cycle.live_execution or cycle.account_state_mutated:
            raise CycleJournalError("journal accepts only non-mutating Phase 57 shadow cycles")
        body_hash = _cycle_hash(cycle)
        existing = self._cycle_hashes.get(cycle.cycle_id)
        if existing is not None:
            if existing != body_hash:
                raise CycleJournalError(
                    "cycle_id already exists with a different full cycle body"
                )
            entry = next(
                row
                for row in self._entries
                if row.cycle_id == cycle.cycle_id and row.stage == "CYCLE_CREATED"
            )
            return JournalAppendReceipt(
                cycle_id=cycle.cycle_id,
                stage="CYCLE_CREATED",
                entry_id=entry.entry_id,
                sequence=entry.sequence,
                duplicate=True,
            )

        self._cycles[cycle.cycle_id] = cycle
        self._cycle_hashes[cycle.cycle_id] = body_hash
        return self._append(
            cycle_id=cycle.cycle_id,
            stage="CYCLE_CREATED",
            artifact_payload=cycle.to_dict(),
            artifact_ref=cycle.cycle_id,
        )

    def mark_paper_applied(
        self,
        cycle_id: str,
        receipt: PaperCycleReceipt,
    ) -> JournalAppendReceipt:
        if receipt.cycle_id != cycle_id:
            raise CycleJournalError("paper receipt belongs to a different cycle")
        if receipt.cycle_hash != self._cycle_hashes.get(cycle_id):
            raise CycleJournalError("paper receipt does not reference the journaled cycle hash")
        return self._append(
            cycle_id=cycle_id,
            stage="PAPER_APPLIED",
            artifact_payload=receipt.to_dict(),
            artifact_ref=receipt.receipt_id,
        )

    def mark_local_projected(
        self,
        cycle_id: str,
        receipt: ProjectionReceipt,
    ) -> JournalAppendReceipt:
        if receipt.cycle_id != cycle_id:
            raise CycleJournalError("projection receipt belongs to a different cycle")
        return self._append(
            cycle_id=cycle_id,
            stage="LOCAL_PROJECTED",
            artifact_payload=asdict(receipt),
            artifact_ref=receipt.projection_hash,
        )

    def mark_reconciliation(
        self,
        cycle_id: str,
        report: ReconciliationBatchReport,
    ) -> JournalAppendReceipt:
        stage: JournalStage = (
            "RECONCILED"
            if report.ready and all(value for _, value in report.checks)
            else "RECONCILIATION_REQUIRED"
        )
        return self._append(
            cycle_id=cycle_id,
            stage=stage,
            artifact_payload=report.to_dict(),
            artifact_ref=content_hash(report.to_dict()),
        )

    def mark_committed(
        self,
        cycle_id: str,
        *,
        state_id: str,
    ) -> JournalAppendReceipt:
        if len(state_id) != 64:
            raise ValueError("committed state_id must be a content hash")
        return self._append(
            cycle_id=cycle_id,
            stage="COMMITTED",
            artifact_payload={"state_id": state_id},
            artifact_ref=state_id,
        )

    def mark_aborted(
        self,
        cycle_id: str,
        *,
        reason: str,
    ) -> JournalAppendReceipt:
        if not reason.strip():
            raise ValueError("abort reason is required")
        return self._append(
            cycle_id=cycle_id,
            stage="ABORTED",
            artifact_payload={"reason": reason},
            artifact_ref=content_hash({"reason": reason}),
        )

    def verify_integrity(self) -> bool:
        previous_id: str | None = None
        latest_by_cycle: dict[str, JournalStage] = {}
        for sequence, entry in enumerate(self._entries):
            if entry.sequence != sequence:
                raise CycleJournalError("journal sequence gap or reorder detected")
            if entry.previous_entry_id != previous_id:
                raise CycleJournalError("journal hash chain is broken")
            if content_hash(entry.identity_payload()) != entry.entry_id:
                raise CycleJournalError("journal entry content hash mismatch")
            if entry.cycle_id not in self._cycles:
                raise CycleJournalError("journal entry references missing cycle body")
            if entry.cycle_hash != _cycle_hash(self._cycles[entry.cycle_id]):
                raise CycleJournalError("journal entry cycle hash does not match stored cycle body")

            prior_stage = latest_by_cycle.get(entry.cycle_id)
            if entry.stage == "CYCLE_CREATED":
                if prior_stage is not None:
                    raise CycleJournalError("cycle contains multiple CYCLE_CREATED entries")
                if entry.artifact_hash != content_hash(self._cycles[entry.cycle_id].to_dict()):
                    raise CycleJournalError("CYCLE_CREATED artifact does not hash full cycle body")
            else:
                if prior_stage is None:
                    raise CycleJournalError("cycle stage appears before CYCLE_CREATED")
                if entry.stage not in _ALLOWED_NEXT[prior_stage]:
                    raise CycleJournalError(
                        f"illegal replayed stage transition {prior_stage} -> {entry.stage}"
                    )
            latest_by_cycle[entry.cycle_id] = entry.stage
            previous_id = entry.entry_id

        expected_latest = {
            cycle_id: entry.stage
            for cycle_id, entry in self._latest_by_cycle.items()
        }
        if latest_by_cycle != expected_latest:
            raise CycleJournalError("journal latest-stage index diverges from replay")
        return True

    def manifest(self) -> dict[str, object]:
        cycle_rows = {
            cycle_id: cycle.to_dict()
            for cycle_id, cycle in sorted(self._cycles.items())
        }
        entry_rows = [entry.to_dict() for entry in self._entries]
        return {
            "schema_version": PHASE66_SCHEMA_VERSION,
            "append_only": True,
            "cycles": cycle_rows,
            "entries": entry_rows,
            "journal_hash": content_hash({
                "cycles": cycle_rows,
                "entries": entry_rows,
            }),
            "shadow_only": True,
            "live_execution": False,
        }


def restore_cycle_journal(
    manifest: Mapping[str, object],
) -> DurableCycleJournal:
    if manifest.get("append_only") is not True:
        raise CycleJournalError("journal manifest is not append-only")
    if manifest.get("shadow_only") is not True or manifest.get("live_execution") is not False:
        raise CycleJournalError("journal manifest crossed the live boundary")

    cycles_raw = manifest.get("cycles")
    entries_raw = manifest.get("entries")
    if not isinstance(cycles_raw, Mapping) or not isinstance(entries_raw, Sequence):
        raise CycleJournalError("journal manifest is missing cycles/entries")

    journal = DurableCycleJournal()
    for cycle_id, raw_cycle in cycles_raw.items():
        if not isinstance(raw_cycle, Mapping):
            raise CycleJournalError("journal cycle payload is invalid")
        cycle = shadow_cycle_from_dict(raw_cycle)
        if cycle.cycle_id != str(cycle_id):
            raise CycleJournalError("journal cycle map key does not match cycle body")
        journal._cycles[cycle.cycle_id] = cycle
        journal._cycle_hashes[cycle.cycle_id] = _cycle_hash(cycle)

    parsed_entries: list[CycleJournalEntry] = []
    for raw_entry in entries_raw:
        if not isinstance(raw_entry, Mapping):
            raise CycleJournalError("journal entry payload is invalid")
        expected_id = str(raw_entry["entry_id"])
        entry = CycleJournalEntry(
            sequence=int(raw_entry["sequence"]),
            stage=str(raw_entry["stage"]),  # type: ignore[arg-type]
            cycle_id=str(raw_entry["cycle_id"]),
            cycle_hash=str(raw_entry["cycle_hash"]),
            previous_entry_id=(
                None
                if raw_entry.get("previous_entry_id") is None
                else str(raw_entry["previous_entry_id"])
            ),
            artifact_hash=str(raw_entry["artifact_hash"]),
            artifact_ref=str(raw_entry["artifact_ref"]),
            schema_version=str(raw_entry.get("schema_version", PHASE66_SCHEMA_VERSION)),
        )
        if entry.entry_id != expected_id:
            raise CycleJournalError("journal entry content hash mismatch")
        parsed_entries.append(entry)

    cycle_rows = {
        cycle_id: cycle.to_dict()
        for cycle_id, cycle in sorted(journal._cycles.items())
    }
    entry_rows = [entry.to_dict() for entry in parsed_entries]
    expected_hash = content_hash({
        "cycles": cycle_rows,
        "entries": entry_rows,
    })
    if manifest.get("journal_hash") != expected_hash:
        raise CycleJournalError("journal manifest hash mismatch")

    journal._entries = parsed_entries
    journal._latest_by_cycle = {}
    journal._artifact_hash_by_stage = {}
    for entry in parsed_entries:
        journal._latest_by_cycle[entry.cycle_id] = entry
        journal._artifact_hash_by_stage[(entry.cycle_id, entry.stage)] = entry.artifact_hash

    journal.verify_integrity()
    return journal
