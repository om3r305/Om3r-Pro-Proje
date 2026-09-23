from __future__ import annotations

from dataclasses import dataclass

from .evidence_ledger import content_hash
from .phase50_execution_reconciliation import ReconciliationBatchReport
from .phase60_shadow_state_ledger import ShadowStateLedger
from .phase61_stateful_paper_venue import PaperVenue
from .phase63_crash_recovery import (
    PaperVenueCheckpoint,
    ShadowRuntimeCheckpoint,
    RuntimeRecoveryError,
    restore_paper_venue,
    restore_runtime_checkpoint,
)
from .phase64_local_execution_projector import (
    LocalExecutionProjector,
    LocalExecutionProjectionError,
)

PHASE65_SCHEMA_VERSION = "brian.phase65-event-sourced-local-recovery.v1"


@dataclass(frozen=True, slots=True)
class LocalProjectionReplayReceipt:
    account_id: str
    cycles_available: int
    cycles_replayed: int
    fills_available: int
    fills_projected: int
    projection_version: int
    projection_hash: str
    complete: bool
    schema_version: str = PHASE65_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


@dataclass(frozen=True, slots=True)
class RecoveredShadowPaperRuntime:
    ledger: ShadowStateLedger
    venue: PaperVenue
    projector: LocalExecutionProjector
    reconciliation: ReconciliationBatchReport
    replay: LocalProjectionReplayReceipt
    pending_cycle_id: str | None
    runtime_hash: str
    schema_version: str = PHASE65_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


def replay_local_execution_history(
    checkpoint: PaperVenueCheckpoint,
    *,
    through_state_version: int | None = None,
) -> tuple[LocalExecutionProjector, LocalProjectionReplayReceipt]:
    """Rebuild the local execution cache from durable paper order/fill events.

    The function first validates the paper checkpoint by replaying the venue
    itself through Phase 63, then independently replays each PaperCycleReceipt
    through Phase 64. No serialized local-position snapshot is trusted.
    """
    # Phase 63 independently validates fill/receipt hashes, ordering, cash and
    # final venue positions before those events are used for local recovery.
    restore_paper_venue(checkpoint)

    total_cycles = len(checkpoint.cycle_receipts)
    if through_state_version is None:
        limit = total_cycles
    else:
        limit = int(through_state_version)
        if limit < 0 or limit > total_cycles:
            raise ValueError(
                "through_state_version must be between 0 and paper state_version"
            )

    fills_by_id = {
        fill.fill_id: fill
        for fill in checkpoint.fill_sequence
    }
    projector = LocalExecutionProjector(checkpoint.config.account_id)
    fills_projected = 0

    for receipt in checkpoint.cycle_receipts[:limit]:
        try:
            projected = projector.process_cycle(receipt, fills_by_id)
        except LocalExecutionProjectionError as exc:
            raise RuntimeRecoveryError(
                f"local execution event replay failed at cycle {receipt.cycle_id}: {exc}"
            ) from exc
        fills_projected += projected.fills_applied

    manifest = projector.manifest()
    replay = LocalProjectionReplayReceipt(
        account_id=checkpoint.config.account_id,
        cycles_available=total_cycles,
        cycles_replayed=limit,
        fills_available=len(checkpoint.fill_sequence),
        fills_projected=fills_projected,
        projection_version=projector.projection_version,
        projection_hash=str(manifest["projection_hash"]),
        complete=limit == total_cycles,
    )
    return projector, replay


def restore_runtime_with_local_projection(
    checkpoint: ShadowRuntimeCheckpoint,
) -> RecoveredShadowPaperRuntime:
    """Restore ledger + paper venue + independent local execution cache.

    Full local event replay must reconcile against the independently restored
    paper venue before the recovered runtime is returned.
    """
    ledger, venue = restore_runtime_checkpoint(checkpoint)
    projector, replay = replay_local_execution_history(checkpoint.paper)
    if not replay.complete:
        raise RuntimeRecoveryError("full runtime restore requires complete local event replay")

    tracked_assets = tuple(sorted(
        set(ledger.head_state.covered_assets)
        | set(venue.positions)
        | set(projector.positions)
    ))
    if not tracked_assets:
        raise RuntimeRecoveryError("restored runtime has no tracked assets")

    reconciliation = venue.reconcile_against_local(
        projector.local_positions(tracked_assets=tracked_assets),
        tracked_assets=tracked_assets,
    )
    if not reconciliation.ready or not all(
        value for _, value in reconciliation.checks
    ):
        raise RuntimeRecoveryError(
            "restored local execution projection does not reconcile with paper venue"
        )

    runtime_hash = content_hash({
        "schema_version": PHASE65_SCHEMA_VERSION,
        "checkpoint_id": checkpoint.checkpoint_id,
        "ledger_hash": ledger.manifest()["ledger_hash"],
        "paper_checkpoint_id": checkpoint.paper.checkpoint_id,
        "projection_hash": replay.projection_hash,
        "reconciliation": reconciliation.to_dict(),
        "pending_cycle_id": ledger.pending_cycle_id,
    })
    return RecoveredShadowPaperRuntime(
        ledger=ledger,
        venue=venue,
        projector=projector,
        reconciliation=reconciliation,
        replay=replay,
        pending_cycle_id=ledger.pending_cycle_id,
        runtime_hash=runtime_hash,
    )


def audit_partial_local_replay(
    checkpoint: ShadowRuntimeCheckpoint,
    *,
    through_state_version: int,
) -> ReconciliationBatchReport:
    """Forensics helper: prove a truncated local replay is detectable.

    The paper venue is restored completely while the local projector intentionally
    stops at an earlier cycle. Phase 50 then exposes any resulting divergence.
    """
    ledger, venue = restore_runtime_checkpoint(checkpoint)
    projector, _ = replay_local_execution_history(
        checkpoint.paper,
        through_state_version=through_state_version,
    )
    tracked_assets = tuple(sorted(
        set(ledger.head_state.covered_assets)
        | set(venue.positions)
        | set(projector.positions)
    ))
    if not tracked_assets:
        raise RuntimeRecoveryError("partial replay audit has no tracked assets")
    return venue.reconcile_against_local(
        projector.local_positions(tracked_assets=tracked_assets),
        tracked_assets=tracked_assets,
    )
