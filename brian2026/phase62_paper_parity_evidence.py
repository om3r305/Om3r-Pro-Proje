from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping
import math

from .evidence_ledger import content_hash
from .phase49_promotion_gate import (
    PaperParityPolicy,
    PaperParityReport,
    ShadowPaperObservation,
    evaluate_shadow_paper_parity,
)
from .phase50_execution_reconciliation import ReconciliationBatchReport
from .phase57_shadow_execution_cycle import ShadowExecutionCycle
from .phase61_stateful_paper_venue import PaperCycleReceipt

PHASE62_SCHEMA_VERSION = "brian.phase62-paper-parity-evidence.v1"


class PaperParityEvidenceConflictError(ValueError):
    pass


def _direction_from_side(side: str) -> int:
    if side == "BUY":
        return 1
    if side == "SELL":
        return -1
    raise PaperParityEvidenceConflictError(f"unsupported execution side {side}")


def _reconciliation_by_asset(
    reconciliation: ReconciliationBatchReport,
) -> dict[str, bool]:
    return {
        row.asset_id: bool(row.resolved)
        for row in reconciliation.results
    }


def build_cycle_parity_observations(
    cycle: ShadowExecutionCycle,
    paper: PaperCycleReceipt,
    reconciliation: ReconciliationBatchReport,
    *,
    reference_prices: Mapping[str, float],
    observed_at: float,
) -> tuple[ShadowPaperObservation, ...]:
    """Convert one shadow->paper->reconciliation cycle into Phase 49 evidence.

    Risk-denied and local-veto legs never reached the paper venue and therefore
    are not execution-parity observations. Every submitted paper leg keeps the
    Phase 57 shadow fill fraction, the Phase 61 paper acknowledgement/fill, and
    Phase 50's per-asset reconciliation resolution in one immutable observation.
    """
    if not math.isfinite(observed_at):
        raise ValueError("observed_at must be finite")
    if cycle.cycle_id != paper.cycle_id:
        raise PaperParityEvidenceConflictError("paper receipt belongs to a different cycle")
    cycle_hash = content_hash(cycle.to_dict())
    if paper.cycle_hash != cycle_hash:
        raise PaperParityEvidenceConflictError("paper receipt cycle hash does not match shadow cycle")
    if len(cycle.items) != len(paper.outcomes):
        raise PaperParityEvidenceConflictError(
            "paper outcome count does not match shadow instruction count"
        )
    if reconciliation.live_execution:
        raise PaperParityEvidenceConflictError(
            "live reconciliation cannot enter the paper parity evidence ledger"
        )

    reconciled = _reconciliation_by_asset(reconciliation)
    observations: list[ShadowPaperObservation] = []

    for item, outcome in zip(cycle.items, paper.outcomes):
        if item.asset_id != outcome.asset_id:
            raise PaperParityEvidenceConflictError("paper outcome asset does not match shadow item")
        if item.instruction_kind != outcome.instruction_kind:
            raise PaperParityEvidenceConflictError(
                "paper outcome instruction kind does not match shadow item"
            )

        if outcome.status == "RISK_DENIED":
            if item.risk_receipt.allowed:
                raise PaperParityEvidenceConflictError(
                    "paper says RISK_DENIED but Phase 56 receipt allowed the order"
                )
            continue
        if outcome.status == "LOCAL_VETO":
            if item.execution_receipt is None or item.execution_receipt.status != "VETO_SLIPPAGE":
                raise PaperParityEvidenceConflictError(
                    "LOCAL_VETO requires the Phase 46 slippage-veto receipt"
                )
            continue

        execution = item.execution_receipt
        if execution is None:
            raise PaperParityEvidenceConflictError(
                "submitted paper outcome has no Phase 46 execution receipt"
            )
        if not item.risk_receipt.allowed:
            raise PaperParityEvidenceConflictError(
                "submitted paper outcome came from a risk-denied instruction"
            )

        if item.asset_id not in reference_prices:
            raise KeyError(f"missing parity reference price for {item.asset_id}")
        reference = float(reference_prices[item.asset_id])
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError(f"invalid parity reference price for {item.asset_id}")

        shadow_direction = _direction_from_side(execution.side)
        paper_direction = shadow_direction if outcome.acknowledged else 0
        reconciliation_complete = bool(reconciled.get(item.asset_id, False))

        observations.append(ShadowPaperObservation(
            intent_id=outcome.paper_order_id,
            shadow_direction=shadow_direction,
            paper_direction=paper_direction,
            reference_price=reference,
            paper_fill_price=(
                float(outcome.average_fill_price)
                if outcome.filled_base > 0 and outcome.average_fill_price is not None
                else None
            ),
            shadow_fill_fraction=float(execution.fill_fraction),
            paper_fill_fraction=float(outcome.fill_fraction),
            paper_acknowledged=bool(outcome.acknowledged),
            reconciliation_complete=reconciliation_complete,
            ambiguous_outcome=not reconciliation_complete,
            observed_at=float(observed_at),
        ))

    return tuple(observations)


@dataclass(frozen=True, slots=True)
class PaperParityEvidenceAppendReceipt:
    cycle_id: str
    cycle_evidence_hash: str
    observations_added: int
    total_observations: int
    duplicate: bool


class PaperParityEvidenceLedger:
    """Append-only bridge from Phase 57/61/50 evidence into Phase 49 parity."""

    def __init__(self) -> None:
        self._observations: list[ShadowPaperObservation] = []
        self._cycle_hashes: dict[str, str] = {}
        self._cycle_observations: dict[str, tuple[ShadowPaperObservation, ...]] = {}

    @property
    def observations(self) -> tuple[ShadowPaperObservation, ...]:
        return tuple(self._observations)

    def append_cycle(
        self,
        cycle: ShadowExecutionCycle,
        paper: PaperCycleReceipt,
        reconciliation: ReconciliationBatchReport,
        *,
        reference_prices: Mapping[str, float],
        observed_at: float,
    ) -> PaperParityEvidenceAppendReceipt:
        observations = build_cycle_parity_observations(
            cycle,
            paper,
            reconciliation,
            reference_prices=reference_prices,
            observed_at=observed_at,
        )
        cycle_evidence_hash = content_hash({
            "schema_version": PHASE62_SCHEMA_VERSION,
            "cycle": cycle.to_dict(),
            "paper": paper.to_dict(),
            "reconciliation": reconciliation.to_dict(),
            "reference_prices": dict(sorted(
                (str(asset), float(price))
                for asset, price in reference_prices.items()
            )),
            "observed_at": float(observed_at),
            "observations": [asdict(row) for row in observations],
        })

        previous = self._cycle_hashes.get(cycle.cycle_id)
        if previous is not None:
            if previous != cycle_evidence_hash:
                raise PaperParityEvidenceConflictError(
                    "cycle_id already exists with different parity evidence"
                )
            return PaperParityEvidenceAppendReceipt(
                cycle_id=cycle.cycle_id,
                cycle_evidence_hash=cycle_evidence_hash,
                observations_added=0,
                total_observations=len(self._observations),
                duplicate=True,
            )

        existing_ids = {row.intent_id for row in self._observations}
        duplicates = sorted(
            row.intent_id
            for row in observations
            if row.intent_id in existing_ids
        )
        if duplicates:
            raise PaperParityEvidenceConflictError(
                f"paper parity intent ids already exist: {duplicates}"
            )

        self._cycle_hashes[cycle.cycle_id] = cycle_evidence_hash
        self._cycle_observations[cycle.cycle_id] = observations
        self._observations.extend(observations)
        return PaperParityEvidenceAppendReceipt(
            cycle_id=cycle.cycle_id,
            cycle_evidence_hash=cycle_evidence_hash,
            observations_added=len(observations),
            total_observations=len(self._observations),
            duplicate=False,
        )

    def evaluate(
        self,
        *,
        policy: PaperParityPolicy = PaperParityPolicy(),
    ) -> PaperParityReport:
        if not self._observations:
            raise ValueError("paper parity evidence ledger has no submitted observations")
        return evaluate_shadow_paper_parity(
            tuple(self._observations),
            policy=policy,
        )

    def manifest(self) -> dict[str, object]:
        rows = [asdict(row) for row in self._observations]
        cycles = dict(sorted(self._cycle_hashes.items()))
        return {
            "schema_version": PHASE62_SCHEMA_VERSION,
            "append_only": True,
            "cycle_count": len(cycles),
            "observation_count": len(rows),
            "cycle_hashes": cycles,
            "observations": rows,
            "ledger_hash": content_hash({
                "cycles": cycles,
                "observations": rows,
            }),
            "shadow_only": True,
            "live_execution": False,
        }
