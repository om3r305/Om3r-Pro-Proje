from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

from .evidence_ledger import content_hash
from .phase50_execution_reconciliation import LocalPositionState
from .phase61_stateful_paper_venue import (
    PaperCycleReceipt,
    PaperFill,
    PaperOrderOutcome,
)

PHASE64_SCHEMA_VERSION = "brian.phase64-local-execution-projector.v1"
ProjectedOrderStatus = Literal[
    "RISK_DENIED",
    "LOCAL_VETO",
    "ACKNOWLEDGED_NO_FILL",
    "VENUE_REJECTED_BALANCE",
    "PARTIAL_FILL",
    "FILLED",
]


class LocalExecutionProjectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProjectedOrder:
    paper_order_id: str
    cycle_id: str
    asset_id: str
    instruction_kind: str
    status: ProjectedOrderStatus
    acknowledged: bool
    requested_base: float
    filled_base: float
    average_fill_price: float | None
    fill_ids: tuple[str, ...]
    schema_version: str = PHASE64_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.paper_order_id.strip() or not self.cycle_id.strip() or not self.asset_id.strip():
            raise ValueError("projected order identity is required")
        if not math.isfinite(self.requested_base) or self.requested_base < 0:
            raise ValueError("requested_base must be finite and non-negative")
        if not math.isfinite(self.filled_base) or self.filled_base < 0:
            raise ValueError("filled_base must be finite and non-negative")
        if self.filled_base > self.requested_base + 1e-12:
            raise ValueError("filled_base cannot exceed requested_base")
        if self.filled_base > 0 and (
            self.average_fill_price is None
            or not math.isfinite(self.average_fill_price)
            or self.average_fill_price <= 0
        ):
            raise ValueError("filled projected order requires average_fill_price")
        if len(self.fill_ids) != len(set(self.fill_ids)):
            raise ValueError("projected order fill ids must be unique")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectedPosition:
    asset_id: str
    quantity: float
    avg_entry_price: float | None
    realized_pnl_quote: float
    source_fill_ids: tuple[str, ...]
    schema_version: str = PHASE64_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("projected position asset_id is required")
        if not math.isfinite(self.quantity) or not math.isfinite(self.realized_pnl_quote):
            raise ValueError("projected position values must be finite")
        if abs(self.quantity) > 1e-12:
            if self.avg_entry_price is None or not math.isfinite(self.avg_entry_price) or self.avg_entry_price <= 0:
                raise ValueError("open projected position requires positive avg_entry_price")
        elif self.avg_entry_price is not None:
            raise ValueError("flat projected position cannot retain avg_entry_price")
        if len(self.source_fill_ids) != len(set(self.source_fill_ids)):
            raise ValueError("projected position fill ids must be unique")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectionReceipt:
    cycle_id: str
    paper_receipt_id: str
    orders_projected: int
    fills_applied: int
    duplicate: bool
    projection_version: int
    projection_hash: str
    schema_version: str = PHASE64_SCHEMA_VERSION


def _paper_fill_identity(fill: PaperFill) -> str:
    return content_hash({
        "schema_version": fill.schema_version,
        "paper_order_id": fill.paper_order_id,
        "asset_id": fill.asset_id,
        "side": fill.side,
        "filled_base": fill.quantity_base,
        "average_fill_price": fill.price,
        "venue_timestamp": fill.timestamp,
    })


def _paper_receipt_identity(receipt: PaperCycleReceipt) -> str:
    return content_hash({
        "schema_version": receipt.schema_version,
        "cycle_id": receipt.cycle_id,
        "cycle_hash": receipt.cycle_hash,
        "outcomes": [row.to_dict() for row in receipt.outcomes],
        "fill_ids": list(receipt.fill_ids),
        "cash_before_usd": receipt.cash_before_usd,
        "cash_after_usd": receipt.cash_after_usd,
        "state_version_before": receipt.state_version_before,
        "state_version_after": receipt.state_version_after,
    })


def _weighted_fill_price(fills: Sequence[PaperFill]) -> float | None:
    quantity = sum(fill.quantity_base for fill in fills)
    if quantity <= 1e-15:
        return None
    return sum(fill.quantity_base * fill.price for fill in fills) / quantity


def _apply_position_fill(
    current: ProjectedPosition,
    fill: PaperFill,
) -> ProjectedPosition:
    delta = fill.quantity_base if fill.side == "BUY" else -fill.quantity_base
    old_qty = current.quantity
    old_avg = current.avg_entry_price
    new_qty = old_qty + delta
    realized = current.realized_pnl_quote

    if abs(old_qty) <= 1e-12:
        new_avg = fill.price if abs(new_qty) > 1e-12 else None
    elif old_qty * delta > 0:
        assert old_avg is not None
        new_avg = (
            abs(old_qty) * old_avg + abs(delta) * fill.price
        ) / (abs(old_qty) + abs(delta))
    else:
        assert old_avg is not None
        closed_qty = min(abs(old_qty), abs(delta))
        old_sign = 1.0 if old_qty > 0 else -1.0
        realized += closed_qty * (fill.price - old_avg) * old_sign
        if abs(new_qty) <= 1e-12:
            new_qty = 0.0
            new_avg = None
        elif old_qty * new_qty > 0:
            new_avg = old_avg
        else:
            new_avg = fill.price

    return ProjectedPosition(
        asset_id=fill.asset_id,
        quantity=float(new_qty),
        avg_entry_price=None if abs(new_qty) <= 1e-12 else float(new_avg),
        realized_pnl_quote=float(realized),
        source_fill_ids=current.source_fill_ids + (fill.fill_id,),
    )


class LocalExecutionProjector:
    """Independent local execution-state projection from paper execution events.

    The projector never reads PaperVenue positions/cash. It consumes only the
    immutable PaperCycleReceipt + referenced PaperFill events, mirroring the
    execution-engine pattern where fills update/create cached positions. This
    makes Phase 50 reconciliation a comparison of two independently derived
    states rather than a copy of venue state back into itself.
    """

    def __init__(self, account_id: str) -> None:
        if not account_id.strip():
            raise ValueError("projector account_id is required")
        self.account_id = account_id
        self._orders: dict[str, ProjectedOrder] = {}
        self._positions: dict[str, ProjectedPosition] = {}
        self._seen_fill_ids: set[str] = set()
        self._cycle_receipt_hashes: dict[str, str] = {}
        self._cycle_projection_receipts: dict[str, ProjectionReceipt] = {}
        self._projection_version = 0

    @property
    def projection_version(self) -> int:
        return self._projection_version

    @property
    def orders(self) -> Mapping[str, ProjectedOrder]:
        return dict(self._orders)

    @property
    def positions(self) -> Mapping[str, ProjectedPosition]:
        return dict(self._positions)

    def position(self, asset_id: str) -> ProjectedPosition:
        return self._positions.get(
            asset_id,
            ProjectedPosition(asset_id, 0.0, None, 0.0, ()),
        )

    def _projection_hash(
        self,
        orders: Mapping[str, ProjectedOrder],
        positions: Mapping[str, ProjectedPosition],
        seen_fill_ids: set[str],
        version: int,
    ) -> str:
        return content_hash({
            "schema_version": PHASE64_SCHEMA_VERSION,
            "account_id": self.account_id,
            "projection_version": version,
            "orders": [
                orders[key].to_dict()
                for key in sorted(orders)
            ],
            "positions": [
                positions[key].to_dict()
                for key in sorted(positions)
            ],
            "seen_fill_ids": sorted(seen_fill_ids),
        })

    def process_cycle(
        self,
        receipt: PaperCycleReceipt,
        fills_by_id: Mapping[str, PaperFill],
    ) -> ProjectionReceipt:
        """Atomically project one paper execution receipt and its fill events."""
        if receipt.live_execution or not receipt.paper_only:
            raise LocalExecutionProjectionError(
                "local projector accepts only paper execution receipts"
            )
        if _paper_receipt_identity(receipt) != receipt.receipt_id:
            raise LocalExecutionProjectionError("paper cycle receipt content hash mismatch")

        receipt_hash = content_hash(receipt.to_dict())
        previous = self._cycle_receipt_hashes.get(receipt.cycle_id)
        if previous is not None:
            if previous != receipt_hash:
                raise LocalExecutionProjectionError(
                    "cycle_id already projected with different receipt evidence"
                )
            return self._cycle_projection_receipts[receipt.cycle_id]

        # Validate the entire cycle before changing local state.
        outcome_order_ids = [row.paper_order_id for row in receipt.outcomes]
        if len(outcome_order_ids) != len(set(outcome_order_ids)):
            raise LocalExecutionProjectionError(
                "paper cycle contains duplicate paper_order_id values"
            )
        if set(receipt.fill_ids) != {
            fill_id
            for outcome in receipt.outcomes
            for fill_id in outcome.fill_ids
        }:
            raise LocalExecutionProjectionError(
                "cycle receipt fill_ids do not match outcome fill references"
            )

        validated_fills: dict[str, PaperFill] = {}
        for fill_id in receipt.fill_ids:
            if fill_id in self._seen_fill_ids:
                raise LocalExecutionProjectionError(
                    f"fill {fill_id} was already applied in a prior cycle"
                )
            try:
                fill = fills_by_id[fill_id]
            except KeyError as exc:
                raise LocalExecutionProjectionError(
                    f"missing local execution fill event {fill_id}"
                ) from exc
            if fill.fill_id != fill_id or _paper_fill_identity(fill) != fill.fill_id:
                raise LocalExecutionProjectionError(
                    f"paper fill content hash mismatch for {fill_id}"
                )
            if fill.cycle_id != receipt.cycle_id:
                raise LocalExecutionProjectionError(
                    f"fill {fill_id} belongs to a different cycle"
                )
            validated_fills[fill_id] = fill

        staged_orders = dict(self._orders)
        staged_positions = dict(self._positions)
        staged_seen = set(self._seen_fill_ids)
        fills_applied = 0

        for outcome in receipt.outcomes:
            if outcome.paper_order_id in staged_orders:
                raise LocalExecutionProjectionError(
                    f"paper order {outcome.paper_order_id} already exists locally"
                )
            fills = tuple(validated_fills[fill_id] for fill_id in outcome.fill_ids)
            for fill in fills:
                if fill.paper_order_id != outcome.paper_order_id:
                    raise LocalExecutionProjectionError(
                        "fill paper_order_id does not match outcome"
                    )
                if fill.asset_id != outcome.asset_id:
                    raise LocalExecutionProjectionError(
                        "fill asset_id does not match outcome"
                    )

            fill_qty = sum(fill.quantity_base for fill in fills)
            weighted_price = _weighted_fill_price(fills)
            if not math.isclose(
                fill_qty,
                outcome.filled_base,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise LocalExecutionProjectionError(
                    "outcome filled_base does not reconcile referenced fills"
                )
            if outcome.filled_base > 0:
                if weighted_price is None or outcome.average_fill_price is None:
                    raise LocalExecutionProjectionError(
                        "filled outcome is missing average price"
                    )
                if not math.isclose(
                    weighted_price,
                    outcome.average_fill_price,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise LocalExecutionProjectionError(
                        "outcome average_fill_price does not reconcile referenced fills"
                    )
            elif fills:
                raise LocalExecutionProjectionError(
                    "zero-fill outcome cannot reference fill events"
                )

            no_fill_statuses = {
                "RISK_DENIED",
                "LOCAL_VETO",
                "ACKNOWLEDGED_NO_FILL",
                "VENUE_REJECTED_BALANCE",
            }
            if outcome.status in no_fill_statuses:
                if outcome.filled_base > 1e-12 or outcome.fill_ids:
                    raise LocalExecutionProjectionError(
                        f"{outcome.status} cannot contain fills"
                    )
            elif outcome.status in {"FILLED", "PARTIAL_FILL"}:
                if not outcome.acknowledged or outcome.filled_base <= 0 or not outcome.fill_ids:
                    raise LocalExecutionProjectionError(
                        f"{outcome.status} requires acknowledged fill evidence"
                    )
                expected_fraction = (
                    outcome.filled_base / outcome.requested_base
                    if outcome.requested_base > 0
                    else 0.0
                )
                if not math.isclose(
                    expected_fraction,
                    outcome.fill_fraction,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise LocalExecutionProjectionError(
                        "outcome fill_fraction does not reconcile quantities"
                    )
            else:
                raise LocalExecutionProjectionError(
                    f"unsupported paper outcome status {outcome.status}"
                )

            staged_orders[outcome.paper_order_id] = ProjectedOrder(
                paper_order_id=outcome.paper_order_id,
                cycle_id=outcome.cycle_id,
                asset_id=outcome.asset_id,
                instruction_kind=outcome.instruction_kind,
                status=outcome.status,
                acknowledged=outcome.acknowledged,
                requested_base=outcome.requested_base,
                filled_base=outcome.filled_base,
                average_fill_price=outcome.average_fill_price,
                fill_ids=outcome.fill_ids,
            )

            for fill in fills:
                current = staged_positions.get(
                    fill.asset_id,
                    ProjectedPosition(fill.asset_id, 0.0, None, 0.0, ()),
                )
                staged_positions[fill.asset_id] = _apply_position_fill(current, fill)
                staged_seen.add(fill.fill_id)
                fills_applied += 1

        new_version = self._projection_version + 1
        projection_hash = self._projection_hash(
            staged_orders,
            staged_positions,
            staged_seen,
            new_version,
        )
        projection_receipt = ProjectionReceipt(
            cycle_id=receipt.cycle_id,
            paper_receipt_id=receipt.receipt_id,
            orders_projected=len(receipt.outcomes),
            fills_applied=fills_applied,
            duplicate=False,
            projection_version=new_version,
            projection_hash=projection_hash,
        )

        self._orders = staged_orders
        self._positions = staged_positions
        self._seen_fill_ids = staged_seen
        self._projection_version = new_version
        self._cycle_receipt_hashes[receipt.cycle_id] = receipt_hash
        self._cycle_projection_receipts[receipt.cycle_id] = projection_receipt
        return projection_receipt

    def local_positions(
        self,
        *,
        tracked_assets: Sequence[str] | None = None,
    ) -> dict[str, LocalPositionState]:
        allowed = (
            None
            if tracked_assets is None
            else {str(asset) for asset in tracked_assets}
        )
        result: dict[str, LocalPositionState] = {}
        for asset, position in self._positions.items():
            if allowed is not None and asset not in allowed:
                continue
            if abs(position.quantity) <= 1e-12:
                continue
            result[asset] = LocalPositionState(
                account_id=self.account_id,
                asset_id=asset,
                quantity=position.quantity,
                avg_entry_price=position.avg_entry_price,
                source_fill_ids=position.source_fill_ids,
            )
        return result

    def manifest(self) -> dict[str, object]:
        projection_hash = self._projection_hash(
            self._orders,
            self._positions,
            self._seen_fill_ids,
            self._projection_version,
        )
        return {
            "schema_version": PHASE64_SCHEMA_VERSION,
            "account_id": self.account_id,
            "projection_version": self._projection_version,
            "orders": {
                key: value.to_dict()
                for key, value in sorted(self._orders.items())
            },
            "positions": {
                key: value.to_dict()
                for key, value in sorted(self._positions.items())
            },
            "seen_fill_ids": tuple(sorted(self._seen_fill_ids)),
            "cycle_receipt_hashes": dict(sorted(self._cycle_receipt_hashes.items())),
            "projection_hash": projection_hash,
            "shadow_only": True,
            "live_execution": False,
        }
