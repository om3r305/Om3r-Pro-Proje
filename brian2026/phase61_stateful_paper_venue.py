from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Sequence
import math

from .evidence_ledger import content_hash
from .phase50_execution_reconciliation import (
    LocalPositionState,
    ReconciliationBatchReport,
    ReconciliationBatchPolicy,
    VenuePositionReport,
    reconcile_execution_state,
)
from .phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from .phase60_shadow_state_ledger import ShadowAccountState

PHASE61_SCHEMA_VERSION = "brian.phase61-stateful-paper-venue.v1"
PaperOrderStatus = Literal[
    "RISK_DENIED",
    "LOCAL_VETO",
    "ACKNOWLEDGED_NO_FILL",
    "VENUE_REJECTED_BALANCE",
    "FILLED",
    "PARTIAL_FILL",
]


@dataclass(frozen=True, slots=True)
class PaperVenueConfig:
    account_id: str = "BRIAN-PAPER"
    starting_cash_usd: float = 500.0
    fee_bps: float = 10.0
    allow_short: bool = True

    def __post_init__(self) -> None:
        if not self.account_id.strip():
            raise ValueError("paper account_id is required")
        if not math.isfinite(self.starting_cash_usd) or self.starting_cash_usd <= 0:
            raise ValueError("starting_cash_usd must be positive")
        if not math.isfinite(self.fee_bps) or self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")


@dataclass(frozen=True, slots=True)
class PaperFill:
    fill_id: str
    paper_order_id: str
    cycle_id: str
    asset_id: str
    side: Literal["BUY", "SELL"]
    quantity_base: float
    price: float
    fee_quote: float
    timestamp: float
    schema_version: str = PHASE61_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all((
            self.fill_id.strip(),
            self.paper_order_id.strip(),
            self.cycle_id.strip(),
            self.asset_id.strip(),
        )):
            raise ValueError("paper fill identity is required")
        if self.side not in ("BUY", "SELL"):
            raise ValueError("paper fill side must be BUY or SELL")
        if not math.isfinite(self.quantity_base) or self.quantity_base <= 0:
            raise ValueError("paper fill quantity must be positive")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("paper fill price must be positive")
        if not math.isfinite(self.fee_quote) or self.fee_quote < 0:
            raise ValueError("paper fill fee must be non-negative")
        if not math.isfinite(self.timestamp):
            raise ValueError("paper fill timestamp must be finite")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperPosition:
    asset_id: str
    quantity: float
    avg_entry_price: float | None
    realized_pnl_quote: float
    source_fill_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("paper position asset_id is required")
        if not math.isfinite(self.quantity) or not math.isfinite(self.realized_pnl_quote):
            raise ValueError("paper position numeric values must be finite")
        if abs(self.quantity) > 1e-12:
            if self.avg_entry_price is None or not math.isfinite(self.avg_entry_price) or self.avg_entry_price <= 0:
                raise ValueError("open paper position requires positive avg_entry_price")
        elif self.avg_entry_price is not None:
            raise ValueError("flat paper position must not retain avg_entry_price")
        if len(self.source_fill_ids) != len(set(self.source_fill_ids)):
            raise ValueError("paper position fill ids must be unique")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperOrderOutcome:
    paper_order_id: str
    cycle_id: str
    asset_id: str
    instruction_kind: str
    status: PaperOrderStatus
    acknowledged: bool
    requested_notional_usd: float
    requested_base: float
    filled_base: float
    fill_fraction: float
    average_fill_price: float | None
    fill_ids: tuple[str, ...]
    reason: str
    schema_version: str = PHASE61_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.paper_order_id.strip() or not self.cycle_id.strip() or not self.asset_id.strip():
            raise ValueError("paper order identity is required")
        for label, value in (
            ("requested_notional_usd", self.requested_notional_usd),
            ("requested_base", self.requested_base),
            ("filled_base", self.filled_base),
            ("fill_fraction", self.fill_fraction),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.fill_fraction > 1 + 1e-12:
            raise ValueError("fill_fraction cannot exceed 1")
        if self.filled_base > 0 and (
            self.average_fill_price is None
            or not math.isfinite(self.average_fill_price)
            or self.average_fill_price <= 0
        ):
            raise ValueError("filled paper order requires average_fill_price")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperCycleReceipt:
    cycle_id: str
    cycle_hash: str
    outcomes: tuple[PaperOrderOutcome, ...]
    fill_ids: tuple[str, ...]
    cash_before_usd: float
    cash_after_usd: float
    state_version_before: int
    state_version_after: int
    receipt_id: str
    schema_version: str = PHASE61_SCHEMA_VERSION
    paper_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PaperVenueConflictError(ValueError):
    pass


class PaperVenue:
    """Stateful paper execution boundary driven by Phase 57 execution receipts.

    The paper venue is the first component allowed to turn a simulated execution
    receipt into persistent paper account state. Applying a cycle is idempotent
    by content hash. Phase 60 still refuses to trust that state until a Phase 50
    reconciliation report confirms the local mirror matches the venue reports.
    """

    def __init__(self, config: PaperVenueConfig = PaperVenueConfig()) -> None:
        self.config = config
        self.cash_usd = float(config.starting_cash_usd)
        self._positions: dict[str, PaperPosition] = {}
        self._fills: dict[str, PaperFill] = {}
        self._cycle_hashes: dict[str, str] = {}
        self._cycle_receipts: dict[str, PaperCycleReceipt] = {}
        self._state_version = 0

    @property
    def state_version(self) -> int:
        return self._state_version

    @property
    def positions(self) -> Mapping[str, PaperPosition]:
        return dict(self._positions)

    @property
    def fills(self) -> tuple[PaperFill, ...]:
        return tuple(self._fills.values())

    @property
    def cycle_receipts(self) -> tuple[PaperCycleReceipt, ...]:
        return tuple(self._cycle_receipts.values())

    def cycle_receipt(self, cycle_id: str) -> PaperCycleReceipt:
        try:
            return self._cycle_receipts[cycle_id]
        except KeyError as exc:
            raise KeyError(f"unknown paper cycle {cycle_id}") from exc

    def position(self, asset_id: str) -> PaperPosition:
        return self._positions.get(
            asset_id,
            PaperPosition(asset_id, 0.0, None, 0.0, ()),
        )

    def _paper_order_id(
        self,
        cycle: ShadowExecutionCycle,
        index: int,
        item: ShadowExecutionCycleItem,
    ) -> str:
        return content_hash({
            "schema_version": PHASE61_SCHEMA_VERSION,
            "cycle_id": cycle.cycle_id,
            "cycle_hash": content_hash(cycle.to_dict()),
            "item_index": index,
            "asset_id": item.asset_id,
            "instruction_kind": item.instruction_kind,
            "risk_receipt": item.risk_receipt.to_dict(),
            "execution_receipt": (
                None
                if item.execution_receipt is None
                else item.execution_receipt.to_dict()
            ),
        })

    def _apply_fill(self, fill: PaperFill) -> None:
        if fill.fill_id in self._fills:
            existing = self._fills[fill.fill_id]
            if existing != fill:
                raise PaperVenueConflictError("fill_id reused with different paper fill")
            return

        current = self.position(fill.asset_id)
        delta = fill.quantity_base if fill.side == "BUY" else -fill.quantity_base
        old_qty = current.quantity
        old_avg = current.avg_entry_price
        new_qty = old_qty + delta

        if not self.config.allow_short and new_qty < -1e-12:
            raise PaperVenueConflictError("paper venue short exposure is disabled")

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

        # Quote-cash accounting: BUY spends cash, SELL receives cash. Fees always
        # reduce cash. Mark-to-market equity is computed separately from cash.
        self.cash_usd -= delta * fill.price
        self.cash_usd -= fill.fee_quote

        self._fills[fill.fill_id] = fill
        updated = PaperPosition(
            asset_id=fill.asset_id,
            quantity=float(new_qty),
            avg_entry_price=None if abs(new_qty) <= 1e-12 else float(new_avg),
            realized_pnl_quote=float(realized),
            source_fill_ids=current.source_fill_ids + (fill.fill_id,),
        )
        self._positions[fill.asset_id] = updated

    def apply_cycle(self, cycle: ShadowExecutionCycle) -> PaperCycleReceipt:
        if not cycle.shadow_only or cycle.live_execution or cycle.account_state_mutated:
            raise PaperVenueConflictError("paper venue accepts only Phase 57 shadow cycles")

        cycle_hash = content_hash(cycle.to_dict())
        previous_hash = self._cycle_hashes.get(cycle.cycle_id)
        if previous_hash is not None:
            if previous_hash != cycle_hash:
                raise PaperVenueConflictError(
                    "cycle_id already exists with different shadow execution evidence"
                )
            return self._cycle_receipts[cycle.cycle_id]

        cash_before = self.cash_usd
        version_before = self._state_version
        outcomes: list[PaperOrderOutcome] = []
        cycle_fill_ids: list[str] = []

        for index, item in enumerate(cycle.items):
            paper_order_id = self._paper_order_id(cycle, index, item)
            risk = item.risk_receipt
            execution = item.execution_receipt

            if not risk.allowed:
                outcomes.append(PaperOrderOutcome(
                    paper_order_id=paper_order_id,
                    cycle_id=cycle.cycle_id,
                    asset_id=item.asset_id,
                    instruction_kind=item.instruction_kind,
                    status="RISK_DENIED",
                    acknowledged=False,
                    requested_notional_usd=risk.requested_notional_usd,
                    requested_base=0.0,
                    filled_base=0.0,
                    fill_fraction=0.0,
                    average_fill_price=None,
                    fill_ids=(),
                    reason="Phase 56 denied before paper venue submission",
                ))
                continue

            if execution is None:
                raise PaperVenueConflictError(
                    f"{item.asset_id} was risk-allowed but has no execution receipt"
                )

            requested_base = float(execution.requested_base)
            if execution.status == "VETO_SLIPPAGE":
                outcomes.append(PaperOrderOutcome(
                    paper_order_id=paper_order_id,
                    cycle_id=cycle.cycle_id,
                    asset_id=item.asset_id,
                    instruction_kind=item.instruction_kind,
                    status="LOCAL_VETO",
                    acknowledged=False,
                    requested_notional_usd=risk.requested_notional_usd,
                    requested_base=requested_base,
                    filled_base=0.0,
                    fill_fraction=0.0,
                    average_fill_price=execution.average_fill_price,
                    fill_ids=(),
                    reason=execution.reason,
                ))
                continue

            if execution.status == "NO_FILL":
                outcomes.append(PaperOrderOutcome(
                    paper_order_id=paper_order_id,
                    cycle_id=cycle.cycle_id,
                    asset_id=item.asset_id,
                    instruction_kind=item.instruction_kind,
                    status="ACKNOWLEDGED_NO_FILL",
                    acknowledged=True,
                    requested_notional_usd=risk.requested_notional_usd,
                    requested_base=requested_base,
                    filled_base=0.0,
                    fill_fraction=0.0,
                    average_fill_price=None,
                    fill_ids=(),
                    reason=execution.reason,
                ))
                continue

            if execution.status not in ("FILLED", "PARTIAL_FILL"):
                raise PaperVenueConflictError(
                    f"unsupported Phase 46 execution status {execution.status}"
                )
            if (
                execution.filled_base <= 0
                or execution.average_fill_price is None
                or execution.venue_timestamp is None
            ):
                raise PaperVenueConflictError(
                    "filled execution receipt is missing quantity, price or venue timestamp"
                )

            fee = (
                execution.filled_base
                * execution.average_fill_price
                * self.config.fee_bps
                / 10_000.0
            )
            required_buy_cash = (
                execution.filled_base * execution.average_fill_price + fee
                if execution.side == "BUY"
                else 0.0
            )
            if (
                required_buy_cash > self.cash_usd + 1e-12
                and not risk.reduce_only
            ):
                outcomes.append(PaperOrderOutcome(
                    paper_order_id=paper_order_id,
                    cycle_id=cycle.cycle_id,
                    asset_id=item.asset_id,
                    instruction_kind=item.instruction_kind,
                    status="VENUE_REJECTED_BALANCE",
                    acknowledged=False,
                    requested_notional_usd=risk.requested_notional_usd,
                    requested_base=requested_base,
                    filled_base=0.0,
                    fill_fraction=0.0,
                    average_fill_price=None,
                    fill_ids=(),
                    reason="paper account cash cannot fund fill plus fees",
                ))
                continue

            fill_id = content_hash({
                "schema_version": PHASE61_SCHEMA_VERSION,
                "paper_order_id": paper_order_id,
                "asset_id": item.asset_id,
                "side": execution.side,
                "filled_base": execution.filled_base,
                "average_fill_price": execution.average_fill_price,
                "venue_timestamp": execution.venue_timestamp,
            })
            fill = PaperFill(
                fill_id=fill_id,
                paper_order_id=paper_order_id,
                cycle_id=cycle.cycle_id,
                asset_id=item.asset_id,
                side=execution.side,
                quantity_base=float(execution.filled_base),
                price=float(execution.average_fill_price),
                fee_quote=float(fee),
                timestamp=float(execution.venue_timestamp),
            )
            self._apply_fill(fill)
            cycle_fill_ids.append(fill_id)
            outcomes.append(PaperOrderOutcome(
                paper_order_id=paper_order_id,
                cycle_id=cycle.cycle_id,
                asset_id=item.asset_id,
                instruction_kind=item.instruction_kind,
                status=execution.status,
                acknowledged=True,
                requested_notional_usd=risk.requested_notional_usd,
                requested_base=requested_base,
                filled_base=float(execution.filled_base),
                fill_fraction=float(execution.fill_fraction),
                average_fill_price=float(execution.average_fill_price),
                fill_ids=(fill_id,),
                reason=execution.reason,
            ))

        self._state_version += 1
        receipt_payload = {
            "schema_version": PHASE61_SCHEMA_VERSION,
            "cycle_id": cycle.cycle_id,
            "cycle_hash": cycle_hash,
            "outcomes": [row.to_dict() for row in outcomes],
            "fill_ids": cycle_fill_ids,
            "cash_before_usd": cash_before,
            "cash_after_usd": self.cash_usd,
            "state_version_before": version_before,
            "state_version_after": self._state_version,
        }
        receipt = PaperCycleReceipt(
            cycle_id=cycle.cycle_id,
            cycle_hash=cycle_hash,
            outcomes=tuple(outcomes),
            fill_ids=tuple(cycle_fill_ids),
            cash_before_usd=float(cash_before),
            cash_after_usd=float(self.cash_usd),
            state_version_before=version_before,
            state_version_after=self._state_version,
            receipt_id=content_hash(receipt_payload),
        )
        self._cycle_hashes[cycle.cycle_id] = cycle_hash
        self._cycle_receipts[cycle.cycle_id] = receipt
        return receipt

    def generate_position_reports(
        self,
        tracked_assets: Sequence[str],
    ) -> dict[str, VenuePositionReport]:
        assets = tuple(sorted({str(asset).strip() for asset in tracked_assets if str(asset).strip()}))
        if not assets:
            raise ValueError("tracked_assets must not be empty")
        return {
            asset: VenuePositionReport(
                account_id=self.config.account_id,
                asset_id=asset,
                quantity=self.position(asset).quantity,
                avg_entry_price=self.position(asset).avg_entry_price,
                explicit=True,
                report_id=content_hash({
                    "schema_version": PHASE61_SCHEMA_VERSION,
                    "account_id": self.config.account_id,
                    "asset_id": asset,
                    "state_version": self._state_version,
                    "quantity": self.position(asset).quantity,
                    "avg_entry_price": self.position(asset).avg_entry_price,
                }),
            )
            for asset in assets
        }

    def reconcile_against_local(
        self,
        local_positions: Mapping[str, LocalPositionState],
        *,
        tracked_assets: Sequence[str],
        unresolved_command_ids: Sequence[str] = (),
        fill_ids: Sequence[str] | None = None,
        policy: ReconciliationBatchPolicy = ReconciliationBatchPolicy(),
        quantity_tolerance: float = 1e-9,
        entry_price_relative_tolerance: float = 1e-4,
    ) -> ReconciliationBatchReport:
        venue = self.generate_position_reports(tracked_assets)
        return reconcile_execution_state(
            local_positions,
            venue,
            account_id=self.config.account_id,
            tracked_assets=tracked_assets,
            unresolved_command_ids=unresolved_command_ids,
            fill_ids=(
                tuple(fill_ids)
                if fill_ids is not None
                else tuple(fill.fill_id for fill in self._fills.values())
            ),
            reports_complete=True,
            policy=policy,
            quantity_tolerance=quantity_tolerance,
            entry_price_relative_tolerance=entry_price_relative_tolerance,
            generate_missing_orders=False,
        )

    def mark_to_market(
        self,
        marks: Mapping[str, float],
    ) -> tuple[float, dict[str, float]]:
        market_values: dict[str, float] = {}
        for asset, position in self._positions.items():
            if abs(position.quantity) <= 1e-12:
                continue
            if asset not in marks:
                raise KeyError(f"missing mark price for open paper position {asset}")
            mark = float(marks[asset])
            if not math.isfinite(mark) or mark <= 0:
                raise ValueError(f"invalid mark price for {asset}")
            market_values[asset] = position.quantity * mark

        equity = self.cash_usd + sum(market_values.values())
        if not math.isfinite(equity) or equity <= 0:
            raise PaperVenueConflictError("paper account equity is non-positive")
        weights = {
            asset: value / equity
            for asset, value in market_values.items()
            if abs(value / equity) > 1e-15
        }
        return float(equity), weights

    def build_reconciled_state(
        self,
        reconciliation: ReconciliationBatchReport,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> ShadowAccountState:
        if not reconciliation.ready or not all(value for _, value in reconciliation.checks):
            raise PaperVenueConflictError(
                "Phase 50 reconciliation must pass before paper state can be committed"
            )
        if reconciliation.live_execution:
            raise PaperVenueConflictError("live reconciliation is outside the paper venue")
        tracked = tuple(sorted(set(reconciliation.tracked_assets)))
        open_assets = {
            asset
            for asset, position in self._positions.items()
            if abs(position.quantity) > 1e-12
        }
        missing = sorted(open_assets - set(tracked))
        if missing:
            raise PaperVenueConflictError(
                f"reconciliation does not cover open paper positions: {missing}"
            )

        equity, weights = self.mark_to_market(marks)
        # Conservative available cash: short-sale proceeds do not expand the
        # next cycle's spendable cash above current equity.
        available_cash = max(0.0, min(self.cash_usd, equity))
        return ShadowAccountState(
            account_id=self.config.account_id,
            observed_at=float(observed_at),
            equity_usd=equity,
            available_cash_usd=available_cash,
            position_weights=tuple(sorted(weights.items())),
            covered_assets=tracked,
            source_kind="RECONCILED_PAPER",
            source_ref=source_ref,
            reconciliation_hash=content_hash(reconciliation.to_dict()),
        )
