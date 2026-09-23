from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

PHASE50_SCHEMA_VERSION = "brian.phase50-execution-reconciliation.v1"
ReconciliationStatus = Literal[
    "MATCHED",
    "RECOVERY_REQUIRED",
    "UNRESOLVED",
    "NO_AUTHORITATIVE_POSITION_REPORT",
]


@dataclass(frozen=True, slots=True)
class LocalPositionState:
    account_id: str
    asset_id: str
    quantity: float
    avg_entry_price: float | None
    source_fill_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.account_id.strip() or not self.asset_id.strip():
            raise ValueError("local position identity is required")
        if not math.isfinite(self.quantity):
            raise ValueError("local quantity must be finite")
        if self.avg_entry_price is not None and (
            not math.isfinite(self.avg_entry_price) or self.avg_entry_price <= 0
        ):
            raise ValueError("local avg_entry_price must be positive when present")
        if abs(self.quantity) > 1e-12 and self.avg_entry_price is None:
            raise ValueError("open local position requires avg_entry_price")


@dataclass(frozen=True, slots=True)
class VenuePositionReport:
    account_id: str
    asset_id: str
    quantity: float
    avg_entry_price: float | None
    explicit: bool = True
    report_id: str = ""

    def __post_init__(self) -> None:
        if not self.account_id.strip() or not self.asset_id.strip():
            raise ValueError("venue position identity is required")
        if not math.isfinite(self.quantity):
            raise ValueError("venue quantity must be finite")
        if self.avg_entry_price is not None and (
            not math.isfinite(self.avg_entry_price) or self.avg_entry_price <= 0
        ):
            raise ValueError("venue avg_entry_price must be positive when present")
        if not self.explicit:
            raise ValueError("Phase 50 accepts only explicit authoritative position reports")


@dataclass(frozen=True, slots=True)
class SyntheticRecoveryEvent:
    account_id: str
    asset_id: str
    quantity_delta: float
    target_quantity: float
    synthetic_price: float
    reason: str
    establishes_realized_pnl: bool = False

    def __post_init__(self) -> None:
        if not self.account_id.strip() or not self.asset_id.strip() or not self.reason.strip():
            raise ValueError("recovery event identity/reason is required")
        if not all(math.isfinite(value) for value in (
            self.quantity_delta, self.target_quantity, self.synthetic_price
        )):
            raise ValueError("recovery values must be finite")
        if self.synthetic_price <= 0:
            raise ValueError("synthetic_price must be positive")
        if self.establishes_realized_pnl:
            raise ValueError("synthetic reconciliation must not invent historical realized PnL")


@dataclass(frozen=True, slots=True)
class PositionReconciliationResult:
    account_id: str
    asset_id: str
    status: ReconciliationStatus
    local_quantity: float
    venue_quantity: float | None
    quantity_difference: float | None
    local_avg_entry_price: float | None
    venue_avg_entry_price: float | None
    quantity_within_tolerance: bool
    entry_price_within_tolerance: bool | None
    recovery_events: tuple[SyntheticRecoveryEvent, ...]
    reasons: tuple[str, ...]
    authoritative_report_present: bool
    schema_version: str = PHASE50_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    @property
    def resolved(self) -> bool:
        return self.status == "MATCHED"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resolved"] = self.resolved
        return payload


def _price_matches(local: float | None, venue: float | None, tolerance: float) -> bool | None:
    if venue is None:
        return None
    if local is None:
        return False
    return math.isclose(local, venue, rel_tol=tolerance, abs_tol=0.0)


def reconcile_position(
    local: LocalPositionState | None,
    venue: VenuePositionReport | None,
    *,
    account_id: str,
    asset_id: str,
    quantity_tolerance: float = 1e-9,
    entry_price_relative_tolerance: float = 1e-4,
    generate_missing_orders: bool = True,
) -> PositionReconciliationResult:
    """Compare one cached position with one authoritative venue report.

    Clean-room behavior follows NautilusTrader's public reconciliation contract:
    an explicit open/flat report is authoritative; a missing report is *not*
    evidence of flat; quantity must reconcile within tolerance; and an explicit
    entry-average mismatch remains unresolved even when quantity matches.
    Synthetic recovery is proposed only when a safe price target exists.
    """
    if not account_id.strip() or not asset_id.strip():
        raise ValueError("account_id and asset_id are required")
    if not math.isfinite(quantity_tolerance) or quantity_tolerance < 0:
        raise ValueError("quantity_tolerance must be non-negative")
    if not math.isfinite(entry_price_relative_tolerance) or entry_price_relative_tolerance < 0:
        raise ValueError("entry_price_relative_tolerance must be non-negative")

    if local is not None and (local.account_id != account_id or local.asset_id != asset_id):
        raise ValueError("local position identity mismatch")
    if venue is not None and (venue.account_id != account_id or venue.asset_id != asset_id):
        raise ValueError("venue position identity mismatch")

    local_qty = 0.0 if local is None else float(local.quantity)
    local_avg = None if local is None else local.avg_entry_price

    if venue is None:
        return PositionReconciliationResult(
            account_id=account_id,
            asset_id=asset_id,
            status="NO_AUTHORITATIVE_POSITION_REPORT",
            local_quantity=local_qty,
            venue_quantity=None,
            quantity_difference=None,
            local_avg_entry_price=local_avg,
            venue_avg_entry_price=None,
            quantity_within_tolerance=False,
            entry_price_within_tolerance=None,
            recovery_events=(),
            reasons=("missing venue position report does not mean flat",),
            authoritative_report_present=False,
        )

    venue_qty = float(venue.quantity)
    venue_avg = venue.avg_entry_price
    difference = venue_qty - local_qty
    quantity_ok = abs(difference) <= quantity_tolerance

    # Flat reports are authoritative and need no entry average.
    if abs(venue_qty) <= quantity_tolerance:
        if quantity_ok:
            return PositionReconciliationResult(
                account_id, asset_id, "MATCHED", local_qty, venue_qty, difference,
                local_avg, venue_avg, True, None, (), (), True,
            )
        if not generate_missing_orders:
            return PositionReconciliationResult(
                account_id, asset_id, "UNRESOLVED", local_qty, venue_qty, difference,
                local_avg, venue_avg, False, None, (),
                ("authoritative flat report conflicts with local exposure",), True,
            )
        price = local_avg
        if price is None:
            return PositionReconciliationResult(
                account_id, asset_id, "UNRESOLVED", local_qty, venue_qty, difference,
                local_avg, venue_avg, False, None, (),
                ("cannot synthesize close without a local entry-price reference",), True,
            )
        event = SyntheticRecoveryEvent(
            account_id, asset_id, difference, venue_qty, price,
            "close local exposure to authoritative venue flat",
        )
        return PositionReconciliationResult(
            account_id, asset_id, "RECOVERY_REQUIRED", local_qty, venue_qty, difference,
            local_avg, venue_avg, False, None, (event,),
            ("apply synthetic recovery then reconcile again",), True,
        )

    # An open venue position cannot be materialized safely without an average
    # entry reference. Nautilus likewise refuses to invent this price.
    if venue_avg is None:
        return PositionReconciliationResult(
            account_id, asset_id, "UNRESOLVED", local_qty, venue_qty, difference,
            local_avg, venue_avg, quantity_ok, None, (),
            ("explicit open position report is missing avg_entry_price",), True,
        )

    price_ok = _price_matches(local_avg, venue_avg, entry_price_relative_tolerance)
    if quantity_ok and price_ok is True:
        return PositionReconciliationResult(
            account_id, asset_id, "MATCHED", local_qty, venue_qty, difference,
            local_avg, venue_avg, True, True, (), (), True,
        )

    if quantity_ok and price_ok is False:
        return PositionReconciliationResult(
            account_id, asset_id, "UNRESOLVED", local_qty, venue_qty, difference,
            local_avg, venue_avg, True, False, (),
            ("quantity matches but authoritative entry average does not",), True,
        )

    if not generate_missing_orders:
        return PositionReconciliationResult(
            account_id, asset_id, "UNRESOLVED", local_qty, venue_qty, difference,
            local_avg, venue_avg, False, price_ok, (),
            ("quantity mismatch and synthetic recovery is disabled",), True,
        )

    events: list[SyntheticRecoveryEvent] = []
    local_sign = 1 if local_qty > quantity_tolerance else -1 if local_qty < -quantity_tolerance else 0
    venue_sign = 1 if venue_qty > quantity_tolerance else -1

    if local_sign != 0 and local_sign != venue_sign:
        # Mirror the venue-reversal recovery shape: close cached exposure, then
        # open the authoritative side at its reported entry average.
        assert local_avg is not None
        events.append(SyntheticRecoveryEvent(
            account_id, asset_id, -local_qty, 0.0, local_avg,
            "close cached position before authoritative direction reversal",
        ))
        events.append(SyntheticRecoveryEvent(
            account_id, asset_id, venue_qty, venue_qty, venue_avg,
            "open authoritative reversed venue position",
        ))
    else:
        # Increasing/opening targets the venue average. Reductions preserve the
        # remaining position's entry average, so the local average is used when
        # available; otherwise the venue average supplies the opening price.
        reducing = local_sign != 0 and abs(venue_qty) < abs(local_qty)
        price = local_avg if reducing and local_avg is not None else venue_avg
        events.append(SyntheticRecoveryEvent(
            account_id, asset_id, difference, venue_qty, price,
            "align cached quantity to authoritative venue position",
        ))

    return PositionReconciliationResult(
        account_id, asset_id, "RECOVERY_REQUIRED", local_qty, venue_qty, difference,
        local_avg, venue_avg, False, price_ok, tuple(events),
        ("apply synthetic recovery then reconcile again",), True,
    )


@dataclass(frozen=True, slots=True)
class ReconciliationBatchPolicy:
    require_authoritative_report_for_tracked_assets: bool = True
    allow_incomplete_history_with_explicit_positions: bool = True


@dataclass(frozen=True, slots=True)
class ReconciliationBatchReport:
    results: tuple[PositionReconciliationResult, ...]
    tracked_assets: tuple[str, ...]
    reports_complete: bool
    unresolved_command_ids: tuple[str, ...]
    duplicate_fill_ids: tuple[str, ...]
    checks: tuple[tuple[str, bool], ...]
    ready: bool
    schema_version: str = PHASE50_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "results": [row.to_dict() for row in self.results],
            "tracked_assets": self.tracked_assets,
            "reports_complete": self.reports_complete,
            "unresolved_command_ids": self.unresolved_command_ids,
            "duplicate_fill_ids": self.duplicate_fill_ids,
            "checks": dict(self.checks),
            "ready": self.ready,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def reconcile_execution_state(
    local_positions: Mapping[str, LocalPositionState],
    venue_positions: Mapping[str, VenuePositionReport],
    *,
    account_id: str,
    tracked_assets: Sequence[str],
    unresolved_command_ids: Sequence[str] = (),
    fill_ids: Sequence[str] = (),
    reports_complete: bool,
    policy: ReconciliationBatchPolicy = ReconciliationBatchPolicy(),
    quantity_tolerance: float = 1e-9,
    entry_price_relative_tolerance: float = 1e-4,
    generate_missing_orders: bool = True,
) -> ReconciliationBatchReport:
    """Produce a fail-closed reconciliation readiness receipt.

    Duplicate fills are surfaced for investigation. Unknown command outcomes and
    unresolved authoritative position reports block readiness. Incomplete
    bounded history may remain acceptable when every tracked asset has an
    explicit authoritative position report, matching Nautilus' documented
    bounded-history safety model.
    """
    assets = tuple(sorted({str(asset).strip() for asset in tracked_assets if str(asset).strip()}))
    if not assets:
        raise ValueError("tracked_assets must not be empty")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for fill_id in fill_ids:
        value = str(fill_id)
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    results = tuple(
        reconcile_position(
            local_positions.get(asset),
            venue_positions.get(asset),
            account_id=account_id,
            asset_id=asset,
            quantity_tolerance=quantity_tolerance,
            entry_price_relative_tolerance=entry_price_relative_tolerance,
            generate_missing_orders=generate_missing_orders,
        )
        for asset in assets
    )

    authoritative_ok = all(row.authoritative_report_present for row in results)
    positions_resolved = all(row.resolved for row in results)
    history_ok = bool(reports_complete) or (
        policy.allow_incomplete_history_with_explicit_positions and authoritative_ok
    )
    unknown = tuple(sorted({str(value) for value in unresolved_command_ids if str(value)}))

    checks = (
        (
            "authoritative_reports_present",
            authoritative_ok if policy.require_authoritative_report_for_tracked_assets else True,
        ),
        ("positions_reconciled", positions_resolved),
        ("history_contract_acceptable", history_ok),
        ("no_unknown_command_outcomes", not unknown),
        ("no_duplicate_fill_ids", not duplicates),
    )
    ready = all(value for _, value in checks)
    return ReconciliationBatchReport(
        results=results,
        tracked_assets=assets,
        reports_complete=bool(reports_complete),
        unresolved_command_ids=unknown,
        duplicate_fill_ids=tuple(sorted(duplicates)),
        checks=checks,
        ready=ready,
    )
