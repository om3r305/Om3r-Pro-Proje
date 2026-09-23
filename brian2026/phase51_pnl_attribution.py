from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import math

from .phase45_execution_contract import TradeIntent

PHASE51_SCHEMA_VERSION = "brian.phase51-pnl-attribution.v1"


@dataclass(frozen=True, slots=True)
class TradeAttribution:
    intent_id: str
    asset_id: str
    direction: int
    quantity_base: float
    reference_entry_price: float
    actual_entry_price: float
    exit_price: float
    fees_quote: float
    market_move_pnl_quote: float
    entry_execution_pnl_quote: float
    gross_pnl_quote: float
    net_pnl_quote: float
    traded_notional_quote: float
    requested_weight_before_risk: float
    final_weight_after_risk: float
    released_weight_to_cash: float
    regime: str
    analyst_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    close_type: str
    schema_version: str = PHASE51_SCHEMA_VERSION
    association_is_not_causal_allocation: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.intent_id.strip() or not self.asset_id.strip() or not self.regime.strip() or not self.close_type.strip():
            raise ValueError("attribution identity, regime and close_type are required")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        for label, value in (
            ("quantity_base", self.quantity_base),
            ("reference_entry_price", self.reference_entry_price),
            ("actual_entry_price", self.actual_entry_price),
            ("exit_price", self.exit_price),
            ("fees_quote", self.fees_quote),
            ("traded_notional_quote", self.traded_notional_quote),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.quantity_base <= 0:
            raise ValueError("quantity_base must be positive")
        if min(self.reference_entry_price, self.actual_entry_price, self.exit_price) <= 0:
            raise ValueError("prices must be positive")
        if self.fees_quote < 0:
            raise ValueError("fees_quote must be non-negative")
        if not self.evidence_ids:
            raise ValueError("attribution requires evidence lineage")
        if not math.isclose(
            self.market_move_pnl_quote + self.entry_execution_pnl_quote,
            self.gross_pnl_quote,
            rel_tol=1e-12,
            abs_tol=1e-10,
        ):
            raise ValueError("gross PnL decomposition does not reconcile")
        if not math.isclose(
            self.gross_pnl_quote - self.fees_quote,
            self.net_pnl_quote,
            rel_tol=1e-12,
            abs_tol=1e-10,
        ):
            raise ValueError("net PnL does not reconcile gross minus fees")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def attribute_closed_trade(
    intent: TradeIntent,
    *,
    quantity_base: float,
    actual_entry_price: float,
    exit_price: float,
    fees_quote: float,
    reference_entry_price: float,
    requested_weight_before_risk: float,
    regime: str,
    analyst_ids: Sequence[str],
    close_type: str,
) -> TradeAttribution:
    """Decompose realized PnL without pretending evidence caused a fixed share.

    Exact accounting identity:
      actual gross PnL
        = market move from the decision reference
        + entry execution effect.

    Fees are then deducted separately. Evidence/analyst ids remain lineage
    associations only; Brian never divides dollars among evidence rows by an
    arbitrary weighting rule.
    """
    if not intent.shadow_only or intent.live_execution:
        raise ValueError("Phase 51 accepts Brian shadow intents only")
    quantity = float(quantity_base)
    reference = float(reference_entry_price)
    actual = float(actual_entry_price)
    exit_px = float(exit_price)
    fees = float(fees_quote)
    if not all(math.isfinite(value) for value in (quantity, reference, actual, exit_px, fees)):
        raise ValueError("trade attribution values must be finite")
    if quantity <= 0 or min(reference, actual, exit_px) <= 0 or fees < 0:
        raise ValueError("trade attribution values are out of range")

    direction = intent.direction
    market_move = direction * (exit_px - reference) * quantity
    execution = direction * (reference - actual) * quantity
    gross = direction * (exit_px - actual) * quantity
    net = gross - fees
    requested_weight = float(requested_weight_before_risk)
    final_weight = float(intent.target_weight)
    if not math.isfinite(requested_weight):
        raise ValueError("requested_weight_before_risk must be finite")
    if requested_weight != 0 and (1 if requested_weight > 0 else -1) != direction:
        raise ValueError("requested pre-risk weight sign must match intent direction")
    released = max(0.0, abs(requested_weight) - abs(final_weight))

    return TradeAttribution(
        intent_id=intent.intent_id,
        asset_id=intent.asset_id,
        direction=direction,
        quantity_base=quantity,
        reference_entry_price=reference,
        actual_entry_price=actual,
        exit_price=exit_px,
        fees_quote=fees,
        market_move_pnl_quote=market_move,
        entry_execution_pnl_quote=execution,
        gross_pnl_quote=gross,
        net_pnl_quote=net,
        traded_notional_quote=quantity * actual,
        requested_weight_before_risk=requested_weight,
        final_weight_after_risk=final_weight,
        released_weight_to_cash=released,
        regime=str(regime),
        analyst_ids=tuple(sorted({str(value) for value in analyst_ids if str(value)})),
        evidence_ids=tuple(sorted(set(intent.evidence_ids))),
        close_type=str(close_type),
    )


@dataclass(frozen=True, slots=True)
class AssociationSummary:
    observations: int
    wins: int
    losses: int
    associated_net_pnl_quote: float
    associated_gross_pnl_quote: float
    associated_fees_quote: float
    association_is_not_causal_allocation: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _association_summary(rows: Sequence[TradeAttribution]) -> AssociationSummary:
    return AssociationSummary(
        observations=len(rows),
        wins=sum(row.net_pnl_quote > 0 for row in rows),
        losses=sum(row.net_pnl_quote < 0 for row in rows),
        associated_net_pnl_quote=sum(row.net_pnl_quote for row in rows),
        associated_gross_pnl_quote=sum(row.gross_pnl_quote for row in rows),
        associated_fees_quote=sum(row.fees_quote for row in rows),
    )


@dataclass(frozen=True, slots=True)
class AttributionReport:
    trades: int
    total_market_move_pnl_quote: float
    total_entry_execution_pnl_quote: float
    total_gross_pnl_quote: float
    total_fees_quote: float
    total_net_pnl_quote: float
    total_traded_notional_quote: float
    released_weight_to_cash: float
    close_type_counts: Mapping[str, int]
    by_regime: Mapping[str, AssociationSummary]
    by_analyst: Mapping[str, AssociationSummary]
    by_evidence: Mapping[str, AssociationSummary]
    schema_version: str = PHASE51_SCHEMA_VERSION
    evidence_associations_are_non_additive: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trades": self.trades,
            "total_market_move_pnl_quote": self.total_market_move_pnl_quote,
            "total_entry_execution_pnl_quote": self.total_entry_execution_pnl_quote,
            "total_gross_pnl_quote": self.total_gross_pnl_quote,
            "total_fees_quote": self.total_fees_quote,
            "total_net_pnl_quote": self.total_net_pnl_quote,
            "total_traded_notional_quote": self.total_traded_notional_quote,
            "released_weight_to_cash": self.released_weight_to_cash,
            "close_type_counts": dict(self.close_type_counts),
            "by_regime": {key: value.to_dict() for key, value in self.by_regime.items()},
            "by_analyst": {key: value.to_dict() for key, value in self.by_analyst.items()},
            "by_evidence": {key: value.to_dict() for key, value in self.by_evidence.items()},
            "evidence_associations_are_non_additive": self.evidence_associations_are_non_additive,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def build_attribution_report(
    trades: Sequence[TradeAttribution],
) -> AttributionReport:
    rows = tuple(trades)
    if not rows:
        raise ValueError("attribution report requires trades")
    ids = [row.intent_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("attribution report requires unique intent ids")

    by_regime_rows: dict[str, list[TradeAttribution]] = {}
    by_analyst_rows: dict[str, list[TradeAttribution]] = {}
    by_evidence_rows: dict[str, list[TradeAttribution]] = {}
    close_counts: dict[str, int] = {}

    for row in rows:
        by_regime_rows.setdefault(row.regime, []).append(row)
        for analyst in row.analyst_ids:
            by_analyst_rows.setdefault(analyst, []).append(row)
        for evidence_id in row.evidence_ids:
            by_evidence_rows.setdefault(evidence_id, []).append(row)
        close_counts[row.close_type] = close_counts.get(row.close_type, 0) + 1

    market_move = sum(row.market_move_pnl_quote for row in rows)
    execution = sum(row.entry_execution_pnl_quote for row in rows)
    gross = sum(row.gross_pnl_quote for row in rows)
    fees = sum(row.fees_quote for row in rows)
    net = sum(row.net_pnl_quote for row in rows)
    if not math.isclose(market_move + execution, gross, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("portfolio attribution gross decomposition failed")
    if not math.isclose(gross - fees, net, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("portfolio attribution net decomposition failed")

    return AttributionReport(
        trades=len(rows),
        total_market_move_pnl_quote=market_move,
        total_entry_execution_pnl_quote=execution,
        total_gross_pnl_quote=gross,
        total_fees_quote=fees,
        total_net_pnl_quote=net,
        total_traded_notional_quote=sum(row.traded_notional_quote for row in rows),
        released_weight_to_cash=sum(row.released_weight_to_cash for row in rows),
        close_type_counts=dict(sorted(close_counts.items())),
        by_regime={
            key: _association_summary(value)
            for key, value in sorted(by_regime_rows.items())
        },
        by_analyst={
            key: _association_summary(value)
            for key, value in sorted(by_analyst_rows.items())
        },
        by_evidence={
            key: _association_summary(value)
            for key, value in sorted(by_evidence_rows.items())
        },
    )
