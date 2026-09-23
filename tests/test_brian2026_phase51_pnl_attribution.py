from __future__ import annotations

import pytest

from brian2026.phase45_execution_contract import TradeIntent
from brian2026.phase51_pnl_attribution import (
    attribute_closed_trade,
    build_attribution_report,
)


def _intent(
    *,
    intent_id: str,
    asset: str,
    direction: int,
    weight: float,
    evidence: tuple[str, ...],
) -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        asset_id=asset,
        direction=direction,
        target_weight=weight,
        expected_edge_bps=25.0,
        confidence=0.75,
        max_slippage_bps=8.0,
        created_at=1_760_000_000.0,
        ttl_seconds=60,
        evidence_ids=evidence,
    )


def test_long_trade_decomposes_market_move_execution_and_fees_exactly() -> None:
    trade = attribute_closed_trade(
        _intent(
            intent_id="btc-1",
            asset="BTCUSDT",
            direction=1,
            weight=0.20,
            evidence=("ev-news", "ev-structure"),
        ),
        quantity_base=2.0,
        reference_entry_price=100.0,
        actual_entry_price=101.0,
        exit_price=110.0,
        fees_quote=1.5,
        requested_weight_before_risk=0.30,
        regime="ALIGNED_UPTREND",
        analyst_ids=("news_analyst", "market_analyst"),
        close_type="TAKE_PROFIT",
    )
    assert trade.market_move_pnl_quote == pytest.approx(20.0)
    assert trade.entry_execution_pnl_quote == pytest.approx(-2.0)
    assert trade.gross_pnl_quote == pytest.approx(18.0)
    assert trade.net_pnl_quote == pytest.approx(16.5)
    assert trade.released_weight_to_cash == pytest.approx(0.10)


def test_short_trade_decomposition_is_sign_correct() -> None:
    trade = attribute_closed_trade(
        _intent(
            intent_id="eth-1",
            asset="ETHUSDT",
            direction=-1,
            weight=-0.15,
            evidence=("ev-deriv",),
        ),
        quantity_base=3.0,
        reference_entry_price=100.0,
        actual_entry_price=99.0,
        exit_price=90.0,
        fees_quote=0.5,
        requested_weight_before_risk=-0.25,
        regime="ALIGNED_DOWNTREND",
        analyst_ids=("derivatives_analyst",),
        close_type="TIME_LIMIT",
    )
    assert trade.market_move_pnl_quote == pytest.approx(30.0)
    # Entering a short below the reference is adverse by 1 quote unit per base.
    assert trade.entry_execution_pnl_quote == pytest.approx(-3.0)
    assert trade.gross_pnl_quote == pytest.approx(27.0)
    assert trade.net_pnl_quote == pytest.approx(26.5)


def test_favorable_execution_is_preserved_as_positive_execution_component() -> None:
    trade = attribute_closed_trade(
        _intent(
            intent_id="btc-2",
            asset="BTCUSDT",
            direction=1,
            weight=0.1,
            evidence=("ev-market",),
        ),
        quantity_base=1.0,
        reference_entry_price=100.0,
        actual_entry_price=99.0,
        exit_price=105.0,
        fees_quote=0.2,
        requested_weight_before_risk=0.1,
        regime="RANGE",
        analyst_ids=("market_analyst",),
        close_type="EARLY_STOP",
    )
    assert trade.entry_execution_pnl_quote == pytest.approx(1.0)
    assert trade.gross_pnl_quote == pytest.approx(6.0)
    assert trade.net_pnl_quote == pytest.approx(5.8)


def test_report_reconciles_totals_and_keeps_associations_noncausal() -> None:
    t1 = attribute_closed_trade(
        _intent(
            intent_id="btc-3",
            asset="BTCUSDT",
            direction=1,
            weight=0.2,
            evidence=("ev-shared", "ev-btc"),
        ),
        quantity_base=1.0,
        reference_entry_price=100.0,
        actual_entry_price=101.0,
        exit_price=110.0,
        fees_quote=1.0,
        requested_weight_before_risk=0.3,
        regime="ALIGNED_UPTREND",
        analyst_ids=("market_analyst", "news_analyst"),
        close_type="TAKE_PROFIT",
    )
    t2 = attribute_closed_trade(
        _intent(
            intent_id="eth-3",
            asset="ETHUSDT",
            direction=-1,
            weight=-0.1,
            evidence=("ev-shared", "ev-eth"),
        ),
        quantity_base=1.0,
        reference_entry_price=200.0,
        actual_entry_price=201.0,
        exit_price=210.0,
        fees_quote=0.5,
        requested_weight_before_risk=-0.2,
        regime="HTF_CONFLICT",
        analyst_ids=("market_analyst", "derivatives_analyst"),
        close_type="STOP_LOSS",
    )
    report = build_attribution_report((t1, t2))
    assert report.total_market_move_pnl_quote + report.total_entry_execution_pnl_quote == pytest.approx(
        report.total_gross_pnl_quote
    )
    assert report.total_gross_pnl_quote - report.total_fees_quote == pytest.approx(
        report.total_net_pnl_quote
    )
    assert report.close_type_counts == {"STOP_LOSS": 1, "TAKE_PROFIT": 1}
    assert report.by_analyst["market_analyst"].observations == 2
    assert report.by_evidence["ev-shared"].observations == 2
    assert report.by_evidence["ev-shared"].association_is_not_causal_allocation is True
    assert report.evidence_associations_are_non_additive is True


def test_evidence_association_is_not_arbitrarily_split_into_pnl_shares() -> None:
    trade = attribute_closed_trade(
        _intent(
            intent_id="multi-evidence",
            asset="SOLUSDT",
            direction=1,
            weight=0.1,
            evidence=("ev-a", "ev-b", "ev-c"),
        ),
        quantity_base=1.0,
        reference_entry_price=100.0,
        actual_entry_price=100.0,
        exit_price=110.0,
        fees_quote=0.0,
        requested_weight_before_risk=0.1,
        regime="MIXED_TRANSITION",
        analyst_ids=("a", "b"),
        close_type="COMPLETED",
    )
    report = build_attribution_report((trade,))
    # Each evidence row is associated with the trade outcome for later statistics.
    # The values are deliberately non-additive: Brian never claims each evidence
    # caused one-third of the dollars.
    assert report.by_evidence["ev-a"].associated_net_pnl_quote == pytest.approx(10.0)
    assert report.by_evidence["ev-b"].associated_net_pnl_quote == pytest.approx(10.0)
    assert report.by_evidence["ev-c"].associated_net_pnl_quote == pytest.approx(10.0)
    assert report.total_net_pnl_quote == pytest.approx(10.0)


def test_duplicate_intent_ids_are_rejected() -> None:
    intent = _intent(
        intent_id="dup",
        asset="BTCUSDT",
        direction=1,
        weight=0.1,
        evidence=("ev",),
    )
    trade = attribute_closed_trade(
        intent,
        quantity_base=1.0,
        reference_entry_price=100.0,
        actual_entry_price=100.0,
        exit_price=101.0,
        fees_quote=0.0,
        requested_weight_before_risk=0.1,
        regime="RANGE",
        analyst_ids=("market",),
        close_type="COMPLETED",
    )
    with pytest.raises(ValueError, match="unique intent ids"):
        build_attribution_report((trade, trade))
