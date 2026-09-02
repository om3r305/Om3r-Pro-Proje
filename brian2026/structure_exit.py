from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

from .market_structure import MarketStructureFeatures
from .portfolio import (
    Action,
    PortfolioBar,
    PortfolioConfig,
    PortfolioResult,
    Side,
    StatefulPortfolioSimulator,
    simulate_portfolio,
)

ExitMode = Literal["fixed", "structure", "hybrid"]


@dataclass(frozen=True, slots=True)
class StructureExitConfig:
    mode: ExitMode = "fixed"
    exit_on_choch: bool = True
    exit_on_bos: bool = True
    exit_on_failed_break: bool = True
    exit_on_momentum_deterioration: bool = True
    trail_confirmed_structure: bool = True
    trail_buffer_atr: float = 0.15

    def __post_init__(self) -> None:
        if self.trail_buffer_atr < 0:
            raise ValueError("trail buffer must be non-negative")


def _structure_exit_reason(
    side: Side,
    bar: PortfolioBar,
    feature: MarketStructureFeatures,
    config: StructureExitConfig,
) -> str | None:
    """Return a causal exit reason for the *actual* open portfolio side.

    Long exits use bearish evidence; short exits use bullish evidence.  A
    failed breakout is bearish (resistance rejection), while a failed
    breakdown is bullish (support reclaim).  This directional mapping avoids
    the inverted semantics present in the first Phase 2.7 draft.
    """
    if side == "LONG":
        if config.exit_on_choch and feature.bearish_choch:
            return "STRUCTURE_BEARISH_CHOCH"
        if config.exit_on_bos and feature.bearish_bos:
            return "STRUCTURE_BEARISH_BOS"
        if config.exit_on_failed_break and feature.failed_breakout:
            return "STRUCTURE_FAILED_BREAKOUT"
        if (
            config.trail_confirmed_structure
            and feature.latest_swing_low is not None
            and feature.atr is not None
            and bar.close
            < feature.latest_swing_low - config.trail_buffer_atr * feature.atr
        ):
            return "STRUCTURE_TRAILING_HL_FAILURE"
        if (
            config.exit_on_momentum_deterioration
            and feature.momentum_deceleration is True
            and feature.acceleration is not None
            and feature.acceleration < 0
        ):
            return "STRUCTURE_MOMENTUM_DETERIORATION"
        return None

    if config.exit_on_choch and feature.bullish_choch:
        return "STRUCTURE_BULLISH_CHOCH"
    if config.exit_on_bos and feature.bullish_bos:
        return "STRUCTURE_BULLISH_BOS"
    if config.exit_on_failed_break and feature.failed_breakdown:
        return "STRUCTURE_FAILED_BREAKDOWN"
    if (
        config.trail_confirmed_structure
        and feature.latest_swing_high is not None
        and feature.atr is not None
        and bar.close
        > feature.latest_swing_high + config.trail_buffer_atr * feature.atr
    ):
        return "STRUCTURE_TRAILING_LH_FAILURE"
    if (
        config.exit_on_momentum_deterioration
        and feature.momentum_deceleration is True
        and feature.acceleration is not None
        and feature.acceleration > 0
    ):
        return "STRUCTURE_MOMENTUM_DETERIORATION"
    return None


def apply_structure_exit(
    actions: Sequence[str],
    features: Sequence[MarketStructureFeatures],
    config: StructureExitConfig = StructureExitConfig(),
) -> tuple[str, ...]:
    """Compatibility helper for the fixed mode only.

    Non-fixed structure exits must be evaluated inside the portfolio state
    machine so TP/SL/max-hold/cooldown state remains authoritative.  Refusing
    signal-only overlays prevents phantom exits after lifecycle closures.
    """
    if len(actions) != len(features):
        raise ValueError("actions/features must align")
    if config.mode != "fixed":
        raise ValueError(
            "stateful structure exits require simulate_structure_aware_portfolio"
        )
    return tuple(actions)


def simulate_structure_aware_portfolio(
    bars: Sequence[PortfolioBar],
    actions: Sequence[Action],
    features: Sequence[MarketStructureFeatures],
    structure_config: StructureExitConfig,
    portfolio_config: PortfolioConfig,
) -> PortfolioResult:
    """Run structure exits against the portfolio's real position lifecycle.

    Lifecycle exits retain priority on each bar.  `structure` mode keeps the
    hard stop and max-hold but disables the fixed profit target; `hybrid` keeps
    the original fixed TP/SL/max-hold and adds causal structure exits.
    """
    if not bars or len(bars) != len(actions) or len(bars) != len(features):
        raise ValueError("bars/actions/features must be non-empty and aligned")
    if structure_config.mode == "fixed":
        return simulate_portfolio(bars, actions, portfolio_config)

    config = (
        replace(portfolio_config, take_profit_pct=1_000_000.0)
        if structure_config.mode == "structure"
        else portfolio_config
    )
    simulator = StatefulPortfolioSimulator(config)
    for bar, action, feature in zip(bars, actions, features):
        reason = (
            _structure_exit_reason(
                simulator.position.side, bar, feature, structure_config
            )
            if simulator.position is not None
            else None
        )
        simulator.step(bar, action, forced_exit_reason=reason)
    return simulator.finish(bars[-1])
