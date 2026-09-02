from __future__ import annotations

import math

from .market_structure import MarketStructureFeatures
from .types import SpecialistVote


def _wait(name: str, reason: str) -> SpecialistVote:
    return SpecialistVote(name, "WAIT", 0.0, 0.0, reason, {})


def _known_state(value: float | None) -> int | None:
    if value is None or not math.isfinite(float(value)):
        return None
    if value > 0.5:
        return 1
    if value < -0.5:
        return -1
    return 0


def market_structure_specialist(
    feature: MarketStructureFeatures,
    *,
    structure_15m: float | None = None,
    structure_1h: float | None = None,
) -> SpecialistVote:
    if feature.state == "UNKNOWN" or feature.atr is None:
        return _wait(
            "market_structure_specialist",
            "unavailable:confirmed_structure,atr",
        )

    bullish = bool(
        feature.bullish_bos
        or feature.bullish_choch
        or feature.bullish_sweep
        or feature.failed_breakdown
        or feature.bullish_breakout_retest
    )
    bearish = bool(
        feature.bearish_bos
        or feature.bearish_choch
        or feature.bearish_sweep
        or feature.failed_breakout
        or feature.bearish_breakout_retest
    )
    edge = (0.45 if bullish else 0.0) - (0.45 if bearish else 0.0)
    edge += 0.20 if feature.state == "UPTREND" else -0.20 if feature.state == "DOWNTREND" else 0.0

    m15 = _known_state(structure_15m)
    h1 = _known_state(structure_1h)
    if m15 is not None and h1 is not None:
        if m15 == h1 == 1:
            edge += 0.15
        elif m15 == h1 == -1:
            edge -= 0.15
        elif m15 * h1 == -1:
            edge *= 0.65

    edge = max(-1.0, min(1.0, edge))
    action = "BUY" if edge > 0.25 else "SELL" if edge < -0.25 else "WAIT"
    used = {
        "bullish_bos": float(feature.bullish_bos),
        "bearish_bos": float(feature.bearish_bos),
        "bullish_choch": float(feature.bullish_choch),
        "bearish_choch": float(feature.bearish_choch),
        "bullish_sweep": float(feature.bullish_sweep),
        "bearish_sweep": float(feature.bearish_sweep),
        "failed_breakdown": float(feature.failed_breakdown),
        "failed_breakout": float(feature.failed_breakout),
        "structure_15m": float(structure_15m) if structure_15m is not None and math.isfinite(float(structure_15m)) else 0.0,
        "structure_1h": float(structure_1h) if structure_1h is not None and math.isfinite(float(structure_1h)) else 0.0,
    }
    confidence = 0.0 if action == "WAIT" else min(1.0, 0.5 + 0.5 * abs(edge))
    return SpecialistVote(
        "market_structure_specialist",
        action,
        confidence,
        edge,
        f"confirmed structure={feature.state}; completed 15m/1h context; candidate evidence only",
        used,
    )


def dip_specialist(
    feature: MarketStructureFeatures,
    *,
    dip_score: float | None = None,
    rally_score: float | None = None,
    structure_15m: float | None = None,
    structure_1h: float | None = None,
) -> SpecialistVote:
    dip = feature.dip_score if dip_score is None else dip_score
    rally = feature.rally_score if rally_score is None else rally_score
    if dip is None or rally is None or not math.isfinite(float(dip)) or not math.isfinite(float(rally)):
        return _wait("dip_specialist", "unavailable:dip_or_rally_evidence")

    edge = float(dip) - float(rally)
    m15 = _known_state(structure_15m)
    h1 = _known_state(structure_1h)
    if m15 is not None and h1 is not None and m15 * h1 == -1:
        edge *= 0.70

    edge = max(-1.0, min(1.0, edge))
    action = "BUY" if edge >= 0.25 else "SELL" if edge <= -0.25 else "WAIT"
    confidence = 0.0 if action == "WAIT" else min(1.0, 0.5 + 0.5 * abs(edge))
    return SpecialistVote(
        "dip_specialist",
        action,
        confidence,
        edge,
        "deterministic dip/rally quality with completed higher-timeframe context; profitability not assumed",
        {
            "dip_score": float(dip),
            "rally_score": float(rally),
            "structure_15m": float(structure_15m) if structure_15m is not None and math.isfinite(float(structure_15m)) else 0.0,
            "structure_1h": float(structure_1h) if structure_1h is not None and math.isfinite(float(structure_1h)) else 0.0,
        },
    )


PHASE27_SPECIALISTS = (market_structure_specialist, dip_specialist)
