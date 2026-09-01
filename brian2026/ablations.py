from __future__ import annotations

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price_returns": ("price", "return_5", "acceleration"),
    "technical": ("ema_fast", "ema_slow", "ema_slope_pct", "rsi", "zscore", "bb_position", "atr_pct", "volume_z", "breakout_score", "recent_high"),
    "order_book": ("spread_bps", "book_imbalance", "wall_score"),
    "regime": ("regime_code",),
    "legacy_predictor": ("legacy_predictor_confidence", "legacy_signal_fired"),
}


def feature_names_for(group: str, available: tuple[str, ...]) -> tuple[str, ...]:
    if group == "combined":
        return tuple(sorted(available))
    if group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group: {group}")
    wanted = FEATURE_GROUPS[group]
    return tuple(name for name in wanted if name in available)