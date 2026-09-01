from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .replay import ReplayAction, ReplayPoint, ReplayResult, ReplaySettings, replay


@dataclass(frozen=True, slots=True)
class DecisionVariant:
    name: str
    action: ReplayAction
    settings: ReplaySettings


def standard_variants(actual_action: ReplayAction, brian_action: ReplayAction,
                      base: ReplaySettings) -> tuple[DecisionVariant, ...]:
    return (
        DecisionVariant("actual_legacy", actual_action, base),
        DecisionVariant("brian", brian_action, base),
        DecisionVariant("wait", "WAIT", base),
        DecisionVariant("delay_1s", brian_action, replace(base, latency_ms=base.latency_ms + 1000)),
        DecisionVariant("tight_tp_sl", brian_action, replace(base, tp_pct=base.tp_pct * 0.8, sl_pct=base.sl_pct * 0.8)),
        DecisionVariant("wide_tp_sl", brian_action, replace(base, tp_pct=base.tp_pct * 1.2, sl_pct=base.sl_pct * 1.2)),
        DecisionVariant("size_half", brian_action, replace(base, position_size=base.position_size * 0.5)),
        DecisionVariant("size_larger", brian_action, replace(base, position_size=base.position_size * 1.25)),
    )


def compare(path: Sequence[ReplayPoint], variants: Sequence[DecisionVariant],
            decision_timestamp: float | None = None) -> dict[str, ReplayResult]:
    """Evaluate every alternative against the exact same path object."""
    return {
        variant.name: replay(path, variant.action, variant.settings,
                             decision_timestamp=decision_timestamp)
        for variant in variants
    }
