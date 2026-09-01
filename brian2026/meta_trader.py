from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math

from .memory import EpisodicMemory
from .types import MarketSnapshot, SpecialistVote, Decision


@dataclass(slots=True)
class MetaConfig:
    min_confidence: float = 0.68
    min_consensus: float = 0.58
    min_abs_score: float = 0.12
    disagreement_wait: float = 0.55


class MetaTrader:
    """Combines specialists and explicitly allows abstention (WAIT)."""

    def __init__(self, memory: EpisodicMemory, cfg: MetaConfig | None = None,
                 base_weights: Dict[str, float] | None = None) -> None:
        self.memory = memory
        self.cfg = cfg or MetaConfig()
        self.base_weights = base_weights or {
            "trend": 1.15,
            "momentum": 0.95,
            "orderbook": 1.10,
            "breakout": 1.05,
            "mean_reversion": 0.90,
        }

    def _weight(self, name: str) -> float:
        base = float(self.base_weights.get(name, 1.0))
        return self.memory.specialist_weight(name, default=base)

    def decide(self, snapshot: MarketSnapshot, votes: list[SpecialistVote]) -> Decision:
        directional = [v for v in votes if v.action != "WAIT" and v.confidence > 0]
        if not directional:
            return Decision(snapshot.symbol, "WAIT", 0.5, 0.0, snapshot.regime, votes,
                            "all specialists abstained")

        weighted_sum = 0.0
        total_weight = 0.0
        buy_weight = 0.0
        sell_weight = 0.0
        for v in directional:
            w = self._weight(v.name) * max(0.05, float(v.confidence))
            signed = abs(float(v.edge)) if v.action == "BUY" else -abs(float(v.edge))
            weighted_sum += w * signed
            total_weight += w
            if v.action == "BUY":
                buy_weight += w
            else:
                sell_weight += w

        score = weighted_sum / max(total_weight, 1e-12)
        dominant = max(buy_weight, sell_weight) / max(total_weight, 1e-12)
        disagreement = 1.0 - abs(buy_weight - sell_weight) / max(total_weight, 1e-12)
        confidence = max(0.5, min(0.99, 0.5 + abs(score) * 0.42 + (dominant - 0.5) * 0.20))

        action = "BUY" if score > 0 else "SELL"
        reasons: list[str] = []
        if abs(score) < self.cfg.min_abs_score:
            reasons.append(f"edge too small {score:.3f}")
        if dominant < self.cfg.min_consensus:
            reasons.append(f"consensus too low {dominant:.2f}")
        if confidence < self.cfg.min_confidence:
            reasons.append(f"confidence too low {confidence:.2f}")
        if disagreement > self.cfg.disagreement_wait:
            reasons.append(f"specialists disagree {disagreement:.2f}")
        if reasons:
            action = "WAIT"

        if action == "WAIT":
            reason = "; ".join(reasons) or "abstain"
        else:
            leaders = sorted(directional, key=lambda v: abs(v.edge) * self._weight(v.name), reverse=True)[:3]
            reason = " + ".join(f"{v.name}:{v.action}/{v.confidence:.2f}" for v in leaders)

        return Decision(snapshot.symbol, action, confidence, score, snapshot.regime, votes, reason)
