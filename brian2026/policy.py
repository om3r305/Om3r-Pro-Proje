from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .learning import ProbabilityPrediction

PolicyAction = Literal["BUY", "SELL", "WAIT"]


@dataclass(frozen=True, slots=True)
class PolicyThresholds:
    buy: float = 0.60
    sell: float = 0.60
    min_margin: float = 0.10


def decide(probability: ProbabilityPrediction, thresholds: PolicyThresholds = PolicyThresholds()) -> PolicyAction:
    if probability.up >= thresholds.buy and probability.up - probability.down >= thresholds.min_margin:
        return "BUY"
    if probability.down >= thresholds.sell and probability.down - probability.up >= thresholds.min_margin:
        return "SELL"
    return "WAIT"


def select_thresholds(probabilities: Sequence[ProbabilityPrediction], labels: Sequence[int],
                      candidates: Sequence[float] = (0.55, 0.60, 0.65, 0.70),
                      *, partition: str = "validation") -> PolicyThresholds:
    if partition != "validation":
        raise ValueError("thresholds may be selected only on validation")
    best = PolicyThresholds(candidates[0], candidates[0])
    best_score = (-1.0, -1.0)
    for threshold in candidates:
        cfg = PolicyThresholds(threshold, threshold)
        actions = [decide(p, cfg) for p in probabilities]
        acted = [(a, y) for a, y in zip(actions, labels) if a != "WAIT"]
        accuracy = sum((a == "BUY" and y == 1) or (a == "SELL" and y == -1) for a, y in acted) / len(acted) if acted else 0.0
        coverage = len(acted) / len(actions) if actions else 0.0
        score = (accuracy, coverage)
        if score > best_score:
            best, best_score = cfg, score
    return best