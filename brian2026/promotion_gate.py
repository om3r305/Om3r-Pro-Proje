from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import json
import time

from .metrics import PerformanceMetrics


@dataclass(slots=True)
class PromotionPolicy:
    min_trades: int = 80
    min_profit_factor: float = 1.15
    min_expectancy: float = 0.0
    max_drawdown_multiplier_vs_incumbent: float = 1.10
    min_walk_forward_pass_ratio: float = 0.75
    auto_apply_live: bool = False


class PromotionGate:
    """Candidate may become a champion, but never live code automatically in v1."""

    def __init__(self, root: str | Path = "runtime/brian2026", policy: PromotionPolicy | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy or PromotionPolicy()
        self.journal = self.root / "promotions.jsonl"

    def review(self, candidate: PerformanceMetrics, incumbent: PerformanceMetrics | None = None,
               walk_forward_pass_ratio: float = 1.0, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        p = self.policy
        reasons: list[str] = []
        if candidate.trades < p.min_trades:
            reasons.append(f"need {p.min_trades} trades, got {candidate.trades}")
        if candidate.profit_factor < p.min_profit_factor:
            reasons.append(f"PF {candidate.profit_factor:.2f} < {p.min_profit_factor:.2f}")
        if candidate.expectancy <= p.min_expectancy:
            reasons.append(f"expectancy {candidate.expectancy:.4f} <= {p.min_expectancy:.4f}")
        if walk_forward_pass_ratio < p.min_walk_forward_pass_ratio:
            reasons.append("walk-forward stability failed")
        if incumbent is not None:
            allowed_dd = max(1e-9, incumbent.max_drawdown * p.max_drawdown_multiplier_vs_incumbent)
            if candidate.max_drawdown > allowed_dd and incumbent.max_drawdown > 0:
                reasons.append("drawdown worse than incumbent tolerance")
            if candidate.expectancy <= incumbent.expectancy:
                reasons.append("expectancy does not beat incumbent")

        status = "CHAMPION_CANDIDATE" if not reasons else "REJECTED"
        rec = {
            "ts": time.time(),
            "status": status,
            "live_applied": False,  # hard rule for foundation phase
            "reasons": reasons,
            "candidate": candidate.to_dict(),
            "incumbent": incumbent.to_dict() if incumbent else None,
            "walk_forward_pass_ratio": walk_forward_pass_ratio,
            "metadata": metadata or {},
        }
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
