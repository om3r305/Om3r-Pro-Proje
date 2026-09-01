from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

from .memory import EpisodicMemory
from .meta_trader import MetaTrader, MetaConfig
from .risk_governor import RiskGovernor, RiskConfig
from .specialists import run_specialists
from .types import Decision, MarketSnapshot, TradeOutcome
from .researcher import Researcher
from .promotion_gate import PromotionGate, PromotionPolicy


class BrianEngine:
    """Shadow-first Brian orchestration.

    decide() never places an order.  It returns a reviewed decision that the
    legacy trader can log/compare.  Execution integration is a separate adapter.
    """

    def __init__(self, config: Dict[str, Any] | None = None,
                 runtime_root: str | Path = "runtime/brian2026") -> None:
        cfg = config or {}
        # Phase 1 is observation/research only. Configuration cannot grant
        # Brian execution authority or produce a non-shadow decision state.
        self.shadow_only = True
        self.memory = EpisodicMemory(runtime_root)
        self.meta = MetaTrader(
            self.memory,
            MetaConfig(**cfg.get("meta", {})),
            base_weights=cfg.get("specialist_weights"),
        )
        self.risk = RiskGovernor(RiskConfig(**cfg.get("risk", {})))
        self.researcher = Researcher(self.memory, runtime_root)
        self.promotion = PromotionGate(runtime_root, PromotionPolicy(**cfg.get("promotion", {})))
        self._decision_cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_json(cls, path: str | Path) -> "BrianEngine":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    def decide(self, snapshot: MarketSnapshot, account: Dict[str, Any] | None = None) -> Decision:
        votes = run_specialists(snapshot)
        decision = self.meta.decide(snapshot, votes)
        allowed, size_scale, risk_reason = self.risk.review(decision, snapshot, account)
        decision.allowed_by_risk = bool(allowed)
        decision.size_scale = float(size_scale) if allowed else 0.0
        decision.shadow_only = self.shadow_only
        if not allowed and decision.action != "WAIT":
            decision.reason = f"{decision.reason}; RISK:{risk_reason}"
        payload = {"snapshot": snapshot.to_dict(), "decision": decision.to_dict()}
        self.memory.append("decision", payload)
        self._decision_cache[decision.decision_id] = payload
        return decision

    def learn(self, outcome: TradeOutcome) -> None:
        cached = self._decision_cache.get(outcome.decision_id)
        if cached is None:
            # Recover from recent JSONL after restart.
            for rec in reversed(self.memory.tail(limit=1000, kind="decision")):
                p = rec.get("payload", {})
                if p.get("decision", {}).get("decision_id") == outcome.decision_id:
                    cached = p
                    break
        dec = (cached or {}).get("decision", {})
        snap = (cached or {}).get("snapshot", {})
        payload = {
            "decision_id": outcome.decision_id,
            "symbol": outcome.symbol,
            "regime": snap.get("regime", "UNKNOWN"),
            "action": dec.get("action", "UNKNOWN"),
            "confidence": dec.get("confidence", 0.0),
            "votes": dec.get("votes", []),
            **outcome.to_dict(),
        }
        self.memory.append("decision_outcome", payload)
        executed_action = str(outcome.metadata.get("executed_action", dec.get("action", "BUY")))
        self.memory.update_specialists(dec.get("votes", []), outcome.won, executed_action=executed_action)

    def research(self, min_cluster: int = 4):
        return self.researcher.propose(min_cluster=min_cluster)
