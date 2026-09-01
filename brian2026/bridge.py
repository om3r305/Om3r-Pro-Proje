from __future__ import annotations

from typing import Any, Dict
import time

from .engine import BrianEngine
from .features import FeatureSnapshot
from .types import MarketSnapshot, TradeOutcome


class LegacyShadowBridge:
    """Adapter for the old Brian/Coins Monster loop.

    review() is shadow-only and never changes the legacy execution decision.
    mark_open()/mark_close() link the legacy trade outcome back to the shadow
    decision so specialist reliability and episodic memory can learn from it.
    """

    def __init__(self, engine: BrianEngine) -> None:
        self.engine = engine
        self._pending_reviews: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._open: Dict[tuple[str, str], Dict[str, Any]] = {}

    def review(self, *, symbol: str, price: float, regime: str,
               legacy_slot: str, legacy_confidence: float,
               features: Dict[str, float] | None = None,
               account: Dict[str, Any] | None = None) -> Dict[str, Any]:
        f = dict(features or {})
        f["legacy_confidence"] = float(legacy_confidence)
        snapshot = MarketSnapshot(
            symbol=symbol,
            price=float(price),
            regime=str(regime),
            features=f,
            context={"legacy_slot": legacy_slot},
        )
        d = self.engine.decide(snapshot, account=account)
        out = d.to_dict()
        self._pending_reviews[(symbol, legacy_slot)] = out
        return out

    def review_snapshot(self, snapshot: FeatureSnapshot,
                        account: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Review a typed point-in-time snapshot without affecting execution."""
        decision = self.engine.decide(snapshot.to_market(), account=account)
        out = decision.to_dict()
        slot = snapshot.legacy_slot or "unknown"
        self._pending_reviews[(snapshot.symbol, slot)] = out
        return out

    def mark_open(self, symbol: str, slot: str, entry_price: float, ts: float | None = None,
                  quantity: float = 0.0) -> str | None:
        key = (symbol, slot)
        review = self._pending_reviews.pop(key, None)
        if not review:
            return None
        self._open[key] = {
            "decision_id": review["decision_id"],
            "entry_price": float(entry_price),
            "open_ts": float(ts or time.time()),
            "shadow_action": review.get("action", "WAIT"),
            "shadow_confidence": float(review.get("confidence", 0.0)),
        }
        self.engine.account.open_position(
            f"{symbol}:{slot}", symbol, "LONG", float(entry_price),
            max(0.0, float(quantity)), float(ts or time.time()),
        )
        self.engine.memory.append("legacy_open_link", {"symbol": symbol, "slot": slot, **self._open[key]})
        return str(review["decision_id"])

    def mark_close(self, symbol: str, slot: str, exit_price: float, pnl_usd: float,
                   exit_reason: str, ts: float | None = None) -> bool:
        key = (symbol, slot)
        opened = self._open.pop(key, None)
        if not opened:
            return False
        now = float(ts or time.time())
        entry = float(opened.get("entry_price", 0.0))
        pnl_pct = ((float(exit_price) / entry - 1.0) * 100.0) if entry > 0 else 0.0
        outcome = TradeOutcome(
            decision_id=str(opened["decision_id"]),
            symbol=symbol,
            pnl_usd=float(pnl_usd),
            pnl_pct=pnl_pct,
            exit_reason=str(exit_reason),
            duration_sec=max(0.0, now - float(opened.get("open_ts", now))),
            metadata={
                "legacy_slot": slot,
                "legacy_exit_price": float(exit_price),
                "shadow_action": opened.get("shadow_action"),
                "shadow_confidence": opened.get("shadow_confidence"),
                "executed_action": "BUY",
            },
        )
        self.engine.account.close_position(f"{symbol}:{slot}", float(pnl_usd),
                                           outcome.fees_usd, now)
        self.engine.learn(outcome)
        return True
