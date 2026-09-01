from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal
import time
import uuid

Action = Literal["BUY", "SELL", "WAIT"]


def _id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    price: float
    ts: float = field(default_factory=time.time)
    timeframe: str = "1m"
    regime: str = "UNKNOWN"
    features: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpecialistVote:
    name: str
    action: Action
    confidence: float
    edge: float = 0.0
    rationale: str = ""
    features_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Decision:
    symbol: str
    action: Action
    confidence: float
    score: float
    regime: str
    votes: list[SpecialistVote]
    reason: str
    ts: float = field(default_factory=time.time)
    decision_id: str = field(default_factory=lambda: _id("dec"))
    allowed_by_risk: bool = False
    size_scale: float = 0.0
    shadow_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["votes"] = [v.to_dict() for v in self.votes]
        return out


@dataclass(slots=True)
class TradeOutcome:
    decision_id: str
    symbol: str
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    duration_sec: float
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    fees_usd: float = 0.0
    ts: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def won(self) -> bool:
        return self.pnl_usd > 0

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["won"] = self.won
        return out
