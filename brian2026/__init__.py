"""Brian 2026: shadow-first adaptive trading research core.

The package is deliberately execution-agnostic: it can observe, decide, learn,
replay and evaluate candidates without placing real orders.
"""

from .engine import BrianEngine
from .features import FeatureSnapshot, from_closed_candles
from .types import MarketSnapshot, Decision, TradeOutcome, SpecialistVote

__all__ = [
    "BrianEngine", "FeatureSnapshot", "from_closed_candles", "MarketSnapshot",
    "Decision", "TradeOutcome", "SpecialistVote",
]
__version__ = "0.1.0"
