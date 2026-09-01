"""Brian 2026: shadow-first adaptive trading research core.

The package is deliberately execution-agnostic: it can observe, decide, learn,
replay and evaluate candidates without placing real orders.
"""

from .engine import BrianEngine
from .types import MarketSnapshot, Decision, TradeOutcome, SpecialistVote

__all__ = ["BrianEngine", "MarketSnapshot", "Decision", "TradeOutcome", "SpecialistVote"]
__version__ = "0.1.0"
