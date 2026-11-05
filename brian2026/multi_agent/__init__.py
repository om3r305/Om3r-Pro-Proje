"""BRIAN multi-agent intelligence (shadow only).

Claude (strategist) and Grok (news intelligence) assess unusual market moves independently, review each
other once, and are checked by deterministic verifier, skeptic and risk roles. The output is an
assessment record; it can never place an order. Off by default, with a zero paid-call budget.
"""
from .config import MultiAgentConfig, UnsafeConfigError
from .coordinator import Coordinator, decide
from .schemas import Decision, DecisionRecord

__all__ = ["Coordinator", "Decision", "DecisionRecord", "MultiAgentConfig", "UnsafeConfigError", "decide"]
