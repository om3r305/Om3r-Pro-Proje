from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import math

from .portfolio import DEVELOPMENT_CUTOFF

Action = Literal["BUY", "SELL", "WAIT"]
Setup = Literal[
    "LIQUIDITY_SWEEP_REVERSAL",
    "FAILED_BREAK_REVERSAL",
    "BREAKOUT_RETEST_CONTINUATION",
    "PULLBACK_CONTINUATION",
    "RANGE_REJECTION",
    "TREND_EXHAUSTION",
    "NO_CLEAR_SETUP",
]


@dataclass(frozen=True, slots=True)
class ExpertOpinion:
    name: str
    bias: float
    confidence: float
    veto: bool
    reasons: tuple[str, ...]
    evidence: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class ScenarioCase:
    name: str
    bias: Literal["BULL", "BEAR", "WAIT"]
    strength: float
    activation: str
    invalidation_level: float | None
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpertDecision:
    timestamp: float
    action: Action
    edge: float
    confidence: float
    agreement: float
    setup: Setup
    regime: str
    thesis: str
    invalidation_level: float | None
    bull_case: ScenarioCase
    bear_case: ScenarioCase
    no_trade_case: ScenarioCase
    experts: tuple[ExpertOpinion, ...]
    contradictions: tuple[str, ...]
    evidence: Mapping[str, float | None]
    shadow_only: bool = True
    schema_version: str = "brian.expert-reasoner.v1"

    def manifest(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExpertReasonerConfig:
    entry_edge: float = 0.46
    min_confidence: float = 0.55
    min_agreement: float = 0.60
    min_directional_experts: int = 2
    contradiction_penalty: float = 0.12
    near_level_atr: float = 0.35
    setup_quality: float = 0.65
    extreme_range_expansion: float = 3.0
    require_completed_htf: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.entry_edge <= 1:
            raise ValueError("entry_edge must be in (0, 1]")
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.min_agreement <= 1:
            raise ValueError("confidence/agreement thresholds must be in [0, 1]")
        if self.min_directional_experts < 1:
            raise ValueError("min_directional_experts must be positive")


WEIGHTS: Mapping[str, float] = {
    "structure_expert": 0.32,
    "trend_expert": 0.24,
    "momentum_expert": 0.18,
    "volume_expert": 0.12,
    "mean_reversion_expert": 0.14,
}


def _finite(snapshot: Mapping[str, object], key: str) -> float | None:
    value = snapshot.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flag(snapshot: Mapping[str, object], key: str) -> bool:
    value = _finite(snapshot, key)
    return bool(value is not None and value > 0.5)


def _state(snapshot: Mapping[str, object], key: str) -> int | None:
    value = _finite(snapshot, key)
    if value is None:
        return None
    if value > 0.5:
        return 1
    if value < -0.5:
        return -1
    return 0


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _confidence(bias: float, evidence_count: int, possible: int) -> float:
    if evidence_count <= 0 or possible <= 0:
        return 0.0
    completeness = min(1.0, evidence_count / possible)
    return min(1.0, 0.35 * completeness + 0.65 * abs(bias))


def structure_expert(snapshot: Mapping[str, object]) -> ExpertOpinion:
    bull_keys = (
        "bullish_bos", "bullish_choch", "bullish_sweep",
        "failed_breakdown", "bullish_breakout_retest",
    )
    bear_keys = (
        "bearish_bos", "bearish_choch", "bearish_sweep",
        "failed_breakout", "bearish_breakout_retest",
    )
    bull = sum(_flag(snapshot, key) for key in bull_keys)
    bear = sum(_flag(snapshot, key) for key in bear_keys)
    state = _state(snapshot, "structure_state")
    edge = 0.18 * (bull - bear)
    if state is not None:
        edge += 0.18 * state
    support = _finite(snapshot, "support_distance_atr")
    resistance = _finite(snapshot, "resistance_distance_atr")
    if support is not None and 0 <= support <= 0.75:
        edge += 0.08
    if resistance is not None and 0 <= resistance <= 0.75:
        edge -= 0.08
    edge = _clip(edge)
    reasons: list[str] = []
    if bull:
        reasons.append(f"{bull} confirmed bullish structure events")
    if bear:
        reasons.append(f"{bear} confirmed bearish structure events")
    if state is not None:
        reasons.append(f"confirmed 5m structure state={state}")
    return ExpertOpinion(
        "structure_expert", edge, _confidence(edge, bull + bear + int(state is not None), 5), False,
        tuple(reasons) or ("insufficient confirmed structure evidence",),
        {"bull_events": float(bull), "bear_events": float(bear), "structure_state": None if state is None else float(state)},
    )


def trend_expert(snapshot: Mapping[str, object], *, single_timeframe: bool = False) -> ExpertOpinion:
    s5 = _state(snapshot, "structure_state")
    s15 = None if single_timeframe else _state(snapshot, "structure_15m")
    s1h = None if single_timeframe else _state(snapshot, "structure_1h")
    ema_slope = _finite(snapshot, "ema_slope")
    weighted: list[tuple[float, int]] = []
    if s5 is not None:
        weighted.append((0.25 if not single_timeframe else 0.80, s5))
    if s15 is not None:
        weighted.append((0.30, s15))
    if s1h is not None:
        weighted.append((0.45, s1h))
    edge = sum(weight * state for weight, state in weighted)
    if ema_slope is not None:
        edge += 0.20 if ema_slope > 0 else -0.20 if ema_slope < 0 else 0.0
    edge = _clip(edge)
    states = [x for x in (s5, s15, s1h) if x is not None]
    reasons = [f"completed timeframe states={states}"]
    if ema_slope is not None:
        reasons.append("EMA slope confirms trend direction" if ema_slope * edge >= 0 else "EMA slope conflicts with structure")
    return ExpertOpinion(
        "trend_expert", edge, _confidence(edge, len(states) + int(ema_slope is not None), 4), False,
        tuple(reasons),
        {"structure_5m": None if s5 is None else float(s5), "structure_15m": None if s15 is None else float(s15), "structure_1h": None if s1h is None else float(s1h), "ema_slope": ema_slope},
    )


def momentum_expert(snapshot: Mapping[str, object], *, use_divergence: bool = True) -> ExpertOpinion:
    dip = _finite(snapshot, "dip_score")
    rally = _finite(snapshot, "rally_score")
    rsi = _finite(snapshot, "rsi")
    acceleration = _finite(snapshot, "acceleration")
    ret1 = _finite(snapshot, "return_1")
    edge = 0.0
    count = 0
    if dip is not None and rally is not None:
        edge += 0.50 * (dip - rally)
        count += 1
    if rsi is not None:
        edge += max(-0.15, min(0.15, (rsi - 50.0) / 200.0))
        count += 1
    if acceleration is not None:
        edge += 0.12 if acceleration > 0 else -0.12 if acceleration < 0 else 0.0
        count += 1
    if ret1 is not None:
        edge += 0.08 if ret1 > 0 else -0.08 if ret1 < 0 else 0.0
        count += 1
    if use_divergence:
        if _flag(snapshot, "bullish_rsi_divergence"):
            edge += 0.20
            count += 1
        if _flag(snapshot, "bearish_rsi_divergence"):
            edge -= 0.20
            count += 1
    edge = _clip(edge)
    return ExpertOpinion(
        "momentum_expert", edge, _confidence(edge, count, 6), False,
        ("dip/rally quality, RSI, acceleration and confirmed divergence",),
        {"dip_score": dip, "rally_score": rally, "rsi": rsi, "acceleration": acceleration, "return_1": ret1},
    )


def volume_expert(snapshot: Mapping[str, object], *, enabled: bool = True) -> ExpertOpinion:
    if not enabled:
        return ExpertOpinion("volume_expert", 0.0, 0.0, False, ("volume ablated",), {})
    relative = _finite(snapshot, "relative_volume")
    zscore = _finite(snapshot, "volume_zscore")
    ret1 = _finite(snapshot, "return_1")
    trend = _state(snapshot, "structure_state")
    edge = 0.0
    count = 0
    if relative is not None and ret1 is not None:
        impulse = min(0.20, max(0.0, relative - 1.0) * 0.10)
        edge += impulse if ret1 > 0 else -impulse if ret1 < 0 else 0.0
        count += 1
    if zscore is not None and ret1 is not None:
        impulse = min(0.12, abs(zscore) * 0.04)
        edge += impulse if ret1 > 0 else -impulse if ret1 < 0 else 0.0
        count += 1
    if _flag(snapshot, "pullback_volume_contraction") and trend is not None:
        edge += 0.16 * trend
        count += 1
    if _flag(snapshot, "selling_exhaustion_proxy"):
        edge += 0.18
        count += 1
    if _flag(snapshot, "buying_exhaustion_proxy"):
        edge -= 0.18
        count += 1
    edge = _clip(edge)
    return ExpertOpinion(
        "volume_expert", edge, _confidence(edge, count, 5), False,
        ("relative volume and exhaustion proxies; no fabricated historical order flow",),
        {"relative_volume": relative, "volume_zscore": zscore, "return_1": ret1},
    )


def mean_reversion_expert(snapshot: Mapping[str, object]) -> ExpertOpinion:
    zscore = _finite(snapshot, "zscore")
    bb = _finite(snapshot, "bb_position")
    support = _finite(snapshot, "support_distance_atr")
    resistance = _finite(snapshot, "resistance_distance_atr")
    state = _state(snapshot, "structure_state")
    edge = 0.0
    count = 0
    if zscore is not None:
        edge += max(-0.35, min(0.35, -zscore * 0.12))
        count += 1
    if bb is not None:
        edge += max(-0.18, min(0.18, (0.5 - bb) * 0.30))
        count += 1
    if support is not None and 0 <= support <= 0.50:
        edge += 0.16
        count += 1
    if resistance is not None and 0 <= resistance <= 0.50:
        edge -= 0.16
        count += 1
    # Mean reversion is deliberately de-emphasized in a confirmed directional trend.
    if state in (-1, 1):
        edge *= 0.65
    edge = _clip(edge)
    return ExpertOpinion(
        "mean_reversion_expert", edge, _confidence(edge, count, 4), False,
        ("z-score/Bollinger location with structural support/resistance",),
        {"zscore": zscore, "bb_position": bb, "support_distance_atr": support, "resistance_distance_atr": resistance},
    )


def _classify_setup(snapshot: Mapping[str, object], config: ExpertReasonerConfig) -> Setup:
    dip = _finite(snapshot, "dip_score")
    rally = _finite(snapshot, "rally_score")
    state = _state(snapshot, "structure_state")
    if _flag(snapshot, "failed_breakdown") or _flag(snapshot, "failed_breakout"):
        return "FAILED_BREAK_REVERSAL"
    if (_flag(snapshot, "bullish_sweep") and _flag(snapshot, "bullish_rsi_divergence")) or (
        _flag(snapshot, "bearish_sweep") and _flag(snapshot, "bearish_rsi_divergence")
    ):
        return "LIQUIDITY_SWEEP_REVERSAL"
    if _flag(snapshot, "bullish_breakout_retest") or _flag(snapshot, "bearish_breakout_retest"):
        return "BREAKOUT_RETEST_CONTINUATION"
    if state == 1 and dip is not None and dip >= config.setup_quality:
        return "PULLBACK_CONTINUATION"
    if state == -1 and rally is not None and rally >= config.setup_quality:
        return "PULLBACK_CONTINUATION"
    if state == 0 and (
        (_flag(snapshot, "inside_support_zone") and (_finite(snapshot, "lower_wick_ratio") or 0.0) >= 0.40)
        or (_flag(snapshot, "inside_resistance_zone") and (_finite(snapshot, "upper_wick_ratio") or 0.0) >= 0.40)
    ):
        return "RANGE_REJECTION"
    if (_flag(snapshot, "selling_exhaustion_proxy") and _flag(snapshot, "bullish_rsi_divergence")) or (
        _flag(snapshot, "buying_exhaustion_proxy") and _flag(snapshot, "bearish_rsi_divergence")
    ):
        return "TREND_EXHAUSTION"
    return "NO_CLEAR_SETUP"


def risk_critic(snapshot: Mapping[str, object], raw_edge: float, config: ExpertReasonerConfig) -> ExpertOpinion:
    s15 = _state(snapshot, "structure_15m")
    s1h = _state(snapshot, "structure_1h")
    reasons: list[str] = []
    veto = False
    risk = 0.0
    if config.require_completed_htf and (s15 is None or s1h is None):
        veto = True
        reasons.append("completed 15m/1h context unavailable")
    if s15 is not None and s1h is not None and s15 * s1h == -1:
        risk += 0.35
        reasons.append("15m and 1h structure conflict")
    bull_events = sum(_flag(snapshot, key) for key in ("bullish_bos", "bullish_choch", "bullish_sweep", "failed_breakdown"))
    bear_events = sum(_flag(snapshot, key) for key in ("bearish_bos", "bearish_choch", "bearish_sweep", "failed_breakout"))
    if bull_events and bear_events:
        risk += 0.30
        reasons.append("simultaneous bullish and bearish structure evidence")
    expansion = _finite(snapshot, "range_expansion")
    if expansion is not None and expansion >= config.extreme_range_expansion:
        risk += 0.25
        reasons.append("extreme range expansion")
    support = _finite(snapshot, "support_distance_atr")
    resistance = _finite(snapshot, "resistance_distance_atr")
    if raw_edge > 0 and resistance is not None and 0 <= resistance < config.near_level_atr and not _flag(snapshot, "bullish_breakout_retest"):
        risk += 0.25
        reasons.append("long candidate crowded into confirmed resistance")
    if raw_edge < 0 and support is not None and 0 <= support < config.near_level_atr and not _flag(snapshot, "bearish_breakout_retest"):
        risk += 0.25
        reasons.append("short candidate crowded into confirmed support")
    if risk >= 0.60:
        veto = True
    return ExpertOpinion(
        "risk_critic", -math.copysign(min(1.0, risk), raw_edge) if raw_edge else 0.0,
        min(1.0, risk), veto,
        tuple(reasons) or ("no material preregistered risk conflict",),
        {"risk_score": risk, "structure_15m": None if s15 is None else float(s15), "structure_1h": None if s1h is None else float(s1h)},
    )


def _scenario_evidence(snapshot: Mapping[str, object], bullish: bool) -> tuple[str, ...]:
    keys = (
        ("bullish_bos", "confirmed bullish BOS"),
        ("bullish_choch", "confirmed bullish CHOCH"),
        ("bullish_sweep", "support-side liquidity sweep"),
        ("failed_breakdown", "failed breakdown reclaim"),
        ("bullish_breakout_retest", "bullish breakout/retest"),
        ("bullish_rsi_divergence", "confirmed bullish RSI divergence"),
    ) if bullish else (
        ("bearish_bos", "confirmed bearish BOS"),
        ("bearish_choch", "confirmed bearish CHOCH"),
        ("bearish_sweep", "resistance-side liquidity sweep"),
        ("failed_breakout", "failed breakout rejection"),
        ("bearish_breakout_retest", "bearish breakout/retest"),
        ("bearish_rsi_divergence", "confirmed bearish RSI divergence"),
    )
    found = [label for key, label in keys if _flag(snapshot, key)]
    return tuple(found) or ("scenario requires additional confirmation",)


def _regime(snapshot: Mapping[str, object]) -> str:
    s1h = _state(snapshot, "structure_1h")
    s15 = _state(snapshot, "structure_15m")
    s5 = _state(snapshot, "structure_state")
    if s1h == s15 == s5 == 1:
        return "ALIGNED_UPTREND"
    if s1h == s15 == s5 == -1:
        return "ALIGNED_DOWNTREND"
    if s1h is not None and s15 is not None and s1h * s15 == -1:
        return "HTF_CONFLICT"
    if s1h == 0 and s15 == 0:
        return "RANGE"
    return "MIXED_TRANSITION"


def reason_market(
    snapshot: Mapping[str, object],
    *,
    timestamp: float,
    config: ExpertReasonerConfig = ExpertReasonerConfig(),
    use_risk_critic: bool = True,
    single_timeframe: bool = False,
    use_volume: bool = True,
    use_divergence: bool = True,
) -> ExpertDecision:
    if float(timestamp) >= DEVELOPMENT_CUTOFF:
        raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden for Phase 2.8")

    experts = (
        structure_expert(snapshot),
        trend_expert(snapshot, single_timeframe=single_timeframe),
        momentum_expert(snapshot, use_divergence=use_divergence),
        volume_expert(snapshot, enabled=use_volume),
        mean_reversion_expert(snapshot),
    )
    denominator = sum(WEIGHTS[item.name] * item.confidence for item in experts)
    raw_edge = 0.0 if denominator <= 0 else sum(
        WEIGHTS[item.name] * item.confidence * item.bias for item in experts
    ) / denominator
    direction = 1 if raw_edge > 0 else -1 if raw_edge < 0 else 0
    supporters = [item for item in experts if direction and item.bias * direction > 0.05]
    opponents = [item for item in experts if direction and item.bias * direction < -0.05]
    agreement_den = sum(WEIGHTS[item.name] * item.confidence for item in experts)
    agreement = 0.0 if agreement_den <= 0 else sum(
        WEIGHTS[item.name] * item.confidence for item in supporters
    ) / agreement_den

    contradictions: list[str] = []
    if opponents:
        contradictions.append("opposing experts=" + ",".join(item.name for item in opponents))
    s15 = _state(snapshot, "structure_15m")
    s1h = _state(snapshot, "structure_1h")
    if s15 is not None and s1h is not None and s15 * s1h == -1:
        contradictions.append("completed 15m/1h structure disagreement")
    if _flag(snapshot, "bullish_choch") and _flag(snapshot, "bearish_choch"):
        contradictions.append("two-sided CHOCH conflict")

    edge = _clip(raw_edge * max(0.0, 1.0 - config.contradiction_penalty * len(contradictions)))
    critic = risk_critic(snapshot, edge, config)
    setup = _classify_setup(snapshot, config)
    confidence = min(1.0, 0.65 * abs(edge) + 0.35 * agreement)
    enough_experts = len(supporters) >= config.min_directional_experts
    blocked = use_risk_critic and critic.veto

    action: Action = "WAIT"
    if not blocked and enough_experts and setup != "NO_CLEAR_SETUP" and confidence >= config.min_confidence and agreement >= config.min_agreement:
        if edge >= config.entry_edge:
            action = "BUY"
        elif edge <= -config.entry_edge:
            action = "SELL"

    support_level = _finite(snapshot, "nearest_support")
    resistance_level = _finite(snapshot, "nearest_resistance")
    invalidation = support_level if action == "BUY" else resistance_level if action == "SELL" else None
    regime = _regime(snapshot)
    bull_evidence = _scenario_evidence(snapshot, True)
    bear_evidence = _scenario_evidence(snapshot, False)
    bull_strength = max(0.0, min(1.0, (edge + 1.0) / 2.0))
    bear_strength = max(0.0, min(1.0, (1.0 - edge) / 2.0))
    bull_case = ScenarioCase(
        "bull_case", "BULL", bull_strength,
        "activate only after confirmed bullish structure/reclaim/retest evidence persists",
        support_level, bull_evidence,
    )
    bear_case = ScenarioCase(
        "bear_case", "BEAR", bear_strength,
        "activate only after confirmed bearish structure/rejection/retest evidence persists",
        resistance_level, bear_evidence,
    )
    wait_strength = min(1.0, 0.35 + 0.15 * len(contradictions) + (0.35 if blocked else 0.0) + (0.15 if setup == "NO_CLEAR_SETUP" else 0.0))
    no_trade_case = ScenarioCase(
        "no_trade_case", "WAIT", wait_strength,
        "wait for completed higher-timeframe alignment and a confirmed objective setup",
        None, tuple(contradictions) or ("no preregistered setup has sufficient agreement",),
    )

    if action == "BUY":
        thesis = f"{setup}: bullish evidence is aligned enough across experts; thesis invalidates at confirmed support failure."
    elif action == "SELL":
        thesis = f"{setup}: bearish evidence is aligned enough across experts; thesis invalidates at confirmed resistance failure."
    else:
        thesis = f"WAIT: {setup}; evidence is incomplete, conflicting, insufficiently agreed, or risk-vetoed."

    evidence = {
        "raw_edge": raw_edge,
        "penalized_edge": edge,
        "agreement": agreement,
        "expert_supporters": float(len(supporters)),
        "expert_opponents": float(len(opponents)),
        "risk_score": _finite(critic.evidence, "risk_score"),
        "structure_15m": None if s15 is None else float(s15),
        "structure_1h": None if s1h is None else float(s1h),
        "dip_score": _finite(snapshot, "dip_score"),
        "rally_score": _finite(snapshot, "rally_score"),
    }
    all_experts: Sequence[ExpertOpinion] = experts + ((critic,) if use_risk_critic else ())
    return ExpertDecision(
        float(timestamp), action, edge, confidence, agreement, setup, regime, thesis,
        invalidation, bull_case, bear_case, no_trade_case, tuple(all_experts),
        tuple(contradictions), evidence,
    )
