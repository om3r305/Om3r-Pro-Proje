from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import math

from .types import MarketSnapshot, SpecialistVote


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _vote(name: str, signed_edge: float, rationale: str, used: dict[str, float]) -> SpecialistVote:
    edge = max(-1.0, min(1.0, signed_edge))
    if abs(edge) < 0.08:
        action = "WAIT"
    else:
        action = "BUY" if edge > 0 else "SELL"
    confidence = 0.5 if action == "WAIT" else _clamp(0.5 + abs(edge) * 0.5)
    return SpecialistVote(name=name, action=action, confidence=confidence, edge=edge,
                          rationale=rationale, features_used=used)


def _unavailable(name: str, *features: str) -> SpecialistVote:
    return SpecialistVote(
        name=name, action="WAIT", confidence=0.0, edge=0.0,
        rationale=f"unavailable:{','.join(features)}", features_used={},
    )


def trend_specialist(s: MarketSnapshot) -> SpecialistVote:
    f = s.features
    if "ema_fast" not in f or "ema_slow" not in f:
        return _unavailable("trend", "ema_fast", "ema_slow")
    price = float(s.price)
    fast = float(f.get("ema_fast", price))
    slow = float(f.get("ema_slow", fast))
    slope = float(f.get("ema_slope_pct", 0.0))
    spread = (fast - slow) / max(abs(price), 1e-12) * 100.0
    edge = math.tanh(spread * 2.2 + slope * 1.5)
    return _vote("trend", edge, f"ema spread={spread:.3f}% slope={slope:.3f}%",
                 {"ema_fast": fast, "ema_slow": slow, "ema_slope_pct": slope})


def momentum_specialist(s: MarketSnapshot) -> SpecialistVote:
    f = s.features
    if "return_5" not in f:
        return _unavailable("momentum", "return_5")
    rsi = float(f.get("rsi", 50.0))
    ret = float(f.get("return_5", f.get("momentum_pct", 0.0)))
    # Momentum is directional but extreme RSI is penalised to avoid chasing.
    directional = math.tanh(ret * 3.0)
    chase_penalty = max(0.0, abs(rsi - 50.0) - 25.0) / 25.0
    edge = directional * (1.0 - 0.45 * chase_penalty)
    return _vote("momentum", edge, f"rsi={rsi:.1f} ret5={ret:.3f}%",
                 {"rsi": rsi, "return_5": ret})


def orderbook_specialist(s: MarketSnapshot) -> SpecialistVote:
    f = s.features
    if "book_imbalance" not in f:
        return _unavailable("orderbook", "book_imbalance")
    imb = float(f.get("book_imbalance", 0.0))  # expected -1..+1
    spread_bps = max(0.0, float(f.get("spread_bps", 0.0)))
    wall = float(f.get("wall_score", 0.0))
    quality = max(0.20, 1.0 - min(spread_bps, 25.0) / 30.0)
    edge = math.tanh((imb * 1.6 + wall * 0.7) * quality)
    return _vote("orderbook", edge, f"imb={imb:.3f} wall={wall:.3f} spread={spread_bps:.1f}bps",
                 {"book_imbalance": imb, "wall_score": wall, "spread_bps": spread_bps})


def breakout_specialist(s: MarketSnapshot) -> SpecialistVote:
    f = s.features
    if not any(name in f for name in ("breakout_score", "volume_z", "acceleration")):
        return _unavailable("breakout", "breakout_score", "volume_z", "acceleration")
    breakout = float(f.get("breakout_score", 0.0))
    vol_z = float(f.get("volume_z", 0.0))
    accel = float(f.get("acceleration", 0.0))
    edge = math.tanh(breakout * 1.2 + vol_z * 0.22 + accel * 0.6)
    return _vote("breakout", edge, f"breakout={breakout:.3f} vol_z={vol_z:.2f} accel={accel:.3f}",
                 {"breakout_score": breakout, "volume_z": vol_z, "acceleration": accel})


def mean_reversion_specialist(s: MarketSnapshot) -> SpecialistVote:
    f = s.features
    if "zscore" not in f and "bb_position" not in f:
        return _unavailable("mean_reversion", "zscore", "bb_position")
    z = float(f.get("zscore", 0.0))
    bb = float(f.get("bb_position", 0.5))  # 0 lower, 1 upper
    regime = str(s.regime).upper()
    regime_scale = 1.0 if regime in {"MEAN", "RANGE", "CHOP"} else 0.35
    edge = math.tanh((-z * 0.7 + (0.5 - bb) * 2.0) * regime_scale)
    return _vote("mean_reversion", edge, f"z={z:.2f} bb={bb:.2f} regime={regime}",
                 {"zscore": z, "bb_position": bb})


DEFAULT_SPECIALISTS: tuple[Callable[[MarketSnapshot], SpecialistVote], ...] = (
    trend_specialist,
    momentum_specialist,
    orderbook_specialist,
    breakout_specialist,
    mean_reversion_specialist,
)


def run_specialists(snapshot: MarketSnapshot,
                    specialists: Iterable[Callable[[MarketSnapshot], SpecialistVote]] = DEFAULT_SPECIALISTS
                    ) -> list[SpecialistVote]:
    out: list[SpecialistVote] = []
    for fn in specialists:
        try:
            out.append(fn(snapshot))
        except Exception as exc:
            out.append(SpecialistVote(name=getattr(fn, "__name__", "specialist"), action="WAIT",
                                      confidence=0.0, edge=0.0, rationale=f"error:{exc}"))
    return out
