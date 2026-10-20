"""Pure statistics for the political-slope experiment (stdlib only, no I/O).

Logistic recalibration: logit P(y=1) = a + b * logit(p). b > 1 means prices are compressed
toward 50% (underconfident); b < 1 means overconfident. The slope is estimated with the same
objective as the paper's sklearn fit (L2 on the slope only, C = 10): minimise
    sum_i NLL_i + 0.5 * b^2 / C
by Newton-Raphson. Uncertainty comes from an EVENT-clustered bootstrap: whole Polymarket events
are resampled, because markets inside one event share an outcome structure.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

C_REG = 10.0
P_CLIP = (0.01, 0.99)  # paper: np.clip(prices / 100, 0.01, 0.99)


def logit(p: float) -> float:
    p = min(max(p, P_CLIP[0]), P_CLIP[1])
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def fit_slope(prices: Sequence[float], outcomes: Sequence[int], c_reg: float = C_REG, max_iter: int = 100) -> tuple[float, float] | None:
    """Return (slope b, intercept a) or None when both outcome classes are not present."""
    if len(prices) != len(outcomes) or len(prices) < 2 or len(set(outcomes)) < 2:
        return None
    xs = [logit(p) for p in prices]
    a, b = 0.0, 1.0
    for _ in range(max_iter):
        ga = gb = 0.0
        haa = hab = hbb = 0.0
        for x, y in zip(xs, outcomes):
            mu = _sigmoid(a + b * x)
            r = mu - y
            w = mu * (1.0 - mu)
            ga += r
            gb += r * x
            haa += w
            hab += w * x
            hbb += w * x * x
        gb += b / c_reg
        hbb += 1.0 / c_reg
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            return None
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a -= da
        b -= db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return b, a


def percentile(sorted_values: Sequence[float], q: float) -> float:
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def cluster_bootstrap_slope(rows: Iterable[tuple[str, float, int]], resamples: int = 2000, seed: int = 20261020, level: float = 0.95) -> dict:
    """rows: (cluster_id, price, outcome). Resample whole clusters with replacement."""
    clusters: dict[str, list[tuple[float, int]]] = {}
    for cid, p, y in rows:
        clusters.setdefault(cid, []).append((p, y))
    ids = sorted(clusters)  # order-independent of input order
    rng = random.Random(seed)
    slopes: list[float] = []
    failed = 0
    for _ in range(resamples):
        ps: list[float] = []
        ys: list[int] = []
        for _ in range(len(ids)):
            for p, y in clusters[ids[rng.randrange(len(ids))]]:
                ps.append(p)
                ys.append(y)
        fit = fit_slope(ps, ys)
        if fit is None:
            failed += 1
            continue
        slopes.append(fit[0])
    slopes.sort()
    if not slopes:
        return {"resamples": resamples, "failed": failed, "lower": None, "upper": None, "level": level}
    alpha = (1.0 - level) / 2.0
    return {"resamples": resamples, "failed": failed, "lower": percentile(slopes, alpha), "upper": percentile(slopes, 1.0 - alpha), "level": level}


# Pre-registered decision rule (fixed before any 2026 data was fetched).
MIN_EVENTS = 30
MIN_MARKETS = 100


def decide(n_markets: int, n_events: int, lower: float | None, upper: float | None) -> tuple[str, str]:
    if n_markets < MIN_MARKETS or n_events < MIN_EVENTS or lower is None or upper is None:
        return "INSUFFICIENT_DATA", f"need >= {MIN_MARKETS} markets and >= {MIN_EVENTS} events with a valid interval"
    if lower > 1.0:
        return "PASS", "event-clustered 95% interval lies above 1: underconfidence (compression toward 50%) replicates"
    if upper < 1.0:
        return "FAIL", "event-clustered 95% interval lies below 1: prices were overconfident, contradicting the claim"
    return "UNVERIFIED", "event-clustered 95% interval includes 1: no reliable departure from calibration"
