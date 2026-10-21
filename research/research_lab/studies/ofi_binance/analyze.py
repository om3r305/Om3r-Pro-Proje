"""Pre-registered offline analysis of the OFI study (arXiv:1011.6402). Python stdlib only.

Input: the collector's per-second buckets (<out>/<day>.<SYMBOL>.bucket.jsonl) and meta (tick size).
Output: result.json with C-01, C-02 and the BRIAN predictive hypothesis, each decided by the rules
in research/research_lab/ledger/arxiv-1011.6402.json. Nothing here trades.

Usage: python analyze.py --data <collector out dir> --out <dir> [--fee-bps 10]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
from collections import defaultdict

DT = 10            # seconds per interval (paper)
WINDOW = 1800      # seconds per estimation window (paper: half hour)
MIN_WINDOW_COVERAGE = 0.95
QUANTILES = (0.90, 0.95, 0.99)
SEED = 20261021


# ── loading ────────────────────────────────────────────────────────────────────────────────────
def load(data_dir: str):
    buckets: dict[str, list[dict]] = defaultdict(list)
    ticks: dict[str, float] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.bucket.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("header"):
                    buckets[r["symbol"]].append(r)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.meta.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("header") and r.get("symbol") and r.get("tickSize"):
                    ticks.setdefault(r["symbol"], float(r["tickSize"]))
    for s in buckets:
        buckets[s].sort(key=lambda b: b["sec"])
    return buckets, ticks


# ── intervals ──────────────────────────────────────────────────────────────────────────────────
def to_intervals(buckets: list[dict], tick: float) -> list[dict]:
    """10 s intervals on the epoch grid. An interval is valid only if all 10 seconds exist and no
    second (re)initialised the state; Delta P needs the previous interval to be valid too."""
    by_sec = {b["sec"]: b for b in buckets}
    if not by_sec:
        return []
    start = (min(by_sec) // DT) * DT
    end = (max(by_sec) // DT) * DT
    out, prev = [], None
    for t in range(start, end + 1, DT):
        secs = [by_sec.get(s) for s in range(t, t + DT)]
        ok = all(b is not None for b in secs) and not any(b["stateReset"] for b in secs if b)
        cur = None
        if ok:
            last = secs[-1]
            cur = {"t": t, "ofi": sum(b["ofi"] for b in secs), "events": sum(b["events"] for b in secs),
                   "depth_sum": sum(b["depthSum"] for b in secs), "mid": last["mid"], "bid": last["bid"], "ask": last["ask"]}
        if cur and prev and prev["t"] == t - DT:
            cur["dp"] = (cur["mid"] - prev["mid"]) / tick
            out.append(cur)
        elif cur:
            cur["dp"] = None
            out.append(cur)
        prev = cur
    return out


def windows(intervals: list[dict]) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for it in intervals:
        groups[it["t"] // WINDOW].append(it)
    out = []
    for w, items in sorted(groups.items()):
        usable = [i for i in items if i["dp"] is not None]
        coverage = len(usable) / (WINDOW / DT)
        events = sum(i["events"] for i in usable)
        out.append({"w": w, "day": (w * WINDOW) // 86400, "items": usable, "coverage": coverage,
                    "valid": coverage >= MIN_WINDOW_COVERAGE,
                    "ad": (sum(i["depth_sum"] for i in usable) / (2 * events)) if events else None})
    return out


# ── statistics ─────────────────────────────────────────────────────────────────────────────────
def ols_white(x: list[float], y: list[float]):
    """y = a + b x by OLS; returns (a, b, r2, White heteroscedasticity-consistent SE of b)."""
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 0:
        return None
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    a = my - b * mx
    res = [yi - a - b * xi for xi, yi in zip(x, y)]
    sst = sum((yi - my) ** 2 for yi in y)
    r2 = 1 - sum(e * e for e in res) / sst if sst > 0 else 0.0
    se_white = math.sqrt(sum(((xi - mx) ** 2) * e * e for xi, e in zip(x, res))) / sxx
    return a, b, r2, se_white


def percentile(v: list[float], q: float) -> float:
    s = sorted(v)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def day_bootstrap(values_by_day: dict[int, list[float]], stat, resamples=2000, seed=SEED):
    days = sorted(values_by_day)
    rng = random.Random(seed)
    out = []
    for _ in range(resamples):
        pick = [days[rng.randrange(len(days))] for _ in days]
        v = stat([values_by_day[d] for d in pick])
        if v is not None:
            out.append(v)
    return (percentile(out, 0.025), percentile(out, 0.975)) if out else (None, None)


# ── decisions (pre-registered) ─────────────────────────────────────────────────────────────────
def c01(wins: list[dict]) -> dict:
    fits = []
    for w in wins:
        if not w["valid"]:
            continue
        f = ols_white([i["ofi"] for i in w["items"]], [i["dp"] for i in w["items"]])
        if f:
            a, b, r2, se = f
            fits.append({"w": w["w"], "day": w["day"], "beta": b, "r2": r2, "z": b / se if se > 0 else float("inf"), "ad": w["ad"]})
    if len(fits) < 200:
        return {"status": "INSUFFICIENT_DATA", "valid_windows": len(fits), "fits": fits}
    med_r2 = statistics.median(f["r2"] for f in fits)
    sig_pos = sum(1 for f in fits if f["beta"] > 0 and f["z"] > 1.959964) / len(fits)
    if med_r2 >= 0.50 and sig_pos >= 0.90:
        st = "PASS"
    elif med_r2 < 0.20 or sig_pos < 0.50:
        st = "FAIL"
    else:
        st = "UNVERIFIED"
    return {"status": st, "valid_windows": len(fits), "median_r2": med_r2, "share_beta_pos_significant": sig_pos, "fits": fits}


def c02(fits: list[dict]) -> dict:
    pts = [f for f in fits if f["beta"] > 0 and f["ad"] and f["ad"] > 0]
    if len(pts) < 200:
        return {"status": "INSUFFICIENT_DATA", "windows": len(pts)}

    def lam(rows):
        flat = [r for day in rows for r in day]
        f = ols_white([math.log(r["ad"]) for r in flat], [math.log(r["beta"]) for r in flat])
        return -f[1] if f else None

    by_day: dict[int, list[dict]] = defaultdict(list)
    for p in pts:
        by_day[p["day"]].append(p)
    lam_hat = lam(list(by_day.values()))
    lo, hi = day_bootstrap(by_day, lam)
    st = "PASS" if lam_hat is not None and lo is not None and lo > 0 else ("FAIL" if hi is not None and hi <= 0 else "UNVERIFIED")
    return {"status": st, "lambda": lam_hat, "interval": [lo, hi], "windows": len(pts), "days": len(by_day)}


def trades(intervals: list[dict], threshold: float, fee_bps: float) -> list[dict]:
    """Signal at the end of interval k when |OFI_k| >= threshold; enter as taker at the touch,
    exit as taker 10 s later. Net of both fees. Only contiguous valid intervals are used."""
    out = []
    for cur, nxt in zip(intervals, intervals[1:]):
        if nxt["t"] != cur["t"] + DT or nxt["dp"] is None or abs(cur["ofi"]) < threshold or cur["ofi"] == 0:
            continue
        if cur["ofi"] > 0:   # buy at ask, sell at bid
            gross = (nxt["bid"] / cur["ask"] - 1) * 1e4
        else:                # sell at bid, buy back at ask
            gross = (cur["bid"] / nxt["ask"] - 1) * 1e4
        out.append({"t": cur["t"], "day": cur["t"] // 86400, "gross_bps": gross, "net_bps": gross - 2 * fee_bps})
    return out


def hypothesis(intervals: list[dict], fee_bps: float) -> dict:
    if not intervals:
        return {"status": "INSUFFICIENT_DATA", "trades": 0}
    first_day = intervals[0]["t"] // 86400
    tune = [i for i in intervals if i["t"] // 86400 < first_day + 7]
    test = [i for i in intervals if i["t"] // 86400 >= first_day + 7]
    if not tune or not test:
        return {"status": "INSUFFICIENT_DATA", "trades": 0, "reason": "need 7 tuning days and a test period"}
    abs_ofi = [abs(i["ofi"]) for i in tune if i["ofi"] != 0]
    best = None
    for q in QUANTILES:
        thr = percentile(abs_ofi, q)
        tr = trades(tune, thr, fee_bps)
        g = statistics.mean(t["gross_bps"] for t in tr) if tr else float("-inf")
        if best is None or g > best[2]:
            best = (q, thr, g)
    q, thr, _ = best
    tt = trades(test, thr, fee_bps)  # threshold frozen from the tuning period
    if len(tt) < 100:
        return {"status": "INSUFFICIENT_DATA", "trades": len(tt), "quantile": q, "threshold": thr}
    by_day: dict[int, list[float]] = defaultdict(list)
    for t in tt:
        by_day[t["day"]].append(t["net_bps"])
    mean = lambda rows: statistics.mean([v for d in rows for v in d]) if any(rows) else None
    lo, hi = day_bootstrap(by_day, mean)
    m = statistics.mean(t["net_bps"] for t in tt)
    st = "PASS" if lo is not None and lo > 0 and m > 0 else ("FAIL" if hi is not None and hi < 0 else "UNVERIFIED")
    return {"status": st, "trades": len(tt), "quantile": q, "threshold": thr, "fee_bps_per_side": fee_bps,
            "mean_net_bps": m, "mean_gross_bps": statistics.mean(t["gross_bps"] for t in tt), "interval": [lo, hi]}


def missing_share(buckets: list[dict]) -> float:
    if not buckets:
        return 1.0
    span = buckets[-1]["sec"] - buckets[0]["sec"] + 1
    return 1 - len({b["sec"] for b in buckets}) / span


def analyze(data_dir: str, fee_bps: float) -> dict:
    buckets, ticks = load(data_dir)
    res = {"experiment_id": "ofi-binance-v1", "symbols": {}, "trade_signal": False, "shadow_only": True}
    for sym, bs in sorted(buckets.items()):
        tick = ticks.get(sym)
        if not tick:
            res["symbols"][sym] = {"status": "REVIEW_REQUIRED", "reason": "tick size unknown"}
            continue
        miss = missing_share(bs)
        iv = to_intervals(bs, tick)
        r1 = c01(windows(iv))
        r2 = c02(r1.get("fits", []))
        r3 = hypothesis(iv, fee_bps)
        review = miss > 0.05
        for r in (r1, r2, r3):
            if review and r["status"] in ("PASS", "FAIL", "UNVERIFIED"):
                r["status"], r["reason"] = "REVIEW_REQUIRED", f"{miss:.1%} of seconds missing (> 5%)"
        r1.pop("fits", None)
        res["symbols"][sym] = {"missing_seconds_share": miss, "tick": tick, "C-01": r1, "C-02": r2, "brian:ofi-predictive-v1": r3}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fee-bps", type=float, default=10.0)  # Binance spot VIP0 published taker fee, per side
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = analyze(a.data, a.fee_bps)
    with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    print(json.dumps(r, indent=2)[:4000])
