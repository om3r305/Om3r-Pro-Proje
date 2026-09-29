"""Delta-neutral funding carry study (long spot / short USD-M perp) on Binance.

Walk-forward only: each 8h settlement the position state is decided from funding
already paid (trailing window), then the *next* settlement's funding is earned.
Costs per full open+close: spot taker in/out + perp taker in/out.
Ignores spot/perp basis drift at entry/exit and borrow/margin costs, so results
are an upper bound; use them to decide whether carry deserves a paper run.

Usage: python research/funding_carry_study.py [--top 20] [--days 365]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import requests

FAPI = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
SPOT_TAKER_BPS = 10.0   # VIP0 spot taker, per side
PERP_TAKER_BPS = 5.0    # VIP0 USD-M taker, per side
ROUND_TRIP_BPS = 2 * (SPOT_TAKER_BPS + PERP_TAKER_BPS)
PERIODS_PER_YEAR = 3 * 365


def top_symbols(n: int) -> list[str]:
    perps = requests.get(f"{FAPI}/fapi/v1/ticker/24hr", timeout=20).json()
    spot = {s["symbol"] for s in requests.get(f"{SPOT}/api/v3/exchangeInfo", timeout=30).json()["symbols"]
            if s["status"] == "TRADING"}
    perps = [p for p in perps if p["symbol"].endswith("USDT") and p["symbol"] in spot]
    perps.sort(key=lambda p: float(p["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in perps[:n]]


def funding_history(symbol: str, days: int) -> pd.Series:
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    rows: list[dict] = []
    while start < end:
        batch = requests.get(f"{FAPI}/fapi/v1/fundingRate",
                             params={"symbol": symbol, "startTime": start, "endTime": end, "limit": 1000},
                             timeout=20).json()
        if not batch:
            break
        rows += batch
        start = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
    s = pd.Series({pd.Timestamp(r["fundingTime"], unit="ms", tz="UTC"): float(r["fundingRate"]) for r in rows})
    return s.sort_index()


def simulate(rates: pd.Series, lookback: int, enter_ann: float, exit_ann: float) -> dict:
    """Hysteresis rule on trailing mean funding (annualised). Decision at t uses rates[:t]."""
    trailing = rates.rolling(lookback).mean().shift(1) * PERIODS_PER_YEAR
    held = False
    pnl_bps, trades, held_periods = [], 0, 0
    for t, r in rates.items():
        tr = trailing.get(t)
        cost = 0.0
        if tr is not None and not np.isnan(tr):
            if not held and tr > enter_ann:
                held, trades, cost = True, trades + 1, ROUND_TRIP_BPS / 2
            elif held and tr < exit_ann:
                held, cost = False, ROUND_TRIP_BPS / 2
        earned = r * 1e4 if held else 0.0   # short perp receives positive funding
        held_periods += held
        pnl_bps.append(earned - cost)
    pnl = pd.Series(pnl_bps, index=rates.index)
    years = len(rates) / PERIODS_PER_YEAR
    daily = pnl.groupby(pnl.index.floor("D")).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0.0
    return {"net_ann_pct": pnl.sum() / 1e2 / years, "sharpe": sharpe, "trades": trades,
            "time_in_market": held_periods / len(rates), "worst_day_bps": daily.min()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()

    symbols = top_symbols(args.top)
    rules = {"always_on": (1, -1e9, -1e9), "trail7d_10/5": (21, 0.10, 0.05), "trail7d_20/10": (21, 0.20, 0.10)}
    rows = []
    for sym in symbols:
        rates = funding_history(sym, args.days)
        if len(rates) < 300:
            continue
        base = {"symbol": sym, "periods": len(rates), "gross_ann_pct": rates.mean() * PERIODS_PER_YEAR * 100,
                "neg_share": (rates < 0).mean()}
        for name, (lb, en, ex) in rules.items():
            r = simulate(rates, lb, en, ex)
            rows.append({**base, "rule": name, **r})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for name in rules:
        sub = df[df.rule == name].sort_values("net_ann_pct", ascending=False)
        print(f"\n=== {name}  (round trip {ROUND_TRIP_BPS:.0f} bps) ===")
        print(sub[["symbol", "gross_ann_pct", "net_ann_pct", "sharpe", "trades", "time_in_market", "neg_share", "worst_day_bps"]]
              .round(3).to_string(index=False))
        print(f"equal-weight portfolio net: {sub.net_ann_pct.mean():.2f}%/yr, median sharpe {sub.sharpe.median():.2f}")
    df.to_json("research/funding_carry_study.json", orient="records", indent=1)


if __name__ == "__main__":
    main()
