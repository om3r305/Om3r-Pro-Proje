"""BTC/ETH trend-following with volatility targeting (daily closes, Binance spot).

Only two assets that exist throughout the sample, so no survivorship bias.
Signal at close t, earns t+1 (no lookahead). 15 bps per side on weight changes.
Reports a small fixed grid, split by year, to show robustness rather than a
single tuned winner. Reads the daily cache written by momentum_pit_study.py.

Usage: python research/btc_eth_trend_study.py [--days 2190]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_pit_study import SIDE_COST, daily

DAYS = 365


def perf(net: pd.Series) -> dict:
    net = net.dropna()
    eq = (1 + net).cumprod()
    yrs = len(net) / DAYS
    yearly = net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "sharpe": net.mean() / net.std() * np.sqrt(DAYS),
            "max_dd": (eq / eq.cummax() - 1).min(), "exposure": np.nan,
            **{f"y{y}": v for y, v in yearly.items()}}


def backtest(w: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    held = w.shift(1).fillna(0.0)
    return (held * rets).sum(axis=1) - (w - held).abs().sum(axis=1) * SIDE_COST


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2190)
    args = ap.parse_args()
    px = pd.DataFrame({s: daily(s, args.days)["close"] for s in ("BTCUSDT", "ETHUSDT")}).dropna()
    rets = px.pct_change().fillna(0.0)
    start = px.index[200]  # common start after the longest lookback warms up
    vol = rets.rolling(30).std() * np.sqrt(DAYS)

    rows = {}
    rows["BTC hold"] = perf(backtest(pd.DataFrame({"BTCUSDT": 1.0, "ETHUSDT": 0.0}, index=px.index), rets)[start:])
    rows["50/50 hold (daily rebal)"] = perf(backtest(pd.DataFrame(0.5, index=px.index, columns=px.columns), rets)[start:])
    for L in (50, 100, 200):
        on = (px > px.rolling(L).mean()).astype(float)
        for sizing in ("equal", "vol40"):
            w = on * 0.5 if sizing == "equal" else on * (0.5 * (0.40 / vol)).clip(upper=0.5)
            # rebalance weekly unless the trend flips, to cap turnover
            weekly = w.where(pd.Series(np.arange(len(w)) % 7 == 0, index=w.index), axis=0)
            flip = on.diff().abs().sum(axis=1) > 0
            w = w.where(flip, weekly, axis=0).ffill().fillna(0.0) if sizing == "vol40" else w
            net = backtest(w, rets)[start:]
            r = perf(net)
            r["exposure"] = w[start:].sum(axis=1).mean()
            rows[f"trend SMA{L} {sizing}"] = r

    df = pd.DataFrame(rows).T
    pct = [c for c in df.columns if c != "sharpe"]
    df[pct] = (df[pct].astype(float) * 100).round(1)
    df["sharpe"] = df["sharpe"].astype(float).round(2)
    pd.set_option("display.width", 220)
    print(f"{start.date()}..{px.index[-1].date()}  costs {SIDE_COST*1e4:.0f} bps/side")
    print(df.to_string())
    df.to_json(Path(__file__).with_name("btc_eth_trend_study.json"), orient="index", indent=1)


if __name__ == "__main__":
    main()
