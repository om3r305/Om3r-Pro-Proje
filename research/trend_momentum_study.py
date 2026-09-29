"""Long-only spot trend / momentum study on liquid Binance USDT pairs (4h bars).

Every rule is decided at bar close t and earns bar t+1's return (no lookahead).
Costs: TAKER_BPS + SLIPPAGE_BPS per side, charged on each change in weight.
All grid points are reported, split into halves, to show stability rather than a
single best cherry-picked run.

Usage: python research/trend_momentum_study.py [--candidates 60] [--universe 20] [--days 2190]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SPOT = "https://api.binance.com"
TAKER_BPS, SLIPPAGE_BPS = 10.0, 5.0
SIDE_COST = (TAKER_BPS + SLIPPAGE_BPS) / 1e4
BARS_PER_YEAR = 6 * 365
STABLES = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "DAIUSDT", "EURUSDT", "USD1USDT", "BFUSDUSDT"}


def top_symbols(n: int) -> list[str]:
    t = requests.get(f"{SPOT}/api/v3/ticker/24hr", timeout=30).json()
    t = [x for x in t if x["symbol"].endswith("USDT") and x["symbol"] not in STABLES]
    t.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [x["symbol"] for x in t[:n]]


CACHE = Path(__file__).with_name(".cache")


def klines(symbol: str, days: int) -> pd.DataFrame:
    cached = CACHE / f"{symbol}_4h_{days}d_{time.strftime('%Y%m%d')}.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    df = _fetch_klines(symbol, days)
    CACHE.mkdir(exist_ok=True)
    df.to_pickle(cached)
    return df


def _fetch_klines(symbol: str, days: int) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    out: list[list] = []
    while start < end:
        k = requests.get(f"{SPOT}/api/v3/klines",
                         params={"symbol": symbol, "interval": "4h", "startTime": start, "limit": 1000},
                         timeout=20).json()
        if not k:
            break
        out += k
        start = k[-1][6] + 1
        if len(k) < 1000:
            break
    idx = [pd.Timestamp(r[6] + 1, unit="ms", tz="UTC") for r in out]
    df = pd.DataFrame({"close": [float(r[4]) for r in out], "qvol": [float(r[7]) for r in out]}, index=idx)
    df = df[~df.index.duplicated()]
    return df[df.index <= pd.Timestamp.now(tz="UTC")].sort_index()


def stats(weights: pd.DataFrame, rets: pd.DataFrame) -> dict:
    w = weights.fillna(0.0)
    gross = (w.shift(1) * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    net = gross - turnover * SIDE_COST
    eq = (1 + net).cumprod()
    years = len(net) / BARS_PER_YEAR
    dd = (eq / eq.cummax() - 1).min()
    sharpe = net.mean() / net.std() * np.sqrt(BARS_PER_YEAR) if net.std() > 0 else 0.0
    half = len(net) // 2
    h1, h2 = net.iloc[:half], net.iloc[half:]
    ann = lambda x: (1 + x).prod() ** (BARS_PER_YEAR / max(len(x), 1)) - 1
    return {"cagr": eq.iloc[-1] ** (1 / years) - 1, "sharpe": sharpe, "max_dd": dd,
            "turnover_yr": turnover.sum() / years, "h1_cagr": ann(h1), "h2_cagr": ann(h2),
            "exposure": w.sum(axis=1).clip(upper=1).mean()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=60, help="candidate pool (by today's volume)")
    ap.add_argument("--universe", type=int, default=20, help="point-in-time universe size")
    ap.add_argument("--days", type=int, default=2190)
    args = ap.parse_args()

    # Point-in-time universe: at each weekly rebalance, the `universe` most traded
    # candidates by trailing 30d quote volume. Coins enter only once they trade;
    # this removes most (not all: delisted coins are absent) selection bias.
    data = {s: klines(s, args.days) for s in top_symbols(args.candidates)}
    px = pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()
    qv = pd.DataFrame({s: d["qvol"] for s, d in data.items()}).reindex(px.index)
    rets = px.pct_change()
    rebal = px.index[::42]  # weekly (42 x 4h)
    liq = qv.rolling(180, min_periods=180).sum()
    is_rebal = pd.Series(px.index.isin(rebal), index=px.index)
    member = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for t in rebal:
        member.loc[t] = 0.0
        member.loc[t, liq.loc[t].dropna().nlargest(args.universe).index] = 1.0
    member = member.ffill().fillna(0.0).astype(bool)
    print(f"candidates={px.shape[1]} universe={args.universe} bars={len(px)} "
          f"{px.index[0].date()}..{px.index[-1].date()}")

    def weekly(target: pd.DataFrame) -> pd.DataFrame:
        """Hold rebalance-bar weights until the next rebalance."""
        w = target.where(is_rebal, axis=0).ffill()
        return w.fillna(0.0)

    rows = []
    rows.append({"rule": "BTC buy&hold", **stats(pd.DataFrame({"BTCUSDT": 1.0}, index=px.index), rets)})
    ew = member.astype(float).div(member.sum(axis=1).replace(0, np.nan), axis=0)
    rows.append({"rule": "PIT EW buy&hold (weekly)", **stats(weekly(ew), rets)})

    # Time-series trend inside the PIT universe: 1/N per member above its SMA(L), checked every bar.
    for L in (50, 100, 200):
        up = (px > px.rolling(L).mean()) & member
        n = member.sum(axis=1).replace(0, np.nan)
        rows.append({"rule": f"TS trend SMA{L} (PIT EW)", **stats(up.astype(float).div(n, axis=0), rets)})
        btc = (px[["BTCUSDT"]] > px[["BTCUSDT"]].rolling(L).mean()).astype(float)
        rows.append({"rule": f"BTC trend SMA{L}", **stats(btc, rets)})

    # Cross-sectional momentum within the PIT universe, gated by BTC > SMA200.
    btc_up = px["BTCUSDT"] > px["BTCUSDT"].rolling(200).mean()
    for lb_days in (14, 30, 60):
        mom = (px / px.shift(lb_days * 6) - 1).where(member)
        for K in (3, 5):
            target = pd.DataFrame(0.0, index=px.index, columns=px.columns)
            for t in rebal:
                if btc_up.loc[t] and mom.loc[t].notna().sum() >= K:
                    target.loc[t, mom.loc[t].nlargest(K).index] = 1.0 / K
            rows.append({"rule": f"XS mom {lb_days}d top{K} +BTC gate", **stats(weekly(target), rets)})

    df = pd.DataFrame(rows).set_index("rule")
    pct = ["cagr", "max_dd", "h1_cagr", "h2_cagr", "exposure"]
    df[pct] = (df[pct] * 100).round(1)
    print(f"\ncosts: {TAKER_BPS:.0f} bps taker + {SLIPPAGE_BPS:.0f} bps slippage per side")
    print(df.round(2).to_string())
    df.to_json("research/trend_momentum_study.json", orient="index", indent=1)


if __name__ == "__main__":
    main()
