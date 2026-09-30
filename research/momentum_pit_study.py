"""Survivorship-free weekly momentum study on every Binance USDT spot pair ever listed.

Candidate pool = all USDT pairs in exchangeInfo, including delisted ones (status
BREAK), minus stablecoins and leveraged tokens. Each week the universe is the top
N by trailing 30d quote volume *as of that week*, so dead coins (LUNA, FTT, ...)
are held exactly when a real trader could have picked them. Daily bars; weights
set at a weekly close earn the following days' returns (no lookahead). Costs:
TAKER_BPS + SLIPPAGE_BPS per side on every weight change. A delisted coin that
stops printing is marked at its last close and liquidated at a HAIRCUT.

Usage: python research/momentum_pit_study.py [--universe 20] [--days 2190]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SPOT = "https://api.binance.com"
TAKER_BPS, SLIPPAGE_BPS = 10.0, 5.0
SIDE_COST = (TAKER_BPS + SLIPPAGE_BPS) / 1e4
DELIST_HAIRCUT = 0.5  # lose half the position value when a held coin stops trading
DAYS_PER_YEAR = 365
STABLE = re.compile(r"^(USDC|BUSD|TUSD|USDP|PAX|DAI|FDUSD|USDS|USDSB|EUR|GBP|AUD|BRL|TRY|USD1|BFUSD|AEUR|EURI|UST|USTC|SUSD|XUSD|RLUSD|USDE)USDT$")
LEVERED = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$")
CACHE = Path(__file__).with_name(".cache")


def all_usdt_pairs() -> list[str]:
    info = requests.get(f"{SPOT}/api/v3/exchangeInfo", timeout=60).json()["symbols"]
    return sorted(s["symbol"] for s in info if s["quoteAsset"] == "USDT"
                  and not STABLE.match(s["symbol"]) and not LEVERED.search(s["symbol"]))


def daily(symbol: str, days: int) -> pd.DataFrame:
    cached = CACHE / f"{symbol}_1d_{days}d_{time.strftime('%Y%m%d')}.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    rows: list[list] = []
    while start < end:
        for attempt in range(5):
            r = requests.get(f"{SPOT}/api/v3/klines",
                             params={"symbol": symbol, "interval": "1d", "startTime": start, "limit": 1000}, timeout=30)
            if r.status_code in (418, 429):
                time.sleep(int(r.headers.get("Retry-After", "5")) + attempt)
                continue
            break
        k = r.json() if r.ok else []
        if not isinstance(k, list) or not k:
            break
        rows += k
        start = k[-1][6] + 1
        if len(k) < 1000:
            break
    idx = pd.to_datetime([x[0] for x in rows], unit="ms", utc=True)
    df = pd.DataFrame({"close": [float(x[4]) for x in rows], "qvol": [float(x[7]) for x in rows]}, index=idx)
    df = df[~df.index.duplicated()].sort_index()
    CACHE.mkdir(exist_ok=True)
    df.to_pickle(cached)
    return df


def run(weights: pd.DataFrame, rets: pd.DataFrame, dead: pd.DataFrame) -> pd.Series:
    """Daily net returns. `dead` marks the first missing bar after a coin's last print."""
    w = weights.fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * rets.fillna(0.0)).sum(axis=1) - (held * dead * DELIST_HAIRCUT).sum(axis=1)
    turnover = (w - held).abs().sum(axis=1)
    return gross - turnover * SIDE_COST


def stats(net: pd.Series) -> dict:
    eq = (1 + net).cumprod()
    years = len(net) / DAYS_PER_YEAR
    ann = lambda x: (1 + x).prod() ** (DAYS_PER_YEAR / max(len(x), 1)) - 1
    half = len(net) // 2
    yearly = net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {"cagr": eq.iloc[-1] ** (1 / years) - 1,
            "sharpe": net.mean() / net.std() * np.sqrt(DAYS_PER_YEAR) if net.std() > 0 else 0.0,
            "max_dd": (eq / eq.cummax() - 1).min(), "h1": ann(net.iloc[:half]), "h2": ann(net.iloc[half:]),
            "worst_year": yearly.min(), **{f"y{y}": v for y, v in yearly.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=int, default=20)
    ap.add_argument("--days", type=int, default=2190)
    args = ap.parse_args()

    pairs = all_usdt_pairs()
    data = {}
    for i, s in enumerate(pairs):
        d = daily(s, args.days)
        if len(d) >= 30:
            data[s] = d
        if i % 100 == 0:
            print(f"fetched {i}/{len(pairs)}", flush=True)
    px = pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()
    qv = pd.DataFrame({s: d["qvol"] for s, d in data.items()}).reindex(px.index)
    rets = px.pct_change(fill_method=None)
    last = px.apply(lambda c: c.last_valid_index())
    # A coin is "dead" on the first bar after its final print if that is before the sample end.
    dead = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for s, t in last.items():
        if t is not None and t < px.index[-1]:
            nxt = px.index[px.index.get_loc(t) + 1]
            dead.loc[nxt, s] = 1.0
    print(f"pairs={len(pairs)} with_data={px.shape[1]} delisted_in_sample={(last < px.index[-1]).sum()} "
          f"{px.index[0].date()}..{px.index[-1].date()}")

    rebal = px.index[::7]
    is_rebal = pd.Series(px.index.isin(rebal), index=px.index)
    liq = qv.rolling(30, min_periods=30).sum()
    alive = px.notna()
    btc = px["BTCUSDT"]
    gates = {"BTC>SMA200d": btc > btc.rolling(200).mean(), "BTC>SMA50d&200d": (btc > btc.rolling(50).mean()) & (btc > btc.rolling(200).mean())}
    vol = rets.rolling(30).std() * np.sqrt(DAYS_PER_YEAR)

    def weekly(target: pd.DataFrame) -> pd.DataFrame:
        w = target.where(is_rebal, axis=0).ffill().fillna(0.0)
        return w.where(alive, 0.0)  # a position in a coin that stopped printing is gone

    results = {"BTC buy&hold": stats(run(pd.DataFrame({"BTCUSDT": 1.0}, index=px.index).reindex(columns=px.columns, fill_value=0.0), rets, dead))}
    ew = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for t in rebal:
        top = liq.loc[t].dropna().nlargest(args.universe).index
        ew.loc[t, top] = 1.0 / len(top) if len(top) else 0.0
    results[f"PIT top{args.universe} EW hold"] = stats(run(weekly(ew), rets, dead))

    for gate_name, gate in gates.items():
        for lb in (14, 30):
            mom = px / px.shift(lb) - 1
            for K in (3, 5, 10):
                for sizing in ("equal", "vol40"):
                    target = pd.DataFrame(0.0, index=px.index, columns=px.columns)
                    for t in rebal:
                        if not gate.loc[t]:
                            continue
                        uni = liq.loc[t].dropna().nlargest(args.universe).index
                        m = mom.loc[t, uni].dropna()
                        if len(m) < K:
                            continue
                        pick = m.nlargest(K).index
                        if sizing == "equal":
                            target.loc[t, pick] = 1.0 / K
                        else:  # inverse-vol weights scaled so the basket targets ~40% annual vol, never levered
                            iv = (1.0 / vol.loc[t, pick]).replace([np.inf], np.nan).dropna()
                            if iv.empty:
                                continue
                            wts = iv / iv.sum()
                            basket_vol = float(np.sqrt((wts ** 2 * vol.loc[t, wts.index] ** 2).sum()))
                            target.loc[t, wts.index] = wts * min(1.0, 0.40 / basket_vol) if basket_vol > 0 else 0.0
                    results[f"mom{lb}d top{K} {sizing} | {gate_name}"] = stats(run(weekly(target), rets, dead))

    df = pd.DataFrame(results).T
    pct = [c for c in df.columns if c != "sharpe"]
    df[pct] = (df[pct].astype(float) * 100).round(1)
    df["sharpe"] = df["sharpe"].astype(float).round(2)
    pd.set_option("display.width", 250)
    print(f"\ncosts {TAKER_BPS:.0f}+{SLIPPAGE_BPS:.0f} bps/side, delist haircut {DELIST_HAIRCUT:.0%}, universe top{args.universe} by 30d volume (PIT)")
    print(df.sort_values("sharpe", ascending=False).to_string())
    df.to_json("research/momentum_pit_study.json", orient="index", indent=1)


if __name__ == "__main__":
    main()
