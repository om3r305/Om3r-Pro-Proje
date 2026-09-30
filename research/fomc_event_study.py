"""BTC around FOMC statements (2021-2026), official dates from federalreserve.gov.

For each scheduled statement (last meeting day, 14:00 America/New_York) measure
BTC returns in fixed windows relative to the statement hour, and compare against
the same clock windows on every non-FOMC day (unconditional baseline). Reports
mean, t-stat vs baseline (Welch) and a sign test, plus the mean absolute move to
show whether the event carries direction or just volatility. Net of a 30 bps
round trip for the "trade it" line.

Usage: python research/fomc_event_study.py [fomccalendars.html]
"""
from __future__ import annotations

import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                       "August", "September", "October", "November", "December"], 1)}
ABBR = {m[:3]: i for m, i in MONTHS.items()}
NY = ZoneInfo("America/New_York")
ROUND_TRIP = 0.0030
CACHE = Path(__file__).with_name(".cache")


def fomc_statements(html: str) -> list[pd.Timestamp]:
    years = [(m.start(), int(m.group(1))) for m in re.finditer(r"(\d{4}) FOMC Meetings", html)]
    out = []
    for i, (pos, y) in enumerate(years):
        seg = html[pos: years[i + 1][0] if i + 1 < len(years) else len(html)]
        for month, days in re.findall(r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<", seg, re.S):
            if "notation" in days.lower() or "-" not in days:
                continue  # unscheduled notation votes have no statement press cycle
            last_day = int(re.match(r"\d+-(\d+)", days.strip()).group(1))
            month = month.strip()
            m = ABBR[month.split("/")[-1][:3]] if "/" in month else MONTHS[month]
            local = datetime(y, m, last_day, 14, 0, tzinfo=NY)
            out.append(pd.Timestamp(local).tz_convert("UTC"))
    return sorted(out)


def btc_hourly(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    cached = CACHE / f"BTCUSDT_1h_{start.date()}_{end.date()}.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    rows, t = [], int(start.timestamp() * 1000)
    stop = int(end.timestamp() * 1000)
    while t < stop:
        k = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": "BTCUSDT", "interval": "1h", "startTime": t, "limit": 1000}, timeout=30).json()
        if not k:
            break
        rows += k
        t = k[-1][0] + 3_600_000
        time.sleep(0.05)
    s = pd.Series([float(r[1]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True), name="open")
    s = s[~s.index.duplicated()]
    CACHE.mkdir(exist_ok=True)
    s.to_pickle(cached)
    return s


def window_return(open_px: pd.Series, anchor: pd.Timestamp, a: int, b: int) -> float:
    """Return from the open of hour anchor+a to the open of hour anchor+b."""
    t0, t1 = anchor + pd.Timedelta(hours=a), anchor + pd.Timedelta(hours=b)
    if t0 not in open_px.index or t1 not in open_px.index:
        return np.nan
    return open_px[t1] / open_px[t0] - 1


def main() -> None:
    html_path = sys.argv[1] if len(sys.argv) > 1 else None
    html = Path(html_path).read_text(encoding="utf8", errors="ignore") if html_path else \
        requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    now = pd.Timestamp.now(tz="UTC")
    events = [e for e in fomc_statements(html) if e < now - pd.Timedelta(days=2)]
    px = btc_hourly(events[0] - pd.Timedelta(days=3), now)
    print(f"FOMC statements: {len(events)}  {events[0].date()} .. {events[-1].date()}  BTC hourly bars: {len(px)}")

    windows = {"pre 24h -> statement": (-24, 0), "statement -> +2h": (0, 2), "+2h -> +24h": (2, 24), "statement -> +24h": (0, 24)}
    event_days = {e.normalize() for e in events}
    rows = []
    for name, (a, b) in windows.items():
        ev = pd.Series([window_return(px, e, a, b) for e in events]).dropna()
        # Baseline: same UTC clock hour on every other day (hour follows each event's DST offset, so use both).
        base = []
        for hour in sorted({e.hour for e in events}):
            for d in pd.date_range(px.index[0].normalize() + pd.Timedelta(days=2), now.normalize() - pd.Timedelta(days=2), freq="D"):
                if d in event_days:
                    continue
                base.append(window_return(px, d + pd.Timedelta(hours=hour), a, b))
        base = pd.Series(base).dropna()
        diff = ev.mean() - base.mean()
        se = np.sqrt(ev.var(ddof=1) / len(ev) + base.var(ddof=1) / len(base))
        rows.append({"window": name, "n_events": len(ev), "event_mean_%": ev.mean() * 100, "base_mean_%": base.mean() * 100,
                     "t_vs_base": diff / se, "up_share": (ev > 0).mean(), "event_abs_%": ev.abs().mean() * 100,
                     "base_abs_%": base.abs().mean() * 100, "abs_ratio": ev.abs().mean() / base.abs().mean(),
                     "trade_long_net_%": (ev.mean() - ROUND_TRIP) * 100})
    df = pd.DataFrame(rows).set_index("window").round(3)
    pd.set_option("display.width", 220)
    print(df.to_string())
    df.to_json(Path(__file__).with_name("fomc_event_study.json"), orient="index", indent=1)


if __name__ == "__main__":
    main()
