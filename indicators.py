# indicators.py — temel indikatörler (deps: numpy, pandas)
import numpy as np
import pandas as pd

def ema(series, n: int):
    return pd.Series(series).ewm(span=max(1, int(n)), adjust=False).mean().to_numpy()

def rsi(series, n: int = 14):
    s = pd.Series(series, dtype=float)
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_dn = dn.ewm(span=n, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0).to_numpy()

def bollinger(series, n: int = 20, k: float = 2.0):
    s = pd.Series(series, dtype=float)
    ma = s.rolling(window=n, min_periods=n).mean()
    sd = s.rolling(window=n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return (ma.to_numpy(), upper.to_numpy(), lower.to_numpy())

def zscore(series, n: int = 50):
    s = pd.Series(series, dtype=float)
    r = s.pct_change()
    m = r.rolling(n, min_periods=n).mean()
    sd = r.rolling(n, min_periods=n).std(ddof=0)
    z = (r - m) / (sd.replace(0, np.nan))
    return z.fillna(0.0).to_numpy()
