# ai_predictor.py — hafif AI filtresi (eğitim gerektirmez, ağırlıklar config'ten)
import numpy as np
import requests
from indicators import ema, rsi, bollinger, zscore

def _klines(symbol: str, interval="1m", limit=120):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=5)
    r.raise_for_status()
    data = r.json()
    closes = np.array([float(x[4]) for x in data], dtype=float)
    highs  = np.array([float(x[2]) for x in data], dtype=float)
    lows   = np.array([float(x[3]) for x in data], dtype=float)
    vols   = np.array([float(x[5]) for x in data], dtype=float)
    return closes, highs, lows, vols

class AiPredictor:
    """
    Logistic benzeri bir skor: sigmoid(w·x). Ağırlıklar config'ten okunur; yoksa mantıklı varsayılanlar.
    """
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.interval = cfg.get("interval", "1m")
        self.limit    = int(cfg.get("limit", 120))
        self.min_prob = float(cfg.get("min_prob", 0.55))
        # Ağırlıklar
        w = cfg.get("weights", {})
        self.w = {
            "ret1":  0.9,
            "ret3":  0.6,
            "rsi":   0.4,
            "ema9":  0.5,
            "ema21": 0.3,
            "bb_pos":0.5,
            "z":     0.6,
            "vol":   0.2,
        } | {k: float(v) for k, v in w.items()}

    @staticmethod
    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

    def score(self, symbol: str):
        c, h, l, v = _klines(symbol, self.interval, self.limit)
        if c.size < 30:
            return 0.5, {"why": "no-data"}

        # Özellikler
        ret1 = (c[-1] / c[-2] - 1.0)
        ret3 = (c[-1] / c[-4] - 1.0) if c.size >= 4 else 0.0
        e9   = ema(c, 9)[-1];  e21 = ema(c, 21)[-1]
        r    = rsi(c, 14)[-1]
        ma20, ub, lb = bollinger(c, 20, 2.0)
        bb_pos = 0.0
        if not np.isnan(ub[-1]) and not np.isnan(lb[-1]) and (ub[-1] - lb[-1]) > 0:
            bb_pos = (c[-1] - lb[-1]) / (ub[-1] - lb[-1])  # 0..1 arası
        z = zscore(c, 50)[-1]
        vol = (v[-1] / (np.mean(v[-21:-1]) + 1e-12)) if v.size >= 22 else 1.0

        # Normalize & w·x
        x = (
            self.w["ret1"]  * (ret1 * 50.0) +
            self.w["ret3"]  * (ret3 * 20.0) +
            self.w["rsi"]   * ((r - 50.0) / 20.0) +
            self.w["ema9"]  * ((c[-1] - e9) / (e9 + 1e-12) * 10.0) +
            self.w["ema21"] * ((c[-1] - e21) / (e21 + 1e-12) * 10.0) +
            self.w["bb_pos"]* (bb_pos - 0.5) * 2.0 +
            self.w["z"]     * z +
            self.w["vol"]   * (vol - 1.0)
        )
        prob = float(self._sigmoid(x))
        why = f"AI p={prob:.2f} | r1={ret1:.3f} r3={ret3:.3f} rsi={r:.1f} volx={vol:.2f} z={z:.2f} bb={bb_pos:.2f}"
        return prob, {"why": why}
