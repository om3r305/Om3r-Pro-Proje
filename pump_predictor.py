# pump_predictor.py — “yakında patlar mı?” olasılığı (0..1)
# Sinyaller: squeeze->expand (BB/Keltner), hacim z-skoru, ivme, tepe kırılımı
import math, time, requests
from typing import List, Dict, Tuple

def _sma(x: List[float], n: int) -> float:
    if not x: return 0.0
    n = min(n, len(x))
    return sum(x[-n:]) / n

def _std(x: List[float], n: int) -> float:
    if len(x) < n or n <= 1: return 0.0
    s = x[-n:]; m = sum(s)/n
    return (sum((v-m)**2 for v in s)/n) ** 0.5

def _atr(h: List[float], l: List[float], c: List[float], n: int) -> float:
    if len(c) < n+1: return 0.0
    trs=[]
    for i in range(1, n+1):
        tr = max(h[-i]-l[-i], abs(h[-i]-c[-i-1]), abs(l[-i]-c[-i-1]))
        trs.append(tr)
    return sum(trs)/n

class PumpPredictor:
    def __init__(self, symbol: str, cfg: Dict):
        self.symbol = symbol
        self.interval = cfg.get("interval","1m")
        self.limit = int(cfg.get("limit",120))
        self.refresh_sec = int(cfg.get("refresh_sec",20))
        self.w = cfg.get("weights", {"squeeze":0.25,"vol":0.35,"accel":0.20,"break":0.20})
        self._last_pull = 0.0
        self._klines = []

    def _pull(self):
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": self.symbol, "interval": self.interval, "limit": self.limit},
                         timeout=5)
        r.raise_for_status()
        self._klines = r.json()
        self._last_pull = time.time()

    def _ensure(self):
        if not self._klines or time.time()-self._last_pull >= self.refresh_sec:
            try: self._pull()
            except Exception: pass

    def score(self) -> Tuple[float, Dict]:
        self._ensure()
        if len(self._klines) < 40:
            return 0.0, {"why": "data<40", "hh": 0.0}
        highs = [float(k[2]) for k in self._klines]
        lows  = [float(k[3]) for k in self._klines]
        closes= [float(k[4]) for k in self._klines]
        vols  = [float(k[5]) for k in self._klines]

        ma20 = _sma(closes, 20); sd20 = _std(closes, 20)
        bb_width = 0.0 if ma20==0 else (4.0*sd20)/ma20
        atr20 = _atr(highs, lows, closes, 20); keltn = 0.0 if ma20==0 else (atr20/ma20)
        ratio = (bb_width/keltn) if keltn>0 else 1.5
        squeeze = max(0.0, min(1.0, 1.5 - ratio))
        prev_sd = _std(closes[:-5] if len(closes)>25 else closes[:-3], 20)
        bb_prev = 0.0 if ma20==0 else (4.0*prev_sd)/ma20
        expand = max(0.0, min(1.0, (bb_width - bb_prev) / (bb_prev + 1e-9)))
        squeeze_comp = 0.5*squeeze + 0.5*expand

        vm=_sma(vols,20); vs=_std(vols,20)
        v_z = 0.0 if vs==0 else (vols[-1]-vm)/vs
        v_score = max(0.0, min(1.0, (v_z-1.0)/3.0))

        roc1 = 0.0 if closes[-2]==0 else (closes[-1]/closes[-2]-1.0)
        roc3 = 0.0 if closes[-4]==0 else (closes[-1]/closes[-4]-1.0)
        accel = max(0.0, min(1.0, (roc1 + 0.5*roc3) / 0.01))  # ~%1 → 1.0

        hh = max(highs[-15:])
        brk = 1.0 if closes[-1] >= hh*1.001 else 0.0

        prob = (self.w.get("squeeze",0.25)*squeeze_comp +
                self.w.get("vol",0.35)*v_score +
                self.w.get("accel",0.20)*accel +
                self.w.get("break",0.20)*brk)
        prob = max(0.0, min(1.0, prob))
        return prob, {"why": f"s={squeeze_comp:.2f} v={v_score:.2f} a={accel:.2f} b={brk:.2f}",
                      "hh": hh, "last": closes[-1]}
