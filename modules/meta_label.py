# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

TRADES_FULL = Path("logs/trades_full_log.csv")
MIN_OBS = 60  # min kapanış

class MetaLabel:
    """
    Basit ve güvenli meta-label filtresi:
      - Son lookback kapanışlarından slot bazlı WR ve PF çıkarır.
      - Rejim uyumlu ağırlık (TREND/CHOP) uygular.
      - Eşikleri config'ten okur; yoksa makul defaults.
    """
    def __init__(self, cfg: Dict[str, Any] | None):
        self.cfg = cfg or {}
        mcfg = (self.cfg.get("modules") or {}).get("meta_label", {})
        self.lookback = int(mcfg.get("lookback_trades", 400))
        self.wr_min = float(mcfg.get("wr_min", 0.56))
        self.pf_min = float(mcfg.get("pf_min", 1.10))
        self.reg_weights = dict(mcfg.get("reg_weights", {"TREND":1.0, "CHOP":0.8, "UNKNOWN":0.9}))

    def _rows(self) -> List[Dict[str,Any]]:
        if not TRADES_FULL.exists(): return []
        out = []
        with TRADES_FULL.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd: out.append(r)
        if self.lookback and len(out) > self.lookback:
            out = out[-self.lookback:]
        return out

    @staticmethod
    def _slot_of(r: Dict[str,Any]) -> str:
        for k in ("slot","Slot","label","reason","enter_reason"):
            v = (r.get(k) or "").lower()
            if "dip" in v: return "dip"
            if "pred" in v: return "pred"
            if "news" in v: return "news"
            if "ob" in v: return "ob"
        return "any"

    def _metrics(self, rows: List[Dict[str,Any]], slot: str, regime: str) -> Tuple[float,float,int]:
        closes = []
        for r in rows:
            side = (r.get("side") or r.get("event") or "").upper()
            if side not in ("SELL","CLOSE"): continue
            s = self._slot_of(r)
            if slot != "any" and s != slot: continue
            try: pnl = float(r.get("pnl") or 0.0)
            except Exception: pnl = 0.0
            closes.append(pnl)
        n = len(closes)
        if n == 0: return 0.5, 1.0, 0
        wr = sum(1 for x in closes if x>0)/float(n)
        pos = sum(x for x in closes if x>0) or 1e-9
        neg = -sum(x for x in closes if x<0) or 1e-9
        pf = pos/neg
        # rejim ağırlığı
        w = float(self.reg_weights.get(regime, self.reg_weights.get("UNKNOWN", 0.9)))
        wr = max(0.0, min(1.0, wr*w + (1-w)*wr*0.9))
        return wr, pf, n

    def predict(self, symbol: str, slot: str, regime: str) -> Tuple[bool,float,Dict[str,float]]:
        rows = self._rows()
        wr, pf, n = self._metrics(rows, slot, regime)
        ok = (n>=MIN_OBS) and (wr>=self.wr_min) and (pf>=self.pf_min)
        return ok, wr, {"wr":wr, "pf":pf, "n":n}
