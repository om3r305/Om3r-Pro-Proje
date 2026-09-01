# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

class ConfidenceCalibrator:
    """
    Hafif kalibrasyon: logistic benzeri sıkıştırma + offset.
    cfg: modules.confidence_calibrator.{gain,offset,lo,hi}
    """
    def __init__(self, cfg: Dict[str,Any] | None):
        self.cfg = cfg or {}
        cc = (self.cfg.get("modules") or {}).get("confidence_calibrator", {})
        self.gain   = float(cc.get("gain", 1.15))
        self.offset = float(cc.get("offset", -0.02))
        self.lo     = float(cc.get("lo", 0.05))
        self.hi     = float(cc.get("hi", 0.98))

    def calibrate(self, p: float) -> float:
        try:
            x = p*self.gain + self.offset
            x = 1.0/(1.0 + pow(2.718281828, -4.0*(x-0.5)))  # soft logistic squeeze
            return max(self.lo, min(self.hi, x))
        except Exception:
            return max(self.lo, min(self.hi, p))
