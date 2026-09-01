# -*- coding: utf-8 -*-
from __future__ import annotations
import random, time
from typing import Dict, Tuple

class Antifragile:
    def __init__(self, cfg: Dict):
        qr = cfg.get("qrng", {}) or {}
        self.qrng_enabled = bool(qr.get("enabled", False))
        self.qrng_weight  = float(qr.get("weight", 0.05))
        self._last = 0.0
        self._predictability = 0.0  # 0..1 (basit sayaç)

    def adjust(self, symbol: str) -> Tuple[float, str]:
        """
        predictability yüksekse → küçük random kırıcı ekle.
        """
        now = time.time()
        dt = max(1.0, now - self._last)
        self._last = now
        # kaba sim: tahmin edilebilirliği biraz yukarı sürükle, sonra çürüt
        self._predictability = max(0.0, min(1.0, self._predictability * 0.98 + 0.02))
        # rastgele hafif kırpma
        jitter = (random.random() - 0.5) * 0.02  # ±0.01
        boost = 0.0
        note = f"pred={self._predictability:.2f},j={jitter:+.3f}"
        # predictability yüksekse boz
        if self._predictability > 0.6:
            boost += jitter
        # qrng etkisi (kapalıysa da 0)
        if self.qrng_enabled:
            boost += (random.random()-0.5) * self.qrng_weight
            note += f",qr={self.qrng_weight:.2f}"
        return boost, note
