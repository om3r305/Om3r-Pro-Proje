# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, random

# Dış veri anahtarların yoksa hemen nötr dönüyoruz
CMC = os.getenv("CMC_API_KEY", "")
NEWS = os.getenv("NEWS_API_KEY", "")
TW   = os.getenv("TWITTER_BEARER", "")

_last = {"t": 0.0, "scores": {"news_shock": 0.0, "macro_risk": 1.0, "flow": "neutral"}}

def get_scores(symbol: str, cfg: dict) -> dict:
    """
    Döndürdüğü sözlük: {"news_shock":0..1, "macro_risk":~1.0 nötr, "flow":"pos|neg|neutral"}
    Anahtar yoksa nötr; çok hafif jitter ile.
    """
    ext_cfg = cfg.get("external", {})
    if not ext_cfg.get("enabled", True):
        return {"news_shock": 0.0, "macro_risk": 1.0, "flow": "neutral"}

    # 15 sn cache
    now = time.time()
    if now - _last["t"] < 15:
        return _last["scores"]

    # Basit/emin fallback — gerçek entegrasyon eklenene kadar nötr + jitter
    # (İleride CMC/News/Twitter API ile doldurulacak)
    news_shock = 0.0
    macro_risk = 1.0
    flow = "neutral"

    # Çok küçük rastgelelik: 0.0–0.05 arası
    news_shock += random.random() * 0.05
    macro_risk += (random.random() - 0.5) * 0.1  # 0.95–1.05
    flow = random.choice(["neutral","neutral","pos","neg"])  # çoğu nötr

    scores = {"news_shock": round(min(max(news_shock,0.0),1.0), 3),
              "macro_risk": round(max(macro_risk, 0.7), 3),
              "flow": flow}
    _last["t"] = now
    _last["scores"] = scores
    return scores
