# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
MODEL.mkdir(parents=True, exist_ok=True)
EVO = MODEL / "evo_champions.jsonl"

def save_champion(genome: Dict[str,Any], metrics: Dict[str,Any]) -> None:
    """
    Başarılı (PF, WR, MaxDD) sonuç üretmiş bir genomu kaydeder.
    Örn: genome={"slot":"ob","tp_pct":0.08,"sl_pct":-0.05, ...}
         metrics={"pf":1.45,"wr":0.61,"trades":120,"symbol":"AVAXUSDT"}
    """
    try:
        rec = {"ts": time.time(), "genome": genome, "metrics": metrics}
        with EVO.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
