# knowledge/vector_store.py
from __future__ import annotations
import json, re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional
from knowledge.schema import Finding

STORE = Path("knowledge"); STORE.mkdir(parents=True, exist_ok=True)
DB = STORE / "findings.jsonl"

WORD = re.compile(r"[A-Za-z0-9_]+")

def _tok(s: str):
    return [w.lower() for w in WORD.findall(s or "") if len(w) > 2]

def add_finding(f: Finding) -> None:
    with DB.open("a", encoding="utf-8") as fobj:
        fobj.write(json.dumps(f, ensure_ascii=False) + "\n")

def quick_query(symbol: str, horizon_min: int = 60) -> Optional[Dict[str, float]]:
    """Basit: son kayıtlar içinden sembol geçenleri topla, impact/confidence ortalaması."""
    if not DB.exists(): return None
    sym = symbol.upper()
    imp = []; conf=[]
    with DB.open("r", encoding="utf-8") as f:
        for ln in f.readlines()[-1000:]:
            try:
                obj = json.loads(ln)
                if sym in [s.upper() for s in obj.get("symbols",[])]:
                    imp.append(float(obj.get("impact", 0.0)))
                    conf.append(float(obj.get("confidence", 0.0)))
            except Exception: pass
    if not imp: return None
    return {"impact": sum(imp)/len(imp), "confidence": sum(conf)/len(conf)}
