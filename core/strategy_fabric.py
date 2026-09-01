# -*- coding: utf-8 -*-
from __future__ import annotations
"""
L60 — strategy_fabric
- Mevcut "blok"lardan (indikator, filtre, giriş/çıkış kuralı) strateji dokur.
- Dinamik StrategyPlugin sınıfları üretir (runtime) ve istenirse *.live.py dosyası yazar (silme yok).
- Tek başına import edildiğinde güvenli no-op çalışır.
"""
import math, json, time, textwrap, inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

# StrategyPlugin tipini opsiyonel import et
try:
    from Proje1.core.strategy_api import StrategyPlugin, StrategySignal  # type: ignore
except Exception:
    from dataclasses import dataclass
    from typing import Literal
    Slot = "pred"
    @dataclass
    class StrategySignal:  # fallback
        symbol: str
        slot: str
        fired: bool
        confidence: float
        reason: str
    class StrategyPlugin:   # fallback
        name: str = "base"
        slot: str = "pred"
        def __init__(self, cfg: Dict[str,Any]): self.cfg = cfg
        def on_symbol(self, symbol, st, market):  # pragma: no cover
            return StrategySignal(symbol, self.slot, False, 0.0, "noop")

IDEAS_DIR = Path("ideas/strategies")
IDEAS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Basit blok kütüphanesi --------------------
# (Bu bloklar başlangıç seti; Synthesizer yeni bloklar ekleyebilir.)
Registry: Dict[str, Callable[..., float]] = {}

def register_block(name: str, fn: Callable[..., float]) -> None:
    """Bir skor/özellik üreten blok kaydet (0..1 aralığına normalize etmeye çalış)."""
    Registry[name] = fn

# Örnek bazı bloklar (indikatör benzeri, dependency-free)
def _norm(v: float, lo=-1.0, hi=1.0) -> float:
    if hi == lo: return 0.5
    x = (v - lo) / float(hi - lo)
    return max(0.0, min(1.0, x))

def block_price_accel(px: float, st, **kw) -> float:
    a = getattr(st, "accel", 0.0)
    return _norm(a, lo=-0.02, hi=0.02)

def block_vol_norm(px: float, st, **kw) -> float:
    v = float(getattr(st, "vol_norm", 0.5))
    return max(0.0, min(1.0, v))

def block_regime_trend_bonus(px: float, st, **kw) -> float:
    reg = (getattr(st, "_last_reg", {}) or {}).get("regime", "UNKNOWN")
    return 0.8 if reg == "TREND" else (0.2 if reg == "CHOP" else 0.5)

# Kütüphane başlangıç kayıtları
register_block("price_accel", block_price_accel)
register_block("vol_norm", block_vol_norm)
register_block("regime_trend_bonus", block_regime_trend_bonus)

# -------------------- Dokuma (weave) arabirimi --------------------
@dataclass
class FabricSpec:
    name: str
    slot: str
    blocks: List[str]              # hangi bloklar toplanacak
    combine: str = "weighted"      # 'weighted' | 'min' | 'max' | 'mean'
    weights: Optional[List[float]] = None
    threshold: float = 0.62        # sinyal eşiği
    base_conf: float = 0.50
    reason: str = "fabric"

def _combine(values: List[float], combine: str, weights: Optional[List[float]]) -> float:
    if not values:
        return 0.0
    if combine == "min":
        return min(values)
    if combine == "max":
        return max(values)
    if combine == "mean":
        return sum(values)/len(values)
    # weighted (default)
    if not weights or len(weights) != len(values):
        return sum(values)/len(values)
    s = sum(w*v for w, v in zip(weights, values))
    ws = sum(weights)
    return s/ws if ws != 0 else sum(values)/len(values)

def weave_runtime_class(spec: FabricSpec):
    """FabricSpec'ten runtime StrategyPlugin sınıfı üret."""
    name = spec.name
    slot_val = spec.slot

    class _Woven(StrategyPlugin):
        NAME = name
        slot = slot_val
        def __init__(self, cfg: Dict[str,Any]):  # cfg ile override imkânı
            super().__init__(cfg)
            self.spec = spec

        def on_symbol(self, symbol: str, st, market) -> StrategySignal:
            px = float(getattr(st, "last_px", 0.0) or 0.0)
            vals = []
            miss = []
            for b in self.spec.blocks:
                fn = Registry.get(b)
                try:
                    vals.append(float(fn(px, st, spec=self.spec)))
                except Exception:
                    miss.append(b)
                    vals.append(0.0)

            score = _combine(vals, self.spec.combine, self.spec.weights)
            fired = bool(score >= self.spec.threshold)
            conf = self.spec.base_conf + 0.4 * max(0.0, score - self.spec.threshold)
            conf = max(0.0, min(1.0, conf))
            rsn = f"{self.spec.reason}; score={score:.2f}; blocks={self.spec.blocks}"
            if miss:
                rsn += f"; missing={miss}"
            return StrategySignal(symbol, self.spec.slot, fired, conf, rsn)

    _Woven.__name__ = f"Woven_{name}"
    return _Woven

def weave_and_optionally_write(spec: FabricSpec, write_live: bool = True) -> str:
    """
    Spec'e göre runtime sınıfı üretir.
    write_live=True ise ideas/strategies/{name}.live.py şeklinde KOD YAZAR (silme yok).
    Dönüş: üretilen sınıfın CANLI dosya yolu (veya boş string).
    """
    cls = weave_runtime_class(spec)
    if not write_live:
        return ""
    code = f'''# -*- coding: utf-8 -*-
# AUTOGEN by strategy_fabric at {time.strftime("%Y-%m-%d %H:%M:%S")}
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
try:
    from Proje1.core.strategy_api import StrategyPlugin, StrategySignal
except Exception:
    from dataclasses import dataclass
    @dataclass
    class StrategySignal:
        symbol: str; slot: str; fired: bool; confidence: float; reason: str
    class StrategyPlugin:
        slot: str = "pred"
        def __init__(self, cfg): self.cfg = cfg

Spec = {json.dumps(spec.__dict__, ensure_ascii=False, indent=2)}

def _combine(vals, combine, weights):
    if not vals: return 0.0
    if combine == "min": return min(vals)
    if combine == "max": return max(vals)
    if combine == "mean": return sum(vals)/len(vals)
    if (not weights) or len(weights) != len(vals): return sum(vals)/len(vals)
    s = sum(w*v for w, v in zip(weights, vals)); ws = sum(weights)
    return (s/ws) if ws else sum(vals)/len(vals)

# Basit yerleşik bloklar (runtime bağımsızlık için inline)
def _norm(v, lo=-1.0, hi=1.0):
    if hi == lo: return 0.5
    x = (v - lo)/float(hi-lo)
    return max(0.0, min(1.0, x))

def price_accel(px, st, **kw):
    a = getattr(st, "accel", 0.0)
    return _norm(a, lo=-0.02, hi=0.02)

def vol_norm(px, st, **kw):
    v = float(getattr(st, "vol_norm", 0.5))
    return max(0.0, min(1.0, v))

def regime_trend_bonus(px, st, **kw):
    reg = (getattr(st, "_last_reg", {{}}) or {{}}).get("regime", "UNKNOWN")
    return 0.8 if reg == "TREND" else (0.2 if reg == "CHOP" else 0.5)

LocalBlocks = {{
    "price_accel": price_accel,
    "vol_norm": vol_norm,
    "regime_trend_bonus": regime_trend_bonus,
}}

class {cls.__name__}(StrategyPlugin):
    NAME = Spec.get("name", "Woven")
    slot = Spec.get("slot", "pred")

    def __init__(self, cfg: Dict[str,Any]):
        super().__init__(cfg)
        self.spec = Spec

    def on_symbol(self, symbol: str, st, market) -> StrategySignal:
        px = float(getattr(st, "last_px", 0.0) or 0.0)
        vals = []
        miss = []
        for b in self.spec.get("blocks", []):
            fn = LocalBlocks.get(b)
            try:
                vals.append(float(fn(px, st, spec=self.spec)))
            except Exception:
                miss.append(b); vals.append(0.0)
        score = _combine(vals, self.spec.get("combine","weighted"), self.spec.get("weights"))
        fired = bool(score >= float(self.spec.get("threshold", 0.62)))
        conf = float(self.spec.get("base_conf", 0.50)) + 0.4*max(0.0, score - float(self.spec.get("threshold", 0.62)))
        conf = max(0.0, min(1.0, conf))
        rsn = f"fabric; score={{score:.2f}}; blocks={{self.spec.get('blocks')}}"
        if miss: rsn += f"; missing={{miss}}"
        return StrategySignal(symbol, self.spec.get("slot","pred"), fired, conf, rsn)
'''
    path = IDEAS_DIR / f"{spec.name}.live.py"
    if not path.exists():
        path.write_text(code, encoding="utf-8")
    else:
        # Silme yok: var olanın altına yeni timestamp'li yorum ekleyelim.
        with path.open("a", encoding="utf-8") as f:
            f.write("\n# --- touched at %s ---\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    return str(path)

def list_blocks() -> List[str]:
    return sorted(list(Registry.keys()))
