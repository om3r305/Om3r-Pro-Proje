# -*- coding: utf-8 -*-
from __future__ import annotations
"""
L60 — l60_synthesizer
- Introspector çıktısından yola çıkarak:
  (1) yeni "blok" fonksiyonları icat eder (runtime kayıt),
  (2) yeni bir dokuma spec (FabricSpec) üretir,
  (3) isterse ideas/strategies/*.live.py dosyasına stratejiyi yazar.
Silme yok; hep ekler ya da yeni dosya üretir.
"""
import re, time, json, random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

try:
    from Proje1.core.strategy_fabric import (
        FabricSpec, register_block, list_blocks, weave_and_optionally_write
    )
except Exception as e:
    # Minimal fallback; ama gerçek kullanım için strategy_fabric gerekir.
    FabricSpec = None  # type: ignore

# ------------ Basit "blok sentezi" fikirleri --------------
def _gen_block_name(prefix="novel") -> str:
    t = time.strftime("%H%M%S")
    rnd = random.randint(10, 99)
    return f"{prefix}_{t}_{rnd}"

def make_novel_block_from_policy(policy: Dict[str,Any]) -> List[str]:
    """
    Policy'deki durumlara göre 1-2 yeni blok icat edip runtime'a kaydeder.
    Dönüş: eklenen blok adları.
    """
    added: List[str] = []
    actions = (policy or {}).get("actions", [])
    needs_momentum = any(a.get("type")=="tune_block_weight" and a.get("block")=="price_accel" and a.get("delta",0)>0 for a in actions)
    needs_chop_guard = any(a.get("type")=="tune_block_weight" and a.get("block")=="regime_trend_bonus" and a.get("delta",0)<0 for a in actions)

    if needs_momentum:
        name = _gen_block_name("momentum_mix")
        def _block(px, st, **kw):
            # fiyat hızını ve rejim bonusunu harmanla (sade, bağımsız)
            accel = getattr(st, "accel", 0.0)
            v = float(getattr(st, "vol_norm", 0.5))
            reg = (getattr(st, "_last_reg", {}) or {}).get("regime","UNKNOWN")
            trend = 0.8 if reg=="TREND" else (0.2 if reg=="CHOP" else 0.5)
            # normalize hız ~[-0.02,0.02], vol 0..1, trend 0..1
            def _norm(a, lo=-0.02, hi=0.02):
                if hi==lo: return 0.5
                x = (a - lo)/float(hi-lo)
                return max(0.0, min(1.0, x))
            s = 0.5*_norm(accel) + 0.3*v + 0.2*trend
            return max(0.0, min(1.0, s))
        register_block(name, _block)
        added.append(name)

    if needs_chop_guard:
        name = _gen_block_name("anti_chop")
        def _block(px, st, **kw):
            reg = (getattr(st, "_last_reg", {}) or {}).get("regime","UNKNOWN")
            return 0.2 if reg=="CHOP" else 0.6  # CHOP'ta baskıla
        register_block(name, _block)
        added.append(name)

    # Eğer hiç eylem yoksa bile, tek bir deneysel blok ekleyelim (entegre çalışsın):
    if not added:
        name = _gen_block_name("exp")
        def _block(px, st, **kw):
            # çok basit: vol_norm ve random jitter
            import random as _r
            v = float(getattr(st, "vol_norm", 0.5))
            return max(0.0, min(1.0, 0.8*v + 0.2*_r.random()))
        register_block(name, _block)
        added.append(name)

    return added

def synthesize_strategy(policy: Dict[str,Any], *,
                        base_name: str = "L60_Weaved",
                        target_slot: str = "pred",
                        threshold: float = 0.64,
                        write_live: bool = True) -> Dict[str,Any]:
    """
    - policy'den yeni blok(lar) üretir,
    - mevcut blok listesi + yenileri ile bir FabricSpec kurar,
    - ideas/strategies/{name}.live.py yazabilir.
    """
    if FabricSpec is None:
        return {"ok": False, "reason": "strategy_fabric_missing"}

    new_blocks = make_novel_block_from_policy(policy)
    # basit bir kombinasyon: mevcut kritik bloklar + yeniler
    candidate_blocks = ["price_accel", "vol_norm", "regime_trend_bonus"] + new_blocks

    spec = FabricSpec(
        name=f"{base_name}_{time.strftime('%Y%m%d_%H%M%S')}",
        slot=target_slot,
        blocks=candidate_blocks,
        combine="weighted",
        weights=[0.30, 0.25, 0.20] + [0.25/ max(1,len(new_blocks))]*len(new_blocks),
        threshold=threshold,
        base_conf=0.52,
        reason="l60_synth"
    )
    path = weave_and_optionally_write(spec, write_live=write_live)
    return {
        "ok": True,
        "spec": spec.__dict__,
        "live_path": path,
        "new_blocks": new_blocks,
        "all_blocks": candidate_blocks
    }
