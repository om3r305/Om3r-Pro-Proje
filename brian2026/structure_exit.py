from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .market_structure import MarketStructureFeatures

ExitMode=Literal["fixed","structure","hybrid"]

@dataclass(frozen=True,slots=True)
class StructureExitConfig:
    mode:ExitMode="fixed"
    exit_on_choch:bool=True
    exit_on_bos:bool=True
    exit_on_zone_failure:bool=True
    exit_on_momentum_deterioration:bool=True

def apply_structure_exit(actions:Sequence[str],features:Sequence[MarketStructureFeatures],
                         config:StructureExitConfig=StructureExitConfig())->tuple[str,...]:
    """Overlay close signals; fixed mode is byte-for-byte action preserving.

    Protective stop/target/max-hold behavior remains owned by the unchanged
    Phase 2.5 portfolio simulator.
    """
    if len(actions)!=len(features):raise ValueError("actions/features must align")
    if config.mode=="fixed":return tuple(actions)
    out=[];side=None
    for action,feature in zip(actions,features):
        exit_long=(config.exit_on_choch and feature.bearish_choch) or (config.exit_on_bos and feature.bearish_bos) or (config.exit_on_zone_failure and feature.failed_breakdown) or (config.exit_on_momentum_deterioration and feature.momentum_deceleration is True)
        exit_short=(config.exit_on_choch and feature.bullish_choch) or (config.exit_on_bos and feature.bullish_bos) or (config.exit_on_zone_failure and feature.failed_breakout) or (config.exit_on_momentum_deterioration and feature.momentum_deceleration is True)
        if side=="LONG" and exit_long:out.append("SELL");side=None
        elif side=="SHORT" and exit_short:out.append("BUY");side=None
        else:
            out.append(action)
            if side is None and action in ("BUY","SELL"):side="LONG" if action=="BUY" else "SHORT"
    return tuple(out)
