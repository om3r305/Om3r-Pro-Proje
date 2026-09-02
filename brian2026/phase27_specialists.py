from __future__ import annotations

from .market_structure import MarketStructureFeatures
from .types import SpecialistVote

def _wait(name,reason):
    return SpecialistVote(name,"WAIT",0.,0.,reason,{})

def market_structure_specialist(feature:MarketStructureFeatures)->SpecialistVote:
    if feature.state=="UNKNOWN" or feature.atr is None:
        return _wait("market_structure_specialist","unavailable:confirmed_structure,atr")
    bullish=(feature.bullish_bos or feature.bullish_choch or feature.bullish_sweep)
    bearish=(feature.bearish_bos or feature.bearish_choch or feature.bearish_sweep)
    edge=(.45 if bullish else 0)-(.45 if bearish else 0)
    edge+=(.2 if feature.state=="UPTREND" else -.2 if feature.state=="DOWNTREND" else 0)
    edge=max(-1.,min(1.,edge));action="BUY" if edge>.2 else "SELL" if edge<-.2 else "WAIT"
    used={"bullish_bos":float(feature.bullish_bos),"bearish_bos":float(feature.bearish_bos),
          "bullish_choch":float(feature.bullish_choch),"bearish_choch":float(feature.bearish_choch),
          "bullish_sweep":float(feature.bullish_sweep),"bearish_sweep":float(feature.bearish_sweep)}
    return SpecialistVote("market_structure_specialist",action,0. if action=="WAIT" else .5+.5*abs(edge),edge,
                          f"confirmed structure={feature.state}; candidate evidence only",used)

def dip_specialist(feature:MarketStructureFeatures)->SpecialistVote:
    if feature.dip_score is None or feature.rally_score is None:
        return _wait("dip_specialist","unavailable:dip_or_rally_evidence")
    edge=feature.dip_score-feature.rally_score
    action="BUY" if edge>=.25 else "SELL" if edge<=-.25 else "WAIT"
    return SpecialistVote("dip_specialist",action,0. if action=="WAIT" else .5+.5*abs(edge),edge,
                          "deterministic dip/rally candidate evidence; profitability not assumed",
                          {"dip_score":feature.dip_score,"rally_score":feature.rally_score})

PHASE27_SPECIALISTS=(market_structure_specialist,dip_specialist)
