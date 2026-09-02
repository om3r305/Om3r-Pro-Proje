from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal, Sequence
import json
import math
import statistics

from .portfolio import DEVELOPMENT_CUTOFF

State = Literal["UPTREND","DOWNTREND","RANGE","TRANSITION","UNKNOWN"]

@dataclass(frozen=True,slots=True)
class StructureConfig:
    left_bars:int=3;right_bars:int=3;atr_period:int=14;break_buffer_atr:float=.15
    zone_width_atr:float=.35;near_zone_atr:float=.75;volume_period:int=20
    def __post_init__(self):
        if min(self.left_bars,self.right_bars,self.atr_period,self.volume_period)<1:raise ValueError("windows must be positive")

@dataclass(frozen=True,slots=True)
class StructureCandle:
    close_timestamp:float;open:float;high:float;low:float;close:float;volume:float
    def __post_init__(self):
        if self.close_timestamp>=DEVELOPMENT_CUTOFF:raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
        if min(self.open,self.high,self.low,self.close)<=0 or self.low>min(self.open,self.close) or self.high<max(self.open,self.close):raise ValueError("invalid OHLC")
        if self.volume<0:raise ValueError("negative volume")

@dataclass(frozen=True,slots=True)
class ConfirmedSwing:
    kind:Literal["HIGH","LOW"];pivot_index:int;pivot_timestamp:float;confirmation_index:int
    confirmation_timestamp:float;price:float;rsi:float|None;label:str;provenance_id:str

@dataclass(frozen=True,slots=True)
class MarketStructureFeatures:
    timestamp:float;state:State;latest_swing_high:float|None;latest_swing_low:float|None
    latest_high_label:str|None;latest_low_label:str|None
    bullish_bos:bool;bearish_bos:bool;bullish_choch:bool;bearish_choch:bool
    nearest_support:float|None;nearest_resistance:float|None
    support_distance_atr:float|None;resistance_distance_atr:float|None
    support_age_bars:int|None;resistance_age_bars:int|None
    support_reactions:int|None;resistance_reactions:int|None
    inside_support_zone:bool|None;inside_resistance_zone:bool|None
    bullish_sweep:bool;bearish_sweep:bool;failed_breakdown:bool;failed_breakout:bool
    bullish_breakout_retest:bool;bearish_breakout_retest:bool
    bullish_rsi_divergence:bool;bearish_rsi_divergence:bool
    atr:float|None;rsi:float|None;body_range_ratio:float;upper_wick_ratio:float
    lower_wick_ratio:float;close_location:float;range_expansion:float|None
    range_contraction:bool|None;displacement:bool|None;inside_bar:bool|None
    outside_bar:bool|None;consecutive_bullish_closes:int;consecutive_bearish_closes:int
    structural_equilibrium_distance_atr:float|None;volume_zscore:float|None
    relative_volume:float|None;volume_expansion:bool|None
    pullback_volume_contraction:bool|None;selling_exhaustion_proxy:bool|None
    buying_exhaustion_proxy:bool|None;momentum_deceleration:bool|None
    momentum_recovery:bool|None;acceleration:float|None;dip_score:float|None
    rally_score:float|None;confirmed_swings:tuple[ConfirmedSwing,...]
    schema_version:str="brian.market-structure.v1"
    def numeric(self):
        out={"structure_state":{"DOWNTREND":-1.,"RANGE":0.,"TRANSITION":0.,"UPTREND":1.,"UNKNOWN":None}[self.state]}
        skip={"timestamp","state","latest_high_label","latest_low_label","confirmed_swings","schema_version"}
        for key,value in asdict(self).items():
            if key not in skip:out[key]=float(value) if isinstance(value,(bool,int,float)) else value
        return out

def _rsi(closes,index,period=14):
    if index<period:return None
    changes=[closes[k]-closes[k-1] for k in range(index-period+1,index+1)]
    gain=sum(max(x,0) for x in changes)/period;loss=sum(max(-x,0) for x in changes)/period
    return 100. if loss==0 and gain else 50. if loss==0 else 100-100/(1+gain/loss)

def _atr(rows,index,period):
    if index+1<period:return None
    values=[]
    for k in range(index-period+1,index+1):
        previous=rows[k-1].close if k else rows[k].close
        values.append(max(rows[k].high-rows[k].low,abs(rows[k].high-previous),abs(rows[k].low-previous)))
    return sum(values)/period

def _state(highs,lows):
    if len(highs)<2 or len(lows)<2:return "UNKNOWN"
    hu=highs[-1].price>highs[-2].price;lu=lows[-1].price>lows[-2].price
    return "UPTREND" if hu and lu else "DOWNTREND" if not hu and not lu else "RANGE" if not hu and lu else "TRANSITION"

def _zone(levels,close,atr,index,rows,width,support):
    if not atr or not levels:return (None,)*5
    # A bounded, explicitly recent structural set prevents ancient levels from
    # dominating and keeps the sequential engine linear-time in long histories.
    recent=levels[-64:]
    eligible=[x for x in recent if x.price<=close] if support else [x for x in recent if x.price>=close]
    if not eligible:return (None,)*5
    level=(max if support else min)(eligible,key=lambda x:x.price);w=atr*width
    reactions=sum(x.low<=level.price+w and x.high>=level.price-w
                  for x in rows[max(level.confirmation_index+1,index-64):index])
    distance=(close-level.price)/atr if support else (level.price-close)/atr
    return level.price,distance,index-level.confirmation_index,reactions,abs(close-level.price)<=w

def _score(values):
    values=[x for x in values if x is not None]
    return None if len(values)<4 else max(0.,min(1.,sum(values)/len(values)))

def compute_market_structure(candles:Sequence[StructureCandle],config:StructureConfig=StructureConfig()):
    rows=tuple(candles)
    if any(b.close_timestamp<=a.close_timestamp for a,b in zip(rows,rows[1:])):raise ValueError("candles must be chronological")
    closes=[x.close for x in rows];volumes=[x.volume for x in rows];swings=[];highs=[];lows=[];broken_h=set();broken_l=set()
    last_bull=last_bear=None;out=[];bull_run=bear_run=0
    for i,candle in enumerate(rows):
        atr=_atr(rows,i,config.atr_period);rsi=_rsi(closes,i);prior=_state(highs,lows);kh=highs[-1] if highs else None;kl=lows[-1] if lows else None
        buffer=atr*config.break_buffer_atr if atr else math.inf
        bull=bool(kh and candle.close>kh.price+buffer and kh.provenance_id not in broken_h)
        bear=bool(kl and candle.close<kl.price-buffer and kl.provenance_id not in broken_l)
        bull_choch=bull and prior=="DOWNTREND";bear_choch=bear and prior=="UPTREND"
        if bull:broken_h.add(kh.provenance_id);last_bull=kh
        if bear:broken_l.add(kl.provenance_id);last_bear=kl
        new=[]
        pivot=i-config.right_bars
        if pivot>=config.left_bars:
            window=rows[pivot-config.left_bars:i+1];pc=rows[pivot]
            kinds=[]
            if pc.high==max(x.high for x in window) and sum(x.high==pc.high for x in window)==1:kinds.append(("HIGH",pc.high))
            if pc.low==min(x.low for x in window) and sum(x.low==pc.low for x in window)==1:kinds.append(("LOW",pc.low))
            for kind,price in kinds:
                history=highs if kind=="HIGH" else lows
                label=("HH" if price>history[-1].price else "LH") if kind=="HIGH" and history else ("HL" if price>history[-1].price else "LL") if history else "UNCLASSIFIED"
                payload={"kind":kind,"pivot_timestamp":pc.close_timestamp,"confirmation_timestamp":candle.close_timestamp,"price":price,"left":config.left_bars,"right":config.right_bars}
                swing=ConfirmedSwing(kind,pivot,pc.close_timestamp,i,candle.close_timestamp,price,_rsi(closes,pivot),label,sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
                history.append(swing);swings.append(swing);new.append(swing)
        bull_div=bear_div=False
        for swing in new:
            history=lows if swing.kind=="LOW" else highs
            if len(history)>1 and swing.rsi is not None and history[-2].rsi is not None:
                if swing.kind=="LOW":bull_div=swing.price<history[-2].price and swing.rsi>history[-2].rsi
                else:bear_div=swing.price>history[-2].price and swing.rsi<history[-2].rsi
        state=_state(highs,lows);support=_zone(lows,candle.close,atr,i,rows,config.zone_width_atr,True);resistance=_zone(highs,candle.close,atr,i,rows,config.zone_width_atr,False)
        bull_sweep=bool(kl and candle.low<kl.price-buffer and candle.close>kl.price);bear_sweep=bool(kh and candle.high>kh.price+buffer and candle.close<kh.price)
        bull_retest=bool(atr and last_bull and candle.low<=last_bull.price+config.zone_width_atr*atr and candle.close>last_bull.price and not bull)
        bear_retest=bool(atr and last_bear and candle.high>=last_bear.price-config.zone_width_atr*atr and candle.close<last_bear.price and not bear)
        span=max(candle.high-candle.low,1e-12);body=abs(candle.close-candle.open);upper=candle.high-max(candle.open,candle.close);lower=min(candle.open,candle.close)-candle.low
        previous_span=rows[i-1].high-rows[i-1].low if i else None;expansion=span/previous_span if previous_span else None
        inside=None if not i else candle.high<rows[i-1].high and candle.low>rows[i-1].low;outside=None if not i else candle.high>rows[i-1].high and candle.low<rows[i-1].low
        bull_run=bull_run+1 if i and candle.close>rows[i-1].close else 0;bear_run=bear_run+1 if i and candle.close<rows[i-1].close else 0
        equilibrium=(highs[-1].price+lows[-1].price)/2 if highs and lows else None
        mean=sum(volumes[i-config.volume_period+1:i+1])/config.volume_period if i+1>=config.volume_period else None
        sample=volumes[i-config.volume_period+1:i+1] if mean is not None else [];sd=statistics.pstdev(sample) if sample else None
        volume_z=(candle.volume-mean)/sd if sd else (0. if mean is not None else None);relative=candle.volume/mean if mean else None
        returns=[closes[k]/closes[k-1]-1 for k in range(max(1,i-2),i+1)];accel=returns[-1]-returns[-2] if len(returns)>1 else None
        decel=abs(returns[-1])<abs(returns[-2]) if len(returns)>1 else None;recovery=returns[-2]<0<returns[-1] if len(returns)>1 else None
        vol_expand=relative>1.5 if relative is not None else None
        pullback=None if relative is None else bool(relative<.8 and ((state=="UPTREND" and candle.close<candle.open) or (state=="DOWNTREND" and candle.close>candle.open)))
        near_s=support[1] is not None and support[1]<=config.near_zone_atr;near_r=resistance[1] is not None and resistance[1]<=config.near_zone_atr
        dip=_score((1. if state=="UPTREND" else 0.,1. if near_s else 0. if support[0] else None,lower/span,1. if bull_sweep else 0.,max(0.,min(1.,(50-rsi)/30)) if rsi is not None else None,1. if decel else 0. if decel is not None else None,1. if recovery else 0. if recovery is not None else None,1. if pullback else 0. if relative is not None else None))
        rally=_score((1. if state=="DOWNTREND" else 0.,1. if near_r else 0. if resistance[0] else None,upper/span,1. if bear_sweep else 0.,max(0.,min(1.,(rsi-50)/30)) if rsi is not None else None,1. if decel else 0. if decel is not None else None,0. if recovery else 1. if recovery is not None else None,1. if pullback else 0. if relative is not None else None))
        out.append(MarketStructureFeatures(candle.close_timestamp,state,highs[-1].price if highs else None,lows[-1].price if lows else None,highs[-1].label if highs else None,lows[-1].label if lows else None,bull and not bull_choch,bear and not bear_choch,bull_choch,bear_choch,support[0],resistance[0],support[1],resistance[1],support[2],resistance[2],support[3],resistance[3],support[4],resistance[4],bull_sweep,bear_sweep,bull_sweep,bear_sweep,bull_retest,bear_retest,bull_div,bear_div,atr,rsi,body/span,upper/span,lower/span,(candle.close-candle.low)/span,expansion,expansion<.8 if expansion is not None else None,None if atr is None else span>=1.5*atr and body/span>=.65,inside,outside,bull_run,bear_run,(candle.close-equilibrium)/atr if equilibrium is not None and atr else None,volume_z,relative,vol_expand,pullback,None if vol_expand is None else bool(vol_expand and lower/span>.5 and candle.close>candle.open),None if vol_expand is None else bool(vol_expand and upper/span>.5 and candle.close<candle.open),decel,recovery,accel,dip,rally,tuple(new)))
    return tuple(out)

def join_completed_timeframes(primary:Sequence[MarketStructureFeatures],fifteen:Sequence[MarketStructureFeatures],hourly:Sequence[MarketStructureFeatures]):
    if any(x.timestamp>=DEVELOPMENT_CUTOFF for rows in (primary,fifteen,hourly) for x in rows):raise ValueError("2026 data forbidden")
    t15=[x.timestamp for x in fifteen];t1h=[x.timestamp for x in hourly];out=[]
    for row in primary:
        i15=bisect_right(t15,row.timestamp)-1;i1h=bisect_right(t1h,row.timestamp)-1;m15=fifteen[i15] if i15>=0 else None;h1=hourly[i1h] if i1h>=0 else None
        known=[x for x in (row.state,m15.state if m15 else None,h1.state if h1 else None) if x not in (None,"UNKNOWN")]
        out.append({"timestamp":row.timestamp,"structure_5m":row,"structure_15m":m15,"structure_1h":h1,"mtf_agreement":len(known)==3 and len(set(known))==1,"htf_trend_ltf_pullback":bool(m15 and h1 and m15.state==h1.state and ((h1.state=="UPTREND" and row.dip_score is not None) or (h1.state=="DOWNTREND" and row.rally_score is not None))),"htf_trend_ltf_reversal":bool(m15 and h1 and m15.state==h1.state and ((h1.state=="UPTREND" and row.bullish_choch) or (h1.state=="DOWNTREND" and row.bearish_choch))),"counter_trend_warning":bool(h1 and row.state in ("UPTREND","DOWNTREND") and h1.state in ("UPTREND","DOWNTREND") and row.state!=h1.state)})
    return tuple(out)
