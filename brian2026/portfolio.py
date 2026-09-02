from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence
import math

Action = Literal["BUY", "SELL", "WAIT"]
Side = Literal["LONG", "SHORT"]
State = Literal["FLAT", "LONG", "SHORT", "COOLDOWN"]
DEVELOPMENT_CUTOFF = 1767225600.0  # 2026-01-01T00:00:00Z
PORTFOLIO_SIMULATOR_VERSION = "brian.stateful-portfolio.v1"


class ChronologicalOutcomeQueue:
    """Releases adaptive-learning outcomes only after their resolution timestamp."""
    def __init__(self) -> None:self._pending:list[tuple[float,Any]]=[]
    def schedule(self,resolution_timestamp:float,outcome:Any)->None:
        if resolution_timestamp>=DEVELOPMENT_CUTOFF:raise ValueError("2026 outcomes are forbidden")
        self._pending.append((float(resolution_timestamp),outcome));self._pending.sort(key=lambda item:item[0])
    def release(self,current_timestamp:float)->tuple[Any,...]:
        ready=[payload for timestamp,payload in self._pending if timestamp<=current_timestamp]
        self._pending=[item for item in self._pending if item[0]>current_timestamp]
        return tuple(ready)


@dataclass(frozen=True, slots=True)
class PortfolioBar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        if self.timestamp >= DEVELOPMENT_CUTOFF:
            raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("invalid OHLC bar")


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    starting_equity: float = 10_000.0
    sizing_mode: Literal["fixed_notional", "equity_fraction"] = "fixed_notional"
    fixed_notional: float = 1_000.0
    equity_fraction: float = 0.10
    max_position_notional: float = 2_000.0
    max_equity_fraction: float = 0.20
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 2.0
    max_holding_bars: int = 12
    cooldown_bars: int = 1
    reversal_enabled: bool = True
    fee_bps: float = 10.0
    assumed_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    stop_first_same_bar: bool = True

    def __post_init__(self) -> None:
        if self.starting_equity <= 0 or self.fixed_notional <= 0 or self.max_position_notional <= 0:
            raise ValueError("invalid account sizing")
        if not 0 < self.equity_fraction <= 1 or not 0 < self.max_equity_fraction <= 1:
            raise ValueError("invalid equity fraction")
        if min(self.stop_loss_pct, self.take_profit_pct) <= 0 or min(self.max_holding_bars, self.cooldown_bars) < 0:
            raise ValueError("invalid lifecycle configuration")


@dataclass(slots=True)
class Position:
    side: Side
    quantity: float
    entry_price: float
    notional: float
    entry_timestamp: float
    entry_fee: float
    entry_spread_cost: float
    entry_slippage_cost: float
    bars_held: int = 0


@dataclass(frozen=True, slots=True)
class PortfolioTrade:
    side: Side
    entry_timestamp: float
    exit_timestamp: float
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    spread_cost: float
    slippage_cost: float
    net_pnl: float
    reason: str
    holding_seconds: float


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: float
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    drawdown: float
    state: State


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    config: dict
    starting_equity: float
    ending_equity: float
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    net_pnl: float
    return_pct: float
    max_drawdown: float
    fees: float
    assumed_spread_cost: float
    slippage_cost: float
    turnover: float
    entries: int
    exits: int
    signals: int
    buy_signals: int
    sell_signals: int
    wait_signals: int
    held_duplicate_signals: int
    average_hold_seconds: float
    exposure_pct: float
    max_consecutive_losses: int
    longest_drawdown_seconds: float
    trades: tuple[PortfolioTrade, ...]
    equity_curve: tuple[EquityPoint, ...]
    simulator_version: str = PORTFOLIO_SIMULATOR_VERSION


class StatefulPortfolioSimulator:
    """Deterministic research-only single-position state machine; no execution surface."""

    def __init__(self, config: PortfolioConfig = PortfolioConfig()) -> None:
        self.cfg=config;self.cash=config.starting_equity;self.realized=0.0;self.position:Position|None=None
        self.cooldown=0;self.trades:list[PortfolioTrade]=[];self.curve:list[EquityPoint]=[]
        self.fees=self.spread_cost=self.slippage_cost=self.turnover=0.0
        self.signals=self.buy=self.sell=self.wait=self.duplicates=self.exposed_bars=0
        self._peak=config.starting_equity;self._max_dd=0.0;self._dd_start:float|None=None;self._longest_dd=0.0

    @property
    def state(self)->State:
        if self.position:return self.position.side
        return "COOLDOWN" if self.cooldown>0 else "FLAT"

    def _unrealized(self, price:float)->float:
        if not self.position:return 0.0
        direction=1.0 if self.position.side=="LONG" else -1.0
        return (price-self.position.entry_price)*self.position.quantity*direction

    def _equity(self, price:float)->float:
        reserved=self.position.notional if self.position else 0.0
        return self.cash+reserved+self._unrealized(price)

    def _mark(self,bar:PortfolioBar)->None:
        equity=self._equity(bar.close);self._peak=max(self._peak,equity);dd=self._peak-equity;self._max_dd=max(self._max_dd,dd)
        if dd>1e-12 and self._dd_start is None:self._dd_start=bar.timestamp
        if dd<=1e-12 and self._dd_start is not None:self._longest_dd=max(self._longest_dd,bar.timestamp-self._dd_start);self._dd_start=None
        self.curve.append(EquityPoint(bar.timestamp,self.cash,equity,self.realized,self._unrealized(bar.close),dd,self.state))

    def _size(self, price:float)->float:
        equity=self._equity(price);desired=self.cfg.fixed_notional if self.cfg.sizing_mode=="fixed_notional" else equity*self.cfg.equity_fraction
        cap=min(self.cfg.max_position_notional,equity*self.cfg.max_equity_fraction)
        fee_rate=self.cfg.fee_bps/10000.0
        return max(0.0,min(desired,cap,self.cash/(1+fee_rate)))

    def _open(self,side:Side,bar:PortfolioBar)->bool:
        if self.position or self.cooldown:return False
        notional=self._size(bar.close)
        if notional<=0:return False
        half_spread=self.cfg.assumed_spread_bps/20000.0;slip=self.cfg.slippage_bps/10000.0
        entry=bar.close*(1+half_spread+slip if side=="LONG" else 1-half_spread-slip)
        qty=notional/entry;fee=notional*self.cfg.fee_bps/10000.0
        if notional+fee>self.cash+1e-9:return False
        self.cash-=notional+fee;self.fees+=fee;self.spread_cost+=notional*half_spread;self.slippage_cost+=notional*slip;self.turnover+=notional
        self.position=Position(side,qty,entry,notional,bar.timestamp,fee,notional*half_spread,notional*slip);return True

    def _close(self,bar:PortfolioBar,base_price:float,reason:str)->None:
        position=self.position
        if position is None:return
        half_spread=self.cfg.assumed_spread_bps/20000.0;slip=self.cfg.slippage_bps/10000.0
        exit_price=base_price*(1-half_spread-slip if position.side=="LONG" else 1+half_spread+slip)
        direction=1.0 if position.side=="LONG" else -1.0;gross=(exit_price-position.entry_price)*position.quantity*direction
        exit_notional=exit_price*position.quantity;fee=exit_notional*self.cfg.fee_bps/10000.0
        spread=exit_notional*half_spread;slippage=exit_notional*slip;all_fees=position.entry_fee+fee
        net=gross-all_fees
        self.cash+=position.notional+gross-fee;self.realized+=net;self.fees+=fee;self.spread_cost+=spread;self.slippage_cost+=slippage;self.turnover+=exit_notional
        self.trades.append(PortfolioTrade(position.side,position.entry_timestamp,bar.timestamp,position.entry_price,exit_price,position.quantity,gross,all_fees,position.entry_spread_cost+spread,position.entry_slippage_cost+slippage,net,reason,bar.timestamp-position.entry_timestamp))
        self.position=None;self.cooldown=self.cfg.cooldown_bars

    def _lifecycle_exit(self,bar:PortfolioBar)->bool:
        p=self.position
        if p is None:return False
        p.bars_held+=1;self.exposed_bars+=1
        if p.side=="LONG":target=p.entry_price*(1+self.cfg.take_profit_pct/100);stop=p.entry_price*(1-self.cfg.stop_loss_pct/100);target_hit=bar.high>=target;stop_hit=bar.low<=stop
        else:target=p.entry_price*(1-self.cfg.take_profit_pct/100);stop=p.entry_price*(1+self.cfg.stop_loss_pct/100);target_hit=bar.low<=target;stop_hit=bar.high>=stop
        if stop_hit and target_hit:reason="STOP" if self.cfg.stop_first_same_bar else "TARGET"
        elif stop_hit:reason="STOP"
        elif target_hit:reason="TARGET"
        elif p.bars_held>=self.cfg.max_holding_bars:reason="MAX_HOLD"
        else:return False
        self._close(bar,stop if reason=="STOP" else target if reason=="TARGET" else bar.close,reason);return True

    def step(self,bar:PortfolioBar,action:Action)->None:
        if self.curve and bar.timestamp<=self.curve[-1].timestamp:raise ValueError("portfolio bars must be strictly chronological")
        if action not in ("BUY","SELL","WAIT"):raise ValueError("invalid signal")
        cooldown_active=self.cooldown>0 and self.position is None
        if cooldown_active:self.cooldown-=1
        self.signals+=1;self.buy+=action=="BUY";self.sell+=action=="SELL";self.wait+=action=="WAIT"
        exited=self._lifecycle_exit(bar)
        if not exited and self.position:
            same=(self.position.side=="LONG" and action=="BUY") or (self.position.side=="SHORT" and action=="SELL")
            opposite=(self.position.side=="LONG" and action=="SELL") or (self.position.side=="SHORT" and action=="BUY")
            if same:self.duplicates+=1
            elif opposite and self.cfg.reversal_enabled:
                new_side:"Side"="SHORT" if self.position.side=="LONG" else "LONG";self._close(bar,bar.close,"REVERSAL")
                if self.cooldown==0:self._open(new_side,bar)
        elif not self.position and not exited and not cooldown_active and self.cooldown==0 and action!="WAIT":self._open("LONG" if action=="BUY" else "SHORT",bar)
        self._mark(bar)

    def finish(self,bar:PortfolioBar)->PortfolioResult:
        if not self.curve or bar.timestamp<self.curve[-1].timestamp:raise ValueError("invalid final bar")
        if self.position:self._close(bar,bar.close,"FORCED_END")
        if not self.curve or self.curve[-1].timestamp<bar.timestamp:self._mark(bar)
        elif self.curve[-1].timestamp==bar.timestamp:self.curve[-1]=EquityPoint(bar.timestamp,self.cash,self._equity(bar.close),self.realized,0.0,max(0.0,self._peak-self._equity(bar.close)),self.state)
        if self._dd_start is not None:self._longest_dd=max(self._longest_dd,bar.timestamp-self._dd_start)
        losses=streak=max_streak=0
        for trade in self.trades:
            streak=streak+1 if trade.net_pnl<0 else 0;max_streak=max(max_streak,streak)
        durations=[t.holding_seconds for t in self.trades];elapsed=max(0.0,self.curve[-1].timestamp-self.curve[0].timestamp) if len(self.curve)>1 else 0.0
        ending=self._equity(bar.close)
        return PortfolioResult(asdict(self.cfg),self.cfg.starting_equity,ending,self.cash,self.realized,0.0,ending-self.cfg.starting_equity,(ending/self.cfg.starting_equity-1)*100,self._max_dd,self.fees,self.spread_cost,self.slippage_cost,self.turnover,len(self.trades),len(self.trades),self.signals,self.buy,self.sell,self.wait,self.duplicates,sum(durations)/len(durations) if durations else 0.0,100*sum(durations)/elapsed if elapsed else 0.0,max_streak,self._longest_dd,tuple(self.trades),tuple(self.curve))

    def break_segment(self,last_included_bar:PortfolioBar)->None:
        """Force flat before an explicitly excluded data segment without resetting account history."""
        if self.position:self._close(last_included_bar,last_included_bar.close,"QUALITY_EXCLUSION")
        self.cooldown=0


def simulate_portfolio(bars:Sequence[PortfolioBar],actions:Sequence[Action],config:PortfolioConfig=PortfolioConfig())->PortfolioResult:
    if len(bars)!=len(actions) or not bars:raise ValueError("bars/actions must be non-empty and aligned")
    simulator=StatefulPortfolioSimulator(config)
    for bar,action in zip(bars,actions):simulator.step(bar,action)
    return simulator.finish(bars[-1])
