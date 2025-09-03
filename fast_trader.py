# fast_trader.py  — Coins Monster (PF + MaxDD rapor, Trading Hours, opsiyonel EV Filter)
import time, sys, json, traceback, requests, os, math
from typing import Dict, Tuple, Optional
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

from dotenv import load_dotenv

from dip_tracker import DipTracker
from pump_predictor import PumpPredictor
from news_hunter import NewsHunter
from watchlist_manager import WatchlistManager
from regime import RegimeDetector
from telegram_utils import tg_setup, tg_send
from candles import candle_bias

STATE_PATH = "runtime_state.json"

# ------------------ helpers ------------------
def _ex_decimals(symbol: str) -> Tuple[int,int]:
    try:
        r = requests.get("https://api.binance.com/api/v3/exchangeInfo",
                         params={"symbol": symbol}, timeout=5)
        f = r.json()["symbols"][0]["filters"]
        pf = next(x for x in f if x["filterType"]=="PRICE_FILTER")
        lf = next(x for x in f if x["filterType"]=="LOT_SIZE")
        def dec(step: float)->int:
            s=f"{step:.12f}".rstrip("0");  return len(s.split(".")[1]) if "." in s else 0
        return dec(float(pf["tickSize"])), dec(float(lf["stepSize"]))
    except Exception:
        return 2, 6

def _get_px(symbol: str) -> float:
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbol": symbol}, timeout=2.5)
        return float(r.json()["price"])
    except Exception:
        return 0.0

def _book_wall_signal(symbol: str, cfg: dict) -> Tuple[bool,str]:
    if not cfg.get("enabled", True): return False, ""
    dist_pct = float(cfg.get("max_dist_pct", 0.2)) / 100.0
    mult = float(cfg.get("imbalance_mult", 2.0))
    min_qty = float(cfg.get("min_wall_qty", 0.0))
    min_bps = float(cfg.get("min_distance_bps", 0)) / 10000.0
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/bookTicker",
                         params={"symbol": symbol}, timeout=3)
        t = r.json(); bid=float(t["bidPrice"]); ask=float(t["askPrice"])
        mid=(bid+ask)/2.0
        r2 = requests.get("https://api.binance.com/api/v3/depth",
                          params={"symbol": symbol, "limit": 100}, timeout=3)
        d = r2.json()

        def near(side):
            arr = d[side]
            out=[]
            for p,q in arr:
                p=float(p); q=float(q)
                dd = (mid-p)/mid if side=="bids" else (p-mid)/mid
                if dd <= max(dist_pct, min_bps):
                    out.append((p,q))
            return out

        nb = near("bids"); na = near("asks")
        bid_qty = sum(q for _,q in nb); ask_qty = sum(q for _,q in na)
        if bid_qty >= max(min_qty, mult*ask_qty) and bid_qty>0:
            return True, f"Buy wall near (bid {bid_qty:.0f} vs ask {ask_qty:.0f})"
        if ask_qty >= max(min_qty, mult*bid_qty) and ask_qty>0:
            return True, f"Sell wall near (ask {ask_qty:.0f} vs bid {bid_qty:.0f})"
    except Exception:
        return False, ""
    return False, ""

def _reason_tag(reason: str)->str:
    m={"DIP":"🟣 DIP","PRED":"🔮 PRED","NEWS":"🚨 NEWS","ORDERBOOK":"🧱 ORDERBOOK",
       "TP":"✅ TP","TP/DECAY":"⏳ TP/DECAY","SL":"🛑 SL","TIME_SL":"⏱️ TIME-SL","BREAKEVEN/MINPROFIT":"⚖️ BE"}
    return m.get(reason.upper(), reason)

def save_state(cash: float, syms: dict):
    try:
        data = {
            "ts": time.time(),
            "cash": cash,
            "positions": {
                s: {k: v for k, v in st.pos.items() if v is not None}
                for s, st in syms.items()
            }
        }
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_state(default_cash: float):
    if not os.path.exists(STATE_PATH):
        return default_cash, {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            j = json.load(f)
        return float(j.get("cash", default_cash)), j.get("positions", {})
    except Exception:
        return default_cash, {}

# ---------- EV FILTER (opsiyonel) ----------
def ev_ok(entry: float, tp_abs: float, sl_abs: float, conf: float,
          fee_pct: float, slip_pct: float, ev_min: float = 0.0005) -> Tuple[bool, float]:
    """
    Basit EV tahmini (USD):
      EV ≈ conf * (tp_net) - (1-conf) * |sl_net|
    conf: 0..1 arası güven (slot'a göre hesaplanıyor)
    tp_abs/sl_abs: mutlak fark (USD), sl_abs negatif olmalı
    """
    tp_net = max(0.0, (tp_abs) * (1.0 - fee_pct) )  # sadeleştirilmiş
    sl_net = abs(sl_abs) * (1.0 + fee_pct)
    ev = conf * tp_net - (1.0 - conf) * sl_net
    return (ev >= ev_min), ev

# ---------------- Symbol Engine ----------------
class SymbolEngine:
    def __init__(self, symbol: str, cfg: dict, news: NewsHunter):
        self.s = symbol; self.cfg = cfg; self.rules = cfg["rules"]
        self.pdec, self.qdec = _ex_decimals(symbol)
        self.pf = "{:." + str(self.pdec) + "f}"; self.qf = "{:." + str(self.qdec) + "f}"
        d = cfg["dip"]
        self.dip = DipTracker(
            require_new_dip_after_start=bool(d.get("require_new_dip_after_start", False)),
            reset_dip_after_sell=bool(d.get("reset_dip_after_sell", True)),
            window_sec=int(d.get("window_sec", 300)) if d.get("window_sec") is not None else None
        )
        self.pred = PumpPredictor(symbol, cfg["predictor"])
        self.news = news
        self.reg = RegimeDetector(symbol)
        self._last_reg = {"regime":"UNKNOWN","ema9":None}

        self.pos: Dict[str, Optional[dict]] = {"dip": None, "pred": None, "news": None, "ob": None}
        self._last_trade_ts = {"dip":0.0, "pred":0.0, "news":0.0, "ob":0.0}
        self.last_px = 0.0

    def fmtp(self,x:float)->str: return self.pf.format(x)
    def fmtq(self,x:float)->str: return self.qf.format(x)

    def levels(self, price: float):
        r = self.rules
        mode = r.get("per_symbol_modes",{}).get(self.s, r.get("level_mode","pct"))
        # dip_rules override (sadece DIP için mutlak offset/tp/sl)
        dip_rules = self.cfg.get("dip_rules", {})
        abs_dip = (dip_rules.get("abs") or {}) if dip_rules else {}

        a,p = r["abs"], r["pct"]
        a_off,a_tp,a_sl = float(a["offset"]), float(a["tp"]), float(a["sl"])
        p_off,p_tp,p_sl = price*(float(p["offset"])/100.0), price*(float(p["tp"])/100.0), price*(float(p["sl"])/100.0)
        if mode=="abs": off,tp,sl = a_off,a_tp,a_sl
        elif mode=="pct": off,tp,sl = p_off,p_tp,p_sl
        elif mode=="hybrid_max": off,tp,sl = max(a_off,p_off),max(a_tp,p_tp),min(a_sl,p_sl)
        else: off,tp,sl = a_off,a_tp,a_sl

        reg = (self._last_reg.get("regime") or "UNKNOWN")
        mults = self.cfg.get("regime", {}).get("multipliers", {
            "TREND":{"tp":1.4,"sl":0.9,"off":1.1},
            "MEAN":{"tp":1.0,"sl":0.8,"off":1.0},
            "CHOP":{"tp":0.6,"sl":0.6,"off":1.2},
            "UNKNOWN":{"tp":1.0,"sl":1.0,"off":1.0}
        })
        m = mults.get(reg, mults["UNKNOWN"])
        tp *= m["tp"]; sl *= m["sl"]; off *= m["off"]

        dyn = r.get("dynamic_tpsl", {"enabled":True,"min_scale":0.7,"max_scale":1.8})
        if dyn.get("enabled",True):
            prob,_ = self.pred.score()
            scale = max(float(dyn.get("min_scale",0.7)), min(float(dyn.get("max_scale",1.8)), 0.6 + 1.2*(prob)))
            tp *= scale; sl *= scale

        return round(off,self.pdec), round(tp,self.pdec), round(sl,self.pdec)

    # --------- signals ----------
    def signal_dip(self, price: float):
        dip_val = self.dip.update(price)
        if dip_val is None or not self.dip.can_buy():
            return (False,"",None)
        # dip_rules varsa offset/tp/sl için mutlak offset kullanacağız (giriş kontrolünde)
        off,_,_ = self.levels(price)
        fired = price >= (dip_val + off)
        ema9 = self._last_reg.get("ema9")
        if ema9 is not None:
            fired = fired and (price >= ema9)
        return fired, f"dip={self.fmtp(dip_val)} need={self.fmtp(dip_val+off)} ema9={('-' if ema9 is None else self.fmtp(ema9))}", None

    def signal_pred(self, price: float):
        prob, info = self.pred.score()
        if prob < float(self.cfg["predictor"].get("enter_prob", 0.7)):
            return (False,"",prob)
        hh = float(info.get("hh", price*1.001))
        return (price >= hh*1.001, f"prob={prob:.2f} hh={self.fmtp(hh)}", prob)

    def signal_news(self): return self.news.maybe_signal(self.s)
    def signal_ob(self, cfg: dict): return _book_wall_signal(self.s, cfg)

    # --------- position mgmt ----------
    def _open(self, slot: str, price: float, cash: float, why: str, label: str, conf: float):
        fee_pct = float(self.cfg.get("fees",{}).get("taker_pct", 0.10)) / 100.0
        slip    = float(self.cfg.get("fees",{}).get("slippage_pct", 0.02)) / 100.0
        adj_price = price * (1.0 + slip)
        qty = round((cash * (1.0 - fee_pct)) / max(adj_price, 1e-12), self.qdec)
        if qty <= 0: return 0.0
        self.pos[slot] = {"avg":adj_price,"qty":qty,"open_ts":time.time(),"spent":cash,"why":label,"conf":conf}
        self._last_trade_ts[slot] = time.time()
        tg_send(f"✅ ALIM [{_reason_tag(label)}]\nParite: {self.s}\nFiyat: {self.fmtp(adj_price)} | Miktar: {self.fmtq(qty)}\nSebep: {why}\n⚖️ Risk: {conf:.2f} | Harcanan: {cash:.2f}")
        return cash

    def maybe_open(self, slot: str, price: float, free_cash: float, why: str, label: str, conf: float) -> float:
        if self.pos[slot] is not None or free_cash <= 0: return 0.0
        return self._open(slot, price, free_cash, why, label, conf)

    def maybe_dca(self, slot: str, price: float, free_cash: float) -> float:
        if self.pos[slot] is None or not self.cfg["dca"]["enabled"]: return 0.0
        pos = self.pos[slot]
        if pos.get("layers",1) >= int(self.cfg["dca"]["max_layers"]): return 0.0
        step_mode = self.cfg["dca"]["step_mode"]; step = float(self.cfg["dca"]["step"])
        last = float(pos["avg"])
        step_abs = last*(step/100.0) if step_mode=="pct" else step
        if price > last - step_abs: return 0.0
        spend = min(free_cash, free_cash * 0.50)
        qty = round(spend/max(price,1e-12), self.qdec)
        if qty <= 0: return 0.0
        old_q, old_a = pos["qty"], pos["avg"]
        new_q = old_q + qty; new_a = (old_a*old_q + price*qty)/new_q
        pos["qty"]=new_q; pos["avg"]=new_a; pos["layers"]=pos.get("layers",1)+1; pos["spent"]=pos.get("spent",0.0)+spend
        tg_send(f"✅ ALIM [DCA {slot.upper()} L{pos['layers']}]\nParite: {self.s}\nFiyat: {self.fmtp(price)} | Miktar: {self.fmtq(qty)}\nOrt. Giriş: {self.fmtp(new_a)} | Toplam Qty: {self.fmtq(new_q)}")
        return spend

    def maybe_close(self, slot: str, price: float):
        pos = self.pos[slot]
        if pos is None: return (False,0.0,"")
        e=float(pos["avg"]); q=float(pos["qty"]); reason=None
        fee_pct = float(self.cfg.get("fees",{}).get("taker_pct", 0.10)) / 100.0
        slip    = float(self.cfg.get("fees",{}).get("slippage_pct", 0.02)) / 100.0
        sell_px = price * (1.0 - slip)

        if slot=="news":
            trail = float(self.cfg["news_mode"]["trailing_pct"])/100.0
            emerg = float(self.cfg["news_mode"]["emergency_sl_pct"])/100.0
            peak = pos.get("peak", e); peak = max(peak, sell_px); pos["peak"]=peak
            if sell_px <= e*(1.0+emerg): reason="SL"
            elif sell_px <= peak*(1.0-trail): reason="TP/DECAY"
        else:
            off,tp,sl = self.levels(price)
            elapsed = time.time()-pos.get("open_ts",time.time())
            be_after = int(self.rules.get("breakeven_after_sec",240))
            be_min   = float(self.rules.get("min_exit_profit_pct",0.05))
            pnl_pct = (sell_px/(e or 1.0)-1.0)*100.0
            if sell_px >= e + tp: reason="TP"
            elif sell_px <= e + sl: reason="SL"
            elif elapsed >= int(self.rules.get("max_hold_sec",900)): reason="TIME_SL"
            elif elapsed >= be_after and pnl_pct >= be_min: reason="BREAKEVEN/MINPROFIT"

        if reason:
            pnl=(sell_px-e)*q * (1.0 - fee_pct)
            pnl_pct=((sell_px/(e or 1.0))-1.0)*100.0
            tg_send(f"🟣 SATIŞ [{_reason_tag(pos.get('why','-'))}]\nParite: {self.s}\nFiyat: {self.fmtp(sell_px)}\nSebep: {_reason_tag(reason)}\nPnL: {pnl:+.2f} USD ({pnl_pct:+.2f}%)")
            self.pos[slot]=None; self.dip.on_sell()
            return (True, pnl, reason)
        return (False, 0.0, "")

# -------------------- Bot --------------------
class Bot:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cash = float(cfg.get("cash", 500.0))
        self.console = bool(cfg.get("console_verbose", True))
        self.alloc = cfg.get("portfolio", {"dip":0.40,"pred":0.30,"news":0.20,"ob":0.10})
        self.entry_frac = cfg.get("entry_frac", {"dip":0.4,"pred":0.4,"news":0.6,"ob":0.5})
        self.sizing = cfg.get("sizing", {"min_mult":0.5,"max_mult":2.0})
        self.max_open_total = int(cfg.get("max_total_open_positions", 3))
        self.max_per_symbol = int(cfg.get("max_open_per_symbol", 2))

        rg = cfg.get("risk", {"daily_max_loss_usd": 10.0, "cooldown_min": 30})
        self.risk_daily_cap = float(rg.get("daily_max_loss_usd", 10.0))
        self.cooldown_min = int(rg.get("cooldown_min", 30))
        self._day0 = time.strftime("%Y-%m-%d")
        self._realized_today = 0.0
        self._cool_until = 0.0

        fc = cfg.get("freq_ctrl", {"min_sec_between_trades": 20, "max_trades_per_hour": 60})
        self.min_gap = int(fc.get("min_sec_between_trades", 20))
        self.max_tph = int(fc.get("max_trades_per_hour", 60))
        self._trade_hist = []  # [(ts, symbol)]

        # rapor
        self._events = []       # dict: ts, kind(open/close), slot, sym, pnl
        self._last_report = time.time()
        self._report_sec = int(cfg.get("report", {}).get("interval_sec", 7200))

        # trading hours
        th = cfg.get("trading_hours", {})
        self.th_enabled = bool(th.get("enabled", False))
        self.th_tz = th.get("tz", "Europe/Berlin")
        self.th_windows = [w.strip() for w in th.get("windows", []) if isinstance(w, str)]

        # EV filter (opsiyonel)
        bt = cfg.get("backtest", {})
        self.use_ev_filter = bool(bt.get("use_ev_filter", False))
        self.ev_min = float(bt.get("ev_min", 0.0005))

        self.news = NewsHunter(cfg.get("news_mode", {}))
        self.wlm = WatchlistManager(cfg)
        self.syms: Dict[str, SymbolEngine] = {}
        self._sync_symbols(self.wlm.active())

        if cfg.get("telegram",{}).get("enabled", True):
            load_dotenv()
            tg_setup()
            tg_send(f"🟣 COINS MONSTER online | Kasa: {self.cash:.2f} | "
                    f"DIP {int(self.alloc['dip']*100)}% / PRED {int(self.alloc['pred']*100)}% / "
                    f"NEWS {int(self.alloc['news']*100)}% / OB {int(self.alloc['ob']*100)}%")

        # state restore
        rest_cash, rest_pos = load_state(self.cash)
        self.cash = rest_cash
        for s, slots in rest_pos.items():
            if s not in self.syms:
                self.syms[s] = SymbolEngine(s, self.cfg, self.news)
            st = self.syms[s]
            for k, v in slots.items():
                st.pos[k] = v
        tg_send(f"♻️ State yüklendi | Kasa: {self.cash:.2f} | Aktif sembol: {len(self.syms)}")

    # ----- utils -----
    def _sync_symbols(self, active_list):
        for s in active_list:
            if s not in self.syms:
                self.syms[s] = SymbolEngine(s, self.cfg, self.news)
                tg_send(f"📈 İzlemeye alındı: {s}")
        for s in list(self.syms.keys()):
            if s not in active_list:
                st = self.syms[s]
                has_pos = any(st.pos[k] is not None for k in ["dip","pred","news","ob"])
                if not has_pos:
                    del self.syms[s]
                    tg_send(f"📉 İzlemeyi bıraktı: {s}")

    def slot_cash(self, slot: str) -> Tuple[float,float]:
        total_slot = self.cash * float(self.alloc.get(slot, 0.0))
        used = 0.0
        for st in self.syms.values():
            pos = st.pos.get(slot)
            if pos: used += float(pos.get("spent",0.0))
        return total_slot, max(0.0, total_slot - used)

    def open_count(self) -> int:
        return sum(1 for st in self.syms.values() for v in st.pos.values() if v is not None)

    def symbol_open_count(self, s: str) -> int:
        st = self.syms[s]
        return sum(1 for v in st.pos.values() if v is not None)

    def can_trade_now(self, s: str, slot: str) -> bool:
        # global cooldown
        if time.time() < self._cool_until:
            return False

        # trading hours (opsiyonel)
        if self.th_enabled and ZoneInfo is not None and self.th_windows:
            try:
                now = datetime.now(ZoneInfo(self.th_tz)).time()
                ok = False
                for w in self.th_windows:
                    try:
                        a,b = w.split("-")
                        hh,mm = map(int, a.split(":")); h2,m2 = map(int, b.split(":"))
                        from datetime import time as _t
                        if _t(hh,mm) <= now <= _t(h2,m2):
                            ok = True; break
                    except Exception:
                        continue
                if not ok:
                    return False
            except Exception:
                pass  # saat filtresi hata verirse engelleme

        st = self.syms[s]
        if time.time() - st._last_trade_ts.get(slot, 0.0) < self.min_gap:
            return False
        cutoff = time.time() - 3600
        self._trade_hist = [x for x in self._trade_hist if x[0] >= cutoff]
        sym_count = sum(1 for t, sym in self._trade_hist if sym == s)
        return sym_count < self.max_tph

    def _size_with_conf(self, slot: str, base_free: float, conf: float) -> float:
        mn = float(self.sizing.get("min_mult", 0.5))
        mx = float(self.sizing.get("max_mult", 2.0))
        base_frac = float(self.entry_frac.get(slot, 0.4))
        mult = mn + (mx - mn) * max(0.0, min(1.0, conf))
        frac = base_frac * mult
        return max(0.0, min(base_free, base_free * frac))

    # ----- REPORT -----
    def _add_event(self, kind:str, slot:str, sym:str, pnl:float=0.0):
        self._events.append({"ts": time.time(), "kind": kind, "slot": slot, "sym": sym, "pnl": pnl})

    def _maybe_report(self):
        if time.time() - self._last_report < self._report_sec: return
        self._last_report = time.time()
        cutoff = time.time() - self._report_sec
        ev = [e for e in self._events if e["ts"]>=cutoff]
        if not ev:
            tg_send(f"📊 2H RAPOR (son {int(self._report_sec/60)} dk)\nHiç işlem yok. "
                    f"\n🔓 Açık pozisyon: {self.open_count()} | 💰 Kasa: {self.cash:.2f}")
            return

        total_pnl = sum(e["pnl"] for e in ev if e["kind"]=="close")
        closes = [e for e in ev if e["kind"]=="close"]
        wins = [e for e in closes if e["pnl"]>0]
        cnt = len(closes)
        wins_n = len(wins)
        winrate = (wins_n/cnt*100.0) if cnt>0 else 0.0
        avgpnl = (total_pnl/cnt) if cnt>0 else 0.0

        # Profit Factor & MaxDD (pencere içi)
        gains = sum(e["pnl"] for e in closes if e["pnl"] > 0)
        losses = sum(-e["pnl"] for e in closes if e["pnl"] < 0)
        pf = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
        cum = 0.0; peak = 0.0; max_dd = 0.0
        for e in sorted(closes, key=lambda x: x["ts"]):
            cum += e["pnl"]
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        # coin stats
        coin = {}
        for e in closes:
            s=e["sym"]; coin.setdefault(s, {"p":0.0,"w":0,"n":0})
            coin[s]["p"] += e["pnl"]; coin[s]["n"] += 1;  coin[s]["w"] += 1 if e["pnl"]>0 else 0
        best = max(coin.items(), key=lambda x:x[1]["p"]) if coin else None
        worst = min(coin.items(), key=lambda x:x[1]["p"]) if coin else None

        # slot stats
        slots = {}
        for e in ev:
            sl=e["slot"]; slots.setdefault(sl, {"p":0.0,"w":0,"n":0})
            if e["kind"]=="close":
                slots[sl]["p"] += e["pnl"]; slots[sl]["n"] += 1; slots[sl]["w"] += 1 if e["pnl"]>0 else 0

        slot_name = {"news":"🚨 NEWS (Haber)","pred":"🔮 PRED (Tahmin)",
                     "dip":"🟣 DIP (Dip)","ob":"🧱 OB (Orderbook)"}

        def bar(v):
            if v>10: return "🟩🟩🟩🟩"
            if v>5:  return "🟩🟩"
            if v>0:  return "🟨"
            if v<-5: return "🟥🟥"
            if v<0:  return "🟥"
            return ""

        coin_lines=[]
        for s,data in sorted(coin.items(), key=lambda x:x[1]["p"], reverse=True):
            coin_lines.append(f"• {'🟢' if data['p']>0 else ('🟡' if abs(data['p'])<1e-9 else '🔴')} {s:<7} | {data['n']} işlem | {data['w']}/{data['n']} win | {data['p']:+.2f} | {bar(data['p'])}")

        slot_order=["news","pred","dip","ob"]
        slot_lines=[]
        for sl in slot_order:
            if sl in slots and slots[sl]["n"]>0:
                d=slots[sl]
                slot_lines.append(f"• {slot_name[sl]:<18} | {d['n']} işlem | ✅ {d['w']}/{d['n']} | {d['p']:+.2f}")

        # slot dağılımı (open sayıları)
        open_counts = {}
        for e in ev:
            if e["kind"]=="open":
                open_counts[e["slot"]] = open_counts.get(e["slot"],0)+1
        total_open = sum(open_counts.values()) or 1
        def pct(sl): return int(100*open_counts.get(sl,0)/total_open)
        slot_dist = f"📈 Slot dağılımı: NEWS 🔵 {pct('news')}% | PRED 🟢 {pct('pred')}% | DIP 🟣 {pct('dip')}% | OB 🟡 {pct('ob')}% | ALL-IN 🧨 0%"

        head = (f"📊 2H RAPOR (son {int(self._report_sec/60)} dk)\n"
                f"🧮 İşlem: {cnt} | 💵 Kâr: {total_pnl:+.2f} USD | "
                f"🟢 WinRate: {winrate:.1f}% | 📈 AvgPnL: {avgpnl:+.2f}\n"
                f"📊 PF: {('INF' if pf==float('inf') else f'{pf:.2f}')} | 📉 MaxDD(pencere): {-max_dd:.2f} USD\n\n")

        best_line = f"🏆 En iyi coin: {best[0]} | ✅ {coin[best[0]]['w']}/{coin[best[0]]['n']} | {coin[best[0]]['p']:+.2f}" if best else "🏆 En iyi coin: -"
        worst_line= f"⚠️ En kötü coin: {worst[0]} | ❌ {coin[worst[0]]['w']}/{coin[worst[0]]['n']} | {coin[worst[0]]['p']:+.2f}" if worst else "⚠️ En kötü coin: -"

        body = "— Coin bazında —\n" + ("\n".join(coin_lines) if coin_lines else "• veri yok") \
             + "\n\n— Slot bazında —\n" + ("\n".join(slot_lines) if slot_lines else "• veri yok") \
             + f"\n\n{slot_dist}\n\n" \
             + f"🔓 Açık pozisyon: {self.open_count()} | 💰 Kasa: {self.cash:.2f}"

        tg_send(head + best_line + "\n" + worst_line + "\n\n" + body)

    # ----- main loop -----
    def run(self):
        while True:
            try:
                if time.time() < self._cool_until:
                    time.sleep(1); self._maybe_report(); continue

                # watchlist
                self.wlm.update()
                self._sync_symbols(self.wlm.active())

                for s, st in self.syms.items():
                    if s in self.cfg.get("ignore_symbols", []):
                        continue

                    snap = st.reg.snapshot()
                    st._last_reg = snap
                    reg = snap.get("regime","UNKNOWN")

                    px = _get_px(s)
                    if px <= 0: continue
                    st.last_px = px

                    # candle bias (BİR KEZ hesapla ve tüm girişlerde kullan)
                    cbias, cconf, ctag = candle_bias(s, self.cfg.get("candles", {}))
                    strict_mode = self.cfg.get("candles", {}).get("strict", True)
                    bonus_conf = (min(1.0, 0.1 + 0.8*cconf) if cbias=="bullish" else 0.0)

                    # heartbeat
                    if self.console:
                        off,tp,sl = st.levels(px)
                        dip = st.dip.current_dip
                        need = None if dip is None else (dip+off)
                        print(f"[HB] {s} reg={reg} px={st.fmtp(px)} dip={('-' if dip is None else st.fmtp(dip))} "
                              f"need={('-' if need is None else st.fmtp(need))} tp={st.fmtp(tp)} sl={st.fmtp(sl)} "
                              f"open={self.open_count()}/{self.max_open_total} cbias={cbias}({cconf:.2f})")

                    if self.open_count() >= self.max_open_total:
                        continue
                    if self.symbol_open_count(s) >= self.max_per_symbol:
                        continue

                    # regime bonus
                    if reg == "TREND":   bonus = 1.2
                    elif reg == "MEAN":  bonus = 1.0
                    elif reg == "CHOP":  bonus = 0.6
                    else:                bonus = 0.9

                    allow_long = True
                    if strict_mode and cbias!="bullish":
                        allow_long = False

                    fees = self.cfg.get("fees", {})
                    fee_pct = float(fees.get("taker_pct", 0.10))/100.0
                    slip_pct = float(fees.get("slippage_pct", 0.02))/100.0

                    # DIP
                    if allow_long:
                        fired, why, _ = st.signal_dip(px)
                        if fired and self.can_trade_now(s,"dip"):
                            _, free = self.slot_cash("dip")
                            # conf tabanı
                            conf = 0.55 + (0.15 if reg=="TREND" else 0.0) + bonus_conf
                            conf = max(0.0, min(1.0, conf))
                            # EV filter (opsiyonel)
                            if self.use_ev_filter:
                                # st.levels(px) → tp/sl fiyat farklarını döndürüyor (mutlaklaştır)
                                _, tp_abs, sl_abs = st.levels(px)
                                ok, ev = ev_ok(entry=px, tp_abs=tp_abs, sl_abs=sl_abs,
                                               conf=conf, fee_pct=fee_pct, slip_pct=slip_pct,
                                               ev_min=self.ev_min)
                                if not ok:
                                    if self.console:
                                        print(f"[EV] skip DIP {s} ev={ev:.4f} conf={conf:.2f}")
                                    pass
                                else:
                                    spend = self._size_with_conf("dip", free, conf) * bonus
                                    if spend>0:
                                        used = st.maybe_open("dip", px, spend, f"{why} | candles={cbias}({ctag})", "DIP", conf)
                                        if used>0:
                                            self._trade_hist.append((time.time(), s))
                                            self._add_event("open","dip",s,0.0)
                            else:
                                spend = self._size_with_conf("dip", free, conf) * bonus
                                if spend>0:
                                    used = st.maybe_open("dip", px, spend, f"{why} | candles={cbias}({ctag})", "DIP", conf)
                                    if used>0:
                                        self._trade_hist.append((time.time(), s))
                                        self._add_event("open","dip",s,0.0)

                    # PRED
                    if allow_long:
                        fired, why, prob = st.signal_pred(px)
                        if fired and self.can_trade_now(s,"pred"):
                            _, free = self.slot_cash("pred")
                            conf = min(1.0, 0.5 + 0.5*(prob or 0.7) + (0.15 if reg=="TREND" else 0.0) + bonus_conf)
                            if self.use_ev_filter:
                                _, tp_abs, sl_abs = st.levels(px)
                                ok, ev = ev_ok(px, tp_abs, sl_abs, conf, fee_pct, slip_pct, self.ev_min)
                                if not ok:
                                    if self.console:
                                        print(f"[EV] skip PRED {s} ev={ev:.4f} conf={conf:.2f}")
                                else:
                                    spend = self._size_with_conf("pred", free, conf) * bonus
                                    if spend>0:
                                        used = st.maybe_open("pred", px, spend, f"{why} | candles={cbias}({ctag})", "PRED", conf)
                                        if used>0:
                                            self._trade_hist.append((time.time(), s))
                                            self._add_event("open","pred",s,0.0)
                            else:
                                spend = self._size_with_conf("pred", free, conf) * bonus
                                if spend>0:
                                    used = st.maybe_open("pred", px, spend, f"{why} | candles={cbias}({ctag})", "PRED", conf)
                                    if used>0:
                                        self._trade_hist.append((time.time(), s))
                                        self._add_event("open","pred",s,0.0)

                    # NEWS (strict kapalıysa bypass; strict ise yine bullish izni arıyoruz)
                    nfired, nwhy = st.signal_news()
                    if nfired and self.can_trade_now(s,"news") and (not strict_mode or cbias=="bullish"):
                        _, free = self.slot_cash("news")
                        conf = 0.9
                        if self.use_ev_filter:
                            _, tp_abs, sl_abs = st.levels(px)
                            ok, ev = ev_ok(px, tp_abs, sl_abs, conf, fee_pct, slip_pct, self.ev_min)
                            if not ok:
                                if self.console:
                                    print(f"[EV] skip NEWS {s} ev={ev:.4f} conf={conf:.2f}")
                            else:
                                spend = self._size_with_conf("news", free, conf)
                                if spend>0:
                                    used = st.maybe_open("news", px, spend, f"{nwhy} | candles={cbias}({ctag})", "NEWS", conf)
                                    if used>0:
                                        self._trade_hist.append((time.time(), s))
                                        self._add_event("open","news",s,0.0)
                        else:
                            spend = self._size_with_conf("news", free, conf)
                            if spend>0:
                                used = st.maybe_open("news", px, spend, f"{nwhy} | candles={cbias}({ctag})", "NEWS", conf)
                                if used>0:
                                    self._trade_hist.append((time.time(), s))
                                    self._add_event("open","news",s,0.0)

                    # ORDERBOOK
                    ofired, owhy = st.signal_ob(self.cfg.get("orderbook", {}))
                    if ofired and self.can_trade_now(s,"ob") and allow_long:
                        _, free = self.slot_cash("ob")
                        conf = 0.65 + (0.15 if reg=="TREND" else 0.0) + bonus_conf
                        conf = max(0.0, min(1.0, conf))
                        if self.use_ev_filter:
                            _, tp_abs, sl_abs = st.levels(px)
                            ok, ev = ev_ok(px, tp_abs, sl_abs, conf, fee_pct, slip_pct, self.ev_min)
                            if not ok:
                                if self.console:
                                    print(f"[EV] skip OB {s} ev={ev:.4f} conf={conf:.2f}")
                            else:
                                spend = self._size_with_conf("ob", free, conf) * bonus
                                if spend>0:
                                    used = st.maybe_open("ob", px, spend, f"{owhy} | candles={cbias}({ctag})", "ORDERBOOK", conf)
                                    if used>0:
                                        self._trade_hist.append((time.time(), s))
                                        self._add_event("open","ob",s,0.0)
                        else:
                            spend = self._size_with_conf("ob", free, conf) * bonus
                            if spend>0:
                                used = st.maybe_open("ob", px, spend, f"{owhy} | candles={cbias}({ctag})", "ORDERBOOK", conf)
                                if used>0:
                                    self._trade_hist.append((time.time(), s))
                                    self._add_event("open","ob",s,0.0)

                # yönetim: DCA + early-exit + exit + risk + state + rapor
                for s, st in self.syms.items():
                    px = st.last_px
                    if px <= 0: continue
                    # bearish early exit
                    if self.cfg.get("candles", {}).get("exit_on_bearish", True):
                        cbias, cconf, ctag = candle_bias(s, self.cfg.get("candles", {}))
                        need_conf = float(self.cfg.get("candles", {}).get("exit_conf", 0.75))
                        ema9 = st._last_reg.get("ema9")
                        if cbias=="bearish" and (cconf>=need_conf) and (ema9 is not None) and (px<ema9):
                            for slot in ["dip","pred","news","ob"]:
                                if st.pos[slot] is None: continue
                                closed, pnl, _ = st.maybe_close(slot, px)
                                if closed:
                                    self.cash = round(self.cash + pnl, 2)
                                    self._add_event("close",slot,s,pnl)
                                    if self.cfg.get("report",{}).get("include_cash_on_sell", True):
                                        tg_send(f"💰 Kasa: {self.cash:.2f} USD")

                    for slot in ["dip","pred","news","ob"]:
                        if st.pos[slot] is None: continue
                        # dca
                        _, free = self.slot_cash(slot)
                        if slot != "news" and self.cfg["dca"]["enabled"]:
                            used = st.maybe_dca(slot, px, free)
                            if used>0: self.cash = round(self.cash, 2)
                        # exit
                        closed, pnl, _ = st.maybe_close(slot, px)
                        if closed:
                            self.cash = round(self.cash + pnl, 2)
                            self._add_event("close",slot,s,pnl)
                            # daily risk
                            day_now = time.strftime("%Y-%m-%d")
                            if day_now != self._day0:
                                self._day0 = day_now
                                self._realized_today = 0.0
                            self._realized_today += pnl
                            if self._realized_today <= -abs(self.risk_daily_cap):
                                self._cool_until = time.time() + self.cooldown_min * 60
                                tg_send(f"🟥 Risk guard: Günlük zarar limiti aşıldı ({self._realized_today:.2f} USD). {self.cooldown_min}dk cool-down.")
                            if self.cfg.get("report",{}).get("include_cash_on_sell", True):
                                tg_send(f"💰 Kasa: {self.cash:.2f} USD")

                save_state(self.cash, self.syms)
                self._maybe_report()
                time.sleep(1)
            except KeyboardInterrupt:
                print("Çıkılıyor..."); break
            except Exception as e:
                print("Hata:", e); traceback.print_exc(); time.sleep(2)

def main():
    cfg_path = "config_live.json"
    if len(sys.argv)>=3 and sys.argv[1]=="--config":
        cfg_path = sys.argv[2]
    with open(cfg_path,"r",encoding="utf-8") as f:
        cfg = json.load(f)
    load_dotenv()
    Bot(cfg).run()

if __name__=="__main__":
    main()
