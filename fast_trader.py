# fast_trader.py — COINS MONSTER (balance göstergeli)
import time, sys, json, traceback, requests, os
from typing import Dict, Tuple, Optional, Callable
from dotenv import load_dotenv

from dip_tracker import DipTracker
from telegram_utils import tg_setup, tg_send
from pump_predictor import PumpPredictor
from news_hunter import NewsHunter
from watchlist_manager import WatchlistManager
from regime import RegimeDetector

STATE_PATH = "runtime_state.json"

# ---------- helpers ----------
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
    dist_pct = float(cfg.get("max_dist_pct", 0.2)) / 100.0
    mult = float(cfg.get("imbalance_mult", 2.0))
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/bookTicker",
                         params={"symbol": symbol}, timeout=3)
        t = r.json(); bid=float(t["bidPrice"]); ask=float(t["askPrice"])
        mid=(bid+ask)/2.0
        r2 = requests.get("https://api.binance.com/api/v3/depth",
                          params={"symbol": symbol, "limit": 100}, timeout=3)
        d = r2.json()
        near_bids = [ (float(p), float(q)) for p,q in d["bids"] if (mid - float(p))/mid <= dist_pct ]
        near_asks = [ (float(p), float(q)) for p,q in d["asks"] if (float(p) - mid)/mid <= dist_pct ]
        bid_qty = sum(q for _,q in near_bids)
        ask_qty = sum(q for _,q in near_asks)
        if bid_qty >= mult*ask_qty and bid_qty>0:
            return True, f"Buy wall near (bid {bid_qty:.0f} vs ask {ask_qty:.0f})"
        if ask_qty >= mult*bid_qty and ask_qty>0:
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

# ---------- Symbol Engine ----------
class SymbolEngine:
    def __init__(self, symbol: str, cfg: dict, news: NewsHunter,
                 get_cash_cb: Optional[Callable[[], float]] = None):
        self.s = symbol; self.cfg = cfg; self.rules = cfg["rules"]
        self.get_cash_cb = get_cash_cb  # 💰 bot’tan canlı kasa göstermek için
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
            "MEAN":{"tp":0.9,"sl":0.7,"off":0.8},
            "CHOP":{"tp":0.6,"sl":0.6,"off":1.2},
            "UNKNOWN":{"tp":1.0,"sl":1.0,"off":1.0}
        })
        m = mults.get(reg, mults["UNKNOWN"])
        tp *= m["tp"]; sl *= m["sl"]; off *= m["off"]

        dyn = r.get("dynamic_tpsl", {"enabled":True,"min_scale":0.6,"max_scale":1.8})
        if dyn.get("enabled",True):
            prob,_ = self.pred.score()
            scale = max(float(dyn.get("min_scale",0.6)), min(float(dyn.get("max_scale",1.8)), 0.6 + 1.2*(prob)))
            tp *= scale; sl *= scale

        return round(off,self.pdec), round(tp,self.pdec), round(sl,self.pdec)

    # ---- sinyaller ----
    def signal_dip(self, price: float):
        dip_val = self.dip.update(price)
        if dip_val is None or not self.dip.can_buy():
            return (False,"",None)
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

    # ---- pozisyon yönetimi ----
    def _open(self, slot: str, price: float, cash: float, why: str, label: str, conf: float) -> float:
        fee_pct = float(self.cfg.get("fees",{}).get("taker_pct", 0.10)) / 100.0
        slip    = float(self.cfg.get("fees",{}).get("slippage_pct", 0.02)) / 100.0
        adj_price = price * (1.0 + slip)
        qty = round((cash * (1.0 - fee_pct)) / adj_price, self.qdec)
        if qty <= 0: return 0.0
        self.pos[slot] = {"avg":adj_price,"qty":qty,"open_ts":time.time(),"spent":cash,"why":label,"conf":conf}
        self._last_trade_ts[slot] = time.time()
        # 💬 Buy mesajı + kasa
        cash_line = ""
        try:
            if self.get_cash_cb:
                cash_line = f"\n💰 Kasa: {self.get_cash_cb():.2f}"
        except Exception:
            pass
        tg_send(
            f"✅ ALIM [{_reason_tag(label)}]\n"
            f"Parite: {self.s}\n"
            f"Fiyat: {self.fmtp(adj_price)} | Miktar: {self.fmtq(qty)}\n"
            f"Sebep: {why}\n⚖️ Risk: {conf:.2f} | Harcanan: {cash:.2f}{cash_line}"
        )
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
        qty = round(spend/price, self.qdec)
        if qty <= 0: return 0.0
        old_q, old_a = pos["qty"], pos["avg"]
        new_q = old_q + qty; new_a = (old_a*old_q + price*qty)/new_q
        pos["qty"]=new_q; pos["avg"]=new_a; pos["layers"]=pos.get("layers",1)+1; pos["spent"]=pos.get("spent",0.0)+spend
        cash_line = ""
        try:
            if self.get_cash_cb:
                cash_line = f"\n💰 Kasa: {self.get_cash_cb():.2f}"
        except Exception:
            pass
        tg_send(
            f"✅ ALIM [DCA {slot.upper()} L{pos['layers']}]\n"
            f"Parite: {self.s}\n"
            f"Fiyat: {self.fmtp(price)} | Miktar: {self.fmtq(qty)}\n"
            f"Ort. Giriş: {self.fmtp(new_a)} | Toplam Qty: {self.fmtq(new_q)}{cash_line}"
        )
        return spend

    def maybe_close_prepare(self, slot: str, price: float):
        """Kapanış kriterlerini değerlendir, kapanacaksa (pnl, reason, sell_px, label) döndür."""
        pos = self.pos[slot]
        if pos is None: return (False, 0.0, "", 0.0, "")
        e=float(pos["avg"]); q=float(pos["qty"])
        fee_pct = float(self.cfg.get("fees",{}).get("taker_pct", 0.10)) / 100.0
        slip    = float(self.cfg.get("fees",{}).get("slippage_pct", 0.02)) / 100.0
        sell_px = price * (1.0 - slip)
        reason = None

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

        if not reason: return (False, 0.0, "", 0.0, "")
        pnl=(sell_px-e)*q * (1.0 - fee_pct)
        label = pos.get('why','-')
        return (True, pnl, reason, sell_px, label)

    def close_commit(self, slot: str):
        """Bot kasa güncelledikten sonra pozisyonu temizle & dip resetle."""
        self.pos[slot]=None
        try: self.dip.on_sell()
        except Exception: pass


# ---------- Bot ----------
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

        self.news = NewsHunter(cfg.get("news_mode", {}))
        self.wlm = WatchlistManager(cfg)
        self.syms: Dict[str, SymbolEngine] = {}
        self._sync_symbols(self.wlm.active())

        if cfg.get("telegram",{}).get("enabled", True):
            tg_setup()
            tg_send(f"🟣 COINS MONSTER online | Kasa: {self.cash:.2f} | "
                    f"DIP {int(self.alloc['dip']*100)}% / PRED {int(self.alloc['pred']*100)}% / "
                    f"NEWS {int(self.alloc['news']*100)}% / OB {int(self.alloc['ob']*100)}%")

        # state restore
        rest_cash, rest_pos = load_state(self.cash)
        self.cash = rest_cash
        for s, slots in rest_pos.items():
            if s not in self.syms:
                self.syms[s] = SymbolEngine(s, self.cfg, self.news, self.display_cash)
            st = self.syms[s]
            for k, v in slots.items():
                st.pos[k] = v
        tg_send(f"♻️ State yüklendi | Kasa: {self.cash:.2f} | Aktif sembol: {len(self.syms)}")

    # --- kasa gösterimi (açık pozisyon harcaması düşülmeden) ---
    def display_cash(self) -> float:
        """Mesajlarda göstereceğimiz kasa (nakit) — PnL işlenmiş, harcanan açık pozisyonları düşmeden."""
        return round(self.cash, 2)

    def _sync_symbols(self, active_list):
        for s in active_list:
            if s not in self.syms:
                self.syms[s] = SymbolEngine(s, self.cfg, self.news, self.display_cash)
                tg_send(f"📈 İzlemeye alındı: {s}")
        for s in list(self.syms.keys()):
            if s not in active_list:
                st = self.syms[s]
                has_pos = any(st.pos[k] is not None for k in ["dip","pred","news","ob"])
                if not has_pos:
                    del self.syms[s]
                    tg_send(f"📉 İzlemeyi bıraktı: {s}")

    # slot bazlı kasa
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
        st = self.syms[s]
        if time.time() - st._last_trade_ts.get(slot, 0.0) < self.min_gap:
            return False
        cutoff = time.time() - 3600
        self._trade_hist = [x for x in self._trade_hist if x[0] >= cutoff]
        sym_count = sum(1 for t, sym in self._trade_hist if sym == s)
        return sym_count < self.max_tph

    # smart sizing
    def _size_with_conf(self, slot: str, base_free: float, conf: float) -> float:
        mn = float(self.sizing.get("min_mult", 0.5))
        mx = float(self.sizing.get("max_mult", 2.0))
        base_frac = float(self.entry_frac.get(slot, 0.4))
        mult = mn + (mx - mn) * max(0.0, min(1.0, conf))
        frac = base_frac * mult
        return max(0.0, min(base_free, base_free * frac))

    def run(self):
        while True:
            try:
                if time.time() < self._cool_until:
                    time.sleep(1); continue

                self.wlm.update()
                self._sync_symbols(self.wlm.active())

                for s, st in self.syms.items():
                    snap = st.reg.snapshot()
                    st._last_reg = snap
                    reg = snap.get("regime","UNKNOWN")

                    px = _get_px(s)
                    if px <= 0: continue
                    st.last_px = px

                    if self.console:
                        off,tp,sl = st.levels(px)
                        dip = st.dip.current_dip
                        need = None if dip is None else (dip+off)
                        print(f"[HB] {s} reg={reg} px={st.fmtp(px)} dip={('-' if dip is None else st.fmtp(dip))} "
                              f"need={('-' if need is None else st.fmtp(need))} tp={st.fmtp(tp)} sl={st.fmtp(sl)} "
                              f"open={self.open_count()}/{self.max_open_total}")

                    if self.open_count() >= self.max_open_total:
                        continue
                    if self.symbol_open_count(s) >= self.max_per_symbol:
                        continue

                    if reg == "TREND":   bonus = 1.2
                    elif reg == "MEAN":  bonus = 1.0
                    elif reg == "CHOP":  bonus = 0.6
                    else:                bonus = 0.9

                    bias = reg
                    allow_long = (bias in ["TREND","MEAN","UNKNOWN"])

                    if allow_long:
                        fired, why, _ = st.signal_dip(px)
                        if fired and self.can_trade_now(s,"dip"):
                            _, free = self.slot_cash("dip")
                            conf = 0.55 + (0.15 if reg=="TREND" else 0.0)
                            spend = self._size_with_conf("dip", free, conf) * bonus
                            if spend>0:
                                used = st.maybe_open("dip", px, spend, why, "DIP", conf)
                                if used>0: self._trade_hist.append((time.time(), s))

                    if allow_long:
                        fired, why, prob = st.signal_pred(px)
                        if fired and self.can_trade_now(s,"pred"):
                            _, free = self.slot_cash("pred")
                            conf = min(1.0, 0.5 + 0.5*(prob or 0.7) + (0.15 if reg=="TREND" else 0.0))
                            spend = self._size_with_conf("pred", free, conf) * bonus
                            if spend>0:
                                used = st.maybe_open("pred", px, spend, why, "PRED", conf)
                                if used>0: self._trade_hist.append((time.time(), s))

                    nfired, nwhy = st.signal_news()
                    if nfired and self.can_trade_now(s,"news"):
                        _, free = self.slot_cash("news")
                        conf = 0.9
                        spend = self._size_with_conf("news", free, conf)
                        if spend>0:
                            used = st.maybe_open("news", px, spend, nwhy, "NEWS", conf)
                            if used>0: self._trade_hist.append((time.time(), s))

                    ofired, owhy = st.signal_ob(self.cfg.get("orderbook", {}))
                    if ofired and allow_long and self.can_trade_now(s,"ob"):
                        _, free = self.slot_cash("ob")
                        conf = 0.65 + (0.15 if reg=="TREND" else 0.0)
                        spend = self._size_with_conf("ob", free, conf) * bonus
                        if spend>0:
                            used = st.maybe_open("ob", px, spend, owhy, "ORDERBOOK", conf)
                            if used>0: self._trade_hist.append((time.time(), s))

                # DCA + exit + risk guard + state
                for st in self.syms.values():
                    px = st.last_px
                    if px <= 0: continue
                    for slot in ["dip","pred","news","ob"]:
                        if st.pos[slot] is None: continue
                        _, free = self.slot_cash(slot)
                        if slot != "news" and self.cfg["dca"]["enabled"]:
                            used = st.maybe_dca(slot, px, free)
                            if used>0:
                                pass

                        closed, pnl, reason, sell_px, label = st.maybe_close_prepare(slot, px)
                        if closed:
                            self.cash = round(self.cash + pnl, 2)  # 💰 kasa güncelle
                            st.close_commit(slot)                  # pozisyonu sil
                            pnl_pct = 0.0
                            try:
                                avg = sell_px - pnl / (st.pos.get(slot, {}).get("qty", 1) or 1)  # güvenli ama gerekmez
                            except Exception:
                                avg = None
                            # tek mesaj: satış + güncel kasa
                            tg_send(
                                f"🟣 SATIŞ [{_reason_tag(label)}]\n"
                                f"Parite: {st.s}\n"
                                f"Fiyat: {st.fmtp(sell_px)}\n"
                                f"Sebep: {_reason_tag(reason)}\n"
                                f"PnL: {pnl:+.2f} USD ({pnl_pct:+.2f}%)\n"
                                f"💰 Kasa: {self.cash:.2f}"
                            )
                            # gün içi risk
                            day_now = time.strftime("%Y-%m-%d")
                            if day_now != self._day0:
                                self._day0 = day_now
                                self._realized_today = 0.0
                            self._realized_today += pnl
                            if self._realized_today <= -abs(self.risk_daily_cap):
                                self._cool_until = time.time() + self.cooldown_min * 60
                                tg_send(f"🟥 Risk guard: Günlük zarar limiti aşıldı ({self._realized_today:.2f} USD). {self.cooldown_min}dk cool-down.")

                save_state(self.cash, self.syms)
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
