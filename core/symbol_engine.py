# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import time
from typing import Dict, Optional, Tuple, Any

import requests  # borsa yardımcıları için

from Proje1.core.utils_io import tg_send, log_event, log_trade, reason_tag

# ----------------- esnek import yardımcısi -----------------
def _try_import(mods: Tuple[str, ...], attr: Optional[str] = None):
    """mods içindeki modüllerden ilk yükleneni döndür. attr verilirse mod.attr döner."""
    for m in mods:
        try:
            mod = importlib.import_module(m)
            return getattr(mod, attr) if attr else mod
        except Exception:
            continue
    return None


# opsiyoneller (yoksa None kalır, stub’lar devreye girer)
DipTracker = _try_import(("dip_tracker", "Proje1.core.dip_tracker"), "DipTracker")
PumpPredictor = _try_import(("pump_predictor", "Proje1.core.pump_predictor"), "PumpPredictor")
RegimeDetector = _try_import(("regime", "Proje1.core.regime"), "RegimeDetector")


# ----------------- borsa yardımcıları -----------------
def _ex_decimals(symbol: str) -> Tuple[int, int]:
    """Binance fiyat/lot basamakları. Hata olursa (2,6) döner."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/exchangeInfo",
            params={"symbol": symbol},
            timeout=5,
        )
        info = r.json()["symbols"][0]["filters"]
        pf = next(x for x in info if x["filterType"] == "PRICE_FILTER")
        lf = next(x for x in info if x["filterType"] == "LOT_SIZE")

        def dec(step_str: str) -> int:
            s = f"{float(step_str):.12f}".rstrip("0")
            return len(s.split(".")[1]) if "." in s else 0

        return dec(pf["tickSize"]), dec(lf["stepSize"])
    except Exception:
        return 2, 6


def _book_wall_signal(symbol: str, cfg: dict) -> Tuple[bool, str]:
    """Orderbook duvar sinyali (yakın güçlü bid/ask)."""
    cfg = cfg or {}
    if not cfg.get("enabled", True):
        return False, ""
    max_dist = float(cfg.get("max_dist_pct", 0.2)) / 100.0
    min_bps = float(cfg.get("min_distance_bps", 6)) / 10_000.0
    mult = float(cfg.get("imbalance_mult", 2.0))
    min_qty = float(cfg.get("min_wall_qty", 12_000))
    try:
        t = requests.get(
            "https://api.binance.com/api/v3/ticker/bookTicker",
            params={"symbol": symbol},
            timeout=3,
        ).json()
        bid = float(t["bidPrice"]); ask = float(t["askPrice"])
        mid = (bid + ask) / 2.0

        d = requests.get(
            "https://api.binance.com/api/v3/depth",
            params={"symbol": symbol, "limit": 100},
            timeout=3,
        ).json()

        def near(side: str):
            out = []
            for p, q in d[side]:
                p = float(p); q = float(q)
                dist = (mid - p) / mid if side == "bids" else (p - mid) / mid
                if dist <= max(max_dist, min_bps):
                    out.append((p, q))
            return out

        nb = near("bids"); na = near("asks")
        bid_qty = sum(q for _, q in nb)
        ask_qty = sum(q for _, q in na)

        if bid_qty >= max(min_qty, mult * ask_qty) and bid_qty > 0:
            return True, f"Buy wall near (bid {bid_qty:.0f} vs ask {ask_qty:.0f})"
        if ask_qty >= max(min_qty, mult * bid_qty) and ask_qty > 0:
            return True, f"Sell wall near (ask {ask_qty:.0f} vs bid {bid_qty:.0f})"
    except Exception:
        pass
    return False, ""


# ----------------- stublar (opsiyonel modül yoksa) -----------------
class _StubRegime:
    def __init__(self, *_a, **_k): ...
    def snapshot(self) -> dict:
        return {"regime": "UNKNOWN", "ema9": None}


class _StubPredictor:
    def __init__(self, *_a, **_k): ...
    def score(self) -> Tuple[float, dict]:
        return 0.5, {"hh": None}  # orta güven


# ----------------- SymbolEngine -----------------
class SymbolEngine:
    def __init__(self, symbol: str, cfg: dict, news):
        self.s = symbol
        self.cfg = cfg or {}

        # ---- RULES (eksikse defaultlar)
        self.rules = (self.cfg.get("rules") or {
            "level_mode": "pct",
            "abs": {"offset": 0.0, "tp": 4.0, "sl": -3.0},
            "pct": {"offset": 0.2, "tp": 0.8, "sl": -0.6},
            "breakeven_after_sec": 240,
            "min_exit_profit_pct": 0.05,
            "max_hold_sec": 900,
            "per_symbol_modes": {}
        })

        # fiyat/lot formatları
        self.pdec, self.qdec = _ex_decimals(symbol)
        self.pf = "{:." + str(self.pdec) + "f}"
        self.qf = "{:." + str(self.qdec) + "f}"

        # DIP
        d = self.cfg.get("dip", {})
        if DipTracker:
            try:
                self.dip = DipTracker(
                    require_new_dip_after_start=bool(d.get("require_new_dip_after_start", False)),
                    reset_dip_after_sell=bool(d.get("reset_dip_after_sell", True)),
                    window_sec=int(d.get("window_sec", 300)) if d.get("window_sec") is not None else None,
                )
            except Exception:
                self.dip = None
        else:
            self.dip = None

        # PRED
        if PumpPredictor:
            try:
                self.pred = PumpPredictor(symbol, self.cfg.get("predictor", {}))
            except Exception:
                self.pred = _StubPredictor()
        else:
            self.pred = _StubPredictor()

        self.news = news  # dış sınıf

        # REGIME
        if RegimeDetector:
            try:
                self.reg = RegimeDetector(symbol)
            except Exception:
                self.reg = _StubRegime()
        else:
            self.reg = _StubRegime()
        self._last_reg: Dict[str, Any] = {"regime": "UNKNOWN", "ema9": None}

        # pozisyonlar
        self.pos: Dict[str, Optional[dict]] = {"dip": None, "pred": None, "news": None, "ob": None}
        # slot-bazlı son trade zamanı
        self._last_trade_ts: Dict[str, float] = {"dip": 0.0, "pred": 0.0, "news": 0.0, "ob": 0.0}

        self.last_px = 0.0
        self._last_guard: Tuple[str, float, str, float] = ("", 0.0, "", 0.0)

    # ---- yardımcılar
    def _touch_last_trade(self, slot: str) -> None:
        if not isinstance(getattr(self, "_last_trade_ts", None), dict):
            self._last_trade_ts = {"dip": 0.0, "pred": 0.0, "news": 0.0, "ob": 0.0}
        if slot not in self._last_trade_ts:
            self._last_trade_ts[slot] = 0.0
        self._last_trade_ts[slot] = time.time()

    def fmtp(self, x: float) -> str:
        try:
            return self.pf.format(float(x))
        except Exception:
            return str(x)

    def fmtq(self, x: float) -> str:
        try:
            return self.qf.format(float(x))
        except Exception:
            return str(x)

    # --- yeni: sinyalleri 3'lü tuple'a sabitleyen küçük yardımcı ---
    def _as3(self, ret):
        """Sinyali (fired, why, extra) üçlüsüne normalize eder."""
        if isinstance(ret, tuple):
            if len(ret) >= 3: return ret[0], ret[1], ret[2]
            if len(ret) == 2: return ret[0], ret[1], None
            if len(ret) == 1: return ret[0], None, None
        return False, None, None

    # ----------------- seviyeler (dinamik tp/sl + rejim) -----------------
    def levels(self, price: float) -> Tuple[float, float, float]:
        r = self.rules
        mode = r.get("per_symbol_modes", {}).get(self.s, r.get("level_mode", "pct"))
        a, p = r.get("abs", {}), r.get("pct", {})
        a_off = float(a.get("offset", 0.0)); a_tp = float(a.get("tp", 4.0)); a_sl = float(a.get("sl", -3.0))
        p_off = price * (float(p.get("offset", 0.2)) / 100.0)
        p_tp  = price * (float(p.get("tp", 0.8)) / 100.0)
        p_sl  = price * (float(p.get("sl", -0.6)) / 100.0)

        if mode == "abs": off, tp, sl = a_off, a_tp, a_sl
        elif mode == "pct": off, tp, sl = p_off, p_tp, p_sl
        elif mode == "hybrid_max": off, tp, sl = max(a_off, p_off), max(a_tp, p_tp), min(a_sl, p_sl)
        else: off, tp, sl = a_off, a_tp, a_sl

        reg = (self._last_reg or {}).get("regime") or "UNKNOWN"
        mults = self.cfg.get("regime", {
            "multipliers": {
                "TREND": {"tp": 1.4, "sl": 0.9, "off": 1.1},
                "MEAN":  {"tp": 0.9, "sl": 0.7, "off": 0.8},
                "CHOP":  {"tp": 0.6, "sl": 0.6, "off": 1.2},
                "UNKNOWN":{"tp": 1.0, "sl": 1.0, "off": 1.0},
            }
        }).get("multipliers", {})
        m = mults.get(reg, mults.get("UNKNOWN", {"tp":1.0,"sl":1.0,"off":1.0}))
        tp *= float(m.get("tp", 1.0)); sl *= float(m.get("sl", 1.0)); off *= float(m.get("off", 1.0))

        dyn = r.get("dynamic_tpsl", {"enabled": True, "min_scale": 0.6, "max_scale": 1.8})
        if dyn.get("enabled", True):
            try:
                prob, _ = self.pred.score() if self.pred else (0.5, {})
                scale = max(float(dyn.get("min_scale", 0.6)),
                            min(float(dyn.get("max_scale", 1.8)), 0.6 + 1.2 * float(prob)))
                tp *= scale; sl *= scale
            except Exception:
                pass

        return round(off, self.pdec), round(tp, self.pdec), round(sl, self.pdec)

    # ---------------- sinyaller ----------------
    def signal_dip(self, price: float):
        """
        Her durumda 3'lü tuple döndürür:
          (fired: bool, reason: str | None, extra: dict | None)
        """
        if not getattr(self, "dip", None):
            return False, None, None

        try:
            dip_val = self.dip.update(float(price))
        except Exception:
            return False, None, None

        if dip_val is None or not self.dip.can_buy():
            return False, None, None

        try:
            off, tp, sl = self.levels(price)  # levels 3'lü döner: (off, tp, sl)
            fired = float(price) <= (float(dip_val) + float(off))

            # Rejim filtresi (ema9 yoksa dip çizgisiyle guard)
            ema9 = (self._last_reg or {}).get("ema9")
            if ema9 is None:
                ema9 = float(dip_val)
            fired = bool(fired and (float(price) >= float(ema9)))

            return self._as3((fired, "DIP", None))
        except Exception:
            return False, None, None

    def signal_pred(self, price: float):
        try:
            prob, info = self.pred.score() if self.pred else (0.0, {})
        except Exception:
            prob, info = 0.0, {}
        need = float((self.cfg.get("predictor", {}) or {}).get("enter_prob", 0.7))
        if prob < need:
            return self._as3((False, "", {"prob": prob}))
        hh = float(info.get("hh", price * 1.001)) if isinstance(info, dict) else (price * 1.001)
        fired = price >= hh * 1.001
        return self._as3((fired, f"prob={prob:.2f} hh={self.fmtp(hh)}", {"prob": prob}))

    def signal_news(self):
        try:
            return self._as3(self.news.maybe_signal(self.s))
        except Exception:
            return False, "", None

    def signal_ob(self, ob_cfg: dict):
        fired, why = _book_wall_signal(self.s, ob_cfg or {})
        return self._as3((fired, why, None))

    # ----------------- al-sat -----------------
    def _open(self, slot: str, price: float, cash: float, why: str, label: str, conf: float) -> float:
        fees = self.cfg.get("fees", {})
        fee_pct = float(fees.get("taker_pct", 0.10)) / 100.0
        slip    = float(fees.get("slippage_pct", 0.02)) / 100.0
        adj_px = price * (1.0 + slip)
        qty = round((cash * (1.0 - fee_pct)) / max(adj_px, 1e-12), self.qdec)
        if qty <= 0:
            return 0.0
        self.pos[slot] = {
            "avg": adj_px,
            "qty": float(qty),
            "open_ts": time.time(),
            "spent": float(cash),
            "why": str(label),
            "conf": float(conf),
        }
        self._touch_last_trade(slot)

        msg = (
            "🟩 <b>ALIM</b> "
            f"[<code>{reason_tag(label)}</code>]\n"
            f"Parite: <b>{self.s}</b>\n"
            f"Fiyat: <code>{self.fmtp(adj_px)}</code> • Qty: <code>{self.fmtq(qty)}</code>\n"
            f"Sebep: {why}\n"
            f"Risk: <code>{conf:.2f}</code> • Harcanan: <code>{cash:.2f}</code> USD"
        )
        tg_send(msg, parse_mode="HTML")
        log_event("open", slot=slot, sym=self.s, price=adj_px, qty=qty, why=why, conf=conf)
        return float(cash)

    def maybe_open(self, slot: str, price: float, free_cash: float, why: str, label: str, conf: float) -> float:
        if self.pos.get(slot) is not None or free_cash <= 0:
            return 0.0
        return self._open(slot, price, free_cash, why, label, conf)

    def maybe_dca(self, slot: str, price: float, free_cash: float) -> float:
        dca = self.cfg.get("dca", {"enabled": False})
        if self.pos.get(slot) is None or not dca.get("enabled", False):
            return 0.0
        pos = self.pos[slot]
        if pos.get("layers", 1) >= int(dca.get("max_layers", 1)):
            return 0.0
        step_mode = dca.get("step_mode", "pct"); step = float(dca.get("step", 0.5))
        last = float(pos["avg"]); step_abs = last * (step / 100.0) if step_mode == "pct" else step
        if price > last - step_abs:
            return 0.0
        spend = min(free_cash, free_cash * 0.50)
        qty = round(spend / max(price, 1e-12), self.qdec)
        if qty <= 0:
            return 0.0
        old_q, old_a = float(pos["qty"]), float(pos["avg"])
        new_q = old_q + qty
        new_a = (old_a * old_q + price * qty) / max(new_q, 1e-12)
        pos.update({
            "qty": new_q,
            "avg": new_a,
            "layers": int(pos.get("layers", 1)) + 1,
            "spent": float(pos.get("spent", 0.0)) + float(spend)
        })
        self._touch_last_trade(slot)

        msg = (
            f"🟨 <b>DCA {slot.upper()}</b> L{pos['layers']}\n"
            f"Parite: <b>{self.s}</b>\n"
            f"Fiyat: <code>{self.fmtp(price)}</code> • Qty: <code>{self.fmtq(qty)}</code>\n"
            f"Ort. Giriş: <code>{self.fmtp(new_a)}</code> • Toplam Qty: <code>{self.fmtq(new_q)}</code>"
        )
        tg_send(msg, parse_mode="HTML")
        log_event("dca", slot=slot, sym=self.s, price=price, qty=qty, layers=pos["layers"])
        return float(spend)

    def _news_trailing_exit(self, pos: dict, px: float) -> Optional[str]:
        trail = float(self.cfg.get("news_mode", {}).get("trailing_pct", 2.5)) / 100.0
        emerg = float(self.cfg.get("news_mode", {}).get("emergency_sl_pct", 0.5)) / 100.0
        e = float(pos["avg"])
        peak = float(pos.get("peak", e)); peak = max(peak, px); pos["peak"] = peak
        if px <= e * (1.0 + emerg):
            return "SL"
        if px <= peak * (1.0 - trail):
            return "TP/DECAY"
        return None

    def maybe_close(self, slot: str, price: float) -> Tuple[bool, float, str]:
        pos = self.pos.get(slot)
        if not pos:
            return False, 0.0, ""
        e = float(pos["avg"]); q = float(pos["qty"])
        fees = self.cfg.get("fees", {})
        fee_pct = float(fees.get("taker_pct", 0.10)) / 100.0
        slip    = float(fees.get("slippage_pct", 0.02)) / 100.0
        sell_px = price * (1.0 - slip)

        reason = None
        if slot == "news":
            reason = self._news_trailing_exit(pos, sell_px)
        else:
            off, tp, sl = self.levels(price)
            elapsed = time.time() - float(pos.get("open_ts", time.time()))
            be_after = int(self.rules.get("breakeven_after_sec", 240))
            be_min   = float(self.rules.get("min_exit_profit_pct", 0.05))
            pnl_pct  = (sell_px / (e or 1.0) - 1.0) * 100.0
            if sell_px >= e + tp:
                reason = "TP"
            elif sell_px <= e + sl:
                reason = "SL"
            elif elapsed >= int(self.rules.get("max_hold_sec", 900)):
                reason = "TIME_SL"
            elif elapsed >= be_after and pnl_pct >= be_min:
                reason = "BREAKEVEN/MINPROFIT"

        if reason:
            pnl = (sell_px - e) * q * (1.0 - fee_pct)
            pnl_pct = ((sell_px / (e or 1.0)) - 1.0) * 100.0
            msg = (
                "🟥 <b>SATIŞ</b> "
                f"[<code>{reason_tag(pos.get('why','-'))}</code>]\n"
                f"Parite: <b>{self.s}</b>\n"
                f"Fiyat: <code>{self.fmtp(sell_px)}</code>\n"
                f"Sebep: <code>{reason_tag(reason)}</code>\n"
                f"PnL: <b>{pnl:+.2f} USD</b> "
                f"(<code>{pnl_pct:+.2f}%</code>)"
            )
            tg_send(msg, parse_mode="HTML")
            log_trade(slot, self.s, float(pnl), extra={"reason": reason, "sell": sell_px, "avg": e, "qty": q})
            self.pos[slot] = None
            try:
                if self.dip:
                    self.dip.on_sell()
            except Exception:
                pass
            self._touch_last_trade(slot)
            return True, float(pnl), reason

        return False, 0.0, ""
