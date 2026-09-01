
# -*- coding: utf-8 -*-
from __future__ import annotations
import time, json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# --- injected: web audit & helpers ---
try:
    from Proje1.core.web_audit import enable as _audit_enable
    _audit_enable()
except Exception:
    pass
def _tick_external_fetch():
    try:
        from Proje1.modules.news_fetcher import fetch_once
        fetch_once()
    except Exception:
        pass
def _tick_equity_guard():
    try:
        from Proje1.modules.equity_guard import guard
        guard()
    except Exception:
        pass
def _tick_kpi():
    try:
        from Proje1.core.reporting_kpi import write_snapshot
        write_snapshot()
    except Exception:
        pass


# --- ABSOLUTE IMPORTS (package: Proje1.core) ---
from Proje1.core.market import get_price
from Proje1.core.state import save_state, load_state
from Proje1.core.utils_io import (
    tg_send, ensure_tg, ev_ok, reason_tag, write_event_cash, tg_ready
)
from Proje1.core.symbol_engine import SymbolEngine
from Proje1.core.alloc import slot_cash, size_with_conf, can_trade_now
from Proje1.core.reporting import Reporter
from Proje1.core.candles_guard import allow_long, should_bearish_exit
from Proje1.core.news_hunter import NewsHunter
from Proje1.core.watchlist_manager import WatchListManager
from Proje1.core.strategy_loader import StrategyLoader
from Proje1.core.logger_utils import log_trade, log_event, log_brain
from Proje1.core.brain_hook import decide_trade, brain_overrides

# Brian 2026 shadow brain (additive; never executes orders)
try:
    from Proje1.brian2026.engine import BrianEngine as _Brian2026Engine
    from Proje1.brian2026.bridge import LegacyShadowBridge as _Brian2026Bridge
except Exception:
    _Brian2026Engine = None  # type: ignore
    _Brian2026Bridge = None  # type: ignore

# dış veri opsiyonel — yoksa stub
try:
    from Proje1.core.external_data import get_scores
except Exception:
    def get_scores(*a, **k):
        return {"news_shock": 0.0, "macro_risk": 1.0, "flow": "neutral"}

from Proje1.core.learning import intra_day_bandit, day_end_grid
from Proje1.core.autopatch_sandbox import sandbox_try
from Proje1.core.brain_learn import adjust_confidence, retrain_daily
# L10 meta/WR cache’i init’te hazırlamak için (EKLENDİ)
try:
    from Proje1.core.brain_learn import refresh_cache as _brain_refresh_cache
except Exception:
    _brain_refresh_cache = None

# L9: self-heal, file-watcher, L8 log & market intel
from Proje1.core.brain_selfheal import (
    ensure_selfheal_watcher, report_exception, SelfHeal, selfheal_heartbeat
)
from Proje1.core.log_ext import write_trades_full_row, log_market_intel
from Proje1.core.file_watcher import ensure_file_watcher

# L10: beyin autocode (opsiyonel)
try:
    from Proje1.core.brain_autocoder import brain_tick as _brain_tick
except Exception:
    _brain_tick = None

# ---------------- L11: EVO entegrasyonu (esnek import) ----------------
_evo_fn = None
try:
    from Proje1.core.evo_runner import tick as _evo_fn  # type: ignore
except Exception:
    try:
        from Proje1.core.evo_runner import evo_step as _evo_fn  # type: ignore
    except Exception:
        try:
            from Proje1.core.evo_runner import evo_tick as _evo_fn  # type: ignore
        except Exception:
            _evo_fn = None  # type: ignore

# ---------------- L12: DRIFT entegrasyonu (esnek import) ----------------
_drift_fn = None
try:
    from Proje1.core.drift_watch import drift_ping as _drift_fn  # type: ignore
except Exception:
    try:
        from Proje1.core.drift_watch import tick as _drift_fn  # type: ignore
    except Exception:
        _drift_fn = None  # type: ignore

# ---------------- L13: Auto-Repair entegrasyonu (yeni) ----------------
from Proje1.core.selfheal_l13 import ensure_autorepair, l13_on_exception, l13_heartbeat

# ---------------- L14: Governor / Auto-Explore (yeni, esnek import) ----------------
_l14_ensure = None
_l14_tick = None
_l14_on_exception = None
_l14_heartbeat = None
try:
    from Proje1.core.l14_governor import ensure_l14 as _l14_ensure  # type: ignore
except Exception:
    _l14_ensure = None
try:
    from Proje1.core.l14_governor import l14_tick as _l14_tick  # type: ignore
except Exception:
    _l14_tick = None
try:
    from Proje1.core.l14_governor import l14_on_exception as _l14_on_exception  # type: ignore
except Exception:
    _l14_on_exception = None
try:
    from Proje1.core.l14_governor import l14_heartbeat as _l14_heartbeat  # type: ignore
except Exception:
    _l14_heartbeat = None

# ---------------- L60: Intent Evolver (esnek import) ----------------
_l60_ensure = None
_l60_tick = None
_l60_on_exception = None
_l60_heartbeat = None
try:
    from Proje1.core.intent_evolver import ensure_l60 as _l60_ensure  # type: ignore
except Exception:
    try:
        from Proje1.core.intent_evolver import ensure_intent as _l60_ensure  # type: ignore
    except Exception:
        _l60_ensure = None
try:
    from Proje1.core.intent_evolver import l60_tick as _l60_tick  # type: ignore
except Exception:
    try:
        from Proje1.core.intent_evolver import intent_tick as _l60_tick  # type: ignore
    except Exception:
        _l60_tick = None
try:
    from Proje1.core.intent_evolver import l60_on_exception as _l60_on_exception  # type: ignore
except Exception:
    try:
        from Proje1.core.intent_evolver import intent_on_exception as _l60_on_exception  # type: ignore
    except Exception:
        _l60_on_exception = None
try:
    from Proje1.core.intent_evolver import l60_heartbeat as _l60_heartbeat  # type: ignore
except Exception:
    try:
        from Proje1.core.intent_evolver import intent_heartbeat as _l60_heartbeat  # type: ignore
    except Exception:
        _l60_heartbeat = None

# ---------------- OMEGA Modules (opsiyonel & güvenli fallback) ----------------
#   Proje1/modules altında bulunuyorsa aktif olur; yoksa sessizce kapalı kalır.
try:
    from Proje1.modules.meta_label import MetaLabeler
except Exception:
    MetaLabeler = None  # type: ignore

try:
    from Proje1.modules.confidence_calibrator import ConfidenceCalibrator
except Exception:
    ConfidenceCalibrator = None  # type: ignore

try:
    from Proje1.modules.bandit_tp_sl import BanditTPSL
except Exception:
    BanditTPSL = None  # type: ignore

try:
    from Proje1.modules.news_guard import NewsGuard
except Exception:
    NewsGuard = None  # type: ignore

try:
    from Proje1.modules.cluster_guard import ClusterGuard
except Exception:
    ClusterGuard = None  # type: ignore

try:
    from Proje1.modules.counterfactual_logger import CounterfactualLogger
except Exception:
    CounterfactualLogger = None  # type: ignore

try:
    from Proje1.modules.error_taxonomy import ErrorTaxonomy
except Exception:
    ErrorTaxonomy = None  # type: ignore


def _as3(ret):
    """Sinyali güvenli şekilde (fired, why, extra) üçlüsüne çevir."""
    if isinstance(ret, tuple):
        if len(ret) >= 3: return ret[0], ret[1], ret[2]
        if len(ret) == 2: return ret[0], ret[1], None
        if len(ret) == 1: return ret[0], None, None
    return False, None, None


# ------------ Risk Guard ------------
class RiskGuard:
    def __init__(self, rc: dict):
        self.cap = float(rc.get("daily_max_loss_usd", 0.0))
        self.cool_min = int(rc.get("cooldown_min", 30))
        self._day = None
        self.realized = 0.0
        self.cool_until = 0.0

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def on_realized(self, pnl: float) -> float | None:
        day = self._today()
        if self._day != day:
            self._day = day
            self.realized = 0.0
            self.cool_until = 0.0
        self.realized += float(pnl)
        if self.cap > 0 and self.realized <= -abs(self.cap):
            self.cool_until = time.time() + max(60, self.cool_min * 60)
            return self.cool_until
        return None


# --------------- Bot ---------------
class Bot:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.cash = float(self.cfg.get("cash", 500.0))
        self.console = bool(self.cfg.get("console_verbose", True))

        # Portfolio & sizing
        self.alloc: Dict[str, float] = self.cfg.get(
            "portfolio", {"dip": 0.40, "pred": 0.30, "news": 0.20, "ob": 0.10}
        )
        self.entry_frac: Dict[str, float] = self.cfg.get(
            "entry_frac", {"dip": 0.40, "pred": 0.40, "news": 0.60, "ob": 0.50}
        )
        self.sizing = self.cfg.get("sizing", {"min_mult": 0.5, "max_mult": 2.0})
        self.max_open_total = int(self.cfg.get("max_total_open_positions", 3))
        self.max_per_symbol = int(self.cfg.get("max_open_per_symbol", 2))

        # EV & fees
        self.use_ev_filter = bool(self.cfg.get("use_ev_filter", False))
        self.ev_min = float(self.cfg.get("ev_min", 0.0))
        fees = self.cfg.get("fees", {}) or {}
        self.fee_pct = float(fees.get("taker_pct", 0.10)) / 100.0
        self.slip_pct = float(fees.get("slippage_pct", 0.02)) / 100.0

        # Risk
        self.risk = RiskGuard(self.cfg.get("risk", {}))

        # Frequency (GERÇEK UYGULAMA)
        fc = self.cfg.get("freq_ctrl", {"min_sec_between_trades": 20, "max_trades_per_hour": 60})
        self.min_gap = int(fc.get("min_sec_between_trades", 20))
        self.max_tph = int(fc.get("max_trades_per_hour", 60))
        self._trade_hist: List[Tuple[float, str]] = []  # (ts, symbol)

        # Reporter
        self.reporter = Reporter(self.cfg.get("report", {}))

        # Telegram
        ensure_tg(self.cfg)
        try:
            if not tg_ready():
                from Proje1.core.logger_utils import log_event as _le
                _le("tg_not_ready", msg="Telegram token/chat eksik veya kapalı")
        except Exception:
            pass

        # Modules (opsiyonel, güvenli fallback)
        mc = dict(self.cfg.get("modules", {}))
        self.mlabel = MetaLabeler(self.cfg) if (MetaLabeler and mc.get("meta_label", True)) else None
        self.calib  = ConfidenceCalibrator(self.cfg) if (ConfidenceCalibrator and mc.get("confidence_calibrator", True)) else None
        self.bandit = BanditTPSL(self.cfg) if (BanditTPSL and mc.get("bandit_tp_sl", True)) else None
        self.nguard = NewsGuard(self.cfg) if (NewsGuard and mc.get("news_guard", True)) else None
        self.cguard = ClusterGuard(self.cfg) if (ClusterGuard and mc.get("cluster_guard", True)) else None
        self.cflog  = CounterfactualLogger(self.cfg) if (CounterfactualLogger and mc.get("counterfactual_logger", True)) else None
        self.etax   = ErrorTaxonomy(self.cfg) if (ErrorTaxonomy and mc.get("error_taxonomy", True)) else None

        # Modules: yardımcı kısım
        self._mod_on = {
            "meta_label": bool(self.mlabel),
            "confidence_calibrator": bool(self.calib),
            "bandit_tp_sl": bool(self.bandit),
            "news_guard": bool(self.nguard),
            "cluster_guard": bool(self.cguard),
            "counterfactual_logger": bool(self.cflog),
            "error_taxonomy": bool(self.etax),
        }
        try:
            log_event("modules_init", **self._mod_on)
        except Exception:
            pass

        # Brian 2026 shadow-first learning layer. This cannot place or veto orders.
        self.brian2026 = None
        self._brian2026_enabled = False
        try:
            b26 = dict(self.cfg.get("brian2026", {}) or {})
            self._brian2026_enabled = bool(b26.get("shadow_enabled", True))
            if self._brian2026_enabled and _Brian2026Engine is not None and _Brian2026Bridge is not None:
                engine_cfg = dict(b26.get("engine", {}) or {})
                engine_cfg["shadow_only"] = True
                self.brian2026 = _Brian2026Bridge(_Brian2026Engine(engine_cfg))
                log_event("brian2026_init", shadow_only=True)
        except Exception as _b26e:
            self.brian2026 = None
            self._brian2026_enabled = False
            try: log_event("brian2026_init_error", error=str(_b26e))
            except Exception: pass

        # Modules
        self.news = NewsHunter(self.cfg.get("news_mode", {}))
        self.wlm = WatchListManager(self.cfg)
        self.syms: Dict[str, SymbolEngine] = {}
        self._sync_symbols(self.wlm.active())
        self.plugins = StrategyLoader(self.cfg)

        # Restore state
        rest_cash, rest_pos = load_state(self.cash)
        self.cash = float(rest_cash)
        for s, slots in (rest_pos or {}).items():
            if s not in self.syms:
                self.syms[s] = SymbolEngine(s, self.cfg, self.news)
            st = self.syms[s]
            for k, v in slots.items():
                st.pos[k] = v

        log_event("restore", cash=self.cash, sym=len(self.syms))
        try:
            tg_send(
                "🟣 <b>Monster Coins Pro</b> — Telegram hazır.\n"
                "🧠 <b>Brian</b> authority=ON\n"
                f"💰 <b>Kasa</b>: <code>{self.cash:.2f} USD</code>",
                parse_mode="HTML",
            )
        except Exception:
            pass

        # L9: Self-heal & File-watcher
        ensure_selfheal_watcher(self.cfg)
        self.sh = SelfHeal.get()
        ensure_file_watcher(self.cfg)

        # L13: Auto-repair
        ensure_autorepair(self.cfg)

        # L14: Governor init (opsiyonel)
        try:
            if _l14_ensure is not None:
                _l14_ensure(self.cfg)
        except Exception:
            pass

        # L60: Intent Evolver init (opsiyonel)
        try:
            if _l60_ensure is not None:
                _l60_ensure(self.cfg)
        except Exception:
            pass

        # L10: brain tick aralığı
        bcfg = (self.cfg.get("brain") or {}) if isinstance(self.cfg, dict) else {}
        self._brain_tick_sec = int(bcfg.get("tick_sec", 30))
        self._last_brain_tick = 0.0

        # L10: meta/WR cache warm
        try:
            if _brain_refresh_cache is not None:
                _brain_refresh_cache(self.cfg)
        except Exception:
            pass

        # L11: EVO scheduler
        ecfg = (self.cfg.get("evo") or {}) if isinstance(self.cfg, dict) else {}
        self._evo_enabled = bool(ecfg.get("enabled", True))
        self._evo_tick_sec = int(ecfg.get("tick_sec", 300))
        self._evo_last = 0.0

        # L12: DRIFT scheduler
        dcfg = (self.cfg.get("drift") or {}) if isinstance(self.cfg, dict) else {}
        self._drift_enabled = bool(dcfg.get("enabled", True))
        self._drift_tick_sec = int(dcfg.get("tick_sec", 120))
        self._drift_last = 0.0

        # L14: Governor scheduler
        l14cfg = (self.cfg.get("l14") or {}) if isinstance(self.cfg, dict) else {}
        self._l14_enabled = bool(l14cfg.get("enabled", True))
        self._l14_tick_sec = int(l14cfg.get("tick_sec", 180))
        self._l14_last = 0.0

        # L60: Intent Evolver scheduler
        l60cfg = (self.cfg.get("l60") or {}) if isinstance(self.cfg, dict) else {}
        self._l60_enabled = bool(l60cfg.get("enabled", True))
        self._l60_tick_sec = int(l60cfg.get("tick_sec", 240))
        self._l60_last = 0.0

        # plugin hot-reload sinyali
        self._plugin_dirs = [
            Path("live/strategies"),
            Path("strategies"),
            Path("core/strategies"),
            Path("model/strategies"),
        ]
        self._plugin_sig = self._scan_plugins_sig()
        self._last_plugin_scan = 0.0
        self._plugin_scan_sec = 30

        self._last_events_emit = 0.0

    # ------------ helpers ------------
    def _sync_symbols(self, active_list):
        for s in active_list:
            if s not in self.syms:
                self.syms[s] = SymbolEngine(s, self.cfg, self.news)
                log_event("watch_add", sym=s)
        for s in list(self.syms.keys()):
            if s not in active_list:
                st = self.syms[s]
                if not any(st.pos.get(k) for k in ["dip", "pred", "news", "ob"]):
                    del self.syms[s]
                    log_event("watch_del", sym=s)

    def open_count(self) -> int:
        return sum(1 for st in self.syms.values() for v in st.pos.values() if v is not None)

    def symbol_open_count(self, s: str) -> int:
        st = self.syms[s]
        return sum(1 for v in st.pos.values() if v is not None)

    def _write_state_snapshot(self) -> None:
        snap: Dict[str, Any] = {"cash": float(self.cash), "positions": {}}
        for sym, st in self.syms.items():
            slots = {}
            for k, pos in st.pos.items():
                if pos:
                    slots[k] = {"avg": float(pos["avg"]), "qty": float(pos["qty"])}
            if slots:
                snap["positions"][sym] = slots
        p = Path("runtime/state.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    # trade rate limits (EKLENDİ)
    def _trade_allowed_now(self) -> bool:
        """min gap ve max trades/hour uygula."""
        now = time.time()
        # min gap
        if self._trade_hist:
            last_ts = self._trade_hist[-1][0]
            if now - last_ts < max(0, self.min_gap):
                return False
        # tph
        one_hour_ago = now - 3600
        self._trade_hist = [(ts, sym) for ts, sym in self._trade_hist if ts >= one_hour_ago]
        if len(self._trade_hist) >= max(1, self.max_tph):
            return False
        return True

    # L60/L14 plugin yeni dosyaları algıla
    def _scan_plugins_sig(self) -> tuple[int, int]:
        latest = 0
        count = 0
        for d in self._plugin_dirs:
            if d.exists() and d.is_dir():
                for f in d.rglob("*.py"):
                    try:
                        st = f.stat()
                        latest = max(latest, int(st.st_mtime))
                        count += 1
                    except Exception:
                        pass
        return (latest, count)

    def _maybe_reload_plugins(self, now: float) -> None:
        if now - self._last_plugin_scan < self._plugin_scan_sec:
            return
        self._last_plugin_scan = now
        new_sig = self._scan_plugins_sig()
        if new_sig != self._plugin_sig:
            try:
                self.plugins = StrategyLoader(self.cfg)
                self._plugin_sig = new_sig
                log_event("plugins_reload", latest=new_sig[0], file_count=new_sig[1])
                try:
                    tg_send("♻️ Yeni stratejiler algılandı, StrategyLoader yeniden yüklendi.")
                except Exception:
                    pass
            except Exception as _e:
                try:
                    tg_send(f"[WARN] StrategyLoader reload başarısız: {_e}")
                except Exception:
                    pass

    # ------ Brian 2026: shadow review / outcome linking ------
    def _brian2026_review(self, s: str, slot: str, st: SymbolEngine, px: float, conf: float, reg: str) -> None:
        if not self.brian2026:
            return
        try:
            alphas = getattr(st, "alphas", {}) or {}
            features = {
                "spread_bps": float(getattr(st, "spread_bps", 0.0) or 0.0),
                "book_imbalance": float(getattr(st, "book_imbalance", 0.0) or 0.0),
                "wall_score": float(getattr(st, "wall_score", 0.0) or 0.0),
                "breakout_score": float(getattr(st, "breakout_score", 0.0) or 0.0),
                "volume_z": float(getattr(st, "volume_z", 0.0) or 0.0),
                "acceleration": float(getattr(st, "acceleration", 0.0) or 0.0),
                "rsi": float(getattr(st, "rsi", 50.0) or 50.0),
                "return_5": float(getattr(st, "return_5", 0.0) or 0.0),
                "zscore": float(getattr(st, "zscore", 0.0) or 0.0),
                "bb_position": float(getattr(st, "bb_position", 0.5) or 0.5),
                "ema_fast": float(getattr(st, "ema_fast", px) or px),
                "ema_slow": float(getattr(st, "ema_slow", px) or px),
                "ema_slope_pct": float(getattr(st, "ema_slope_pct", 0.0) or 0.0),
                "atr_pct": float(getattr(st, "atr_pct", 0.0) or 0.0),
                "legacy_alpha_pred": float(alphas.get("pred", 0.0) or 0.0) if isinstance(alphas, dict) else 0.0,
                "legacy_alpha_dip": float(alphas.get("dip", 0.0) or 0.0) if isinstance(alphas, dict) else 0.0,
            }
            account = {
                "daily_pnl_pct": (float(self.risk.realized) / max(float(self.cash), 1e-9)) * 100.0,
                "drawdown_pct": 0.0,
                "open_positions": self.open_count(),
            }
            d = self.brian2026.review(
                symbol=s, price=px, regime=reg, legacy_slot=slot,
                legacy_confidence=conf, features=features, account=account
            )
            log_brain("brian2026_shadow", {"symbol": s, "slot": slot, "decision": d})
        except Exception as _e:
            try: log_event("brian2026_review_error", symbol=s, slot=slot, error=str(_e))
            except Exception: pass

    def _brian2026_mark_open(self, s: str, slot: str, px: float) -> None:
        if not self.brian2026:
            return
        try:
            self.brian2026.mark_open(s, slot, px)
        except Exception:
            pass

    def _brian2026_mark_close(self, s: str, slot: str, px: float, pnl: float, reason: str) -> None:
        if not self.brian2026:
            return
        try:
            self.brian2026.mark_close(s, slot, px, pnl, reason)
        except Exception:
            pass

    # ------ OMEGA: Guard & MetaLabel & Calib & Bandit entegrasyonu ------
    def _pretrade_pipeline(self, symbol: str, slot: str, st: SymbolEngine, px: float,
                           spend: float, tp: float, sl: float, conf: float, reg: str) -> tuple[Optional[dict], str]:
        """
        Sinyal → (opsiyonel) kalibrasyon, guard'lar, meta-label → öneri (proposal)
        Bir şey bloklarsa (None, reason) döner ve counterfactual 'skip' loglanır.
        """
        if spend <= 0 or px <= 0:
            return None, "invalid_spend_or_px"
        if not self._trade_allowed_now():
            return None, "rate_limit"

        qty = spend / px
        try:
            ext_scores = get_scores(symbol, self.cfg)
        except Exception:
            ext_scores = {"news_shock": 0.0, "macro_risk": 1.0, "flow": "neutral"}

        market_ctx = {
            "regime": reg,
            "vol": getattr(st, "vol_norm", 0.5),
            "spread_bps": getattr(st, "spread_bps", 1.0),
            "alphas": getattr(st, "alphas", {"pred": 0.5, "dip": 0.5}),
            "ext": ext_scores,
            "px": px,
        }

        # Haber guard (bloklarsa çık)
        if self.nguard and not self.nguard.allow(symbol, market_ctx):
            return None, "news_guard"

        # Cluster guard (örn. korelasyon / yığılma)
        if self.cguard and not self.cguard.allow(symbol, slot, market_ctx):
            return None, "cluster_guard"

        # Confidence kalibrasyonu (varsa)
        if self.calib:
            try:
                conf = float(self.calib.calibrate(symbol, slot, conf, market_ctx))
            except Exception:
                pass

        # Bandit TP/SL önerisi (varsa)
        if self.bandit:
            try:
                b = self.bandit.suggest(symbol, slot, px, market_ctx)
                if isinstance(b, dict):
                    tp = float(b.get("tp", tp))
                    sl = float(b.get("sl", sl))
            except Exception:
                pass

        proposal = {"side": "BUY", "qty": qty, "price": px, "tp": tp, "sl": sl, "confidence": conf}

        # Meta-labeling (varsa)
        if self.mlabel:
            try:
                features = {
                    "regime": reg,
                    "vol": market_ctx["vol"],
                    "spread_bps": market_ctx["spread_bps"],
                    "alphas": market_ctx["alphas"],
                    "ext": market_ctx["ext"],
                    "slot": slot,
                    "px": px,
                    "tp": tp,
                    "sl": sl,
                    "conf": conf,
                }
                ml_ok = bool(self.mlabel.approve(symbol, features))
                if not ml_ok:
                    return None, "meta_label_reject"
            except Exception:
                # meta-label hatası varsa işlem durdurma—devam et
                pass

        return proposal, "ok"

    # ------ Brian decision gate ------
    def _brain_gate(
        self, s: str, slot: str, st: SymbolEngine, px: float,
        spend: float, tp: float, sl: float, conf: float, reg: str
    ) -> dict | None:
        # Pretrade pipeline
        prop, reason = self._pretrade_pipeline(s, slot, st, px, spend, tp, sl, conf, reg)
        if prop is None:
            # counterfactual skip
            try:
                if self.cflog:
                    self.cflog.log("skip", {
                        "symbol": s, "slot": slot, "why": reason, "regime": reg,
                        "px": px, "spend": spend, "tp": tp, "sl": sl, "conf": conf
                    })
            except Exception:
                pass
            return None

        # Market intel: yüksek shock'u not al
        try:
            ext_scores = get_scores(s, self.cfg)
            if (ext_scores or {}).get("news_shock", 0.0) > 0.8:
                log_market_intel(kind="shock", sym=s, msg="news_shock", payload=ext_scores)
        except Exception:
            pass

        market_ctx = {
            "regime": reg,
            "vol": getattr(st, "vol_norm", 0.5),
            "spread_bps": getattr(st, "spread_bps", 1.0),
            "alphas": getattr(st, "alphas", {"pred": 0.5, "dip": 0.5}),
            "ext": (ext_scores if 'ext_scores' in locals() else {"news_shock":0.0,"macro_risk":1.0,"flow":"neutral"}),
        }

        # Brian karar kapısı
        dec = decide_trade(s, slot, prop, market_ctx, self.cfg)
        log_brain("route_decision", {"symbol": s, "slot": slot, "proposal": prop, "market": market_ctx, "decision": dec})

        if dec.get("action") == "reject":
            try:
                if self.cflog:
                    self.cflog.log("skip", {
                        "symbol": s, "slot": slot, "why": dec.get("reason","reject"), "regime": reg,
                        "px": px, "spend": spend, "tp": prop.get("tp"), "sl": prop.get("sl"),
                        "conf": prop.get("confidence")
                    })
            except Exception:
                pass
            return None
        if dec.get("action") == "adjust":
            for k, v in (dec.get("update") or {}).items():
                prop[k] = v

        # Brian 2026 watches the exact trade the legacy brain is about to take.
        # Its decision is logged only; it cannot alter prop in shadow mode.
        self._brian2026_review(s, slot, st, px, float(prop.get("confidence", conf)), reg)

        # take log
        try:
            if self.cflog:
                self.cflog.log("take", {
                    "symbol": s, "slot": slot, "regime": reg,
                    "px": px, "tp": prop.get("tp"), "sl": prop.get("sl"),
                    "conf": prop.get("confidence"), "qty": prop.get("qty")
                })
        except Exception:
            pass
        return prop

    # --------------- run ---------------
    def run(self):
        last_learn_ping = 0.0
        last_daily_retrain = None
        last_aux_tick = 0.0

        while True:
            try:
                now = time.time()

                # Full-tier auxiliary jobs (were previously defined but never scheduled).
                if now - last_aux_tick >= 900:
                    _tick_external_fetch()
                    _tick_equity_guard()
                    _tick_kpi()
                    last_aux_tick = now

                # L9 heartbeat
                try: selfheal_heartbeat()
                except Exception: pass
                # L13 heartbeat
                try: l13_heartbeat()
                except Exception: pass
                # L14 heartbeat (opsiyonel)
                try:
                    if _l14_heartbeat is not None:
                        _l14_heartbeat()
                except Exception: pass
                # L60 heartbeat (opsiyonel)
                try:
                    if _l60_heartbeat is not None:
                        _l60_heartbeat()
                except Exception: pass

                # L10: brain tick
                if _brain_tick is not None and (now - self._last_brain_tick) >= max(5, self._brain_tick_sec):
                    try:
                        ctx = {
                            "ts": now,
                            "cash": self.cash,
                            "open_positions": self.open_count(),
                            "symbols": list(self.syms.keys()),
                        }
                        _brain_tick(self.cfg, ctx)
                    except Exception:
                        pass
                    self._last_brain_tick = now

                # L11: EVO
                if (_evo_fn is not None) and self._evo_enabled and (now - self._evo_last) >= max(30, self._evo_tick_sec):
                    try:
                        evo_ctx = {
                            "ts": now,
                            "cash": self.cash,
                            "open_positions": self.open_count(),
                            "symbols": list(self.syms.keys()),
                        }
                        try: _evo_fn(self.cfg, evo_ctx)  # type: ignore
                        except TypeError: _evo_fn(self.cfg)  # type: ignore
                    except Exception:
                        pass
                    self._evo_last = now

                # L12: DRIFT
                if (_drift_fn is not None) and self._drift_enabled and (now - self._drift_last) >= max(30, self._drift_tick_sec):
                    try:
                        dctx = {
                            "ts": now,
                            "cash": self.cash,
                            "open_positions": self.open_count(),
                            "symbols": list(self.syms.keys()),
                        }
                        try: _drift_fn(self.cfg, dctx)  # type: ignore
                        except TypeError: _drift_fn(self.cfg)  # type: ignore
                    except Exception:
                        pass
                    self._drift_last = now

                # L14: Governor
                if (self._l14_enabled and (_l14_tick is not None) and
                    (now - self._l14_last) >= max(15, self._l14_tick_sec)):
                    try:
                        gctx = {
                            "ts": now,
                            "cash": self.cash,
                            "open_positions": self.open_count(),
                            "symbols": list(self.syms.keys()),
                        }
                        _l14_tick(self.cfg, gctx)  # type: ignore
                    except Exception:
                        pass
                    self._l14_last = now

                # L60: Intent Evolver
                if (self._l60_enabled and (_l60_tick is not None) and
                    (now - self._l60_last) >= max(15, self._l60_tick_sec)):
                    try:
                        ictx = {
                            "ts": now,
                            "cash": self.cash,
                            "open_positions": self.open_count(),
                            "symbols": list(self.syms.keys()),
                            "alloc": dict(self.alloc),
                            "entry_frac": dict(self.entry_frac),
                        }
                        _l60_tick(self.cfg, ictx)  # type: ignore
                    except Exception:
                        pass
                    self._l60_last = now

                # plugin reload kontrolü
                self._maybe_reload_plugins(now)

                # risk cooldown
                if self.risk.cool_until and now < self.risk.cool_until:
                    if now - self._last_events_emit > 5:
                        write_event_cash(self.cash, open_positions=self.open_count())
                        self._last_events_emit = now
                    time.sleep(1)
                    self.reporter.maybe_report(self.cash, self.open_count())
                    continue

                # bandit / sandbox
                if now - last_learn_ping > 60:
                    try: intra_day_bandit(self.cfg)
                    except Exception: pass
                    try: sandbox_try(self.cfg)
                    except Exception: pass
                    last_learn_ping = now

                # day-end ops
                gm = time.gmtime()
                if gm.tm_hour == 23 and gm.tm_min >= 55:
                    try: day_end_grid(self.cfg)
                    except Exception: pass
                day_key = time.strftime("%Y-%m-%d", gm)
                if last_daily_retrain != day_key:
                    try: retrain_daily(self.cfg)
                    except Exception: pass
                    last_daily_retrain = day_key

                # watchlist sync
                self.wlm.update()
                self._sync_symbols(self.wlm.active())

                # -------- scan symbols --------
                for s, st in self.syms.items():
                    if s in self.cfg.get("ignore_symbols", []):
                        continue

                    px = float(get_price(s) or 0.0)
                    if px <= 0:
                        continue
                    st.last_px = px

                    snap = st.reg.snapshot()
                    st._last_reg = snap
                    reg = snap.get("regime", "UNKNOWN")

                    allow, (cbias, cconf, ctag), bonus_conf = allow_long(self.cfg.get("candles", {}), s)
                    st._last_guard = (cbias, cconf, ctag, bonus_conf)

                    if self.open_count() >= self.max_open_total:
                        continue
                    if self.symbol_open_count(s) >= self.max_per_symbol:
                        continue

                    enter_bonus = 1.15 if reg == "TREND" else (0.8 if reg == "CHOP" else 0.95)

                    # DIP
                    if allow:
                        fired, why, _ = _as3(st.signal_dip(px))
                        if fired and can_trade_now(self.cfg, st, "dip"):
                            _total_slot, free = slot_cash(self.syms, self.cash, self.alloc, "dip")
                            base_conf = max(0.0, min(1.0, 0.55 + (0.15 if reg == "TREND" else 0.0) + bonus_conf))
                            conf = adjust_confidence(self.cfg, s, "DIP", why or "", base_conf)

                            _off, tp_abs, sl_abs = st.levels(px)
                            ok_ev = True
                            if self.use_ev_filter:
                                ok_ev, _ = ev_ok(px, tp_abs, sl_abs, conf, self.fee_pct, self.slip_pct, self.ev_min)
                            if ok_ev:
                                spend = size_with_conf(self.entry_frac, self.sizing, "dip", free, conf) * enter_bonus
                                prop = self._brain_gate(s, "dip", st, px, spend, tp_abs, sl_abs, conf, reg)
                                if prop:
                                    qty = max(0.0, float(prop["qty"]))
                                    used = st.maybe_open("dip", px, qty * px, f"{why} | {cbias}({ctag})", reason_tag("DIP"), conf)
                                    if used > 0:
                                        self._brian2026_mark_open(s, "dip", px)
                                        try:
                                            write_trades_full_row(
                                                event="open", sym=s, slot="dip", side="BUY",
                                                price=px, qty=qty,
                                                avg=st.pos["dip"]["avg"] if st.pos.get("dip") else 0.0,
                                                pnl=0.0, regime=reg, confidence=conf,
                                                spread_bps=getattr(st, "spread_bps", 0.0),
                                                reason="open", enter_reason=str(why),
                                                tp_abs=tp_abs, sl_abs=sl_abs,
                                                vol_norm=getattr(st, "vol_norm", 0.0),
                                                alphas=getattr(st, "alphas", None)
                                            )
                                        except Exception: pass
                                        self._trade_hist.append((time.time(), s))
                                        self.reporter.add("open", "dip", s, 0.0)

                    # PRED
                    if allow:
                        fired, why, meta = _as3(st.signal_pred(px))
                        if fired and can_trade_now(self.cfg, st, "pred"):
                            _total, free = slot_cash(self.syms, self.cash, self.alloc, "pred")
                            prob = (meta or {}).get("prob", 0.7)
                            base_conf = min(1.0, 0.5 + 0.5 * float(prob) + (0.15 if reg == "TREND" else 0.0) + bonus_conf)
                            conf = adjust_confidence(self.cfg, s, "PRED", why or "", base_conf)

                            _off, tp_abs, sl_abs = st.levels(px)
                            ok_ev = True
                            if self.use_ev_filter:
                                ok_ev, _ = ev_ok(px, tp_abs, sl_abs, conf, self.fee_pct, self.slip_pct, self.ev_min)
                            if ok_ev:
                                spend = size_with_conf(self.entry_frac, self.sizing, "pred", free, conf) * enter_bonus
                                prop = self._brain_gate(s, "pred", st, px, spend, tp_abs, sl_abs, conf, reg)
                                if prop:
                                    qty = max(0.0, float(prop["qty"]))
                                    used = st.maybe_open("pred", px, qty * px, f"{why} | {cbias}({ctag})", reason_tag("PRED"), conf)
                                    if used > 0:
                                        self._brian2026_mark_open(s, "pred", px)
                                        try:
                                            write_trades_full_row(
                                                event="open", sym=s, slot="pred", side="BUY",
                                                price=px, qty=qty,
                                                avg=st.pos["pred"]["avg"] if st.pos.get("pred") else 0.0,
                                                pnl=0.0, regime=reg, confidence=conf,
                                                spread_bps=getattr(st, "spread_bps", 0.0),
                                                reason="open", enter_reason=str(why),
                                                tp_abs=tp_abs, sl_abs=sl_abs,
                                                vol_norm=getattr(st, "vol_norm", 0.0),
                                                alphas=getattr(st, "alphas", None)
                                            )
                                        except Exception: pass
                                        self._trade_hist.append((time.time(), s))
                                        self.reporter.add("open", "pred", s, 0.0)

                    # NEWS
                    nfired, nwhy, _ = _as3(st.signal_news())
                    if nfired and can_trade_now(self.cfg, st, "news") and allow:
                        try:
                            log_market_intel(kind="news", sym=s, msg=nwhy or "news_signal",
                                             payload={"price": px, "regime": reg})
                        except Exception:
                            pass

                        _tot, free = slot_cash(self.syms, self.cash, self.alloc, "news")
                        base_conf = 0.90
                        conf = adjust_confidence(self.cfg, s, "NEWS", nwhy or "", base_conf)

                        _off, tp_abs, sl_abs = st.levels(px)
                        ok_ev = True
                        if self.use_ev_filter:
                            ok_ev, _ = ev_ok(px, tp_abs, sl_abs, conf, self.fee_pct, self.slip_pct, self.ev_min)
                        if ok_ev:
                            spend = size_with_conf(self.entry_frac, self.sizing, "news", free, conf)
                            prop = self._brain_gate(s, "news", st, px, spend, tp_abs, sl_abs, conf, reg)
                            if prop:
                                qty = max(0.0, float(prop["qty"]))
                                used = st.maybe_open("news", px, qty * px, f"{nwhy} | {cbias}({ctag})", reason_tag("NEWS"), conf)
                                if used > 0:
                                    self._brian2026_mark_open(s, "news", px)
                                    try:
                                        write_trades_full_row(
                                            event="open", sym=s, slot="news", side="BUY",
                                            price=px, qty=qty,
                                            avg=st.pos["news"]["avg"] if st.pos.get("news") else 0.0,
                                            pnl=0.0, regime=reg, confidence=conf,
                                            spread_bps=getattr(st, "spread_bps", 0.0),
                                            reason="open", enter_reason=str(nwhy),
                                            tp_abs=tp_abs, sl_abs=sl_abs,
                                            vol_norm=getattr(st, "vol_norm", 0.0),
                                            alphas=getattr(st, "alphas", None)
                                        )
                                    except Exception: pass
                                    self._trade_hist.append((time.time(), s))
                                    self.reporter.add("open", "news", s, 0.0)

                    # ORDERBOOK
                    ofired, owhy, _ = _as3(st.signal_ob(self.cfg.get("orderbook", {})))
                    if ofired and can_trade_now(self.cfg, st, "ob") and allow:
                        _t, free = slot_cash(self.syms, self.cash, self.alloc, "ob")
                        base_conf = max(0.0, min(1.0, 0.65 + (0.15 if reg == "TREND" else 0.0) + bonus_conf))
                        conf = adjust_confidence(self.cfg, s, "OB", owhy or "", base_conf)

                        _off, tp_abs, sl_abs = st.levels(px)
                        ok_ev = True
                        if self.use_ev_filter:
                            ok_ev, _ = ev_ok(px, tp_abs, sl_abs, conf, self.fee_pct, self.slip_pct, self.ev_min)
                        if ok_ev:
                            spend = size_with_conf(self.entry_frac, self.sizing, "ob", free, conf) * enter_bonus
                            prop = self._brain_gate(s, "ob", st, px, spend, tp_abs, sl_abs, conf, reg)
                            if prop:
                                qty = max(0.0, float(prop["qty"]))
                                used = st.maybe_open("ob", px, qty * px, f"{owhy} | {cbias}({ctag})", reason_tag("ORDERBOOK"), conf)
                                if used > 0:
                                    self._brian2026_mark_open(s, "ob", px)
                                    try:
                                        write_trades_full_row(
                                            event="open", sym=s, slot="ob", side="BUY",
                                            price=px, qty=qty,
                                            avg=st.pos["ob"]["avg"] if st.pos.get("ob") else 0.0,
                                            pnl=0.0, regime=reg, confidence=conf,
                                            spread_bps=getattr(st, "spread_bps", 0.0),
                                            reason="open", enter_reason=str(owhy),
                                            tp_abs=tp_abs, sl_abs=sl_abs,
                                            vol_norm=getattr(st, "vol_norm", 0.0),
                                            alphas=getattr(st, "alphas", None)
                                        )
                                    except Exception: pass
                                    self._trade_hist.append((time.time(), s))
                                    self.reporter.add("open", "ob", s, 0.0)

                    # plugin stratejiler
                    for ps in self.plugins.run_for_symbol(s, st):
                        slot = ps.get("slot", "pred")
                        if slot not in ("dip", "pred", "news", "ob"):
                            slot = "pred"
                        if not allow or not can_trade_now(self.cfg, st, slot):
                            continue

                        _t2, free = slot_cash(self.syms, self.cash, self.alloc, slot)
                        base_conf = max(0.0, min(1.0, 0.5 + float(ps.get("confidence", 0.0))))
                        conf = adjust_confidence(self.cfg, s, f"PLUG:{slot.upper()}", ps.get("reason", ""), base_conf)

                        _off, tp_abs, sl_abs = st.levels(px)
                        ok_ev = True
                        if self.use_ev_filter:
                            ok_ev, _ = ev_ok(px, tp_abs, sl_abs, conf, self.fee_pct, self.slip_pct, self.ev_min)
                        if not ok_ev:
                            continue

                        spend = size_with_conf(self.entry_frac, self.sizing, slot, free, conf)
                        prop = self._brain_gate(s, slot, st, px, spend, tp_abs, sl_abs, conf, reg)
                        if prop:
                            qty = max(0.0, float(prop["qty"]))
                            used = st.maybe_open(
                                slot, px, qty * px,
                                f"[STRAT] {ps.get('reason', '')} | {cbias}({ctag})",
                                f"PLUG:{slot.upper()}",
                                conf,
                            )
                            if used > 0:
                                self._brian2026_mark_open(s, slot, px)
                                try:
                                    write_trades_full_row(
                                        event="open", sym=s, slot=slot, side="BUY",
                                        price=px, qty=qty,
                                        avg=st.pos[slot]["avg"] if st.pos.get(slot) else 0.0,
                                        pnl=0.0, regime=reg, confidence=conf,
                                        spread_bps=getattr(st, "spread_bps", 0.0),
                                        reason="open", enter_reason=str(ps.get("reason", "")),
                                        tp_abs=tp_abs, sl_abs=sl_abs,
                                        vol_norm=getattr(st, "vol_norm", 0.0),
                                        alphas=getattr(st, "alphas", None)
                                    )
                                except Exception: pass
                                self._trade_hist.append((time.time(), s))
                                self.reporter.add("open", slot, s, 0.0)

                # EXIT/DCA & risk
                for s, st in self.syms.items():
                    px = float(st.last_px or 0.0)
                    if px <= 0:
                        continue

                    ema9 = st._last_reg.get("ema9")
                    reg = st._last_reg.get("regime", "UNKNOWN")
                    cbias, cconf, _ctag, _bonus = getattr(st, "_last_guard", ("", 0.0, "", 0.0))

                    # bearish guard
                    if should_bearish_exit(self.cfg.get("candles", {}), s, ema9, cbias, cconf, px):
                        for slot in ["dip", "pred", "news", "ob"]:
                            if st.pos[slot] is None:
                                continue
                            closed, pnl, _ = st.maybe_close(slot, px)
                            if closed:
                                self._brian2026_mark_close(s, slot, px, pnl, "bearish_exit")
                                self.cash = round(self.cash + pnl, 2)
                                self.reporter.add("close", slot, s, pnl)
                                try:
                                    write_trades_full_row(
                                        event="close", sym=s, slot=slot, side="SELL",
                                        price=px, qty=0.0, avg=0.0,
                                        pnl=float(pnl), regime=reg, confidence=0.0,
                                        spread_bps=getattr(st, "spread_bps", 0.0),
                                        reason="bearish_exit", enter_reason="",
                                        tp_abs=0.0, sl_abs=0.0,
                                        vol_norm=getattr(st, "vol_norm", 0.0),
                                        alphas=getattr(st, "alphas", None)
                                    )
                                except Exception: pass
                                try: tg_send(f"💰 Kasa: {self.cash:.2f} USD")
                                except Exception: pass

                    for slot in ["dip", "pred", "news", "ob"]:
                        if st.pos[slot] is None:
                            continue

                        # DCA
                        _tt, free = slot_cash(self.syms, self.cash, self.alloc, slot)
                        used = st.maybe_dca(slot, px, free)
                        if used > 0:
                            try:
                                write_trades_full_row(
                                    event="dca", sym=s, slot=slot, side="BUY",
                                    price=px, qty=(used / px if px > 0 else 0.0),
                                    avg=st.pos[slot]["avg"] if st.pos.get(slot) else 0.0,
                                    pnl=0.0, regime=reg, confidence=0.0,
                                    spread_bps=getattr(st, "spread_bps", 0.0),
                                    reason="dca", enter_reason="auto_dca",
                                    tp_abs=0.0, sl_abs=0.0,
                                    vol_norm=getattr(st, "vol_norm", 0.0),
                                    alphas=getattr(st, "alphas", None)
                                )
                            except Exception:
                                pass

                        # EXIT
                        closed, pnl, reason = st.maybe_close(slot, px)
                        if closed:
                            self._brian2026_mark_close(s, slot, px, pnl, reason)
                            self.cash = round(self.cash + pnl, 2)
                            self.reporter.add("close", slot, s, pnl)
                            # error taxonomy (kayıp kapama ise etiketle)
                            try:
                                if self.etax and float(pnl) < 0:
                                    tag = self.etax.classify(
                                        symbol=s, slot=slot, regime=reg, pnl=float(pnl),
                                        px=float(px), avg=float(st.pos.get(slot,{}).get("avg",0.0)) if st.pos.get(slot) else 0.0,
                                        spread_bps=getattr(st, "spread_bps", 0.0),
                                        reason=str(reason)
                                    )
                                    log_event("loss_tag", symbol=s, slot=slot, tag=tag, pnl=float(pnl))
                            except Exception:
                                pass
                            try:
                                write_trades_full_row(
                                    event="close", sym=s, slot=slot, side="SELL",
                                    price=px, qty=0.0, avg=0.0,
                                    pnl=float(pnl), regime=reg, confidence=0.0,
                                    spread_bps=getattr(st, "spread_bps", 0.0),
                                    reason=str(reason), enter_reason="",
                                    tp_abs=0.0, sl_abs=0.0,
                                    vol_norm=getattr(st, "vol_norm", 0.0),
                                    alphas=getattr(st, "alphas", None)
                                )
                            except Exception: pass

                            cool = self.risk.on_realized(pnl)
                            if cool:
                                try:
                                    tg_send(f"🟥 Risk guard: günlük limit! {self.risk.cool_min}dk cooldown.")
                                except Exception:
                                    pass
                                log_event("risk_cooldown", pnl_today=self.risk.realized)
                            try: tg_send(f"💰 Kasa: {self.cash:.2f} USD")
                            except Exception: pass

                # persist/report
                save_state(self.cash, self.syms)
                self._write_state_snapshot()
                self.reporter.maybe_report(self.cash, self.wlm.open_count())

                if now - self._last_events_emit > 5:
                    write_event_cash(self.cash, open_positions=self.open_count())
                    self._last_events_emit = now

                time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                # L9: SelfHeal guard + raporlama
                try:
                    if getattr(self, "sh", None):
                        with self.sh.guard("loop"):
                            raise e
                except Exception:
                    pass
                try: report_exception("loop", e)
                except Exception: pass
                try: tg_send(f"[ERR] bot.loop: {e}")
                except Exception: pass
                # L13: exception hook
                try: l13_on_exception("loop", e, self.cfg)
                except Exception: pass
                # L14: exception hook
                try:
                    if _l14_on_exception is not None:
                        _l14_on_exception("loop", e, self.cfg)  # type: ignore
                except Exception:
                    pass
                # L60: exception hook
                try:
                    if _l60_on_exception is not None:
                        _l60_on_exception("loop", e, self.cfg)  # type: ignore
                except Exception:
                    pass
                time.sleep(2)


