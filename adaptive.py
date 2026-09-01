# adaptive.py — Coins Monster Auto-Tuner (global + per-symbol)
import os, time

class AutoTuner:
    """
    Global (PF/WR/MaxDD) ve per-symbol (coin bazlı) oto ayar.
    - Global: entry_frac ölçeği, predictor.enter_prob, EV eşiği, min trade gap
    - Per-symbol: her coin için entry multiplier (0.6..1.4) ve loss-streak cooldown
    Log: logs/tune.log + telegram kısa bildirim
    """
    def __init__(self, cfg: dict, bot_ref):
        self.cfg = cfg
        self.bot = bot_ref
        os.makedirs("logs", exist_ok=True)
        self.tunepath = os.path.join("logs", "tune.log")

        # Global tuner
        c = cfg.get("auto_tune", {})
        self.enabled_global = bool(c.get("enabled", False))
        self.eval_minutes = int(c.get("eval_minutes", 30))
        self.lookback_closes = int(c.get("lookback_closes", 40))
        t = c.get("targets", {})
        self.pf_min = float(t.get("pf_min", 1.15))
        self.win_low = float(t.get("winrate_low", 0.38))
        self.win_high = float(t.get("winrate_high", 0.55))
        self.maxdd_floor = float(t.get("maxdd_floor_usd", -4.0))
        a = c.get("adjust", {})
        self.step_entry = float(a.get("entry_frac_step", 0.05))
        self.step_prob  = float(a.get("enter_prob_step", 0.02))
        self.step_ev    = float(a.get("ev_min_step", 0.0002))
        self.step_gap   = int(a.get("min_gap_step", 2))
        b = c.get("bounds", {})
        self.b_entry_min = float(b.get("entry_frac_min", 0.30))
        self.b_entry_max = float(b.get("entry_frac_max", 0.80))
        self.b_prob_min  = float(b.get("enter_prob_min", 0.60))
        self.b_prob_max  = float(b.get("enter_prob_max", 0.80))
        self.b_ev_min    = float(b.get("ev_min_min", 0.0005))
        self.b_ev_max    = float(b.get("ev_min_max", 0.0030))
        self.b_gap_min   = int(b.get("min_gap_min", 8))
        self.b_gap_max   = int(b.get("min_gap_max", 30))

        # Per-symbol tuner
        ps = cfg.get("per_symbol_tune", {})
        self.enabled_ps = bool(ps.get("enabled", True))
        self.ps_wr_good = float(ps.get("wr_good", 0.55))
        self.ps_wr_bad  = float(ps.get("wr_bad", 0.35))
        self.ps_pnl_bad = float(ps.get("pnl_bad_usd", -1.0))
        self.ps_step    = float(ps.get("step", 0.05))
        self.ps_min     = float(ps.get("mult_min", 0.60))
        self.ps_max     = float(ps.get("mult_max", 1.40))
        self.ps_loss_n  = int(ps.get("cooldown_after_n_losses", 3))
        self.ps_cool_m  = int(ps.get("cooldown_minutes", 60))

        self.last_tune = 0.0

    # --------------- utils ---------------
    def _log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            with open(self.tunepath, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
        if getattr(self.bot, "console", False):
            print("[AUTO]", msg)
        try:
            from telegram_utils import tg_send
            tg_send(f"🛠️ AutoTune: {msg}")
        except Exception:
            pass

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    # ---- GLOBAL MOVES ----
    def _scale_entry_fracs(self, up: bool):
        ef = dict(self.bot.entry_frac)
        step = self.step_entry if up else -self.step_entry
        new = {}
        for k, v in ef.items():
            nv = self._clamp(float(v) + step, self.b_entry_min, self.b_entry_max)
            new[k] = round(nv, 4)
        self.bot.entry_frac = new
        self.cfg["entry_frac"] = new
        self._log(f"entry_frac -> {new}")

    def _bump_enter_prob(self, up: bool):
        pr = dict(self.cfg.get("predictor", {}))
        cur = float(pr.get("enter_prob", 0.70))
        nv = self._clamp(cur + (self.step_prob if up else -self.step_prob),
                         self.b_prob_min, self.b_prob_max)
        pr["enter_prob"] = round(nv, 4)
        self.cfg["predictor"] = pr
        self._log(f"predictor.enter_prob -> {nv:.2f}")

    def _bump_ev_min(self, up: bool):
        bt = dict(self.cfg.get("backtest", {}))
        cur = float(bt.get("use_ev_filter", True) and bt.get("ev_min", 0.0015) or 0.0)
        nv = self._clamp(cur + (self.step_ev if up else -self.step_ev),
                         self.b_ev_min, self.b_ev_max)
        bt["use_ev_filter"] = True
        bt["ev_min"] = round(nv, 6)
        self.cfg["backtest"] = bt
        self.bot.use_ev_filter = True
        self.bot.ev_min = float(bt["ev_min"])
        self._log(f"EV ev_min -> {nv:.4f}")

    def _bump_min_gap(self, up: bool):
        fc = dict(self.cfg.get("freq_ctrl", {}))
        cur = int(fc.get("min_sec_between_trades", 20))
        nv = int(self._clamp(cur + (self.step_gap if up else -self.step_gap),
                             self.b_gap_min, self.b_gap_max))
        fc["min_sec_between_trades"] = nv
        self.cfg["freq_ctrl"] = fc
        self.bot.min_gap = nv
        self._log(f"freq_ctrl.min_sec_between_trades -> {nv}s")

    # ---- PER-SYMBOL MOVES ----
    def _ps_bump_mult(self, sym: str, up: bool):
        m = self.bot.entry_mult.get(sym, 1.0)
        nm = self._clamp(m + (self.ps_step if up else -self.ps_step), self.ps_min, self.ps_max)
        self.bot.entry_mult[sym] = round(nm, 4)
        self._log(f"{sym}: entry_mult -> {nm:.2f}")

    def _ps_cooldown(self, sym: str):
        until = time.time() + self.ps_cool_m * 60
        self.bot.sym_cool_until[sym] = until
        self._log(f"{sym}: cooldown {self.ps_cool_m} dk")

    # --------------- entrypoint ---------------
    def on_report(self, pf: float, winrate: float, maxdd_pos_usd: float,
                  closes_count: int, coin_stats: dict, loss_streaks: dict):
        """
        pf: Profit Factor
        winrate: 0..1
        maxdd_pos_usd: pencere içi max drawdown (pozitif sayı olarak)
        coin_stats: {SYM: {'p':pnl_usd, 'w':wins, 'n':trades, 'wr':0..1}}
        loss_streaks: {SYM: consecutive_losses_int}
        """
        now = time.time()
        # GLOBAL
        if self.enabled_global:
            if now - self.last_tune >= self.eval_minutes * 60 and closes_count >= max(10, int(self.lookback_closes * 0.5)):
                actions = []
                if maxdd_pos_usd >= abs(self.maxdd_floor):
                    # DD kötü → frekansı düşür, boyutu küçült, kaliteyi artır
                    self._bump_min_gap(True)
                    self._scale_entry_fracs(False)
                    self._bump_ev_min(True)
                    actions.append("MaxDD guard")
                elif pf < self.pf_min:
                    self._bump_ev_min(True)
                    self._bump_enter_prob(True)
                    self._scale_entry_fracs(False)
                    actions.append("PF improve")
                elif pf >= self.pf_min and winrate >= self.win_high:
                    self._scale_entry_fracs(True)
                    self._bump_enter_prob(False)
                    self._bump_ev_min(False)
                    actions.append("Boost")
                else:
                    # nötr: hafif kalite
                    self._bump_ev_min(True)
                    actions.append("Neutral")

                if actions:
                    self._log(f"GLOBAL pf={pf:.2f} wr={winrate:.2f} dd={maxdd_pos_usd:.2f} → {', '.join(actions)}")
                self.last_tune = now

        # PER-SYMBOL
        if self.enabled_ps and coin_stats:
            for sym, st in coin_stats.items():
                n = int(st.get("n", 0))
                if n < 3:
                    continue
                wr = float(st.get("wr", 0.0))
                pnl = float(st.get("p", 0.0))
                # kötü: winrate düşük veya pnl kötü
                if wr <= self.ps_wr_bad or pnl <= self.ps_pnl_bad:
                    self._ps_bump_mult(sym, up=False)
                elif wr >= self.ps_wr_good and pnl > 0:
                    self._ps_bump_mult(sym, up=True)

                # Loss streak cooldown
                ls = int(loss_streaks.get(sym, 0))
                if ls >= self.ps_loss_n:
                    self._ps_cooldown(sym)
