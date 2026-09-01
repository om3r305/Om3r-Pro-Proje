# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time

class HotApply:
    def __init__(self, bot, cfg_path: str = "config_live.json"):
        self.bot = bot
        self.cfg_path = cfg_path
        self._cfg_mtime = os.path.getmtime(cfg_path) if os.path.exists(cfg_path) else 0
        self._last_check = 0.0

    def _load_cfg(self):
        try:
            with open(self.cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def apply(self, new_cfg: dict):
        if not new_cfg:
            return
        b = self.bot

        # 1) frekans
        fc = new_cfg.get("freq_ctrl", {})
        b.min_gap = int(fc.get("min_sec_between_trades", b.min_gap))
        b.max_tph = int(fc.get("max_trades_per_hour", b.max_tph))

        # 2) portföy/limitler
        b.alloc = new_cfg.get("portfolio", b.alloc)
        b.entry_frac = new_cfg.get("entry_frac", b.entry_frac)
        b.sizing = new_cfg.get("sizing", b.sizing)
        b.max_open_total = int(new_cfg.get("max_total_open_positions", b.max_open_total))
        b.max_per_symbol = int(new_cfg.get("max_open_per_symbol", b.max_per_symbol))

        # 3) sembollere yeni cfg’yi yansıt
        for st in b.syms.values():
            st.cfg = new_cfg

        # 4) rapor aralığı (opsiyonel)
        rep = new_cfg.get("report", {})
        if rep:
            try:
                b.reporter.interval = int(rep.get("interval_sec", b.reporter.interval))
            except Exception:
                pass

        # 5) bot referansını güncelle
        b.cfg = new_cfg

    def tick(self, interval_sec: int = 5):
        now = time.time()
        if now - self._last_check < interval_sec:
            return
        self._last_check = now
        try:
            if not os.path.exists(self.cfg_path):
                return
            mt = os.path.getmtime(self.cfg_path)
            if mt != self._cfg_mtime:
                new_cfg = self._load_cfg()
                self._cfg_mtime = mt
                if new_cfg:
                    self.apply(new_cfg)
                    print("[HOT-APPLY] config_live.json değişti → canlıya uygulandı.")
        except Exception as e:
            print("[HOT-APPLY] hata:", e)
