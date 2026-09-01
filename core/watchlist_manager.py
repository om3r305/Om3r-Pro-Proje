# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import json, time, random

# Proje kökü (core klasörünün bir üstü)
ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "runtime_state.json"

DEFAULT_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","ADAUSDT","AVAXUSDT","DOGEUSDT"
]

class WatchListManager:
    """
    Basit watchlist yöneticisi:
      - cfg['watchlist'] altında:
          { "symbols": [...],        # statik liste
            "max_active": 7,         # aynı anda takip edilecek sembol sayısı
            "exclude": ["BUSDUSDT"], # hariç tutulacaklar
            "rotate_sec": 120        # bu kadar saniyede bir listeyi döndür
          }
      - cfg['symbols'] varsa yedek olarak onu kullanır.
      - open_count(): runtime_state.json'dan açık pozisyon sayısını okur (bot değişikliği gerekmez)
      - Opsiyonel: note_open/note_close ile içeriden sayaç tutabilir.
    """

    def __init__(self, cfg: dict):
        wl = cfg.get("watchlist", {}) if isinstance(cfg, dict) else {}
        # Kaynak liste
        symbols = (
            wl.get("symbols")
            or cfg.get("symbols")
            or DEFAULT_SYMBOLS
        )
        symbols = [s.strip().upper() for s in symbols if isinstance(s, str)]
        # Hariç tutulanlar
        exclude = set(x.strip().upper() for x in wl.get("exclude", []) if isinstance(x, str))

        # Son hal: exclude çıkar
        self._all: List[str] = [s for s in symbols if s not in exclude]

        # Aktif pencere
        self._max_active: int = int(wl.get("max_active", max(1, min(10, len(self._all) or 1))))
        self._rotate_sec: int = int(wl.get("rotate_sec", 0))  # 0 => rotate yok
        self._last_rotate: float = 0.0

        # Mevcut aktif liste
        self._active: List[str] = self._pick_initial()

        # Opsiyonel internal sayaç (bot isterse kullanır)
        self._open_counts: Dict[str, int] = {}  # sym -> kaç slot açık

    # ------- iç yardımcılar -------
    def _pick_initial(self) -> List[str]:
        if not self._all:
            return []
        if self._max_active >= len(self._all):
            return list(self._all)
        # deterministic ama karışık başlangıç:
        pool = list(self._all)
        random.shuffle(pool)
        return pool[: self._max_active]

    def _rotate_if_needed(self):
        if self._rotate_sec <= 0:
            return
        now = time.time()
        if now - self._last_rotate < self._rotate_sec:
            return
        self._last_rotate = now

        if len(self._all) <= self._max_active:
            # hepsi zaten aktif
            self._active = list(self._all)
            return

        # basit bir “kaydırmalı” rotasyon
        try:
            start_idx = (self._all.index(self._active[0]) + self._max_active) % len(self._all)
        except Exception:
            start_idx = 0
        new_active = []
        i = start_idx
        while len(new_active) < self._max_active and self._all:
            new_active.append(self._all[i % len(self._all)])
            i += 1
        self._active = new_active

    # ------- public API -------
    def update(self):
        """Watchlist'i (gerekirse) döndürür/yeniler."""
        self._rotate_if_needed()

    def active(self) -> List[str]:
        """Şu an takip edilen aktif semboller."""
        return list(self._active)

    # ----- açık pozisyon sayısı -----
    def open_count(self) -> int:
        """
        Kaç tane açık pozisyon olduğunu döndürür.
        1) Öncelik: iç sayaç (note_open/note_close ile beslenirse)
        2) Aksi halde: runtime_state.json dosyasından okur (bot değişikliği gerekmez)
        """
        # 1) İç sayaç kullanılmakta ise:
        if self._open_counts:
            return sum(1 for v in self._open_counts.values() if v and v > 0)

        # 2) Dosyadan say (fallback)
        try:
            if not STATE_PATH.exists():
                return 0
            j = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            pos = j.get("positions", {})
            cnt = 0
            for _sym, slots in pos.items():
                # slots: {"dip": {...} | None, "pred": {...} | None, ...}
                if any(slots.get(k) for k in slots):
                    cnt += 1
            return cnt
        except Exception:
            return 0

    # ----- opsiyonel: bot'tan besleme -----
    def note_open(self, symbol: str):
        """Bot bir pozisyon açtığında çağırılabilir (opsiyonel hızlandırma)."""
        s = symbol.upper()
        self._open_counts[s] = self._open_counts.get(s, 0) + 1

    def note_close(self, symbol: str):
        """Bot bir pozisyon kapattığında çağırılabilir (opsiyonel hızlandırma)."""
        s = symbol.upper()
        self._open_counts[s] = max(0, self._open_counts.get(s, 0) - 1)

    def set_active(self, symbols: List[str]):
        """İstersen aktif listeyi dışarıdan sabitle."""
        self._active = [s.strip().upper() for s in symbols if isinstance(s, str)]
