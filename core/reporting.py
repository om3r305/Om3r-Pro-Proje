# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import List, Dict, Any

# TG gönderimi (projede var)
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(_msg: str, parse_mode="HTML"):  # fallback
        pass


class Reporter:
    """
    Periyodik özet raporları TG'ye yollar.
    add() ile olayları (open/close) biriktir; maybe_report() belli aralıkla gönderir.
    """

    def __init__(self, cfg: dict):
        cfg = cfg or {}
        # kaç saniyede bir rapor?
        self.interval = int(cfg.get("interval_sec", 7200))  # 2 saat default
        # rapora dahil edilecek maksimum kapanış sayısı (listeyi taşırmasın)
        self.max_lines = int(cfg.get("max_lines", 18))

        # İlk rapor hemen gitmesin diye başlangıç zamanını baz alıyoruz
        self._last = time.time()           # <-- ESKİ: 0.0 (hemen rapor atıyordu)
        self._events: List[Dict[str, Any]] = []

    # -------- public api --------
    def add(self, kind: str, slot: str, sym: str, pnl: float = 0.0):
        """kind: 'open' | 'close'"""
        self._events.append({
            "ts": time.time(),
            "kind": str(kind),
            "slot": str(slot),
            "sym":  str(sym),
            "pnl":  float(pnl),
        })

    def maybe_report(self, cash_ref: float, open_count: int):
        now = time.time()
        if now - self._last < self.interval:
            return
        self._last = now

        cutoff = now - self.interval
        # pencere içindeki olaylar
        window: List[Dict[str, Any]] = [e for e in self._events if e["ts"] >= cutoff]
        # pencere dışındaki olayları temizle
        self._events = window

        closes = [e for e in window if e["kind"] == "close"]
        if not closes:
            tg_send(
                f"📊 Rapor ({int(self.interval/60)} dk)\n"
                f"• İşlem yok.\n"
                f"🔓 Açık pozisyon: {open_count}\n"
                f"💰 Kasa: {cash_ref:.2f} USD"
            , parse_mode="HTML")
            return

        # ----- temel metrikler -----
        total_pnl = sum(e["pnl"] for e in closes)
        n = len(closes)
        wins = sum(1 for e in closes if e["pnl"] > 0)
        wr = (wins / n * 100.0) if n > 0 else 0.0

        gains = sum(e["pnl"] for e in closes if e["pnl"] > 0)
        losses = sum(-e["pnl"] for e in closes if e["pnl"] < 0)
        pf = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

        # MaxDD (rapor penceresi)
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for e in sorted(closes, key=lambda x: x["ts"]):
            cum += e["pnl"]
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        # ----- coin bazında -----
        per_coin: Dict[str, Dict[str, float]] = {}
        for e in closes:
            s = e["sym"]
            d = per_coin.setdefault(s, {"p": 0.0, "n": 0.0, "w": 0.0})
            d["p"] += e["pnl"]; d["n"] += 1; d["w"] += 1 if e["pnl"] > 0 else 0

        # en iyi / en kötü
        best = max(per_coin.items(), key=lambda kv: kv[1]["p"]) if per_coin else None
        worst = min(per_coin.items(), key=lambda kv: kv[1]["p"]) if per_coin else None

        # ----- slot bazında -----
        per_slot: Dict[str, Dict[str, float]] = {}
        for e in window:  # open+close
            sl = e["slot"]
            d = per_slot.setdefault(sl, {"p": 0.0, "n": 0.0, "w": 0.0})
            if e["kind"] == "close":
                d["p"] += e["pnl"]; d["n"] += 1; d["w"] += 1 if e["pnl"] > 0 else 0

        # slot açılış dağılımı
        opened: Dict[str, int] = {}
        for e in window:
            if e["kind"] == "open":
                opened[e["slot"]] = opened.get(e["slot"], 0) + 1
        opened_total = sum(opened.values()) or 1

        def pct(sl: str) -> int:
            return int(100 * opened.get(sl, 0) / opened_total)

        # ----- render helpers -----
        def bar(v: float) -> str:
            # küçük görsel skor çubuğu
            if v > 10: return "🟩🟩🟩🟩"
            if v > 5:  return "🟩🟩"
            if v > 0:  return "🟨"
            if v < -5: return "🟥🟥"
            if v < 0:  return "🟥"
            return "▪️"

        slot_name = {
            "news": "🚨 NEWS", "pred": "🔮 PRED", "dip": "🟣 DIP", "ob": "🧱 OB"
        }

        # coin satırları
        coin_lines: List[str] = []
        for sym, data in sorted(per_coin.items(), key=lambda kv: kv[1]["p"], reverse=True)[:self.max_lines]:
            tag = "🟢" if data["p"] > 0 else ("🟡" if abs(data["p"]) < 1e-9 else "🔴")
            coin_lines.append(
                f"• {tag} {sym:<8} | {int(data['n'])} işlem | "
                f"{int(data['w'])}/{int(data['n'])} win | {data['p']:+.2f} {bar(data['p'])}"
            )

        # slot satırları
        slot_lines: List[str] = []
        for sl, data in per_slot.items():
            if int(data["n"]) <= 0:
                continue
            nm = slot_name.get(sl, sl.upper())
            slot_lines.append(
                f"• {nm:<10} | {int(data['n'])} işlem | ✅ {int(data['w'])}/{int(data['n'])} | {data['p']:+.2f}"
            )

        # başlık
        pf_str = "INF" if pf == float("inf") else f"{pf:.2f}"
        head = (f"📊 Rapor ({int(self.interval/60)} dk)\n"
                f"🧮 İşlem: {n} | 💵 Kâr: {total_pnl:+.2f} USD\n"
                f"🟢 WinRate: {wr:.1f}% | 📈 PF: {pf_str} | 📉 MaxDD: {-max_dd:.2f} USD\n")

        best_line = ("🏆 En iyi: "
                     f"{best[0]} • {int(per_coin[best[0]]['w'])}/{int(per_coin[best[0]]['n'])} "
                     f"| {per_coin[best[0]]['p']:+.2f}") if best else "🏆 En iyi: -"
        worst_line = ("⚠️ En kötü: "
                      f"{worst[0]} • {int(per_coin[worst[0]]['w'])}/{int(per_coin[worst[0]]['n'])} "
                      f"| {per_coin[worst[0]]['p']:+.2f}") if worst else "⚠️ En kötü: -"

        slot_dist = (f"📈 Açılış dağılımı → NEWS 🔵 {pct('news')}% | "
                     f"PRED 🟢 {pct('pred')}% | DIP 🟣 {pct('dip')}% | OB 🟡 {pct('ob')}%")

        body = (
            "— Coin bazında —\n" + ("\n".join(coin_lines) if coin_lines else "• veri yok") +
            "\n\n— Slot bazında —\n" + ("\n".join(slot_lines) if slot_lines else "• veri yok") +
            f"\n\n{slot_dist}\n\n"
            f"🔓 Açık pozisyon: {open_count} | 💰 Kasa: {cash_ref:.2f} USD"
        )

        # parse_mode kullanmıyoruz → Telegram entity hatası olmaz
        tg_send(head + best_line + "\n" + worst_line + "\n\n" + body, parse_mode="HTML")
