# backtest.py  (v5)
import argparse, csv, json, os, time
from pathlib import Path
from typing import List, Dict, Tuple

from metrics_utils import (
    equity_curve_from_trades, max_drawdown,
    profit_factor, win_rate, avg_win_loss, save_equity_png
)


# İsteğe bağlı: DipTracker kullanmadan basit bir DIP simülasyonu yapacağız.
# (Mevcut projendeki DipTracker ile birebir aynısını kullanmıyoruz ki
#  CSV ile offline çalışabilsin. Mantık: lokal dip (rolling low)+offset aşılırsa giriş.)
def simulate_dip_from_csv(rows: List[Dict[str, float]], offset_abs: float, tp_abs: float, sl_abs: float,
                          stake_usd: float, fee_pct: float = 0.001):
    """
    rows: [{'ts':int,'open':float,'high':float,'low':float,'close':float}]
    offset_abs: giriş tetikleyici (dip+offset)
    tp_abs: take profit mutlak
    sl_abs: stop-loss mutlak (negatif beklenir; ör: -6.0)
    stake_usd: her işlemde kullanılan USD
    """
    pos_open = None  # {'entry':float,'qty':float,'ts':int, 'dip':float}
    trades = []
    cash = 0.0  # net realize PnL toplamı (simülasyon içi)
    last_dip = None

    # rolling dip: son 90 barın en düşüğü (1m için ~1.5 saat)
    WIN = 90
    lows = []

    for r in rows:
        ts = int(r['ts'])
        o, h, l, c = float(r['open']), float(r['high']), float(r['low']), float(r['close'])
        lows.append(l)
        if len(lows) > WIN:
            lows.pop(0)

        dip = min(lows) if lows else l
        last_dip = dip

        # açık pozisyon yoksa giriş koşulu: fiyat (close) >= dip+offset_abs
        if pos_open is None:
            need = dip + offset_abs
            if c >= need:
                entry_px = c * (1.0 + fee_pct)  # taker + minimal slip
                qty = stake_usd / entry_px if entry_px > 0 else 0.0
                if qty > 0:
                    pos_open = {'entry': entry_px, 'qty': qty, 'ts': ts, 'dip': dip}
            continue

        # pozisyon varsa tp/sl kontrolü
        entry = pos_open['entry'];
        qty = pos_open['qty']
        tp_px = entry + tp_abs
        sl_px = entry + sl_abs  # sl_abs negatif beklenir
        sell_px = c * (1.0 - fee_pct)

        sell = None
        if sell_px >= tp_px:
            sell = ('TP', sell_px)
        elif sell_px <= sl_px:
            sell = ('SL', sell_px)

        if sell:
            reason, px = sell
            pnl = (px - entry) * qty
            trades.append({'ts': ts, 'pnl': pnl, 'reason': reason, 'entry': entry, 'exit': px})
            cash += pnl
            pos_open = None

    # gün sonu pozisyon varsa, kapatmıyoruz -> flat kalmak istersen burada realize edebilirsin
    return trades, cash, last_dip


def load_csv_series(path: Path) -> List[Dict[str, float]]:
    """
    Beklenen kolon isimleri (esnek):
      - ts/open/high/low/close  (veya)  open_time/open/high/low/close
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            def fnum(x):
                try:
                    return float(x)
                except:
                    return 0.0

            ts = int(row.get("ts") or row.get("open_time") or row.get("time") or 0)
            out.append({
                "ts": ts,
                "open": fnum(row.get("open")),
                "high": fnum(row.get("high")),
                "low": fnum(row.get("low")),
                "close": fnum(row.get("close")),
            })
    return out


def read_trade_log(log_path: Path, symbols: List[str]) -> Dict[str, List[dict]]:
    by_sym = {s: [] for s in symbols}
    if not log_path.exists():
        return by_sym
    with open(log_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            s = r.get("symbol") or r.get("sym") or ""
            if s in by_sym and (r.get("kind") == "close" or r.get("event") == "close"):
                try:
                    pnl = float(r.get("pnl", "0") or 0.0)
                except:
                    continue
                ts = float(r.get("ts") or r.get("close_ts") or time.time())
                by_sym[s].append({"ts": ts, "pnl": pnl})
    return by_sym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True, help="ör: BTCUSDT ETHUSDT SOLUSDT")
    ap.add_argument("--cash", type=float, default=500.0)
    ap.add_argument("--config", type=str, default="config_live.json")
    ap.add_argument("--profile", type=str, default="", help="opsiyonel etiket")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    interval = (cfg.get("general", {}) or {}).get("interval", "1m")
    data_dir = Path((cfg.get("general", {}) or {}).get("data_dir", "data"))
    os.makedirs("logs", exist_ok=True)

    # Önce canlı bot logu varsa ondan rapor çıkar
    log_file = Path("logs/trades.csv")
    sym_list = [s.strip().upper() for s in args.symbols]
    by_sym_log = read_trade_log(log_file, sym_list)
    any_log = any(len(v) > 0 for v in by_sym_log.values())

    print(f">>> BACKTEST v5 (metrics + PnL, profile='{args.profile or 'none'}') <<<")
    print(os.path.basename(args.config))

    header = f"{'SYMBOL':<8} {'TRADES':>7} {'PF':>6} {'WIN%':>7} {'AVGWIN':>10} {'AVGLOS':>10} {'MAXDD':>10} {'PNL(USD)':>10} {'FINAL_CASH':>12}"
    print(header)

    total_pnl = 0.0
    for s in sym_list:
        trades = []
        eq_curve = []
        if any_log and by_sym_log.get(s):
            # log'tan oku
            trades = by_sym_log[s]
            eq_curve = equity_curve_from_trades(trades, args.cash)
            pnl = sum(t["pnl"] for t in trades)
            final_cash = args.cash + pnl
        else:
            # CSV'den DIP simüle et
            p = data_dir / f"{s}_{interval}.csv"
            if not p.exists():
                print(
                    f"{s:<8} {'-':>7} {'-':>6} {'-':>7} {'-':>10} {'-':>10} {'-':>10} {0.0:>10.2f} {args.cash:>12.2f}  (veri yok)")
                continue
            rows = load_csv_series(p)
            dr = cfg.get("dip_rules", {}) or {}
            offset_abs = float((dr.get("abs", {}) or {}).get("offset", 1.0))
            tp_abs = float((dr.get("abs", {}) or {}).get("tp", 6.0))
            sl_abs = float((dr.get("abs", {}) or {}).get("sl", -4.0))
            fee = float((cfg.get("fees", {}) or {}).get("taker_pct", 0.10)) / 100.0

            tlist, pnl_sum, _ = simulate_dip_from_csv(
                rows, offset_abs=offset_abs, tp_abs=tp_abs, sl_abs=sl_abs,
                stake_usd=min(args.cash, args.cash * 0.5), fee_pct=fee
            )
            trades = tlist
            eq_curve = equity_curve_from_trades(trades, args.cash)
            pnl = pnl_sum
            final_cash = args.cash + pnl

        pf = profit_factor(trades)
        wr = win_rate(trades)
        aw, al = avg_win_loss(trades)
        mdd_abs, mdd_pct = max_drawdown(eq_curve)
        total_pnl += pnl

        # grafiği kaydet
        out_png = save_equity_png(eq_curve, f"logs/equity_{s}.png")
        # trades.csv'ye de özet düşelim
        try:
            csv_path = Path("logs") / f"summary_{s}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                wrt = csv.writer(f)
                wrt.writerow(["ts", "equity"])
                for ts, eq in eq_curve:
                    wrt.writerow([ts, eq])
        except Exception:
            pass

        def fmt_pf(x):
            if x == float("inf"):
                return "INF"
            return f"{x:.2f}"

        print(
            f"{s:<8} {len(trades):>7} {fmt_pf(pf):>6} {wr:>7.1f} {aw:>10.2f} {al:>10.2f} {(-mdd_abs):>10.2f} {pnl:>10.2f} {final_cash:>12.2f}")

    print(f"\nTOTAL PNL: {total_pnl:+.2f} USD")
    print("Not: logs/ klasörüne equity PNG/CSV dosyaları bırakıldı.")


if __name__ == "__main__":
    main()
