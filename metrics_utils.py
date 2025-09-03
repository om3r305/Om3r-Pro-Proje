# metrics_utils.py
# Basit performans metrikleri + PnL eğrisi çizimi

from typing import List, Tuple, Optional
import os

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def equity_curve_from_trades(trades: List[dict], start_cash: float) -> List[Tuple[float, float]]:
    """
    trades: [{ 'ts':close_ts or open_ts, 'pnl':float }]
    döner: list[(ts, equity_usd)]
    """
    eq = start_cash
    curve = []
    for t in sorted(trades, key=lambda x: x.get("ts", 0.0)):
        eq += float(t.get("pnl", 0.0))
        curve.append((float(t.get("ts", 0.0)), eq))
    return curve


def max_drawdown(curve: List[Tuple[float, float]]) -> Tuple[float, float]:
    """(abs_max_dd_usd, max_dd_pct)"""
    if not curve:
        return 0.0, 0.0
    peak = curve[0][1]
    max_dd = 0.0
    for _, v in curve:
        peak = max(peak, v)
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    start_eq = curve[0][1]
    mdp = (max_dd / start_eq * 100.0) if start_eq > 0 else 0.0
    return round(max_dd, 4), round(mdp, 2)


def profit_factor(trades: List[dict]) -> float:
    gains = sum(float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) > 0)
    losses = sum(-float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) < 0)
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def win_rate(trades: List[dict]) -> float:
    n = len(trades)
    if n == 0:
        return 0.0
    w = sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0)
    return 100.0 * w / n


def avg_win_loss(trades: List[dict]) -> Tuple[float, float]:
    wins = [float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) > 0]
    losses = [float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) < 0]
    aw = sum(wins) / len(wins) if wins else 0.0
    al = sum(losses) / len(losses) if losses else 0.0
    return aw, al  # al negatif döner


def save_equity_png(curve: List[Tuple[float, float]], out_path: str) -> Optional[str]:
    if not curve:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not _HAS_MPL:
        # matplotlib yoksa sessizce geç
        try:
            with open(out_path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
                for ts, eq in curve:
                    f.write(f"{ts},{eq}\n")
            return out_path.replace(".png", ".txt")
        except Exception:
            return None

    xs = [x for x, _ in curve]
    ys = [y for _, y in curve]
    plt.figure(figsize=(8, 3))
    plt.plot(xs, ys)
    plt.title("Equity Curve")
    plt.xlabel("t")
    plt.ylabel("USD")
    plt.tight_layout()
    try:
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception:
        plt.close()
        return None
