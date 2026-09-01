"""Synthetic smoke/demo for Brian 2026. No exchange connection and no orders."""
from brian2026 import BrianEngine, MarketSnapshot


def main() -> None:
    brian = BrianEngine.from_json("brian2026/config.json")
    snap = MarketSnapshot(
        symbol="BTCUSDT",
        price=60000.0,
        regime="TREND",
        features={
            "ema_fast": 60120.0,
            "ema_slow": 59880.0,
            "ema_slope_pct": 0.18,
            "rsi": 61.0,
            "return_5": 0.22,
            "book_imbalance": 0.35,
            "wall_score": 0.15,
            "spread_bps": 2.0,
            "breakout_score": 0.40,
            "volume_z": 1.4,
            "acceleration": 0.12,
            "zscore": 0.8,
            "bb_position": 0.70,
            "atr_pct": 0.9
        },
    )
    d = brian.decide(snap, account={"daily_pnl_pct": 0.0, "drawdown_pct": 0.0, "open_positions": 0})
    print(d.to_dict())


if __name__ == "__main__":
    main()
