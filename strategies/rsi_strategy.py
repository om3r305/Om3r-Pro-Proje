# strategies/rsi_strategy.py
# Basit RSI stratejisi — alım: RSI<30, satım: RSI>70

class RSIStrategy:
    NAME = "RSI_Basic"

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.period = int(self.cfg.get("period", 14))
        self.overbought = float(self.cfg.get("overbought", 70))
        self.oversold = float(self.cfg.get("oversold", 30))
        self.values = []

    def update(self, price: float):
        self.values.append(price)
        if len(self.values) > self.period * 3:
            self.values.pop(0)

    def signal(self) -> dict:
        if len(self.values) < self.period + 1:
            return {"buy": False, "sell": False, "info": "warmup"}

        gains, losses = [], []
        for i in range(1, self.period + 1):
            diff = self.values[-i] - self.values[-i - 1]
            (gains if diff > 0 else losses).append(abs(diff))

        avg_gain = sum(gains) / self.period if gains else 0.0
        avg_loss = sum(losses) / self.period if losses else 1e-9
        rs = avg_gain / avg_loss if avg_loss > 0 else 0.0
        rsi = 100 - (100 / (1 + rs))

        return {
            "buy": rsi <= self.oversold,
            "sell": rsi >= self.overbought,
            "rsi": round(rsi, 2),
            "info": f"RSI={rsi:.2f}"
        }
