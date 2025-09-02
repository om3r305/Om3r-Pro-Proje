# optimizer.py
import pandas as pd

class Optimizer:
    """
    Trade loglarını analiz edip parametre önerir.
    """

    def __init__(self):
        self.history = []

    def load_trades(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path)
            self.history = df.to_dict("records")
        except Exception:
            self.history = []

    def suggest_params(self, coin: str):
        """
        Coin bazlı en iyi parametreleri bulmaya çalış.
        """
        df = pd.DataFrame(self.history)
        if df.empty or "symbol" not in df.columns:
            return {"buy_offset": 1.0, "tp": 4.0, "sl": -6.0}

        cdf = df[df["symbol"] == coin]
        if cdf.empty:
            return {"buy_offset": 1.0, "tp": 4.0, "sl": -6.0}

        avg_pnl = cdf.groupby("buy_offset")["pnl"].mean()
        if avg_pnl.empty:
            return {"buy_offset": 1.0, "tp": 4.0, "sl": -6.0}

        best_offset = avg_pnl.idxmax()
        # şimdilik tp/sl sabit, geliştirilebilir
        return {"buy_offset": best_offset, "tp": 4.0, "sl": -6.0}
