# strategy_manager.py
import numpy as np

class StrategyManager:
    """
    Basit strateji seçici.
    Modlar:
      - dip_offset (default)
      - breakout
      - mean_reversion
    """

    def __init__(self):
        self.current_mode = "dip_offset"

    def choose(self, prices: list[float]) -> str:
        """
        Basit sinyal ile strateji seçimi.
        - ATR düşükse breakout
        - Trend kuvvetliyse dip_offset
        - Yan piyasa ise mean_reversion
        """
        if len(prices) < 20:
            return self.current_mode

        arr = np.array(prices)
        atr = np.std(arr[-20:]) / np.mean(arr[-20:]) * 100
        change = (arr[-1] - arr[0]) / arr[0] * 100

        if atr < 0.5:   # çok sakin → breakout dene
            self.current_mode = "breakout"
        elif abs(change) > 2:  # güçlü trend → dip_offset
            self.current_mode = "dip_offset"
        else:
            self.current_mode = "mean_reversion"

        return self.current_mode
