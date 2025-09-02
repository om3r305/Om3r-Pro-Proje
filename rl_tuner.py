# rl_tuner.py

class RLTuner:
    """
    Basit RL benzeri tuner: art arda kazan/zarar durumuna göre parametre oynar.
    """

    def __init__(self, base_offset=1.0, tp=4.0, sl=-6.0):
        self.base_offset = base_offset
        self.tp = tp
        self.sl = sl
        self.win_streak = 0
        self.loss_streak = 0

    def update(self, result: str):
        """
        result: "WIN" veya "LOSS"
        """
        if result == "WIN":
            self.win_streak += 1
            self.loss_streak = 0
        elif result == "LOSS":
            self.loss_streak += 1
            self.win_streak = 0

        # ayarlamalar
        if self.win_streak >= 5:
            self.base_offset *= 1.1  # daha geniş al
            self.tp *= 1.1
            self.win_streak = 0
        elif self.loss_streak >= 3:
            self.base_offset *= 0.9  # daralt
            self.tp *= 0.9
            self.loss_streak = 0

        return {
            "buy_offset": round(self.base_offset, 4),
            "tp": round(self.tp, 4),
            "sl": round(self.sl, 4),
        }
3
