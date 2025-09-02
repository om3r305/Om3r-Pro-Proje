# dip_tracker.py

class DipTracker:
    """
    Basit dip izleyici:
    - Satıştan sonra aynı dipten hemen almayı engeller (kilit).
    - SON alış dipinden daha DÜŞÜK yeni yerel dip oluşunca kilit otomatik açılır.
    """

    def __init__(self,
                 require_new_dip_after_start: bool = False,
                 reset_dip_after_sell: bool = True,
                 window_sec: int | None = None):
        self.require_new_after_start = bool(require_new_dip_after_start)
        self.reset_dip_after_sell = bool(reset_dip_after_sell)
        self.window_sec = window_sec

        self.current_dip: float | None = None
        self.new_dip_ready: bool = False
        self.wait_new_dip: bool = self.require_new_after_start
        self._last_buy_dip: float | None = None
        self._since_sell_min: float | None = None

        self._tol = 1e-6

    def update(self, price: float):
        p = float(price)

        if self.wait_new_dip:
            if self._since_sell_min is None or p < self._since_sell_min - self._tol:
                self._since_sell_min = p
                self.current_dip = p
                self.new_dip_ready = True

            if (self._last_buy_dip is None) or (self._since_sell_min < self._last_buy_dip - self._tol):
                self.wait_new_dip = False  # kilit aç
            return self.current_dip

        if self.current_dip is None:
            self.current_dip = p
            self.new_dip_ready = True
        elif p < self.current_dip - self._tol:
            self.current_dip = p
            self.new_dip_ready = True

        return self.current_dip

    def can_buy(self) -> bool:
        return (not self.wait_new_dip) and self.new_dip_ready and (self.current_dip is not None)

    def consume_new_flag(self):
        self.new_dip_ready = False

    def record_buy_dip(self, dip: float | None):
        self._last_buy_dip = float(dip) if dip is not None else None

    def on_sell(self):
        if self.reset_dip_after_sell:
            self.wait_new_dip = True
            self.new_dip_ready = False
            self._since_sell_min = None
        else:
            self.wait_new_dip = False
            self._since_sell_min = None
