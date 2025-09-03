import math
import numpy as np
import pandas as pd

# ==============================
# Candle Bias Hesaplama
# ==============================
def candle_bias(symbol: str, config: dict = None):
    """
    Verilen sembol için candle bias hesaplar.
    Config opsiyoneldir, verilmezse default ayarlar kullanılır.
    """

    # Varsayılan config
    if config is None:
        config = {
            "bias_len": 20,   # Kaç mum ortalaması alınacak
            "ema_len": 50     # EMA uzunluğu
        }

    # Burada normalde ccxt veya başka data kaynağından fiyatlar çekilir
    # Şimdilik placeholder yapıyoruz
    closes = np.random.random(config["bias_len"]) * 100  # random kapanış fiyatları
    df = pd.DataFrame({"close": closes})

    # EMA hesaplama
    df["ema"] = df["close"].ewm(span=config["ema_len"], adjust=False).mean()

    # Bias mantığı: kapanış EMA'nın üstünde mi altında mı?
    if df["close"].iloc[-1] > df["ema"].iloc[-1]:
        bias = "bullish"
    else:
        bias = "bearish"

    # Geri dönüş formatı fast_trader.py ile uyumlu
    return bias, config, f"{symbol}_bias"
