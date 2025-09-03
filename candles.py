import ccxt
import time

# Binance borsası
exchange = ccxt.binance({
    "enableRateLimit": True
})


def fetch_ohlcv(symbol: str, timeframe: str = "1m", limit: int = 100):
    """
    Binance'ten OHLCV (mum verisi) çeker.
    Her mum: [timestamp, open, high, low, close, volume]
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return ohlcv
    except Exception as e:
        print(f"[HATA][fetch_ohlcv] {symbol} için veri alınamadı: {e}")
        return []


def candle_bias(klines):
    """
    Basit Candle Bias:
    - Son mum kapanış > açılış → bullish (+1)
    - Son mum kapanış < açılış → bearish (-1)
    - Eşitse → nötr (0)
    """
    if not klines or len(klines) == 0:
        return 0

    try:
        open_price = float(klines[-1][1])  # son mum açılış
        close_price = float(klines[-1][4])  # son mum kapanış

        if close_price > open_price:
            return 1
        elif close_price < open_price:
            return -1
        else:
            return 0
    except Exception as e:
        print(f"[HATA][candle_bias]: {e}")
        return 0


def avg_price(klines, count: int = 5):
    """
    Son X mumun ortalama kapanış fiyatını döndürür.
    """
    if not klines or len(klines) < count:
        return None
    try:
        closes = [float(k[4]) for k in klines[-count:]]
        return sum(closes) / len(closes)
    except Exception as e:
        print(f"[HATA][avg_price]: {e}")
        return None


if __name__ == "__main__":
    # Küçük test
    candles = fetch_ohlcv("ETH/USDT", "1m", 10)
    print("Son 10 mum:", candles)
    print("Bias:", candle_bias(candles))
    print("Son 5 ortalama:", avg_price(candles, 5))
