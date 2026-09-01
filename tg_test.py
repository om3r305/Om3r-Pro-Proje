# tg_test.py — Telegram bağlantı testi
from telegram_utils import tg_setup, tg_send
from dotenv import load_dotenv
from pathlib import Path

# .env dosyasını yükle (kesin olsun diye proje kökünden)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

print("Telegram ayarları yükleniyor...")
if not tg_setup():
    print("❌ Telegram setup başarısız. .env dosyasını kontrol et.")
else:
    ok = tg_send("✅ Telegram test: Sistem online")
    print("OK" if ok else "FAIL")
