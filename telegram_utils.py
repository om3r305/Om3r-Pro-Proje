import os
import requests

_TG_TOKEN = None
_TG_CHAT_ID = None

def tg_setup():
    """
    .env:
      TELEGRAM_TOKEN=123456:ABC-xyz
      TELEGRAM_CHAT_ID=987654321
    """
    global _TG_TOKEN, _TG_CHAT_ID
    _TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
    _TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    if not _TG_TOKEN or not _TG_CHAT_ID:
        print("⚠️ Telegram token/chat_id eksik! Mesaj gönderilmeyecek.")
    else:
        print(f"✅ Telegram ayarlandı | chat_id={_TG_CHAT_ID}")

def tg_send(msg: str):
    if not _TG_TOKEN or not _TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": _TG_CHAT_ID, "text": msg}, timeout=5)
        if r.status_code != 200:
            print(f"⚠️ Telegram gönderim hatası: {r.text}")
    except Exception as e:
        print(f"⚠️ Telegram exception: {e}")
