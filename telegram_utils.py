# telegram_utils.py — Monster Coins Pro (hardened / fail-safe)

from __future__ import annotations
import os, time, json, typing, requests, pathlib
from typing import Optional, Union

# --- .env yükle (varsa) ------------------------------------------------------
def _load_env():
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    # bu dosyanın yanındaki/üstündeki klasörlerde .env ara
    here = pathlib.Path(__file__).resolve()
    cands = [
        here.parent / ".env",
        here.parent.parent / ".env",
        pathlib.Path.cwd() / ".env",
    ]
    for p in cands:
        if p.exists():
            load_dotenv(str(p))
            break

_load_env()

# --- ENV ---------------------------------------------------------------------
_TG_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() in ("1","true","yes","on")
_TG_TOKEN   = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
_TG_CHATID  = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
# Varsayılanı boş = parse_mode kullanma → markdown hatası riski yok
_TG_PARSE   = (os.getenv("TELEGRAM_PARSE_MODE") or "").strip()
_TG_SILENT  = (os.getenv("TELEGRAM_SILENT") or "false").lower() in ("1","true","yes","on")

_BASE     = "https://api.telegram.org/bot{token}/{method}"
_MAX_CHARS = 4000      # 4096 güvenli sınır
_TIMEOUT   = 10
_RETRIES   = 3
_BACKOFF   = 1.5

_last_message_id: Optional[int] = None


# --- durum helpers -----------------------------------------------------------
def is_ready() -> tuple[bool, str]:
    """(ready, reason) döner; kapalıysa/kayıpsa sebebi string olarak verir."""
    if not _TG_ENABLED:
        return False, "disabled"
    if not _TG_TOKEN:
        return False, "missing TELEGRAM_BOT_TOKEN"
    if not _TG_CHATID:
        return False, "missing TELEGRAM_CHAT_ID"
    return True, "ok"

def _req(method: str, *, data=None, files=None) -> dict:
    url = _BASE.format(token=_TG_TOKEN, method=method)
    last_err = None
    for i in range(_RETRIES):
        try:
            r = requests.post(url, data=data, files=files, timeout=_TIMEOUT)
        except NameError:
            # küçük typo koruması
            r = requests.post(url, data=data, files=files, timeout=_TIMEOUT)
        except Exception as e:
            last_err = str(e)
            time.sleep((_BACKOFF ** i) * 0.5)
            continue

        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return {"ok": True}
        else:
            last_err = f"HTTP {r.status_code} {r.text[:200]}"
        time.sleep((_BACKOFF ** i) * 0.5)
    raise RuntimeError(f"Telegram API hatası: {last_err}")

def _split_chunks(text: str, n: int = _MAX_CHARS) -> list[str]:
    t = str(text)
    if len(t) <= n:
        return [t]
    parts, cur, cur_len = [], [], 0
    for line in t.splitlines(True):
        if cur_len + len(line) > n and cur:
            parts.append("".join(cur)); cur=[]; cur_len=0
        cur.append(line); cur_len += len(line)
    if cur:
        parts.append("".join(cur))
    return parts


# --- public API --------------------------------------------------------------
def tg_setup() -> bool:
    """Kritik değil: sadece hazırsa bir karşılama gönderir; asla exception fırlatmaz."""
    ok, why = is_ready()
    if not ok:
        print(f"[TG] setup atlandı: {why}")
        return False
    try:
        tg_send("🟣 Monster Coins Pro — Telegram bağlantısı aktif.", silent=True)
        return True
    except Exception as e:
        print("[TG] setup uyarı:", e)
        return False


def tg_send(text: str,
            *,
            parse_mode: Optional[str] = None,
            silent: Optional[bool] = None,
            disable_web_preview: bool = True) -> bool:
    """Uzun mesajları böler; hazır değilse sessizce False döner; exception fırlatmaz."""
    ok, _ = is_ready()
    if not ok:
        return False

    pm = parse_mode if parse_mode is not None else _TG_PARSE  # "" → hiç gönderme
    sl = _TG_SILENT if silent is None else bool(silent)

    sent_any = False
    for chunk in _split_chunks(text):
        data = {
            "chat_id": _TG_CHATID,
            "text": chunk,
            "disable_web_page_preview": "true" if disable_web_preview else "false",
            "disable_notification": "true" if sl else "false",
        }
        if pm:
            data["parse_mode"] = pm
        try:
            res = _req("sendMessage", data=data)
            global _last_message_id
            try:
                _last_message_id = int(res.get("result", {}).get("message_id", 0)) or _last_message_id
            except Exception:
                pass
            sent_any = True
        except Exception as e:
            print("[TG] send fail:", e)
    return sent_any


def tg_edit_last(new_text: str,
                 *,
                 parse_mode: Optional[str] = None,
                 disable_web_preview: bool = True) -> bool:
    ok, _ = is_ready()
    if not ok or not _last_message_id:
        return False
    pm = parse_mode if parse_mode is not None else _TG_PARSE
    data = {
        "chat_id": _TG_CHATID,
        "message_id": _last_message_id,
        "text": new_text[:_MAX_CHARS],
        "disable_web_page_preview": "true" if disable_web_preview else "false",
    }
    if pm:
        data["parse_mode"] = pm
    try:
        _req("editMessageText", data=data)
        return True
    except Exception as e:
        print("[TG] edit fail:", e)
        return False


def tg_photo(photo: Union[str, bytes],
             caption: Optional[str] = None,
             *,
             parse_mode: Optional[str] = None,
             silent: Optional[bool] = None) -> bool:
    ok, _ = is_ready()
    if not ok:
        return False
    pm = parse_mode if parse_mode is not None else _TG_PARSE
    sl = _TG_SILENT if silent is None else bool(silent)

    files = None
    data = {
        "chat_id": _TG_CHATID,
        "disable_notification": "true" if sl else "false",
    }
    if caption:
        data["caption"] = caption[:1024]
        if pm: data["parse_mode"] = pm

    try:
        if isinstance(photo, (bytes, bytearray)):
            files = {"photo": ("image.jpg", photo)}
        else:
            path = str(photo)
            if not os.path.exists(path):
                print(f"[TG] photo yok: {path}")
                return False
            files = {"photo": (os.path.basename(path), open(path, "rb"))}
        _req("sendPhoto", data=data, files=files)
        return True
    except Exception as e:
        print("[TG] photo fail:", e)
        return False
    finally:
        try:
            if isinstance(photo, str) and files and hasattr(files["photo"][1], "close"):
                files["photo"][1].close()
        except Exception:
            pass


def tg_doc(path: str,
           caption: Optional[str] = None,
           *,
           parse_mode: Optional[str] = None,
           silent: Optional[bool] = None) -> bool:
    ok, _ = is_ready()
    if not ok:
        return False
    if not os.path.exists(path):
        print(f"[TG] doc yok: {path}")
        return False

    pm = parse_mode if parse_mode is not None else _TG_PARSE
    sl = _TG_SILENT if silent is None else bool(silent)

    data = {
        "chat_id": _TG_CHATID,
        "disable_notification": "true" if sl else "false",
    }
    if caption:
        data["caption"] = caption[:1024]
        if pm: data["parse_mode"] = pm

    files = {"document": (os.path.basename(path), open(path, "rb"))}
    try:
        _req("sendDocument", data=data, files=files)
        return True
    except Exception as e:
        print("[TG] doc fail:", e)
        return False
    finally:
        try: files["document"][1].close()
        except Exception: pass


def tg_status() -> dict:
    """Bot/Chat hızlı sağlık kontrolü (getMe + typing). Exception fırlatmaz."""
    ok, why = is_ready()
    out = {"enabled": _TG_ENABLED, "ready": ok, "why": None if ok else why}
    if not ok:
        return out
    try:
        me = _req("getMe")
        out["me"] = me.get("result", {})
    except Exception as e:
        out["me_err"] = str(e)
    try:
        _req("sendChatAction", data={"chat_id": _TG_CHATID, "action": "typing"})
        out["chat_ok"] = True
    except Exception as e:
        out["chat_err"] = str(e)
    return out


# --- CLI ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Telegram utils quick test")
    ap.add_argument("--msg", help="Metin mesaj gönder", default=None)
    ap.add_argument("--photo", help="Foto yolunu gönder", default=None)
    ap.add_argument("--doc", help="Doküman yolunu gönder", default=None)
    ap.add_argument("--status", action="store_true", help="Durum sorgula")
    args = ap.parse_args()

    if args.status:
        print(json.dumps(tg_status(), ensure_ascii=False, indent=2))

    if args.msg:
        print("send:", tg_send(args.msg))

    if args.photo:
        print("photo:", tg_photo(args.photo, caption="📸 Foto"))

    if args.doc:
        print("doc:", tg_doc(args.doc, caption="📄 Doküman"))
