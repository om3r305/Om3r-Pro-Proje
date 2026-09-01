# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- Proje kökünü sys.path'e ÖNCE ekle (idempotent) ---------------------------
_pkg_dir = Path(__file__).resolve().parent      # .../Proje1
_proj_root = _pkg_dir.parent                    # proje kökü
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

# --- (ops.) .env yükle: API anahtarları vs. ----------------------------------
# dotenv yoksa sessizce geçer; varsa CMC_API_KEY gibi değişkenler erken yüklenir.
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None
if load_dotenv:
    # Öncelik: proje kökünde .env varsa onu yükle
    env_path = _proj_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))  # tip: CMC_API_KEY burada olabilir
    else:
        load_dotenv()  # çalışma dizinindeki .env

# --- AutoPkg: alias+finder kur, eksik modülleri skeleton ile oluştur ----------
# Not: Modülün kendisini import edip onun üzerinden çağırıyoruz.
from Proje1.core import auto_pkg_bootstrap as _autopkg
_autopkg.preload_missing_modules()  # created/aliased sayılarını TG'ye de yollar

# --- Paket içi mutlak importlar ----------------------------------------------
from Proje1.core.bot import Bot
from Proje1.core.utils_io import tg_send
from Proje1.core.brain_selfheal import ensure_selfheal_watcher, report_exception


def _resolve_config_path(arg_path: str | Path) -> Path:
    """--config ile gelen göreli/yol karışık hallerini sağlam çöz."""
    p = Path(arg_path)
    if p.exists():
        return p
    cand = _pkg_dir / str(arg_path)
    if cand.exists():
        return cand
    cwd_cand = Path.cwd() / str(arg_path)
    if cwd_cand.exists():
        return cwd_cand
    raise FileNotFoundError(
        f"Config not found: {arg_path} (tried: {p}, {cand}, {cwd_cand})"
    )


def load_config(path: str | Path) -> dict:
    p = _resolve_config_path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="Proje1/config_live.json",
        help="JSON config path (relative allowed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 🧠 Self-heal’i başta devreye al
    selfheal_ok = ensure_selfheal_watcher()
    print(f"[main] selfHeal check -> {selfheal_ok}")

    # Bilgilendir: kritik env anahtarları var mı?
    if os.getenv("CMC_API_KEY"):
        print("[main] CMC_API_KEY yüklendi (env).")
    else:
        print("[main] CMC_API_KEY bulunamadı (env). Haber/CMC opsiyonel kaynak devre dışı kalabilir.")

    bot = Bot(cfg)

    try:
        bot.run()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
    except Exception as e:
        try:
            report_exception("main", e)
        except Exception:
            pass
        try:
            tg_send(f"[ERR] main: {e}", parse_mode="HTML")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
