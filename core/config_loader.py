# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, glob
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

from Proje1.core.guardrails import get as cfg_get

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "config"
RUNTIME = ROOT / "runtime"

def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def _load_overrides_jsonl(path: Path) -> dict:
    out = {}
    if not path.exists(): return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            kv = rec.get("set") or rec.get("overrides") or rec.get("kv") or {}
            if isinstance(kv, dict):
                out.update(kv)
    except Exception:
        pass
    return out

def _apply_kv(cfg: dict, kv: dict) -> dict:
    for k, v in (kv or {}).items():
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = v
    return cfg

def load_config() -> dict:
    base = _read_json(CONF_DIR / "config_live.json", {})
    brain = _read_json(CONF_DIR / "brain_config.json", {})
    cfg = _merge(base, brain)

    # .env -> telegram
    token = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    chat  = os.getenv("TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if token or chat:
        tg = dict(cfg.get("telegram", {}))
        if token: tg["token"] = token
        if chat: tg["chat_id"] = chat
        cfg["telegram"] = tg

    # runtime overrides jsonl
    kv = _load_overrides_jsonl(RUNTIME / "runtime_overrides.jsonl")
    if kv: cfg = _apply_kv(cfg, kv)

    return cfg
