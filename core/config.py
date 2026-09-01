# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os
from pathlib import Path

def _read_json(path: Path, default: dict=None) -> dict:
    if not path or not path.exists(): return default or {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default or {}

def _apply_env_overrides(cfg: dict) -> dict:
    # Telegram env varsa otomatik enable
    tg = cfg.setdefault("telegram", {})
    if "TELEGRAM_BOT_TOKEN" in os.environ: tg["enabled"] = True
    return cfg

def _apply_runtime_overrides(cfg: dict) -> dict:
    """Brian'ın yazdığı canlı override'lar (runtime/runtime_overrides.jsonl)."""
    ov = Path("runtime/runtime_overrides.jsonl")
    if not ov.exists(): return cfg
    try:
        for line in ov.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            sets = rec.get("set", {})
            for key, val in sets.items():
                node = cfg
                parts = key.split(".")
                for p in parts[:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = val
    except Exception:
        pass
    return cfg

def load_cfg(path: str | None) -> dict:
    p = Path(path or "config_live.json")
    cfg = _read_json(p, {})
    cfg = _apply_env_overrides(cfg)
    cfg = _apply_runtime_overrides(cfg)

    # Brian vars (L2/L3)
    brain = cfg.setdefault("brain", {})
    brain.setdefault("authority", True)
    brain.setdefault("auto_overrides", True)
    brain.setdefault("log_path", "logs/brain_log.jsonl")
    brain.setdefault("veto_conf_min", 0.55)
    brain.setdefault("adjust", {
        "chop_qty_mult": 0.60,
        "trend_tp_mult": 1.06,
        "trend_sl_mult": 0.97,
        "spread_qty_penalty_bps": 8
    })

    # L4 dış veri
    ext = cfg.setdefault("external", {})
    ext.setdefault("enabled", True)
    ext.setdefault("news", True)
    ext.setdefault("macro", True)
    ext.setdefault("social", False)
    ext.setdefault("fallback_neutral", True)

    # L5/L6 öğrenme & autopatch
    learn = cfg.setdefault("learning", {})
    learn.setdefault("intra_day", {"enabled": True, "update_every_min": 10, "min_trades": 8})
    learn.setdefault("day_end",   {"enabled": True, "grid_candidates": 12,
                                   "rollout_guard": {"wr": 0.48, "pf": 1.10, "maxdd": -3.0}})
    learn.setdefault("autopatch", {"enabled": True, "sandbox_minutes": 20,
                                   "rollback_if": {"wr": 0.42, "pf": 1.05}})

    # Risk failsafe
    cfg.setdefault("risk", {}).setdefault("daily_max_loss_usd", 10.0)
    cfg["risk"].setdefault("cooldown_min", 30)
    return cfg
