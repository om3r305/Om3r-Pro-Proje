# === BEGIN FILE: tools/config_merge.py
from __future__ import annotations
import json, sys
from pathlib import Path
from copy import deepcopy

# L9 şeması için minimal defaultlar (yalnızca eksik alanları tamamlar)
DEFAULTS = {
    "runtime": {"overrides_path": "runtime/runtime_overrides.jsonl"},
    "market_intel": {"enabled": True, "path": "logs/market_intel.jsonl", "max_kb": 512},
    "logging": {"trades_full_log": "logs/trades_full_log.csv"},
    "file_watcher": {
        "enabled": True,
        "scan_interval_sec": 20,
        "scan_dirs": ["core", "live", "model", "scripts"],
        "include_globs": ["**/*.py", "**/*.json", "**/*.yaml", "**/*.yml"],
        "exclude_globs": ["**/__pycache__/**", "**/.patch_backups/**", "**/.git/**"],
        "quality_rules": {
            "min_lines": {
                "core/file_watcher.py": 120,
                "core/log_ext.py": 120,
                "core/brain_selfheal.py": 150
            },
            "must_contain": {
                "core/log_ext.py": ["def write_trades_full_row", "def log_market_intel"],
                "core/brain_selfheal.py": ["def report_exception", "def ensure_selfheal_watcher"]
            }
        },
        "actions": {
            "on_stale": "patch",
            "on_broken": "disable",
            "auto_patch": {"enabled": True, "backup_dir": ".patch_backups", "max_backups": 20},
            "auto_disable": {"enabled": True, "write_tag_comment": True},
            "auto_adapter": {"enabled": True, "dir": "live/adapters"}
        },
        "notify": {
            "mode": "hybrid",
            "telegram": True,
            "digest_time_utc": "20:00",
            "journal_path": "logs/filewatch_journal.jsonl",
            "rate_limit_per_min": 8
        }
    },
    "brain": {
        "learn": {"meta_path": "model/brain_meta.json"},
        "auto_weights": {
            "enabled": True, "lookback_trades": 400, "temperature": 0.6, "alpha": 0.35,
            "floors": {"dip":0.10,"pred":0.15,"news":0.10,"ob":0.10},
            "caps":   {"dip":0.55,"pred":0.60,"news":0.45,"ob":0.45},
            "entry_frac_floor_cap": [0.30, 1.70]
        }
    },
    "evo": {"champions_path": "model/evo_champions.jsonl"}
}

def deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if k not in dst:
            dst[k] = deepcopy(v)
        else:
            if isinstance(dst[k], dict) and isinstance(v, dict):
                deep_merge(dst[k], v)
    return dst

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/config_merge.py <in_config.json> <out_config.json>")
        sys.exit(1)
    inp = Path(sys.argv[1]); outp = Path(sys.argv[2])
    cfg = json.loads(inp.read_text(encoding="utf-8"))
    merged = deep_merge(cfg, DEFAULTS)
    outp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] wrote {outp}")

if __name__ == "__main__":
    main()
# === END FILE: tools/config_merge.py
