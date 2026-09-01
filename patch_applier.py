#!/usr/bin/env python3
# patch_applier.py — Monster Coins Pro
# Aksiyonlar: json_merge, append_if_missing, edit_text
# Ekstralar: --dry-run, --backup-dir, --rules, --verify-only

from __future__ import annotations
import argparse, os, sys, re, json, shutil, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    import yaml
except Exception:
    print("pyyaml gerekli: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ENC = "utf-8"

def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def _ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _read_text(p: Path) -> str:
    return p.read_text(encoding=ENC) if p.exists() else ""

def _write_text(p: Path, data: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding=ENC)

def _backup(src: Path, backup_dir: Path) -> Path:
    _ensure_dir(backup_dir)
    dst = backup_dir / f"{src.name}.{_stamp()}.bak"
    shutil.copy2(src, dst)
    return dst

def _deep_merge(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _deep_merge(a.get(k), v) if k in a else v
        return out
    return b

def _json_merge(fp: Path, patch: Dict[str, Any], dry: bool, backups: Path) -> Tuple[bool,str]:
    cur: Dict[str, Any] = {}
    if fp.exists():
        try: cur = json.loads(_read_text(fp))
        except Exception as e: return False, f"JSON parse hatası: {fp}: {e}"
    merged = _deep_merge(cur, patch)
    if dry:
        return True, f"[dry] {fp} (json_merge) yazılmadı"
    if fp.exists(): _backup(fp, backups)
    _write_text(fp, json.dumps(merged, ensure_ascii=False, indent=2))
    return True, f"OK json_merge → {fp}"

def _append_if_missing(fp: Path, blob: str, dry: bool, backups: Path) -> Tuple[bool,str]:
    cur = _read_text(fp)
    if blob.strip() in cur:
        return True, f"skip (zaten var) → {fp}"
    if dry:
        return True, f"[dry] append → {fp}"
    if fp.exists(): _backup(fp, backups)
    _write_text(fp, (cur + ("\n" if cur and not cur.endswith("\n") else "") + blob))
    return True, f"OK append → {fp}"

def _edit_text(fp: Path, pattern: str, repl: str, multiple: bool, dry: bool, backups: Path) -> Tuple[bool,str]:
    cur = _read_text(fp)
    flags = re.MULTILINE | re.DOTALL
    if multiple:
        new, n = re.subn(pattern, repl, cur, flags=flags)
    else:
        new, n = re.subn(pattern, repl, cur, count=1, flags=flags)
    if n == 0:
        return False, f"eşleşme yok: {fp} / {pattern}"
    if dry:
        return True, f"[dry] edit ({n} değişim) → {fp}"
    if fp.exists(): _backup(fp, backups)
    _write_text(fp, new)
    return True, f"OK edit ({n}) → {fp}"

def apply_rule(root: Path, rule: Dict[str, Any], dry: bool, backups: Path) -> Tuple[bool,str]:
    rel = rule.get("file")
    if not rel: return False, "rule.file eksik"
    fp = (root / rel).resolve()
    action = (rule.get("action") or "").lower()

    if action == "json_merge":
        return _json_merge(fp, rule.get("patch") or {}, dry, backups)
    elif action == "append_if_missing":
        return _append_if_missing(fp, rule.get("blob") or "", dry, backups)
    elif action == "edit_text":
        return _edit_text(fp, rule.get("pattern") or "", rule.get("repl") or "",
                          bool(rule.get("multiple", False)), dry, backups)
    else:
        return False, f"bilinmeyen action: {action}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="patch_rules.yaml")
    ap.add_argument("--backup-dir", default=".patch_backups")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    root = Path(".").resolve()
    rules_path = root / args.rules
    if not rules_path.exists():
        print(f"kural dosyası yok: {rules_path}", file=sys.stderr)
        sys.exit(2)

    with rules_path.open("r", encoding=ENC) as f:
        doc = yaml.safe_load(f) or {}
    rules = doc.get("rules") or []
    if not isinstance(rules, list):
        print("rules bekleniyor (liste).", file=sys.stderr); sys.exit(2)

    ok_all = True
    for i, rule in enumerate(rules, 1):
        if args.verify_only:
            print(f"[verify] #{i} {rule.get('file')} -> {rule.get('action')}")
            continue
        ok, msg = apply_rule(root, rule, args.dry_run, Path(args.backup_dir))
        ok_all = ok_all and ok
        print(("✔" if ok else "✗"), msg)

    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
