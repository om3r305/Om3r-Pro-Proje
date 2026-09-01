# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys, shutil
from pathlib import Path

ENC_HEADER = "# -*- coding: utf-8 -*-\n"
SKIP_DIRS = {"venv", ".git", "__pycache__", ".mypy_cache", ".pytest_cache"}
VALID_EXT = {".py"}

def has_bom(raw: bytes) -> bool:
    return raw.startswith(b"\xef\xbb\xbf")

def ensure_utf8_header(text: str) -> str:
    lines = text.splitlines(True)
    if not lines:
        return ENC_HEADER
    head = "".join(lines[:2])
    if "coding:" in head or "coding=" in head or "utf-8" in head.lower():
        return text
    return ENC_HEADER + text

def clean_file(fp: Path, in_place: bool, backup: bool) -> bool:
    raw = fp.read_bytes()
    changed = False
    if has_bom(raw):
        raw = raw[3:]
        changed = True
    text = raw.decode("utf-8", errors="replace")
    text_norm = text.replace("\r\n", "\n")
    if text_norm != text:
        text = text_norm
        changed = True
    text2 = ensure_utf8_header(text)
    if text2 != text:
        text = text2
        changed = True
    if changed and in_place:
        if backup:
            shutil.copy2(fp, fp.with_suffix(fp.suffix + ".bak"))
        fp.write_text(text, encoding="utf-8", newline="\n")
    return changed

# ... mevcut importlar ve sabitler aynı ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=False)   # <- required=False
    ap.add_argument("--file", type=str, required=False, help="Tek bir .py dosyasını düzelt")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--backup", action="store_true", default=True)
    args = ap.parse_args()

    files = []
    if args.file:
        fp = Path(args.file).resolve()
        if not fp.exists():
            print(f"[fix-enc] file not found: {fp}")
            sys.exit(2)
        files = [fp]
    else:
        root = Path(args.root).resolve()
        if not root.exists():
            print(f"[fix-enc] root not found: {root}")
            sys.exit(2)
        for fp in root.rglob("*.py"):
            if fp.is_file() and fp.suffix == ".py" and not any(d.name in SKIP_DIRS for d in fp.parents):
                files.append(fp)

    total = 0
    changed = 0
    for fp in files:
        total += 1
        if clean_file(fp, in_place=args.in_place, backup=args.backup):
            print(f"[fix-enc] would change: {fp}" if not args.in_place else f"[fix-enc] changed: {fp}")
            changed += 1

    mode = "apply" if args.in_place else "dry-run"
    print(f"[fix-enc] mode={mode} scanned={total} "
          f"{'would_change' if mode=='dry-run' else 'changed'}={changed}")

