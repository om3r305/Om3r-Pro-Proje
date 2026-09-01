# === BEGIN FILE: tools/make_release.py
from __future__ import annotations
import shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / f"release_{time.strftime('%Y%m%d_%H%M%S')}"

INCLUDE = [
    "core", "live", "model", "scripts", "logs",
    "config_live.json", "main.py", "requirements.txt",
]
EXCLUDE_PATTERNS = ["__pycache__", ".git", ".patch_backups"]

def copy_tree(src: Path, dst: Path):
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(x in p.as_posix() for x in EXCLUDE_PATTERNS):
            continue
        t = dst / rel
        if p.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        else:
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for item in INCLUDE:
        sp = ROOT / item
        if not sp.exists():
            continue
        if sp.is_dir():
            copy_tree(sp, OUT / item)
        else:
            (OUT / item).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, OUT / item)
    print(f"[release] ready at: {OUT}")

if __name__ == "__main__":
    main()
# === END FILE: tools/make_release.py
