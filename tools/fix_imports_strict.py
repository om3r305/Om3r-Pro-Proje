# Proje1/tools/fix_imports_strict.py
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # .../Proje1
CORE = ROOT / "core"

SKIP_DIRS = {"venv", ".git", "__pycache__"}

# NOT: Zaten Proje1.core geçen satırları değiştirmiyoruz.
PATTERNS = [
    # from Proje1.core.x import Y  -> from Proje1.core.x import Y
    (re.compile(r'(^|\s)from\s+core(\.[\w\.]*)\s+import\s+(?!.*Proje1\.core)', re.M),
     lambda m: f"{m.group(1)}from Proje1.core{m.group(2)} import "),
    # from core import Y -> from Proje1.core import Y
    (re.compile(r'(^|\s)from\s+core\s+import\s+(?!.*Proje1\.core)', re.M),
     lambda m: f"{m.group(1)}from Proje1.core import "),
    # import core.x  -> import Proje1.core.x
    (re.compile(r'(^|\s)import\s+core(\.[\w\.]*)(?![\w\.])', re.M),
     lambda m: f"{m.group(1)}import Proje1.core{m.group(2)}"),
    # import core -> import Proje1.core
    (re.compile(r'(^|\s)import\s+core(\s|$)', re.M),
     lambda m: f"{m.group(1)}import Proje1.core{m.group(2)}"),
    # from .core.x import Y / from ..core.x import Y -> from Proje1.core.x import Y
    (re.compile(r'(^|\s)from\s+(\.+)core(\.[\w\.]*)\s+import\s+', re.M),
     lambda m: f"{m.group(1)}from Proje1.core{m.group(3)} import "),
]

def should_skip(path: Path) -> bool:
    parts = {p.name for p in path.parents}
    return any(d in parts for d in SKIP_DIRS)

def process_file(fp: Path) -> bool:
    text = fp.read_text(encoding="utf-8")
    orig = text
    for rx, repl in PATTERNS:
        text = rx.sub(repl, text)
    if text != orig:
        fp.write_text(text, encoding="utf-8")
        return True
    return False

def main():
    changed = 0
    for py in CORE.rglob("*.py"):
        if should_skip(py):
            continue
        if process_file(py):
            print(f"[fix] {py}")
            changed += 1
    print(f"[fix] done. changed_files={changed}")

if __name__ == "__main__":
    main()
