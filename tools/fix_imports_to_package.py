# tools/fix_imports_to_package.py  (SAFE v2)
from __future__ import annotations
import argparse, re
from pathlib import Path

# --- Güvenli dönüşüm kuralları ---
PATTERNS = [
    (re.compile(r'^(\s*)from\s+core\s+import\s+', re.M),
     r'\1from Proje1.core import '),
    (re.compile(r'^(\s*)from\s+core\.', re.M),
     r'\1from Proje1.core.'),
    (re.compile(r'^(\s*)import\s+core\.', re.M),
     r'\1import Proje1.core.'),
    (re.compile(r'^(\s*)import\s+core(\s|$)', re.M),
     r'\1import Proje1.core as core\2'),
    (re.compile(r'^(\s*)from\s+\.\s*([A-Za-z0-9_]+)\s+import\s+', re.M),
     r'\1from Proje1.core.\2 import '),
    (re.compile(r'^(\s*)from\s+\.(?=[A-Za-z0-9_])', re.M),
     r'\1from Proje1.core.'),
    (re.compile(r'^(\s*)from\s+\.\.\s*core(\.|(\s+import\s+))', re.M),
     r'\1from Proje1.core\1'),
    (re.compile(r'^(\s*)from\s+\.\.\s*import\s+([A-Za-z0-9_.,\s]+)', re.M),
     r'\1from Proje1.core import \2'),
]

EXCLUDE_DIR_PARTS = {
    "venv", ".venv", "__pycache__", "site-packages", "dist-packages", ".fix_imports_backup"
}

def is_excluded(path: Path) -> bool:
    parts = set(p.lower() for p in path.parts)
    return any(x in parts for x in EXCLUDE_DIR_PARTS)

def should_touch(p: Path, root: Path) -> bool:
    # Sadece Proje1/ altındaki .py dosyaları, ve exclude klasörleri hariç
    try:
        rel = p.relative_to(root)
    except Exception:
        return False
    if is_excluded(rel):
        return False
    return str(rel).replace("\\", "/").startswith("Proje1/") and p.suffix == ".py"

def transform(text: str) -> tuple[str, int]:
    count = 0
    for pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        count += n
    return text, count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root (Proje1 klasörünü içerir)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup_dir", default=".fix_imports_backup")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    py_files = list(root.rglob("*.py"))

    touched_total = 0
    changed_files = []

    for f in py_files:
        if not should_touch(f, root):
            continue
        src = f.read_text(encoding="utf-8")
        new, n = transform(src)
        if n > 0 and new != src:
            touched_total += n
            changed_files.append((f, n))
            if args.apply:
                backup = root / args.backup_dir / f.relative_to(root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    backup.write_text(src, encoding="utf-8")
                f.write_text(new, encoding="utf-8")

    print(f"[fix-safe] scanned={len(py_files)} changed_files={len(changed_files)} total_subs={touched_total}")
    for f, n in changed_files[:50]:
        print(f"  - {f} ({n} patch)")
    if len(changed_files) > 50:
        print(f"  ... and {len(changed_files)-50} more")

    print("\n[hint] Paket olarak çalıştırın:")
    print("  python -m Proje1.main --config config_live.json")
    print("  python tools/project_doctor.py  # tekrar kontrol")

if __name__ == "__main__":
    main()
