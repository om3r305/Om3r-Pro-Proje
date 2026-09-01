# Proje1/tools/force_normalize.py
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "Proje1"

# sadece core içindeki dosyaları normalize ediyoruz
TARGETS = ["core", "strategies", "tools"]

changed = 0
for folder in TARGETS:
    for pyfile in (ROOT / folder).rglob("*.py"):
        text = pyfile.read_text(encoding="utf-8")
        new = text

        # relative importları düzelt
        new = re.sub(r"from\s+\.\s*([a-zA-Z0-9_]+)\s+import",
                     rf"from {PKG}.{folder}.\1 import", new)

        new = re.sub(r"from\s+core\.([a-zA-Z0-9_]+)\s+import",
                     rf"from {PKG}.core.\1 import", new)

        if new != text:
            pyfile.write_text(new, encoding="utf-8")
            changed += 1
            print(f"[fix] {pyfile}")

print("Toplam değişen dosya:", changed)
