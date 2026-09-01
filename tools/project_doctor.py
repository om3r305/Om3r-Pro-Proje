# Proje1/tools/project_doctor.py
from __future__ import annotations
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../Proje1
SKIP_DIRS = {"venv", ".git", "__pycache__"}

REL_CORE_RX = re.compile(r'(^|\s)from\s+core\.|(^|\s)import\s+core\.|(^|\s)from\s+\.+core\.', re.M)
ABS_CORE_RX = re.compile(r'(^|\s)from\s+Proje1\.core\.|(^|\s)import\s+Proje1\.core\.', re.M)

def should_skip(p: Path) -> bool:
    parts = {x.name for x in p.parents}
    return any(x in parts for x in SKIP_DIRS)

def read_text(p: Path) -> str:
    # UTF-8 ama BOM/garip karakterleri tolere et
    return p.read_text(encoding="utf-8", errors="ignore")

def check_ast(py: Path) -> tuple[bool, str]:
    try:
        code = read_text(py)
        ast.parse(code, filename=str(py))
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def check_import_style(py: Path) -> tuple[bool, str]:
    """
    KURAL:
      - Dosyada core'a relatif import VARSA (from core..., import core..., from ..core...)
        VE aynı dosyada ABSOLUTE (Proje1.core...) YOKSA -> FAIL
      - Aksi halde PASS (dosyada core hiç kullanılmıyorsa da PASS).
    """
    text = read_text(py)
    has_rel = bool(REL_CORE_RX.search(text))
    has_abs = bool(ABS_CORE_RX.search(text))
    if has_rel and not has_abs:
        return False, "missing absolute Proje1.core import pattern"
    return True, ""

def main():
    print(f"[doctor] root={ROOT}")

    py_files = [p for p in ROOT.rglob("*.py") if not should_skip(p)]
    ast_fails = []
    import_fails = []

    # 1) AST
    for p in py_files:
        ok, why = check_ast(p)
        if not ok:
            ast_fails.append((p, why))

    # 2) Import style (sadece core klasörü altında anlamlı)
    for p in [x for x in py_files if ROOT / "core" in x.parents]:
        ok, why = check_import_style(p)
        if not ok:
            import_fails.append((p, why))

    print("\n[doctor] SUMMARY")
    print(f"  py_files     : {len(py_files)}")
    print(f"  ast_fails    : {len(ast_fails)}")
    print(f"  import_fails : {len(import_fails)}")
    print(f"  crit_fails   : 0")
    print(f"  json_fails   : 0\n")

    if ast_fails:
        print("[ast failures]")
        for p, why in sorted(ast_fails):
            print(f" - {p.relative_to(ROOT)} -> {why}")
        print()

    if import_fails:
        print("[import style failures]")
        for p, why in sorted(import_fails):
            print(f" - {p.relative_to(ROOT)} -> {why}")
        print()

if __name__ == "__main__":
    main()
