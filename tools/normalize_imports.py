import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # proje kökü
PKG  = "proje1"

IN_PKG_DIRS = {"proje1", "proje1/core", "proje1/strategies"}
TOOLS_DIRS  = {"proje1/tools"}

def is_in_pkg(p: Path) -> bool:
    try:
        rel = p.relative_to(ROOT)
    except ValueError:
        return False
    return any(str(rel).startswith(d + "/") for d in IN_PKG_DIRS)

def is_in_tools(p: Path) -> bool:
    try:
        rel = p.relative_to(ROOT)
    except ValueError:
        return False
    return any(str(rel).startswith(d + "/") for d in TOOLS_DIRS)

# Basit kalıplar
ABS_CORE = re.compile(r"\bfrom\s+core\.([a-zA-Z0-9_\.]+)\s+import\b")
ABS_PROJ = re.compile(rf"\bfrom\s+{PKG}\.core\.([a-zA-Z0-9_\.]+)\s+import\b")
REL_DOT  = re.compile(r"\bfrom\s+\.\s*core\.([a-zA-Z0-9_\.]+)\s+import\b")

def to_relative(line: str) -> str:
    # 'from Proje1.core.xxx import y'  -> 'from .core.xxx import y'
    line = ABS_CORE.sub(r"from .core.\1 import", line)
    # 'from proje1.core.xxx import y' -> 'from .core.xxx import y'
    line = ABS_PROJ.sub(r"from .core.\1 import", line)
    return line

def to_absolute_for_tools(line: str) -> str:
    # 'from Proje1.core.xxx import y'  -> 'from proje1.core.xxx import y'
    line = ABS_CORE.sub(rf"from {PKG}.core.\1 import", line)
    # 'from .core.xxx import y' -> 'from proje1.core.xxx import y'
    line = REL_DOT.sub(rf"from {PKG}.core.\1 import", line)
    return line

def process_file(fp: Path):
    src = fp.read_text(encoding="utf-8")
    new = src
    if is_in_pkg(fp):
        new = "\n".join(to_relative(l) for l in src.splitlines())
    elif is_in_tools(fp):
        new = "\n".join(to_absolute_for_tools(l) for l in src.splitlines())
    else:
        return False
    if new != src:
        fp.write_text(new, encoding="utf-8")
        print("[fix]", fp)
        return True
    return False

def main():
    changed = 0
    for py in ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in py.parts):
            continue
        if py.name == "__init__.py":
            continue
        changed += bool(process_file(py))
    print(f"done. changed={changed}")

if __name__ == "__main__":
    sys.exit(main())
