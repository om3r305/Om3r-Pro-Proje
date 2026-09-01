# -*- coding: utf-8 -*-
from __future__ import annotations
import shutil, sys
from pathlib import Path
def main():
    backup=Path(".patch_backups")
    if not backup.exists():
        print("No backups found."); return 1
    targets=sorted(backup.glob("**/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    count=0
    for p in targets:
        if p.is_file():
            rel=p.relative_to(backup)
            dest=Path(".")/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest); count+=1
    print(f"Restored {count} files from backups.")
    return 0
if __name__ == "__main__":
    sys.exit(main())
