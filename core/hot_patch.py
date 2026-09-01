# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, time, shutil, tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

class HotPatchError(Exception): ...

class HotPatch:
    """
    Desteklenen ops:
      - replace {pattern, repl}
      - insert_after {anchor, text}
      - ensure_import {import}
      - json_set {key, value}
      - create_file {text}
      - append {text}
    """
    def __init__(self, root: Optional[str]=None, backup_dir: Optional[str]=None):
        self.root = Path(root or Path(__file__).resolve().parent.parent)
        self.backup_dir = Path(backup_dir or (self.root/"backups"))
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _read(self, p: Path) -> str:
        with open(p, "r", encoding="utf-8") as f: return f.read()

    def _write_atomic(self, p: Path, data: str) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="hp_", dir=str(p.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        with open(tmp, "w", encoding="utf-8") as f: f.write(data)
        if p.exists():
            bkp = self.backup_dir / (p.name + f".{int(time.time())}.bak")
            try: shutil.copy2(p, bkp)
            except Exception: pass
        os.replace(tmp, p)

    def _write_json_atomic(self, p: Path, obj: Dict[str, Any]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="hp_", dir=str(p.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        with open(tmp, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
        if p.exists():
            bkp = self.backup_dir / (p.name + f".{int(time.time())}.bak.json")
            try: shutil.copy2(p, bkp)
            except Exception: pass
        os.replace(tmp, p)

    def apply(self, ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results=[]
        for op in ops:
            kind = op.get("op")
            rel = op.get("path")
            if not kind or not rel: raise HotPatchError("op/path gerekli")
            path = (self.root / rel).resolve()

            if kind == "create_file":
                text = op.get("text","")
                self._write_atomic(path, text)
                results.append({"op":kind,"path":rel,"created":True})
                continue

            if not path.exists() and kind not in ("create_file",):
                raise HotPatchError(f"dosya yok: {rel}")

            if kind == "replace":
                pat = re.compile(op["pattern"], re.MULTILINE | re.DOTALL)
                src = self._read(path)
                new = pat.sub(op.get("repl",""), src)
                if new == src: results.append({"op":kind,"path":rel,"changed":False}); continue
                self._write_atomic(path, new); results.append({"op":kind,"path":rel,"changed":True})

            elif kind == "insert_after":
                anchor = re.compile(op["anchor"], re.MULTILINE)
                src = self._read(path)
                m = anchor.search(src)
                if not m: results.append({"op":kind,"path":rel,"changed":False,"err":"anchor yok"}); continue
                pos = m.end()
                new = src[:pos] + op.get("text","") + src[pos:]
                self._write_atomic(path, new); results.append({"op":kind,"path":rel,"changed":True})

            elif kind == "append":
                src = self._read(path)
                new = src + op.get("text","")
                self._write_atomic(path, new); results.append({"op":kind,"path":rel,"changed":True})

            elif kind == "ensure_import":
                imp = op["import"].strip()
                src = self._read(path)
                if imp in src:
                    results.append({"op":kind,"path":rel,"changed":False}); continue
                new = imp + "\n" + src
                self._write_atomic(path, new); results.append({"op":kind,"path":rel,"changed":True})

            elif kind == "json_set":
                obj = json.loads(self._read(path))
                key = op["key"]; val = op.get("value")
                cur = obj
                parts = key.split(".")
                for p in parts[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p]={}
                    cur = cur[p]
                cur[parts[-1]] = val
                self._write_json_atomic(path, obj); results.append({"op":kind,"path":rel,"changed":True})

            else:
                raise HotPatchError(f"desteklenmeyen op: {kind}")
        return results
