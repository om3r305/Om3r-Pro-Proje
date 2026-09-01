# file_ops.py — Monster Coins Pro • Safe FS + Patcher
import os, io, re, json, shutil, time, hashlib
from typing import List, Dict, Any, Optional, Tuple

REPO_ROOT = os.path.abspath(os.getcwd())
BACKUP_DIR = os.path.join(REPO_ROOT, "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

def _norm(path: str) -> str:
    p = os.path.abspath(os.path.join(REPO_ROOT, path))
    if not p.startswith(REPO_ROOT):
        raise ValueError("Path escapes repo root")
    return p

def _rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT)

def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _backup_path(abs_path: str) -> str:
    stamp = _stamp()
    rel = _rel(abs_path).replace("\\", "/")
    bdir = os.path.join(BACKUP_DIR, stamp)
    os.makedirs(os.path.join(bdir, os.path.dirname(rel)), exist_ok=True)
    return os.path.join(bdir, rel)

def _sha256(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()[:16]

def list_dir(path: str = ".") -> Dict[str, Any]:
    absdir = _norm(path)
    if not os.path.isdir(absdir):
        raise FileNotFoundError("Directory not found")
    out = {"cwd": REPO_ROOT, "dir": _rel(absdir), "entries": []}
    for name in sorted(os.listdir(absdir)):
        ap = os.path.join(absdir, name)
        out["entries"].append({
            "name": name,
            "type": "dir" if os.path.isdir(ap) else "file",
            "size": os.path.getsize(ap) if os.path.isfile(ap) else None
        })
    return out

def read_file(path: str, max_bytes: int = 1024*1024*4) -> Dict[str, Any]:
    ap = _norm(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError("File not found")
    with open(ap, "rb") as f:
        data = f.read(max_bytes+1)
    truncated = len(data) > max_bytes
    if truncated: data = data[:max_bytes]
    return {
        "path": _rel(ap),
        "size": os.path.getsize(ap),
        "truncated": truncated,
        "sha256_16": _sha256(data),
        "content": data.decode("utf-8", errors="replace")
    }

def write_file(path: str, content: str, make_dirs: bool = True, backup: bool = True) -> Dict[str, Any]:
    ap = _norm(path)
    if make_dirs:
        os.makedirs(os.path.dirname(ap), exist_ok=True)
    data = content.encode("utf-8")
    existed = os.path.exists(ap)
    old_hash = None
    if existed and backup:
        bp = _backup_path(ap)
        os.makedirs(os.path.dirname(bp), exist_ok=True)
        shutil.copy2(ap, bp)
        with open(ap, "rb") as f: old_hash = _sha256(f.read())
    with open(ap, "wb") as f:
        f.write(data)
    return {
        "path": _rel(ap),
        "bytes": len(data),
        "was_existing": existed,
        "old_hash": old_hash,
        "new_hash": _sha256(data)
    }

def delete_path(path: str) -> Dict[str, Any]:
    ap = _norm(path)
    if not os.path.exists(ap):
        return {"deleted": False, "reason": "not found"}
    # backup before delete
    bp = _backup_path(ap)
    os.makedirs(os.path.dirname(bp), exist_ok=True)
    if os.path.isdir(ap):
        shutil.copytree(ap, bp, dirs_exist_ok=True)
        shutil.rmtree(ap)
        kind = "dir"
    else:
        shutil.copy2(ap, bp)
        os.remove(ap)
        kind = "file"
    return {"deleted": True, "kind": kind, "backup": _rel(bp)}

def make_dir(path: str) -> Dict[str, Any]:
    ap = _norm(path)
    os.makedirs(ap, exist_ok=True)
    return {"created": True, "dir": _rel(ap)}

# --------- Patch helpers ----------
def json_ops(path: str, ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    ap = _norm(path)
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    doc: Any = {}
    existed = os.path.exists(ap)
    if existed:
        with open(ap, "r", encoding="utf-8") as f:
            try: doc = json.load(f)
            except Exception: doc = {}

    def _set(obj: Any, jpath: str, value: Any):
        parts = jpath.split(".")
        cur = obj
        for i, k in enumerate(parts):
            last = (i == len(parts)-1)
            if last:
                cur[k] = value
            else:
                if k not in cur or not isinstance(cur[k], dict):
                    cur[k] = {}
                cur = cur[k]

    for op in ops:
        if op.get("op") == "set":
            _set(doc, op["path"], op.get("value"))
        else:
            # Future: add/remove/merge …
            pass

    before = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
    # backup old
    if existed:
        bp = _backup_path(ap); os.makedirs(os.path.dirname(bp), exist_ok=True); shutil.copy2(ap, bp)
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    after = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
    return {"path": _rel(ap), "new_hash": _sha256(after)}

def replace_regex(path: str, pattern: str, replacement: str, count: int = 0) -> Dict[str, Any]:
    ap = _norm(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError("File not found")
    with open(ap, "r", encoding="utf-8") as f:
        txt = f.read()
    new_txt, n = re.subn(pattern, replacement, txt, count=count, flags=re.DOTALL)
    if n > 0:
        bp = _backup_path(ap); os.makedirs(os.path.dirname(bp), exist_ok=True); shutil.copy2(ap, bp)
        with open(ap, "w", encoding="utf-8") as f:
            f.write(new_txt)
    return {"path": _rel(ap), "replaced": n}

def append_if_missing(path: str, blob: str) -> Dict[str, Any]:
    ap = _norm(path)
    existed = os.path.exists(ap)
    old = ""
    if existed:
        with open(ap, "r", encoding="utf-8") as f: old = f.read()
    if blob in old:
        return {"path": _rel(ap), "appended": False}
    bp = _backup_path(ap);
    if existed:
        os.makedirs(os.path.dirname(bp), exist_ok=True); shutil.copy2(ap, bp)
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    with open(ap, "a", encoding="utf-8") as f:
        if existed and not old.endswith("\n"): f.write("\n")
        f.write(blob if blob.endswith("\n") else blob + "\n")
    return {"path": _rel(ap), "appended": True}

def insert_after(path: str, anchor: str, insert: str, once: bool = True) -> Dict[str, Any]:
    ap = _norm(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError("File not found")
    with open(ap, "r", encoding="utf-8") as f:
        txt = f.read()
    idx = txt.find(anchor)
    if idx < 0:
        return {"path": _rel(ap), "inserted": False, "reason": "anchor-not-found"}
    cut = idx + len(anchor)
    new_txt = txt[:cut] + ("\n" if not txt[cut:cut+1] == "\n" else "") + insert + "\n" + txt[cut:]
    if once and insert in txt:
        return {"path": _rel(ap), "inserted": False, "reason": "already-present"}
    bp = _backup_path(ap); os.makedirs(os.path.dirname(bp), exist_ok=True); shutil.copy2(ap, bp)
    with open(ap, "w", encoding="utf-8") as f:
        f.write(new_txt)
    return {"path": _rel(ap), "inserted": True}
