# change_engine.py — Autonomous Patch Engine (no-confirm apply)
import os, re, json, shutil, time, argparse, datetime as dt
from typing import Any, Dict, List

try:
    import yaml  # pyyaml
except Exception:
    yaml = None

BACKUP_DIR = "patch_backups"
LOG_FILE   = os.path.join("logs", "patch_engine.log")

def ts():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_text(path:str)->str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(path:str, txt:str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def json_load(path:str)->Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def json_dump(path:str, obj:Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def log(msg:str):
    ensure_dir(os.path.dirname(LOG_FILE))
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts()}] {msg}\n")
    print(msg)

def backup_file(path:str):
    ensure_dir(BACKUP_DIR)
    base = os.path.basename(path)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(BACKUP_DIR, f"{base}.{stamp}.bak")
    shutil.copy2(path, dst)
    return dst

def apply_replace_regex(path:str, pattern:str, repl:str, count:int=0)->int:
    txt = read_text(path)
    new, n = re.subn(pattern, repl, txt, count=count, flags=re.DOTALL)
    if n>0:
        backup_file(path)
        write_text(path, new)
    return n

def apply_insert_after(path:str, anchor:str, insert:str, once:bool=True)->int:
    txt = read_text(path)
    idx = txt.find(anchor)
    if idx < 0:
        return 0
    idx2 = idx + len(anchor)
    new = txt[:idx2] + insert + txt[idx2:]
    backup_file(path)
    write_text(path, new)
    return 1 if once else new.count(insert)

def apply_append_if_missing(path:str, blob:str)->int:
    txt = read_text(path)
    if blob in txt:
        return 0
    backup_file(path)
    write_text(path, txt.rstrip()+"\n"+blob+"\n")
    return 1

def json_set(obj:Any, jptr:str, value:Any):
    """
    jptr ör: rules.per_symbol_modes.ETHUSDT => obj['rules']['per_symbol_modes']['ETHUSDT']
    oluşturur/üzerine yazar
    """
    parts = jptr.split(".")
    cur = obj
    for i, p in enumerate(parts):
        last = (i == len(parts)-1)
        if last:
            cur[p] = value
        else:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]

def apply_json_ops(path:str, ops:List[Dict[str,Any]])->int:
    obj = json_load(path)
    changed = 0
    for op in ops:
        if op.get("op") == "set":
            json_set(obj, op["path"], op["value"])
            changed += 1
    if changed:
        backup_file(path)
        json_dump(path, obj)
    return changed

def apply_yaml_ops(path:str, ops:List[Dict[str,Any]])->int:
    if yaml is None:
        log("WARN: pyyaml yok, YAML kuralı atlandı.")
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    changed = 0
    for op in ops:
        if op.get("op") == "set":
            json_set(data, op["path"], op["value"])
            changed += 1
    if changed:
        backup_file(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return changed

def apply_rule(project_root:str, rule:Dict[str,Any])->int:
    target = os.path.join(project_root, rule["file"])
    if not os.path.isfile(target):
        log(f"SKIP (missing): {rule['file']}")
        return 0

    action = rule["action"]
    n = 0
    if action == "replace_regex":
        n = apply_replace_regex(
            target,
            pattern = rule["pattern"],
            repl    = rule["replacement"],
            count   = int(rule.get("count", 0))
        )
    elif action == "insert_after":
        n = apply_insert_after(
            target,
            anchor = rule["anchor"],
            insert = rule["insert"],
            once   = bool(rule.get("once", True))
        )
    elif action == "append_if_missing":
        n = apply_append_if_missing(
            target,
            blob = rule["blob"]
        )
    elif action == "json_ops":
        n = apply_json_ops(target, rule["ops"])
    elif action == "yaml_ops":
        n = apply_yaml_ops(target, rule["ops"])
    else:
        log(f"SKIP (unknown action): {action}")

    if n>0:
        log(f"APPLY {action} -> {rule['file']} (ops:{n})")
    else:
        log(f"NOCHANGE {action} -> {rule['file']}")
    return n

def load_rules(path:str)->Dict[str,Any]:
    if path.endswith(".json"):
        return json_load(path)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml gerekli (pip install pyyaml)")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        raise RuntimeError("Desteklenmeyen rules formatı (json|yaml)")

def run(project_root:str, rules_path:str, auto:bool=True)->int:
    ensure_dir("logs")
    rules = load_rules(rules_path)
    applied = 0
    for rule in rules.get("rules", []):
        # koşul kontrolü (ör: metric tabanlı) — istersen buraya genişletebiliriz
        applied += apply_rule(project_root, rule)
    log(f"DONE: {applied} değişiklik")
    return applied

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--rules", default="patch_rules.yaml")
    ap.add_argument("--auto", action="store_true")
    args = ap.parse_args()
    run(args.root, args.rules, auto=args.auto)
