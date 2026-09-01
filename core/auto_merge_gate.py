# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, json, time, re
from pathlib import Path
from typing import Dict, Any, List, Tuple

# TG sessiz fallback
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, **k): pass

INBOX   = Path("runtime/autocoder_inbox.jsonl")      # L13’ün bıraktığı patch önerileri (satır başı JSON)
APPLIED = Path("runtime/autocoder_applied.jsonl")    # uygulananlar günlüğü
BACKUPS = Path(".patch_backups")                     # güvenlik yedeği
BACKUPS.mkdir(parents=True, exist_ok=True)
APPLIED.parent.mkdir(parents=True, exist_ok=True)

# --------- helpers ---------
def _cfg(d: Dict[str,Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _f(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        try: return float(x)
        except Exception: return default

def _s(x) -> str:
    try: return str(x)
    except Exception: return ""

def _read_rows(path: Path) -> List[Dict[str,Any]]:
    if not path.exists(): return []
    out: List[Dict[str,Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append(r)
    return out

def _metrics(rows: List[Dict[str,Any]]) -> Tuple[float,float,float,int]:
    closes: List[float] = []
    for r in rows:
        side = _s(r.get("side") or r.get("event") or "")
        sideU = side.upper() if hasattr(side, "upper") else _s(side).upper()
        if sideU in ("SELL","CLOSE"):
            pnl = _f(r.get("pnl") or r.get("realized_pnl"))
            closes.append(pnl)

    n = len(closes)
    if n == 0: return 0.5, 1.0, 0.0, 0

    wins = sum(1 for x in closes if x > 0)
    wr = wins / float(n)

    gross_pos = sum(x for x in closes if x > 0) or 1e-9
    gross_neg = -sum(x for x in closes if x < 0) or 1e-9
    pf = gross_pos / gross_neg

    cum=0.0; peak=0.0; maxdd=0.0
    for x in closes:
        cum += x
        peak = max(peak, cum)
        maxdd = min(maxdd, cum - peak)
    return wr, pf, maxdd, n

def _source_path(cfg: Dict[str,Any]) -> Path:
    tfl = Path(_cfg(cfg, "logging.trades_full_log", "logs/trades_full_log.csv"))
    if tfl.exists(): return tfl
    return Path(_cfg(cfg, "paths.events_csv", "logs/events.csv"))

def _quality_gate_ok(cfg: Dict[str,Any]) -> Tuple[bool, Dict[str,float|int]]:
    look = int(_cfg(cfg, "autocoder.gate.lookback_trades", 250))
    wr_min = float(_cfg(cfg, "autocoder.gate.wr_min", 0.52))
    pf_min = float(_cfg(cfg, "autocoder.gate.pf_min", 1.20))
    dd_min = float(_cfg(cfg, "autocoder.gate.maxdd_min", -3.5))
    n_min  = int(_cfg(cfg, "autocoder.gate.min_trades", 60))

    rows = _read_rows(_source_path(cfg))
    if look > 0 and len(rows) > look:
        rows = rows[-look:]

    wr, pf, dd, n = _metrics(rows)
    ok = (n >= n_min) and (wr >= wr_min) and (pf >= pf_min) and (dd >= dd_min)
    return ok, {"wr":wr, "pf":pf, "dd":dd, "n":n, "wr_min":wr_min, "pf_min":pf_min, "dd_min":dd_min, "n_min":n_min}

def _backup_file(path: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    dest = BACKUPS / f"{path.name}.{ts}.bak"
    dest.write_bytes(path.read_bytes())
    return dest

def _apply_op(target: Path, op: Dict[str,Any]) -> bool:
    """
    Modlar:
      - replace_full:   {"mode":"replace_full", "content":"<yeni_tam_metin>"}
      - search_replace: {"mode":"search_replace", "pattern":"...", "replacement":"...", "flags":"i"}
      - append_after:   {"mode":"append_after", "anchor":"regex", "content":"..."}
    """
    txt = target.read_text(encoding="utf-8")
    mode = _s(op.get("mode")).lower()

    if mode == "replace_full":
        new_txt = _s(op.get("content",""))
    elif mode == "search_replace":
        flags = re.IGNORECASE if "i" in _s(op.get("flags","")).lower() else 0
        new_txt = re.sub(op.get("pattern",""), op.get("replacement",""), txt, flags=flags)
    elif mode == "append_after":
        m = re.search(op.get("anchor",""), txt, flags=re.MULTILINE)
        if not m:
            new_txt = txt + _s(op.get("content",""))
        else:
            idx = m.end()
            new_txt = txt[:idx] + _s(op.get("content","")) + txt[idx:]
    else:
        new_txt = txt

    if new_txt != txt:
        _backup_file(target)
        target.write_text(new_txt, encoding="utf-8")
        return True
    return False

def _read_inbox(limit:int) -> List[Dict[str,Any]]:
    if not INBOX.exists(): return []
    out=[]
    with INBOX.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except Exception: continue
            if len(out)>=limit: break
    return out

def _consume(consumed: List[Dict[str,Any]]) -> None:
    if not INBOX.exists() or not consumed: return
    all_lines = [json.loads(l) for l in INBOX.read_text(encoding="utf-8").splitlines() if l.strip()]
    left = [j for j in all_lines if j not in consumed]
    INBOX.write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in left), encoding="utf-8")

def _append_applied(rec: Dict[str,Any]) -> None:
    rec = {"ts": time.time(), **rec}
    with APPLIED.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

_COOL_UNTIL = 0.0

def tick(cfg: Dict[str,Any], ctx: Dict[str,Any] | None = None) -> None:
    global _COOL_UNTIL
    cool_sec = int(_cfg(cfg, "autocoder.gate.cooldown_sec", 300))
    if time.time() < _COOL_UNTIL:
        return

    ok, met = _quality_gate_ok(cfg)
    if not ok:
        try:
            tg_send(
                f"🧪 <b>Quality-Gate</b> kapalı: wr={met['wr']:.2f} (≥{met['wr_min']:.2f}) • "
                f"pf={met['pf']:.2f} (≥{met['pf_min']:.2f}) • dd={met['dd']:.2f} (≥{met['dd_min']:.2f}) • "
                f"n={int(met['n'])} (≥{int(met['n_min'])})",
                parse_mode="HTML"
            )
        except Exception: pass
        _COOL_UNTIL = time.time() + cool_sec
        return

    batch = _read_inbox(limit=int(_cfg(cfg, "autocoder.gate.max_apply_per_tick", 1)))
    if not batch:
        _COOL_UNTIL = time.time() + cool_sec
        return

    for pr in batch:
        target = Path(_s(pr.get("target","")))
        ops    = pr.get("ops") or []
        reason = _s(pr.get("reason","autocoder"))
        if (not target.exists()) or (not ops):
            continue

        changed=False
        for op in ops:
            try:
                changed = _apply_op(target, op) or changed
            except Exception as e:
                try: tg_send(f"❌ Auto-Merge hata: {target} • {e}")
                except Exception: pass

        _append_applied({"target": str(target), "reason": reason, "ops": ops, "changed": changed, "metrics": met})
        try:
            if changed:
                tg_send(
                    f"✅ <b>Auto-Merge</b> uygulandı • <code>{target}</code>\n"
                    f"Sebep: <i>{reason}</i>\n"
                    f"Gate: wr={met['wr']:.2f}, pf={met['pf']:.2f}, dd={met['dd']:.2f}, n={int(met['n'])}",
                    parse_mode="HTML"
                )
            else:
                tg_send(f"ℹ️ Auto-Merge: {target} için değişecek bir şey yoktu (no-op).")
        except Exception: pass

    _consume(batch)
    _COOL_UNTIL = time.time() + cool_sec
