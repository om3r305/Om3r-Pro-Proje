# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, random, importlib.util, traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

try:
    from Proje1.core.logger_utils import log_event, log_brain
except Exception:
    def log_event(kind, **kw): pass
    def log_brain(tag, payload): pass

try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, **k): pass

# --- Block Forge: güvenli import + fallback ---
try:
    from Proje1.core.block_forge import (
        load_registry, ensure_block, run_block,
        propose_block_from_diag
    )
except Exception:
    # YOKSA güvenli fallback’ler (no-op ama akışı bozmadan dosya üretir)
    def load_registry() -> dict:
        return {"blocks": {}}
    def ensure_block(name: str, spec: dict, code: str) -> None:
        # fallback: sadece logs’a kaydet gibi davran
        pass
    def run_block(st, name: str, params: dict) -> dict:
        # fallback: sahte çıktılar
        return {"mid": getattr(st, "ema", lambda n: st.last_px)(20) if hasattr(st, "ema") else getattr(st, "last_px", 0.0),
                "up": None, "dn": None, "hi": None, "lo": None, "r": 50, "k": getattr(st, "last_px", 0.0),
                "o": 0.0, "atr": getattr(st, "vol_norm", 0.2), "value": 1.0}
    def propose_block_from_diag(diag: dict|None) -> tuple[str, dict, str]:
        # basit öneri
        name = f"dyn_bb_{int(time.time())}"
        spec = {"period": 20, "dev": 2.0}
        code = "# dyn block (fallback)\n"
        return name, spec, code

AUTOGEN_DIR = Path("Proje1/core/strategies_autogen")
AUTOGEN_DIR.mkdir(parents=True, exist_ok=True)
(AUTOGEN_DIR / "__init__.py").touch(exist_ok=True)

JOURNAL_PATH = Path("logs/l60_intent_journal.jsonl")
DESIGNS_PATH = Path("logs/l60_designs.jsonl")
POLICY_PATH  = Path("model/l60_policy.json")
STATS_PATH   = Path("model/l60_stats.json")

DEFAULT_POLICY = {
    "change_styles": {"punish": 0.30, "replace": 0.35, "branch": 0.35},
    "risk_bias": {
        "TREND":   {"enter": +0.10, "tp": +0.10, "sl": -0.05},
        "CHOP":    {"enter": -0.15, "tp": -0.15, "sl": +0.10},
        "MEAN":    {"enter": -0.05, "tp": -0.05, "sl": +0.05},
        "UNKNOWN": {"enter":  0.00, "tp":  0.00, "sl":  0.00}
    },
    "gate": {"min_trades": 100, "wr": 0.52, "pf": 1.20, "maxdd": -5.0},
    "throttle": {"max_new_per_day": 4},
    "meta": {"selector": "softmax", "styles_pool": ["punish","replace","branch"], "bandit": {"c": 1.2}}
}

def _read_json(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return json.loads(json.dumps(default))

def _write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _append_j(path: Path, rec: Dict[str,Any]):
    rec = {"ts": time.time(), **rec}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())

def _analyze_failures(cfg: dict) -> Dict[str, Any]:
    import csv
    path = Path((cfg.get("logging") or {}).get("trades_full_log","logs/trades_full_log.csv"))
    heat = {"slot": {}, "reg": {}}
    if not path.exists():
        return {"hot": None, "why": "no_trades"}
    neg_rows = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            ev = (r.get("event") or "").lower()
            if ev not in ("close","sell","close_all"): continue
            try:
                pnl = float((r.get("pnl") or 0) or 0)
            except Exception:
                pnl = 0.0
            if pnl >= 0: continue
            neg_rows += 1
            slot = (r.get("slot") or r.get("label") or r.get("reason") or "").lower() or "pred"
            reg  = (r.get("regime") or "UNKNOWN").upper()
            heat["slot"][slot] = heat["slot"].get(slot, 0) + abs(pnl)
            heat["reg"][reg]   = heat["reg"].get(reg, 0) + abs(pnl)
    if neg_rows == 0:
        return {"hot": None, "why": "no_loss"}
    worst_slot = max(heat["slot"], key=lambda k: heat["slot"][k]) if heat["slot"] else None
    worst_reg  = max(heat["reg"],  key=lambda k: heat["reg"][k])  if heat["reg"]  else None
    return {"hot": {"slot": worst_slot, "reg": worst_reg}, "why": "loss_hotspot", "heat": heat}

BASE_INDICATORS = [
    ("rsi",      {"period":[8,14,21,28]}),
    ("atr",      {"period":[7,14,21], "k":[1.0,1.5,2.0]}),
    ("bb",       {"period":[10,20,30], "dev":[1.5,2.0,2.5]}),
    ("donchian", {"period":[20,30,55]}),
    ("obv",      {"ma":[5,10,20]}),
    ("kama",     {"er":[10,20], "fast":[2,5], "slow":[30,50]}),
    ("super",    {"period":[7,10,14], "mult":[2.0,2.5,3.0]}),
]
ENTRY_RULES = ["cross_up","band_break_up","range_break","vol_surge"]

def _choice(lst: List[Any]) -> Any: return random.choice(lst)
def _sample_param(pdef: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {k: _choice(v) for k, v in pdef.items()}

def _design_new_graph(diag: Dict[str,Any]|None, policy: Dict[str,Any]) -> Dict[str,Any]:
    reg = ((diag or {}).get("hot") or {}).get("reg") or "UNKNOWN"
    slot = ((diag or {}).get("hot") or {}).get("slot") or "pred"

    dyn = []
    reg_blocks = load_registry().get("blocks", {})
    for bname in reg_blocks.keys():
        dyn.append((bname, {"dyn": True}))

    palette = BASE_INDICATORS + dyn
    g: Dict[str,Any] = {"slot": slot, "nodes": [], "edges": [], "meta": {"reg": reg}}
    core: List[Tuple[str,Dict[str,Any]]] = []

    if reg == "TREND":
        core.extend([("donchian", _sample_param({"period":[20,30,55]})),
                     ("kama",     _sample_param({"er":[10,20], "fast":[2,5], "slow":[30,50]})),
                     ("super",    _sample_param({"period":[7,10,14], "mult":[2.0,2.5,3.0]}))])
    elif reg == "CHOP":
        core.extend([("bb",  _sample_param({"period":[10,20,30], "dev":[1.5,2.0,2.5]})),
                     ("rsi", _sample_param({"period":[8,14,21,28]}))])
    elif reg == "MEAN":
        core.extend([("rsi", _sample_param({"period":[8,14,21]})),
                     ("bb",  _sample_param({"period":[10,20], "dev":[1.5,2.0]}))])
    else:
        core.append(_choice(palette)); core.append(_choice(palette))

    need_novel = (len([1 for n,_ in core if n not in [c[0] for c in BASE_INDICATORS]]) == 0)
    if need_novel or random.random() < 0.45:
        bname, bspec, bcode = propose_block_from_diag(diag)
        try:
            ensure_block(bname, bspec, bcode)
            core.append((bname, {"dyn":True}))
            log_brain("l60_new_block", {"name":bname, "spec":bspec})
        except Exception as e:
            log_event("l60_block_emit_fail", err=str(e))

    if reg == "TREND":
        entry, exits = "range_break", ["trail_kama","rr_static"]
    elif reg == "CHOP":
        entry, exits = "band_break_up", ["trail_bb","breakeven"]
    elif reg == "MEAN":
        entry, exits = "cross_up", ["rr_static","breakeven"]
    else:
        entry = _choice(ENTRY_RULES)
        exits = [_choice(["trail_bb","trail_kama","stop_atr","rr_static","breakeven"]), _choice(["trail_bb","rr_static"])]

    for i,(name, params) in enumerate(core):
        g["nodes"].append({"id": f"N{i}", "ind": name, "params": params})
    g["edges"] = [{"from":"N0","to":"N1"}] + ([{"from":"N1","to":"N2"}] if len(core)>=3 else [])
    g["entry"] = {"rule": entry}
    g["exit"]  = {"rules": exits}
    return g

HEADER = '''# AUTOGEN STRATEGY — L60 (no template)
# intent_id: {intent_id}
# created_ts: {ts}
# meta: {meta}
# SHADOW_MODE: {shadow}
# rationale: {rationale}
'''

CLASS_TMPL = '''
from __future__ import annotations
from typing import Dict, Any
from Proje1.core.block_forge import run_block as _dyn_block

class {cls}:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {{}}
        self.shadow = {shadow}
        self.slot = "{slot}"
        self.intent_id = "{intent_id}"
        self.meta = {meta}

    def _rsi(self, st, p): return getattr(st, "rsi", lambda n: 50)(int(p.get("period", 14)))
    def _atr(self, st, p):
        fn = getattr(st, "atr", None)
        return float(fn(int(p.get("period",14)))) if callable(fn) else max(0.001, float(getattr(st, "vol_norm", 0.2)))
    def _bb(self, st, p):
        fn = getattr(st, "bb", None)
        if callable(fn):
            return fn(int(p.get("period",20)), float(p.get("dev",2.0)))
        ema = getattr(st, "ema", lambda n: st.last_px)(int(p.get("period",20)))
        dev = float(p.get("dev",2.0))
        width = dev * max(0.001, getattr(st,"vol_norm",0.2)) * float(getattr(st,"last_px",0.0))
        return (ema, ema+width, ema-width)
    def _donchian(self, st, p):
        fn = getattr(st, "donchian", None)
        if callable(fn): return fn(int(p.get("period",20)))
        rn = max(0.001, getattr(st,"vol_norm",0.2)) * float(getattr(st,"last_px",0.0))
        return (float(getattr(st,"last_px",0.0)) + rn, float(getattr(st,"last_px",0.0)) - rn)
    def _kama(self, st, p):
        fn = getattr(st, "kama", None)
        return fn(int(p.get("er",10)), int(p.get("fast",2)), int(p.get("slow",30))) if callable(fn) else getattr(st,"ema", lambda n: st.last_px)(10)
    def _super(self, st, p):
        atr = self._atr(st, {{}})
        base = getattr(st, "ema", lambda n: st.last_px)(int(p.get("period",10)))
        mult = float(p.get("mult",2.0))
        return (base+mult*atr, base-mult*atr)
    def _obv(self, st, p):
        fn = getattr(st, "obv", None)
        return float(fn()) if callable(fn) else 0.0

    def _cross_up(self, a, b): return a > b
    def _band_break_up(self, price, upper): return price > upper
    def _range_break(self, price, hi, lo, bias=0.001): return price > (hi*(1.0-bias))
    def _vol_surge(self, st, mult=1.6): return max(0.001, getattr(st,"vol_norm",0.2)) > 1.6*0.2

    def run_for_symbol(self, sym: str, st) -> dict|None:
        if not hasattr(st, "last_px") or float(getattr(st,"last_px",0)) <= 0: return None
        px = float(st.last_px)

{nodes_block}
{entry_block}
        if not entry_signal: return None

{exit_block}
        reason = f"L60:{self.intent_id}:{self.slot}:{self.meta.get('reg')}"
        return {{"slot":"{slot}", "confidence": base_conf, "reason": reason}}
'''

def _build_nodes_code(g: Dict[str,Any]) -> str:
    lines = []
    for node in g["nodes"]:
        nid = node["id"]; ind = node["ind"]; params = node.get("params") or {}
        if params.get("dyn"):
            p = {"p": 14, "fast":5, "slow":30, "dev":2.0, "period":20, "ma":10}
            p.update(params)
            lines += [
                f'        _res_{nid} = _dyn_block(st, "{ind}", {json.dumps(p)})',
                f'        {nid}_mid = _res_{nid}.get("mid")',
                f'        {nid}_up  = _res_{nid}.get("up")',
                f'        {nid}_dn  = _res_{nid}.get("dn")',
                f'        {nid}_hi  = _res_{nid}.get("hi")',
                f'        {nid}_lo  = _res_{nid}.get("lo")',
                f'        {nid}_r   = _res_{nid}.get("r")',
                f'        {nid}_k   = _res_{nid}.get("k")',
                f'        {nid}_o   = _res_{nid}.get("o")',
                f'        {nid}_atr = _res_{nid}.get("atr")',
                f'        {nid}_v   = _res_{nid}.get("value")',
            ]
        elif ind == "bb":
            lines += [f'        {nid}_mid, {nid}_up, {nid}_dn = self._bb(st, {json.dumps(params)})']
        elif ind == "donchian":
            lines += [f'        {nid}_hi, {nid}_lo = self._donchian(st, {json.dumps(params)})']
        elif ind == "super":
            lines += [f'        {nid}_up, {nid}_dn = self._super(st, {json.dumps(params)})']
        elif ind == "atr":
            lines += [f'        {nid}_atr = self._atr(st, {json.dumps(params)})']
        elif ind == "rsi":
            lines += [f'        {nid}_r = self._rsi(st, {json.dumps(params)})']
        elif ind == "kama":
            lines += [f'        {nid}_k = self._kama(st, {json.dumps(params)})']
        elif ind == "obv":
            lines += [f'        {nid}_o = self._obv(st, {json.dumps(params)})']
        else:
            lines += [f"        # unknown node '{ind}' skipped"]
    lines += [
        "        base_conf = 0.53",
        "        if self.meta.get('reg') == 'TREND': base_conf += 0.10",
        "        if self.meta.get('reg') == 'CHOP':  base_conf -= 0.08",
        "        if self.meta.get('reg') == 'MEAN':  base_conf -= 0.02",
    ]
    return "\n".join(lines)

def _build_entry_code(g: Dict[str,Any]) -> str:
    rule = g["entry"]["rule"]
    c = ["        entry_signal = False"]
    if rule == "band_break_up":
        c += [
            "        if 'N0_up' in locals(): entry_signal = self._band_break_up(px, N0_up)",
            "        elif 'N1_up' in locals(): entry_signal = self._band_break_up(px, N1_up)",
            "        elif 'N0_hi' in locals(): entry_signal = self._range_break(px, N0_hi, N0_lo)",
        ]
    elif rule == "range_break":
        c += [
            "        if 'N0_hi' in locals(): entry_signal = self._range_break(px, N0_hi, N0_lo)",
            "        elif 'N1_hi' in locals(): entry_signal = self._range_break(px, N1_hi, N1_lo)",
        ]
    elif rule == "cross_up":
        c += [
            "        if 'N0_r' in locals(): entry_signal = self._cross_up(N0_r, 50)",
            "        elif 'N0_k' in locals(): entry_signal = px > N0_k",
            "        elif 'N0_v' in locals(): entry_signal = (N0_v or 0) > 0",
        ]
    elif rule == "vol_surge":
        c += ["        entry_signal = self._vol_surge(st)"]
    else:
        c += ["        # default no-op"]
    return "\n".join(c)

def _build_exit_code(g: Dict[str,Any]) -> str:
    lines = [
        "        tp_pct = 0.08; sl_pct = -0.06",
        "        if self.meta.get('reg') == 'CHOP': tp_pct = 0.05; sl_pct = -0.04",
        "        elif self.meta.get('reg') == 'TREND': tp_pct = 0.10; sl_pct = -0.07",
    ]
    for r in g["exit"]["rules"]:
        if r == "trail_bb":
            lines += ["        if 'N0_mid' in locals(): sl_pct = min(sl_pct, -0.05)"]
        elif r == "trail_kama":
            lines += ["        if 'N1_k' in locals() or 'N0_k' in locals(): sl_pct = min(sl_pct, -0.055)"]
        elif r == "stop_atr":
            lines += ["        if 'N0_atr' in locals(): sl_pct = min(sl_pct, -0.06)"]
        elif r == "breakeven":
            lines += ["        # breakeven ipucu; asıl mantık bot tarafında"]
        elif r == "rr_static":
            lines += ["        # sabit kalsın"]
    return "\n".join(lines)

def _emit_strategy_file(graph: Dict[str,Any], rationale: str, shadow: bool, intent_id: str) -> Path:
    cls = f"AutoL60_{int(time.time())}_{random.randint(100,999)}"
    header = HEADER.format(
        intent_id=intent_id, ts=time.time(), meta=json.dumps(graph["meta"]),
        shadow=str(bool(shadow)), rationale=rationale.replace("\n"," ")
    )
    code = CLASS_TMPL.format(
        cls=cls, intent_id=intent_id, slot=graph["slot"], meta=json.dumps(graph["meta"]),
        nodes_block=_build_nodes_code(graph),
        entry_block=_build_entry_code(graph),
        exit_block=_build_exit_code(graph),
        shadow=str(bool(shadow))
    )
    p = AUTOGEN_DIR / f"{cls}.py"
    p.write_text(header + code, encoding="utf-8")
    return p

def _dry_import(path: Path) -> Tuple[bool, str]:
    try:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        cls = [getattr(mod, k) for k in dir(mod) if k.startswith("AutoL60_")]
        if not cls: return False, "class_not_found"
        _ = cls[0]({})
        return True, "ok"
    except Exception as e:
        return False, f"import_fail:{e}"

def _append_design(file: Path, intent_id: str, graph: Dict[str,Any], rationale: str):
    _append_j(DESIGNS_PATH, {"kind":"shadow_register","file":str(file),"intent_id":intent_id,"graph":graph,"rationale":rationale})

def _promote_decision(stats: Dict[str,Any], policy: Dict[str,Any], intent_id: str) -> Optional[str]:
    gate = policy.get("gate", DEFAULT_POLICY["gate"])
    s = (stats.get("by_intent") or {}).get(intent_id)
    if not s: return None
    if s.get("trades",0) >= gate["min_trades"] and s.get("wr",0) >= gate["wr"] and s.get("pf",0) >= gate["pf"] and s.get("maxdd",0) >= gate["maxdd"]:
        return "promote"
    if s.get("trades",0) >= gate["min_trades"] and (s.get("wr",1) < gate["wr"]-0.04 or s.get("pf",1) < gate["pf"]-0.1):
        return "demote"
    return None

def _softmax_pick(weights: Dict[str,float]) -> str:
    keys = list(weights.keys())
    vals = [max(1e-6, float(weights[k])) for k in keys]
    s = sum(vals)
    r = random.random()*s
    acc = 0.0
    for k,v in zip(keys, vals):
        acc += v
        if r <= acc: return k
    return keys[-1]

def _bandit_ucb_pick(stats: Dict[str,Any], policy: Dict[str,Any]) -> str:
    pool = policy.get("meta",{}).get("styles_pool", ["punish","replace","branch"])
    C = float(policy.get("meta",{}).get("bandit",{}).get("c", 1.2))
    s = _read_json(STATS_PATH, {"styles":{}}).get("styles",{})
    total = sum([v.get("n",0) for v in s.values()]) + 1
    best, best_ucb = None, -1e9
    for a in pool:
        d = s.get(a, {"n":0,"r":0.0})
        n = d.get("n",0); r = d.get("r",0.0)
        ucb = r + C * (((total**0.5) / (1+n)) if (1+n) else 0.0)
        if ucb > best_ucb:
            best_ucb, best = ucb, a
    return best or pool[0]

def _meta_evolve(policy: Dict[str,Any], outcome: Optional[str]):
    styles = policy.get("change_styles", DEFAULT_POLICY["change_styles"]).copy()
    meta   = policy.get("meta", DEFAULT_POLICY["meta"]).copy()

    if outcome == "good":
        styles["replace"] = round(min(0.85, styles.get("replace",0.33)+0.03), 3)
        styles["branch"]  = round(min(0.85, styles.get("branch",0.33)+0.02), 3)
    elif outcome == "bad":
        styles["punish"]  = round(min(0.85, styles.get("punish",0.34)+0.02), 3)

    ssum = sum(styles.values()) or 1.0
    for k in list(styles.keys()):
        styles[k] = round(styles[k]/ssum, 3)

    pool = meta.get("styles_pool", ["punish","replace","branch"])
    if "rewrite" not in pool and random.random() < 0.25:
        pool.append("rewrite"); styles["rewrite"] = 0.15
    if "meta_shift" not in pool and random.random() < 0.10:
        pool.append("meta_shift"); styles["meta_shift"] = 0.10

    if outcome == "bad" and random.random() < 0.5:
        meta["selector"] = "bandit"
    elif outcome == "good" and random.random() < 0.3:
        meta["selector"] = "softmax"

    policy["change_styles"] = styles
    meta["styles_pool"] = pool
    policy["meta"] = meta
    _write_json(POLICY_PATH, policy)

def _pick_style(policy: Dict[str,Any]) -> str:
    meta = policy.get("meta", DEFAULT_POLICY["meta"])
    styles = policy.get("change_styles", DEFAULT_POLICY["change_styles"])
    selector = meta.get("selector","softmax")
    if selector == "bandit":
        return _bandit_ucb_pick({}, policy)
    return _softmax_pick(styles)

# -------- public API (bot.py beklediği isimler) --------
def ensure_l60(cfg: dict|None = None) -> None:
    # İlk çalıştırmada klasörleri/sayfaları hazırla
    AUTOGEN_DIR.mkdir(parents=True, exist_ok=True)
    (AUTOGEN_DIR / "__init__.py").touch(exist_ok=True)
    _ = _read_json(POLICY_PATH, DEFAULT_POLICY)
    _ = _read_json(STATS_PATH, {"by_intent":{}, "daily_counts":{}, "styles":{}})

def l60_heartbeat() -> None:
    # Şimdilik no-op (ileride throttling/health koyabiliriz)
    return

def l60_on_exception(where: str, err: Exception, cfg: dict|None = None) -> None:
    try:
        msg = f"[L60] exception @ {where}: {err}"
        log_event("l60_exception", where=where, err=str(err))
        tg_send(msg)
    except Exception:
        pass

# ------------- main tick -------------
def l60_tick(cfg: dict, ctx: dict):
    policy = _read_json(POLICY_PATH, DEFAULT_POLICY)
    stats  = _read_json(STATS_PATH, {"by_intent":{}, "daily_counts":{}, "styles":{}})

    day = _today_key()
    counts = stats.get("daily_counts", {})
    today_count = int(counts.get(day, 0))
    if today_count >= int(policy.get("throttle",{}).get("max_new_per_day",4)):
        return

    diag = _analyze_failures(cfg)
    hot = diag.get("hot") if diag.get("why") == "loss_hotspot" else None

    style = _pick_style(policy)
    if random.random() < 0.15:
        _meta_evolve(policy, outcome=None)

    graph = _design_new_graph(diag if hot else None, policy)
    intent_id = f"INTENT-{int(time.time())}-{random.randint(1000,9999)}"
    rationale = f"style={style} hot={hot} selector={policy.get('meta',{}).get('selector')}"

    p = _emit_strategy_file(graph, rationale, shadow=True, intent_id=intent_id)
    ok, why = _dry_import(p)
    if not ok:
        _append_j(JOURNAL_PATH, {"kind":"emit_fail","intent":intent_id,"why":why,"file":str(p)})
        try: tg_send(f"⚠️ L60 emit/import fail: {why}")
        except Exception: pass
        return

    _append_j(JOURNAL_PATH, {"kind":"emit_ok","intent":intent_id,"file":str(p)})
    _append_j(DESIGNS_PATH, {"kind":"shadow_register","file":str(p),"intent_id":intent_id,"graph":graph,"rationale":rationale})
    try: tg_send(f"🧪 L60 shadow strategy ready • {intent_id}")
    except Exception: pass
    log_brain("l60_emit", {"intent": intent_id, "graph": graph, "style": style})

    if random.random() < 0.1:
        _meta_evolve(policy, outcome="bad")

    counts[day] = today_count + 1
    stats["daily_counts"] = counts
    _write_json(STATS_PATH, stats)
