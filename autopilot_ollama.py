# autopilot_ollama.py — Monster Coins Pro • Ollama entegre AutoPilot
# Çalışma: Her 10 dakikada bir son istatistikleri toparlar, Ollama’ya sorar,
# JSON yanıtı patch önerisine çevirir, patch'i uygular (dry_run=false ise).
# Not: %100 kazanç garanti edilemez; hedef ve risk yönetimi optimize edilir.

import os, time, json, csv, datetime, traceback, requests
from trader_prompt import build_trader_prompt

CFG_PATH = os.environ.get("MCP_CONFIG", "config_live.json")
PATCH_LOG = os.environ.get("MCP_AP_LOG", "autopilot_log.jsonl")

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

# --------------- I/O helpers ---------------
def load_config(path=CFG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg, path=CFG_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def log_jsonl(obj, path=PATCH_LOG):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_csv_rows(path, limit=2000):
    rows=[]
    if not os.path.exists(path): return rows
    with open(path, "r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows[-limit:]

# --------------- metrics snapshot ---------------
def snapshot_metrics(cfg):
    # trades.csv & events.csv opsiyonel
    log_dir = cfg.get("logging", {}).get("dir", "logs")
    tcsv    = cfg.get("logging", {}).get("trade_csv", os.path.join(log_dir, "trades.csv"))
    ecsv    = cfg.get("logging", {}).get("events_csv", os.path.join(log_dir, "events.csv"))

    trades = read_csv_rows(tcsv, limit=2000)
    # basit PF/WR hesap
    pnl = [float(x.get("pnl", 0.0)) for x in trades]
    gains = sum(x for x in pnl if x>0)
    losses = -sum(x for x in pnl if x<0)
    pf = (gains / losses) if losses>0 else (float("inf") if gains>0 else 0.0)
    wins = sum(1 for x in pnl if x>0)
    wr = (wins/len(pnl)*100.0) if pnl else 0.0

    # maxDD (kümülatif)
    cum=0.0; peak=0.0; maxdd=0.0
    for x in pnl:
        cum += x; peak = max(peak, cum); maxdd = max(maxdd, peak - cum)
    maxdd = -maxdd

    # slot/coin kaba istatistikleri
    slot_stats = {}
    coin_stats = {}
    for row in trades:
        slot = row.get("slot","?")
        sym  = row.get("sym","?")
        p    = float(row.get("pnl",0.0))
        if slot not in slot_stats: slot_stats[slot]={"p":0.0,"n":0,"w":0}
        slot_stats[slot]["p"]+=p; slot_stats[slot]["n"]+=1; slot_stats[slot]["w"]+=1 if p>0 else 0
        if sym not in coin_stats: coin_stats[sym]={"p":0.0,"n":0,"w":0}
        coin_stats[sym]["p"]+=p;  coin_stats[sym]["n"]+=1; coin_stats[sym]["w"]+=1 if p>0 else 0

    metrics = {
        "time": datetime.datetime.now().isoformat(),
        "count": len(pnl),
        "pf": (None if pf==float("inf") else pf),
        "winrate": wr,
        "maxdd": maxdd,
        "slot_stats": slot_stats,
        "coin_stats": coin_stats
    }
    recent = {
        "last_trades": trades[-10:],
        "open_positions": 0  # canlıdan okunmuyorsa 0 geç
    }
    return metrics, recent

# --------------- Ollama ask ---------------
def ask_ollama(model: str, prompt: str, host: str = OLLAMA_HOST, timeout=90):
    url = host.rstrip("/") + "/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    r = requests.post(url, json=data, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j.get("response","").strip()

def parse_llm_json(s: str):
    # Son satırda JSON bekliyoruz; değilse güvenli fallback
    import json, re
    try:
        # En sondaki {...} bloğunu yakala
        m = re.findall(r"\{[\s\S]*\}$", s)
        if m:
            return json.loads(m[-1])
        return json.loads(s)
    except Exception:
        return {"action":"keep","explain":"parse_fail","_raw":s}

# --------------- Patch uygulama ---------------
def apply_patch_to_config(cfg, patch: dict):
    # weights -> portfolio
    w = (patch.get("weights") or {})
    if w:
        portfolio = cfg.get("portfolio", {})
        for k in ("dip","pred","news","ob"):
            if k in w: portfolio[k]=float(w[k])
        # normalize
        sm = sum(portfolio.values()) or 1.0
        for k in portfolio: portfolio[k] = round(portfolio[k]/sm, 4)
        cfg["portfolio"]=portfolio

    # risk
    rk = patch.get("risk") or {}
    if "max_daily_loss_usd" in rk:
        cfg.setdefault("risk", {})["daily_max_loss_usd"] = float(rk["max_daily_loss_usd"])

    # tpsl
    tpsl = patch.get("tpsl") or {}
    rr = cfg.get("rules", {})
    if "rules" in tpsl:
        rr_abs = tpsl["rules"].get("abs") or {}
        rr_pct = tpsl["rules"].get("pct") or {}
        if "tp" in rr_abs: rr.setdefault("abs", {})["tp"] = float(rr_abs["tp"])
        if "sl" in rr_abs: rr.setdefault("abs", {})["sl"] = float(rr_abs["sl"])
        if "tp" in rr_pct: rr.setdefault("pct", {})["tp"] = float(rr_pct["tp"])
        if "sl" in rr_pct: rr.setdefault("pct", {})["sl"] = float(rr_pct["sl"])
    if "dynamic_tpsl" in tpsl:
        rr["dynamic_tpsl"] = tpsl["dynamic_tpsl"]
    cfg["rules"]=rr

    # freq_ctrl
    fq = patch.get("freq_ctrl") or {}
    if fq:
        cfg.setdefault("freq_ctrl", {})
        for k in ("min_sec_between_trades","max_trades_per_hour"):
            if k in fq: cfg["freq_ctrl"][k]=int(fq[k])

    # guards
    gd = patch.get("guards") or {}
    if gd:
        cfg.setdefault("candles", {})
        if "candles_strict" in gd:     cfg["candles"]["strict"] = bool(gd["candles_strict"])
        if "exit_on_bearish" in gd:    cfg["candles"]["exit_on_bearish"] = bool(gd["exit_on_bearish"])

    # per-symbol modes
    smodes = patch.get("symbol_modes") or {}
    if smodes:
        cfg.setdefault("rules", {}).setdefault("per_symbol_modes", {})
        cfg["rules"]["per_symbol_modes"].update(smodes)

    return cfg

# --------------- main loop ---------------
def main():
    cfg = load_config()
    ap = cfg.get("autopilot", {})
    if not ap.get("enabled", True):
        print("[AP] disabled in config"); return

    dry  = bool(ap.get("dry_run", False))
    step = int(os.environ.get("AP_PERIOD_SEC", "600"))  # 10 dakika
    print(f"[AP] Ollama model={OLLAMA_MODEL} period={step}s dry_run={dry}")

    while True:
        try:
            cfg = load_config()
            metrics, recent = snapshot_metrics(cfg)
            prompt = build_trader_prompt(metrics=metrics, recent=recent, config=cfg)
            resp_txt = ask_ollama(OLLAMA_MODEL, prompt)
            out = parse_llm_json(resp_txt)

            log_jsonl({
                "ts": datetime.datetime.now().isoformat(),
                "metrics": metrics,
                "llm_raw": resp_txt,
                "llm_json": out
            })

            act = (out.get("action") or "keep").lower()
            print("[AP] action:", act, "| explain:", out.get("explain"))
            if act == "patch":
                new_cfg = apply_patch_to_config(cfg, out)
                if dry:
                    print("[AP] dry_run=True — patch uygulanmadı (sadece log).")
                else:
                    save_config(new_cfg)
                    print("[AP] patch uygulandı ve config_live.json güncellendi.")
            time.sleep(step)
        except KeyboardInterrupt:
            print("Çıkılıyor..."); break
        except Exception as e:
            print("AP hata:", e)
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()
