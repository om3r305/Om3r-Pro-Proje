# meta_brain_v2.py — Online Deney Motoru: A/B/C politika dener, kazananı büyütür
import os, csv, json, time, math
from collections import defaultdict, deque
from datetime import datetime, timedelta

POLICY_SET = {
    # Her politika ufak farklı: portföy, entry, predictor ve hız
    "A": {
        "portfolio": {"dip":0.55,"pred":0.30,"news":0.10,"ob":0.05},
        "entry_frac":{"dip":0.70,"pred":0.55,"news":0.60,"ob":0.40},
        "predictor":{"enter_prob":0.66},
        "freq_ctrl":{"min_sec_between_trades":18,"max_trades_per_hour":80},
        "sizing":{"max_mult":2.2}
    },
    "B": {
        "portfolio": {"dip":0.40,"pred":0.40,"news":0.15,"ob":0.05},
        "entry_frac":{"dip":0.60,"pred":0.60,"news":0.60,"ob":0.40},
        "predictor":{"enter_prob":0.70},
        "freq_ctrl":{"min_sec_between_trades":22,"max_trades_per_hour":70},
        "sizing":{"max_mult":2.0}
    },
    "C": {
        "portfolio": {"dip":0.35,"pred":0.25,"news":0.30,"ob":0.10},
        "entry_frac":{"dip":0.55,"pred":0.50,"news":0.70,"ob":0.40},
        "predictor":{"enter_prob":0.62},
        "freq_ctrl":{"min_sec_between_trades":25,"max_trades_per_hour":60},
        "sizing":{"max_mult":1.8}
    }
}

CFG = {
    "events_csv": "logs/events.csv",
    "override_path": "runtime_overrides.json",
    "window_min": 120,          # performans penceresi
    "update_every_min": 10,     # kaç dakikada bir ağırlık/override güncelle
    "min_trades": 8,            # politika başına min kapanış lazım
    "softmax_temp": 0.8,        # skor→ağırlık dönüşümü
    "explore_eps": 0.15,        # %15 keşif: bazen 2. seçeneği dener
    "guard": {                  # güvenlik kelepçeleri
        "pf_floor": 0.70,
        "wr_floor": 0.28,
        "maxdd_cap_usd": 8.0
    }
}

def _read_recent(csv_path, window_min):
    if not os.path.exists(csv_path): return []
    cut = datetime.utcnow() - timedelta(minutes=window_min)
    out=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ts = datetime.fromtimestamp(float(row["ts"]))
            except:
                try:
                    ts = datetime.fromisoformat(row["ts"])
                except:
                    continue
            if ts >= cut and (row.get("kind","").lower()=="close"):
                out.append(row)
    return out

def _score_policy(rows, pid):
    sel = [x for x in rows if (x.get("policy_id") or "base")==pid]
    n = len(sel)
    if n==0:
        return {"n":0,"pnl":0.0,"pf":0.0,"wr":0.0,"maxdd":0.0,"score":-1e9}
    pnl = [float(x.get("pnl",0.0)) for x in sel]
    wins = [p for p in pnl if p>0]
    losses = [-p for p in pnl if p<0]
    pf = (sum(wins)/sum(losses)) if sum(losses)>0 else (float("inf") if sum(wins)>0 else 0.0)
    wr = (len(wins)/n) if n>0 else 0.0
    # maxdd
    cum=0.0; peak=0.0; maxdd=0.0
    for p in pnl:
        cum += p
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    # skor: PF’yı ve wr’yi ödüllendir, maxdd’yi cezalandır
    pf_clip = min(3.0, (pf if pf!=float("inf") else 3.0))
    score = 1.2*pf_clip + 0.8*wr - 0.08*maxdd
    return {"n":n,"pnl":sum(pnl),"pf":pf,"wr":wr,"maxdd":maxdd,"score":score}

def _softmax(scores, temp):
    # scores: dict pid→score
    import math
    keys=list(scores.keys())
    vals=[scores[k] for k in keys]
    mx=max(vals) if vals else 0.0
    ex=[math.exp((v-mx)/max(1e-9,temp)) for v in vals]
    s=sum(ex) or 1.0
    return {k: ex[i]/s for i,k in enumerate(keys)}

def _choose(weights, eps):
    # eps-greedy: %eps oranında ikinciyi seç
    ranked = sorted(weights.items(), key=lambda x:-x[1])
    if len(ranked)==0: return "A"
    top = ranked[0][0]
    if len(ranked)>1 and (random.random()<eps):
        return ranked[1][0]
    return top

def _guard_ok(metrics, guard):
    pf = metrics["pf"]; wr = metrics["wr"]; dd = metrics["maxdd"]
    if pf!=float("inf") and pf < guard["pf_floor"]: return False
    if wr < guard["wr_floor"]: return False
    if dd > guard["maxdd_cap_usd"]: return False
    return True

def _write_override(pid):
    obj = {"policy_id": pid}
    obj.update(POLICY_SET[pid])
    with open(CFG["override_path"],"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[MBv2] override → policy {pid}")

def main():
    last = 0
    while True:
        try:
            now = time.time()
            if now - last < CFG["update_every_min"]*60:
                time.sleep(2); continue
            last = now

            rows = _read_recent(CFG["events_csv"], CFG["window_min"])

            # Her politika için metrik + skor
            metrics = {}
            for pid in POLICY_SET.keys():
                m = _score_policy(rows, pid)
                metrics[pid]=m

            # Yeterli veri yoksa rotasyon yap (A→B→C→A…)
            if all(metrics[pid]["n"] < CFG["min_trades"] for pid in POLICY_SET.keys()):
                current = None
                try:
                    if os.path.exists(CFG["override_path"]):
                        current = json.load(open(CFG["override_path"],"r",encoding="utf-8")).get("policy_id")
                except: pass
                order = ["A","B","C"]
                nxt = order[(order.index(current)+1)%3] if current in order else "A"
                _write_override(nxt);
                continue

            # Softmax ile ağırlık çıkar
            sm = _softmax({pid:metrics[pid]["score"] for pid in POLICY_SET.keys()}, CFG["softmax_temp"])
            # En iyi + sağlıklı olanı seç
            ranked = sorted(POLICY_SET.keys(), key=lambda k: -sm.get(k,0.0))
            choose_from = [pid for pid in ranked if _guard_ok(metrics[pid], CFG["guard"])]
            if not choose_from:
                # herkes kötü → en az kötü olan
                choose_from = [ranked[0]]
            chosen = _choose({pid:sm[pid] for pid in choose_from}, CFG["explore_eps"])
            _write_override(chosen)

            # Konsol özet
            for pid in POLICY_SET.keys():
                m=metrics[pid]
                print(f"[{pid}] n={m['n']} pnl={m['pnl']:+.2f} pf={(m['pf'] if m['pf']!=float('inf') else 999):.2f} wr={m['wr']*100:.1f}% dd={-m['maxdd']:.2f} sc={m['score']:.2f}")
            print("weights:", sm, "chosen:", chosen)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("MetaBrain v2 err:", e)
        time.sleep(2)

if __name__=="__main__":
    main()
