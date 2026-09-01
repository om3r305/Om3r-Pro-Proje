# tools_runtime.py — MetaBrain araçları: metrics, override, patch, dosya, web
import os, json, time, csv, requests
from typing import Dict, Any
from patch_applier import apply_patch

LOG_DIR = "logs"
EVENTS_CSV = os.path.join(LOG_DIR, "events.csv")

class Tools:
    def __init__(self,
                 events_csv: str = EVENTS_CSV,
                 overrides_path: str = "runtime_overrides.json",
                 state_path: str = "runtime_state.json",
                 config_path: str = "config_live.json"):
        self.events_csv = events_csv
        self.overrides_path = overrides_path
        self.state_path = state_path
        self.config_path = config_path

    # ---------- READ ----------
    def get_status(self) -> Dict[str, Any]:
        out = {"cash": None, "open_positions": 0, "by_symbol": {}}
        try:
            if os.path.exists(self.state_path):
                j = json.load(open(self.state_path, "r", encoding="utf-8"))
                out["cash"] = j.get("cash")
                pos = j.get("positions", {})
                cnt = 0
                for s, slots in pos.items():
                    n = sum(1 for v in slots.values() if v)
                    if n>0:
                        out["by_symbol"][s] = n
                        cnt += n
                out["open_positions"] = cnt
        except Exception as e:
            out["error"] = f"state read err: {e}"
        return out

    def get_metrics(self, window_min: int = 120) -> Dict[str, Any]:
        rows = []
        cutoff = time.time() - window_min*60
        try:
            if os.path.exists(self.events_csv):
                with open(self.events_csv, "r", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for x in r:
                        try:
                            ts = float(x["ts"])
                        except:
                            continue
                        if ts >= cutoff and x.get("kind","").lower()=="close":
                            rows.append(x)
        except Exception as e:
            return {"ok": False, "msg": f"metrics read err: {e}"}

        n = len(rows)
        pnl = [float(x.get("pnl", 0.0)) for x in rows]
        wins = [p for p in pnl if p>0]
        losses = [-p for p in pnl if p<0]
        pf = (sum(wins)/sum(losses)) if sum(losses)>0 else (float("inf") if sum(wins)>0 else 0.0)
        wr = (len(wins)/n) if n>0 else 0.0
        cum=0.0; peak=0.0; maxdd=0.0
        for p in pnl:
            cum += p
            peak = max(peak, cum)
            maxdd = max(maxdd, peak - cum)

        by_slot = {}
        for x in rows:
            sl = x.get("slot","-")
            by_slot.setdefault(sl, {"n":0,"p":0.0,"w":0})
            by_slot[sl]["n"] += 1
            by_slot[sl]["p"] += float(x.get("pnl",0))
            if float(x.get("pnl",0))>0: by_slot[sl]["w"] += 1

        by_coin = {}
        for x in rows:
            s = x.get("sym","-")
            by_coin.setdefault(s, {"n":0,"p":0.0,"w":0})
            by_coin[s]["n"] += 1
            by_coin[s]["p"] += float(x.get("pnl",0))
            if float(x.get("pnl",0))>0: by_coin[s]["w"] += 1

        return {
            "ok": True,
            "n": n,
            "pnl": sum(pnl),
            "pf": pf,
            "winrate": wr,
            "max_dd": maxdd,
            "by_slot": by_slot,
            "by_coin": by_coin
        }

    def read_config(self) -> Dict[str, Any]:
        try:
            return {"ok": True, "config": json.load(open(self.config_path,"r",encoding="utf-8"))}
        except Exception as e:
            return {"ok": False, "msg": f"config read err: {e}"}

    def read_file(self, path: str, max_bytes: int = 150_000) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {"ok": False, "msg": "file not found"}
        try:
            data = open(path, "r", encoding="utf-8").read()
            if len(data)>max_bytes:
                data = data[:max_bytes] + "\n\n...<truncated>..."
            return {"ok": True, "path": path, "content": data}
        except Exception as e:
            return {"ok": False, "msg": f"read err: {e}"}

    # ---------- WRITE ----------
    def write_override(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with open(self.overrides_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return {"ok": True, "msg": "override written"}
        except Exception as e:
            return {"ok": False, "msg": f"override write err: {e}"}

    def apply_patch(self, file_path: str, pattern: str, repl: str) -> Dict[str, Any]:
        return apply_patch(file_path, pattern, repl)

    def create_file(self, path: str, content: str) -> Dict[str, Any]:
        if os.path.exists(path):
            return {"ok": False, "msg": "file already exists"}
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"ok": True, "msg": f"file {path} created"}
        except Exception as e:
            return {"ok": False, "msg": str(e)}

    # ---------- WEB ----------
    def web_get(self, url: str, timeout: int = 12, max_chars: int = 6000) -> Dict[str, Any]:
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent":"MetaBrain/1.0"})
            txt = r.text
            if len(txt)>max_chars:
                txt = txt[:max_chars] + "\n\n...<truncated>..."
            return {"ok": True, "status": r.status_code, "text": txt}
        except Exception as e:
            return {"ok": False, "msg": f"web err: {e}"}
