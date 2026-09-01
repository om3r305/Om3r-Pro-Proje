# autopilot.py — Monster Coins Pro (full, updated)
# © Kanka • 05.09.2025
# İşlev: trades.csv’den hızlı metrikleri topla -> (opsiyonel) LLM’e plan yazdır ->
#        patch_rules.yaml uygulanır (patch_applier varsa full, yoksa mini).
# Not: OpenAI anahtarı boş/dummy ise otomatik OLLAMA'ya düşer.

from __future__ import annotations
import os, re, json, csv, time, argparse, datetime, traceback
from typing import Any, Dict, List, Optional

# ---- opsiyonel bağımlılıklar (yoksa sorun yapmasın) ----
try:
    import yaml   # pyyaml
except Exception:
    yaml = None

try:
    import requests
except Exception:
    requests = None

# ---- dış modüller (opsiyonel) ----
try:
    from llm_engine import LLMEngine  # Ollama/OpenAI fallback motoru
except Exception:
    LLMEngine = None  # yoksa LLM kullanmadan devam ederiz

try:
    from patch_applier import apply_rules   # full patch motoru
except Exception:
    apply_rules = None                      # yoksa mini'ye düşeriz

CFG_PATH_DEFAULT = "config_live.json"
TRADES_CSV_DEFAULT = os.path.join("logs", "trades.csv")


# ==============================
# Yardımcılar
# ==============================
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def now_utc_str(fmt: str = "%Y-%m-%d %H:%M:%S"):
    # utcnow() deprecated uyarısından kaçın
    return datetime.datetime.now(datetime.UTC).strftime(fmt)


# ==============================
# Metrik Okuma (hızlı)
# ==============================
def read_quick_metrics(
    path: str = TRADES_CSV_DEFAULT,
    lookback_hours: int = 6
) -> Dict[str, float]:
    """
    logs/trades.csv (ts, slot, sym, pnl, ...) bekler.
    Hata halinde metrikleri 0 döndürür (fail-safe).
    """
    out = {"trades": 0, "pnl_sum": 0.0, "win": 0, "loss": 0, "avg_pnl": 0.0, "pf": 1.0, "maxdd": 0.0}
    try:
        if not os.path.exists(path):
            return out

        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=lookback_hours)
        bal = 0.0
        peak = 0.0
        min_equity = 0.0
        pnl_sum = 0.0
        wins = 0
        losses = 0
        n = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_raw = row.get("ts") or row.get("time") or ""
                try:
                    # ISO tarih bekliyoruz; fallback olarak parse etmeyip atla
                    ts = datetime.datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    continue

                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.UTC)

                if ts < cutoff:
                    continue

                # pnl sütunu
                try:
                    pnl = float(row.get("pnl", "0") or 0)
                except Exception:
                    pnl = 0.0

                n += 1
                pnl_sum += pnl
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1

                bal += pnl
                peak = max(peak, bal)
                min_equity = min(min_equity, bal)

        out["trades"] = n
        out["pnl_sum"] = round(pnl_sum, 4)
        out["win"] = wins
        out["loss"] = losses
        out["avg_pnl"] = round(pnl_sum / n, 6) if n else 0.0
        # kaba PF tahmini (negatif toplamı 1 kabul ederek blow-up'ları şişirmeyelim)
        neg = abs(min(0.0, pnl_sum))
        pos = max(0.0, pnl_sum)
        out["pf"] = round((pos + 1e-6) / (neg + 1e-6), 3)
        out["maxdd"] = round(min_equity, 4)  # USD cinsinden aşağı sarkma
    except Exception as e:
        print("[AutoPilot] trades.csv okunamadı:", e)
    return out


# ==============================
# Planlama (opsiyonel LLM)
# ==============================
PLAN_SCHEMA = """\
You are MetaBrain, an expert trading autopilot for a crypto scalper bot.
Given quick metrics, produce a short actionable plan in Turkish bullet points.
Be concrete. Then list the patch files you expect to change.
JSON response with keys: "notes" (string), "suggest" (string[]), "files" (string[]).
"""

def make_plan_with_llm(cfg: Dict[str, Any], quick: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """
    LLMEngine varsa (Ollama/OpenAI), JSON plan üretmeyi dener.
    Yoksa None döner.
    """
    if LLMEngine is None:
        return None
    try:
        engine = LLMEngine(cfg)
        prompt = (
            PLAN_SCHEMA + "\n\n"
            + json.dumps({"metrics": quick}, ensure_ascii=False, indent=2)
        )
        raw = engine.ask(prompt, max_tokens=800) or ""
        # cevabı JSON yakalamaya çalış
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return {"notes": raw.strip(), "suggest": [], "files": []}
        plan = json.loads(m.group(0))
        return plan
    except Exception as e:
        print("[LLM] plan üretilemedi:", e)
        return None


# ==============================
# Patch Uygulama
# ==============================
def _mini_apply_rules(rules_yaml: str, dry: bool = True) -> List[str]:
    """
    Basit/uyumluluk modu: sadece bilgi mesajı döner.
    Projede patch_applier yoksa bloklamasın.
    """
    return ["[Patch] (mini) devrede — patch_applier.py bulunamadı; hiç bir dosya değiştirilmedi."]


def run_patch_engine(rules_yaml: str, dry: bool) -> None:
    print("[AutoPilot] Yamalar uygulanıyor...")
    if apply_rules is not None:
        logs = apply_rules(root=".", rules_yaml=rules_yaml, dry=dry)
        for line in logs:
            print(line)
    else:
        for line in _mini_apply_rules(rules_yaml, dry):
            print(line)
    print("[AutoPilot] bitti. 1800s bekleyip tekrar çalıştırabilirsiniz.")


# ==============================
# CLI & Main
# ==============================
def parse_args():
    ap = argparse.ArgumentParser(description="Monster Coins Pro - AutoPilot")
    ap.add_argument("--config", default=CFG_PATH_DEFAULT, help="config_live.json yolu")
    ap.add_argument("--rules",  default="patch_rules.yaml", help="patch kuralları yaml")
    ap.add_argument("--dry-run", action="store_true", help="dosyaları değiştirmeden deneme")
    ap.add_argument("--lookback-hours", type=int, default=6, help="metrik bakış penceresi (saat)")
    return ap.parse_args()


def main():
    args = parse_args()
    rules_path = args.rules
    dry = bool(args.dry_run)
    cfg_path = args.config
    lookback = int(args.lookback_hours)

    try:
        cfg = load_json(cfg_path)
    except Exception as e:
        print(f"[FATAL] config yüklenemedi: {cfg_path} -> {e}")
        return

    print(f"[{now_utc_str()}] AutoPilot started (cooldown=1800s, dry={dry})")
    # LLM sağlayıcı bilgisini logla (fallback lm_engine içinde)
    provider = "openai"
    if not (cfg.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")):
        provider = "ollama"
    model = cfg.get("BRAIN_MODEL", cfg.get("brain_model", "llama2:70b"))
    print(f"[LLM] provider: {provider} | model: {model}")

    # 1) hızlı metrikler
    quick = read_quick_metrics(TRADES_CSV_DEFAULT, lookback)
    print(f"  -> metrics (last {lookback}h): trades={quick['trades']} pnl={quick['pnl_sum']} "
          f"wr~={(quick['win']/(quick['trades'] or 1)):.2f} pf={quick['pf']} maxdd={quick['maxdd']}")

    # 2) (opsiyonel) plan iste
    plan = make_plan_with_llm(cfg, quick)
    if plan:
        print("\n* Uygulanacak Kurallar (Applicable Rules):")
        print("  - Notlar:", (plan.get("notes") or "").strip()[:400])
        suggest = plan.get("suggest") or []
        for s in suggest[:6]:
            print("   •", s)
        files = plan.get("files") or []
        if files:
            print("  - Dokunulacak dosyalar:", ", ".join(files[:10]))
        print()

    # 3) patch uygula
    run_patch_engine(rules_path, dry)

    # 4) cool down (autopilot döngüsü dışarıdan sistemd/pm2 ile tetiklenebilir)
    # time.sleep(1800)


if __name__ == "__main__":
    main()
