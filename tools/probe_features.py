# tools/probe_features.py
import json, sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # proje kökünü PYTHONPATH'e ekle

from Proje1.core.feature_registry import register_defaults, bootstrap_from_cfg, snapshot, snapshot_text

def _load_cfg() -> dict:
    # config_live.json veya config.json arasından ilk bulduğunu yükle
    for name in ("config_live.json", "config.json"):
        if os.path.exists(name):
            with open(name, "r", encoding="utf-8") as f:
                import json
                return json.load(f)
    return {}

if __name__ == "__main__":
    register_defaults()
    cfg = _load_cfg()

    # EV filtresi gibi cfg kökündeki basit bayraklar:
    # örn: cfg["use_ev_filter"] -> feature "ev_filter".enabled ayarı için:
    if cfg.get("use_ev_filter") is not None:
        # küçük köprü: registry “ev_filter” için enabled = True/False yapacağız
        # bootstrap sonrası reason alanı “ok/disabled via cfg” dolduruluyor
        pass

    bootstrap_from_cfg(cfg)
    mode = (sys.argv[1] if len(sys.argv) > 1 else "pretty").lower()

    if mode == "json":
        print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
    else:
        print(snapshot_text())
        print("\nİpucu: JSON için `python tools/probe_features.py json`")
