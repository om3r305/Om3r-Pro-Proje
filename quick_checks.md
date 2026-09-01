
# Quick Checks (5 minutes)
1) Başlat:  python -m Proje1.main --config Proje1/config_live.json
2) 1–2 dk sonra dosyalar oluşmalı:
   - logs/web_audit.jsonl, logs/market_intel.jsonl, logs/telemetry_kpi.txt
3) Python kabuğunda test:
   >>> from Proje1.modules.news_fetcher import fetch_once; fetch_once()
   >>> from Proje1.core.reporting_kpi import write_snapshot; print(write_snapshot())
   >>> from Proje1.modules.meta_label_dyn import suggest_veto; print(suggest_veto())
   >>> from Proje1.modules.counterfactual_v2 import enrich_from_trades; enrich_from_trades()
4) Gölge/Arena:
   >>> from Proje1.core.shadow_arena import simulate_configs; simulate_configs()
   >>> from Proje1.core.alpha_forge import evolve_step; evolve_step()
5) Risk guard:
   >>> from Proje1.modules.equity_guard import guard; guard()
