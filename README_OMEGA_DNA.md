# OMEGA_DNA — Tek Ana Config

## Nasıl kullanılır?
- Çalıştırırken loader’ına **tek dosya** ver:
  - `Proje1/config/OMEGA_DNA.json`
- `runtime/runtime_overrides.jsonl` **üstten patch** atar (drift/evo/brain).
- Eski json’lar (config_live.json, brain_config.json, autopilot_config.json) **dosya olarak kalsın** ama loader artık **onları okumuyor**.

## Değişenler
- `brain_hook.py`: 
  - veto_conf_min **autopatch (lo/hi)** clamp tek kaynaktan,
  - brain.adjust parametreleri güvenli **clamp**,
  - `dd_soft_usd` için **soft guard** (opsiyonel).
- `drift_watch.py`:
  - **autopatch bounds** DNA’dan okunur,
  - override akışı aynen devam.
- `bot.py`:
  - Senin son sürümün ile uyumlu (**tg_ready** kontrolü var).
