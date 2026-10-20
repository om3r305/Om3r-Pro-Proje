# Açık depo gizli bilgi taraması — 21 Ekim 2026

Kapsam: `om3r305/Om3r-Pro-Proje` (public).
- Bütün dalların bütün geçmişi tarandı: 2.916 commit, 223 dal. Yalnızca **eklenen satırlara** bakıldı.
- Yöntem: kendi yazdığımız, düzenli ifade tabanlı bir tarayıcı (`scratchpad/secret_scan.py`, depoya eklenmedi).
- Aranan kalıplar: JWT, `sb_secret_`/`sb_publishable_`, GitHub, OpenAI, Anthropic, AWS, Google, Telegram, Slack ve Stripe anahtarları, özel anahtar blokları, şifreli Postgres URL'leri, `key/secret/token/password = "…"` atamaları, anahtar yakınında 64 karakter hex.
- Hiçbir değer tam olarak yazdırılmadı.

## Sonuç: gerçek (çalışan) bir gizli bilgi bulunmadı

| Bulgu | Adet | Değerlendirme |
|---|---|---|
| `sb_secret_…` (tests/) | 37 değer | Hepsi uydurma test verisi: alfabe dizileri ya da `…_never_echo`, `phase98`, `superproject` gibi kelimeler. Gerçek CORE anahtarı değil |
| `sb_publishable_…` (frontend) | 1 | Supabase'in yayımlanabilir anahtarı; tarayıcıda olması tasarım gereği |
| `mcp-da…` "DASHBOARD_KEY" (frontend) | — | `localStorage` alanının **adı**; anahtar değil. Karması CORE'daki üç auth kaydıyla eşleşmiyor (doğrulandı) |
| `…_KEY_SHA256 = "<64 hex>"` (cron_auth, realtime_internal_auth, cloudflare) | 18 | Anahtarların kendisi değil, doğrulama **karmaları** ve olay kimlikleri |
| `postgres://postgres:postgres@localhost` (CI workflow'ları) | 1 | Geçici CI test veritabanı; gerçek sistem değil |
| `.env` (2025-09 `fb11a65` → 2026-09 `4364529` silindi) | 1 dosya | `TELEGRAM_TOKEN` **boş**; `TELEGRAM_CHAT_ID` 10 haneli değer. Chat id tek başına erişim vermez. Düşük risk; geçmişte kalıyor |
| `config_live.json` `learning.autopatch.key` | — | Ayar yolu adı (`brain.veto_conf_min`); gizli bilgi değil |

## Kalan riskler ve öneriler

1. **Karmalar ve kaba kuvvet:** Doğrulama karmaları herkese açık. Anahtarlar uzun ve rastgeleyse (≥ 32 karakter) bu risk değil; kısa ya da tahmin edilebilir bir anahtar varsa döndürülmeli. Anahtarlar bende olmadığı için bu doğrulanamadı.
2. **GitHub secret scanning + push protection:** Public depolarda ücretsiz. Açmak bir hesap ayarı olduğu için kullanıcı kararı: Settings → Code security.
3. **Kapsam dışı kalanlar:** Issue/PR yorumları, Actions logları ve artifact'ları, diğer depolar. Düzenli ifade taraması özel biçimli ya da base64 kodlanmış gizli bilgileri kaçırabilir.
4. **İleride:** Tarayıcı CI'a hafif bir kontrol olarak eklenebilir; yeni commit'lerde aynı kalıpları arar.
