# Prediction Intelligence — canlıya alma planı (Faz 1b)

Durum: **PLAN — uygulanmadı.** Hazırlanma: 19 Ekim 2026. Kaynak: PR #150 (`dd990d4`).

## Hedef proje: CORE (`qbcjuxhvhwagvqbjyemo`)

- Fonksiyon global anahtarı `brian_evolution_runtime_config` tablosundan okuyor. Bu tablo yalnızca CORE'da var. REALTIME'da (`dliediwlldojkfjzlznm`) yok, fonksiyon orada her çağrıda güvenli şekilde kapanırdı.
- Örnek alınan eye `brian-outcome-price-eye` da CORE'da çalışıyor: `verify_jwt=false` ve internal key ile yetkilendirme.

## Ön kontroller (salt okuma, 19 Ekim'de yapıldı)

| Kontrol | Sonuç |
|---|---|
| `brian_prediction_*` tablo/fonksiyon var mı | Yok, isim çakışması yok |
| `brian-prediction-eye` fonksiyonu var mı | Yok |
| Lease fonksiyonları (`brian_*_collector_lease`) | Var |
| `brian_reject_mutation` | Var. Gövdesi migrasyondakiyle aynı, `create or replace` davranışını değiştirmez |
| `brian_system_enabled` | `true` |
| DB boyutu / son migrasyon | 5704 MB / `20261019190608` |

Migrasyon **yalnızca ekleme yapar.** Mevcut hiçbir tablo, cron işi veya fonksiyon değişmez.

## Adımlar

**0. Önkoşul (senin onayın):** PR #150 incelenip `brian-2026`'ya merge edilir. Yayın merge SHA'sından yapılır, böylece ana dal ve canlı aynı kodu taşır.

**1. Yedek:**
- `brian-2026` HEAD'ine `backup/brian-2026-10-19-prediction-oncesi` git tag'i atılır ve push edilir.
- `C:\Brian\brian-backups\2026-10-19-prediction-oncesi\` klasörüne `brian-2026.bundle` (doğrulanmış) ve CORE durum JSON'u yazılır. JSON'da migrasyon listesi, public tablo ve fonksiyon adları, `brian_reject_mutation` md5'i, cron işleri ve edge fonksiyon listesi bulunur.

**2. Migrasyon:** Supabase `apply_migration` ile `brian_prediction_intelligence` uygulanır. Ardından repodaki dosya adı uygulanan sürüm numarasına çevrilir; bu, repodaki mevcut "applied history" pratiği.

**3. Son kontroller (SQL):**
- 6 tablonun hepsinde RLS açık.
- Kontrol satırı: modül kapalı, cüzdan takibi kapalı, `live_execution=false`.
- `anon` ve `authenticated` için `has_table_privilege` sonucu false.
- 4 append-only trigger var, 2 RPC `SECURITY DEFINER` ve `search_path` sabit.
- Supabase güvenlik advisor'ı çalıştırılır.

**4. Fonksiyon yayını:** `brian-prediction-eye`, `verify_jwt=false` ile yayımlanır (internal key yetkisi, örnek eye ile aynı). Kaynak dosyalar: `index.ts`, `_shared/prediction_market.ts`, `prediction_collector.ts`, `collector_lease.ts`, `realtime_internal_auth.ts`. Yayımlanan kodun test edilen kaynakla aynı olduğu doğrulanır.

**5. Duman testi:**
- Anahtarsız POST → `401` (bunu ben yapabilirim).
- Anahtarlı POST → `MODULE_DISABLED` beklenir. Internal key bende yok ve istemiyorum, bu yüzden bu testi istersen sen çalıştırırsın.

**6. Hiçbir şey açılmaz:** `module_enabled=false` kalır. Zamanlayıcı eklenmez. Toplamayı açmak ayrı bir karar: önce Faz 1c'deki saklama sınırı gelir, sonra yalnızca piyasa anlık görüntüleri açılır, cüzdan takibi kapalı kalır.

## Geri dönüş

| Durum | Yapılacak |
|---|---|
| Fonksiyon sorunlu | `brian-prediction-eye` silinir. Onu çağıran hiçbir şey yok |
| Migrasyon sorunlu, **veri yok** (modül hiç açılmadı) | Aşağıdaki SQL çalışır, `schema_migrations` satırı silinir. `brian_reject_mutation` paylaşımlı olduğu için **dokunulmaz** |
| Modül açıldıktan sonra sorun | Proje doktrini gereği kanıt silinmez. `module_enabled=false` yapılır, tablolar kalır |
| Kod | `brian-2026` üzerindeki merge commit `git revert` edilir; tag yedekten geri dönülebilir |

```sql
drop function if exists public.brian_record_prediction_batch_v1(jsonb, jsonb, jsonb, jsonb);
drop function if exists public.brian_upsert_prediction_wallet_v1(text, text, text, jsonb, timestamptz);
drop table if exists public.brian_prediction_wallet_trades;
drop table if exists public.brian_prediction_market_snapshots;
drop table if exists public.brian_prediction_markets;
drop table if exists public.brian_prediction_source_health;
drop table if exists public.brian_prediction_wallets;
drop table if exists public.brian_prediction_control;
```

## Risk ve maliyet

- **Risk düşük:** Değişiklik yalnızca ekleme yapıyor ve modül kapalı geliyor. CORE disk yükü açılana kadar sıfır. Açıldıktan sonra 15 dakikada bir 60 piyasa çekilirse günde yaklaşık 5.800 satır (birkaç MB) eder.
- **Ek maliyet yok:** Yeni ücretli servis yok. Polymarket uç noktaları ücretsiz ve herkese açık.

## Yayın kaydı — 19 Ekim 2026 (uygulandı)

| Adım | Sonuç |
|---|---|
| Yedek | Tag `backup/brian-2026-10-19-prediction-oncesi` → `6af5466` push edildi; `C:\Brian\brian-backups\2026-10-19-prediction-oncesi\` içinde `brian-2026.bundle` (verify: okay) + `core-state.json` |
| Merge | PR #150 → `brian-2026`, merge commit `6e10481` (CI: Prediction, Brian 2026, ALPHA v2 yeşil, `a1b03ed`) |
| Migrasyon | CORE `qbcjuxhvhwagvqbjyemo`, sürüm `20261019195019` (`brian_prediction_intelligence`); dosya adı bu sürüme hizalandı |
| Son kontroller | 6 tabloda RLS açık; anon/authenticated SELECT yok; service_role kanıt tablolarında UPDATE/DELETE yok; 4 append-only trigger; 2 RPC `SECURITY DEFINER` + sabit `search_path`, yalnızca service_role EXECUTE; kontrol satırı hepsi kapalı; `brian_reject_mutation` md5 değişmedi (`47210aae…`) |
| Advisor (security) | Yeni tablolar için yalnızca INFO "RLS enabled no policy" (diğer 116 tabloyla aynı, kasıtlı). Yeni WARN yok; mevcut 2 WARN önceden var olan fonksiyonlara ait |
| Fonksiyon | `brian-prediction-eye` v1 ACTIVE, `verify_jwt=false`, ezbr `739d3d7f…` |
| Duman testi | Anahtarsız POST → 401, yanlış anahtar → 401, GET → 405. Anahtarlı çağrı yapılmadı (internal key kullanılmadı) |
| Durum | `module_enabled=false`, cron işi yok, prediction tablolarında 0 satır, lease olayı yok |

## Yayın kaydı — Faz 1c (19 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Yedek | Tag `backup/brian-2026-10-19-prediction-1c-oncesi` → `311c34e`; `C:\Brian\brian-backups\2026-10-19-prediction-1c-oncesi\` (bundle verify: okay + `core-state.json`, eski RPC md5 `d842b802…`) |
| Merge | PR #152 → `0c25ed9` (CI üçü de yeşil, `c1e7b69`; ilk koşudaki hata collation'a bağlı test sıralamasıydı, düzeltildi) |
| Migrasyon | CORE sürüm `20261019200517` (`brian_prediction_retention`); dosya adı hizalandı |
| Son kontroller | `max_storage_mb=256`; `brian_prediction_storage_bytes()` = 122880 bayt; yeni RPC md5 `0191f445…`, `SECURITY DEFINER` + sabit `search_path`; iki fonksiyon için yalnızca service_role EXECUTE; modül kapalıyken RPC `OPERATOR_STOPPED`; 4 arşiv politikası eklendi (toplam 21); `brian-cold-archive-daily` cron'u **pasif kaldı** |
| Fonksiyon | `brian-prediction-eye` v2 ACTIVE (ezbr `310089e7…`), `STORAGE_CAP_REACHED` durumunu tanıyor; anahtarsız POST → 401, GET → 405 |
| Geri dönüş | `core-state.json` içindeki adımlar: v1 RPC'yi yeniden uygula, `brian_prediction_storage_bytes` ve `max_storage_mb`'ı kaldır, `brian_prediction%` politikalarını sil, eye v1'i tag'den yeniden yayımla |

## Yayın kaydı — Faz 1d (19 Ekim 2026): zamanlayıcı + açılış

| Adım | Sonuç |
|---|---|
| Yedek | Tag `backup/brian-2026-10-19-prediction-1d-oncesi` → `7b284c5`; `C:\Brian\brian-backups\2026-10-19-prediction-1d-oncesi\` (bundle verify: okay + `core-state.json`) |
| Merge | PR #154 → `3c58ec7` (CI üçü de yeşil, `47c4d63`) |
| Fonksiyon | Migrasyondan **önce** yayımlandı: `brian-prediction-eye` v3 (ezbr `7e39e7e8…`), cron anahtarını kabul ediyor. Anahtarsız → 401, yanlış cron anahtarı → 401 |
| Migrasyon | CORE sürüm `20261019201911` (`brian_prediction_schedule`); dosya adı hizalandı |
| Durum | `module_enabled=true`, `wallet_tracking_enabled=false`; cron `brian-prediction-eye-15m` (jobid 103), `5,20,35,50 * * * *`, aktif |
| İlk gerçek tur | 20:19:22 UTC'de cron komutuyla aynı SQL elle çalıştırıldı (istek 186518) → HTTP 200 `SUCCESS`: 6 etiket, 688 piyasa görüldü, 60 tutuldu, 0 reddedildi; 60 piyasa + 60 anlık görüntü + 6 sağlık satırı saklandı, 0 işlem. Modül depolaması 311.296 bayt |
| Geri dönüş | `select cron.unschedule('brian-prediction-eye-15m')` + `module_enabled=false`. Toplanan kanıt append-only olduğu için kalır |
| İlk zamanlanmış tur | 20:20:00 UTC cron jobid 103 `succeeded` → HTTP 200 `SUCCESS`, 6 sağlık satırı; yeni piyasa/anlık görüntü **0**. Sebep: elle tetiklenen turdan 38 saniye sonra çalıştı ve Gamma yanıtı CDN'de önbellekli (`Cache-Control: max-age=300`), yani `updatedAt` aynı kaldı. Aynı gözlem yinelenmedi; bu idempotentliğin beklenen sonucu. 15 dakikalık aralıkta yeni anlık görüntüler beklenir, sonraki turlarda doğrulanacak |

## Yayın kaydı — Faz 2a (19 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Yedek | Tag `backup/brian-2026-10-19-prediction-2a-oncesi` → `1c5dabf`; bundle verify: okay + `core-state.json` |
| Merge | PR #156 → `4e6e21a` (CI üçü de yeşil) |
| Migrasyon | CORE sürüm `20261019202909` (`brian_prediction_probability_changes`); dosya adı hizalandı |
| Canlı kontrol (20:29 UTC) | Sağlık özeti: 6 kaynak, hepsi OK, ortalama gecikme 58–98 ms; tur kapsamı gözlenen 2 / beklenen 1 (fazlalık, 20:19'daki elle tetiklenen turdan); depolama 311.296 / 268.435.456 bayt. Değişim fonksiyonu: 60 piyasanın hepsi `INSUFFICIENT_HISTORY` (veri 1 saatten genç; beklenen sonuç, 0 değişim değil) |
| 20:35 zamanlanmış tur | `succeeded`; 60 yeni anlık görüntü (yeni `source_updated_at` 20:34:46.224Z), 2 yeni piyasa ilk 60'a girdi. Toplam 120 anlık görüntü; 58 piyasada ≥2 gözlem |
| Bilinen sınır | Tur sayımı dakika bazında; elle tetiklenen turlar da sayılır |

## Yayın kaydı — Faz 2a.1 (19 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Yedek | Tag `backup/brian-2026-10-19-prediction-2a1-oncesi` → `4e6e21a`; bundle verify: okay. Önceki fonksiyon md5 `ff568cf3…` |
| Merge | PR #157 → `b52023e` (CI üçü de yeşil) |
| Migrasyon | CORE sürüm `20261019204328` (`brian_prediction_probability_changes_elapsed`); yeni md5 `66157adc…`; anon EXECUTE yok; dosya adı hizalandı |
| Canlı kontrol | Yeni alanlar (`baseline_*_at`, `elapsed_*_seconds`, `relative_change_*`, `change_unit`) dönüyor; veri henüz 1 saatten genç olduğu için hepsi `NULL`/`INSUFFICIENT_HISTORY` |
| Geri dönüş | 2a migrasyonundaki fonksiyon gövdesini yeniden uygula (`20261019202909_…`) |

## Yayın kaydı — Faz 2c (19–20 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Yedek | Tag `backup/brian-2026-10-19-prediction-2c-oncesi` → `6b7482c`; bundle verify: okay + `core-state.json` (eski `storage_bytes` md5 `8f2b2d2f…`, eye v3) |
| Merge | PR #160 → `6b7482c` (CI üçü de yeşil) |
| Migrasyon | CORE sürüm `20261019210906` (`brian_prediction_resolutions`); dosya adı hizalandı |
| Son kontroller | Tabloda RLS açık; anon/authenticated SELECT yok; service_role UPDATE yok; append-only trigger var. 4 fonksiyon `SECURITY DEFINER` ve yalnızca service_role EXECUTE. Arşiv politikası: 3650 gün, silme yok. `brian_reject_mutation` md5 değişmedi. Advisor: yeni WARN yok, yalnızca beklenen INFO |
| Kuru kontrol | Şu an aday 0; 29 Ekim 06:00 itibarıyla 5 aday (ilk bitişler 29 Ekim 03:59) |
| Fonksiyon | `brian-prediction-eye` v4 ACTIVE (ezbr `823b9319…`). Anahtarsız → 401, yanlış cron anahtarı → 401, GET → 405 |
| Gece doğrulaması (20 Ekim 06:45 UTC) | Cron jobid 103: 42/42 `succeeded` (20:20–06:35). Görülebilen 24 yanıtın hepsi `SUCCESS` ve `resolution: NO_CANDIDATES` içeriyor (v4 canlı). 2.577 anlık görüntü, 2,39 MB / 256 MB. Değişim fonksiyonu: 61 FRESH, 16 STALE, 3 INSUFFICIENT_HISTORY; tur kapsamı 44/42 (fazlalık elle tetiklenen 2 turdan) |
| Bilinen sınır | İlk 60'tan düşen piyasalar STALE oluyor. Bitişten 24 saat önce anlık görüntüsü yoksa kalibrasyonda `no_forecast_before_lead` olarak dışarıda kalırlar; sonuçları yine kaydedilir |

## Yayın kaydı — UI durum ucu + NIULAI4 pilot sink (20 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Yedek | Tag `backup/niulai4-ingest-oncesi` → `b1dd6f7`; bundle + `core-state.json` (`C:\Brian\brian-backups\2026-10-20-niulai4-ingest-oncesi\`) |
| Merge | PR #164 → `4ad5e6e` (8 CI kontrolünün hepsi yeşil) |
| Migrasyon | CORE sürüm `20261020074329` (`brian_niulai4_pilot_ingest`); dosya adı hizalandı. Yalnızca yeni tablolar ve fonksiyonlar; mevcut nesneler değişmedi |
| Son kontroller | Auth satırı `enabled=false`, hash `NULL` → pilot durumu `NOT_CONFIGURED`. `anon` SELECT/EXECUTE yok. `wallet_tracking_enabled=false`. `brian_prediction_status_v1` = `OK` (6 kaynak OK, 2.757 anlık görüntü, 2,53 MB / 256 MB, 0 sonuç) |
| Fonksiyon | `brian-prediction-status` v1 ACTIVE (ezbr `132d8a3f…`). GET → 405, anahtarsız/yanlış anahtar → 401, OPTIONS → 204. 200 yolu dashboard anahtarıyla dışarıdan **denenmedi** (anahtar kullanılmadı). SQL fonksiyonları CORE'da doğrudan çağrılarak doğrulandı; şekillendirme birim testli |
| Yayınlanmadı | `brian-niulai4-pilot-ingest`: yayın aracı izin sınıflandırıcısınca reddedildi; yeniden denenmedi. Anahtar kararı verilene kadar gerekmiyor (çağrılırsa 404). Pilot sink açıkken bu uç yoksa sink `HTTP_404` sayar, ölçüm etkilenmez |
| Gözlem | 07:33–07:43 UTC arası CORE'da birden fazla modülde pg_net zaman aşımı / DNS gecikmesi görüldü (migrasyondan önce başladı). 07:35 prediction turu kayıt üretmedi; son tur 07:20. Durum, 40 dk eşiği aşılırsa `STALE` olur |
| Geri dönüş | Yeni nesneleri düşür: `drop function brian_prediction_status_v1, brian_niulai4_pilot_status_v1, brian_record_niulai4_pilot_v1; drop table brian_niulai4_pilot_reports, _heartbeats, _runs, _auth;` ve `brian-prediction-status` fonksiyonunu sil. Mevcut veriye etkisi yok |

## Yayın kaydı — durum okuma yükü düzeltmesi (20 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Olay | 07:30–08:05 UTC: `brian-frontier-engineering-status` p50 34 sn, 150 sn edge sınırında çok sayıda 5xx; muhasebe ve otonomi uçları da 150 sn'de kesildi. DB'de 5 dakikada 16–29 `statement timeout`; `CONNECT_TIMEOUT` ve bağlantı havuzu zaman aşımları. Sayfa okuma ~14 ms (I/O sınırı), önbellek isabetli sorgu bile 3,4 sn (CPU sınırı). DB 5,8 GB, `shared_buffers` 224 MB, `max_connections` 60 |
| Yedek | Tag `backup/status-read-load-oncesi` → `db111c8`; `C:\Brian\brian-backups\2026-10-20-status-read-load-oncesi\` (bundle, eski görünüm SQL'i, eski fonksiyon kaynağı, `core-state.json`) |
| Merge | PR #167 → `0dbac9e` (8 CI kontrolünün hepsi yeşil) |
| Migrasyon | CORE sürüm `20261020080320` (`brian_status_read_load`); dosya adı hizalandı. İki görünüm skip-scan biçimine çevrildi (177 / 186 satır), `security_invoker` korundu, `anon` erişimi yok. `brian_evolution_hypothesis_snapshots_observed_idx` eklendi |
| Fonksiyon | `brian-frontier-engineering-status` v14 → v15 (ezbr `2b2723d9…`). v14'ün canlı kaynağı repo ile birebir aynıydı. 45 sn paylaşılan önbellek, single flight, açıkça işaretli `STALE` geri dönüşü |
| Geri dönüş | `old_latest_views.sql` dosyasını yeniden uygula; v14 kaynağını (`engineering-status.index.ts`) yeniden yayınla; index'i düşür. Veri değişmedi |
| Kalan risk | Kök kısıt compute boyutu. Ücretli yükseltme kullanıcı kararıdır, yapılmadı |

## Yayın kaydı — NIULAI4 ingest anahtarı ve ucu (20 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Onay | Kullanıcı: "sen üret kur kankam canlıya alabilirsin de" (Seçenek A). Pilotu başlatma onayı ayrıdır ve verilmedi |
| Anahtar | 48 karakter, `secrets.token_urlsafe`; yalnızca kullanıcının profilinde `~/.brian/niulai4-pilot-ingest.key`. Repoya, loga ya da sohbete yazılmadı |
| CORE | `brian_niulai4_pilot_auth`: SHA-256 `a677ddb8…`, `enabled=true`, `configured_at` 08:11:08 UTC |
| Fonksiyon | `brian-niulai4-pilot-ingest` v1 ACTIVE (ezbr `ccdffe24…`); dosyalar repo ile birebir |
| Doğrulama | Anahtarsız → 401, yanlış anahtar → 401, doğru anahtar + geçersiz `run_id` → 400 `BAD_RUN_ID`, doğru anahtar + shadow dışı → 400 `NOT_SHADOW_ONLY`. Tablolarda 0 satır; pilot durumu `NOT_CONFIGURED` → `NOT_STARTED` |
| İptal | `update brian_niulai4_pilot_auth set enabled=false` (anahtar anında geçersiz). Döndürme: yeni anahtar üret, karmayı değiştir |

## Yayın kaydı — hazine scorecard v2 dönemi (20 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Onay | Kullanıcı (Codex aracılığıyla): yeni 7 günlük ölçüm, mevcut sunucu altyapısında, ücretsiz sınırlar içinde; PC'ye bağımlı değil. NIULAI4 ve gerçek para kapsam dışı |
| Neden | v1 (`forward-20260922-v1-treasury`, 22–29 Eylül) gözlemleri 26 Eylül 14:20 UTC'de kesildi: pg_cron `brian-shadow-scorecard-5m` (jobid 95) `active=false` idi ve kayıt defterinde yoktu. v1 olduğu gibi bırakıldı; boşluk doldurulmadı |
| Yedek | Tag `backup/scorecard-v2-oncesi` → `7acd0c2`; `C:\Brian\brian-backups\2026-10-20-scorecard-v2-oncesi\` (bundle, scorecard v6 kaynağı, `core-state.json`) |
| Merge | PR #172 → `05f5770` (8 CI kontrolünün hepsi yeşil) |
| Fonksiyon | `brian-shadow-scorecard` v6 → v7 (ezbr `00e6d99c…`): her motor için en yeni protokol. Tek satırlık değişiklik |
| Migrasyon | CORE sürüm `20261020082424`; dosya adı hizalandı. Protokol `forward-20261020-v2-treasury`: 2026-10-20 08:24:24 UTC → 2026-10-27 08:24:24 UTC; kurallar v1 ile aynı (doğrulandı) |
| Zamanlayıcı | CORE pg_cron jobid 95 `brian-shadow-scorecard-5m`, `12,22,32,42,52 * * * *`, `active=true`. Var olan `brian_capture_shadow_observations()` çalışır (8 sn zaman aşımı, tek index okuması ~0,2 ms). ~6 satır/saat, 7 günde ~1.000 satır. Edge, harici host ya da ücretli kaynak yok |
| İlk kayıt | 08:24:24 UTC; equity 4938,06, kaynak zamanı 08:22:01 (2,4 dk), politika `brian.treasury-shadow.v3`, açık pozisyon 3 |
| Bilinen sınır | Dönem açık pozisyonla başladı → model kuralı `OPEN_POSITIONS_AT_START` → karar `INCOMPLETE` olarak kalır. Ölçüm betimleyici olarak sürer |
| Geri dönüş | `cron.alter_job(95, active := false)`; scorecard v6 kaynağını yeniden yayınla. Protokol satırları append-only'dir, silinmez |

## Yayın kaydı — Research Lab defteri (20 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Yedek | Tag `backup/research-ledger-oncesi` → `37047a6`; `C:\Brian\brian-backups\2026-10-20-research-ledger-oncesi\` (bundle, durum ucu v1 kaynağı, `core-state.json`) |
| Merge | PR #178 → `cb607ff` (9 CI kontrolü yeşil) |
| Migrasyon | CORE `20261020211802` (`brian_research_ledger`) ve `20261020211823` (`brian_research_ledger_writer_grants`); dosya adları hizalandı |
| Bulgu ve düzeltme | Supabase varsayılan yetkileri yeni fonksiyonlarda `service_role`'a EXECUTE veriyor; iki yazma fonksiyonu açık kalmıştı. Hemen geri alındı. Test artık bu varsayılanı taklit ediyor; düzeltme olmadan başarısız olduğu doğrulandı |
| Yetkiler (CORE'da doğrulandı) | Yazma fonksiyonları: anon/authenticated/service_role hayır. `brian_research_status_v1`: yalnızca service_role. Tablolar: anon SELECT yok |
| Defter | `arxiv:2602.19520`: 2 iddia, 9 varsayım (ledger sha256 `b13f3646…`). Sonuç: `run-1-truncated` = `REVIEW_REQUIRED` |
| Fonksiyon | `brian-prediction-status` v2 (ezbr `c50e072f…`): ek `research` bölümü. GET 405, anahtarsız 401, OPTIONS 204 |
| Geri dönüş | Durum ucunun v1 kaynağını yeniden yayınla. Tablolar append-only; silinmez, gerekirse fonksiyonların EXECUTE'u kapatılır |

## Yayın kaydı — fiyat canlılık kuralı + deney koşucusu (20 Ekim 2026)

| Adım | Sonuç |
|---|---|
| Fiyat ucu | PR #181 → `42dddc5`. REALTIME `brian-realtime-market-marks-public` v3 → v4 (ezbr `baf7c09d…`). Yedek: `C:\Brian\brian-backups\2026-10-20-marks-live-age-oncesi\`. Canlı doğrulama: gümüş/altın 1.680 sn → `live:false`; BTC 139 sn → `live:false` (önceden hepsi `true`) |
| Koşucu | CORE `20261020213549`: jobid 44 (koşucu) ve 45 (kurul) `active=true`. Elle tetikleme 21:35 UTC: SUCCESS, 1,1 sn, 1 deney ölçüldü, 2 sonuç yazıldı. Canlı kod zaten `PAGINATED_LABELS_DECISION_TIME_COST_V2` yamasını içeriyor; kod değişmedi |
| Geri dönüş | `cron.alter_job(44, active := false)`, `cron.alter_job(45, active := false)`; fiyat ucu için yedekteki v3 kaynağını yeniden yayınla |
