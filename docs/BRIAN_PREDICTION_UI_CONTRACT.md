# Brian UI ↔ backend sözleşmesi — Prediction Intelligence ve NIULAI4 pilotu

Sürüm `brian-prediction-status.v1` (20 Ekim 2026). Arayüz Codex'te (`monster-coins-pro/**`, `codex/brian-ui-*` dalları), backend Claude'da (`supabase/**`, `research/**`). Bu dosya ikisi arasındaki tek doğruluk kaynağıdır. Alan eklemek geriye uyumludur; alan silmek ya da anlamını değiştirmek `schema_version` artışı gerektirir.

Temel kural: **erişilemeyen ya da henüz kurulmamış veri açıkça öyle gösterilir.** Varsayılan, yer tutucu ya da "çalışıyor" görüntüsü üretilmez. Her bölümün kendi `status` alanı ve zaman damgası vardır; arayüz rengini ve metnini yalnızca bunlardan türetir.

## 1. Uç: `brian-prediction-status`

| | |
|---|---|
| URL | `https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-prediction-status` (Vercel üzerinden `/api/brian/brian-prediction-status`) |
| Yöntem | `POST`, gövde `{}`. `OPTIONS` CORS ön isteği desteklenir |
| Yetki | `x-brian-dashboard-key` başlığı. Diğer dashboard uçlarıyla aynı anahtar (`brian_dashboard_auth.control-v3`) |
| Önbellek | Sunucuda 30 sn. Yanıttaki `cache_age_seconds` önbellek yaşını verir. Önerilen yoklama: 60 sn |
| Yazma | **Yok.** Tek bir salt okunur, repeatable-read işlem |
| Kaynak | `supabase/functions/brian-prediction-status/index.ts`, `_shared/prediction_status.ts`, migration `20261020074329_brian_niulai4_pilot_ingest.sql` |

### HTTP durumları

| Kod | Gövde | Arayüz |
|---|---|---|
| 200 | Aşağıdaki şema | Bölümleri tek tek göster |
| 401 | `{"error":"UNAUTHORIZED_DASHBOARD"}` | "Dashboard anahtarı doğrulanamadı". Veri gösterme |
| 405 | `{"error":"METHOD_NOT_ALLOWED"}` | Programlama hatası |
| 503 | `{"error":"STATUS_UNAVAILABLE","detail":…}` | "Durum şu an alınamıyor". Son başarılı yanıtı **eski** olarak göster (zaman damgasıyla) ya da hiç gösterme |
| Ağ hatası / zaman aşımı | — | 503 ile aynı davran |

### 200 yanıt şeması (üst seviye)

```jsonc
{
  "schema_version": "brian-prediction-status.v1",
  "observed_at": "2026-10-20T08:00:00.000Z",   // DB işlem zamanı; tüm bölümler bu ana göre
  "cache_age_seconds": 0,
  "shadow_only": true, "live_execution": false, "trade_signal": false,  // her zaman bu değerler
  "prediction":   { "status": "...", ... },    // §2
  "calibration":  { "status": "...", ... },    // §3
  "niulai4_pilot": { "status": "...", ... }    // §4
}
```

Her bölüm bağımsız okunur. Bir bölüm okunamazsa yalnızca o bölüm şöyle döner:

```json
{ "status": "UNAVAILABLE", "reason": "READ_FAILED:PostgresError" }
```

`reason` değerleri: `READ_FAILED:<hata sınıfı>`, `PREDICTION_STATUS_MISSING`, `CALIBRATION_INPUTS_MISSING`, `PILOT_STATUS_MISSING`. Arayüz bu durumda "veri alınamadı" yazar, sayı göstermez.

## 2. `prediction` — Polymarket gözlem modülü

| `status` | Anlamı | Önerilen gösterim |
|---|---|---|
| `DISABLED` | `module_enabled = false` | Gri, "Kapalı" |
| `NO_DATA` | Açık ama hiç tur kaydı yok | Gri, "Henüz veri yok" |
| `STALE` | Son tur 40 dakikadan eski (`stale_after_seconds: 2400`; tur aralığı 15 dk) | **Sarı**, "Gecikmiş — son tur: …" |
| `DEGRADED` | Son turdaki kaynaklardan en az biri `OK` değil | Sarı, `last_cycle_sources` içindeki sorunlu kaynakları listele |
| `OK` | Son tur ≤ 40 dk ve tüm kaynaklar `OK` | Yeşil |
| `UNAVAILABLE` | Okunamadı | Kırmızı değil, gri/kesik: "Durum alınamadı" |

Alanlar:

| Alan | Tip | Not |
|---|---|---|
| `as_of` | timestamptz | `observed_at` ile aynı an |
| `control.module_enabled`, `control.wallet_tracking_enabled` | bool | Cüzdan takibi şu an **kapalı**. Arayüz bunu "kapalı" diye göstermeli, ima etmemeli |
| `control.max_markets`, `control.tag_slugs`, `control.max_storage_mb`, `control.updated_at` | | Yapılandırma, salt okunur |
| `last_cycle_at` | timestamptz \| null | Son toplayıcı turu (kaynak sağlık kaydı zamanı) |
| `last_cycle_sources[]` | `{source,status}` | `status` ∈ `OK, DEGRADED, BLOCKED, RATE_LIMITED, ERROR, SKIPPED` |
| `snapshots.total`, `snapshots.last_24h`, `snapshots.last_received_at` | | Piyasa anlık görüntüleri |
| `resolutions.total`, `resolutions.last_received_at` | | Sonuçlanan piyasalar. İlk sonuçlar 29 Ekim 2026'dan önce beklenmiyor; o zamana kadar `total: 0` normaldir ve "henüz sonuçlanan piyasa yok" yazılmalı |
| `storage.bytes`, `storage.cap_bytes` | bigint | Tavan 256 MB. Doluluk oranı = bytes / cap_bytes |
| `health_24h` | object | `brian_prediction_health_summary_v1` çıktısı: kaynak başına satır, `ok`/`not_ok`, gecikme, tur sayıları |

## 3. `calibration` — piyasa olasılıklarının kalibrasyonu

| `status` | Anlamı |
|---|---|
| `INSUFFICIENT_DATA` | 20'den az bağımsız event. Aralık yok (`report.brierInterval = null`). Arayüz "yetersiz veri (n event)" yazar, Brier skorunu öne çıkarmaz |
| `SUFFICIENT` | Event-kümeli bootstrap aralığı mevcut |
| `UNAVAILABLE` | Okunamadı |

Alanlar:
- `as_of`, `lead_seconds` (86400): tahmin, kapanıştan 24 saat önceki son anlık görüntüdür.
- `method`, `excluded.{no_forecast_before_lead, null_probability, non_binary_resolutions}`.
- `report.{markets, events, brier, logLoss, baseRateBrier, brierInterval, bins[], tradeSignal:false}`.

**Bu bir işlem sinyali değildir.** Piyasanın ne kadar kalibre olduğunu ölçer, kâr hakkında bir şey söylemez.

## 4. `niulai4_pilot` — yerel gözlem pilotu

Pilot Supabase dışında, sürekli çalışan yerel bir süreçtir. CORE'a yalnızca kendi sağlık ve özet kayıtlarını gönderebilir (§5).

| `status` | Anlamı | Önerilen gösterim |
|---|---|---|
| `NOT_CONFIGURED` | Ingest anahtarı kurulmamış ya da kapalı; hiçbir kayıt gelemez | Gri, "Kurulmadı — pilot başlatılmadı" |
| `NOT_STARTED` | Anahtar hazır, hiç çalışma kaydı yok | Gri, "Başlatılmadı (onay bekliyor)" |
| `RUNNING` | Çalışan kayıt ve heartbeat ≤ 180 sn (`stale_after_seconds: 180`) | Yeşil, "Gözlemde" + `last_heartbeat.age_seconds` |
| `STALE` | Kayıt çalışıyor görünüyor ama heartbeat > 180 sn | **Sarı**, "Pilot sessiz — süreç kapalı ya da bağlantı yok". Asla "çalışıyor" gösterme |
| `STOPPED` | Erken durdu. `run.stop_reason` nedeni verir | `ACCESS_DENIED`, `RATE_LIMITED`, `WS_UNSTABLE`, `IDENTITY_UNVERIFIED`, `DISK_CAP` için **kırmızı uyarı**; `OPERATOR` ve `HEARTBEAT_LOST` için sarı |
| `COMPLETED` | Süre doldu (`DURATION_ELAPSED`) | Nötr, "Tamamlandı" |
| `UNAVAILABLE` | Okunamadı | Gri |

Alanlar:

| Alan | Not |
|---|---|
| `ingest_configured` | bool |
| `run` | `null` ya da `{run_id, rules_version, rules_fingerprint, host_label, state, stop_reason, started_at, ended_at, last_heartbeat_at}` |
| `last_heartbeat` | `null` ya da `{seq, observed_at, received_at, age_seconds, health}`. Health içeriği: `stats.{wsMessages, wsReconnects, spotReconnects, coinbaseErrors, unsyncedDeltas, polls, pollErrors, observations, captures, tailSamples}`, `bytes`, `spotFresh`, `lastSocketMsgAgeMs`, `pendingCaptures`, `sink.{sent, stored, rejected, rateLimited, failed, lastStatus, lastError}` |
| `latest_report` | `null` ya da `{seq, kind, observed_at, received_at, summary}`. `FINAL` varsa o, yoksa en yeni `INTERIM` döner |
| `counts.{heartbeats, reports}` | |

Pilot özetinin (`latest_report.summary`) yorumu:
- `INTERIM` özette `verdict` her zaman `INTERIM_NO_VERDICT`'tir. Arayüz bunu sonuç gibi gösteremez ("avantaj var/yok" yazılmaz).
- Ayrı bloklar: `feasibility` (`PASS`/`FAIL`/`INCONCLUSIVE_REDESIGN`/`INSUFFICIENT_DATA`), `absoluteNet`, `baselineSuperiority`, `power` (σ_w, gereken pencere sayısı), latency sayıları (`seenAfterWindowClosed`, `lateForPrimaryDelay`), d başına ret nedenleri.
- `h2WalletIndependentTiming: "NOT_TESTED"` ve `tradeSignal: false` her zaman bulunur.
- d = 0 satırları `OPTIMISTIC_UPPER_BOUND` etiketlidir; arayüz bu etiketi göstermelidir.

## 4b. `research` — Research Lab defteri (v1'e eklendi, geriye uyumlu)

Kaynak: `brian_research_status_v1`. Okunamazsa diğer bölümler gibi `UNAVAILABLE` + `reason` döner.

| `status` | Anlamı |
|---|---|
| `NO_DATA` | Henüz kayıtlı makale/iddia yok |
| `OK` | En az bir iddia kayıtlı (sonuçlar aşağıda, tek tek) |

Alanlar:
- `papers[]`: `{paper_id, title, authors, url, license, code_url, code_commit, data_cutoff, missing_info, ledger_path, ledger_sha256, assumptions[]}`. `assumptions[]` öğeleri `{assumption_id, claim_id, leverage, detail, chosen, rationale}`.
- `claims[]`: `{claim_id, paper_id, tier, statement, metric, repro_risk, reported, status, status_reason, result_at, experiment_id, code_commit, run_url, evidence_class, metrics, runs}`.
  - `status`: `NOT_STARTED | RUNNING | INSUFFICIENT_DATA | PASS | FAIL | UNVERIFIED | REVIEW_REQUIRED`. Hiç test edilmemiş iddia `NOT_STARTED` görünür.
  - `runs`: bu iddia için kaç test yapıldığı. Önceki sonuçlar (başarısızlar dahil) silinmez; burada her zaman **en son** sonuç gösterilir.
  - `evidence_class`: `OUT_OF_SAMPLE_ADAPTED_REPLICATION` (yöntem uyarlanarak yeni veride), `OUT_OF_SAMPLE_REPLICATION`, `IN_SAMPLE_REPRODUCTION`, `DEMO`. Arayüz bunu etiket olarak göstermelidir. `DEMO` hiçbir zaman gerçek sonuç gibi sunulamaz.
- `counts`: duruma göre iddia sayısı.
- `trade_signal: false`. Araştırma sonucu bir işlem sinyali değildir.

Arayüz kuralları:
- `PASS` yeşil, `FAIL` kırmızı, `UNVERIFIED` ve `INSUFFICIENT_DATA` gri, `REVIEW_REQUIRED` sarı. Hiçbiri "kârlı" diye etiketlenemez.
- Varsayımlar (`leverage: high`) sonuçla birlikte görünür olmalı. Uyarlanmış bir test, makalenin birebir tekrarı gibi sunulamaz.

## 5. Yazma yolu (yalnızca pilot için; arayüz kullanmaz)

`brian-niulai4-pilot-ingest` (20 Ekim 08:12 UTC'den beri CORE'da v1 ACTIVE; anahtar kullanıcı onayıyla kuruldu, karması CORE'da. Pilot başlatılmadığı için durum `NOT_STARTED`):
- `POST` ve `x-brian-pilot-key` başlığı. CORS yok, tarayıcıdan çağrılmaz.
- Anahtar yalnızca SHA-256 hash olarak `brian_niulai4_pilot_auth` tablosunda durur. Migration **anahtar oluşturmaz**: satır `enabled=false, hash=null` gelir; bu durumda her istek `503 PILOT_INGEST_NOT_CONFIGURED` alır.
- Kabul edilen türler: `START`, `HEARTBEAT` (≤ 1 / 30 sn), `REPORT` (≤ 1 / 5 dk; `FINAL` yalnızca bir kez), `STOP`.
- Reddedilen her şey bir neden koduyla döner, kaydedilmez:
  - `NOT_SHADOW_ONLY`, `FINGERPRINT_MISMATCH`, `OBSERVED_AT_OUT_OF_RANGE` (±2 dk ileri / 1 saat geri)
  - `ANOTHER_RUN_ACTIVE`, `BAD_*`, `FINAL_ALREADY_STORED`
- Tablolar `anon` ve `authenticated` rollerine kapalı. `service_role` okuyabilir, doğrudan yazamaz; tek yazma yolu `brian_record_niulai4_pilot_v1`.
- Heartbeat'ler 30 gün tutulur. Var olan hiçbir tabloya dokunulmaz.

## 6. Arayüzün kullandığı mevcut uçlar (20 Ekim envanteri; değiştirilmedi)

`frontier-v4.html` script'lerinden çıkarıldı. Hepsi CORE'da `ACTIVE`.
- `brian-frontier-heartbeat-public` dışındakilerin hepsi `POST` ve `x-brian-dashboard-key` istiyor. `GET` isteği 405 alır; bu bir arıza değildir.
- `brian-evolution-alpha-intelligence-status` `verify_jwt=true` (401: JWT gerekir).
- Bu uçların alan sözleşmeleri kendi modüllerine aittir ve bu PR'da değişmedi.

| Uç | Alan / oda | Yetki |
|---|---|---|
| `brian-frontier-heartbeat-public` | Genel nabız (alpha, hazine özeti, toplayıcılar, haber hattı) | Açık. 20 Ekim 07:31 UTC `status: OK` |
| `brian-control-center` | Komuta / kontrol | dashboard key |
| `brian-system-control` | Sistem aç/kapa | dashboard key |
| `brian-frontier-news` | Haberler | dashboard key |
| `brian-frontier-engineering-status`, `brian-development-status` | Yazılım / mühendislik | dashboard key |
| `brian-frontier-autonomy-status` | Otonomi | dashboard key |
| `brian-treasury-accounting`, `brian-evolution-treasury-status` | Muhasebe / hazine | dashboard key |
| `brian-realtime-market-marks-public` (REALTIME projesi) | Hazine pozisyon fiyatları | Açık, POST |
| `brian-shadow-scorecard` | Gölge karne / kanıt | dashboard key |
| `brian-lab-readiness-status`, `brian-evolution-lab-status` | Lab | dashboard key |
| `brian-evolution-status`, `brian-evolution-ocean-status`, `brian-evolution-alpha-intelligence-status` | Evrim / okyanus / alpha | dashboard key (alpha: JWT) |
| `brian-world-status` | Dünya | dashboard key |
| RPC `brian_meeting_shadow_upsert` | Toplantı gölge defteri | PostgREST |

Heartbeat'te görülen ve arayüzün doğru göstermesi gereken iki gerçek durum:
- `control.core_dispatcher_active: false`, `realtime_scheduler_active: true`. CORE dispatcher kapalı; bu zamanlayıcı alanı ve bu PR'da dokunulmadı.
- `managed_jobs_active: 15 / managed_jobs_total: 30`.

## 6b. 20 Ekim yavaşlık olayı ve `brian-frontier-engineering-status` değişikliği

**Ölçüm** (edge ve Postgres logları, 07:00–08:00 UTC):
- `brian-frontier-engineering-status` 1 saatte 71 çağrı aldı; p50 34 sn, 26 çağrı 150 sn edge sınırında 5xx döndü.
- `brian-treasury-accounting` ve `brian-frontier-autonomy-status` da 150 sn'de kesildi; `brian-shadow-scorecard` 33 sn sürdü.
- 07:30'dan önce DB'de yavaş sorgu yok denecek kadar azdı. 07:30'dan sonra 5 dakikada 16–29 `statement timeout` görüldü.
- Fonksiyon loglarında `write CONNECT_TIMEOUT` ve "Timed out acquiring connection from connection pool" var.
- Vercel `/api/brian/*` proxy'si yalnızca ~60–150 ms ekliyor; neden proxy değil.

**Kök neden:**
- CORE küçük bir compute üzerinde: 5,8 GB veritabanı, `shared_buffers` 224 MB, `max_connections` 60. Disk ve CPU payı tükenince sayfa okuma ~14 ms'ye çıkıyor.
- Mühendislik ucu açık her sekmeden dakikada bir çağrılıyordu; her çağrıda paralel 10 okuma yapıyordu ve bunların üçü tam tarama + sıralama gerektiriyordu (`DISTINCT ON` görünümleri ve index'siz `observed_at` sıralaması).
- 07:30'da ikinci bir tarayıcı açılınca yük iki katına çıktı. Okumalar birikip bağlantı havuzunu doldurdu; diğer uçlar da bağlanamadı.

**Backend düzeltmesi** (migration `…_brian_status_read_load.sql` ve uç kodu):
- İki "latest" görünüm aynı sonucu veren index skip-scan biçimine çevrildi. Satırlar, sütunlar ve sıralama birebir aynı; testli.
- `brian_evolution_hypothesis_snapshots(observed_at desc)` index'i eklendi.
- Uç artık sonucu 45 sn paylaşıyor ve eşzamanlı istekler tek hesaplamayı bekliyor.
- Yeni alan `cache_age_seconds` (sn). Yeniden hesaplama başarısız olursa son iyi sonuç en fazla 15 dk boyunca **`status: "STALE"`, `stale: true`, `stale_reason`** ile döner. Arayüz bunu "eski veri (x dk önce)" olarak göstermeli, "ONLINE" gibi göstermemeli.
- Son iyi sonuç yoksa eskisi gibi 500 ve `status: "DEGRADED"` döner.

**Arayüz için öneriler (Codex):**
- Ağır durum uçlarını gizli sekmede yoklama (`document.visibilityState`).
- Aynı uca paralel istek atma; bir istek bitmeden yenisini başlatma.
- Yoklama aralığını önbellek süresinden kısa tutma (≥ 45–60 sn).

## 6c. `brian-realtime-market-marks-public` — canlılık kuralı (20 Ekim, v4)

- `live` artık fiyatın **şu anki yaşından** hesaplanıyor: `age_seconds <= live_max_age_seconds` (120). Emtia ve endeks fiyatları için ayrıca `session_state == "REGULAR"` şartı var. Kayıtlı `data_latency_seconds` değeri yalnızca bilgi amaçlı; kararı etkilemiyor.
- Yeni alanlar:
  - Her fiyatta `age_seconds` (sayı ya da `null`).
  - Yanıtın kökünde `live_max_age_seconds`.
  - Zaman damgası okunamazsa ya da 30 sn'den fazla ileri tarihliyse `age_seconds: null, live: false`.
- Beklenen görünüm:
  - Emtia kaynağı yapısı gereği ~10 dk gecikmeli, bu yüzden pratikte hep `live: false` olur. Arayüz "gecikmeli (x dk)" göstermeli.
  - Kripto tick akışı 2–3 dakikada bir güncellendiği için kripto da sık sık `live: false` görünebilir.
- Arayüzün PR175'teki 2 dk kuralıyla aynı. İki taraf artık aynı kararı veriyor.

## 6d. Evrim deney koşucusu (20 Ekim)

Deney koşucusu (jobid 44, her saat :27) ve terfi kurulu (jobid 45, her saat :34) yeniden açıldı; 26 Eylül'den beri kapalıydı. `brian-evolution-lab-status` ve `brian-evolution-status` uçlarında yeni deney sonuçları görünmeye başlayacak. Terfi hâlâ **insan onayı** istiyor; otomatik terfi yok.

## 7. Zaman ve tazelik kuralları (tüm bölümler)

- Tüm zamanlar ISO-8601 UTC. Arayüz yerel saate çevirebilir ama kaynağı UTC'dir.
- Tazelik sunucuda hesaplanır (`status` + `stale_after_seconds`). Arayüz kendi saatine göre yeniden "taze" kararı vermez; yalnızca "x dk önce" metni üretir.
- Yanıt 503 ya da ağ hatası verdiyse ekrandaki son veri "eski (son başarılı: …)" etiketiyle kalabilir. Yeni veri gibi gösterilemez.
- Sayısal alan `null` ise arayüz "—" ya da "veri yok" gösterir, `0` göstermez.

## 8. `integration` — tüm sistemin bağlılık durumu (23 Ekim, `brian.integration-status.v1`)

`brian-prediction-status` yanıtına `integration` alanı eklendi (aynı POST, aynı anahtar). Kaynak: `public.brian_treasury_integration_status_v1()`.

- `integrated` (bool): kasaya veri besleyen her parça (`feeds_treasury`: treasury, unified, evidence, prediction_eye) `OK` ve izlenen cron işlerinin hepsi aktifse `true`. UI bunu tek rozet olarak göstermeli: **Entegre** / **Kopukluk var**.
- `components.<ad>.status`: `OK` | `STALE` | `OFF` | `LOCAL_ONLY`. `LOCAL_ONLY` hata değildir (OFI şu an PC'de çalışıyor); gri göster.
- `components.treasury`: `equity_usd`, `cash_usd`, `deployment_pct`, `blocked_reasons[0]` (neden nakitte durduğunu tek cümleyle gösterir).
- `components.evidence`: `segments`, `segments_clearing_cost` (masrafı geçen sinyal ailesi sayısı; 0 ise kasa bilerek nakitte), `lanes.{MULTIASSET_EVENT_REACTION,CANONICAL_ALPHA}.n7d`.
- `components.pm_calibration`, `niulai4`, `ofi`, `opportunity_audit`: ayrı deneyler; `role` metni neden kasaya doğrudan bağlı olmadıklarını açıklar, olduğu gibi gösterilebilir.
- Okuma başarısızsa `integration = {status:"UNAVAILABLE", integrated:false}` gelir; UI bunu "bilinmiyor" göstermeli, asla "entegre" değil.
- Hiçbir alan işlem sinyali değildir (`trade_signal:false`).
