# Brian — Prediction Intelligence (Faz 0 bulguları ve Faz 1 durumu)

Durum: 19 Ekim 2026. Modül **yalnızca okuma + gölge (shadow)** çalışır. Gerçek para, emir, cüzdan anahtarı, canlı karar değişikliği yoktur.

## Faz 0 — doğrulanan gerçekler (19 Ekim 2026, Berlin'den ölçüldü)

| Kaynak | Sonuç |
|---|---|
| Gamma API `markets/keyset` | HTTP 200, erişim açık |
| Gamma API `public-search` | HTTP 200 |
| CLOB `/time`, `/book`, `/prices-history` | HTTP 200, herkese açık |
| Data API `/activity`, `/positions`, `/value`, `/traded`, `/v1/leaderboard` | HTTP 200 |

**Kritik bulgu:** Gamma `GET /markets` (offset) kullanımdan kaldırıldı (`deprecation: true`, `sunset: 2026-05-01`). Keyset uç noktası `markets/keyset` kullanılmalı. İstek parametresi **`after_cursor`**, cevap alanı **`next_cursor`**. `cursor` adı **sessizce yok sayılır** ve hep ilk sayfa döner; bu yüzden yineleyici daha önce görülen imleci reddeder.

**NIULAI4 kimliği (doğrulandı):** `public-search` tam ad eşleşmesi + 42 karakterlik tam cüzdan adresi + `/activity` satırlarının aynı cüzdan ve adı bildirmesi. Cüzdan: `0x44832d0d2ec11187c1e77d786feb15f6a50254c6`. Benzer adlar (`Niulai`, `niulai666`…) farklı hesaplardır ve eşleştirilmez.

**Kamuya açık ilk gözlemler (OBSERVED_VERIFIED; strateji değil):** liderlik tablosu (ALL) sıra 1963, hacim ≈ 4,18 M USD, PnL ≈ 97 bin USD (hacme oranı ≈ %2,3); işlem edilen piyasa sayısı 11104; gördüğümüz işlemler ağırlıkla "Bitcoin Up or Down" 5 dakikalık piyasalar. Pozisyon satırlarında giriş ücreti (`entryFeesUsdc`) görünüyor. Bu sayılar bir hipotez değildir ve takip edilebilir bir avantaj kanıtı değildir: 5 dakikalık piyasalarda herkese açık veri gecikmesi avantajı yok edebilir (Faz 2 sorusu 9).

## Faz 1 — bu PR'ın kapsamı

`supabase/functions/_shared/prediction_market.ts` (saf, enjekte edilebilir `fetch`):

- Host izin listesi: yalnızca `gamma-api`, `data-api`, `clob.polymarket.com`, https, kimlik bilgisi yok.
- 429/5xx/ağ hatasında sınırlı yeniden deneme: `Retry-After` (saniye veya HTTP tarihi) + üstel geri çekilme.
- 401/403/451 = `BLOCKED`: yeniden denenmez, coğrafi kısıt aşılmaya çalışılmaz.
- Gamma piyasa normalleştirme: eksik değer `null` kalır, `0` olmaz; olasılık [0,1] dışıysa `null`.
- Keyset sayfalama (`after_cursor`/`next_cursor`), `maxPages` sınırı, tekrar eden imleçte durma.
- Cüzdan işlemleri: yalnızca `type=TRADE` işlem sayılır; REDEEM/MERGE/… ayrı sayılır; kararlı idempotency anahtarı; yinelenenler birleştirilir.
- Kimlik doğrulama: tam ad + tam adres + etkinlik çapraz kontrolü; aksi halde `UNVERIFIED`/`AMBIGUOUS`. Adres asla tamamlanmaz veya tahmin edilmez.

Bu PR veritabanına yazmaz, zamanlayıcı eklemez, panele dokunmaz, mevcut karar akışına bağlanmaz.

## Testler

```
deno test supabase/functions/_shared/prediction_market.test.ts                       # çevrimdışı, CI
PREDICTION_LIVE=1 deno test --allow-net --allow-env supabase/functions/_shared/prediction_market.test.ts   # canlı duman testi
```

## Faz 1b — depolama ve sınırlı toplayıcı

**Migrasyon** `20261019195019_brian_prediction_intelligence.sql`:

| Tablo | Tür | Not |
|---|---|---|
| `brian_prediction_control` | durum (tek satır) | Hepsi **kapalı** başlar; `live_execution` açılamaz (CHECK) |
| `brian_prediction_wallets` | durum | Yalnızca 42 karakterlik tam adres; VERIFIED için doğrulama zamanı zorunlu; watermark sadece ileri gider |
| `brian_prediction_markets` | kanıt, append-only | Piyasa tanımı + olay bağlamı |
| `brian_prediction_market_snapshots` | kanıt, append-only | Olasılıklar; eksik değer NULL; kaynak güncelleme zamanı ve tazelik saniyesi |
| `brian_prediction_wallet_trades` | kanıt, append-only | Yalnızca VERIFIED cüzdan + takip açıkken saklanır |
| `brian_prediction_source_health` | kanıt, append-only | Kaynak başına durum, HTTP, deneme, gecikme |

RLS açık; anon/authenticated erişemez; service_role yalnızca gerekli izinlere sahip; yazıcı RPC'ler `SECURITY DEFINER` + sabit `search_path`.

**Toplayıcı** `_shared/prediction_collector.ts` + `brian-prediction-eye`: global Brian anahtarı **ve** `module_enabled` açık değilse hiçbir şey yapmaz; collector lease kullanır; tur başına sınırlı istek (6 etiket sayfası, en fazla 60 piyasa, 2 kimlik isteği, en fazla 2×500 işlem). Watermark yalnızca işlemler veritabanına kabul edildikten sonra ilerler; bir sayfa başarısızsa ilerlemez.

**API bulguları (19 Ekim):** `markets/keyset` `tag_slug`'ı **sessizce yok sayar**, `events/keyset` uygular — konu keşfi olaylar üzerinden. Data API geçmiş offset sınırı 5000 → zaman tabanlı (`start` + ASC) devam.

**Canlı kuru çalıştırma (DB'siz) + yerel Postgres:** 6 etiketten 688 piyasa görüldü, 60 tutuldu, 0 reddedildi; NIULAI4 VERIFIED; 24 saatlik pencerede 1000 işlem (kesildi, sonraki tura devreder). Aynı canlı paket yerel Postgres 16'ya iki kez yazıldı: 1. tur 60/60/1000/11 saklandı, 2. tur 0 — idempotent.

**Bilinen eksikler / dikkat:**
- NIULAI4 günde binlerce işlem yapıyor (≈13 saatte 2000). Cüzdan takibi açılmadan önce **saklama süresi/boyut sınırı** eklenmeli.
- Zamanlayıcı (cron) eklenmedi, fonksiyon yayımlanmadı, migrasyon canlı veritabanına uygulanmadı. Bunlar kullanıcı onayıyla ayrı adım.
- Veriler henüz hiçbir karara bağlanmadı.

## Faz 1c — sınırlı depolama

Mevcut soğuk arşiv (`brian-cold-archive`) 3 Ekim'den beri çalışmıyor (cron pasif). Bu yüzden iki ayrı sınır var:

1. **Sert sınır:** `brian_prediction_control.max_storage_mb` (varsayılan **256 MB**, 1–2048). Kaydedici her çağrıda modülün kendi tablolarını ölçer. Sınırda yeni piyasa/anlık görüntü/işlem **saklanmaz**, `STORAGE_CAP_REACHED` döner. Sağlık satırları yine yazılır, neden görünür kalır. Yer açmak için **hiçbir şey silinmez**. Fonksiyon bu durumda watermark'ı ilerletmez. Bir paket sınırı en fazla bir paket kadar aşabilir.
2. **Arşiv politikaları:** `brian_archive_policies` kayıtları eklendi. Mevcut "arşivle, doğrula, 24 saat bekle, sil" yolu, operatör arşivi çalıştırdığında bu tablolara da uygulanır:

| Tablo | Sıcak süre | Silme |
|---|---|---|
| `brian_prediction_market_snapshots` | 90 gün | doğrulanmış arşivden sonra |
| `brian_prediction_wallet_trades` | 30 gün | doğrulanmış arşivden sonra |
| `brian_prediction_source_health` | 14 gün | doğrulanmış arşivden sonra |
| `brian_prediction_markets` | 365 gün | **asla** (anlık görüntüler referans veriyor) |

Bu migrasyon hiçbir şey silmez ve hiçbir zamanlayıcıyı açmaz.

## Faz 1d — zamanlayıcı ve açılış (yalnız piyasa anlık görüntüleri)

- `brian_prediction_control`: `module_enabled=true`, `wallet_tracking_enabled=false` (migrasyon cüzdan takibini açıkça kapatır).
- CORE pg_cron işi `brian-prediction-eye-15m`, her saat 5, 20, 35 ve 50. dakikalarda çalışır. Diğer CORE eye'ları gibi vault'taki cron anahtarıyla çağırır; önceki istek pg_net kuyruğundaysa o turu atlar.
- `brian-prediction-eye` artık cron anahtarını da kabul ediyor; internal key de geçerliliğini koruyor.
- **Durdurma:** `module_enabled=false`, global `brian_system_enabled=false` ya da `select cron.unschedule('brian-prediction-eye-15m')`.
- Faz 1c'deki 256 MB sınırı her yazmada uygulanır.

## Faz 2a — anlık-zaman olasılık değişimi ve veri sağlığı

İki salt-okuma fonksiyonu (yalnızca service_role):

- `brian_prediction_probability_changes_v1(as_of, limit, stale_after)`: Her piyasa için "Yes" olasılığı ve 1, 6, 24 saatlik değişim. **Yalnızca `received_at <= as_of` olan anlık görüntüler kullanılır**, yani Brian'ın o anda gerçekten bildiği veri. Gelecek bilgisi sızamaz.
  - O kadar eski bir gözlem yoksa değişim `NULL` olur ve durum `INSUFFICIENT_HISTORY` olarak işaretlenir; 0 sayılmaz.
  - Son gözlemi 45 dakikadan eski olan piyasa `STALE` işaretlenir. Olasılık yoksa durum `NO_PROBABILITY` olur.
  - Çıktı `trade_signal: false` taşır; bu bir işlem sinyali değildir.
- `brian_prediction_health_summary_v1(as_of, window)`: Kaynak başına durum dağılımı, son gözlem ve gecikme bilgisi verir. Ayrıca beklenen ve gözlenen tur sayısını (15 dakikalık plana göre, toplamanın başladığı andan itibaren), anlık görüntü tazeliğini ve depolama/sınır durumunu raporlar.

## Sıradaki tek adım

Faz 2b: ekonomik olay eşleştirmesi. Polymarket piyasalarını Brian'ın mevcut World/intel olaylarıyla zaman ve konu üzerinden eşleştirip "haber öncesi ve sonrası olasılık" karşılaştırması yapılacak. Panel (Faz 4) bu fonksiyonları okuyacak.
