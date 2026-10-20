# NIULAI4 sınırlı gözlem deneyi — PLAN (başlatılmadı)

Durum: **yalnızca plan ve uygulanabilirlik değerlendirmesi** (20 Ekim 2026).
- Deney başlatılmadı.
- `wallet_tracking_enabled` kapalı kalıyor.
- Üretim kodu, CORE veritabanı ve zamanlayıcılar değişmedi.
- Ücretli servis yok.

Aşağıdaki "probe" sayıları, bu planı hazırlarken yerelde 3 dakika çalışan salt okuma ölçümlerinden geliyor. Örnekleri küçük ve kanıt değil.

## 0. Uygulanabilirlik probe'ları (20 Ekim ~06:55 UTC, yerel, anahtarsız)

| Soru | Ölçüm | Sonuç |
|---|---|---|
| Piyasa ve çözüm kuralı | `btc-updown-5m-<epoch>` (Gamma `/markets/slug/…`) | 5 dakikalık pencere; çözüm kaynağı **Chainlink BTC/USD TWAP 60s Data Stream**. Kural: "TWAP ≥ başlangıç fiyatı → Up". **Spot fiyat değil.** `feeSchedule`: rate 0,07, yalnızca taker, rebate 0,2. `orderMinSize` 5, tick 0,01 |
| Oracle'a ücretsiz erişim | `data.chain.link/streams/btc-usd-twap-60s-streams` | Sayfa 429 döndü. Data Streams API kimlik bilgisi istiyor. **Gerçek zamanlı oracle değeri ücretsiz erişilebilir değil**, yalnızca spot proxy kullanılabilir |
| Spot referansı | Binance `bookTicker` REST, Coinbase `ticker` | İkisi de Almanya'dan HTTP 200, anahtarsız |
| Kamuya açık aktivite gecikmesi | `/activity?user=…&type=TRADE` 3 sn'de bir, 59 istek, 0 hata | Probe süresince 6 yeni işlem; `ilk görülme − işlem zamanı` **25,5–26,5 sn** (3 sn'ye kadar poll payı dahil). n çok küçük, tek oturum |
| Emir defteri akışı | `wss://ws-subscriptions-clob.polymarket.com/ws/market`, anahtarsız, 60 sn, tek piyasanın 2 token'ı | İlk mesaj 275 ms. **13.236 mesaj** (560 `book`, 12.398 `price_change`, 279 `last_trade_price`); kitap 99 seviye. Sunucu zaman damgası yerel saatin ~160 ms **önünde**, yani yerel saat kayık. Gecikme ölçmek için saat senkronu (NTP) şart |
| Hız sınırları (resmî sayfa) | Data API `/v2/activity` 200 istek/10 sn; CLOB `/book` 1.500/10 sn; Gamma `/markets` 300/10 sn | Plandaki yük bunların %3'ünün altında |

**En önemli uygulanabilirlik bulgusu:** NIULAI4 ilk girişlerinin medyanı pencere bitiminden 55 sn önce. Kamuya açık akış ise işlemi yaklaşık 26 sn sonra gösteriyor. Takip eden biri işlemi gördüğünde pencereden yaklaşık **29 sn** kalıyor; bazı girişlerde pencere çoktan kapanmış oluyor. Bu yüzden deneyin ilk sorusu "avantaj var mı?" değil, **"görüldükten sonra uygulanabilir bir fiyat kalıyor mu?"** olmalı.

## 1. Hipotez (önceden sabit)

- **H1:** NIULAI4'ün kamuya açık bir işlemi Brian'a ulaştıktan d saniye sonra aynı outcome alınsın. Kaydedilmiş gerçek emir defterine göre, maliyet sonrası ortalama sonuç **eşleştirilmiş baseline'dan büyük** mü?
- **H0:** Fark yok ya da negatif.
- **Kopyalanabilir avantaj sayılmayacaklar:** NIULAI4'ün geçmiş kârı ve kendi giriş fiyatı. Ölçüm yalnızca Brian'ın görebildiği an ve o andaki erişilebilir fiyattan yapılır.
- **Ayrı ve ikincil soru (H2):** NIULAI4 gözlemi, aynı andaki spot/defter bilgisine ek bilgi taşıyor mu? Bunun için yalnızca spot ve defter verisiyle kurulan bir baseline'a NIULAI4 özelliği eklenip eklenmemesi karşılaştırılır.

## 2. Kapsam ve bütçe

| Öğe | Pilot değeri | Gerekçe |
|---|---|---|
| Piyasalar | Yalnızca **BTC 5 dakikalık Up/Down** | NIULAI4 satırlarının %89'u 5 dakikalık; ETH ve 15 dakikalık piyasalar pilotu ikiye katlar |
| Eşzamanlı piyasa | Aktif pencere + bir sonraki = 2 | Pencere geçişlerinde veri kaçmasın |
| Saatler | Her gün 12:00–24:00 UTC (12 saat) | ABD seansı ve NIULAI4 aktivitesinin yoğun olduğu saatler. Kesin dağılım ilk gün ölçülür |
| Süre | **7 gün pilot**, sonra karar | Uygulanabilirlik ve gecikme ölçümü için yeterli; avantaj testi için yeterli değil (bkz. §6) |
| Aktivite isteği | 2 sn'de bir → 5 istek/10 sn | Limit 200/10 sn |
| Websocket | 1 bağlantı, 4 token | Ücretsiz, anahtarsız |
| Spot | Binance `bookTicker` websocket + Coinbase REST 5 sn'de bir (çapraz kontrol) | Ücretsiz |
| Ham mesaj | **Saklanmaz** (yaklaşık 13 bin/dk) | Kitap bellekte tutulur, yalnızca örnekler yazılır |
| Saklanan örnek | Her pencerenin son 120 sn'sinde saniyede bir: üst 5 seviye × 2 outcome, spot. NIULAI4 işlemi görülünce ek olarak d ∈ {0, 1, 5, 15, 30, 60} sn anlarında tam derinlik | Yaklaşık 1,5 MB/saat → 7 gün × 12 saat ≈ **130 MB** (gzip'li JSONL) |
| Kaynak | RAM < 150 MB, tek çekirdekte < %5 (JSON ayrıştırma ~220 mesaj/sn) | Yerel makinede rahat |
| Disk tavanı | 500 MB | Aşılırsa durur |
| Ücret | 0 | Ücretsiz uç noktalar + yerel süreç |

**Çalışma yeri:**
- Kullanıcının bilgisayarında ayrı, taşınabilir bir Deno süreci; kalıp `workers/brian-l2-capture` ile aynı.
- Veri **yerel dosyalara** yazılır (`C:\Brian\brian-research\niulai4-pilot\`, repo dışında).
- **CORE'a yazılmaz.** Supabase edge sürekli websocket tutamaz; 15 dakikalık cron da bu iş için uygun değil.

## 3. Kaynaklar ve alanlar (mevcut altyapıyla eşleme)

| Alan | Kaynak | Mevcut Brian | Durum |
|---|---|---|---|
| Doğrulanmış profil/cüzdan | `verifyProfileIdentity` (`_shared/prediction_market.ts`) | ✔ var | Yeniden kullanılır; her gün başında yeniden doğrulanır |
| İşlem: tx hash, token, yön, fiyat, adet, `timestamp` | `/activity` (`normalizeWalletActivityRow`, idempotency anahtarı) | ✔ var | Yeniden kullanılır |
| İlk görülme zamanı ve gecikme (`first_seen − timestamp`) | Yerel poll saati (NTP'ye senkron) | ✖ yok | Eklenecek. **`timestamp` saniye çözünürlüğünde**; gecikme ±1 sn belirsiz |
| Sözleşme/oracle kuralı, pencere başlangıç ve bitişi | Gamma `/markets/slug/…`: `eventStartTime`, `endDate`, `description`, `resolutionSource`, `feeSchedule` | ~ kısmen (Gamma normalizer) | Pencere ve ücret alanları eklenecek |
| Gerçek çözüm | Gamma `/markets/{id}` (`normalizeGammaResolution`) | ✔ var (Faz 2c) | Yeniden kullanılır |
| Oracle değeri (Chainlink TWAP) | — | ✖ | **Ücretsiz erişim yok.** Proxy olarak spot 60 sn TWAP; proxy ile gerçek çözüm uyumu ayrıca ölçülür |
| BTC spot referansı | Binance `bookTicker` WS + Coinbase | ~ Binance L2 işçisi ve `binance_l2_wire.ts` var | Yeniden kullanılır; Coinbase çapraz kontrol |
| İki outcome için bid/ask ve derinlik | CLOB market WS (`book`, `price_change`) | ~ `l2_book.ts` var (Binance için) | Polymarket olay şekli için küçük bir reducer eklenecek |
| Ücret | `polymarketTakerFeeUsdc(C, p, 0.07)` | ✔ var | Yeniden kullanılır; oran her piyasada `feeSchedule.rate`'ten okunur |
| Veri sağlığı | Bağlantı kopması, sıra boşluğu, kitap hash'i, kayıp saniye | ~ CORE'da `source_health` var | Yerel sağlık kaydı; aynı alan adları |

## 4. Ölçüm

NIULAI4'ün her işlemi için (t_trade = işlem zamanı, t_seen = Brian'ın ilk gördüğü an):

1. **Gerçek gecikme:** L = t_seen − t_trade (saat senkronu ile). Dağılım raporlanır: p50, p90, eksik kalanlar.
2. **Duyarlılık:** d ∈ {0, 1, 5, 15, 30, 60} sn için, t_seen + d anında **kaydedilmiş defterden** sabit S = 10 adet (minimum emir 5) için doldurma fiyatı hesaplanır. Doldurma ask derinliği boyunca yürünerek yapılır. Fiyat sınırı en fazla +0,02; daha kötüyse "uygulanamaz" sayılır. d + L pencere bitişini geçiyorsa "geç" sayılır.
3. **Maliyet:**
   - Ücret = S × 0,07 × p(1−p).
   - Kayma = doldurma fiyatı − en iyi ask.
   - Spread kaydedilir.
   - Taker varsayılır (takipçi pasif bekleyemez).
   - Rebate hesaba katılmaz (taker olduğumuz için).
4. **İki outcome:** NIULAI4 aynı pencerede iki tarafı da alıyorsa takipçi de iki tarafı alır. İkisinin sermayesi birlikte rezerve edilir ve pozisyon çifti tek birim olarak raporlanır.
5. **Sermaye:**
   - Sanal kasa 100 USDC.
   - Her giriş, maliyet + ücret kadar parayı çözüme kadar rezerve eder.
   - Yeterli nakit yoksa işlem "reddedildi" sayılır.
   - Brian hazinesinin rezerv mantığı aynen uygulanır.
6. **Sonuç:** Pencere çözülünce payoff 1/0; net = payoff − maliyet − ücret.
7. **Geriye dönük veri:** CLOSED pozisyonlardaki `first_entry_at` yalnızca **davranışı betimlemek** için kullanılır (zamanlama, iki taraf, büyüklük). O işlemler için kamuya açık gecikme bilinmediğinden **avantaj ölçümü yapılmaz**.

## 5. Kontroller

- **Eşleştirilmiş baseline'lar** (aynı pencere, aynı t_seen + d anı, aynı S, aynı maliyet modeli):
  - **B0, favori:** O anda fiyatı yüksek olan outcome'u almak.
  - **B1, spot yönü:** Spot 60 sn TWAP proxy'si pencere açılışının üstündeyse Up, altındaysa Down.
  - **B2, rastgele:** Sabit tohumla rastgele outcome; maliyetin alt sınırını gösterir.
- **H2 testi:** B1'in kuralına NIULAI4'ün yön bilgisi eklenir ("B1 + NIULAI4 aynı yönse al"). Fark ancak bu sayede ölçülür.
- **Önceden sabitlenecekler:** S, d kümesi, fiyat toleransı, "çözüme kadar tut" çıkışı, saat aralığı, baseline tanımları. Pilot başlamadan bu dosyada dondurulur; değişiklik ancak yeni sürüm adıyla yapılabilir.
- **Ayrı dönemler:** Pilotun 7 günü yalnızca uygulanabilirlik içindir. Avantaj testi, kuralları pilot verisine bakmadan sabitlenmiş ayrı bir ileriye dönük dönemde yapılır.
- **Bağımlılık:**
  - Küme birimi pencere (aynı penceredeki işlemler tek olay).
  - Gün etkisi için gün bazlı ikinci bootstrap.
  - Altı farklı d ve üç baseline için çoklu karşılaştırma uyarısı; birincil karşılaştırma önceden tek seçilir (**d = 5 sn, B1'e karşı**).

## 6. Başarı / başarısızlık / yetersiz veri

**Neden yüksek eşik?** Fiyat 0,5 civarında tek bir ikili işlemin getirisinin standart sapması yaklaşık %100. %95 güvenle X puanlık bir farkı ayırt etmek için yaklaşık (1,96/X)² bağımsız pencere gerekir:

| Bağımsız pencere | Ayırt edilebilir en küçük fark |
|---|---|
| ~1.000 | ~±%6 |
| ~10.000 | ~±%2 |

NIULAI4'ün kendi kâr/hacim oranı yaklaşık %2,3. Yani gerçekçi bir avantajı göstermek on binlerce pencere gerektirebilir.

| Durum | Ölçüt (öneri) |
|---|---|
| **Pilot geçti (uygulanabilir)** | Gecikme p50 ölçüldü ve günler arasında tutarlı; işlemlerin en az %30'unda d = 5 sn'de tolerans içinde fiyat var; veri kapsamı %95 üzeri (kayıp saniye, kopma) |
| **Pilot başarısız** | d = 5'te uygulanabilir işlem oranı %10'un altında ya da gecikme pencere süresini aşıyor. Sonuç: "kamuya açık veriyle takip edilemez" diye kaydedilir, deney biter |
| **Avantaj VAR (ileriye dönük test)** | ≥1.000 bağımsız pencere ve ≥14 gün; pencere kümeli %95 aralığın **alt sınırı > 0** (birincil karşılaştırmada) **ve** gün bazlı aralık da > 0 |
| **Avantaj YOK** | Aynı veride %95 aralığın üst sınırı < 0 |
| **Yetersiz veri** | Yukarıdakilerin hiçbiri; büyük ihtimalle sonuç bu olur. Açıkça böyle raporlanır |

## 7. İşletim kuralları

- **Başlama koşulları:**
  - Kullanıcı onayı.
  - Kurallar bu dosyada dondurulmuş olmalı.
  - NTP senkronu doğrulanmış olmalı (kayma < 50 ms).
  - Disk alanı ≥ 1 GB.
  - Profil-cüzdan bugün yeniden doğrulanmış olmalı.
- **Durdurma:**
  - Süre doldu ya da disk tavanı aşıldı.
  - 429 ya da 403 alındı (403 = erişim reddi: hemen ve kalıcı olarak durur; kısıt aşılmaya çalışılmaz).
  - Kimlik doğrulanamadı.
  - Websocket 10 dakikada 5'ten fazla koptu.
  - Kullanıcı istedi.
- **Backoff:**
  - REST: mevcut `fetchPredictionJson` (Retry-After + üstel bekleme).
  - Websocket: yeniden bağlanırken 1→60 sn üstel bekleme. Her kopma bir veri boşluğu olarak kaydedilir; boşluktaki işlemler ölçüme alınmaz.
- **Mevcut sistemin etkilenmediğini denetleme:**
  - Pilot CORE'a bağlanmaz, CORE anahtarı kullanmaz.
  - Günlük kontrol: cron jobid 103'ün turları başarılı mı, `brian_prediction_*` satır sayıları beklenen hızda mı, `wallet_tracking_enabled` hâlâ `false` mı.
  - Host izin listesi `gamma-api`, `data-api`, `clob`, `ws-subscriptions-clob.polymarket.com`, Binance ve Coinbase ile sınırlı.
- **Rapor (günlük + final):** Gecikme dağılımı; d başına uygulanabilir oran, kayma, ücret ve spread; baseline'larla fark ve kümeli aralık; bağımsız pencere ve gün sayısı; veri sağlığı (kapsam, kopma, boşluk); proxy (spot TWAP) ile gerçek çözüm uyum oranı; "yetersiz veri" açıkça belirtilir.

## 8. Önerilen en küçük ücretsiz pilot

| Konu | Öneri |
|---|---|
| Kapsam | BTC 5 dakikalık Up/Down, aktif + sonraki pencere, 12:00–24:00 UTC, **7 gün** |
| Çalışma | Yerel Deno süreci; dosyaya yazar; CORE'a dokunmaz |
| Yük | 5 istek/10 sn + 1 websocket + spot websocket; yaklaşık 130 MB disk; < 150 MB RAM; 0 maliyet |
| Ölçülen | Gerçek gecikme, d = 0/1/5/15/30/60 sn uygulanabilirliği, maliyet, baseline farkı (yalnızca betimleyici) |
| Karar | Pilot uygulanabilirlik ölçütlerini geçerse ayrı ve kuralları sabitlenmiş en az 14 günlük ileriye dönük test; geçemezse dur |

| Bilinen eksik | Etkisi |
|---|---|
| Chainlink TWAP oracle'ına ücretsiz gerçek zamanlı erişim yok | Spot proxy kullanılır; proxy-çözüm uyumsuzluğu ayrıca raporlanır |
| İşlem `timestamp` saniye çözünürlüğünde | Gecikme ±1 sn belirsiz |
| NIULAI4 maker mı taker mı bilinmiyor; emir defteri kimliği yok | Takipçinin taker maliyeti NIULAI4'ünkiyle karşılaştırılamaz |
| Avantaj testi on binlerce pencere gerektirebilir | Büyük olasılıkla sonuç "yetersiz veri" olur |
| Yerel süreç PC kapanınca durur | Kapsam boşlukları raporlanır, doldurulmaz |
| Websocket saat kayması (~160 ms) | NTP senkronu başlama koşulu |

| Başlamadan önce gerekli | Kim |
|---|---|
| Plan ve kuralların onayı (bu dosya) | Kullanıcı |
| Yerel pilot kodu (`research/niulai4_pilot/`, testli, üretime bağlı değil) | Claude, onaydan sonra ayrı PR |
| NTP senkronu ve disk kontrolü | Claude, başlamadan hemen önce |
| Profil-cüzdan günlük yeniden doğrulama | Pilot sürecinin kendisi |
| `wallet_tracking_enabled` **kapalı kalır** (pilot CORE'a hiç yazmaz) | — |
