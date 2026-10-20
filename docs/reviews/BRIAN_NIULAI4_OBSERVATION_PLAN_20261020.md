# NIULAI4 sınırlı gözlem deneyi — PLAN v2 (başlatılmadı)

Durum: **yalnızca plan, kod ve test** (20 Ekim 2026). Kural sürümü `niulai4-pilot.v2` (`research/niulai4_pilot/rules.ts`).
- Deney ve canlı veri toplama başlatılmadı.
- `wallet_tracking_enabled` kapalı; CORE, üretim kodu ve zamanlayıcılar değişmedi.
- Ücretli servis yok.

v1'e göre değişenler, Codex incelemesi 1–7. maddeler: mutlak kârlılık ayrı bir koşul; H2 ertelendi; ilk görülme sırasına göre karar ve sızıntı testi; gerçekleşebilirlik ret kuralları; oracle erişiminin belgeden yeniden değerlendirilmesi; güç hesabı ve tek final analiz; gecikme ve kaynak tahminlerinin dili.

## 0. Uygulanabilirlik probe'ları (20 Ekim ~06:55 UTC, yerel, anahtarsız; n küçük, kanıt değil)

| Soru | Ölçüm | Sonuç |
|---|---|---|
| Piyasa ve çözüm kuralı | `btc-updown-5m-<epoch>` (Gamma `/markets/slug/…`) | 5 dk pencere. Çözüm kaynağı **Chainlink BTC/USD TWAP 60s**; "TWAP ≥ başlangıç → Up"; spot değil. `feeSchedule`: rate 0,07, yalnızca taker. `orderMinSize` 5, tick 0,01 |
| Oracle erişimi | Resmî belge [market-data/realtime-data](https://docs.polymarket.com/market-data/realtime-data), 20 Ekim | 60 sn'lik Chainlink TWAP akışı **var**: `wss://ws-live-v2.polymarket.com/ws`, kanal `price.crypto.twap` (TS'de `prices.crypto.twap`), `window_seconds = 60`. Belgeye göre bu abonelik **kimlik doğrulaması istiyor** (CLOB API key + secret + passphrase; `SecureClient`/`AsyncSecureClient` ≥ 0.11.0). Belgede **ücret, kota ve limit bilgisi yok**. CLOB API kimlik bilgisi bir cüzdan imzasıyla üretilir; bu, hesap/anahtar oluşturmak demek ve **yapılmadı**. Önceki "ücretsiz erişim yok" ifadesi **geri alındı**: doğru ifade "erişim kimlik doğrulaması gerektiriyor, maliyeti belgelenmemiş, doğrulanmadı". Akışın sembolünün sözleşmedeki `btc-usd-twap-60s-streams` ile **aynı seri olduğu da doğrulanmadı**. Karar kullanıcıda; o zamana kadar spot proxy kullanılıyor ve sınırlaması açıkça belirtiliyor |
| Spot referansı | Binance `bookTicker`, Coinbase `ticker` | İkisi de Almanya'dan HTTP 200, anahtarsız |
| Kamuya açık aktivite gecikmesi | `/activity`, 3 sn'de bir, 59 istek | **n = 6** yeni işlem, tek oturum: 25,5–26,5 sn (poll aralığı dahil). **Genellenemez**; dağılım pilotta ölçülecek |
| Emir defteri akışı | CLOB market websocket, anahtarsız, 60 sn | ~13.000 mesaj/dk, 99 seviye. Belgeye göre 10 sn'de bir text `PING` gerekiyor; pilot gönderiyor (v1'de eksikti) |
| Saat | Windows `w32tm`, Cloudflare + PTB | Yerel saat yaklaşık +202–207 ms geride; örnekler arası ±4 ms. Düzeltme olarak uygulanıyor |
| Hız sınırları (resmî sayfa) | Data API `/v2/activity` 200/10 sn; CLOB `/book` 1.500/10 sn; Gamma `/markets` 300/10 sn | Plandaki yük bunların çok altında |

## 1. Soru — DARALTILDI

- **Pilotun sorusu (koşullu takip):** NIULAI4'ün kamuya açık bir işlemi Brian'a ulaştıktan sonra, kaydedilmiş gerçek defterde, emir gecikmesi ve maliyetler dahil, o işlemi izlemek **(a) gerçekleşebilir mi, (b) mutlak olarak kârlı mı, (c) aynı anlarda basit kurallardan iyi mi?** Üçü **ayrı** raporlanır.
- **Geçmiş kâr ve NIULAI4'ün kendi giriş fiyatı** kanıt sayılmaz.
- **H2 (cüzdandan bağımsız zamanlama katkısı) bu pilotta TEST EDİLMİYOR.** Yalnızca NIULAI4'ün işlem yaptığı anlara koşullu karşılaştırma, zamanlamanın katkısını ayıramaz. H2 için cüzdandan bağımsız, önceden belirlenmiş karar anları gerekir: örneğin her pencerenin sabit saniyelerinde spot+defter kuralı ile aynı kurala NIULAI4 bilgisi eklenmiş sürüm aynı bütçe ve maliyetle karşılaştırılır; işlem yapılmayan anlar ve katılım oranı dahil edilir. Kapsamı büyüttüğü için sonraya bırakıldı. Rapor `h2WalletIndependentTiming: NOT_TESTED` yazar.

## 2. Kapsam ve bütçe

| Öğe | Değer | Not |
|---|---|---|
| Piyasa | Yalnızca BTC 5 dk Up/Down; aktif + sonraki pencere | |
| Saat | 12:00–24:00 UTC | NIULAI4 aktivitesinin dağılımı pilotta ölçülecek |
| Süre | 7 günlük pilot (uygulanabilirlik + varyans tahmini) | Avantaj testi ayrı ve sonra (§6) |
| İstek | Aktivite 2 sn'de bir (5/10 sn); Coinbase 5 sn'de bir; Gamma pencere başına 1; çözüm pencere başına 1 | |
| Websocket | CLOB market (4 token, 10 sn'de PING) + Binance `bookTicker` | Ücretsiz, anahtarsız |
| Saklanan | Ham mesaj yok. Pencerelerin son 120 sn'sinde saniyede bir üst 5 seviye; her işlem için 6 yakalama | |
| Disk | **Tavan** 500 MB (kod zorluyor). Tahmin: kabaca 1–2 MB/saat | **Tahmin, ölçülmedi** |
| RAM / CPU | Tahmin: < 150 MB, tek çekirdekte birkaç % | **Tahmin, ölçülmedi.** Gerçek tüketim pilotun sağlık kaydında raporlanacak |
| Çalışma yeri | Kullanıcının bilgisayarı, yerel Deno süreci, yerel dosyalar | CORE'a yazılmaz |
| Ücret | 0 | |

## 3. Alanlar

Mevcut koddan yeniden kullanılanlar:
- kimlik doğrulama (`verifyProfileIdentity`),
- işlem normalizasyonu ve idempotency (`normalizeWalletActivityRow`),
- çözüm normalizasyonu (`normalizeGammaResolution`),
- resmî ücret fonksiyonu (`polymarketTakerFeeUsdc`),
- yeniden deneme mantığı (`fetchPredictionJson`).

Pilotta yeni olanlar:
- ilk görülme zamanı (NTP düzeltmeli) ve gecikme,
- pencere başlangıç/bitiş ve `feeSchedule` kontrolü,
- CLOB websocket defter reducer'ı,
- spot proxy (Binance + Coinbase),
- soket sessizlik süresi ve yeniden bağlanma boşlukları.

**Oracle değeri yok** (§0).

## 4. Ölçüm ve gerçekleşebilirlik kuralları (kodda zorunlu)

Her işlem için: t_trade = kamuya açık zaman (1 sn çözünürlük), t_seen = Brian'ın ilk gördüğü an.

1. **Karar anı ve emir gecikmesi:** Karar t_seen + d anında verilir (d ∈ {0, 1, 5, 15, 30, 60} sn). Emrin borsaya **300 ms** sonra ulaştığı varsayılır (önceden kayıtlı varsayım). Defter t_seen + d + 300 ms anında okunur. **d = 0 "iyimser üst sınır"** olarak etiketlenir.
2. **Ret kuralları (hiçbiri "düzeltilmez"):**

   | Durum | Sonuç |
   |---|---|
   | Emir pencere kapandıktan sonra varır | `LATE_WINDOW_CLOSED` |
   | Yakalama planlanan zamandan önce, planlanan + 500 ms'den sonra ya da plandan farklı bir zamanda | `CAPTURE_TIME_INVALID` — geç alınmış snapshot'tan geçmiş dolum üretilmez |
   | Soketten 2 sn'den uzun süre mesaj gelmemiş ya da bu süre bilinmiyor | `STALE_BOOK` |
   | Bağlanmadan / yeniden bağlanmadan sonra tam `book` snapshot'ı gelmemiş | `BOOK_UNSYNCED` |
   | +0,02 tolerans içinde 10 adedin tamamı dolmuyor | `UNFILLABLE` — kısmi dolum dolum sayılmaz |
   | Piyasanın ücret oranı bilinmiyor | `NO_FEE_RATE` — varsayılmaz |

3. **Maliyet:** Gerçekten yürünen seviyelerin toplamı. Ücret seviye bazında `C × rate × p(1−p)` ile hesaplanır. Kayma **yalnızca raporlanır**; zaten maliyetin içinde olduğu için iki kez düşülmez.
4. **Karar sırası ve sızıntı:** Gözlemler **ilk görülme sırasıyla** işlenir. Her karar yalnızca o gözlemi ve onun planlı yakalamasını kullanır. Aynı pencerenin öteki tarafı sonradan görülse bile geçmiş karar değişmez (test var). Rastgele baseline'ın yazı-turası `hash(obsId, d)`'den gelir; sıradan bağımsızdır.
5. **Sermaye:** Her (strateji, d) çiftinin kendi 100 USDC sanal kasası var. Aynı kasadaki girişler ortak sermayeyi paylaşır; maliyet + ücret pencere çözülene kadar rezerve edilir. Çözülmemiş pencerenin sermayesi kilitli kalır.
6. **Geriye dönük veri:** Yalnızca davranışı betimlemek için kullanılır. Gecikme bilinmediğinden avantaj ölçümü yapılmaz.

## 5. Kontroller (koşullu)

- **Baseline'lar** (aynı an, aynı miktar, aynı maliyet):
  - B0: favori outcome.
  - B1: spot TWAP proxy yönü.
  - B2: rastgele.
- **Birincil karşılaştırma:** d = 5 sn, FOLLOW − B1, eşleştirilmiş fark. Önceden sabit; diğer karşılaştırmalar betimleyicidir.
- **Bağımlılık:** Pencere kümeli **ve** gün kümeli bootstrap; ikisinin de geçmesi gerekir.

## 6. Başarı ölçütleri ve istatistik

Üç ayrı soru, üç ayrı karar:

| Soru | Ölçüt |
|---|---|
| **Uygulanabilirlik** (pilot) | d = 5'te en az 100 deneme şartıyla: ≥ %30 gerçekleşebilir → `PASS`; < %10 → `FAIL` (deney biter); **%10–30 → `INCONCLUSIVE_REDESIGN`** (kurallar yeniden tasarlanır, test başlamaz); < 100 deneme → `INSUFFICIENT_DATA` |
| **Mutlak net** (test) | FOLLOW'un maliyet sonrası ortalama getirisinin pencere kümeli **ve** gün kümeli %95 alt sınırı > 0 |
| **Baseline üstünlüğü** (test) | Eşleştirilmiş farkın (FOLLOW − B1) aynı iki alt sınırı > 0 |
| **Avantaj** | **İkisi birden** sağlanmalı. Takipçi −%2, baseline −%5 → fark +%3, ama avantaj **yok** (test var) |

**Güç hesabı:**
- (1,96/X)² yaklaşımı kaldırıldı.
- Gerekli bağımsız pencere sayısı = ((z₀,₉₇₅ + z₀,₈₀) · σ_w / δ)².
  - σ_w: **pencere ortalamalarının** standart sapması. Pencere içi bağımlılığı ve değişen katılımı içerir.
  - δ: asgari ekonomik anlamlı etki, **%3 net getiri/maliyet** (önceden kayıtlı). Bunun altındaki bir avantajı, modellenmemiş sürtünmeler (gecikme varyansı, kuyruk sırası) muhtemelen yer.
- Örnek: σ_w = 1 → **8.721 pencere**. Gün bağımlılığı gün kümeli aralıkla ayrıca denetlenir.
- σ_w ve gereken pencere sayısı **pilottan** tahmin edilir ve test başlamadan **dondurulur** (`--required-windows N`). Test verisine bakılarak değiştirilemez.
- 1.000 pencere / 14 gün **otomatik kanıt değildir**. 14 gün yalnızca alt sınırdır.

**Analiz ufku:**
- Test süresi, dondurulan pencere sayısına ulaşınca biten **tek bir final analizdir** (`--final`).
- Ara raporlar `INTERIM_NO_VERDICT` yazar; **başarı için erken durdurma yok**.
- Erken durdurma yalnızca zarar ya da uygulanamazlık için (§7).

## 7. İşletim

- **Başlama koşulları:**
  - Kullanıcının açık onayı.
  - `<out>/APPROVED_RULES_FINGERPRINT` dosyası v2 kural parmak iziyle eşleşmeli (onaydan sonra yazılır).
  - NTP belirsizliği ≤ 50 ms.
  - En az 1 GB boş disk.
  - Profil-cüzdan bugün doğrulanmış olmalı.
- **Durdurma:**
  - Süre doldu ya da disk tavanı aşıldı.
  - 401/403/451: hemen ve kalıcı durur, kısıt aşılmaz.
  - Arka arkaya 5 kez 429.
  - Websocket 10 dakikada 5'ten fazla koptu.
  - Kimlik değişti.
  - Operatör istedi.
- **Yeniden bağlanma:** 1→60 sn üstel bekleme. Her bağlantı sıfırdan başlar; tam snapshot gelene kadar defter kullanılmaz. Boşluklar sağlık kaydına yazılır.
- **Mevcut sisteme etki:** Pilot CORE anahtarı kullanmaz, CORE'a bağlanmaz. Günlük kontroller: cron jobid 103, `brian_prediction_*` satır artışı, `wallet_tracking_enabled = false`.
- **Rapor içeriği:**
  - Gecikme: ölçülen sayı, p50/p90/maks, **pencere kapandıktan sonra görülen** ve **birincil d'de geç kalan** sayılar ayrı.
  - d başına ret nedenleri; ücret, kayma, spread.
  - Mutlak net ve üstünlük, ayrı ayrı ve aralıklarıyla.
  - Güç: σ_w ve önerilen gereken pencere sayısı.
  - `h2: NOT_TESTED`, `tradeSignal: false`.

## 8. En küçük ücretsiz pilot ve koşullar

| Konu | Öneri |
|---|---|
| Pilot | BTC 5 dk, 12–24 UTC, 7 gün, yerel; yalnızca uygulanabilirlik ve σ_w tahmini |
| Sonrası | `PASS` ve onay → gereken pencere sayısı dondurulur → ayrı test; `INCONCLUSIVE_REDESIGN` → yeniden tasarla; `FAIL` → dur |

| Bilinen sınır | Durum |
|---|---|
| Oracle | Belgelenmiş ama kimlik doğrulamalı akış var; kullanılmıyor, spot proxy kullanılıyor; seri eşleşmesi doğrulanmadı |
| Gecikme | n = 6 ön ölçüm; genellenemez |
| Emir gecikmesi | 300 ms varsayım; ölçülmedi |
| Kuyruk önceliği | Ask'ı görmek dolum garantisi değil; tolerans ve tam miktar kuralı bunu yalnızca kısmen yansıtıyor |
| Kaynak tüketimi | Tahmin; ölçülmedi |
| H2 | Test edilmiyor |
| `runtime.ts` canlı döngüsü | Hiç çalıştırılmadı; ilk onaylı başlatmanın ilk saati aynı zamanda ilk canlı testi |
