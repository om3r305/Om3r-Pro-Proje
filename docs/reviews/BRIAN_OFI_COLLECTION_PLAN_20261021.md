# Emir akışı dengesizliği (OFI) — veri toplama ve test planı (BAŞLATILMADI)

Tarih: 21 Ekim 2026. Durum: **yalnızca plan.** Veri toplama kullanıcı onayı olmadan başlamaz.

## 1. Makale ve iddialar

**Makale (künyesi doğrulandı):** Cont, Kukanov, Stoikov — "The Price Impact of Order Book Events", *Journal of Financial Econometrics* 12(1): 47–88, 2014; arXiv:1011.6402. Veri: NYSE TAQ, 50 hisse.

**Doğrulanmış iddia (özet sayfası):** Kısa aralıklarda fiyat değişimini çoğunlukla en iyi alış/satış seviyesindeki emir akışı dengesizliği (OFI) belirler. İlişki doğrusaldır; eğim piyasa derinliğiyle ters orantılıdır. Sonuç farklı zaman ölçeklerinde ve hisselerde geçerlidir.

**Ön koşul:** OFI'nin tam tanımı, regresyon biçimi, aralık uzunlukları ve raporlanan açıklama gücü, veri toplanmadan **önce** makale metninden deftere (`research/research_lab/ledger/arxiv-1011.6402.json`) yazılacak. PDF metni otomatik çıkarılamadı; elle okunacak. Ezberden sayı kullanılmayacak.

| İddia | Kaynağı | Para ile ilgisi |
|---|---|---|
| **C-01:** Aynı aralıktaki OFI, orta fiyat değişimini doğrusal olarak açıklar; eğim derinlikle ters orantılı | Makale (eş zamanlı ilişki) | Doğrudan değil. Fiyat *aynı anda* hareket eder, önceden bilinmez |
| **C-02:** Bir önceki aralığın OFI'si bir sonraki aralığın fiyat değişimini, **maliyetten sonra** öngörür | **BRIAN'ın kendi hipotezi**, makalenin iddiası değil | Doğrudan. Para kazanma sorusu bu |

**Dürüst beklenti:** C-01'in kripto piyasasında da büyük ölçüde tutması olasıdır. C-02 ise çok daha zor: öngörülebilir hareket genelde spread ve komisyondan küçük kalır. C-02'nin `FAIL` çıkması da değerli bir sonuçtur; bu yoldan para kaybetmeyi önler.

## 2. Veri

- **Kaynak:** Binance herkese açık websocket `btcusdt@bookTicker` ve `ethusdt@bookTicker`. En iyi alış/satış fiyatı ve miktarı, her güncellemede gelir. Anahtar yok, ücretsiz, Almanya'dan erişilebilir; NIULAI4 pilotu bu akışı zaten kullanıyor.
- **Ham veri saklanmaz.** Saniyelik özet tutulur:
  - OFI toplamı ve güncelleme sayısı
  - Saniye sonundaki orta fiyat, spread ve L1 miktarları
  - Boşluk ve yeniden bağlanma bayrakları
  - `u` (update id) süreklilik kontrolü
- **Hacim (tahmin):** sembol başına ~86.400 satır/gün, ~7 MB/gün. 2 sembol × 14 gün ≈ 200 MB. Disk tavanı 500 MB, kod tarafından zorunlu.
- **Süre:** 14 gün. 10 saniyelik aralıkla sembol başına ~120 bin aralık ve ~670 yarım saatlik pencere eder.
- **Çalıştığı yer:** NIULAI4 pilotu ile aynı makine; şimdilik bilgisayar, bulut hesabı açılınca oraya.
  - CORE (Supabase) ve GitHub Actions uygun değil: biri uzun süreli websocket taşıyamaz, diğerinde Binance ABD'den 451 veriyor.
  - Pilotun sürecinden **ayrı** bir süreç olarak çalışır; pilotun kurallarına ve parmak izine dokunmaz.
- **CORE'a gönderilen:** Yalnızca günlük özet ve sağlık bilgisi, mevcut pilot ingest kalıbıyla (opsiyonel, küçük).

## 3. Ön kayıtlı analiz

Kurallar, veri toplanmadan önce commit'lenir.

- **C-01 (makaleye sadık):**
  - Her yarım saatlik pencerede ΔP₁₀ₛ ~ OFI₁₀ₛ regresyonu, makaledeki biçimiyle.
  - Raporlananlar: pencerelerin medyan R²'si, pozitif eğimli pencere oranı, eğim ile 1/derinlik arasındaki ilişki.
  - `PASS` eşikleri, makalenin raporladığı değerlerden önceden türetilir; sonuçtan sonra değiştirilmez.
- **C-02 (BRIAN hipotezi):**
  - OFIₜ ile ΔPₜ₊₁ (sonraki 10 sn) arasındaki ilişki.
  - Basit kural: OFI eşiği aşarsa, spread'i karşılayan taker işlem; sanal ve tamamen gölge.
  - Maliyet: gerçek spread + Binance taker komisyonu (o günkü resmi oran).
  - Ölçüt: gün kümeli bootstrap ile %95 aralığın alt sınırı > 0 bps (maliyet sonrası).
  - İlk 7 gün kural ayarı, son 7 gün **dokunulmamış test**. Aynı veride ayar ve test yapılmaz.
- **Durumlar:** Research Lab ile aynı (`PASS` / `FAIL` / `UNVERIFIED` / `INSUFFICIENT_DATA` / `REVIEW_REQUIRED`).
  - Kayıt boşluğu toplam sürenin %5'ini aşarsa sonuç `REVIEW_REQUIRED` olur.

## 4. Korumalar

- Emir yok, anahtar yok, gerçek para yok.
- Bağlantı koparsa üstel bekleme ile yeniden bağlanır; boşluk kaydedilir, asla doldurulmaz.
- Saat NTP ile düzeltilir. 401/403/451 alınırsa kalıcı durur.
- Disk tavanına ulaşınca durur.
- Kural parmak izi ve onay dosyası kullanılır (pilotla aynı kalıp).

## 5. İş ve onay

| Adım | Süre | Onay |
|---|---|---|
| Makaleden tanım ve sayıları deftere yazmak | ~1 saat | — |
| Toplayıcı (Deno, pilot altyapısı) + testler | ~1 gün | — |
| Analiz (Python stdlib) + testler + ön kayıt | ~0,5 gün | — |
| **Veri toplamayı başlatmak (14 gün)** | — | **Kullanıcı onayı gerekir** |
| Sonuç | 14 gün sonra | — |
