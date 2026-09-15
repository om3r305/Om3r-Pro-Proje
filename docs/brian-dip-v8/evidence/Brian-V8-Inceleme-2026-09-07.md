# Brian DIP V8 — Kod incelemesi ve geliştirme sırası

7 Eylül 2026 · Ömer için hazırlanmıştır.

**Karar: V8’in yapı okuma yaklaşımı korunmalı; mevcut sürümün işlem sonuçları ve öğrenme ölçümü, aşağıdaki doğruluk sorunları düzeltilmeden güvenilir kabul edilmemeli.** Öncelik yeni indikatör eklemekten önce session, tez, işlem ve değerlendirme zincirini tutarlı hale getirmektir.

İncelenen repo: [om3r305/Om3r-Pro-Proje](https://github.com/om3r305/Om3r-Pro-Proje), branch `brian-2026`, commit [`0c1b1e88e758a71e7cf9732300b77f8f6e2f597d`](https://github.com/om3r305/Om3r-Pro-Proje/commit/0c1b1e88e758a71e7cf9732300b77f8f6e2f597d). V8 [PR #76](https://github.com/om3r305/Om3r-Pro-Proje/pull/76) bu commit ile merge edilmiş. Snapshot saklama değişikliği [PR #77](https://github.com/om3r305/Om3r-Pro-Proje/pull/77), inceleme sırasında açık; V8 karar motorunu değiştirmiyor.

Kapsam V8 kodu ve ona bağlanan mevcut DIP arayüzü/session API’sidir. Üretim DB’sine sorgu, silme, migration, restart veya deploy yapılmadı; repo içeriği değiştirilmedi. Ekran görüntüsü DB bağlantısının o anda reddedildiğini gösterir; mevcut canlı DB/worker durumunu tek başına doğrulamaz. Phase 3.7, Control Center ve ana Brian öğrenme hafızası bu çalışmanın değişiklik kapsamı dışındadır. SHADOW ONLY ve ETHUSDT sınırları korunur.

**Kanıtın gücü ve sınırı**

PR #76’nın son head commit’inde GitHub’dan okunan üç CI sonucu başarılı: [Brian 2026 CI](https://github.com/om3r305/Om3r-Pro-Proje/actions/runs/34129035914), [ALPHA v2 CI](https://github.com/om3r305/Om3r-Pro-Proje/actions/runs/34129035834), [DIP V5 Isolation CI](https://github.com/om3r305/Om3r-Pro-Proje/actions/runs/34129035793). Bunlar V8’in aşağıdaki davranışlarını ispatlamıyor.

`test_brian2026_dip_server_authoritative_v7.py` içindeki 8 test fonksiyonu yerelde doğrudan çalıştırıldı ve geçti. Ortamda pytest bulunmadığı için bu sonuç bir pytest çalıştırması olarak sunulmuyor. Testler esas olarak kaynak metnindeki ifadeleri ve iki tarayıcı dosyasının sözdizimini denetliyor. CI’nin açık Deno kontrol listesinde V8 worker ve foresight dosyaları bulunmuyor.

Ayrıca gerçek kaynak dosyaları Node 24’te TypeScript sözdiziminden arındırılıp izole VM içinde yürütüldü: **15 sorun senaryosu ve 2 olumlu kontrol, toplam 17/17 senaryoda beklenen gözlem yeniden üretildi.** İşlem senaryolarında yapı algılayıcısının çıkışı sabitlendi; amaç giriş/çıkış, boyutlandırma ve kayıt davranışını ayrı sınamaktı. Pivot kontrolü gerçek `structure()` / `classifyPivots()` fonksiyonlarını kullandı. DB ve borsa bellek içi örneklerle temsil edildi; gerçek Postgres eşzamanlılık testi, gerçek tarayıcı testi, Deno type-check veya piyasa performans testi yapılmadı. Rakamlar sentetik fiyat/kasa örnekleridir; gerçek ETH fiyatı veya canlı işlem sonucu değildir.

**Korunacak doğru parçalar**

| Tasarım | Neden korunmalı? | Mevcut sınır |
|---|---|---|
| ETHUSDT ve temiz V8 session kontrolü | V7 geçmişi ile yeni deneyi karıştırmayı önlüyor. V7 config’i worker tarafından gerçekten reddediliyor. | Eski arayüz katmanları ETH sınırını tekrar bozabiliyor. |
| Sunucuda çalışan SHADOW worker | Telefon/sayfa kapanmasına bağlı işlem sahipliğini ortadan kaldırma yönü doğru. | Session, lease ve atomik kayıt sözleşmesi tamamlanmalı. |
| Native 1m/5m/15m/1h/4h verisi ve 3/3 pivot | Kapanmamış son mumun yapı kararını değiştirmemesi ve sağ tarafta üç mum beklenmesi olumlu kontrollerde doğrulandı. | Mum kapanışı zaman damgasıyla doğrulanmıyor; son satırın açık olduğu varsayılıyor. Pivot teyidinin doğal gecikmesi var. |
| HH/HL/LH/LL, BOS/CHOCH ve sweep seviyeleri | Dip fikrini somut fiyat yapısına bağlıyor. | Setup sırası, karşıt kanıt ve invalidation geometrisi eksik. |
| Tezde hedef/iptal, maliyet ve R:R | Denetlenebilir bir karar kaydı için doğru temel. | Aynı tez için worker, çizim ve ölçüm farklı durumlara kayabiliyor. |
| Ham görüş ile tarihsel isabetin ayrılması | Elle verilen puanın başarı olasılığı sanılmasını önlemeye çalışıyor. | Mevcut tarihsel oran gerçek anlamda koşullara göre kalibre edilmiş olasılık değil. |
| V7 ve V8 ölçütlerinin ayrılması; yapay gelecek mumların kaldırılması | Eski 8 dakikalık yön isabeti yeni hedef/iptal ölçümüne karışmıyor. | V8 zaman penceresi ve örnek tekilliği düzeltilmeli. |

**1. P0 — Eski arayüz katmanı V8 sözleşmesini bozuyor**

[`dip.html`](https://github.com/om3r305/Om3r-Pro-Proje/blob/0c1b1e88e758a71e7cf9732300b77f8f6e2f597d/monster-coins-pro/dip.html#L95), V8 içeriğini taşıyan `dip-foresight-v7.js` dosyasından sonra [`dip-focus3-stability-v7.js`](https://github.com/om3r305/Om3r-Pro-Proje/blob/0c1b1e88e758a71e7cf9732300b77f8f6e2f597d/monster-coins-pro/dip-focus3-stability-v7.js#L38) dosyasını yüklüyor. Bu dosya `const prevForesight=v7Foresight` satırında artık bulunmayan eski fonksiyona başvuruyor.

Yeniden üretim sonucu: **`v7Foresight is not defined`**. Üstelik hatadan önce `restore()` ve `start()` sarmalayıcıları kurulmuş oluyor. Restore çalıştırıldığında evren **XRPUSDT / ETHUSDT / DOGEUSDT** oluyor. Dolayısıyla hata, dosyanın tüm etkilerini geri almıyor.

Gerekli değişiklik: Eski Focus-3 yüklemesini V8 sayfasından kaldır; hâlâ gereken çizim hız sınırlamasını ve session kapsamlı görüntü desteğini V8’e taşı. Bu, eski öğrenme verisinin silinmesi anlamına gelmez. Restart payload’ı, API’nin sakladığı config, restore sonucu ve radar tek coin sözleşmesini birlikte doğrulasın. Backend `settings()` de V8 config’ini sürüme göre kabul etsin; bugün `server_authoritative` ve `browser_execution` gibi alanları atıyor, `universe_size` için eski 6 alt sınırını uyguluyor.

Kabul: Gerçek HTML yükleme sırası hata üretmez; restart/resume/restore sonrasında yalnız ETH vardır; UI worker sürümünü ve son başarılı snapshot yaşını gösterir.

**2. P0 — Pozisyon kapanışı ve tez kilidi aynı turda tutarlı değil**

[`worker run()`](https://github.com/om3r305/Om3r-Pro-Proje/blob/0c1b1e88e758a71e7cf9732300b77f8f6e2f597d/supabase/functions/brian-dip-shadow-worker/index.ts#L28), kilidi mevcut pozisyonu kapatmadan önce kontrol ediyor. Kapanış sırasında yeni kilit koyuyor ama giriş koşulunu bu yeni kilitle tekrar değerlendirmiyor.

Yeniden üretim: 91 dakikalık pozisyon `THESIS_EXPIRED` ile kapandı, aynı çalışmada **SELL → BUY**, aynı `thesis_id` ile yeniden açıldı. Yeni 5m kapanışı veya yapı gerekmemiş oldu.

Ayrıca invalidation kontrolü `Math.abs(px-inv)` kullanıyor; stop’un yön bakımından doğru tarafta olmasını aramıyor. Son kapanmış mumda reclaim oluşup güncel fiyat yeniden seviyenin altına indiğinde **100.020001 giriş, 100.5 stop, 103 hedef** ile LONG açılabildi.

Gerekli değişiklik: Kapanış sonrası aynı tez için giriş kesin olarak engellensin; kilit giriş anında tekrar doğrulansın. LONG için `stop < gerçek giriş < hedef`, SHORT için ters sıralama zorunlu olsun. Spread/slippage sonrası gerçek giriş kullanılsın. Piyasa yapısının kimliği ile her bağımsız tez oluşumunun kimliği ayrı tutulsun; kapanmış tez yeni bir olay olmadan tekrar canlanmasın.

Kabul: Stop, hedef ve süre dolumunda bir kez kapanış; aynı turda aynı tezle yeniden giriş yok; ters seviyeler açık WAIT nedeni üretir.

**3. P0 — İşlem motoru ile ileri görüş farklı zamanları ölçüyor**

Worker stop/hedef için yalnız o andaki book midpoint’i kontrol ediyor. Aradaki mumun yüksek/düşük fiyatlarını işlem kapanışına uygulamıyor. Yeniden üretimde mum **104** gördü, hedef **103**, güncel fiyat **100** oldu; pozisyon açık kaldı. [`foresight resolveDue()`](https://github.com/om3r305/Om3r-Pro-Proje/blob/0c1b1e88e758a71e7cf9732300b77f8f6e2f597d/supabase/functions/brian-dip-foresight/index.ts#L24) ise high/low ile sonucu değerlendiriyor. Böylece tezin isabeti ile aynı tezin işlemi ayrışabilir.

Resolver’ın üç zaman kusuru da yeniden üretildi:

- Tahminin oluştuğu dakikanın tamamını atlıyor. Tam dakika başında üretilse bile bir sonraki dakikadan başlıyor.
- Henüz kapanmamış mumdan sonuç çıkarıp `resolved_at` değerini mumun sonuna yazabiliyor: saat **12:00:20** iken sonuç zamanı **12:00:59.999** olabiliyor.
- Süre **12:00:20**’de bitse bile 12:00 mumunun sonraki 40 saniyesini içeren high/low değerini kullanabiliyor; bu veri, hedefin son tarihten önce vurulduğunu ispatlamaz.

Binance Spot verisinde aggregate trade zamanları ve kline açılış/kapanış zamanları bulunur; sözleşme bu zamanlara dayanmalı. [Binance Spot market-data belgeleri](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market)

Gerekli değişiklik: Tez, giriş ve ölçüm için aynı etkinleşme zamanı ve fiyat kaynağı kullanılsın. Son işlenen piyasa zamanından itibaren bariyerler sırayla değerlendirilsin. Dakikanın kısmi başlangıç/bitişinde işlem zaman damgalı trade verisi veya açıkça tanımlanmış muhafazakâr bir politika kullanılsın. Aynı mumda hem hedef hem stop için gözlem sırası bilinmiyorsa kesin isabet üretilmesin; belirsizlik ayrıca raporlansın. Açık mum üzerinde ara izleme ile kapanmış kanıt ayrı olsun.

Kabul: Worker ve evaluator aynı sabit piyasa akışında aynı bariyer sonucunu verir; gelecekte `resolved_at` yok; ilk/son dakika yanlış etiket üretmez.

**4. P0 — Kayıt hatası sanal kasayı ve geçmişi ayırabiliyor**

[`latestRuntime()`](https://github.com/om3r305/Om3r-Pro-Proje/blob/0c1b1e88e758a71e7cf9732300b77f8f6e2f597d/supabase/functions/brian-dip-shadow-worker/index.ts#L24) sorgudaki `q.error` değerini kontrol etmiyor. Yeniden üretimde kayıtlı kasa **700** ve açık pozisyon varken okuma hatası, kasa **1000**, pozisyon **null** olarak döndü.

Event ve snapshot ayrı isteklerle yazılıyor; event kimlikleri rastgele. BUY yazılıp snapshot yazımı hata verdiğinde sonraki deneme pozisyonu geri yükleyemiyor. Bellek içi hata senaryosunda **iki farklı BUY event’i ve tek snapshot** oluştu.

Gerekli değişiklik: Okuma hatası `NO_STATE/READ_FAILED` ile yeni işlem açılmasını durdursun; gerçekten ilk session ile kaydı okunamayan session ayrılmalı. Tez oluşumu, işlem geçişi, ücret/kasa ve son durum aynı işlem bütünlüğü içinde veya yeniden oynatılabilir tek bir olay zinciriyle kaydedilsin. Her geçiş deterministik kimlikle tekrar denendiğinde ikinci kez işlenmesin. Commit anında session’ın hâlâ aktif olduğu ve worker’ın lease sahibi olduğu doğrulansın. Lease kaybında mevcut kod yalnız heartbeat timer’ını durduruyor; çalışan karar akışını iptal etmiyor. Pause da çalışmanın başında bir kez okunuyor; bu ikisi kod incelemesiyle saptanan, ayrıca üretimde denenmemiş yarış riskleridir.

Bu bir veri temizliği önerisi değildir; V8 sanal portföyünün doğru kalması için çalışma sözleşmesidir.

Kabul: Okuma/yazma hatası ve tekrar deneme mevcut açık pozisyonu kaybettirmez, sahte kasa sıfırlaması veya çift işlem üretmez.

**5. P1 — Öğrenme başarısızlığı karar kapısına yeterince yansımıyor**

[`calibration()`](https://github.com/om3r305/Om3r-Pro-Proje/blob/0c1b1e88e758a71e7cf9732300b77f8f6e2f597d/supabase/functions/brian-dip-shadow-worker/index.ts#L25) setup/yön/venue için geçmiş hit oranını okuyor. 40 örnekte oran hesaplanıyor, R:R alt sınırı 2’den 1.6’ya geçiyor. Başarı oranının yeterli olması için bir kapı bulunmuyor; pozisyon formülü en az %5 seçiyor.

Yeniden üretim: **40 örnek, 0 başarılı sonuç, olasılık 0** iken kasa 1000 üzerinden **50 notional** ile yeni LONG açıldı. Ayrı senaryoda akış tamamen satış yönündeyken (`ofi=-1`) sweep’in taban ham puanı **0.61** giriş eşiğini geçti. Mevcut akış bilgisi bir destek puanı; zorunlu teyit veya karşıt kanıt cezası değil.

Gerekli değişiklik: Öğrenme çıktısı sabit ve sürümlü setup/rejim/venue gruplarında, belirsizlik aralığı ve maliyet sonrası beklenen sonuçla değerlendirilsin. Yetersiz örnek, veri hatası ve ölçülmüş kötü performans ayrı durumlar olsun. Kötü setup’ın observation-only kalması veya açık bir veto üretmesi tanımlansın. `n=40` tek başına eşiği gevşetmesin. Giriş yapılan ve veto edilen adaylar ayrılarak yanlış veto/kaçan fırsatlar da ölçülsün. Karşıt/missing flow nasıl ele alınacak, bu bir açık politika olsun.

Worker en fazla 500 sonucu sırasız alıyor; arayüz istatistiği son 300 sonucu sıralı alıyor. Rejim ve raw conviction dilimi filtrelenmiyor. Bu yüzden ekrandaki oranla boyutlandırma oranı da farklılaşabilir. Ölçüm, veri seçimi ve politika versiyonları ortaklaştırılmalı.

Kabul: 40 başarısız sonuç deneyi daha güçlü hale getirmez; doğrulanmış kötü grup nedenli WAIT üretir; gösterilen ve kullanılan kalibrasyon aynı örnek kümesidir.

**6. P1 — “Tek tez” kayıt, ekran ve gerçek pozisyon boyunca korunmuyor**

Worker, `thesis_id` veya `thesis_state` değişmedikçe yeni tez kaydetmiyor. ID yalnız sembol/setup/yön/iptal/hedef içeriyor. Giriş aralığı, akış, conviction ve kalibrasyon değişse de DB’deki son tez eskide kalabiliyor.

Yeniden üretim: Aynı tez sürerken örnek sayısı **39 → 40**, worker olasılığı **null → 1** oldu; DB’de tek tez kaydı ve **null** olasılık kaldı. Giriş aralığı da eskiydi. Foresight bu eski tez tablosunu, işlem motoru güncel hesaplamayı kullanıyor. UI ayrıca açık pozisyonun dondurulmuş stop/hedefini temel almak yerine güncel foresight seviyelerini çiziyor.

Foresight, yalnız en son tezi tarıyor; aynı session/tez/ölçüt için bir kere kayıt açıyor. Aynı yapı daha sonra meşru biçimde tekrar oluşursa yeni örneği kaçırabilir; aynı anda iki çağrı yapılırsa ikisi de kayıt yok görüp iki örnek açabilir. Bellek içi eşzamanlı senaryoda **aynı tez için iki foresight kaydı** oluştu; migration’da bunu engelleyen birleşik unique constraint yok. UI her 10 saniyede çağırdığı POST endpoint’inde hem resolver hem persist çalıştığı için browser görüntülemesi de ölçüm yazımı başlatıyor.

Gerekli değişiklik: Değişmez tez oluşumu kaydı + sürümlü güncel görünüm + pozisyonun açıldığı anda dondurulan plan açıkça ayrı tutulsun. Foresight örneği worker’ın karar anında, tekil oluşum kimliğiyle oluşturulsun. Browser yalnız sonuç okusun; resolver sunucuda, tekrar çalıştırmaya dayanıklı olsun.

Kabul: Grafikteki aktif işlem seviyeleri işlem defteriyle eşleşir; açılan her uygun tez oluşumu bir kez ölçülür; ekran açmak örnek sayısını değiştirmez.

**7. P1 — SHORT piyasası ve maliyet hesabı tutarsız**

Worker SHORT’a `venue=PERP` yazıyor; bütün fiyat, mum, book ve trade verisi `/api/v3/...` Spot uçlarından geliyor. [`Binance USDⓈ-M kline belgesi`](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/market-data/rest-api/Kline-Candlestick-Data) vadeli piyasa için ayrı `/fapi/v1/klines` sözleşmesini tanımlar.

Yeniden üretimde `allow_shadow_short=false` config’i olmasına rağmen PERP etiketli SHORT açıldı. Funding, basis ve vadeli veri tazeliği kontrolü yok. Kapanış ücreti mevcut pozisyonun venue’sinden değil, o turdaki yeni adayın yönünden türetiliyor. Giriş slippage’ı hem giriş fiyatına hem ücret benzeri kesintiye eklenirken çıkış midpoint’ten hesaplanıyor; maliyet modeli simetrik değil.

Gerekli değişiklik: İlk temiz deneyde Spot LONG ve aşağı yönlü gözlemle sınırlı kalmak en sade seçenek. ETH perp SHORT isteniyorsa ayrı piyasa adaptörü, funding/basis, gerçek bid/ask kapanışı ve pozisyon üzerinde dondurulmuş ücret varsayımları tamamlanmalı. SHADOW sınırı her iki seçenekte de aynıdır.

Kabul: SHORT kapalı ayarına uyulur; her sonuç kullandığı piyasa verisiyle aynı venue’ye aittir; ücret modeli giriş ve çıkışta tutarlı uygulanır.

**8. P1 — Boyut sınırı ve veri sağlığı sözleşmesi eksik**

`Math.max(10, cash*frac)` minimum notional uygulaması üst sınırı aşabiliyor. Yeniden üretim: kasa **50**, kalibrasyon dönemi üst sınırı **%8**, açılan notional **10 = %20**. `actual_fraction` ise hesaplanan teorik fraction’ı yazıyor; gerçekleşen oranı değil.

Gerekli değişiklik: Önce risk bütçesi ve stop mesafesinden miktar, sonra notional üst sınırı uygulanmalı; borsa minimumunun altında kalırsa WAIT olmalı. Emir minimumu üst sınırı büyütmemeli. Gerçek oran `notional/equity` olarak kaydedilmeli.

Kod incelemesinde mum/trade yaş sınırı, minimum yeterli mum, zaman boşluğu, sıralama, çapraz timeframe gecikmesi ve book geçerliliği için merkezi kontrol bulunmadı. HTTP 200 olması temiz piyasa kanıtı sayılmamalı. Bunun için timestamp/closedTime, veri kaynağı ve beklenen frekans sözleşmesi eklenmeli. `NO_DATA`, `STALE_DATA`, `NO_SIGNAL`, `CONFLICT`, `COST_HURDLE` gibi nedenler DIP ekranında açıkça görünmeli; Control Center değişikliği gerektirmez.

**Uygulama sırası ve tamamlanma ölçütleri**

| Sıra | İş paketi | Bitti sayılması için |
|---|---|---|
| 1 | V8 arayüz ve session sözleşmesi | Eski Focus-3 runtime hatası yok; start/restore ETH-only; doğru sürüm ve son veri zamanı görünür. |
| 2 | Pozisyon/tez geçişleri ve güvenilir kayıt | Aynı turda tekrar giriş yok; yönlü stop/hedef kontrolü; retry ve okuma/yazma hatası tek defter sonucunu korur. |
| 3 | Ortak fiyat yolu ve sonuç değerlendirici | İntrabar hedef/stop, ilk/son kısmi dakika, aynı mum belirsizliği ve pause boşluğu tutarlı. |
| 4 | Tez oluşumu, kalibrasyon ve risk kapıları | Tekil örnek; güncel görünüm; başarısız grup veto; gerçek notional sınırı; Spot/PERP ayrımı. |
| 5 | Yeni ETHUSDT V8 SHADOW deneyi | Aynı doğrulanmış kod sürümü UI/worker/resolver’da; migration ve temiz session operasyonel olarak doğrulanmış; browser kapalıyken smoke-test başarılı. |
| 6 | Prospektif karşılaştırma | Sabit politika sürümüyle net beklenti, maliyet, drawdown, MFE/MAE, tutma süresi, veto ve kaçan fırsat raporu; aynı veri/maliyet varsayımlı basit baseline’lar. |

Önce bu doğruluk paketleri uygulanmalı. Yeni coinler, yapay gelecek mumlar, ek ağırlıklı sensör komiteleri ve kanıt gelmeden agresif boyut artırımı bu aşamaya eklenmemeli. Eski V7 öğrenme verileri arşiv kanıtı olarak korunmalı; V8 performans hesabına karıştırılmamalı. Ana Brian öğrenme/audit/reliability/calibration kayıtları temizlik kapsamında silinmemeli.

PR #77 snapshot hacmini azaltmaya yönelik yararlı bir değişikliktir; bu rapordaki karar, kilit, zamanlama, tekillik ve kalibrasyon sorunlarını çözmez. DB kurtarma çalışması ayrı yürürken V8 doğruluk düzeltmeleri bellek içi/sentetik senaryolarla hazırlanabilir. Canlı doğrulama ve yeni session, çalışan DB ve uygulanmış migration gerektirir.

Başarı ölçütü sık işlem açması değildir: doğru ve tekrar üretilebilir sonuç defteri, net maliyet sonrası gözlenen avantaj ve belirsizliğin görünür olmasıdır. Bu inceleme kârlılık kanıtı üretmedi.

**Tekrar çalıştırma**

Eşlik eden doğrulama arşivinde `audit-repros.cjs` ve `audit-repro-results.json` bulunur. Kaynak repo yukarıdaki commit’e sabitlenmişken Node 24 ile `node audit-repros.cjs /absolute/path/to/repo` çalıştırılır. Program gerçek ağa/DB’ye erişmez. Başarılı bir sorun senaryosu, mevcut hatanın yeniden üretildiği anlamına gelir; düzeltme testinin geçtiği anlamına gelmez. Düzeltmeler uygulanırken ilgili beklentiler güvenli davranışı doğrulayacak regresyon testlerine çevrilmelidir.
