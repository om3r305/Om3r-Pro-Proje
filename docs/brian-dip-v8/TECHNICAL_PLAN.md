# V8 doğruluk ve uygulama planı

**Tasarım önerisi; uygulanmış kod değildir.** Kaynak bulguları [özgün incelemede](evidence/Brian-V8-Inceleme-2026-09-07.md), kısıtlar [karar kaydında](DECISIONS.md). Aşağıdaki alan ve fonksiyon adları sözleşmeyi anlatır; yeni migration/RPC'nin mevcut olduğu anlamına gelmez.

## T1 — UI, session ve çalışma modu

İlgili yüzeyler: `monster-coins-pro/dip.html`, `dip-focus3-stability-v7.js`, V8 içeren `dip-foresight-v7.js`, `dip-server-authoritative-v7.js`, `supabase/functions/brian-dip-trader/index.ts`.

- [ ] V8 sayfasından eski Focus-3 yüklemesini çıkar. Gerekli çizim hız sınırını V8'e taşı; geçersiz `v7Foresight` referansı ve XRP/ETH/DOGE restore sarmalayıcıları kalmasın.
- [ ] Session sözleşmesini sürümle. `engine_version`, ETHUSDT, server authority, browser execution ve sizing policy API'de sessizce atılmasın. Eski `universe_size >= 6` zorlaması V8'e uygulanmasın. Sunucu uyumsuz ayarı açıkça reddetsin.
- [ ] `OBSERVE` (yeni sanal pozisyon açmaz) ile `SHADOW_PAPER` (yalnız sanal defter geçişleri) açıkça ayrılsın. Her ikisinde gerçek emir yasak. Yeni giriş kapısı sunucuda denetlensin.
- [ ] Açık pozisyon varken observe/pause geçişi onu silmesin. İzleme, kapanış ve mutabakat yükümlülüğü ayrı tanımlansın; eski sürüme kör rollback yapılmasın.
- [ ] DIP ekranında sürüm, session, son başarılı piyasa verisi/snapshot yaşı, çalışma modu ve WAIT nedeni gösterilsin. Control Center değişmesin.

Kabul: gerçek HTML yükleme sırasıyla browser testi; start/restart/restore/resume sonrasında tek ETH; V7 config reddi; sayfanın açılması ölçüm kaydı yazmasın. Service worker/cache varsa aynı UI sürümünü sunduğu doğrulansın.

## T2 — Önce defter bütünlüğü, sonra state machine

İlgili yüzey: `supabase/functions/brian-dip-shadow-worker/index.ts`, DIP session/lease ve ileride eklenecek V8'e özel transaction katmanı.

### Tek muhasebe sözleşmesi

`latestRuntime()` için üç durum ayrıdır: yeni ve gerçekten boş session, okunmuş mevcut durum, okuma hatası. Hata veya açıklanamayan eksik kayıt `READ_FAILED/RECONCILIATION_REQUIRED` üretir; başlangıç kasasına dönüş ve pozisyonu null kabul ederek yeni giriş yasaktır.

Önerilen yapı: append-only geçiş defteri ve onunla **aynı DB transaction'ında** güncellenen session durumu. Son durum yeniden oynatmayla doğrulanabilir. Saatlik runtime/snapshot raporlama görünümü muhasebenin tek kurtarma kaynağı olamaz.

Bir geçiş en az `session_id`, `expected_state_version`, `lease_epoch`, `transition_id`, `occurrence_id`, varsa `position_id`, `market_cursor`, karar/sonuç politikası sürümü ve muhasebe etkilerini taşır. Geçiş kimliği aynı mantıksal işlem için retry boyunca değişmez; rastgele worker invocation ID'sinden türetilmez.

Önerilen transaction adımları:

1. Aynı `transition_id` daha önce aynı payload ile commit edilmişse önceki sonucu döndür. Farklı payload ile aynı ID bir tutarlılık hatasıdır.
2. Session durum satırını kilitle veya eşdeğer seri geçiş güvencesi uygula. Commit sırasında version, aktif session, pause/giriş yetkisi ve lease epoch doğrula.
3. İzinli state geçişini, tez kilidini, veri zamanı ve pozisyon limitlerini doğrula. Stale worker yeni bir kayıt commit edemesin.
4. Geçiş olayı, ücret/kasa/pozisyon etkisi, tez yaşam döngüsü ve gerekli tekil forecast intent'i ile güncel durumu birlikte yaz; version artır. Bir parça hata verirse tamamını geri al.
5. Snapshot/raporlama türevleri commit edilmiş defterden üretilebilsin. Asenkron yazım gerekiyorsa transaction'a bağlı outbox ve idempotent tüketici kullanılsın; best-effort çift POST kullanılmasın.

Borsa/network çağrısı DB transaction'ı içinde yapılmaz. Version çatışmasında eski karar körlemesine yeni duruma yazılmaz; yeni durum ve tazelik okunarak tekrar değerlendirilir. Runtime/RPC yazma yetkisi yalnız uygun sunucu rolüne verilir; browser doğrudan muhasebe olayları ekleyemez.

### Kimlikler ve değişmezlik

| Kimlik / kayıt | Anlamı ve kuralı |
|---|---|
| `structure_id` | Yapısal aile/seviyeler; tek başına bağımsız örnek kimliği değil |
| `episode_id` | Aynı piyasa olayından türeyen korelasyonlu karar grubu; split ve belirsizlik hesabında kullanılır |
| `occurrence_id` | Karar anında oluşmuş değişmez tez örneği; yeni gerçek tetikleyici yeni oluşum açar |
| `thesis_revision` | Güncel aday görünümünün sürümü; eski karar özelliklerini geriye dönük değiştirmez |
| `position_id` | Açılmış gerçek sanal defter pozisyonu; giriş planı/maliyet modeli dondurulur |
| `transition_id` | Aynı mantıksal durum geçişinin retry kimliği; DB'de tekillik zorunlu |
| `forecast_id` | Oluşum + ölçüt sürümü + önceden belirlenmiş ufuk/politika için tek ölçüm |

Meşru yeniden girişler defterde ayrı işlemdir; bağımsız eğitim örneğiymiş gibi sayılmaması için episode ile gruplanır. Gerçek işlemleri tek satırda birleştirerek kayıpları gizlemek veya yalnız fiyat hash'i değişmedi diye yeni gerçek olayı yutmak yasaktır.

### Durum geçişleri

| Önce | Olay / kontrol | Sonra | Zorunlu sonuç |
|---|---|---|---|
| Aday | Henüz yapı teyitli değil veya veri yetersiz | Gözlem | Neden kaydı; sanal giriş yok |
| Teyitli oluşum | Veri, veto, maliyet, geometri, session, lease ve boyut kapıları açık | Pozisyon açık | Bir giriş ve dondurulmuş plan |
| Pozisyon açık | Hedef, iptal veya süre sonu kesinleşti | Pozisyon kapalı; oluşum tüketilmiş | Bir kapanış; kilit aynı transaction'da |
| Tüketilmiş/iptal olmuş oluşum | Aynı worker turu veya aynı tetikleyici | Kilitli | Aynı oluşumla yeniden giriş yok |
| Kilitli yapı | Yeni teyitli yapı olayı ve yeni kapanmış 5m tetikleyicisi | Yeni oluşum değerlendirmesi | Zamanın tek başına geçmesi kilidi kaldırmaz |
| Herhangi bir durum | Okuma/yazma hatası, lease kaybı, açıklanamayan state | Mutabakat bekliyor | Mevcut kanıt korunur; yeni giriş kapalı |

Uygulama ilk sürümünde kapanıştan sonra o worker turunda tüm yeni girişleri ertelemek sade bir seçimdir. En azından aynı oluşum için bu yasak zorunludur. LONG için `stop < gerçekleşebilir giriş < hedef`, SHORT için ters sıralama aranır. `abs(entry-stop)` yön kontrolünün yerine geçmez.

Kabul: kasa 700/açık pozisyon + read error kasayı 1000 yapmaz; BUY yazımı sonrası hata ve timeout/retry ikinci BUY üretmez; eşzamanlı iki worker tek geçiş yapar; lease kaybı/pause yarışında yetkisiz giriş commit olmaz; hedef/stop/expiry yalnız bir kez kapatır; aynı turda ölü oluşum açılmaz. Bunlar disposable gerçek Postgres ile de sınanmalıdır; MemoryDB bunun yerine geçmez.

## T3 — Ortak piyasa yolu, zaman ve sonuç değerlendirme

Worker ve foresight aynı sürümlü evaluator'ı çağırır. Girdi: venue, yön, bariyerler, etkinleşme/son tarih, sıralı piyasa gözlemleri ve son işlenmiş cursor. Çıktı: sonuç, olay zamanı, kanıt aralığı, çözüm politikası sürümü ve belirsizlik nedeni. Tam olarak aynı girdi aynı sonucu üretir.

- [ ] Karar zamanı, piyasadaki olay zamanı, verinin sisteme kullanılabilir olduğu zaman ve kaydın yazıldığı zaman ayrı tutulsun. `observed_at`/`recorded_at` birbirinin yerine kullanılmasın.
- [ ] Her aralık için sınır dahil etme kuralı sürümlensin. Karar öncesi ve son tarih sonrası fiyatlar etikete giremesin. İlk oluşum dakikası toptan atlanmasın.
- [ ] Kısmi başlangıç/bitiş mumunda zaman damgalı trade yolu kullan veya kanıtın yetersizliğini açıkça kaydet. Tüm mumun high/low değerini kısmi aralığa mal etme.
- [ ] İki bariyer aynı mumda görülüyor ama sıra bilinmiyorsa `AMBIGUOUS`; eksik/gap veri `INDETERMINATE` olsun. Bunlar kesin HIT değildir. Değerlendirmeden sessizce çıkarılıp hit oranı şişirilemez; sayıları ve iyimser/kötümser sınırlar raporlanır.
- [ ] Canlı ara gözlem ile nihai etiket ayrı olsun; henüz gerçekleşmemiş mum kapanışına `resolved_at` yazılamasın. `event_at`, çözüm zamanı ve deadline farkları saklansın.
- [ ] Worker aradaki hedef/stop dokunuşlarını mevcut midpoint'e bakarak kaybetmesin. Kesinti/pause sonrası eksik yol tamamlanamıyorsa bilinmeyen sonuç kesin sonuca çevrilmesin.

**Tez tahmini ile işlem getirisi farklı ölçütlerdir.** Aynı evaluator kullanılsa bile forecast karar anında, pozisyon daha sonra dolmuş olabilir; fill, süre, spread ve maliyet farkı nedeniyle sonuçlar meşru olarak ayrışabilir. Eşitlik testi aynı etkinleşme zamanı ve bariyerler içindir. Diğer farklar `NO_FILL`, giriş gecikmesi veya maliyet gibi açık gerekçeyle raporlanır; zorla eşitlenmez.

Kabul: intrabar hedef; stop; ilk dakika; kısmi son dakika; açık mum; aynı mum iki bariyer; boş aralık; sıra/dedup; geç gelen veri; pause/resume ve deterministic replay testleri. Eski forecast'lerin yeniden çözümü [araştırma planındaki](RESEARCH_PLAN.md) sürümlü karşılaştırma ile yapılır; geçmişin üzerine yazılmaz.

## T4 — Tek tez görünümü ve sunucuda ölçüm

- [ ] Karar anında yapı, giriş aralığı, hedef/iptal, raw conviction, calibration sonucu/veri kümesi sürümü, akış ve veri kalitesi değişmez kayda bağlansın. Özellikler sonuçtan sonra güncellenmesin.
- [ ] Güncel aday görünümü ayrı sürümlensin; DB'deki eski calibration/entry worker'ın güncel hesabıymış gibi sunulmasın.
- [ ] Açık pozisyon çizgileri dondurulmuş pozisyon planından çizilsin. Son aday ve forecast çizgileri kullanılıyorsa ayrı ve açık etiketlensin.
- [ ] Forecast karar olayından sunucuda üretilsin. Yalnız `latestThesis` taramasıyla aradaki adaylar kaybedilmesin; karar cursor'u/outbox tüketimi olsun.
- [ ] Oluşum/ölçüt/ufuk/politika tekilliği DB constraint'iyle korunsun. `SELECT exists` ardından rastgele `INSERT` yeterli değil.
- [ ] Browser metrikleri salt okusun. Sunucu scheduler ve retry, browser kapalıyken de çalışsın. Mevcut migration'da cron tanımı var; üretimde çalıştığı ayrıca doğrulanacak.

## T5 — Veri sağlığı, calibration, risk, venue ve maliyet

**Veri:** kaynak/venue kimliği, gerçek `closeTime`, yeterli warm-up, yaş, sıra, tekrar, mum aralığı boşlukları, timeframe gecikmesi, trade zamanı, book spread/bid/ask ve son başarılı veri zamanı kontrol edilir. Eşikler config/manifest ile sürümlenir; evrensel sabit değer diye sunulmaz. `NO_DATA`, `STALE_DATA`, `NO_SIGNAL`, `CONFLICT`, `CALIBRATION_UNAVAILABLE`, `COST_HURDLE` ayrıdır.

**Calibration:** örnek sayısı yeterlilik ve belirsizlik içindir; tek başına risk kapısını gevşetmez. Sorgu hatası sıfır örnek veya sıfır olasılık sanılmaz. Setup/yön/venue/rejim ve model/etiket sürümüyle tanımlanan aynı, sıralı örnek kümesi UI ve worker'da kullanılır. Kötü sonucu doğrulanmış grup gerekçeli veto/observation üretir. 40/40 kayıp senaryosu sırf `n >= 40` diye işleme geçemez. Ham 0.66 puanı kalibre edilmiş %66 başarı olasılığı değildir.

**Boyut:** önce izinli kayıp bütçesi, geçerli stop mesafesi ve maliyet; sonra notional/equity üst sınırı, kullanılabilir nakit ve miktar hassasiyeti uygulanır. Yuvarlama risk/notional sınırını yükseltemez. Minimum notional karşılanmıyorsa WAIT. Cash 50 ve %8 üst sınırla 10 notional açılamaz. `actual_fraction = gerçekleşen notional / tanımlanmış equity` yazılır; cash ve mark-to-market equity karıştırılmaz. Pozisyon riskinin stopta sınırlanması gap/slippage kaybını garanti etmez; stres varsayımı ayrıca tutulur.

**Venue:** ilk temiz deney için önerilen kapsam Spot LONG; aşağı yön adaylar gözlem olarak kalır. `allow_shadow_short=false` kesin uygulanır. Spot veriyle PERP işlem etiketi kullanılamaz. İleride SHORT ancak ayrı ETH PERP adaptörü, aynı venue book/trade/candle, funding/basis ve kapanış modeli ile ayrı deney olur; SHADOW sınırı değişmez.

**Maliyet:** alış ask, satış bid tarafı ve tanımlı slippage/impact ile modellenir. Slippage fill fiyatına uygulandıysa ayrıca ücret gibi tekrar kesilmez. Giriş/çıkış ücret ve venue politikası pozisyon üzerinde dondurulur; sonraki adayın yönüne göre değişmez. Net P&L ile ücret/finansman/varsayım dökümü replay ile eşleşir.

Kabul: eksik/eski/ters sıralı veri giriş açmaz; karşıt akışın etkisi açık politikaya uyar; bozuk calibration sorgusu veto sebebidir; min-notional/cap ve rounding senaryoları; short-disable; sonraki aday değişirken açık pozisyon ücreti sabit; gidiş-dönüş maliyette çift slippage yok; V8 worker'ın kendi testinde gerçek emir çağrısı sıfır.

## Uygulama PR'larında kanıt

Her T paketi için değişen dosyalar, düzeltilen senaryolar, yeni davranış beklentisi ve gerçek çalıştırma sonucu kaydedilsin. Tarihsel harness'in 17/17 sonucu yeni sürümün regresyon sonucu olarak kullanılamaz. Mevcut CI'ın yeşil olması yeterli değildir: V8 worker/resolver/session için type-check, davranış testleri, gerçek browser sözleşmesi ve atomik/lease işleri için gerçek yerel Postgres testi eklenmelidir. Test için canlı DB gerekmez.
