# V8 uzmanlık ve araştırma planı

**Aşağıdakiler sınanacak hipotezlerdir; kanıtlanmış performans veya uygulanmış özellik değildir.** Ön koşul [teknik plandaki](TECHNICAL_PLAN.md) defter, etiket, tekillik, veri ve risk doğruluğudur. Bozuk etiketle eğitip aynı bozuk etiketle test edilen bir model iyi görünüp yanlış öğrenebilir.

## Hedef ve ölçüm tanımı

Ömer'in **%70 ve üzeri kazanma oranı** hedefi korunur. Başarı iddiası; maliyet sonrası pozitif beklenti, sınırlı büyük kayıp/drawdown, anlamlı kapsama ve belirsizliği raporlanmış yeni veri sonuçlarıyla birlikte değerlendirilecektir. %70 bir performans vaadi veya hedef/stop geometrisini değiştirme izni değildir.

Bir işlem, girişten tamamen kapanışa kadar tek pozisyondur. Kısmi kapanışlar ayrı kazanımlar sayılmaz. Net gerçekleşmiş getirisi pozitif kapanış kazanım; negatif kapanış kayıp; sıfır sonuç ayrı raporlanır. Ana kazanma oranının paydası sıfır sonuçlar dahil tüm tamamen kapanmış uygun işlemlerdir. Açık pozisyonlar bu orana kapanmış işlem gibi katılmaz; mark-to-market equity, açık zarar ve yaşları ayrıca rapordan eksik edilmez. Deney bitişindeki açık pozisyon politikası başlangıçta dondurulur.

`Beklenti = p(kazanç) × ortalama net kazanç − p(kayıp) × ortalama net kayıp büyüklüğü`.

70 kez +10 ve 30 kez −30, maliyet öncesinde bile −200 eder. Sabit +2.5R/−1R ve maliyet yokken başabaş oranı yaklaşık %28.6'dır; R:R tek başına gerçek isabet oranını belirlemez. Daha iyi aday seçmek hit oranını artırabilir; hedefi yakınlaştırmak ve stopu uzaklaştırmak zorunlu veya kabul edilen yöntem değildir.

| Ölçüt | Raporlanacak ayrıntı |
|---|---|
| Net expectancy ve profit factor | İşlem/episode dağılımı, maliyet varsayımı, belirsizlik; kayıp yokken PF'nin kararsız/tanımsız niteliği |
| Win rate | Net tanım, pay/payda, sıfır sonuç, belirsizlik aralığı, piyasa rejimi ve zaman dilimi |
| Risk | Mark-to-market maksimum drawdown, kayıp kuyruğu, en kötü dönem, maruziyet ve tutma süresi |
| Calibration | Tahmin edilen olasılık ile aynı ölçütte gözlenen sıklık; Brier/log loss ve güvenilirlik dilimleri |
| Seçicilik | Toplam uygun aday, girilen/veto edilen, fill olmayan, geç kalınan; precision/coverage ilişkisi |
| Yol ve icra | MFE/MAE, spread/slippage, kısmi fill, adverse selection, belirsiz sonuç sayısı |

`100 örnek` veya `PF > 1.3` evrensel geçiş kapısı değildir. PF 1.3 gibi bir eşik seçilirse önceden kaydedilmiş deney ölçütüdür; belirsizlik, örnek bağımlılığı ve maliyet stresinin yerine geçmez. Aynı piyasa olayından türeyen kararları bağımsız sayan dar hata aralıkları kullanılmaz. V7'nin `legacy-v7-direction-8m` oranı V8'in `target-before-invalidation-v8` oranıyla veya net trade win rate ile birleştirilmez.

## R0 — Yapı okuyucunun gerçekten ne bildiğini sına

HH/HL, BOS/CHOCH, sweep ve native timeframe yaklaşımı korunur. Bu etiketleri üretmek tek başına grafik uzmanlığı veya öğrenme kanıtı değildir.

1. Point-in-time geçmiş veri ile etiketleri üret. Üç sağ mumla teyit edilen pivot ancak o mumlar kapandıktan ve veri erişilebilir olduktan sonra bilinebilir; sinyali geçmişte pivotun oluştuğu ana geri yazma.
2. Başlangıçta işlemlere bağlamadan, etiket sonrası getiri/bariyer dağılımını aynı rejim, ufuk, saat ve maliyet varsayımlı karşılaştırma örnekleriyle ölç.
3. Kuralları yalnız eğitim/doğrulama döneminde seç; dokunulmamış sonraki dönem ve rejimlerde sınamayı ayrı tut.
4. Toplamda fark yoksa koşullu bir alt grupta avantaj olabileceğini hipotez olarak araştır; veri dilimlerini sınırsız tarayarak en iyi sonucu seçme. Yeni model eklenmeden önce aday üreticisinin değeri hakkında kanıt yaz.

### Eski resolver sonuçlarını yeniden değerlendirme

Özgün V7/V8 kayıtlarını ve eski etiket sürümünü koru. Karar anı, bariyerler ve gerekli piyasa yolu gerçekten mevcutsa yeni evaluator ile **ayrı sürümlü türev sonuç** üret. Eski/yeni etiketlerin uyuşmazlıklarını, ilk/son dakika farklarını ve veri eksiklerini say. Gerekli zaman çözünürlüğü yoksa sonucu bilinmeyen bırak; geçmişe eksik veri uydurma.

Eski %17.65 gibi oranların hangi session/ölçüt/etiket sürümüne ait olduğu doğrulanmadan bu replay'in onu açıklayacağı söylenemez. Browser tetiklemeleri ve tekillik sorunları örnekleme desenini bozmuş olabilir; eski arşiv üzerinde bu sınırlamalar yazılı tutulur.

## R1 — Dar ETH uzmanlığı

İlk hipotez: ETH'de satış baskısının zayıfladığı ve süpürülen seviyenin geri kazanıldığı `SWEEP_RECLAIM` adaylarının bazı bağlamlarda maliyet sonrası değeri var mı?

Ana trendle uyum/karşıtlık, range/transition, volatilite, saat, yakın direnç ve reclaim sonrası devam ayrı açıklayıcı değişkenlerdir. `FAILED_BREAK` ve `BOS_RETEST` sonraki aday araştırmalarıdır; başlangıçta üçünü birden başarılı ilan etme. Dar kapsam ölçümü sadeleştirir, aşırı uyumu otomatik önlemez. ETH-only sınırı kalır; örnek hızını artırmak için 20–30 coin'e genişleme bu plana alınmamıştır.

## R2 — Öğrenen gir/geç seçicisi (meta-labeling)

Birinci katman adayın yapı/yön/iptal/hedefini üretir. İkinci katman, karar anında bilinen koşullarda bu adayın maliyet sonrası değerini ve gir/geç kararını araştırır. Tahmin olasılığı otomatik olarak aynı sayıda pozisyon oranı veya sabit %70 eşiği anlamına gelmez.

Başlangıç için repodaki `brian2026/learning.py` içindeki `LogisticRegressionBaseline`, `GradientBoostingBaseline`, calibration ve metadata araçları değerlendirilebilir. Mevcut yön sınıfları V8'in net trade/barrier hedeflerine doğrudan uymaz; veri/etiket adaptasyonu ve bağımsız doğrulama gerekir. Bu altyapının repoda bulunması V8'in bugün onunla öğrendiğini göstermez. Frozen Phase 3.7 politikası değiştirilemez.

Özellik adayları: yön, yapı mesafeleri, hedef/stop geometrisi, volatilite, rejim, zamanında bilinen akış, spread ve öngörülen maliyet. Sonradan gerçekleşmiş fill/maliyet/gelecek mum bilgisi karar özelliğine sızamaz. Eksik veri gerçek sıfırla karıştırılmaz.

Model eğitim/doğrulama/test ayrımı, train-only preprocessing ve doğrulamada calibration ile çalışır. Seçim ve boyut politikaları ayrı sürümlenir; öğrenen model hard risk sınırını veya SHADOW modunu değiştiremez. Yeni model önce aynı adaylar üzerinde kayıtlı karşılaştırma yapar; kendi kendini aktif modele terfi ettiremez.

## R3 — Emir akışının zaman içindeki davranışı

Anlık oran yerine gerçekleşmiş satışa fiyat tepkisi, bid yenilenmesi, baskının hızlanması/yavaşlaması, reclaim sonrası alım devamı ve farklı kısa pencerelerde süreklilik araştırılır. Satış hacmi artarken fiyatın düşüşünün yavaşlaması absorption hipotezini destekleyebilir; gizli iceberg veya oyuncu niyetini kanıtlamaz.

5 dakikalık CVD ile anlık taker alış/satış oranı zorunlu iki seçenek değildir. Normalize edilmiş birden fazla pencere aday özellik olabilir; katkısı aynı veri üzerinde ayrı ölçülür. Trade delta ile limit emir ekleme/iptallerini de içeren L2 OFI aynı şey değildir. Görünen duvar veya iptal edilmiş emir gerçekleşmiş işlem sayılmaz.

Repo içindeki `_shared/l2_book.ts`, `binance_l2_wire.ts`, `l2_capture_session.ts`, `l2_raw_segment.ts` ve `dynamic_cost.ts` yeniden kullanım açısından incelenebilir. Frozen tüketiciler değiştirilmez. Akışın gerektirdiği olay çözünürlüğü saklama tasarımıyla birlikte seçilir; her şeyi daha sık DB'ye yazmak çözüm kabul edilmez.

## R4 — Zor örnekler ve karar anının dondurulması

Başarılı dönüş/boşa çıkan tepki, gerçek kırılma/failed break ve yüksek güvenle yanılma çiftleri toplanır. **Kayıt tasarımı bugün hazırlanabilir; eğitim doğrulanmış etiketleri bekler.**

Her karar kanıtı en az occurrence/episode, `decision_time`, piyasa zaman aralığı, `data_available_at`, feature/model/policy sürümü, kaynak kimliği ve içeriğin hash/ref bilgisini taşır. Sonuç ayrı eklenir. Aynı ham veriyi yüzlerce snapshot'ta çoğaltmak yerine gerekli girdiyi tekrar üretmeye yetecek ortak veri referansları ve sınırlandırılmış kanıt parçaları saklanır.

Eğitimde zor örnekleri artırmak test/calibration dağılımını da yapaylaştırma izni değildir. Temsili, dokunulmamış değerlendirme kümesi korunur; yeniden örnekleme/ağırlıklandırma kaydedilir. Aynı olayın yüzlerce görüntüsü yüzlerce bağımsız ders sayılmaz. Bu yöntem tek başına önceki confidence/hit uyuşmazlığını çözmüş sayılmaz.

## R5 — Giriş zamanlaması ve fill modeli

Aynı aday için hemen gir / geri çekilmeyi bekle / bırak seçenekleri; net sonuç, fill olasılığı, kaçan fırsat ve dolumdan sonra aleyhe hareket üzerinden karşılaştırılır. Market ve limit yolları aynı maliyet ve sermaye kısıtlarına bağlanır.

Fiyatın limit seviyesine dokunması kesin fill değildir. Queue-ahead, kısmi dolum, iptaller ve veri gecikmesi modellenir. L2 verisinden sanal sıra tam olarak bilinemiyorsa belirsizlik ve muhafazakâr varsayım raporlanır; simülasyon borsa dolumunun kesin kopyası diye sunulmaz. İlk uygulama yalnız SHADOW'dur.

## R6 — Her eklemenin katkısı ve deney disiplini

| Karşılaştırma | Asıl soru |
|---|---|
| Temel V8 | Dondurulmuş yapı adayları tek başına ne sağlıyor? |
| V8 + seçici | İşlem kapsamı ne kadar azalırken net beklenti/calibration iyileşiyor? |
| Önceki sürüm + akış | Ek veri ve saklama maliyetine karşı yeni bilgi var mı? |
| Önceki sürüm + zamanlama | Fill ve kaçan fırsatlar dahil net icra değeri artıyor mu? |

Aynı aday kümesi, dönem, venue, sermaye kısıtı ve maliyet varsayımı kullanılır. Veto edilen adayların karşı-olgusal sonucu yalnız aynı doğrulanmış yol ve tanımlanmış fill varsayımıyla değerlendirilir; gerçekleşmiş işlem P&L'sine eklenmez. Birden fazla parçanın etkileşimi sınanacaksa ek karşılaştırma da önceden kaydedilir.

Her deney manifestosu şunları içerir:

- Hipotez, `experiment_id`, önceki baseline, code/data/feature/label/model/policy hash'leri.
- Eğitim/doğrulama/dokunulmamış test dönemleri; episode gruplaması ve ufuk çakışmasına uygun purge/embargo.
- Venue, uygun aday tanımı, fill/maliyet/risk varsayımları ve açık/ambiguous/eksik sonuç politikası.
- Denenecek ayar sayısı, her varyantın kimliği; başarısız, iptal ve olumsuz sonuçlar dahil deneme günlüğü.
- Önceden tanımlanmış ana ölçütler, kapsama, belirsizlik, maliyet stresi ve değerlendirme/durdurma takvimi.
- Sonuç, sınırlamalar, ek katkı var/yok kararı; ileride aynı başarısız fikrin unutulmasını önleyecek not.

Geçmişte kontamine ilan edilmiş 2026 holdout'u tekrar dokunulmamış sayma; mevcut eski fazların tarih sınırlarını değiştirme. Yeni V8 deneyi kendine ait yeni veri/ayrım manifestosu oluşturur. Ablation aşırı uyum riskini tek başına ortadan kaldırmaz; çok denemeyi kaydetmek ve yeni veride sınamak zorunludur.

## Araştırma sırası ve kaynaklar

Doğruluk kapıları → yapı etiketlerini sınama → temiz SHADOW baseline → dar uzmanlık/seçici → akış → zamanlama. Bulgular bu sırayı değiştirebilir; her değişiklik gerekçesiyle kaydedilir. Bir modülün katkısı yoksa araştırma sonucu saklanır ve karar yoluna alınmaz. Daha büyük model veya daha çok indikatör başlı başına ilerleme değildir.

Yöntem kaynakları, Brian'ın kârlılığının kanıtı değildir:

- [QuantResearch — meta-labeling, örnek tekilliği ve validation yöntemleri](https://www.quantresearch.org/Innovations.htm).
- [Bailey ve diğerleri — backtest overfitting](https://www.davidhbailey.com/dhbpapers/overfit-tools.pdf).
- [NIST — binomial oran belirsizliği](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm); bağımlı olaylar için bağımsız binomial varsayımı ayrıca değerlendirilmelidir.
- [Briola, Bartolucci ve Aste — Deep Limit Order Book Forecasting](https://arxiv.org/abs/2403.09267); forecast başarısı ile işlem değeri ayrı ölçülür.
- [Limit fill olasılığı için survival analysis araştırması](https://arxiv.org/abs/2306.05479).
