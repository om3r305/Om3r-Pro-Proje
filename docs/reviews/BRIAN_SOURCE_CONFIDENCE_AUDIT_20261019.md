# Brian — kaynak kalitesi ve mekanizma güveni, 19 Ekim 2026

## Yeni doğrulanmış eksik

CORE anlık görüntü: 2026-10-19T08:59:32.958272+00:00 (Berlin 10:59:32). Son altı saatten en yeni 300 olay çerçevesi ile 100 mekanizma ve en yeni 10 mekanizmanın kaynak bağlantıları incelendi. Rastgele veya bağımsız örnek değildir; iki ardışık mekanizma anı aynı kaynakları tekrar kullanabilir.

| Son 300 olay çerçevesi | Sayı |
|---|---:|
| Kaynak URI'si bulunan | 300 |
| Yayın zamanı bulunan | 298 |
| UNVERIFIED_DISCOVERY | 292 |
| INDEPENDENT_PROFESSIONAL | 4 |
| OFFICIAL_PRIMARY | 4 |

Bu sınıflar sistemde kayıtlı etiketlerdir; bağımsız doğruluk denetimi yapılmış değildir. Doğrulanmamış etiket, haberin yanlış olduğunu göstermez; resmî etiket de tek başına iddiayı doğrulamaz.

Son 100 mekanizmada beş farklı neden metni görüldü. Son 10 mekanizmanın hepsinde güven puanı 0.97; hiçbirinde OFFICIAL_PRIMARY olarak etiketlenmiş bağlantılı olay yok. Enerji, para politikası ve ürün tanıtımı mekanizmalarında bağlantılar tamamen discovery sınıfında. AI ve jeopolitikte yalnızca birer professional etiketli kayıt var. Link sayıları bağımsız kaynak sayısı değildir; tekrarların etkisi ayrıca ölçülmeli.

Kanonik `supabase/functions/_shared/world_brain.ts` kodunda puan şu kuraldan geliyor:

`clamp(0.25 + 0.12 * min(4, event_count) + 0.08 * min(3, breadth))`

Bu bir sezgisel puandır; %97 doğru olma, kâr etme veya nedensel mekanizmanın tutması olasılığı değildir. Örnek, puanın yüksek kaynağa bağlı doğrulama olmadan doygunluğa ulaşabildiğini gösterir. Hangi iddianın yanlış olduğunu veya kayıp işlemlerin nedenini kanıtlamaz.

`counter_evidence_required=true` bütün 100 mekanizmada mevcut, fakat bu alan karşı kanıtın gerçekten elde edildiğini kaydetmez. Mevcut senaryo kodu genel varsayım ve geçersiz kılma metinleri üretiyor; bunlar olaya özel kaynaklı karşı kanıt değildir. Mekanizma tablolarında doğrudan ALPHA etkisi ve canlı yürütme kapalı; bu inceleme kapıları değiştirmedi.

## Sisteme eklenen gerçek kayıtlar

09:01 UTC'de yalnızca sistem ve araştırma etkin bayrakları birlikte true iken:

- Bir append-only `SOURCE_CONFIDENCE_GAP` öğrenme gözlemi kaydedildi: `b468c51a0b3e09643076bda9e24613f7d31d57c9317046779b4baa6bb1520865`.
- Üç hedefli araştırma görevi açıldı ve yeniden okunarak QUEUED / attempts=0 durumları doğrulandı.

| Araştırma | Görev kimliği | İlk kontrol sırası |
|---|---|---:|
| Kaynak bağımsızlığı ve tekrarlar | `78ba55847266aa02498455d22aba987edac12924d9d5d078a45d1a07b693d9bc` | 3 |
| Mekanizma güveninin kalibrasyonu | `8d4f59007c3d9f59254f1bac3b1294aded01462772caa4fc4f19bf4fde55545b` | 4 |
| Olaya özel ekonomik dosya ve karşı kanıt | `b68d3bb42d34dabc596ed1086ffb7bd55b85167220fbc7de19b2373157ae1c01` | 5 |

Bunlar mevcut 24/54 UTC araştırma döngüsüne katıldı; yeni cron veya rakip kod işçisi kurulmadı. Sıra anlık görüntüdür, kesin çalışma saati değildir. Kaynak arama ve README/makale metadatası okuma kapsamı değişmedi. Henüz bu üç göreve ait yeni sonuç elde edilmedi; belge bulunması kabul ölçütlerinin tamamlandığı anlamına gelmeyecek.

Mevcut bağlam birleştiricisi yalnızca kendisinin ürettiği görev-hypothesis eşleşmelerini yazılımcı bağlamına bağlıyor. Bu üç operatör görevi araştırılabilir ve belgeleri saklanabilir, ancak yazılımcıya otomatik bağlandıkları iddia edilmiyor. Aşağıdaki doküman ve görev bağlantıları bu eksik bağlantının da açık gereksinimidir.

## Yazılımcı için küçük uygulama kabulü

İlk değişiklik, mevcut puanı yatırım başarısı olasılığı gibi göstermeyi önlemeli; körlemesine yeni bir sayı veya tavan seçip kalibrasyon varmış gibi sunmamalı.

1. Puanı `HEURISTIC_UNCALIBRATED` gibi açık bir türle ayır. Arayüzde 'sezgisel kanıt puanı; doğruluk olasılığı değil' anlamı görünür olsun. Geçmiş satırlar değiştirilmeden yeni sunum/çıktı sözleşmesi kurulabilir.
2. Kaynak güvenilirliği, kopya/sindikasyon bağımsızlık grubu ve mekanizma hipotezini ayrı alanlar olarak tut. Discovery başlıklarının sayısı doğrulanmış kanıt sayısı olmasın.
3. Birincil kaynak uygun ve erişilebilir değilse `PRIMARY_UNAVAILABLE` / `UNRESOLVED` gibi açık durum kullan; bunu 'yanlış' veya 'doğrulandı' diye çevirmeme testini ekle.
4. Genel `counter_evidence_required` alanıyla gerçek karşı kanıt makbuzunu ayır. Bir makbuz URI, yayın/yakalama zamanı, içerik özeti/hash, destek/karşı çıkış rolü ve çözümlenmemiş çelişki taşımalı.
5. Yeniden yayımlanan aynı haber, çok sayıda discovery başlığı, çelişen kaynak ve eksik yayın zamanı örnekleri; sayının artmasıyla kendiliğinden 'doğrulandı' sonucu üretmemeli. Birincil kaynak gerçekten elde edilirse de olasılık kalibrasyonu otomatik olarak geçmiş sayılmamalı.
6. Görevden gelen araştırma makbuzunun doğru hipoteze bağlandığını veya henüz bağlanmadığını açık göster; `UNVERIFIED_EXTERNAL` durumu korunmalı. Model ağırlığı eğitildiği veya tam metin okunmuş olduğu iddia edilmesin.

Bu çalışma canlı strateji eşiklerini/puanlarını değiştirmedi, kod yayımlamadı ve parasal işlem açmadı. Ürün tarafında kanıt, hata hafızası ve araştırma kuyruğu ilerletildi. Rutin kod uygulamasını Ömer'in diğer sohbete devrettiği yazılımcı ele alacak.

## Doğrulama ve devam

Veritabanı eklemesi tek işlemde 1 yeni episode / 3 yeni mission döndürdü. İkinci SELECT kimlikleri, durumları ve shadow_only=true / live_execution=false değerlerini doğruladı. İsimleri sabit görev kimlikleriyle çakışma halinde tekrar ekleme yapılmıyor. Eski sonuçlar, puanlar ve muhasebe silinmedi/değiştirilmedi.

[Ayrıntılı denetim makbuzu](BRIAN_SOURCE_CONFIDENCE_AUDIT_20261019.json). Öncelik sırası [ürün kabul belgesindeki](BRIAN_PRODUCT_ACCEPTANCE_20261019.md) sonuç ölçümü ve ekonomik araştırma gereksinimlerini tamamlar; bu bulgu bütün projenin tamamlandığı anlamına gelmez.
