# Kararlar ve yorumların kanıt sınırı

2026-09-07 konuşmalarının birleştirilmiş kaydıdır. Bu dosya yorumların tamamını alıntılamak yerine kabul edilen işleri ve düzeltilmesi gereken çıkarımları korur. Yeni bulgu geldiğinde eski kararı silmek yerine tarihli değişiklik kaydı eklenir.

## Sabit kapsam

| Karar | Sonuç |
|---|---|
| SHADOW ONLY | Gerçek emir yok; üretim ortamında çalıştırma izni gerçek işleme geçiş izni değildir |
| Temiz ETHUSDT V8 session | V7 session devamı ve çok coinli restore yok |
| Phase 3.7 / Control Center korunur | Bu çalışma bu yüzeylerin davranışını veya frozen politikalarını değiştirmez |
| Ana Brian hafızası korunur | Learning/reliability/audit/calibration ve öğretici kanıtlar temizlik bahanesiyle silinmez |
| DB kurtarma ayrı iş | Bu paket canlı DB'ye dokunmaz; destek sonucu beklenirken kod/izole test hazırlanabilir |
| Önce doğru defter ve etiket | Sonuç doğrulanmadan meta-labeling/calibration eğitimi yok |
| %70+ koşullu hedef | Net beklenti, kayıp kuyruğu, kapsama ve belirsizlikle beraber ölçülür; garanti değil |

## Diğer yorumlarda korunacak ve düzeltilecek noktalar

| Yorum / iddia | Bu paketin tutumu |
|---|---|
| Dört model de planı beğendi; doğru yoldayız | Yaklaşım makul. Bağımsız veri ve test olmadan yorum birliği teknik/performance kanıtı değildir. |
| 17/17 senaryo bütün sistemi kanıtladı | 15 hata üretimi + 2 olumlu kontrol, sabit commit ve mock ortam. Gerçek DB concurrency, uçtan uca browser ve kârlılık kanıtı değil. |
| V8 güncellemesiyle grafik uzmanlığı ortaya çıkar | Yapı okuma uygulanabilir bir temsil sağlar; doğru ölçüm uzmanlığı görünür kılar. Avantajı R0 ve yeni veri sınavı gösterecek. |
| Foresight yalnız açık browser'a bağlı | Browser POST'u yazımı tetikliyor. Ayrıca migration'da server cron tanımı var; canlı cron durumu doğrulanmadı. Tam bağımlılık iddiası kurulamaz. Browser yine salt okunur olmalı. |
| Resolver hataları %17.65'i açıkladı | Zaman kusurları doğrulandı; o oranın nedenini ve düzeltme sonrası oranı kanıtlamadı. Önce aynı ölçüt/session ve replay verisi gerekir. |
| R:R 2.5 doğal olarak %35–40 hit verir | R:R gerçek hit oranını belirlemez; sabit +2.5/−1 için maliyetsiz başabaş yaklaşık %28.6. |
| %70 ancak stop açarak elde edilir | Daha iyi aday seçimi de hit oranını değiştirebilir. Hedef/stop manipülasyonu kabul edilmez. |
| 40 başarısız örnek doğrudan daha büyük pozisyon açtırıyor | Kanıt: R:R eşiği gevşiyor, p=0 yeni girişi veto etmiyor; örnekte fraction %8'den %5'e düşüyor. Sorun sonuç kapısı yokluğu, otomatik daha büyük pozisyon iddiası değil. |
| Short-disable ihlali SHADOW ihlalini kanıtlar | Konfigürasyon kusuru ciddi. İncelemede gerçek emir yolu gösterilmedi; V8'e özgü hard-SHADOW davranış testi gerekli. |
| Aynı evaluator tüm forecast ve trade sonuçlarını eşitlemeli | Aynı yol, zaman ve bariyer girdisinde eşitlik zorunlu. Farklı giriş/fill/maliyet koşullarında meşru fark nedenleri ayrı kaydedilir. |
| Dar uzmanlık overfitting'i engeller; ablation yeterlidir | Kapsamı sadeleştirir. Çoklu deneme, seçim yanlılığı ve kontamine test riski yine vardır. |
| 100 örnek / PF 1.3 geçiş için yeterli | Evrensel kanıt sınırı değildir; bağımlılık, rejim, belirsizlik, net maliyet ve kayıp kuyruğu gerekir. |
| Satış artıp fiyat düşmüyorsa kesin iceberg var | Akışa fiyat tepkisi araştırma hipotezidir; gizli likidite veya oyuncu niyetinin kesin kanıtı değil. |
| 20–30 coin forecast ile örnek hızını artıralım | Mevcut ETH-only kararına ve saklama kısıtına alınmadı. Çapraz coin örnekleri bağımsız ETH uzmanlığı kanıtı olmaz. |
| Phase 1–2 bu hafta kesin biter | Kod, erişim ve test görülmeden süre garantisi yok. Paketler kabul kanıtıyla kapanır. |

## Korunacak V8 çekirdeği

Native 1m/5m/15m/1h/4h veri, teyitli pivot, HH/HL/LH/LL, BOS/CHOCH/sweep, yapıdan hedef/iptal, sunucu sahipliği, tez ve maliyetin denetlenebilir olması, V7/V8 metrik ayrımı ve yapay gelecek mumlarının kaldırılması korunur. Ham conviction ile ölçülmüş başarı olasılığı ayrı gösterilir. Bu seçimler henüz net avantaj ispatı değildir.

## Şimdi eklenmeyecekler

- Yeni coinler, daha fazla gösterge/LLM katmanı, sırf işlem sayısı artsın diye gevşeyen kapılar.
- Martingale, kaybedeni büyütme veya sonradan stopu uzaklaştırarak hit oranı üretme.
- Aynı fiyatın türevi RSI/MACD gibi sinyalleri bağımsız uzman oyları sayma.
- Yapay/sinüs gelecek mumları, kanıtsız yüzde güven ve eksik veri yerine uydurma piyasa girdisi.
- Ayar denemelerini saklayıp en güzel backtest'i sunma; her kayıpta model değiştirme.
- Kısmi kapanışları çoklu kazanım sayma; açık zararları, sıfır/ambiguous sonuçları veya başarısız denemeleri gizleme.
- İşlem kotası, sonradan değişen başarı tanımı, modelin risk sınırını veya çalışma modunu kendi kendine değiştirmesi.
- Snapshot sıkıştırmasını doğruluk çözümü sayma veya yedeksiz tarihsel silme.

## Daha sonra ayrıca değerlendirilecekler

ETH PERP SHORT adaptörü; başarılı temel deneyi aşan yeni setup'lar; daha karmaşık sıra/fill modeli; temsil ve model karmaşıklığı. Bunlar net ek katkı, veri ve maliyet gereksinimi görüldüğünde ayrı deneydir. Şimdiki sınırlar bu listeyle kendiliğinden genişlemez.
