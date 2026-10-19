# Brian — canlı veri ve sonuç kapsamı, 19 Ekim 2026

## Sorumluluk ve amaç

Ömer 19 Ekim 10:27 Berlin talimatı: otonom yazılımcı kurulumu Ömer'de; bu çalışma veri, araştırma, fırsat değerlendirme ve sonuç doğrulama tarafını sürdürür. Yazılımcı yapılandırması değiştirilmedi. Ürün hedefi [ürün sözleşmesinde](../BRIAN_VISION_AND_DELIVERY.md) kalır. Bu rapor sistemin tamamlandığı veya kâr ettiği anlamına gelmez.

Kaynak kod tabanı: `135b35df162e51cb1474c65e6c0c4102e4af8ae7`. Canlı CORE ve REALTIME üzerinde yalnızca SELECT sorguları çalıştırıldı; veri/şema/yayın/işlem anahtarları değiştirilmedi.

## Gerçek ölçüm

CORE anlık görüntü: 2026-10-19T08:27:52.872427+00:00 (Berlin 10:27:52). Son altı saatteki en yeni 300 karar; rastgele örnek değildir. Aynı kararlar farklı ufuklarda tekrar sayılır, toplamlar benzersiz işlem sayısı değildir.

| Ufuk | Süresi dolmuş karar | Sonucu kaydedilmiş | Core'da uygun son fiyat yok | Son fiyat var, denetçi sonucu bekliyor |
|---|---:|---:|---:|---:|
| 5 dk | 292 | 177 | 115 | 0 |
| 15 dk | 268 | 168 | 100 | 0 |
| 60 dk | 156 | 90 | 63 | 3 |

Bu örnekte eksik giriş referansı ve sonuçlanmış yönlü kararlarda eksik maliyet metadatası görülmedi. Bu, maliyet modelinin gerçekleşmiş dolum maliyetiyle doğrulandığı anlamına gelmez. Son fiyat kontrolü üç mevcut Core yolunu kapsar: ALPHA karar fiyatı, intrabar olayları, bağımsız audit fiyatları. Terminal aralık hedef zamandan sonraki 120 saniyedir; tolerans genişletilmedi. `TERMINAL_PRESENT_AWAITING_AUDITOR` yalnızca fiyat bulunduğunu söyler; tüm denetçi koşullarının geçildiğini kanıtlamaz.

Ayrı 08:27:24 UTC RPC sorgusu: 45 maliyeti kayıtlı yönlü 5 dk sonucu, ortalama modellenmiş net -25.13 baz puan. Örnek seçilimli ve örtüşmelidir; strateji performans tahmini, gerçekleşmiş zarar veya nedensel açıklama olarak kullanılamaz.

Araştırma: 111 belge makbuzu; son belge 2026-10-19T08:24:07.243+00:00. Son bağımsız Core fiyatı 2026-10-19T08:25:37.163+00:00. Belge sayısı kalite veya öğrenilmiş doğru bilgi sayısı değildir.

## Core / Realtime karşılaştırması

Core'da eksik görünen 278 karar-ufuk aralığı Realtime intrabar kayıtlarıyla tek tek eşleştirildi. 274 aralıkta Realtime intrabar fiyatı da bulunmadı; dört aralık için iki gerçek fiyat kaydı bulundu (BNB ve XRP, her biri 5/15 dk aralıklarına uyuyor). Bu dört aralık dört farklı fiyat değildir.

İki kaynak olay:

- `4dadd6db579d184f1f83f3a96fb879819822bc436e7845f52ec9891bcb2fd755`
- `0ae2ea25d0bb46996057a4e6905416c011024a0d24322fe5865cd73f57b04e07`

Her ikisi de Realtime'da 08:27:05.375 UTC'de gözlenmiş, normal Core köprüsü tarafından 08:29:02 UTC'de özgün olay kimliği ve gözlem zamanı korunarak kaydedilmişti. Son SELECT bu iki kaydı doğruladı. Manuel backfill yapılmadı. Yaklaşık 117 saniyelik gecikme, ilk anlık görüntüdeki dört eksik aralığı açıklıyor; diğer 274 aralığın tamamının nedenini açıklamıyor. Diğer Realtime tabloları/ham arşivler bu karşılaştırmada taranmadı; 'hiçbir yerde fiyat yok' iddiası yapılmaz.

## Toplayıcı tasarımı ve eksik

Güncel kanonik `brian-intrabar-eye` kaynak kodu, BTC/ETH/SOL/BNB/XRP'yi sabit tutup kalan evrenden 10 varlığı dönüşümlü seçiyor; toplam 15. Realtime 08:25, 08:27 ve 08:28 UTC başarılı çalışma kayıtları 15 seçili / 15 kullanılabilir varlık gösterdi. Bu mimari, dönen varlıklarda dar terminal aralığına uygun fiyat bulunamamasına katkıda bulunabilir; tek başına nedensel kanıt değildir. Sabit varlıklarda da boşluk var, dolayısıyla sadece radar rotasyonunu suçlamak yanlış olur.

## Sonraki somut geliştirme kabulü

Yazılımcının ele alacağı davranış: sinyal üretiminden bağımsız, bekleyen kararların 5/15/60 dk hedeflerini izleyen fiyat kaydı. Mevcut toplu bookTicker yanıtının kullanımı araştırılabilir; yeni ücretli sağlayıcı veya daha fazla ağır kline/aggTrades isteği varsayılmamalı.

1. Kararı olan varlığın izlenmesi radar rotasyonundan bağımsız olmalı; ücretsiz kaynak ve sınırlar açık olmalı.
2. Gözlem zamanı ile sisteme ulaşma zamanı ayrı korunmalı; gelecekte edinilen fiyat geriye dönük biliniyormuş gibi yazılmamalı.
3. Kaynak olayları ve mevcut muhasebe değişmez kalmalı; kayıp tarihsel tick uydurulmamalı, terminal toleransı genişletilmemeli.
4. Her varlık/ufuk için olgun, sonuçlanmış, fiyatı eksik, maliyeti eksik ve denetçi bekleyen sayılar görünür olmalı.
5. Dönüşümlü varlık, kısa servis kesintisi, gecikmeli sync, yinelenen olay ve STOP senaryoları gerçek testlerle doğrulanmalı.
6. Önceden sabitlenmiş bir ileri dönem penceresinde kapsam karşılaştırılmalı; örtüşen son-300 örneklerini kâr/iyileşme kanıtı saymamalı.

Bu çalışma eksiklerin teşhisi ve tekrarlanabilir okuma sorgusunu teslim eder; bağımsız fiyat toplayıcısı henüz uygulanmadı. Otonom yazılımcı çalışmalarına çakışan kod değişikliği yapılmadı.

## Kanıt dosyaları

- [Tekrar çalıştırılabilir salt-okunur Core SQL](BRIAN_OUTCOME_COVERAGE_20261019.sql)
- [İlk anlık görüntü, varlık/ufuk satırları](BRIAN_OUTCOME_COVERAGE_20261019.json)

SQL tanı içindir; yetkili sunucu ortamında çalıştırılmalıdır. Yeni bir public RPC veya istemci izni oluşturmaz. SQL canlı veritabanında başarıyla çalıştırıldı ve tüm gruplarda örnek/olgun/sonuç/eksik aritmetiği kontrol edildi. Çalışan ürün kodu değişmediğinden yeni bir ürün test koşusu veya canlı yayın yapılmadı.
