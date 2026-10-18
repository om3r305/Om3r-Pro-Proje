# Brian yeniden başlatma incelemesi — 18 Ekim 2026

## Karar

Mevcut stratejileri kazanç beklentisiyle yeniden başlatma: HAYIR.
Haber araştırması, kaynak doğrulama ve bağımsız ileriye dönük sanal test için yeniden tasarım: teknik olarak yapılabilir. Başarı henüz kanıtlanmadı.

Bu inceleme brian-2026 dalındaki 839a99f sürümü, CORE ve REALTIME veritabanlarına salt okunur sorgular ve Agent Reach'in kendi belgelerine dayanır. Tüm kodun güvenlik denetimi, bütün stratejilerin bağımsız performans denetimi veya gerçek borsa hesap mutabakatı değildir. Ücretli servis açılmadı, işleyici yeniden başlatılmadı, para/emir gönderilmedi.

## Ölçülen bulgular

CORE ve REALTIME üzerinde incelenen tüm cron işleri kapalı. CORE kaynak toplayıcısının son görülen kaydı 11 Ekim; dünkü kapatma işleminin ayrıntıları bu incelemede doğrulanmadı. Veritabanı servisleri ise API'de ACTIVE_HEALTHY görünüyor. İşlerin kapalı olması barındırma faturalarının sıfırlandığını kanıtlamaz; faturalama incelenmedi.

CORE `brian_scoreboard_net_30d`, sorgu anındaki son 30 günde şu sonuçları veriyor. Birim karar başına baz puan; 100 baz puan = %1. Bunlar borsada gerçekleşmiş alım-satımların toplam getirisi değildir.

| Ufuk | Yön | Ölçüm | Ortalama brüt bps | Kaydedilen tahmini maliyet bps | Ortalama net bps |
|---|---|---:|---:|---:|---:|
| 5 dk | Long | 2422 | -0.49 | 20.46 | -20.95 |
| 5 dk | Short | 2717 | 0.66 | 20.44 | -19.78 |
| 15 dk | Long | 2306 | -0.04 | 20.38 | -20.42 |
| 15 dk | Short | 2587 | -0.55 | 20.40 | -20.95 |
| 1 saat | Long | 1767 | 0.40 | 20.51 | -20.11 |
| 1 saat | Short | 2008 | -0.32 | 20.49 | -20.81 |

Bu 13.807 ölçüm 6.636 farklı karara ait. Aynı kararın farklı ufukları ve ardışık kararlar bağımsız işlemler sayılamaz. Sorgu kapsamındaki karar tarihleri 18 Eylül–11 Ekim; bütün kayıtlar shadow_only=true/live_execution=false. Kullanıcının bildirdiği gerçek para kaybı bu kayıtlarla doğrulanamaz veya reddedilemez. Borsa işlem ekstresi ayrıca gerekir.

Brüt öngörü ortalaması sıfıra yakın; yaklaşık 20 bps maliyeti karşılamıyor. Bu bulgu yalnızca daha fazla kaynak eklemenin yeterli olduğunu desteklemez. Maliyet varsayımının gerçekleşen ücret/spread/slippage ile doğruluğu ayrıca test edilmeli; maliyeti görünürde azaltıp sistemi başarılı göstermek uygun değildir.

Trend sanal kasasının son kaydı 10 Ekim: NAV 9.610,91; başlangıca göre -%3,757. Aynı kayıtta BTC tutma -%0,807, 50/50 BTC-ETH -%3,632. Bu kısa ve sınırlı örnekte trend motoru da bu iki karşılaştırmayı geçmiyor.

22–29 Eylül için kayıtlı sabit test protokollerinin son ölçümü 26 Eylül'de kalmış. Arena'nın o ölçümdeki kaynak zamanı 23 Eylül. Bu eksik pencereyi tamamlanmış, kesintisiz başarı testi sayamayız. İnceleme geçmiş testleri sıfırlamaz veya negatif sonuçları silmez.

## Haber hattı

Canlı kaynak kaydında ACTIVE/PUBLIC_NO_KEY olan 48 uç nokta var. Bunların 25'i HTML_DIFF_PENDING, 5'i JSON_PENDING, 1'i JSON_DIFF_PENDING: toplam 31'inin adaptör etiketi hâlâ bekliyor. ACTIVE, tek başına veri alındığının kanıtı değil. Genel RSS okuyucusu yalnız GENERIC_FEED olanları seçiyor (bu grupta 8 aktif kaynak). Diğer mevcut okuyucular ayrı; 31 bekleyen kaynağın hiç çalışmadığına dair tüm sistem çapında bir iddia burada yapılmıyor, ancak registry etiketleri bağlantının tamamlandığını göstermiyor.

`brian-source-observer-v2` son kayıtta 244 öğe görmüş, 86 olay raporlamış. Kodda bu sayaç aday olay dizisinin uzunluğu; yalnız yeni eklenen benzersiz olay sayısı olarak okunmamalı. Registry son kaydı DEGRADED, Bloomberg yönlendirmesi hata olarak kaydedilmiş.

Önceki `MAIN_DASHBOARD_PIPELINE_20260927.md` incelemesi 24 saatlik örnekte ortanca yayın→gözlem gecikmesini 11 dakika ölçmüş. Bu bugünkü hız ölçümü değildir. Zamanlanmış işler kapalıyken güncel tazelik/erişim SLA'sı doğrulanamaz.

Kod riskleri:

- Genel okuyucu tarihi bulunmayan öğeyi kabul ediyor; `pit_verified:true` basılması yayın zamanının bilindiğini göstermiyor.
- Kaynak sağlık kaydı eskidiyse bile sorgu yalnız son boolean değerlere bakıyor; sağlığın yaşı ayrıca ölçülmeli.
- World Brain son 12 saatten en fazla 200 yeni olay alıyor. Her kaynağı eklemeden önce giriş hızı/işleme kapasitesi ve atlanan olay sayısı ölçülmeli.
- Kaynak metni üzerindeki keyword yönleri ve genel tema eşlemesi, olaya özgü fiyat etkisinin veya işlem avantajının kanıtı değil.
- Kayıtlardaki `decision_evidence_locked` etiketinin bütün aşağı akış tüketicilerinde gerçekten uygulanması ayrıca doğrulanmalı; etiket güvenlik sınırı olarak varsayılmamalı.

## Agent Reach kararı

Birincil belgeler:
- https://github.com/Panniantong/Agent-Reach
- https://github.com/Panniantong/Agent-Reach/blob/main/pyproject.toml

Python >=3.10 isteyen MIT lisanslı bir kurulum/erişim katmanı. Web için Jina Reader, RSS için feedparser, video için yt-dlp, GitHub için gh ve sosyal platformlar için ayrı araçlar kullanır. Bir işlem stratejisi, lisanslı düşük gecikmeli piyasa hattı veya bütün haber kaynaklarını kapsayan servis değildir. Bazı kaynaklarda oturum, platform erişimi ve kota gerekir. Yazılımın ücretsiz olması model, sunucu ve veri kullanım maliyetlerinin sıfır olduğu anlamına gelmez.

Mevcut Deno Edge Functions içine `pip install` ile bağlanamaz. Bir Python araştırma işçisi veya mevcut yerel bilgisayarda, sabitlenmiş sürüm ve ayrı çalışma ortamıyla bağlanabilir. Tarayıcıya hesap çerezi veya sunucu anahtarı konmaz. Bu inceleme sırasında Agent Reach kurulmadı ve üretim haber hattına bağlanmadı.

Önerilen bağlantı sırası:
1. Çalışan ücretsiz RSS/Atom kaynaklarını mevcut okuyuculardan kullan; aynı akışı ikinci kez toplama.
2. Eksik HTML/JSON adaptörlerini kaynak bazında tamamla; erişim, orijin, gerçek yayın zamanı ve hız ölçümü geçmeden 'çalışıyor' gösterme.
3. Agent Reach'i ek web/sosyal keşif için ayrı araştırma işçisinde dene. Çıktıyı URL, kaynak, published_at (bilinmiyorsa null), observed_at, içerik özeti/hash ve erişim durumu ile kaydet. Keşif verisi kendiliğinden doğrulanmış yön sinyali olmasın.
4. Ücretsiz yolların güncel erişimini kapalı işlem motorundan bağımsız doğrula. Sağlık hatasını sessizce başarıya çevirme; kotaları veya ücretleri otomatik büyütme.

## Yapılan değişiklik

`shadow-scorecard.js` ortak raporu sayfaya göre ayırır: ana sayfa yalnız Brian Hazine, ALPHA karar değerlendirmesi ve trend testini; /dip ve /dip.html yalnız iki Dip motorunu gösterir. Bilinmeyen motor raporları varsayılan olarak gösterilmez. Dip sayfasının kendi mevcut kasası ve al/sat görünümü korunur. Menüde Dip bağlantısı kalır; ayrı hosting/repo kurulmadı.

Sanal karar tablosundaki 'gerçek sonuç' ve 'anlamlı kârlı' ifadeleri düzeltilir. Kaydedilen tahmini maliyetin düşüldüğü, sonuçların portföy kârı olmadığı ve örtüşen örneklerin bağımsız sayılamadığı açıkça belirtilir.

31 ilgili Node testi geçti: sayfa ayrımı, test raporu muhasebesi, dashboard sağlık/durdurma davranışı ve haber→çoklu varlık sanal karar hattı. Bunlar kârlılık veya canlı veri entegrasyonu testi değildir.

## Yeniden tasarım için kabul koşulları

- Kaynak listesi: çalışıyor/bekliyor/engelli ve son gerçek veri zamanı; gözlenen benzersiz olay ve gecikme dağılımı.
- Ayrı Brian ve Dip ekranları; ayrı değerlendirmeler ve strateji sürümleri. Eski kayıplar/testler korunur.
- Veri toplayıcıları işlem motorundan ayrı çalıştırılabilir olmalı; 'başlat' bütün modülleri körlemesine açmamalı.
- Kaynaksız/tarihsiz, yinelenmiş veya fiyatı bayat veri işlem kanıtına terfi etmemeli.
- Haber var/yok karşılaştırması aynı fiyat, maliyet, tarih aralığı ve risk kurallarıyla yapılmalı. Herkesin okuduğu haberi almak tek başına avantaj sayılmaz.
- Strateji ve eşikler önceden sabitlenmeli; tarih sıralı, görülmemiş örneklerde ve ardından ileriye dönük sanal kayıtlarda ölçülmeli. Aynı pencereye tekrar tekrar ayar yapıp bağımsız test denmemeli.
- Tüm ücretler, spread, kayma, fonlama ve gecikme dahil edilmeli. Nakitte kalma ve basit tutma karşılaştırmaları, azami düşüş ve devir hızı raporlanmalı.
- Birkaç kazanan işlem veya 7 gün/100 kapanış tek başına yeterli değil; örnek büyüklüğü ve belirsizlik stratejiye göre değerlendirilmeli. Pozitif test, gelecekte kazanç garantisi değildir.

Bu koşullar yerine gelmeden 'istediğimiz seviyede çalışacak' demek için kanıt yok. Projenin korunması ve sadeleştirilmesi mümkün; mevcut sonuçları yeni özellik isimleriyle başarı saymak mümkün değil.
