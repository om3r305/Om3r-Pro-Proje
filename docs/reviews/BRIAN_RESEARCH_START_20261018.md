# Brian araştırma başlangıcı — 18 Ekim 2026

Ömer'in 09:32 Berlin talebi üzerine kontrollü araştırma başlangıcı yapıldı. Gerçek para veya borsa emri yok. Hedef veri hattını doğrulamak ve maliyet sonrası avantajı ölçmek; kazanç kanıtlandığı iddia edilmiyor.

## Yayın

- Dip/Brian rapor ayrımı PR #136 ile brian-2026 dalına birleştirildi: d30bb712b6435ad8e3b1ea2bd27b4de5000a4851.
- Vercel monster-coins-pro yeniden açıldı. Production dpl_6nBwU1cpfdTdczZ5St3hHtpw3Cq9 READY. Ana domain https://monster-coins-pro-seven.vercel.app/.
- İlk build, mevcut Ignored Build Step nedeniyle iptal edildi. İkinci istek bu dağıtım için `exit 1` ile build'in yapılmasını sağladı. Genel proje ayarı değiştirilmedi.
- Yayındaki frontier-v4.html HTTP 200 ve `shadow-scorecard.js?v=20261018-isolated` içeriyor. Yayındaki JS HTTP 200; Brian allowlist'i ve Dip ayrımı doğrulandı. Tam tarayıcı görsel testi yapılmadı.

## Çalışma modu

REALTIME `brian-realtime-core-scheduler` version 18 / uygulama `v13-research-only` yayımlandı. Auth korunuyor; mevcut internal-key doğrulaması kullanılıyor. Planlayıcı sadece açık araştırma allowlist'ini çağırır: direct_wire, readiness_cost, alpha_sync, official_primary, world, source_observer, source_registry, meeting_sync, watchdog, discovery. Pozisyon motorları treasury/multiasset, Dip, recovery fanout ve archive çağrılmıyor. Bu, sistem genelinde gelecekteki manuel komutları engelleyen bir güvenlik duvarı değil; yayımlanan scheduler'ın açık çalışma kapsamıdır.

Açılan mevcut cron işleri:
- REALTIME: brian-realtime-orchestrator-1m, brian-realtime-core-scheduler-1m.
- CORE: brian-intel-label-v2-5m (mevcut gerçek programı 15 dakika), brian-missed-opportunity-auditor-v3-5m (mevcut gerçek programı 10 dakika).

Diğer Core eski toplayıcı cronları ve tüm Dip cronları kapalı bırakıldı; Realtime sahipliğiyle çakışan toplayıcılar açılmadı. Treasury kasası sıfırlanmadı; promotion/learning kapıları değiştirilmedi. GitHub Engineer iş akışları açılmadı. Yeni ücretli hizmet veya plan değişikliği yapılmadı; mevcut barındırmanın maliyetsiz olduğu iddia edilmiyor.

## İlk doğrulama

- İlk Realtime döngüsünde ALPHA zaman aşımı ve intrabar JWT-future hatası görüldü. Sonraki döngüde intrabar SUCCESS ve 5 taze ALPHA WAIT kararı doğrulandı; ilk hata kayıtları silinmedi.
- Kaynak registry 16 uç nokta yokladı; 15'i sağlıklı. Yahoo Finance RSS HTTP 404 halen açık eksik.
- Kaynak observer SUCCESS, 105 aday olay raporladı. Bu sayaç yeni benzersiz kayıt sayısı değildir.
- 07:35 UTC başlangıcından ilk ara sayımda Core'da 232 yeni olay, 182 yeni World frame, 6 karar vardı. Bunlar benzersiz hikâye, fırsat veya kâr sayısı değildir.
- Realtime scheduler/orchestrator art arda HTTP 200 SUCCESS yanıtları verdi; araştırma sürümü yanıt üzerinden doğrulandı. Araştırma scheduler'ında doğrudan haber sağlayıcısı bozulunca 207 DEGRADED doğru biçimde raporlanabiliyor.

## Sonuç ölçümündeki düzeltme

Auditor ilk çağrıda DB statement timeout ile FAILED_CLOSED verdi. 5 saniye sınırla kontrol sorgusu da zaman aşımına uğradı. Sorunlu pending-audit fonksiyonundaki iki decision join'ine, seed seçiminin zaten sağladığı gözlem zaman sınırları eklendi. Böylece tarih aralığı dışında geniş decision taraması önlenir; ufuklar, maliyet, sınıflandırma ve izinler değiştirilmez.

Uygulanan migration'lar: 20261018074047_bound_alpha_audit_seed_joins.sql ve 20261018074417_expose_alpha_audit_seed_time_bounds.sql. İlk düzeltme tek başına yeterli olmadı; olgun kararlar girince zaman aşımı tekrarlandı. Seed taramalarındaki zaman sınırı CTE satırından değil, doğrudan aynı 2–36 saat sınırlı ifade ile verildiğinde seed seçimi hızlı tamamlandı. Son fonksiyon servis hesabıyla 5 saniye sınırında doğrulandı. Sonra gerçek auditor HTTP 200 NO_PENDING_AUDITS ve collector SUCCESS döndü (07:44:38 UTC). Bu, ölçüm görevinin tekrar çalışmasıdır; henüz sonuç ürettiğinin kanıtı değildir.

Ayrıca Realtime karar akışı 2 dakikalıkken Core aktarımı 5 dakikadaydı. Auditor terminal fiyatını hedef ufuktan sonraki 120 saniyede aradığı için Core'daki seyrek fiyatlar ölçülemeyen kararlar yaratabilir. Aktarım her tek dakikaya alındı (2 dakika aralık); çift dakikalarda tek Core haber/sınıflandırma görevi dönüşümlü çalışır. Ufuklar veya terminal toleransı gevşetilmedi. Ölçüm hâlâ bire bir gerçekleşen borsa fill'i değildir ve kesintilerde eksik kalabilir. Eski ölçülemeyen kararlar kâr/zarar uydurularak doldurulmadı.

Security advisor ilgili fonksiyon için yeni uyarı göstermedi; proje genelindeki mevcut uyarılar bu değişiklikle giderilmiş sayılmıyor.

34 Node kontrolü geçti: araştırma planının 60 dakikasında pozisyon çağrısı olmaması, veri görevlerinin korunması, sayfa ayrımı, skor muhasebesi, dashboard sağlık/durdurma davranışı ve haber→sanal karar hattı. Bunlar kârlılık testi değildir.

## Takip

Mevcut, eski iki çalışmalık Dip takip görevi yeni Brian saatlik çalışma takibine dönüştürüldü; yeni görev açma denemesi planın 5 görev sınırına takıldığı için ilgili mevcut görev kullanıldı. Görev etkin, Europe/Berlin, saatlik. Yalnız yeni ve anlamlı eksik/kesinti/kaçan fırsat/karşılaştırma raporlanacak. İzleme salt okunur; ücretli servis, gerçek emir, risk değişikliği veya kullanıcı durdurduğunda otomatik yeniden açma yok.

## Açık işler

1. Yahoo 404 ve diğer sağlayıcıların gerçek erişimini onar; çalışmayan kaynağı çalışıyor gösterme.
2. Pending HTML/JSON kaynakları için mevcut Realtime adaptörleriyle registry etiketlerini uzlaştır; listedeki 31 pending etiketini 31 kesin kopuk bağlantı olarak sunma.
3. Kaynak → olay → World frame → ALPHA karar → gerçekleşen fiyat sonucu zincirini ölç. Haberin hızlı olması ile sonradan önemli görünmesini ayır.
4. Agent Reach kurulmadı/bağlanmadı. Ayrı Python araştırma işçisi, sabit sürüm, kaynak/zaman/hash sözleşmesi ve ölçülmüş ek katkı gerektirir. Mevcut RSS'leri çoğaltmak bir iyileşme sayılmaz.
5. Sabit stratejiyle haberli/habersiz test ve nakit/tutma karşılaştırması hazırla. İşlem motorları ancak sonuç ölçümü ve maliyet modeli güvenilir olduktan sonra ayrı sanal denemede değerlendirilir.


## Durdurma ve sınırlar

Bu başlangıç cron seviyesinde seçilidir. Eski genel yönetim düğmelerinin tüm yeni seçili işleri kapsadığı uçtan uca kanıtlanmadı; sonraki kontrol işi bunu da açık eksik saymalıdır. Tam operasyonel durdurma için REALTIME'da iki scheduler cronunu ve CORE'da burada sayılan iki ölçüm cronunu `cron.alter_job(..., active := false)` ile kapatmak gerekir. Sadece Core `system_enabled` bayrağına güvenilmemeli. Genel start düğmesi bu araştırma allowlist'inin dışındaki eski işleri açabileceği için bu rapordaki mod, genel bir start komutuna eşdeğer değildir.
