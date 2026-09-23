# DB düzeldikten sonra V8'e devam ve yayın kapıları

**Bu bir uygulama runbook'udur; çalıştırılmış operasyon kaydı değildir.** DB erişimi açıldığı için bu dokümantasyon branch'i doğrudan deploy edilmez. Önce [teknik planın](TECHNICAL_PLAN.md) uygulama PR'ları ve aşağıdaki kanıtlar tamamlanır. Üretim ortamında da yalnız **ETHUSDT / SHADOW ONLY** çalışılır.

## A — Mevcut durumu yeniden belirle

- [ ] Güncel `brian-2026` SHA, PR #76/#77 durumu, deploy edilmiş UI/worker/resolver sürümleri ve migration geçmişini kaydet.
- [ ] DB bağlantısı/yazılabilirlik, disk headroom ve servis sağlığını gerçek ölçümle doğrula. Destek cevabı veya dashboard görüntüsü tek başına yeterli değil.
- [ ] Maintenance için kapatılmış writer/cron envanterini ve varsa açık V7/V8 sanal pozisyonları belirle. Kayıtlar sessizce başlangıç kasasına dönmesin.

Bu aşama canlı read/write erişimi hazır olduğunda uygulanır; bu paketin hazırlanması sırasında yapılmadı.

## B — Önceki DB kurtarma planını koru

Önce öğretici learning/calibration/reliability/audit ve önemli logları içeren doğrulanmış yedek/export. Restore edilebilirliği, kapsamı ve bütünlüğü kontrol edilmeden silme yok. Sonra tüm tabloların gerçek relation/index/TOAST boyutları ve satır tahminleri ölçülür; yalnız `brian_dip_*` tablolarına bakılmaz.

Eğitim değeri taşımadığı doğrulanan eski test/runtime/snapshot/event/session ve V7 test artıkları, retention sınıfı ve referansları incelenerek temizlenir. Test etiketi tek başına bir kaydın değersiz olduğunu kanıtlamaz. Ana Brian hafızası ve önemli özgün sonuç kanıtı korunur. Bu dosya toplu DELETE/TRUNCATE komutu sağlamaz.

Silme ardından gereken alan kazanımı ayrıca doğrulanır. Normal VACUUM alanı çoğunlukla relation içinde yeniden kullanıma açar; dosya sistemine alan dönüşü otomatik varsayılmaz. Reclaim yöntemi disk payı ve kilit gereksinimleri görülerek seçilir; `VACUUM FULL` gibi rewrite işleri yetersiz alanla körlemesine başlatılmaz. Son DB ve disk ölçümü ilk ölçümle karşılaştırılır.

[PR #77](https://github.com/om3r305/Om3r-Pro-Proje/pull/77), incelenen haliyle V8 session başına UTC saatte bir snapshot satırını günceller. Eski saatler/oturumlar yine birikir; bu **tam retention değildir**. Öğretici geçiş/sonuç defteri ayrı korunurken runtime/telemetry/raw veri için son durum + saatlik/günlük özet + açık TTL/arşiv politikası gerekir. Başka writer'ların hacimleri de bütçelenir. Saatlik satır update'inin JSONB/TOAST churn'ü ölçülür; sadece satır sayısına bakılmaz.

## C — Kod ve schema kabul kapıları

| Kapı | Gerekli kanıt | Şu an |
|---|---|---|
| UI/session | Gerçek browser'da ETH restore/start/resume, doğru sürüm, browser ölçüm yazımı yok | Bekliyor |
| Defter | Retry, timeout, eşzamanlı worker, lease/pause ve partial-write testleri; replay/kasa eşitliği | Bekliyor |
| Evaluator | Aynı yol için deterministik sonuç; ilk/son dakika, açık mum ve ambiguous politikası | Bekliyor |
| Kimlik | Tez/forecast tekilliği; pozisyon planı/grafik tutarlılığı | Bekliyor |
| Risk/veri | Stale/gap/read error giriş veto; min-notional cap; short-disable; simetrik maliyet | Bekliyor |
| Hard SHADOW | V8 worker'da config değiştirme denemeleri dahil gerçek emir çağrısı sıfır | Bekliyor |
| Schema | Disposable Postgres transaction/constraint/migration testi; deploy uyumluluk planı | Bekliyor |
| Araştırma | R0 yapı sınavı ve baseline manifestosu; kanıt sınırları yazılı | Bekliyor |

Özgün 17 senaryo bir defect-reproduction arşividir; bu tablodaki kapıların geçmiş olduğu anlamına gelmez. Yeni testler güvenli davranış beklentisiyle yazılır. Type-check listesine V8 worker, foresight, session API ve ortak evaluator eklenir; gerçek Postgres concurrency ayrıca sınanır.

## D — Migration ve eşleşen sürüm

- [ ] `supabase/migrations/202609070074_brian_dip_v8_chart_reader.sql` için remote migration geçmişi ve mevcut schema doğrulansın. Uygulanmış migration yeniden körlemesine çalıştırılmasın; uygulanmamışsa test edilmiş sıraya eklensin.
- [ ] PR #77'deki `202609070075_brian_dip_hourly_snapshot_compaction.sql`, yeni transaction/snapshot tasarımıyla birlikte test edilsin. Mutable saatlik snapshot trigger'ı atomik defterin garantilerini zayıflatmamalı.
- [ ] Diğer prerequisites, özellikle session/lease ve foresight tabloları, gerçek migration sırası üzerinden doğrulansın. Yalnız numaralara bakarak tüm bağımlılıklar uygulanmış sayılmasın.
- [ ] Yeni düzeltmeler için additive, sürümlü migration hazırlanmış ve disposable DB'de denenmiş olsun; eski migration dosyalarını değiştirmek remote schema'yı düzeltmiş sayılmaz.
- [ ] Önce yeni girişler kapalı/OBSERVE iken schema ve sunucu bileşenleri sürüm uyumuyla yayınlansın. Worker/resolver/session API/UI için uyumlu code/policy/schema sürümleri kaydedilsin; uyumsuz istemci açık hata alsın.
- [ ] UI cache/service worker da aynı contract sürümünü sunsun. Eski V7 katmanı yeni session'ı tekrar değiştiremesin.

## E — Writer'ları kontrollü aç ve yeni session başlat

Önce gerekli temel servisler, ardından yük/disk/latency/error ölçümleriyle diğer writer'lar açılır. Hangi writer'ın ne zaman açıldığı ve sonrasındaki büyüme kaydedilir. Eski maintenance öncesi tüm cron'ları bir anda açmak bu runbook'un parçası değildir.

Yeni V8 session'ı **yalnız ETHUSDT** ile oluştur; V7 session'ı resume etme. İlk önerilen trade kapsamı Spot LONG, aşağı yön yalnız gözlem; short flag kapalı. Başlangıç sanal equity açık ve doğrulanmış olsun. Eski kasayı sıfırlayarak aynı session'ı temizmiş gibi gösterme.

Smoke-test:

- [ ] Sunucu worker çağrısı, geçerli session ve lease ile çalışır; `engine_version` uyumu doğrulanır.
- [ ] Piyasa yolu/yaş kontrolleri ve WAIT nedenleri görünür; iyi veri yoksa işlem zorlanmaz.
- [ ] Tez oluşumu, forecast intent/sonucu, olay ve runtime projeksiyonu beklenen tekillikte ilerler. State version ve market cursor geriye gitmez.
- [ ] Browser tamamen kapalıyken scheduler/resolver ilerler; sayfayı açmak yeni forecast yazımı başlatmaz.
- [ ] Açık pozisyon varsa UI seviyeleri onun dondurulmuş planıyla aynıdır. Kasa/ücret/pozisyon replay'i tutar.
- [ ] Yeni girişler sadece bütün kabul kapıları açıkken SHADOW_PAPER modunda serbesttir. Piyasada uygun aday yoksa sırf smoke için giriş uydurulmaz; zorunlu giriş/çıkış vakaları yerel fixture ile doğrulanır.
- [ ] Gerçek emir çağrısı yok; disk büyümesi ve hata oranı ölçülmüş bütçe içinde.

Smoke başarıları uygulama SHA, session ID, policy/evaluator sürümü ve kanıt konumuyla kaydedilir. Henüz çalıştırılmamış adımlar tamamlandı diye işaretlenmez.

## F — Sorunda durdurma ve yeniden devam

DB/state okunamıyorsa, lease/version uyuşmazlığı varsa veya veri bozuksa yeni giriş kapısı kapanır. Mevcut pozisyon ve olay kanıtı korunur. İzleme/çıkışın güvenle sürdürülebildiği ve veri yolunun tamamlanabildiği doğrulanır; doğrulanamıyorsa açık durum mutabakat gerektirir. Güvenli sonuca ilişkin eksik bilgi uydurulmaz.

Rollback sürüm uyumlu olmalıdır. Açık V8 pozisyonunu anlamayan eski worker'a geçip session'ı restore etmek veya DB'yi silerek restart etmek kabul edilmez. Olay kaydı, hata nedeni ve eksik kanıt tutulur; düzeltilmiş sürüm aynı defteri tekrar oynatarak ilerler.

Sonraki aşama yalnız [araştırma planındaki](RESEARCH_PLAN.md) dondurulmuş ETH SHADOW baseline ve prospektif karşılaştırmadır. Gerçek paraya geçiş bu paketin kapsamında değildir.
