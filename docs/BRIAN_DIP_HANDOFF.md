# Brian DIP — devam kaydı

Son doğrulama: 2026-09-09 09:54 UTC.

## Kırmızı çizgiler
ETHUSDT, SHADOW ONLY. Gerçek emir yok. Phase 3.7, Control Center ve ana Brian öğrenme hafızasına dokunma. DB temizliği / kasa sıfırlama bu işin kapsamında değil.

## Kaydedilmiş ve yayınlanmış paket
PR #80 birleştirilmiş: merge 0d43b59e8b8d9f00844b2ed18dadc871a7615096; test edilen PR head bda994644cfbadfb1c3edc1e49b882d12a1d5d13. Dokuz GitHub kontrolünün tamamı başarılı. Sonraki resume düzeltmesi 6e510e3108c90d297c8fadc22c5b9f5931ed4717. Bu kayıttan önce ana branch dc95d384c4885dc38f8743d56c370cf1de08be81.

Tamamlananlar: işlem defterinden kalıcı episode giriş kontrolü; soğuk kalibrasyonda %8, yeterli kalibrasyonda %20 gross notional tavanı; mevcut %0.5 stop-risk bütçesi; yakın yapısal hedefi atlamama; karar/resolver revizyonuna göre kalibrasyon ayrımı; işaretli gerçekleşen funding muhasebesi; liveview sözdizimi ve tablo düzeltmesi; grafik USD-M Perp kaynağı; açık pozisyonda pause ve aynı session resume desteği.

Supabase migration kayıtları: brian_dip_v83_repair (20260908222735), brian_dip_v83_resume_append_only_fix (20260908225958). Repodaki dosya zaman damgaları farklıdır; sırf bu nedenle migrationları yeniden uygulama.

## Canlı doğrulama
Production: https://monster-coins-pro-seven.vercel.app/dip
Vercel deployment dpl_HdvZZB3cXqb7QoEvLtaps3U27oyH, READY, production alias bağlı. Canlı dip-v83-liveview.js HTTP 200; son repo içeriğiyle karşılaştırıldı.
Worker: dip-v8.3-integrity-20260908.5; status OK; veri taze; market/path error null; SHADOW_PAPER; ETHUSDT; live_execution=false. Worker ve foresight cronları dakikalık ve aktif.
Aktif session başka oturumda açılmış: dip-v83-20260909063856-c6a03008 (06:38:56 UTC). Son ölçümde kasa 500, realized 0, işlem 0, açık pozisyon yok; veto NO_SIGNAL. Sinyal yokluğu tek başına arıza değildir; işlem üretmek için eşikleri gevşetme.
Önceki session dip-v83-20260908212621-42012158 korunmuş: kasa 496.29265556722265, realized -3.707344432777365, 4 kapanış. Eski zararlar yeni session performansıyla karıştırılmamalı.

## Sonraki çalışma
Bu onarım paketini tekrar yazma / eski f59d5bc sürümündeki CI hatalarını tekrar düzeltme: sonraki PR head bunları çözmüş.
Bir sonraki iş, kullanıcı istediğinde yeni revizyondaki benzersiz episode/forecast sonuçlarını ve veto dağılımını dar kapsamlı ölçmek. Meta-labeling, akış uzmanlığı ve %70 başarı henüz kanıtlanmış özellikler değildir. Önce temiz prospektif kanıt; yeni coin/indikatör veya agresif boyut ekleme.
Her paket sonunda bu dosyada commit, test, gerçek yayın ve sonraki tek adımı güncelle. Geçici scratch checkout'u canlı sürüm sanma.


## İlk ölçüm tamamlandı — 2026-09-09 09:56 UTC
64 aday / 42 benzersiz episode; 0 ekonomik olarak uygun giriş. Ortanca hedef 5.855 bps, modellenen maliyet 22.942 bps. Resolver zaman kontrollerinde bu örneklemde ihlal yok. Ayrıntı: [ilk ölçüm](BRIAN_DIP_MEASUREMENT_20260909.md).
Sıradaki tek iş: structure.ts hedef seviyelerinin süpürülme/kırılma yaşam döngüsünü karar anı verisiyle replay testinde sınamak; doğrudan eşik gevşetmek veya meta-labeling eklemek değil. Üretim bu ölçüm sırasında değiştirilmedi.


## Seviye replay tamamlandı — 2026-09-09
8 sentetik test mevcut kod üzerinde geçti: dokunulmamış, fitille aşılmış ve kapanışla aşılmış eski pivotların LONG/SHORT hedef seçiminde ayrılmadığı yeniden üretildi. Pivot teyit zamanı ve karar-anı prefix kontrolleri de geçti. Canlı performansa etkisi ve equal-level çiftleri henüz sınanmadı.
Kanıt: [seviye replay raporu](BRIAN_DIP_LEVEL_REPLAY_20260909.md); test: tests/replay/dip_v83_level_lifecycle.test.ts.
Sıradaki tek paket: işlem kararını değiştirmeyen, teyit sonrası seviye yaşam döngüsü gözlem alanları ve equal-level nedensellik testleri. Önceki replay'i baştan yazma; eski seviyeleri toptan silme, uzak hedefe atlama veya risk eşiğini gevşetme. Bu tur üretime müdahale edilmedi.
