# DIP hedef ve maliyet inceleme paketi — 2026-09-09

Kapsam: ETHUSDT SHADOW. Phase 3.7, Control Center, session, kasa, öğrenme tabloları ve emir yolları değişmez.

## Gerçek kayıt incelemesi
Berlin 17:35–18:35 / UTC 15:35–16:35: 22 karar, 18 episode; 1 episode daha önce başlamış. Bu saatte ilk kez başlayan 17 episode: 12 hedef-önce, 4 iptal-önce, 1 açık (sorgu anındaki sonuçlar). %75 hedef isabeti bir işlem kazanma oranı değildir. İşlem yok, kasa 500 USDT. Kaynak: brian_dip_v8_decisions, brian_dip_v8_runtime ve ledger salt-okunur sorguları; bağımsız fiyat replay yapılmadı.
22/22 hedef mesafesi modellenen maliyetin altında. Ortanca hedef 3.7123 bps; maliyet 22.2997–22.4171 bps. 18 kararda seçili hedefin CLOSE_CROSSED geçmişi mevcut. Tekrarlanan episode sonuçları ayrı bağımsız örnek değildir.

## Uygulanan
- UI: TARGET_BELOW_COST açıklaması tam kuralı söyler: hedef mesafesi < 2.5 × modellenen maliyet. Ekran hesaplanan minimumu gösterir. Null maliyet sıfır sayılmaz.
- Offline ölçüm: scripts/dip_opportunity_review/review.py. İlk session episode kaydını seçer; 5/15/30 tam kapalı dakikada yönlü kapanış, lehe/aleyhe sapma ve maliyet eşiği karşılaştırması çıkarır. Oluşum mumu high/low kullanılmaz, eksik seri MISSING_DATA, dolmamış süre PENDING. Sonuç as_of sonrasında kaydedilmişse başarı göstermez. İşlem simülatörü değildir; stop/target sırası veya gerçekleşebilir fill iddia etmez.
- Altı davranış testi geçti. Canlı DB'ye yeni veri yazılmaz. Production yayın yapılmadı.

## Kullanım
`python scripts/dip_opportunity_review/review.py input.json > report.json`
`python -m unittest discover -s scripts/dip_opportunity_review -p 'test_*.py'`
Girdi: start/end/as_of UTC ISO veya milisaniye; complete_session_history=true; symbol=ETHUSDT; market_source=BINANCE_USDM_PERP; decisions (session'ın pencere öncesi kararları dahil), bars (sıralı Binance USD-M 1m t/ct/o/h/l/c, fiyatlar sayı). Decision alanları: session_id, episode_id, decision_at, symbol, direction, entry_price, evidence.cost_bps, evidence.market_source, isteğe bağlı evidence.veto/level_observation, resolved_at/hit. Veri kaynağı ve tam session geçmişi girdiyi hazırlayan tarafından doğrulanmalı. Araç canlıya bağlanmaz ve veri indirmez. Üretilen gözlemler model eğitimi veya veto kaldırma için otomatik kullanılmaz.

## Hâlâ doğrulama gerektirenler
1. Gerçek USD-M hesap komisyonu: mevcut ayar 10 bps/yön. Kullanıcının gerçek tarifesi bilinmediğinden değiştirilmedi.
2. CLOSE_CROSSED seviyeleri için yeni geçerlilik/tekrar teyit politikası: bu gözlem tek başına seviyeyi geçersiz kılmaz. Mevcut yakın hedef otomatik uzak hedefe çevrilmedi.
3. Gerçek mum ihracıyla veto edilen ilk episode'ların ileri dönem karşılaştırması: araç hazır, bu PR'da gerçek veri replay sonucu üretilmedi.
4. Maliyet bileşenlerinin entry slippage dahil muhasebeyle tutarlılığı ayrıca incelenmeli. Araç bu nedenle net P&L üretmez.

Sıradaki adım: gerçek ücret tarifesini doğrula, aynı piyasa ve sabit pencereli verilerle aracı çalıştır; sonuçlara göre hedef politikası için ayrı, ileri dönem deneyi hazırla. %75'i kâr kanıtı olarak sunma; eşikleri işlem açtırmak amacıyla düşürme.
