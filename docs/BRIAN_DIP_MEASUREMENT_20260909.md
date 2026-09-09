# DIP V8.3 — ilk prospektif ölçüm

Ölçüm: 9 Eylül 2026, 09:56 UTC. Session: dip-v83-20260909063856-c6a03008.
Kod tabanı: 6632cc6777fbb60341a1484d1eeb988247d45243. Karar revizyonu: dip-v8-dual-integrity-20260908.5.

## Sonuç
Motor çalışıyor ve adayları kaydediyor. Bu örneklemde işlem açılmamasını açıklayan somut neden ekonomik risk/ödül vetosudur. Bu bulgu, bütün stratejinin doğru veya kârlı olduğunu kanıtlamaz.

- 64 karar kaydı; episode başına ilk karara indirgenince 42 olay.
- 40 olay çözülmüş: 23 hedef önce, 17 iptal/başarısız sonuç; 2 bekleyen. Bu %57.5 bir **ham bariyer isabetidir**, işlem kazanma oranı veya maliyet sonrası kâr değildir.
- EARLY_REVERSAL: 30 episode, 28 çözülmüş, 19 hedef önce / 9 başarısız.
- SWEEP_RECLAIM: 11 episode, 11 çözülmüş, 3 hedef önce / 8 başarısız.
- BOS_RETEST: 1 episode, 1 hedef önce. Tek örnekten başarı çıkarımı yapılamaz.
- 64/64 kayıtta ECONOMIC_RR_TOO_LOW; 62/64 kayıtta TARGET_BELOW_COST. Aynı kayıtta birden fazla veto bulunabilir.
- Diğer vetolar: CALIBRATING 64, DIRECTION_REFEREE_REJECT 20, COUNTER_STRUCTURE 10, RAW_CONVICTION_LOW 9, OPPOSING_MULTI_FLOW 2.
- Ortanca hedef uzaklığı 5.855 bps (%0.05855); ortanca modellenen gidiş-dönüş maliyeti 22.942 bps (%0.22942).
- Ekonomik giriş kapılarını geçen kayıt: 0. İşlem: 0. Kasa: 500; açık pozisyon yok.
- Son tarihten 3 dakika fazla gecikmiş çözümsüz kayıt: 0. Karardan önce resolved_at: 0. Son tarihten sonra eventAt: 0. Bunlar sorgulanan zaman kontrolleridir; tam replay/eşzamanlılık testi değildir.

## Maliyet varsayımı
Session fee_bps=10 (tek yön), slippage_bps=1 (tek yön). Ücret tek başına gidiş-dönüş 20 bps; spread/funding/slippage ilave edilir. Bu, yapılandırılmış SHADOW varsayımıdır; kullanıcının gerçek borsa komisyon kademesini doğrulamaz. İşlem üretmek amacıyla düşürülmemeli; gerçek piyasa/hesap tarifesi ayrı doğrulanmalıdır.

Örnek 09:56 SHORT adayı: giriş 2490.120963, hedef 2490, iptal 2495. Hedef uzaklığı 0.486 bps, modellenen maliyet 22.762 bps. Ham yapı puanı 0.87 olmasına rağmen WAIT kararı verilmiş. Bu olayda yapı puanı başarı olasılığı sayılmamış ve maliyet vetosu uygulanmış.

## Sıradaki tek teknik inceleme
Hedeflerin ekonomik mesafesi dar. structure.ts geçmiş pivotlardan/equal high-low çiftlerinden seviyeler seçiyor; seviye daha sonra süpürüldü mü, kırıldı mı, hâlâ likidite hedefi mi soruları için açık bir yaşam döngüsü kaydı yok. Bu, kodda görülen bir temsil eksikliğidir; 64 adayın kaçını yanlış veto ettirdiği henüz ölçülmedi.

Sonraki küçük paket: karar anında mevcut mumlarla seviye yaşam döngüsünü replay testinde sınamak. Dokunulmamış, süpürülmüş, kırılmış ve yeni teyitli retest seviyeleri ayrılmalı; gelecekteki mumlar özelliklere sızmamalı. Yakın gerçek engeller sırf ekonomik eşiği geçmek için atlanmamalı. Karşılaştırma mevcut hedef seçimine karşı yapılmalı; kanıt olmadan canlı hedef politikasını değiştirme.

Bu tur üretim kodu, risk eşiği, session, bakiye, cron veya DB şeması değiştirilmedi. Phase 3.7 / Control Center / ana Brian öğrenme kapsam dışı. Bir sonraki sohbette onarım paketini veya CI incelemesini baştan yapma.
