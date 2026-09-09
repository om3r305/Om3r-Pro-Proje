# DIP V8.3 — seviye yaşam döngüsü replay sonucu

Tarih: 2026-09-09. İncelenen commit: cb1bfaee26b0f7b01e00e79ac2dbb2c7810105ba.
Kapsam: hedef seçimi ve karar anında mevcut mumlar. Gerçek piyasa performans testi değildir.

## Uygulanan doğrulama
Mevcut structure.ts ve decision.ts fonksiyonlarını değiştirmeden doğrudan çağıran 8 sentetik Deno testi çalıştırıldı: 8 başarılı, 0 başarısız. Testlerin geçmesi aşağıdaki mevcut davranışı yeniden üretir; bu davranışın stratejik olarak doğru olduğunu onaylamaz. Çalıştırma --no-check ile yapıldı; yeni tam CI/typecheck iddiası yok.

| Senaryo | LONG | SHORT |
|---|---|---|
| Teyit edilmiş, sonradan dokunulmamış pivot | Hedef seçiliyor | Hedef seçiliyor |
| Teyitten sonra fitille aşılmış pivot | Aynı eski seviye hedef seçiliyor | Aynı eski seviye hedef seçiliyor |
| Teyitten sonra kapanışla aşılmış pivot | Aynı eski seviye hedef seçiliyor | Aynı eski seviye hedef seçiliyor |

LONG örneği: pivot 104; daha sonra high 106 ve close 105; sonraki giriş referansı 102. Eski 104 yine yakın hedef seçiliyor. SHORT simetriği: pivot 96; daha sonra low 94 ve close 95; giriş referansı 98. Eski 96 yine hedef seçiliyor.

Ek iki kontrol: pivot üçüncü sağ mum kapanmadan görünmüyor; geleceği farklı iki fiyat yolunun aynı karar-anı prefix'i aynı yapıyı üretiyor. Bu iki fixture'ın nedensellik kontrolüdür, bütün sistem için look-ahead olmadığı kanıtı değildir.

## Kesin bulgu ve sınırı
Pivot sözleşmesinde teyit sonrası dokunma, süpürülme, kapanışla kırılma veya yeniden teyitli retest bilgisi yok. Hedef seçimi bu geçmişleri ayırmıyor. Bu bir temsil eksikliğidir.
Bunun 64 gerçek adaydan kaçını yanlış veto ettirdiği henüz ölçülmedi. Eski seviyenin artık hiçbir piyasa değeri taşımadığı da ispatlanmadı: kırılmış seviye yeni destek/direnç rolü kazanabilir. Dolayısıyla bütün dokunulmuş seviyeleri silmek veya yakın engeli atlayıp uzak hedefe geçmek bu rapordan çıkarılamaz.
Equal-high/low çiftlerinin yaşam döngüsü bu ilk fixture paketinde sınanmadı; sonraki pakete dahildir.

## Sıradaki tek uygulama paketi: gözlem amaçlı seviye geçmişi
1. Pivot kimliğini ve confirmed_at zamanını koru; olayları yalnız teyitten sonra ve karar zamanına kadar olan kapalı mumlarda hesapla.
2. Dokunma, fitille aşma ve kapanışla aşmayı zamanlarıyla ayrı alanlarda kaydet. Aynı mum içindeki bilinmeyen sıralamayı uydurma.
3. Eski seviyeyi yeni bir retest adayı olarak ayır; yeni teyit olmadan otomatik geçerli likidite hedefi veya otomatik yok sayılan engel ilan etme.
4. Equal-high/low oluşumunun ikinci pivot teyidinden önce bilinmediğini test et; sonradan eşleşme kurarak geçmişe bilgi taşıma.
5. İlk aşamada giriş/hedef/stop/episode hash ve risk kapılarını değiştirme. Mevcut hedef ile yaşam döngüsü açıklamasını aynı adayda karşılaştır. Test: gözlem alanı eklendiğinde işlem kararı birebir aynı kalmalı.
6. Etki ölçülürse ayrı karar revizyonuyla hedef politikası deneyini değerlendir. Başarı ölçütü daha çok işlem değil; doğru açıklanmış seviye ve maliyet sonrası doğrulanmış sonuçtur.

## Tekrar çalıştırma
```sh
deno test --no-check tests/replay/dip_v83_level_lifecycle.test.ts
```

Bu tur canlı kod, DB, session, cron veya risk ayarı değiştirilmedi. ETHUSDT / SHADOW ONLY ve Phase 3.7 / Control Center / ana Brian hafızası sınırları geçerli.
