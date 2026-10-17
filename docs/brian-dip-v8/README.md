# Brian DIP V8 — kalıcı çalışma ve devam paketi

Tarih: **2026-09-07**. Kapsam: DIP V8 ve bağlandığı DIP session API/UI.

Bu paket, Ömer'in V8 incelemesini, sonraki tartışmalardaki kararları ve geliştirme önerilerini kaybolmadan GitHub'da tutma talebini karşılar. DB sorunu çözüldüğünde çalışmaya bu dizinden devam edilir.

**Durum: Dokümantasyon ve tarihsel yeniden üretim kanıtı hazır. Burada tarif edilen V8 düzeltmeleri henüz uygulanmadı; bu paket deploy edilebilir bir düzeltme sürümü değildir.** Supabase desteğine başvurulduğu ve DB'nin kullanılamadığı bilgisi kullanıcı bildirimidir; bu paket hazırlanırken canlı DB durumu sorgulanmadı.

## Önce burayı oku

| Dosya | İçerik |
|---|---|
| [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md) | P0/P1 işleri, state machine, atomik defter, ortak zaman/fiyat yolu ve davranış kabul kriterleri |
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | %70+ hedefinin koşulları, yapı okuyucu sınavı, uzmanlık, seçici model, akış, zor örnekler, zamanlama ve karşılaştırma |
| [RELEASE_RUNBOOK.md](RELEASE_RUNBOOK.md) | DB sonrası migration/session/deploy sırası, test kapıları ve güvenli durdurma |
| [DECISIONS.md](DECISIONS.md) | Sabit sınırlar, yorumlarda düzeltilen çıkarımlar, ertelenen ve reddedilen işler |
| [evidence/README.md](evidence/README.md) | Kanıtın sınırı, dosya bütünlüğü ve tekrar çalıştırma |
| [İlk kod incelemesi](evidence/Brian-V8-Inceleme-2026-09-07.md) | İncelenen commit'e bağlı özgün rapor; değiştirilmeden korunmuştur |

## Değişmeyen sınırlar

- **SHADOW ONLY:** gerçek emir yok. Daha sonra üretim ortamına deploy edilmesi de bu sınırı değiştirmez.
- İlk temiz V8 deneyinin sembolü **yalnız ETHUSDT**. Eski V7 session'ı sürdürülmez.
- **Phase 3.7 ve Control Center değişiklik kapsamı dışındadır.** Ana Brian öğrenme, reliability, audit ve değerli calibration kanıtları korunur.
- Canlı DB'ye bu paketle migration, silme, restart veya writer açma işlemi yapılmaz. Tarihsel temizlik için doğrulanmış yedek ön koşuldur.
- DB'nin açılması doğruluk testlerinin yerine geçmez. Yeni model eğitimi için tutarlı ve sürümlü sonuç etiketleri ön koşuldur.

## Kaynak ve mevcut durum

- Repo: [om3r305/Om3r-Pro-Proje](https://github.com/om3r305/Om3r-Pro-Proje).
- İncelenen ve paket hazırlanırken tekrar doğrulanan `brian-2026` commit'i: [`0c1b1e88e758a71e7cf9732300b77f8f6e2f597d`](https://github.com/om3r305/Om3r-Pro-Proje/commit/0c1b1e88e758a71e7cf9732300b77f8f6e2f597d).
- [PR #76](https://github.com/om3r305/Om3r-Pro-Proje/pull/76): V8 chart reader; merge edilmiş.
- [PR #77](https://github.com/om3r305/Om3r-Pro-Proje/pull/77): bu kayıt tarihinde açık. Saatlik snapshot sıkıştırması; karar motorundaki P0/P1 düzeltmelerini içermiyor. Daha sonraki durumu tekrar kontrol edilmelidir.
- `074_brian_dip_v8_chart_reader` migration'ının üretimde beklediği kullanıcı tarafından bildirildi; uygulanma durumu DB erişimi açıldığında doğrulanacak.
- Önceki kaynak çalıştırmasında **15 sorun senaryosu + 2 olumlu kontrol = 17/17 beklenen gözlem** elde edildi. Bu sonuç hataların giderildiğini, gerçek DB eşzamanlılığını veya kârlılığı göstermez.

## Çalışma sırası

| Sıra | Paket | Mevcut durum |
|---|---|---|
| 0 | Bu kararları ve özgün kanıtı GitHub'da koru | Bu dokümantasyon PR'ı |
| 1 | V8 UI/session sözleşmesi ve açık sunucu çalışma modu | Bekliyor |
| 2 | Önce defter bütünlüğü; sonra tez/pozisyon geçişleri ve kilit | Bekliyor |
| 3 | Ortak fiyat yolu, zaman penceresi, tekil tez/forecast kaydı | Bekliyor |
| 4 | Veri sağlığı, risk, calibration veto, Spot/maliyet tutarlılığı | Bekliyor |
| 5 | Doğrulanmış geçmiş veride yapı etiketlerinin öngörü değerini sına | 1–4'ün ilgili doğruluk kapılarına bağlı |
| 6 | DB ve release kapıları sonrası temiz ETHUSDT V8 SHADOW deneyi | DB + kod/test kapılarına bağlı |
| 7 | Uzmanlık → seçici model → akış → zamanlama; her katkıyı ayrı ölç | Güvenilir etiket ve ayrılmış değerlendirme verisine bağlı |

DB beklenirken 1–4'ün kodu ve izole testleri hazırlanabilir. Zor örneklerin kayıt tasarımı ve deney manifestosu da hazırlanabilir. Bozuk etiketlerle model eğitimi ve yeni bir güvenilirlik iddiası başlatılmaz.

## Sonraki geliştiricinin ilk işi

Bu README ve `DECISIONS.md` dosyasını oku; güncel branch/PR/migration durumunu tekrar doğrula. `TECHNICAL_PLAN.md` içindeki ilk açık paketten küçük, davranış testi olan bir uygulama PR'ı aç. Tarihsel hata üretimlerini değiştirme; düzeltilmiş davranış için ayrı regresyon testleri ekle. Tamamlanan her işin yanına uygulama commit'i ve test kanıtı yaz. Bu dokümandaki bir checkbox, yalnız metni yazıldığı için tamamlanmış sayılmaz.
