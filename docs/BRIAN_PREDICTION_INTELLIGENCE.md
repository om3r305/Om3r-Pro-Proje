# Brian — Prediction Intelligence (Faz 0 bulguları ve Faz 1 durumu)

Durum: 19 Ekim 2026. Modül **yalnızca okuma + gölge (shadow)** çalışır. Gerçek para, emir, cüzdan anahtarı, canlı karar değişikliği yoktur.

## Faz 0 — doğrulanan gerçekler (19 Ekim 2026, Berlin'den ölçüldü)

| Kaynak | Sonuç |
|---|---|
| Gamma API `markets/keyset` | HTTP 200, erişim açık |
| Gamma API `public-search` | HTTP 200 |
| CLOB `/time`, `/book`, `/prices-history` | HTTP 200, herkese açık |
| Data API `/activity`, `/positions`, `/value`, `/traded`, `/v1/leaderboard` | HTTP 200 |

**Kritik bulgu:** Gamma `GET /markets` (offset) kullanımdan kaldırıldı (`deprecation: true`, `sunset: 2026-05-01`). Keyset uç noktası `markets/keyset` kullanılmalı. İstek parametresi **`after_cursor`**, cevap alanı **`next_cursor`**. `cursor` adı **sessizce yok sayılır** ve hep ilk sayfa döner; bu yüzden yineleyici daha önce görülen imleci reddeder.

**NIULAI4 kimliği (doğrulandı):** `public-search` tam ad eşleşmesi + 42 karakterlik tam cüzdan adresi + `/activity` satırlarının aynı cüzdan ve adı bildirmesi. Cüzdan: `0x44832d0d2ec11187c1e77d786feb15f6a50254c6`. Benzer adlar (`Niulai`, `niulai666`…) farklı hesaplardır ve eşleştirilmez.

**Kamuya açık ilk gözlemler (OBSERVED_VERIFIED; strateji değil):** liderlik tablosu (ALL) sıra 1963, hacim ≈ 4,18 M USD, PnL ≈ 97 bin USD (hacme oranı ≈ %2,3); işlem edilen piyasa sayısı 11104; gördüğümüz işlemler ağırlıkla "Bitcoin Up or Down" 5 dakikalık piyasalar. Pozisyon satırlarında giriş ücreti (`entryFeesUsdc`) görünüyor. Bu sayılar bir hipotez değildir ve takip edilebilir bir avantaj kanıtı değildir: 5 dakikalık piyasalarda herkese açık veri gecikmesi avantajı yok edebilir (Faz 2 sorusu 9).

## Faz 1 — bu PR'ın kapsamı

`supabase/functions/_shared/prediction_market.ts` (saf, enjekte edilebilir `fetch`):

- Host izin listesi: yalnızca `gamma-api`, `data-api`, `clob.polymarket.com`, https, kimlik bilgisi yok.
- 429/5xx/ağ hatasında sınırlı yeniden deneme: `Retry-After` (saniye veya HTTP tarihi) + üstel geri çekilme.
- 401/403/451 = `BLOCKED`: yeniden denenmez, coğrafi kısıt aşılmaya çalışılmaz.
- Gamma piyasa normalleştirme: eksik değer `null` kalır, `0` olmaz; olasılık [0,1] dışıysa `null`.
- Keyset sayfalama (`after_cursor`/`next_cursor`), `maxPages` sınırı, tekrar eden imleçte durma.
- Cüzdan işlemleri: yalnızca `type=TRADE` işlem sayılır; REDEEM/MERGE/… ayrı sayılır; kararlı idempotency anahtarı; yinelenenler birleştirilir.
- Kimlik doğrulama: tam ad + tam adres + etkinlik çapraz kontrolü; aksi halde `UNVERIFIED`/`AMBIGUOUS`. Adres asla tamamlanmaz veya tahmin edilmez.

Bu PR veritabanına yazmaz, zamanlayıcı eklemez, panele dokunmaz, mevcut karar akışına bağlanmaz.

## Testler

```
deno test supabase/functions/_shared/prediction_market.test.ts                       # çevrimdışı, CI
PREDICTION_LIVE=1 deno test --allow-net --allow-env supabase/functions/_shared/prediction_market.test.ts   # canlı duman testi
```

## Sıradaki tek adım

Faz 1b: `prediction_*` tabloları (RLS, idempotent, değiştirilemez gözlem) için migrasyon ve sınırlı sayıda piyasayı aralıklı çeken toplayıcı. Toplayıcı mevcut collector kira (lease) mekanizmasını yeniden kullanmalı.
