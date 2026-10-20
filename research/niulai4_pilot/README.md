# NIULAI4 gözlem pilotu (yerel, salt okuma, BAŞLATILMADI)

Plan: [`docs/reviews/BRIAN_NIULAI4_OBSERVATION_PLAN_20261020.md`](../../docs/reviews/BRIAN_NIULAI4_OBSERVATION_PLAN_20261020.md). Kurallar `rules.ts` içinde dondurulmuş (`niulai4-pilot.v2`). Kurallar değişirse `RULES_VERSION` de değişmeli.

**Pilot şunları yapmaz:**
- Emir göndermez ve hiçbir anahtar kullanmaz.
- CORE'a (Supabase) yazmaz, zamanlayıcıya dokunmaz.
- `wallet_tracking_enabled`'a dokunmaz.
- İzin listesi dışında hiçbir sunucuya istek atmaz: `rules.ts → PILOT_ALLOWED_HOSTS`.

## Modlar

```
deno run --allow-net --allow-read --allow-write --allow-run=powershell,w32tm research/niulai4_pilot/main.ts --selftest --out C:/Brian/brian-research/niulai4-pilot
deno run --allow-net --allow-read --allow-write --allow-run=powershell,w32tm research/niulai4_pilot/main.ts --start    --out C:/Brian/brian-research/niulai4-pilot
deno run --allow-read --allow-write research/niulai4_pilot/main.ts --report --out C:/Brian/brian-research/niulai4-pilot
```

- `--selftest` yalnızca ön kontrol yapar, hiçbir şey kaydetmez. Kontroller: saat (Windows NTP, `w32tm`; Cloudflare + PTB), disk, profil-cüzdan doğrulaması, kural parmak izi.
- `--start` şu koşulda çalışır: `<out>/APPROVED_RULES_FINGERPRINT` dosyası, `--selftest`'in yazdığı `rulesFingerprint` ile birebir aynı olmalı. Bu dosya ancak operatör onayından sonra yazılır. Bu dosya olmadan, saat belirsizliği 50 ms'yi aşarsa, disk 1 GB'ın altındaysa ya da kimlik doğrulanmazsa pilot başlamaz.
- `--report` kayıtlı dosyalardan çevrimdışı değerlendirme yapıp `summary.json` yazar.
  - Varsayılan olarak **ara rapordur** (`INTERIM_NO_VERDICT`, karar yok).
  - Karar yalnızca `--final --required-windows N` ile verilir. N, pilottan **önceden dondurulmuş** olmalı.
  - Üç sonuç ayrı raporlanır: uygulanabilirlik, mutlak net, baseline üstünlüğü. Avantaj ancak son ikisi birlikte sağlanırsa var sayılır.
  - H2 = `NOT_TESTED`; çıktı her zaman `tradeSignal: false` taşır.

```
deno run --allow-read --allow-write research/niulai4_pilot/main.ts --report --out <dir> --final --required-windows 8721
```

## CORE'a sonuç gönderme (isteğe bağlı, varsayılan KAPALI)

`--start ... --core-sink-key-file <yol> [--host-label pc]` ile pilot kendi sağlık ve ara özetini CORE'a yollar. Arayüz bunları `brian-prediction-status` üzerinden okur (sözleşme: [`docs/BRIAN_PREDICTION_UI_CONTRACT.md`](../../docs/BRIAN_PREDICTION_UI_CONTRACT.md)).
- Anahtar dosyası yoksa pilot yalnızca yerel çalışır.
- Anahtar CORE'da yalnızca SHA-256 olarak durur ve ancak operatör onayıyla etkinleştirilir. Repo'ya ya da kayıt dosyalarına yazılmaz.
- Sink'in hataları ve HTTP durumları ölçümü durdurmaz, StopGuard'a gitmez.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `rules.ts` | Dondurulmuş kurallar, host izin listesi, kural parmak izi |
| `windows.ts` | 5 dk pencere hesabı, slug, aktif saatler, piyasa alanlarıyla çapraz kontrol |
| `book.ts` | CLOB market websocket reducer'ı (`book` + `price_change`), sabit miktarlı ask yürüme |
| `spot.ts` | Spot proxy (Binance + Coinbase), 60 sn TWAP, B1 yönü. **Oracle değil.** Chainlink TWAP akışı kimlik doğrulaması istiyor ve kullanılmıyor (plan §0) |
| `evaluate.ts` | FOLLOW / B0 / B1 / B2. İlk görülme sırası; emir gecikmesi; ret kuralları (geç/erken yakalama, sessiz soket, senkronsuz defter); seviye bazında maliyet ve ücret; (strateji, d) başına kasa; güç hesabı; ayrı kararlar |
| `stats.ts` | Pencere ve gün kümeli bootstrap |
| `guards.ts` | Durdurma kuralları, yeniden bağlanma beklemesi, saat kayması, ön kontrol |
| `recorder.ts` | Yerel, byte tavanlı JSONL (her dosyada kural başlığı) |
| `watcher.ts` | Kamuya açık aktiviteyi izler; ilk tur yalnızca mevcut durumu öğrenir |
| `runtime.ts` | Çalışma döngüsü: websocket (10 sn'de PING), poll, örnekleme, yakalama (first_seen + d + 300 ms), çözüm, sağlık |
| `sink.ts` | İsteğe bağlı CORE sink: sağlık (60 sn) ve ara özet (30 dk) → `brian-niulai4-pilot-ingest`. Anahtar dosyası verilmezse kapalı; tek sabit URL; hataları pilotu etkilemez |
| `main.ts` | Komut satırı |

## Test durumu

- `pilot_test.ts`: saf parçaların hepsi test ediliyor.
- `runtime.ts`'in canlı döngüsü **hiç çalıştırılmadı**, çünkü pilot başlatılmadı. İlk onaylı başlatmanın ilk saati aynı zamanda bu döngünün ilk canlı testidir; sağlık kayıtları ile izlenmeli.
