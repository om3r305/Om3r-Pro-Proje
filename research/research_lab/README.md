# BRIAN Research Lab

Makalelerdeki iddiaları **tekrarlanabilir, ön kayıtlı ve izole** deneylerle sınar. Sonuçlar ne çıkarsa çıksın saklanır: başarısız bir test, başarılı olanla aynı ağırlıkta kayıttır.

Plan ve gerekçe: [`docs/reviews/BRIAN_VISUAL_RESEARCH_LAB_PHASE0_20261020.md`](../../docs/reviews/BRIAN_VISUAL_RESEARCH_LAB_PHASE0_20261020.md).

## Akış

1. **Defter** (`ledger/<paper>.json`, şema `brian.research-ledger.v1`): makale künyesi, tek tek iddialar, varsayımlar ve ön kayıt (karar kuralı).
   - Defter, veri çekilmeden **önce** commit'lenir; git zaman damgası ön kaydın kanıtıdır.
   - Sonradan yapılan değişiklik yalnızca kuralı **sıkılaştırabilir** ve `amendments` altına yazılır.
2. **Deney** (`studies/<id>/`): yalnızca Python standart kütüphanesi; dış paket ve tedarik zinciri riski yok.
   - Yalnızca izin listesindeki kamuya açık hostlara, anahtarsız ve salt okuma ile bağlanır.
   - Her deneyin çevrimdışı testi vardır (`test_experiment.py`).
3. **İzole çalıştırma:** `.github/workflows/brian-research-lab.yml`.
   - `workflow_dispatch` ile tetiklenir. Gizli anahtar ve yazma izni yok; süre sınırı var. Sonuç artifact olarak alınır.
4. **Kayıt:** Operatör sonucu inceledikten sonra CORE'a yazar.
   - Fonksiyonlar `brian_record_research_ledger_v1` ve `brian_record_research_result_v1` (migration `…_brian_research_ledger.sql`).
   - Tablolar yalnızca ekleme kabul eder. Aynı kimlikle içerik değiştirmek reddedilir.
5. **Arayüz:** `brian-prediction-status` yanıtının `research` bölümü (sözleşme: `docs/BRIAN_PREDICTION_UI_CONTRACT.md` §4b).

## Durumlar

`NOT_STARTED`, `RUNNING`, `INSUFFICIENT_DATA`, `PASS`, `FAIL`, `UNVERIFIED` (aralık 1'i içeriyor ya da kararsız), `REVIEW_REQUIRED` (örneklem kesik, yöntem sorunu).

**Bir `PASS` işlem sinyali değildir.** Kalibrasyon bulgusu maliyet sonrası bir avantaj anlamına gelmez ve hiçbir zaman otomatik terfi ettirilmez.

## Bütçe (her çalıştırma)

| Kaynak | Sınır |
|---|---|
| Çalıştırma süresi | 180 dk (workflow `timeout-minutes`) |
| HTTP isteği | 60.000; istekler arası en az 0,12 sn; 401/403/451 alınırsa yeniden denenmez |
| GPU / büyük indirme | 0; gerekirse kullanıcı onayı şart |
| Ücretli API / model | 0 |
| Veri | Yalnızca kamuya açık veri. Depo public olduğu için BRIAN'ın gizli verisi buraya girmez |

## Deneyler

| Kimlik | İddia | Veri | Durum |
|---|---|---|---|
| `pm-political-slope-oos-v1` | arXiv:2602.19520 C-01: Polymarket siyasi piyasaları az emin (lojistik eğim > 1) | 2026'da kapanan, makalenin hiç görmediği piyasalar; kapanıştan 30 saat önceki fiyat | Sonuç `ledger` ve CORE kaydında |

Yerel çalıştırma:

```
cd research/research_lab/studies/pm_political_slope
python -m unittest test_experiment
python run.py --out ./out
```
