# NIULAI4 ve üç Polymarket deposu — bağımsız değerlendirme ve katkı raporu

19 Ekim 2026. Girdi: Codex'in `BRIAN_NIULAI4_DERIN_INCELEME.md` raporu ve kanıt klasörü. Bu belge raporu tekrar ettirmez; iddiaları **pinlenmiş commit'lerden ve resmî API'den bağımsız olarak** yeniden üretir, aynı karşı örnekleri **Brian'ın kendi koduna** uygular ve yalnızca küçük, testli yöntemleri alır.

Değişmeyenler: cüzdan takibi (kapalı), gerçek para, risk kapıları, zamanlayıcılar, toplayıcılar. Ücretli hizmet açılmadı. Hiçbir sonuç kâr kanıtı değildir.

## 1. Dış depolardaki karşı örnekler (bağımsız yeniden üretim)

Kaynaklar `raw.githubusercontent.com` üzerinden, rapordaki commit'lerle indirildi: crollila `7311a844…`, l-marque `91b315e5…`.

| İddia | Bağımsız sonuç |
|---|---|
| crollila `max_drawdown([-100])` = 0 | **Doğrulandı.** `equity.cummax()` başlangıçtaki 0 tepesini içermiyor (`analysis/backtest.py:136-141`) |
| crollila `size_trades` ortak nakit ayırmıyor | **Doğrulandı.** Her aday `fraction × bankroll / price` ile ayrı boyutlanıyor; `run()` içinde toplam sınırı yok (`:71-86`, `:214-215`) |
| crollila ücret modeli resmî belgeye uymuyor | **Doğrulandı.** Kod `ceil_cents(rate × C × p)` diyor (`pmcal/fees.py:71`). Resmî sayfa `C × feeRate × p × (1−p)` diyor; 5 ondalık, maker ücretsiz, kategori oranları (kripto 0,07, jeopolitik 0). Oransal model ücreti 1/(1−p) kat fazla gösteriyor: p=0,5'te 2 kat, p=0,95'te 20 kat |
| crollila olay bazlı bootstrap | **Doğrulandı.** `pmcal/stats.py` aynı olayın satırlarını küme olarak yeniden örnekliyor; `n_clusters` raporluyor |
| crollila T-kesme testi | **Doğrulandı.** `tests/test_no_lookahead.py`, veri T'de kesildiğinde T'ye kadar olan özelliklerin değişmediğini sınıyor |
| l-marque `open_position` sermaye rezerve etmiyor | **Doğrulandı.** Fonksiyon `state["capital"]` değerini hiç kontrol etmiyor ya da düşmüyor. NO maliyeti gerçek NO ask yerine `1 − yes_price` |

## 2. Aynı karşı örnekler Brian'ın kendi kodunda

Brian'ın gerçek fonksiyonları çalıştırıldı (`_shared/evolution_lab.ts` → `measureOutcomeSet`, `_shared/evolution_treasury.ts` → `planTreasuryCycle`):

| Test | Sonuç | Değerlendirme |
|---|---|---|
| Tek −100 bps karar → maksimum düşüş | **%1,00** | ✔ Doğru. Tepe başlangıç sermayesiyle (100) başlıyor; crollila hatası Brian'da **yok** |
| Hazine: 10.000 USD, aynı anda 100 tam güvenli aday | 1 pozisyon açıldı, toplam 9.990,01 USD, nakit 0 | ✔ Nakit rezervi doğru; crollila/l-marque hatası Brian'da **yok** |
| Aynı 3 kararın sırası değişince düşüş | Zaman sırasıyla **%9,75**, karışık sırayla **%5,00** | ⚠ `measureOutcomeSet` girdiyi kendisi sıralamıyor. v2 koşucu ölçmeden önce `observedAt`'e göre sıralıyor (`brian-evolution-experiment-runner-v2/index.ts:130`), v1 koşucu sıralamıyor. Koşucu cron'u pasif; son sonuç 26 Eylül. **Canlı etki yok, gizli kırılganlık.** Diğer geliştiricinin alanı olduğu için değiştirilmedi; öneri olarak kayda geçti |
| Aynı anda 100 karar, her biri −100 bps → düşüş | **%63,4** | ⚠ Eşzamanlı kararlar ardışık, tam sermayeli bahisler gibi bileşikleniyor. Düşüşü portföy modeline göre olduğundan büyük gösterir (temkinli tarafta). Semantik sınır olarak kayda geçti, değiştirilmedi |

## 3. NIULAI4 muhasebe uzlaştırması (resmî API, 19 Ekim ~20:55 UTC)

Profil–cüzdan eşleşmesi zaten doğrulanmıştı (`0x44832d0d2ec11187c1e77d786feb15f6a50254c6`). Codex'in açık bıraktığı fark incelendi:

| Kalem | Değer |
|---|---|
| CLOSED pozisyonların `realized_pnl` toplamı (Codex, 14.176 satır, sayfalama tam) | 108.073,82 |
| **Redeem edilmemiş açık pozisyonlar** (`/positions?sizeThreshold=0`, 233 satır; hepsi `redeemable`, `curPrice=0`, değer 0) — `cashPnl` toplamı | **−10.688,30** |
| Toplam | **97.385,52** |
| Resmî seri `realized_pnl` (v2/user-pnl, son nokta 20:27:48 UTC) | 96.801,73 |
| Kalan fark | 583,79 (**%0,6**) |
| Ücret: CLOSED 32.480,32 + açık 297,86 | 32.778,18 |
| Resmî seri `fees` | 32.774,85 (fark 3,33) |
| `economic_pnl` = realized + wallet_income (maker 636,14 + taker 3.168,67 + ödül 5,21 + referans 79,97 + sponsorlu 85,18) | 100.691,71 ✔ tutarlı |

- **Bulgu:** Farkın %95'i, sonuçlanıp sıfır değere düşen ama hiç redeem edilmemiş 233 kaybeden pozisyondan geliyor. Bunlar CLOSED listesinde görünmüyor, bu yüzden CLOSED toplamı kârı **olduğundan yüksek** gösteriyor. Kalan ~584'ün olası nedeni, seri (20:27) ile pozisyon çekimi (~20:55) arasındaki zaman farkı; doğrulanmadı.
- **Hâlâ bilinmeyen:** Yatırılan sermaye (`deposits` ve `withdrawals` null), dolayısıyla getiri yüzdesi ve gerçek risk hesaplanamaz.
- **Zamanlama:** Medyan ilk giriş pencere bitiminden 55 sn önce. Brian'ın 15 dakikalık toplayıcısı bu davranışı **gözlemleyemez, kopyalayamaz da**. Cüzdan takibi açılsa bile veri olayın dakikalarca arkasından gelir. Kapalı kalması doğru.

## 4. Brian'a alınanlar (bu PR)

1. **T-kesme değişmezlik testi** (crollila'nın yöntemi, Brian'ın SQL fonksiyonuna yeniden yazıldı): `tests/test_brian_prediction_changes_postgres.py::test_t_cut_invariance_future_rows_never_change_the_past`. Üç rastgele tohum ve her birinde üç kesim noktası kullanılır. Kesimden sonraya yeni satırlar eklenir ve as-of görünümünün **her alanı** aynı kalmalıdır.
   - **Mutasyon kontrolü:** Yerel test veritabanında zaman filtresi bilerek kaldırıldı; test sızıntıyı yakaladı ("as-of view changed after adding future rows").
2. **Resmî Polymarket ücret fonksiyonu** `polymarketTakerFeeUsdc(shares, price, feeRate, role)` (`_shared/prediction_market.ts`): `C × feeRate × p × (1−p)`, 5 ondalık, minimum altı 0, maker 0. `feeRate` zorunlu girdi; varsayılan oran gömülü değil. Şu an yalnızca test ve gelecekteki simülasyon için; üretimde çağrılmıyor. Eye yeniden yayımlanmadı.

## 5. Alınmayanlar ve neden

| Öğe | Karar |
|---|---|
| crollila simülatörünün tamamı | Alınmadı: düşüş, sermaye ve ücret hataları var. Brian'ın hazinesi ve laboratuvarı bu hatalardan muaf |
| crollila olay bazlı bootstrap | **Faz 2c'de** (sonuçlanmış piyasalar biriktiğinde Brier/kalibrasyon): `brian_prediction_markets.event_id` zaten saklanıyor, kümeleme anahtarı hazır. Şu an sonuçlanmış örnek yok |
| l-marque fiyat aralığı stratejisi | Alınmadı: notebook'ta sabit yazılmış sonuçlar, geriye `asof` toleranssız, sonradan bilinen hacim filtresi, NO = 1−YES, sermaye rezervi yok |
| tlafargue cüzdan devamlılığı | Yalnızca yöntem: geçmiş PnL ile cüzdan seçme, dönemleri ayır. Kapsamı NegRisk+combo olduğu için NIULAI4'e genellenmez |
| Cüzdan takibini açmak | Hayır: 55 sn'lik davranış 15 dakikalık toplamayla gözlemlenemez; yüksek frekanslı toplayıcı kurulması istenmiyor |

## 6. Sıradaki tek adım (öneri)

Faz 2c tasarımı: Brian'ın izlediği makro piyasalar sonuçlandıkça kalibrasyon ölçümü (Brier, güvenilirlik eğrisi). Örnekler `event_id` ile kümelenir ve olay bazlı bootstrap güven aralığı verilir. Yalnızca karar anındaki (`received_at ≤ T`) olasılık kullanılır. Yeterli sonuçlanmış olay yoksa çıktı "yetersiz veri" olur.
