# Açık kaynak aday değerlendirmesi — 19 Ekim 2026

Kaynak: Codex araştırma notu (23 aday). Bu belge o notu mevcut koda karşı kontrol eder. Lisans ve sürüm iddialarını GitHub API'den doğrular. İki aday küçük, izole deneylerle ölçüldü. Hiçbir üretim servisi değişmedi, yeni toplayıcı ya da zamanlayıcı yok, ücretli hizmet yok. Cüzdan takibi, gerçek para ve risk kapıları değişmedi.

## 1. Mevcut bağımlılık kontrolü

- Polymarket SDK'larının hiçbiri kullanılmıyor (`py-clob-client`, `clob-client`, `ts-sdk`, `py-sdk`). Connector düz REST + `fetch` ile çalışıyor. Arşivlenmiş SDK uyarısı bizi etkilemiyor.
- `requirements.txt` içinde `scikit-learn>=1.5,<2` ve `ccxt==4.3.81` zaten var; ikisi için yeni bağımlılık gerekmez. ccxt'nin en son sürümü 4.5.88, bizdeki sabit sürüm eski ama çalışan Binance akışı için güncelleme gerekçesi yok.
- Haber gövdesi `brian-world-research/reader.ts` içinde `linkedom@0.18.12` ile `textContent` üzerinden okunuyor.
- Kopya haber için `brian_intel_events` tablosunda `content_fingerprint` ve `corroboration_key` alanları zaten var.

## 2. Lisans / durum doğrulaması (GitHub API, 19 Ekim)

| Depo | SPDX | Arşiv | Son sürüm | Not doğru mu? |
|---|---|---|---|---|
| Polymarket/py-clob-client | MIT | **evet** | v0.34.6 | ✔ arşivli |
| Polymarket/clob-client | MIT | **evet** | v5.8.2 | ✔ arşivli |
| Polymarket/ts-sdk | MIT | hayır | sürüm etiketi yok | ✔ (0.x, sabitleme gerekir) |
| Polymarket/py-sdk | MIT | hayır | polymarket-client v0.13.0 | ✔ |
| adbar/trafilatura | Apache-2.0 | hayır | v2.3.1 | ✔ |
| huggingface/sentence-transformers | Apache-2.0 | hayır | v6.1.0 | ✔ (model lisansı ayrı) |
| dgunning/edgartools | MIT | hayır | v5.62.0 | ✔ |
| mortada/fredapi | Apache-2.0 | hayır | v0.5.2 (2024) | ✔ (az bakım) |
| ccxt/ccxt | MIT | hayır | v4.5.88 | ✔ |
| unionai-oss/pandera | MIT | hayır | v0.34.1 | ✔ |
| scikit-learn | BSD-3-Clause | hayır | 1.9.1 | ✔ |
| statsmodels | BSD-3-Clause | hayır | v0.15.0 | ✔ |
| ranaroussi/quantstats | Apache-2.0 | hayır | v0.0.86 | ✔ (0.0.x) |
| skfolio | BSD-3-Clause | hayır | v1.9.6 | ✔ |
| online-ml/river | BSD-3-Clause | hayır | 0.26.1 | ✔ |
| deepcharles/ruptures | BSD-2-Clause | hayır | v1.1.10 | ✔ |
| duckdb | MIT | hayır | v1.5.6 | ✔ |
| optuna | MIT | hayır | v5.0.0 | ✔ |
| mlflow | Apache-2.0 | hayır | v3.17.1 | ✔ |
| nautilus_trader | LGPL-3.0 | hayır | v1.231.0 | ✔ |
| microsoft/qlib | MIT | hayır | v0.9.8 | ✔ |
| freqtrade | GPL-3.0 | hayır | 2026.9 | ✔ |
| bmoscon/cryptofeed | NOASSERTION | hayır | v3.0.1 | ✔ GitHub standart lisans tanımıyor; not AGPL + ek şart diyor, elenir |
| polakowo/vectorbt | NOASSERTION | hayır | v1.2.0 | ✔ Commons Clause; elenir |
| FinGPT | MIT (kod) | hayır | v1.0.0 | ✔ model/veri ayrı |
| ProsusAI/finBERT | Apache-2.0 (kod) | hayır | — | ✔ ağırlıklar CC BY-NC-SA; elenir |
| hudson-and-thames/mlfinlab | NOASSERTION | hayır | — (2023'ten beri itme yok) | ✔ elenir |

Kayıt: Codex notundaki lisans iddialarının tamamı GitHub'ın bildirdiği lisansla tutarlı. NOASSERTION olanlarda ilgili LICENSE metni okunmadan kullanılmamalı.

## 3. Deney A — resmî Polymarket py-sdk ile connector karşılaştırması

- **Kurulum:** İzole bir venv'e `polymarket-client==0.13.0` kuruldu. Yalnızca `PublicClient` kullanıldı; anahtar, cüzdan ve emir işlevlerine dokunulmadı.
- **Yöntem:** Bizim CORE'da sakladığımız 5 piyasa (`2589810–2589814`) için `get_market(id=…)` çağrıldı. Sonuç, aynı turun (20:35 UTC) son anlık görüntüsüyle alan alan karşılaştırıldı.
- **Sonuç:** 5/5 piyasada `condition_id`, iki outcome'un token ID'leri ve Yes/No fiyatları **birebir aynı**. İstek süresi 30–100 ms. SDK'nın `list_activity` imzası da bizim yöntemimizi (`start` + `sort_direction=ASC` ile zaman tabanlı devam) destekliyor.
- **Karar:** Çalışan REST connector korunur. py-sdk yalnızca **çevrimdışı çapraz kontrol aracı** olarak kalır; üretime ikinci toplayıcı eklenmez. ts-sdk'nın Deno uyumluluğu test edilmedi; gerekmiyor.

## 4. Deney B — Trafilatura ile mevcut `linkedom` okuyucusu

- **Yöntem:** `brian_intel_events` içinden son 3 günde görülmüş, farklı host'lardan 12 gerçek kaynak URL'si seçildi. Her sayfa bir kez indirildi. Aynı HTML'e iki çıkarıcı da uygulandı: Trafilatura 2.3.1 (izole venv, `favor_precision`) ve Brian'ın `reader.ts` → `documentText`. Etiketli bir altın veri seti **yok**, bu yüzden bulgular nicel ve gözle karşılaştırmadır; doğruluk oranı iddia edilmez.

| Ölçüt | Trafilatura | Mevcut okuyucu |
|---|---|---|
| Yayın tarihi bulunan sayfa | **10 / 12** | 0 / 12 (tarih çıkarmıyor) |
| Kelime yapışması | yok | var ("SPEECHFrom", "crisis2 days agoShareSave", "PolicyTether") |
| Arayüz kalıntısı (Share/Save/Key Points) | büyük ölçüde temizliyor | metne karışıyor |
| 403 engel sayfası | metin döndürüyor (Bloomberg 478, BLS 494 karakter) — **tehlikeli** | BLS ve Konsey sayfalarını desenle reddediyor; Bloomberg 403'ü kabul etti (625 karakter) |
| Hız | 2–90 ms (ilk çağrıda 754 ms) | 1–26 ms |
| Çalışma ortamı | Python | Deno edge (mevcut) |

- **Erişim:** 12 sayfadan 3'ü 403 döndü (Bloomberg, BLS, consilium). Hiçbir çıkarıcı erişim sorununu çözmez; not doğru.
- **Karar:** Trafilatura'yı üretime eklemek için edge'in yanına bir Python servisi gerekir. Bu, istenmeyen ikinci bir okuyucu demek, bu yüzden **önerilmez**. Ölçülen iki kazanç bağımlılık eklemeden mevcut `reader.ts`'e taşınabilir:
  1. Blok elemanlar arasında boşluk koruyarak kelime yapışmasını gidermek.
  2. `article:published_time`, `datePublished` (JSON-LD) ve `<time datetime>` meta verisinden yayın tarihini, kaynağıyla birlikte çıkarmak.
- **Bulgu:** Bloomberg'ün 403 sayfası mevcut okuyucunun desen filtresinden geçti. Çağıran kodun HTTP durumunu ayrıca kontrol edip etmediği bu deneyde doğrulanmadı; ayrı bir kontrol konusudur.
- **Not:** `reader.ts` World araştırma akışının parçası. Bu değerlendirmede ona dokunulmadı; iyileştirme ayrı, testli bir PR olmalı.

## 5. Diğer adaylar — karar

| Aday | Karar | Gerekçe |
|---|---|---|
| scikit-learn | **sonra, yeni bağımlılık yok** | Zaten requirements'ta. Brier/log loss/güvenilirlik için önce sonuçlanmış Polymarket piyasaları birikmeli (şu an 0). Aynı olayın anlık görüntüleri bağımsız örnek sayılmayacak. |
| sentence-transformers | **beklet** | Etiketli haber çifti seti yok. `content_fingerprint`/`corroboration_key` zaten var; önce mevcut kopya ayırmanın hata oranı ölçülmeli. Model indirme ve CPU maliyeti edge'e uymuyor. |
| statsmodels | **sonra** | Olay–varlık eşleştirmesi (Faz 2b) olmadan anormal getiri çalışması yapılamaz. |
| QuantStats | **sonra** | 0.0.x sürümünde; maliyetleri işlenmiş günlük portföy serisi önce üretilmeli. |
| fredapi | **aday, anahtar gerekir** | ALFRED "o gün bilinen sürüm" değerli, ama ücretsiz anahtar gerekiyor ve kullanıcı onayı lazım. Son sürümü 2024. |
| EdgarTools | **beklet** | SEC erişim sorununu çözdüğü varsayılamaz (BLS/SEC 403 benzeri). Aynı ağdan küçük test gerekir. |
| ccxt güncelleme | **gerek yok** | Çalışan akış var; yalnızca eksik bir borsa ihtiyacı doğarsa. |
| Pandera | **gerek yok** | Aynı kontroller SQL CHECK + TS normalleştirmede zaten var (null ≠ 0, olasılık aralığı, idempotency). |
| skfolio, River, ruptures, Optuna, Qlib, MLflow, DuckDB | **sonra** | Önce yeterli ileriye dönük sonuç ve baseline gerekiyor. Şu an veri yok. |
| NautilusTrader, Freqtrade | **sonra** | LGPL/GPL; yalnızca somut bir simülasyon eksiği gösterilirse ayrı süreçte. |
| cryptofeed, vectorbt, FinBERT ağırlıkları, mlfinlab | **ele** | Lisans koşulları (AGPL+ek şart / Commons Clause / NC / özel). |
| py-clob-client, clob-client | **ele** | Arşivli. |

## 6. Codex notundaki Faz 1d/2a önerileri

- "Zamanlanmış turda yeni gözlem kimliği" → **doğrulandı.** 20:35 UTC cron jobid 103 `succeeded`, 60 yeni anlık görüntü geldi. Örnek: `polymarket:2589810|src:2026-10-19T20:34:46.224Z` (`received_at` 20:35:01.741). Önceki 20:20 turu aynı CDN yanıtını aldığı için 0 yeni satır yazdı; aynı gözlemin tekrarı başarı sayılmadı.
- "Gerçek geçen süre; yüzde puan ≠ göreli değişim" → **uygulandı (Faz 2a.1).** `baseline_*_at`, `elapsed_*_seconds` ve `relative_change_*` alanları eklendi. `change_*` olasılık puanı olarak kalıyor (0.10 = 10 yüzde puanı). Baz değer 0 ise göreli değişim `NULL`.

## 7. Önerilen sıradaki tek adım

`reader.ts` için ayrı, testli PR: boşluk koruyan metin birleştirme ve kaynaklı yayın tarihi çıkarma. Ölçüt, bu 12 sayfalık setin genişletilmiş ve **elle etiketlenmiş** 30 sayfalık sürümünde tarih doğruluğu ve yapışık kelime sayısı. Bu iş World araştırma akışına dokunduğu için ayrı onayla yapılmalı.
