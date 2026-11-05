# BRIAN Visual Intelligence & Research Lab — PHASE 0 kaynak denetimi (Claude: araştırma ve backend)

Tarih: 20 Ekim 2026. Kapsam: master notun (v2.0) 15. maddesindeki PHASE 0. Bu belge **yalnızca rapordur**:
- Canlı BRIAN değişmedi, deploy yapılmadı.
- Dış depoların hiçbir kodu çalıştırılmadı. Depolar ayrı bir klasöre sığ biçimde klonlanıp okundu.

İş bölümü:
- **Claude:** Paper2Agent, Paper2Code, reproducibility deposu, TAPE'in sinyal tarafı, video kaynağı, veri modeli ve deney motoru.
- **Codex:** PRISM, Collision, SWARM, HORIZON ve 49 inç motoru. Bu depolar için burada yalnızca lisans, güvenlik ve "gerçek veri mi" kontrolü var.

## 1. Depolar — doğrulanmış bağlantılar, lisanslar, incelenen sürüm

| Depo | Bağlantı | Lisans | İncelenen commit | Not |
|---|---|---|---|---|
| ATSMATRIX PRISM | https://github.com/anyel1to/ATSMATRIX-PRISM | MIT | `ea70e6a` | 9 dosya, 6★ |
| Collision Engine | https://github.com/anyel1to/atsmatrix-agent-visualizeR--ANYEL1TO | MIT | `efc9d4d` | 5 dosya, 58★ |
| ATSMATRIX TAPE | https://github.com/anyel1to/ATSMATRIX-TAPE | MIT | `031d6d8` | 22 dosya (video/GIF dahil) |
| ATSMATRIX SWARM | https://github.com/anyel1to/ATSMATRIX-SWARM | MIT | `571b4ce` | `server.js` + `package.json` da var |
| ATSMATRIX HORIZON | https://github.com/anyel1to/ATSMATRIX-HORIZON | MIT | `925caf5` | tek `horizon.html` |
| ATSMATRIX HELIX | https://github.com/anyel1to/ATSMATRIX-HELIX | MIT | `35f8510` | |
| ATSMATRIX AGENT RING | https://github.com/anyel1to/ATSMATRIX-AGENT-RING | MIT | `f2033da` | |
| Paper2Agent | https://github.com/jmiao24/Paper2Agent | MIT | `8c2d059` | 3.847★; bağımlılık PyMuPDF **AGPL-3.0** (§4) |
| Paper2Code (PaperCoder) | https://github.com/going-doer/Paper2Code | Apache-2.0 | `ba91699` | 4.993★; ICLR 2026 |
| Reproducibility Agent | https://github.com/usama-shiranai90/paper-to-code-reproducibility-agent | Apache-2.0 (dosya metni; GitHub API `NOASSERTION` gösteriyor) | `d137345` | 1★; çalışma günlüğü boş |

Lisans gereği MIT ve Apache-2.0 bildirimleri, kod parçası alınırsa ilgili dosyanın başında ve `THIRD_PARTY_NOTICES` içinde korunmalı. Apache-2.0'da ayrıca `NOTICE` dosyası ve "değiştirildi" notu gerekir. Önerim kod kopyalamak yerine **yöntemi yeniden yazmak** (§9).

## 2. Paper2Agent — gerçekte ne yapıyor

- **Ne olduğu:** Bir servis değil; Claude Code veya Codex'e yüklenen **beceri paketi**: markdown talimatları ve yardımcı Python betikleri.
  - `skills/paper2agent/SKILL.md` girişe göre yönlendiriyor: makale → Paper2Skill, kod deposu → Paper2MCP.
  - Ajan başlatabilen bir host gerektiriyor.
- **Paper2Skill** (`paper2skill/SKILL.md`, `scripts/paper_bundle.py`, 578 satır):
  - PDF, ek dosyalar ve tablolardan sayfa sayfa **insan veya ajan incelemesi zorunlu** bir okuma paketi kuruyor.
  - Kaynaklar SHA-256 ile sabitleniyor. İnceleme kanıtları teslimatın dışında tutuluyor.
  - `verify --strict` beş durum veriyor: `mechanical_failure`, `unreviewed`, `unresolved_discrepancies`, `reviewed_with_limitations`, `reviewed`.
  - Betikte ağ çağrısı ya da kod çalıştırma yok; yalnızca XML ad alanı adresleri geçiyor. "Makaledeki kod veya talimatları çalıştırmaz" kuralı açıkça yazılı.
- **Paper2MCP** (`paper2mcp/SKILL.md`, `references/orchestration.md`, `scripts/verify_mcp_server.py`, `verify_workflow.py`):
  - Bir araştırma deposundaki **mevcut** fonksiyonları test edilmiş MCP araçlarına sarıyor.
  - Kurallar sıkı:
    - "Do not invent scientific algorithms to fill repository gaps".
    - Doğrulayıcı ajan, kodu yazan ajandan farklı olmalı.
    - Deneme sınırları var: ortam 3, çalıştırma 5, doğrulama 6.
    - Raporların hash'leri değiştirilemiyor; başarısız denemeler saklanıyor.
- **Güvenlik:** Paper2MCP **dış deponun kodunu doğrudan çalıştırıyor** ("Run the bound upstream code"). Yerleşik bir sandbox yok; container yalnızca isteğe bağlı eklenti.
- **Alan uyumu:** Örneklerin hepsi biyoloji (TISSUE, Scanpy, AlphaGenome); finans örneği yok.
- **BRIAN'a uygunluk:**
  - Paper2Skill'in **inceleme disiplini** (kaynak hash'i, sayfa bazlı inceleme, sıkı doğrulama durumları) doğrudan model alınabilir.
  - Paper2MCP yalnızca makalenin **kod deposu varsa** anlamlı ve mutlaka izole ortamda çalışmalı (§8).

## 3. Paper2Code (PaperCoder) — gerçekte ne yapıyor

- **Akış** (`scripts/run.sh`):
  - `0_pdf_process` → `1_planning` → `1.1_extract_config` → `2_analyzing` → `3_coding`.
  - `4_debugging` ayrı çalışıyor: kullanıcının verdiği hata çıktısını alıp modele SEARCH/REPLACE düzeltmesi yazdırıyor.
  - `eval.py` ise bir modelin 1–5 arası **görüş puanı** (o3-mini-high, 8 örnek ortalaması). Bu doğruluk kanıtı değil.
- **Model bağımlılığı:**
  - Bütün `_llm` olmayan betikler sabit biçimde `OpenAI(api_key=os.environ["OPENAI_API_KEY"])` kullanıyor. Anthropic yolu yok.
  - Açık model yolu vLLM ile `DeepSeek-Coder-V2-Lite`, `TP_SIZE=2`, yani **2 GPU**.
  - PDF→JSON dönüşümü için GROBID (Java servisi) gerekiyor.
- **Maliyet:** README'nin kendi tahmini makale başına o3-mini ile 0,50–0,70 $. Ücretli; GPU yolu bizde yok.
- **Güvenlik:**
  - Üretilen kodu kendisi çalıştırmıyor; bu iyi.
  - Ancak `4_debugging.parse_and_apply_changes`, modelin döndürdüğü dosya adını `os.path.join(debug_dir, filename)` ile birleştiriyor. `../` içeren bir ad çalışma klasörünün dışına yazabilir (yol aşımı). Kullanılacaksa yol doğrulaması eklenmeli.
- **BRIAN'a uygunluk:** Planlama → analiz → kodlama aşamalandırması **referans** olarak değerli. Pipeline'ın kendisi alınmamalı: yalnızca OpenAI, ücretli, sandbox yok, değerlendirmesi LLM puanı.

## 4. Lisans notu — PyMuPDF

Paper2Agent MIT lisanslı, ama `paper_bundle.py` dosyası `pymupdf==1.28.2` ve `pymupdf4llm` kullanıyor. Bunlar **AGPL-3.0 ya da ticari lisanslı**.
- Yerel ve tek kullanıcılı bir araç olarak çalıştırmak sorun değil.
- Ağ üzerinden hizmet olarak sunmak ya da dağıtmak AGPL yükümlülüğü doğurur. Bu yüzden BRIAN'ın bir edge function'ına ya da sunucu servisine **konmamalı**.

## 5. Reproducibility Agent — gerçekte ne yapıyor

- **İçerik:** Codex ve Claude Code eklentisi olarak paketlenmiş beceriler:
  - `paper-ingest`, `claim-extraction`, `method-reconstruction`, `reproduction-planning`, `discrepancy-analysis`, `trackio-logbook`, `context-packet`.
  - Bir orkestratör ajan.
- **Değerli parçalar:**
  - **İddia defteri** (`skills/claim-extraction/references/claim-schema.md`): her satırda tek bir iddia; `tier`, `metric`, `dataset`, `reported{value,unit,uncertainty,source}`, `comparison`, `repro_risk`, `depends_on`.
  - **Varsayım defteri**, kaynak izi (`TRACE.md`), ve yalnızca ekleme yapılan çalışma günlüğü.
  - **Bütçe** (`loop-budget.md`): çalıştırma başına 3 deneme, 15 dk; GPU **0**; büyük indirme **0**, ikisi de onaysız yapılamaz.
- **Durum:**
  - `loop-run-log.jsonl` dosyasında şema satırı dışında kayıt yok; araç sahada kullanılmamış.
  - Örnek yalnızca `examples/toy-paper`.
  - Varsayılan rekonstrüksiyon PyTorch tabanlı, yani ML odaklı.
- **BRIAN'a uygunluk:** İddia ve varsayım şeması, BRIAN'daki eksik halkayı tam karşılıyor (§7). Kod alınmadan şema yeniden yazılmalı.

## 6. Görsel ve simülasyon depoları — gerçek veri, güvenlik, Codex için uyarılar

| Depo | Gerçek veri mi? | Güvenlik bulgusu | Teknik not |
|---|---|---|---|
| PRISM | Hayır (deterministik simülasyon, `Math.random` yok) | `innerHTML` 2 kullanım | C++ "çekirdek" yalnızca `make kernel` ile derlenen bir nesne dosyası. **WASM yok**; `prism.js` C++'ı kullanmıyor, JS'e yeniden yazılmış (README: "matching JS port") |
| Collision Engine | Hayır (14 rastgele kaynak) | **Gemini API anahtarını `localStorage`'da saklıyor**, anahtar URL'de (`?key=`); sayfa açılışında **geliştiricinin adresine olay gönderiyor** (`anyel1to.github.io/.../api/event`) | Bu kısımlar BRIAN'a kesinlikle alınmamalı |
| HORIZON | Kısmen: Binance `ticker/24hr` ve Wikipedia araması gerçek, görsel olaylar rastgele | Collision ile aynı Gemini/`localStorage` ve dış telemetri | Order Flow Aurora vb. görseller gerçek ölçüm değil |
| SWARM | Hayır (boids simülasyonu) | `server.js` yol temizlemesi zayıf; yalnızca yerel geliştirme için, canlıya konmamalı | `js/spatialGrid.js` (uzamsal ızgara) ve typed array yaklaşımı performans için değerli |
| HELIX / AGENT RING | Hayır | `innerHTML` kullanımları | İş akışı görselleştirme referansı |
| TAPE | Hayır: `engine/tape.cpp` "No market data", 336 adım sentetik | — | Bkz. §6.1 |

### 6.1 TAPE sinyali (ALPHA hipotez kaynağı olarak)

- **Sinyal formülü** (`signal/src/main.rs`): `score = clamp(1.6·imbalance + 90·ret8 + 46·ret24, ±3)`.
  - Yön eşiği ±0,05, pozisyon eşiği güven > 0,42.
  - Rejim kuralı: düşük oynaklık ve küçük `ret24` → yatay; aynı işaretli momentum → trend.
- **Sabitler elle seçilmiş.** Kâr/zarar hesabında **maliyet, spread ve komisyon yok** (`pnl += prev_pos * ret`).
- **Sonuç:** Strateji değil, **hipotez listesi**:
  - Defter dengesizliği yönü öngörür mü?
  - Kısa ve uzun momentum uyumu trend rejimini işaretler mi?
- **Test edilebilirlik:**
  - Momentum ve oynaklık özellikleri `brian_micro_book_ticks` (REALTIME, ~367 bin satır, 3 Ekim'den beri) üzerinde test edilebilir.
  - **Dengesizlik test edilemez:** tabloda orta fiyat ve spread var ama alış/satış miktarı yok. Yeni veri toplamak gerekir; bu da onay ister.

## 7. BRIAN'da zaten var olanlar — ikinci bir sistem kurmamak için

Research Lab'ın omurgası CORE'da **mevcut** (sayılar 20 Ekim, salt okuma):

| İhtiyaç | Mevcut yapı | Durum |
|---|---|---|
| Hipotez kaydı | `brian_evolution_hypothesis_snapshots` (mekanizma, kanıt, karşı kanıt, başarı ölçütleri, aşama, belirsizlik) | ~8.558 satır |
| Deney | `brian_evolution_experiments` (control vs challenger; `REPLAY` / `STRESS` / `PROSPECTIVE_SHADOW`; asgari örnek, rejim, kontaminasyon kuralları) | 36 deney, son kayıt 18 Ekim |
| Sonuç | `brian_evolution_experiment_results` (maliyet sonrası net/brüt kenar, max düşüş, sızıntı tespiti, veri kalitesi, kararlılık) | 1.652 sonuç, **son sonuç 26 Eylül** |
| Karar | `brian_evolution_promotion_decisions` | 159 `REJECT`, 665 `KEEP_EXPERIMENTAL`, 0 terfi |
| Kaynak kütüphanesi | `brian_world_source_candidates` / `_assessments` (güven puanı, yetki sınıfı, erişim modu) | `source_kind` serbest metin → makaleler `RESEARCH_PAPER` türü olarak şema değişmeden eklenebilir |
| Kod üretimi ve inceleme | `brian_evolution_codegen_requests`, `_code_candidates`, `_engineering_runs` (compile/test/replay/stress/review, insan onayı) | Mühendislik ajanının alanı; dokunulmadı |

**Eksik halka:**
- Makale **iddiaları** ve **varsayımları** için yer yok.
- Rapordaki doğrulama durumları (`NOT_STARTED` … `REVIEW_REQUIRED`) hiçbir tabloda yok.

**Önemli açık sorun:** Deney koşucusu 26 Eylül'den beri sonuç yazmıyor. Aynı tarihte hazine scorecard kayıt işi de kapanmıştı (bugün v2 dönemiyle yeniden açıldı). Research Lab'ın deney hattı bu koşucuya dayanacağı için önce bunun nedeni bulunmalı. Bu, canlı zamanlayıcı alanı olduğu için performans denetimi bitene kadar yalnızca incelenecek.

## 8. Önerilen mimari (Claude tarafı)

```text
makale (açık erişim) ──► [A] Okuma paketi        yerel, Claude Code oturumu (ek API ücreti yok)
                            Paper2Skill disiplini: kaynak SHA-256, sayfa incelemesi, strict doğrulama
                         ──► [B] İddia + varsayım defteri   CORE: yeni 2 küçük tablo (append-only)
                         ──► [C] Hipotez                     mevcut brian_evolution_hypothesis_snapshots
                         ──► [D] Deney tanımı                mevcut brian_evolution_experiments (mode=REPLAY)
                         ──► [E] İzole çalıştırma            GitHub Actions (public repo → ücretsiz), sırsız
                         ──► [F] Sonuç                       mevcut brian_evolution_experiment_results
                         ──► [G] İddia durumu                PASS / FAIL / INSUFFICIENT_DATA / ...
                         ──► [H] UI                          salt okunur durum ucu (Codex sözleşmesiyle)
```

- **[A] Model maliyeti:** Okuma ve çıkarma işi, kullanıcının mevcut Claude Code planı içinde yapılır. Ayrı bir ücretli API (OpenAI/Anthropic) çağrılmaz. Paper2Code'un OpenAI hattı **kullanılmaz**.
- **[E] İzolasyon:**
  - Ortam: `workflow_dispatch` ile tetiklenen ayrı bir workflow; `permissions: contents: read`; **sır yok**; `timeout-minutes: 15`; tek dosyalık sabitlenmiş girdi.
  - Bağımlılıklar hash'li bir kilit dosyasından kurulur. Çıktı yalnızca artifact olarak alınır; CORE'a yazma ayrı ve güvenilir bir adımda yapılır.
  - Depo **public** olduğu için girdiler ve çıktılar da herkese açıktır. Bu yüzden yalnızca zaten kamuya açık piyasa verisi kullanılır: BRIAN'ın gizli kayıtları ya da hazine verisi bu runner'a gönderilmez.
- **Bütçe** (reproducibility deposundan): çalıştırma başına 3 deneme ve 15 dk, günde 60 dk. GPU 0. Büyük indirme 0; aşılırsa onay gerekir.
- **Doğrulama:** Sonucu üreten adım ile doğrulayan adım ayrı tutulur. Aynı örneklemde hem uydurulmuş hem test edilmiş sonuç "başarı" sayılmaz; eğitim ve test bölümleri önceden kaydedilir.

## 9. Ne alınır, ne alınmaz

| Bileşen | Karar | Gerekçe |
|---|---|---|
| Paper2Skill inceleme disiplini (hash, sayfa incelemesi, strict durumlar) | **Model al, yeniden yaz** | Kaynak izlenebilirliği; AGPL bağımlılığı servis tarafına girmesin |
| Paper2MCP "mevcut kodu sar, algoritma uydurma" + ayrı doğrulayıcı | **Kural olarak al** | Sahte rekonstrüksiyonu önler |
| Paper2MCP'nin dış kodu host'ta çalıştırması | **Alma** | Sandbox yok → GitHub Actions izolasyonu |
| Paper2Code pipeline'ı | **Alma** (yalnızca aşama referansı) | Yalnızca OpenAI, ücretli, LLM puanı, yol aşımı riski |
| Reproducibility iddia/varsayım şeması + bütçe | **Al, yeniden yaz** | Eksik halkayı tam doldurur |
| TAPE sinyal sabitleri | **Alma** | Elle seçilmiş, maliyetsiz, sentetik; yalnızca hipotez olarak test |
| Collision/HORIZON Gemini ve dış telemetri kodu | **Kesinlikle alma** | Tarayıcıda anahtar, üçüncü tarafa veri |

## 10. Paper Lab videosu — kaynak araştırması

- `starlabx.com` kendini açıkça simülasyon olarak etiketliyor: "SIMULATED DATASPACE", "STREAMING 0 EVENTS · SIMULATED", "ILLUSTRATIVE / SOURCE CHECK REQUIRED". Yazar, depo ya da makale listesi yok.
- "64 makale / 12 ajan / 11 yöntem" sayılarıyla eşleşen bir açık kaynak depo **bulunamadı**. Benzer çalışmalar (CodeScientist, ajanlarla ML makalesi tekrarlama çalışmaları) farklı sayılara sahip.
- **Sonuç:** Orijinal depo **doğrulanmadı**. Bu madde açık kalıyor; benzer görünümlü hiçbir şey orijinal kaynak gibi sunulmamalı.

## 11. 49 inç görsel prototip — backend tarafından Codex'e notlar

- **Hız iddiası:** "C++/WASM ile hızlı" iddiası PRISM için geçerli değil; demo saf JS. WASM'a ancak ölçülmüş bir darboğaz varsa geçilmeli; bu master notun 2. maddesiyle uyumlu.
- **Veri kaynağı:** Görsel düğümler mevcut uçlardan beslenmeli; animasyon sırf canlı görünmek için "aktif" göstermemeli. Kullanılabilecek uçlar:
  - `brian-prediction-status` (sözleşme: `docs/BRIAN_PREDICTION_UI_CONTRACT.md`)
  - `brian-world-status`
  - `brian-frontier-heartbeat-public`
  - `brian-shadow-scorecard`
- **Gecikmeli fiyatlar:** `brian-realtime-market-marks-public` uçundaki emtia fiyatları ~10 dk gecikmeli. `live` bayrağı kayıt anındaki gecikmeye göre hesaplanıyor. Görselde "canlı" etiketi için Codex'in PR175'teki "kaynak zamanı ≤ 2 dk" kuralı kullanılmalı. Backend düzeltmesi önerildi ama onay bekliyor.
- **Yük:** Yoklama görünürlükle sınırlanmalı ve tek uçuşla yapılmalı (sözleşme §6b). 20 Ekim olayı, ağır uçların sekme başına sık yoklanmasından çıktı.

## 12. Paper Lab deney motoru — ilk deney (PHASE 2) için aday

- **Ölçüt:** Açık erişimli olmalı, merkezi iddiası **BRIAN'ın zaten topladığı ya da kamuya açık** veriyle sınanabilmeli, GPU gerektirmemeli.
- **Aday 1** (yeni veri toplamadan): Tahmin piyasası **kalibrasyonu ve favori/uzun-oran yanlılığı** iddiası.
  - BRIAN'ın Polymarket anlık görüntüleri ve olay kümeli kalibrasyon motoru hazır.
  - İlk sonuçlanan piyasalar 29 Ekim'den sonra gelecek.
  - Uyarı: makale başka bir piyasadaysa bu bir "tekrar" değil "başka alanda test" olur ve öyle etiketlenmeli.
- **Aday 2** (ek veri ister): Emir akışı dengesizliğinin kısa vadeli fiyat etkisi (Cont, Kukanov ve Stoikov; arXiv 1011.6402, açık erişim).
  - TAPE'in dengesizlik hipotezini doğrudan sınar.
  - L1 alış/satış miktarı gerektirir. Binance `bookTicker` ücretsiz, ama yeni veri toplama **onay ister**.
- **Öneri:** Aday 1 ile başlamak; yeni toplama, maliyet ya da risk gerektirmiyor.
- Makalenin raporladığı sayılar PHASE 2'de makaleden çıkarılıp iddia defterine yazılacak; bu raporda **ezberden sayı verilmedi**.

## 13. İlk küçük geliştirme PR'ının kapsamı (onaydan sonra)

1. **Migration:** `brian_research_claims` ve `brian_research_assumptions`.
   - Append-only; `anon`/`authenticated` erişimi kapalı.
   - İddia durumu: `NOT_STARTED | RUNNING | INSUFFICIENT_DATA | PASS | FAIL | UNVERIFIED | REVIEW_REQUIRED`.
   - Makale kaynağı `brian_world_source_candidates`'e `source_kind='RESEARCH_PAPER'` ile bağlanır; hipotez ve deney mevcut tablolara referans verir.
2. **Salt okunur durum RPC'si ve UI sözleşmesi** (mevcut kalıpla aynı): her iddia için kaynak, durum, son deney ve sonuç.
3. **Postgres testleri:** şema, yetkiler, append-only ve durum geçişleri.
4. **Yapılmayacaklar:** Bu PR'da deney koşucusu, zamanlayıcı, edge function yayını ve dış kod çalıştırma yok. Canlıya uygulama ancak performans denetimi bittikten ve sen onayladıktan sonra.

## 14. Açık teknik sorular

1. **Koşucu:** Deney koşucusu neden 26 Eylül'den beri sonuç yazmıyor? (Zamanlayıcı alanı; yalnızca inceleme.)
2. **İlk makale:** Aday 1 mi, Aday 2 mi? Aday 2 yeni veri toplama onayı ister.
3. **İzolasyon yeri:** GitHub Actions kabul edilebilir mi? Public olduğu için çıktılar açık olur.
4. **Model kullanımı:** Okuma ve çıkarma işi Claude Code oturumunda yapılacak; ayrı bir API anahtarı kullanılmayacak. Uygun mu?
5. **Video kaynağı:** Orijinal Paper Lab videosunun deposu doğrulanamadı; araştırma açık kalıyor.
6. **Depo görünürlüğü:** `Om3r-Pro-Proje` herkese açık. Gizli bilgi olmadığı denetlenmeli ya da görünürlük yeniden değerlendirilmeli (kullanıcı kararı).
7. **Fiyat etiketi:** Backend'deki `live` bayrağı düzeltmesi için onay bekleniyor.
