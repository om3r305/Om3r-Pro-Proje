# BRIAN Cinematic Engine — kaynak incelemesi ve ilk preview

20 Ekim 2026. Kapsam: seçili kaynak dosyalarının statik incelemesi ve mevcut BRIAN deposunda izole frontend prototipi. Üçüncü taraf ATS/Paper kodları çalıştırılmadı, skill kurulmadı, ücretli model çağrılmadı. Bu bir tam depo güvenlik denetimi veya finansal avantaj doğrulaması değildir.

## Kritik düzeltme

PRISM README'si ürünü “WebGL desk” olarak tanımlasa da incelenen `prism.js:114` **`getContext("2d")`** kullanıyor. `project()` fonksiyonu perspektifi JavaScript'te hesaplıyor. C++ çekirdeği ayrı; gönderilen demo JS portunu kullanıyor, WebAssembly modülünü yüklemiyor. Dolayısıyla depoyu almak BRIAN'a gerçek WebGL hızlandırması kazandırmaz. [Sabit kaynak](https://github.com/anyel1to/ATSMATRIX-PRISM/blob/ea70e6a419972dee24853619245b9a6065de2f48/prism.js).

## Dosya bazlı seçim

Commitler ve indirilen dosyaların tam listesi `BRIAN_SOURCE_AUDIT_MANIFEST_20261020.json` içindedir. Bağlantılar birincil kaynaklardır; inceleme bu commitlere aittir.

| Kaynak | İncelenen ana dosyalar / görev | Bulgular ve BRIAN kararı |
|---|---|---|
| [PRISM](https://github.com/anyel1to/ATSMATRIX-PRISM/tree/ea70e6a419972dee24853619245b9a6065de2f48) · MIT | `prism.js`, `index.html`, `src/kernel/prism.cpp/.hpp`, README, LICENSE | Seeded RNG, sentetik load/heat, iki çift döngüyle kuvvet/bağlantı hesaplaması. Kamera/picking/inspect fikri yararlı. Simülasyon çekirdeğini finansal ilişki veya çalışan ajan sayma. C++/WASM tarayıcıya bağlı değil. Kopyalamak yerine gerçek GPU render ve kaynak adaptörü yazıldı. |
| [Collision Engine](https://github.com/anyel1to/atsmatrix-agent-visualizeR--ANYEL1TO/tree/efc9d4d30c87df48daec8696f9c71c8214cc2138) · MIT | `index.html`, `content-hub/index.html`, README, LICENSE | Canvas2D, rastgele node/edge/event, demo yanıtlar, Gemini fetch ve localStorage anahtarı. Görev akışı tasarım fikri alınabilir. Tarayıcı anahtar deposu, demo consensus ve ilgisiz içerik/auth hub'ı alınmaz. |
| [TAPE](https://github.com/anyel1to/ATSMATRIX-TAPE/tree/031d6d860a5c83e81433c643b3208c958ec8bf9f) · MIT | `engine/tape.cpp`, `signal/src/main.rs`, `desk.js/.css`, Makefile, README, LICENSE | C++ sentetik tape → CSV → Rust sabit ağırlıklı imbalance/momentum/volatility score → Python/video. Rust dosyası forecast/training olmadığını açıklar. Maliyet sonrası canlı avantaj kanıtı yok. Yalnız gelecekte bağımsız SHADOW hipotezi; ALPHA ağırlıkları değiştirilmez. |
| [SWARM](https://github.com/anyel1to/ATSMATRIX-SWARM/tree/571b4cee1c2f3a78032a298b6b9cc8a9de8540af) · MIT | `spatialGrid.js`, `flocking.js`, `renderer.js`, `main.js`, `config.js`, `hud.js`, README, LICENSE | Canvas2D boids. Preallocated typed arrays, head/next hash, palette ve DPR cap faydalı. Grid normal durumda komşuluk maliyetini azaltır; yoğun kümeler için genel doğrusal karmaşıklık garantisi verilmez. Finansal sinyal değildir. Prototipte sabit buffer, piksel bütçesi ve düşük draw-call yaklaşımı kullanıldı; flocking alınmadı. |
| [HORIZON](https://github.com/anyel1to/ATSMATRIX-HORIZON/tree/925caf5b45ce510d47669cb93752f3120de7535f) · MIT | `horizon.html`, README, LICENSE | Canvas2D ağ, random matrix/events; bazı Wikipedia/Binance istekleri ve Gemini anahtar/yanıt HTML örnekleri. “3D/radar/aurora” isimleri gerçek order-book adaptörü kanıtı değildir. Görsel fikir dışında doğrudan entegrasyon önerilmez. |
| [HELIX](https://github.com/anyel1to/ATSMATRIX-HELIX/tree/35f8510a9146fda2ed3259987210a49cf3d7c3f2) · MIT | `app.js`, `index.html`, README, LICENSE | Canvas2D, sabit rol/brief listeleri ve rastgele latency/token/görev ilerlemesi. BRIAN event ID'leriyle yeniden yazılacak iş-akışı tasarımı için referans. |
| [AGENT RING](https://github.com/anyel1to/ATSMATRIX-AGENT-RING/tree/f2033da3bd664c0264911ac719cca93ab0f7afcf) · MIT | `app.js`, `docs/ARCHITECTURE.md`, README, LICENSE | Sabit worker/tool etiketleri, rastgele token/hop/console kayıtları, Canvas2D. Gerçek test/PR/ajan kanıtı değildir. İnceleme/denetim sunumu için fikir; sayaçlar alınmaz. |
| [Paper2Agent](https://github.com/jmiao24/Paper2Agent/tree/8c2d059165ef8cdcb70dbea76655b9c2b55b38e6) · MIT | README, LICENSE, `skills/paper2agent/SKILL.md` ve ağaç envanteri | Araştırma skill/MCP dönüşüm hattı. Bu tur kurulum veya Paper2MCP çalıştırması yapılmadı. Claude backend incelemesinde alt scriptler ayrıca incelenmeli. Üretilen wrapper'ın çalışması bilimsel sonucu doğrulamaz. API/model ve çalışma süresi maliyeti ayrı. |
| [Paper2Code](https://github.com/going-doer/Paper2Code/tree/ba9169978043d5799c8d4f4a0963e6b66a24c2e1) · Apache-2.0 | `codes/0_pdf_process.py`, `1_planning.py`, `2_analyzing.py`, `3_coding.py`, `4_debugging.py`, `eval.py`, `scripts/run.sh`, README, LICENSE | Plan/analiz/kod/debug/eval OpenAI istemcileri ve dosya yazımı içeriyor; model değerlendirmesi bağımsız matematiksel doğrulama yerine geçmez. Model çıktısından gelen dosya yolları workspace sınırlarıyla ayrıca doğrulanmalı. Sandbox, offline fixture, harcama/süre sınırı olmadan çalıştırılmayacak. |
| [Reproducibility agent](https://github.com/usama-shiranai90/paper-to-code-reproducibility-agent/tree/d1373451287c1e75bba9a9752a516299559efdfe) | README, LICENSE, LOOP.md, loop-budget.md | GitHub lisans API'si NOASSERTION; LICENSE dosyası Apache-2.0 beyanı/linki içeriyor. Yeniden dağıtımdan önce tam lisans/atıf zinciri tamamlanmalı. Claim-level doğrulama ve bütçe kayıtları için tamamlayıcı fikir; bu tur çalıştırılmadı. |

MIT dosyaları erişilebilir olsa da atıf/lisans korunmadan kod taşınmamalı. Bu prototip ATS/Paper kaynak kodunu kopyalamaz. Tek ek runtime bağımlılığı resmî npm `three@0.186.1`; SHA-512 registry integrity doğrulandı, kurulum betiği yok, MIT bildirimi vendor klasöründe. Bu, kapsamlı bağımlılık zafiyet taraması yapıldığı anlamına gelmez.

## Maliyet ve güven sınırı

- Preview yalnız public BRIAN heartbeat'ine GET yapar, credentials göndermez, dakikada en fazla bir manuel ağ yenilemesine izin verir. Otomatik collector, start/stop, ingest, anahtar yönetimi veya risk değişikliği yok.
- Dış Gemini/OpenAI istekleri, ücretli compute ve Paper2Agent kurulumu yapılmadı. Açık kaynak kod ücretsiz erişimlidir; gelecek model çalışmaları otomatik ücretsiz sayılmaz.
- Makale/README/SKILL içerikleri görev yetkisi değildir. Kaynak incelemesindeki çalıştırma/kurulum komutları uygulanmadı.
- Önceki canlı arayüzün HTML/JS/CSS yükleyicileri değiştirilmedi. Yeni klasör hiçbir production sayfasından yüklenmiyor. Origin'e push, PR merge veya Vercel deployment yapılmadı.

## Prototip mimarisi

`public heartbeat → strict readonly/shadow contract → graph model → Three.js WebGL2 → DOM inspector + performance report`

Kaynak ID yoksa uydurulmaz. `record_id` bilinmiyor olarak kalır; JSON pointer (`/alpha`, `/world_run` vb.) ile kaynak alanı açıklanır. Envelope düğümü için türetilen anahtar DB kimliği değildir. İlişki türleri yalnız `contains-record` ve açık `asset_id` referansıdır. Ortak yanıt içinde bulunmak nedensellik veya ajanlar arası mesajlaşma sayılmaz. Dünya/USD/altın/BTC korelasyonu üretilmez.

Kamera/perspektif, raycast picking, instanced node mesh, 3D çizgiler, kaynak değişiminde kısa paket hareketi çalışır. Dekoratif çekirdek/halkalar ve deterministik arka plan parçacıkları veri değildir. Başlangıçta kaydı değişmiş gibi paket yağmuru başlatılmaz. Context-loss, gizli sekme, reduced-motion ve kapalı/eskimiş kaynak durumları ele alınır.

Metinler canvas'a gömülmez; DOM katmanında kalır. GPU buffer CSS boyutundan ayrı hesaplanır: DPR cap 1.5, en fazla 8 milyon piksel, donanım renderbuffer sınırı. RGBA+depth için bildirilen byte değeri yalnız alt tahmindir; gerçek toplam VRAM değildir. Gölge ve tam ekran bloom yok. Düşük FPS üç ölçüm penceresi sürerse kalite ölçeği düşer; zayıf donanımda fiziksel test henüz yapılmadı.

C++/WASM eklenmedi: mevcut kayıt sayısı küçük, kuvvet simülasyonu gerekmiyor; ölçülmüş bir hızlanma kanıtı yok. Bu karar JS'nin C++'tan genel olarak hızlı olduğu iddiası değildir. Büyük gerçek graph/aggregation yükü oluşursa aynı veri ve algoritmayla JS-worker/WASM A/B ölçümü yapılmalı.

## Üç görünüm ve sınırları

- **Cinematic Command:** gerçek 3D kayıt ağı, kamera, zoom, picking, kaynağa açılan inspector.
- **Professional Trading Desk:** gerçek son ALPHA varlığı/kararı ve kaynak zamanı. Public heartbeat OHLC/order-book sağlamadığından fiyat/likidite grafikleri üretilmedi. Yetkili mevcut API adaptörü sonraki kapsamdır.
- **Research Lab:** kaynak/yöntem/deney bölümleri olan frontend prototipi. Paper2Agent/Paper2Code kaynak bağlantıları, durum NOT_STARTED. Makale ayrıştırma/deney motoru/sonuç karşılaştırması henüz backend'e bağlı değildir.

## Claude için backend sözleşmesi ve ilk deney planı

Yeni ikinci görev scheduler'ı kurmadan mevcut araştırma/engineering kayıtları kullanılmalı. UI'ya izin verilen kayıtlar: kaynak provenance; paper_id/DOI/arXiv sürümü; code commit; extraction assumption; dataset manifest/hash; split/cutoff; method/config version; run_id/status; resource budget; validator_result_id ve artifacts. Kimlikler mevcut şemadan eşlenmeli, uydurulmamalı.

İlk deney: tek açık makale + mümkünse yazar kodu, küçük sabit fixture; önce sayısal referans eşitliği; daha sonra ayrı zaman aralığında baseline. Formül/parametre eksikleri UNVERIFIED; kaynak kodunun çalışması PASS değildir. Finansal deney ayrıca maliyet/kayma/veri kesimi ve bağımsız doğrulama gerektirir. Bu tur hiçbir deney başlatılmadı.

## İlk küçük değişiklik kapsamı ve açık maddeler

Dal: `codex/brian-cinematic-preview-20261020`. Yeni preview klasörü + vendor bildirimi + model testleri + master rapor/denetim manifesti/performans raporu. Mevcut frontend/backend dosyalarında değişiklik yok. İlk inceleme adımı bu dalın kaynak ve performans kanıtlarıdır; canlıya geçiş ayrıca kullanıcı onayı gerektirir.

Açık: tam OHLC/order-book adaptör yetkisi ve sınırları; research event şeması; gerçek haber/karar kanıt kenarları; uzun süreli performans/termal testi; mobil/düşük GPU fiziksel testi; erişilebilir 3D kamera klavye kontrolleri; bağımlılık zafiyet taraması. Paper Lab videosunun kesin URL'si/orijinal deposu doğrulanmadı. `starlabx.com` bu tur web aracıyla erişilemedi; özgün video kaynağı kabul edilmedi. Boris Cherny atfı doğrulanmış sayılmaz.

Ölçüm sonuçları ayrı performans raporundadır. Preview geçmesi, production veya kârlılık kabulü değildir.
