# Brian entegrasyon kontrolü — 18 Ekim 2026

Bu rapor aynı günkü BRIAN_RESEARCH_START_20261018.md raporunun çalışma modu ve durdurma bölümlerini günceller. Kullanıcının tüm modülleri ayağa kaldırma talebiyle SHADOW_INTEGRATED modu açıldı. Gerçek para veya borsa emirleri açılmadı; kârlılık kanıtlanmış değildir.

## Çalışan bağlantılar

07:58 ve son olarak 08:13 UTC üretim heartbeat verisi, mevcut frontier-status-truth.js durum hesabıyla kontrol edildi: Dünya, Davranış, ALPHA, Hazine, Araştırma ve Okyanus altısı da güncel. Bu, bağlantı/son çalışma sağlığıdır; tüm kaynakların tamamlandığı veya kararların kârlı olduğu anlamına gelmez. Üretim /api/brian/brian-frontier-heartbeat-public HTTP 200 / OK; news_pipeline HEALTHY, pipeline_stalled=false. 08:07 örneğinde backlog 7 olay, en eski 56 saniye; eşik 720 saniye.

| Hat | Doğrulama |
|---|---|
| Dünya | Collector SUCCESS 08:04:15 UTC; yeni frame yazımı var |
| Davranış | SUCCESS; taze fiyat/taker/hacim bağlamı, bağımsız yön oyu sayılmaz |
| ALPHA | Core'a aktarılmış SUCCESS 08:05:23; kararlar ve değişmez fiyat referansı mevcut |
| Hazine | SUCCESS 08:02:01; kasa/equity 4943.521744958152 USD, başlangıç 5000; sıfırlanmadı |
| Araştırma | Orchestrator, researcher ve sandbox ayrı SUCCESS kayıtları üretti |
| Okyanus | SUCCESS; ocean-integrated-20261018 adlı 24 saatlik gözlem başladı, önceki testler korundu |

Tarayıcıda üretim sayfası açıldı fakat bu tarayıcı oturumunda kullanıcının dashboard anahtarı bulunmadığından uygulamanın kendi kilit ekranında kalındı. Kilit aşılmadı. Doğrulama canlı API → gerçek arayüz durum mantığı ve otomatik regresyon kontrollerini kapsar; giriş yapılmış tam görsel akış testi yapıldı denmez.

## Ortak işletim kontrolü

20261018075418_integrated_shadow_operator_control.sql uygulandı. Core start/stop artık sadece sekiz açıkça seçilmiş işi yönetir: treasury, evolution orchestrator, researcher, sandbox, ocean, intel-label, missed auditor ve trend-shadow. Eski yinelenen Core veri cronları açılmadı. RT orchestrator ve RT Core scheduler her döngüde Core'daki yetkili kontrol uç noktasından bayrağı okur; kontrol ulaşılamazsa sağlayıcı işi başlatmaz. Çalışmakta olan bir HTTP isteği anında iptal edilmiş sayılmaz.

Gerçek STOP → iki RT invoker çağrısı → yanıt okuma → START testi yapıldı: her iki RT görevi HTTP 200 OPERATOR_STOPPED döndü. Start yeniden uygulandı; sistem son durumda açık. Stop sırasında heartbeat de yenilenir; heartbeat yenileme hatası STOP işlemini geri almaz. Dip işleri değiştirilmedi.

Core gerçek programları: Hazine 2-59/5 dakika; orchestrator 13,33,53; researcher 24,54; sandbox 29,59; Ocean 11,26,41,56; intel-label 7,22,37,52; auditor 7,17,27,37,47,57; trend günlük 00:12 UTC. İş adındaki eski 1m/5m ibaresi gerçek cadence değildir. RT iki cron her dakika tetiklenir; Core veri görevi tek dakikalarda ALPHA aktarımı, çift dakikalarda dönüşümlü bir görevdir. Çakışma lease/rate guard'ları korunur. Hazine promotion kapısı KEEP_EXPERIMENTAL olarak kapalı kaldı; araştırma/sandbox otomatik kod uygulama izni almadı.

## Haber kaynakları

Direct Wire'a toplam 6500ms sınırı paylaşan, boş/HTML yanıtı RSS başarısı saymayan yedek deneme eklendi. Yahoo Finance birincil RSS hâlâ HTTP 404. Yahoo News topstories yedeği RSS döndürüyor; bu genel haber akışıdır, eski finans kapsamının aynısı değildir. Gerçek URL, fallback_used ve primary_errors provider kayıtlarına yazılır; fallback yayıncısı Yahoo News olarak korunur. Bloomberg, CNBC ve Al Jazeera döngüsü de başarılı. İlk Yahoo yedek örneğinde tazelik penceresine giren haber yoktu; haber uydurulmadı.

08:06 UTC'de aşağıdaki 14 RT resmî uç noktanın last_success_at alanı güncel, HTTP 200 ve last_error=null idi: BIS, Bank of Canada, Bank of England, Bank of Japan, Bundesbank, CFTC Federal Register, Consilium, ECB, Fed monetary, Fed speeches, Minneapolis Fed, New York Fed, SEC press, US Treasury press. Bu anlık sağlayıcı başarısı gelecekte kesintisizlik garantisi değildir.

20261018080622_mirror_verified_realtime_official_ownership.sql bu 14 doğrulanmış kaynağın doğal endpoint kimliklerini, URL/adapter ve RT collector sahipliğini Core registry'ye eşitledi. İkinci bir worker açılmaz; Core generic-feed worker zaten collector_owner=null olanları seçer. ACTIVE registry etiketi tek başına tüm kaynakların ingest edildiği kanıtı sayılmaz. Diğer HTML_DIFF_PENDING/JSON_PENDING/AUTHENTICATED_WEBSOCKET_PENDING adaptörleri tamamlandı sayılmıyor.

## Karar → maliyet → sonuç zinciri

İlk auditor timeout'u önceki iki migration ile giderildi. Karar aktarımı 2 dakikaya çekildi; 5/15/60 dakika ufukları ve +120 saniye terminal toleransı korunuyor.

Yeni incelemede WAIT kararlarında maliyet kanıtının yokluğu saptandı. Compiler artık en fazla mevcut sekiz varlık için karar öncesi L2 örneği alır; aynı snapshot'ın BUY/SELL maliyetlerinden yüksek olanını metadata.audit_cost içine AUDIT_ONLY / L2_OBSERVED olarak kaydeder. Bu, belirli 3–20 USD sanal ticket ve mevcut 10bps ücret varsayımıyla hesaplanan bir tahmindir; gerçek borsa fill'i değildir. WAIT ana execution-cost alanı ve strateji eşikleri değişmez; audit quote emir izni değildir. Yönlü kararların mevcut execution maliyeti önceliklidir.

Auditor Core'daki en fazla 60 pending kararın metadata'sını yükler. Audit-only maliyeti yalnız WAIT/VETO için, kaynağı/provenansı doğru ve örnek karar öncesinde en fazla 120 saniyelikse kabul eder. Gelecek, eski, geçersiz veya eksik maliyet sıfır sayılmaz. Sonuç metadata'sı maliyet kaynağını taşır. Eski kayıtlara sonradan maliyet eklenmedi, geçmiş zarar/kararlar silinmedi. 08:04:05 UTC yeni WAIT kararlarında gerçek L2 provenance görüldü; 08:05 Core aktarımında da korundu.

08:11 UTC gerçek auditor çağrısı HTTP 200 RESOLVED döndü: 5 sonuç ve 5 missed-opportunity receipt yazıldı. BTC/ETH/BNB/SOL/XRP için 300 saniyelik sonuçlar observed_audit_only maliyet kaynağıyla WAIT_JUSTIFIED_BY_COST sınıfında. Bu ilk örnek kâr veya kaçırılmış büyük fırsat değildir; fiyat hareketi tahmini maliyeti aşmamıştır. Aynı çağrıda 95 olgun ufuk hâlâ unresolved: eski maliyetsiz kararlar, seyrek dönen radar varlıkları veya fiyat boşlukları; veri uydurulmadı. 15/60 dakika sonuçları ilk beş dakikalık örnekle tamamlanmış sayılamaz.

08:10 döngüsünde türev toplayıcısı iki spot sembolü için DEGRADED döndü (SOXLBUSDT/SNDKBUSDT). Spot radarın vadeli sözleşme listesi olmadığı görüldü. Toplayıcı, Binance exchangeInfo kaynağındaki TRADING / PERPETUAL / USDT sözleşmeleriyle kapsamı kesiştirecek şekilde düzeltildi. Kapsam dışı semboller coverage.skipped_non_active_usdt_perpetual alanında açıkça raporlanır; diğer sağlayıcı hataları yine DEGRADED olur, katalog yoksa işlem FAILED olur. Boş kapsam başarı diye sunulmaz. Yayımlanan v2 gerçek çağrısı HTTP 200 CAPTURED döndü: 15 radar adayından 13 aktif sözleşme, 18 gözlem, degraded_sources=[]; iki kapsam dışı sembol açıkça bildirildi. Resmî sözleşme bilgisi: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information.

## Yayın ve kontroller

- CORE brian-core-scheduler-bridge v7, missed-opportunity-auditor-v3 v9.
- RT brian-realtime-derivatives-eye v2; brian-realtime-orchestrator v15; brian-realtime-core-scheduler v19; brian-direct-wire-eye v3; gerçek compiler slug'ı brian-alpha-decision-compiler v9. Repo kaynak dizini brian-realtime-alpha-decision-compiler'dır; bu ad doğrudan deploy slug'ı olarak kullanılmamalıdır. Yanlış isimle oluşan alias 410 DECOMMISSIONED döndüren iş yapmayan uç noktaya çevrildi.
- JWT kapıları yeniden tasarlanmadı: önceden verify_jwt=false olan bu uç noktaların mevcut internal/cron anahtar doğrulaması korunur; yeni operator_status da aynı doğrulamaya tabidir. Veritabanı fonksiyonunun execute izni yalnız service_role'dadır.
- Security advisor tekrar çalıştırıldı. Projede önceden bulunan RLS/no-policy bilgi kayıtları ve iki public/authenticated SECURITY DEFINER uyarı sınıfı sürüyor. Bu çalışma genel güvenlik incelemesini tamamlamış sayılmaz. İlgili rehberler: https://supabase.com/docs/guides/database/database-linter?lint=0028_anon_security_definer_function_executable ve https://supabase.com/docs/guides/database/database-linter?lint=0029_authenticated_security_definer_function_executable.
- 46 Node kontrolü: ortak STOP/fail-closed, nedensel audit maliyeti, RSS fallback/yayıncı, vadeli sözleşme kapsamı, scheduler cadence, Dip/Brian ayrımı, skor muhasebesi, dashboard veri eskimesi ve haber→sanal karar hattı. Bunlar kârlılık testi değildir.
- Frontend Dip ayrımı önceki PR #136 üretiminde mevcut; bu değişiklikler Supabase backend'e doğrudan uygulandı. Ana site https://monster-coins-pro-seven.vercel.app/.

## Açık sınırlar ve takip

Kapsamı tamamlanmamış kaynak adaptörleri, eski Yahoo Finance adresi, Agent Reach işçisi ve yeterli maliyet sonrası strateji kanıtı açık işlerdir. Agent Reach bu çalışmada kurulmuş veya bağlanmış değildir. Resmî proje tekrar incelendi: https://github.com/Panniantong/Agent-Reach; bu mevcut Supabase veri adaptörlerine otomatik entegrasyon anlamına gelmez. Tüm dünya haberlerine ücretsiz ve kesintisiz erişim sağlandığı söylenemez. Önceki 30 günlük ölçümde maliyet sonrası avantaj negatifti; yeni altyapı sağlığı bunu kâra dönüştürmüş sayılmaz.

Mevcut Brian saatlik çalışma takibi yeni SHADOW_INTEGRATED moduna güncellendi; salt okunur izleme yeni ve anlamlı eksik/kesinti/kaçan fırsat/ölçüm raporlayacak. Kullanıcı stop verirse otomatik açma, gerçek emir, eşik/risk/bakiye değişikliği veya ücretli servis yok. Görev enabled=true; servis yanıtında next_run_time=null döndüğü için gelecekteki teslimat zamanı ayrıca garanti edilmez.
