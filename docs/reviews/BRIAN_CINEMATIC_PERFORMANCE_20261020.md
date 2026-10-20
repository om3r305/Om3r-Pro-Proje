# BRIAN Cinematic Engine — ilk performans ölçümü

20 Ekim 2026. İzole localhost preview; production deploy yok. Kısa ölçüm, 60 FPS için ön kabul kanıtıdır; uzun süreli veya büyük finansal graph kapasite garantisi değildir.

## Donanım ve yöntem

Windows aktif mod: **5120×1440 @240Hz**. RTX4090, 24564MiB VRAM. Tarayıcı rendererı ANGLE / NVIDIA RTX4090 / Direct3D11 olarak doğrulandı. DPR=1.25. WMI preferred-mode 3840×1080 kaydı aktif çözünürlük diye kullanılmadı.

Her çözünürlükte yaklaşık 15 saniyelik ölçüm, gerçek boyutlu same-origin iframe içindedir. Dış otomasyon tarayıcısı görüntüyü kırpabilir; bu fiziksel tam ekran/monitörün bütün alanının aynı anda görünür olduğu test değildir. Iframe layout viewportu ve gerçek GPU drawing buffer boyutu ayrı kaydedildi. 5120 layout, yalnız CSS ile büyütülmüş bir bitmap değildir.

Sahnede **7 gerçek kaynak düğümü, 6 yapısal/açık varlık referansı kenarı**, 1200 dekoratif nokta vardır. Kamera dönüşü açık; hedef 60 FPS frame limiti. Metinler DOM üzerindedir. Çekirdek/halkalar/arka plan parçacıkları veri değildir. Bu örnek 1000 gerçek entity veya karmaşık order-book yükünü temsil etmez.

| Layout viewport | Duration s | FPS | Frame p95 ms | JS ms/frame | GPU ms/draw | Buffer | Quality |
|---|---:|---:|---:|---:|---:|---|---:|
| 5120x1440 | 15.79 | 60.16 | 16.90 | 0.216 | 0.031 | 3742x1390 | 1 |
| 3840x1080 | 15.25 | 60.13 | 16.90 | 0.211 | 0.039 | 2782x940 | 1 |
| 3440x1440 | 15.62 | 60.17 | 16.90 | 0.251 | 0.045 | 2482x1390 | 1 |
| 1920x1080 | 15.04 | 60.15 | 16.90 | 0.253 | 0.018 | 1431x940 | 1 |

GPU sorgusu `EXT_disjoint_timer_query_webgl2`; disjoint sonuçlar kullanılmaz. Her pencerede 15–16 GPU örneği var. JS iş süresi yalnız uygulamanın frame fonksiyonunu kapsar; browser/compositor/system CPU değildir. Ortalama GPU draw süresi cihazın GPU kullanım yüzdesi değildir. İlk shader derlemesi bu ısınmış ölçüm penceresinden önce gerçekleşmiştir.

## Ayrı cihaz yükü örneklemesi

Preview açıkken 80 adet sistem/device örneği alındı. Başka uygulamalar ve açık sekmeler de bu değerleri etkiler; prototipe ait kaynak tüketimi diye atfedilemez. Ham zaman damgaları JSON içindedir. Tüm çözünürlük koşularıyla birebir eşzamanlı eşleştirme yapılmadı.

| Device metric | Mean | Max |
|---|---:|---:|
| System CPU | 3.33 % | 8.27 % |
| Device GPU | 29.39 % | 47.00 % |
| Total used VRAM | 2719.74 MiB | 2765.00 MiB |
| Device power | 36.09 W | 75.10 W |

Browser buffer tahmini RGBA+depth için alt sınırdır; gerçek VRAM tahsisini veya shader/driver belleğini içermez. GPU bellek yüzdesi uydurulmadı.

## Kabul ve eksikler

- 60 FPS hedefi küçük veri kümesinde dört layout ölçümünde yaklaşık sağlandı. Kısa pencere, frame sayımı ve vsync farkı nedeniyle 60'ın çok az üstündeki sonuçlar görülür.
- Orbit/reset, zoom, hareket durdurma, gerçek Dünya kaydı seçimi ve inspector alanları UI üzerinden doğrulandı. Üç görünüm geçişi çalıştı.
- Direct preview konsolu kontrolde boştu. İlk iframe aracında kaynağı belirtilmeyen bir MutationObserver hatası görüldü; preview/vendor kaynaklarında MutationObserver yok. Atıf kesinleştirilemedi; sayfayı veya ölçümü durdurmadı.
- Düşük GPU fiziksel testi, uzun/termal çalışma, binlerce gerçek kayıt, mobil kamera etkileşimi ve context-loss geri kazanım testi yapılmadı. Adaptif kalite yolu mevcut; zayıf cihazda doğrulandığı iddia edilmez.
- Public kaynak OHLC/order-book sağlamıyor; Desk yalnız mevcut karar/kasa kapsamını gösteriyor. Research Lab backend ve makale deneyi bağlı değil; NOT_STARTED.
- C++/WASM ölçülmüş hızlanma olmadan eklenmedi. Mevcut JS işi düşük; yeni hesap yükü için kontrollü karşılaştırma gerekir.
- NIULAI4/makale deneyi veya collector başlatılmadı. Canlı sürüm, risk kapıları ve kasa kayıtları değiştirilmedi.

## Tekrar çalıştırma

Depo kökünden:

```powershell
python -m http.server 8800 --bind 127.0.0.1 --directory monster-coins-pro
```

Prototip: http://127.0.0.1:8800/cinematic-preview/index.html

Ölçüm kafesi: http://127.0.0.1:8800/cinematic-preview/bench.html

HTML dosyasını file:// ile açmayın. Ham ölçümler `cinematic-preview/performance/`, sabit Three.js sürümü/integrity `vendor/manifest.json` içindedir. Yayın/merge ayrıca kullanıcı onayı gerektirir.
