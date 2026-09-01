# Proje1 Dashboard

Profesyonel, "flashlı" bir pano. Dış bağımlılık yok (saf HTML+CSS+JS).

## Özellikler
- **Login sahnesi** (hafif koruma; sadece ön yüz)
- **Canlı PnL / WinRate / PF / MaxDD** özet kartları
- **Slot dağılımı (PRED/DIP/NEWS/OB)** pasta grafiği
- **PnL Zaman Serisi** çizgisel grafik
- **Sembol Isı Haritası** (PnL / WinRate bazlı)
- **Canlı Trade Akışı** (filtreleme ve arama)
- **Event & Telegram Özetleri**
- **Runtime Overrides Görüntüleyici (read-only)**

## Veri Kaynakları
1. **WebSocket (tercih edilir):** `ws://localhost:8765` (config_live.json > ui.socket_port).  
   Backend WS mesajları JSON olmalı (örnek: metrics, trades, events).
2. **Fallback (dosya okuma):**
   - `logs/trades.csv`
   - `logs/events.csv`
   - `logs/telegram_out/YYYY-MM-DD.jsonl` (en güncel dosyayı seçer)
   - `runtime/runtime_overrides.jsonl`

> Not: Dosya okuma için bu klasörü bir HTTP server ile servis etmeniz önerilir:
> ```bash
> # proje kökünde
> python -m http.server 8766
> # sonra http://localhost:8766/dashboard/ açın
> ```
> (Tarayıcı güvenlik politikaları nedeniyle "file://" ile fetch engellenebilir.)

## Çalıştırma
- **Yerel sunucu:** `python -m http.server 8766`
- **Adres:** `http://localhost:8766/dashboard/`
- **Login:** Herhangi bir isim + erişim kodu (örn. `boss` / `letmein`) — sadece ön yüz doğrulaması.

## Konfig
- WS adresi: `ws://localhost:8765` (config_live.json > ui.socket_port ile uyumlu)
- Poll aralığı: 5sn (dosya fallback)
- En fazla 10.000 satır JSONL/CSV işlenir (performans için)

## Notlar
- Pano, kod tabanına dokunmaz; yalnızca `dashboard/` klasörü eklendi.
- UI Tailwind benzeri stiller **kullanılmadan** saf CSS ile tasarlandı.
- Grafikler basit Canvas/SVG ile çiziliyor (harici kütüphane yok).