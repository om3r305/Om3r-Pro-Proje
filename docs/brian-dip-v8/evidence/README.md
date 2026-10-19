# 2026-09-07 tarihli V8 kanıt arşivi

Bu dizindeki özgün rapor, harness ve JSON; `0c1b1e88e758a71e7cf9732300b77f8f6e2f597d` commit'ine yönelik önceki incelemeden **baytları değiştirilmeden** alınmıştır. Dosya ve kaynak hash'leri [manifest.json](manifest.json) içindedir. Sonraki kararlar üst dizindeki belgelerde tutulur; tarihsel sonuçlar yeni uygulama sonucuymuş gibi güncellenmez.

| Dosya | Amaç |
|---|---|
| [Brian-V8-Inceleme-2026-09-07.md](Brian-V8-Inceleme-2026-09-07.md) | Özgün 8 bulgu grubu, olumlu tasarımlar, CI ve inceleme sınırı |
| [audit-repros.cjs](audit-repros.cjs) | Node 24 ile gerçek kaynağı izole VM'de çalıştıran tarihsel hata üretimleri |
| [audit-repro-results.json](audit-repro-results.json) | İlk çalıştırmanın 17 senaryo sonucu |
| [manifest.json](manifest.json) | Audit commit'i, arşiv dosyalarının SHA-256 değerleri ve kullanılan kaynak dosyaları |

## Ne kanıtlandı?

15 sorun senaryosu ve 2 olumlu kontrol için beklenen gözlemler üretildi. Pozisyon senaryolarında yapı okuyucunun çıktısı sabitlendi; pivot kontrolünde gerçek yapı fonksiyonları kullanıldı. Kaynak TypeScript, Node'un `stripTypeScriptTypes` fonksiyonuyla çalıştırıldı. DB, borsa, zaman ve HTTP bellek içi fixture ile temsil edildi; ağ çağrısı engellendi.

Bu gerçek Postgres transaction/constraint testi, gerçek tarayıcı testi, Deno type-check, üretim smoke-test'i veya kârlılık backtest'i değildir. Eşzamanlı MemoryDB senaryosu gerçek Postgres isolation garantilerini ölçmez. Özgün JSON'daki `verified: true`, senaryonun beklenen tarihsel gözlemi verdiği anlamındadır. Hata senaryosunda bu, **hata hâlâ üretildi** demektir.

Tarihsel testler normal uygulama CI'ına düzeltme testi diye eklenmez. Uygulama PR'ları güvenli davranışı bekleyen ayrı regresyon testleri ekler; özgün kanıt bu dizinde değişmeden kalır.

## Güvenli yeniden çalıştırma

Node 24 ve Git gerekir. Repo kökünde aşağıdaki örnek çalıştırılabilir. İncelenen commit yerel Git'te yoksa önce o commit fetch edilmelidir. Yalnız kaynak checkout'u için GitHub erişimi gerekebilir; harness DB/borsa erişimi kullanmaz.

```sh
audit_tmp="$(mktemp -d)"
git worktree add --detach "$audit_tmp/source" 0c1b1e88e758a71e7cf9732300b77f8f6e2f597d
cp docs/brian-dip-v8/evidence/audit-repros.cjs "$audit_tmp/audit-repros.cjs"
node "$audit_tmp/audit-repros.cjs" "$audit_tmp/source"
```

Yeni çıktı `$audit_tmp/audit-repro-results.json` olur. Script çalıştığı dizindeki JSON'u yazdığı için script'i yukarıdaki gibi geçici dizine kopyala; arşivdeki script'i doğrudan çalıştırıp özgün JSON'un üzerine yazma. Script audit commit'ini kontrol eder; source worktree'de yerel kod değişikliği olmamalı ve manifestteki kaynak hash'leri eşleşmelidir. Rastgele event ID gibi alanlar yeni çalıştırmada farklı olabilir; bu yüzden tüm JSON'un aynı hash'e sahip olması beklenmez.

## Senaryo → düzeltme paketi

| Senaryo | Paket |
|---|---|
| `legacy_ui_undefined_function_and_focus_override` | T1 |
| `runtime_read_error_silently_resets_state` | T2 |
| `same_cycle_expiry_reopens_locked_thesis` | T2 |
| `long_entry_accepts_stop_above_fill` | T2/T5 |
| `worker_ignores_intrabar_target` | T3 |
| `forty_losses_still_enable_calibrated_entry` | T5 |
| `minimum_notional_breaks_eight_percent_cap` | T5 |
| `sweep_enters_despite_opposing_flow` | T5/R3 |
| `same_thesis_persistence_freezes_calibration_and_entry` | T4 |
| `event_survives_snapshot_failure_and_duplicates_on_retry` | T2 |
| `short_uses_spot_data_and_ignores_disabled_short_setting` | T5 |
| `resolver_skips_first_prediction_minute` | T3 |
| `resolver_can_record_future_resolved_at_from_open_candle` | T3 |
| `resolver_includes_bar_remainder_after_deadline` | T3 |
| `parallel_forecast_persist_duplicates_same_thesis` | T4 |
| `positive_worker_rejects_v7_session` | Korunacak T1 kontrolü |
| `positive_pivot_confirmation_and_open_bar_exclusion` | Korunacak yapı kontrolü; gerçek closeTime kontrolü ayrıca gerekli |

`T*` paketleri [teknik planda](../TECHNICAL_PLAN.md), `R*` araştırmaları [araştırma planında](../RESEARCH_PLAN.md) açıklanır.
