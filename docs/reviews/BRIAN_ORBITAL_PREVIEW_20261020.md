# BRIAN Orbital Intelligence — Preview 02

20 October 2026. Isolated frontend deliverable, built after reviewing the ATS MATRIX Decision Desk, FOMOBILL, GPU Night Shift and Claude + JEV public videos. This design is an original implementation; no ATS code, video assets or simulated performance figures were imported.

## Open the deliverable

- Direct interface: http://127.0.0.1:8800/cinematic-preview/v2.html
- Resolution studio: http://127.0.0.1:8800/cinematic-preview/review.html
- Existing prototype remains available at `cinematic-preview/index.html`.

The local HTTP server must remain running. Opening the HTML through `file://` is not supported. Restart from the BRIAN checkout with:

```powershell
python -m http.server 8800 --bind 127.0.0.1 --directory monster-coins-pro
```

No existing production file, database, scheduler, backend function, credential or remote branch was changed. Nothing was published. No NIULAI4/Paper Lab experiment, paid model invocation, agent job or trade was started.

## What changed

The scene now has a dotted 3D globe, atmospheric limb shader, orbital instrumentation, a depth grid, original coarse continent silhouettes, 5,827 decorative points, and source-backed module nodes. The globe, camera motion, geographic silhouettes and orbital layout are decorative. Module positions do not claim geographic location. Lines denote membership of the same source response or an explicit ALPHA asset reference; they do not imply causality or agent communication.

Record nodes and their labels are selectable. The interface includes source filters, smooth focus, camera rotation, zoom, reset, pause, freshness states, raw JSON inspection, source-path copy, collapsible record activity, a performance dialog, JSON export, replay import, and visible 60-second automatic refresh control. Motion honors reduced-motion preference, and selected records can be focused while motion is paused.

The ultrawide layout uses a scene plus a separate decision/world-metrics column. At 5120×1440 the complete Command layout was measured at 1437 CSS pixels in height with no horizontal page overflow. The 390-pixel iframe test measured a 375-pixel body and 375-pixel scroll width, including an actual `OPEN_SHORT` response. Internal module-list scrolling is intentional.

Three workspaces are available:

1. **Command:** source topology, selected-record inspector, ALPHA evidence meter, actual world-run counters, module record activity and behavior context.
2. **Decision Desk:** actual shadow equity/cash/start/positions, latest ALPHA record, and an equity series built only from distinct timestamps observed during this page session. A single observation is labelled as such. This is not a market-price or OHLC chart.
3. **Research Lab:** real research/discovery/orchestration/sandbox/ocean collector states, with an explicitly unstarted experiment chain. The public source does not expose a paper library, experiment results, or a working Paper Lab backend.

## Data contract and boundary

The only network data source is the existing public heartbeat, requested with GET, omitted credentials, a 15-second timeout and a maximum of one request per minute per page while visible:

https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-frontier-heartbeat-public

The adapter requires `status=OK`, a valid source timestamp, `read_only=true`, `shadow_only=true` and `live_execution=false`. Invalid responses fail closed; retained records after transport failure become degraded. Cached, future and stale records cannot produce current activity. Public data is escaped before HTML interpolation; raw records are rendered with `textContent`.

The observed response produced **15 records and 14 explicit relations**: the response envelope; world, ALPHA, treasury, behavior and news fields; eight existing collectors; and the explicit asset reference. Source record IDs are preserved when supplied, including treasury `snapshot_id`; an absent ID is displayed as absent. Evidence score is labelled as evidence, not a probability of trading success. All money is labelled SHADOW.

New record timestamps may create short packets along existing source relations. Random trades, prices, news, agent activities and P&L paths are not generated. Recorded JSON imports are limited to 2 MB, marked REPLAY, and disable automatic refresh. There are no execution, deployment or experiment endpoints in this interface.

## Validation and performance

Seven meaningful model tests pass: source mapping/relations/IDs; fail-closed authority; stale/future/degraded behavior; numeric financial fields; and DPR, drawing-buffer pixel and hardware dimension limits. Both new JavaScript files pass `node --check`. Existing production application tests were not rerun because existing production files were not changed.

Browser checks covered module filters, scene-label selection, record selection, raw JSON toggle, camera focus/zoom/reset/pause, the three workspaces, settings telemetry, actual source population, 390-pixel mobile overflow, and the resolution studio. The direct preview console returned no errors during the checked session. This is not a claim of an exhaustive accessibility/security audit or every browser/device being verified.

Measured renderer: ANGLE / NVIDIA GeForce RTX 4090 / Direct3D11. The original host inspection reported Windows active mode 5120×1440 @240 Hz. Each measurement below is approximately 15 seconds using actual rendering at the iframe's stated CSS viewport. DPR and actual GPU buffer dimensions are included in the raw reports; they must not be inferred from the target monitor size. Studio presentation scaling does not change the inner CSS layout but may influence browser compositing behavior.

The scene uses locally vendored Three.js, already verified in the Phase 0 dependency manifest. It does not add C++/WebAssembly: measured JavaScript work does not establish a bottleneck that would justify that complexity. DPR is capped, drawing-buffer pixels are bounded, and sustained low measured FPS reduces the rendering scale. DOM text stays at CSS resolution. Weak physical GPUs, long thermal runs, full physical fullscreen, thousands of real records, and cross-browser acceptance remain untested.

CPU numbers measure JavaScript frame work only. GPU numbers measure draw-query duration only; neither is a device utilization percentage. Host-wide CPU/GPU percentages from the earlier Phase 0 report are not attributed to this updated design. No new device-percentage claims are made.

Raw data: `monster-coins-pro/cinematic-preview/performance/orbital-v2-measurements.json`. The original four development measurements are retained in `orbital-v2-development-measurements.json`; final layout measurements are reported below.

| CSS viewport | Duration | Render FPS | Frame p95 | JS / frame | GPU / draw | Drawing buffer |
|---|---:|---:|---:|---:|---:|---|
| 5120×1440 | 15.16 s | 60.10 | 16.70 ms | 0.568 ms | 0.398 ms | 2931×730 |
| 3840×1080 | 15.48 s | 60.13 | 16.90 ms | 0.586 ms | 0.334 ms | 2001×580 |
| 3440×1440 | 15.51 s | 60.17 | 16.90 ms | 0.591 ms | 0.510 ms | 1724×730 |
| 1920×1080 | 15.29 s | 60.16 | 16.80 ms | 0.561 ms | 0.345 ms | 1283×695 |

Final runs used browser DPR ≈1.0, 15 source records, 14 relations, quality scale 1.0, 59 draw calls and 5,827 decorative points. These are short iframe render/layout tests, not a claim of sustained performance on every device.


## Next integration boundary

Production integration requires separate user approval. Actual article titles, market prices/OHLC, liquidity depth, paper lists and experiment outcomes require the existing BRIAN backend's authorized data contracts. The prototype deliberately exposes their absence rather than substituting synthetic investment data. These remaining backend integrations are separate from this working frontend preview.
