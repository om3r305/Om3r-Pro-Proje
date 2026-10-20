# BRIAN Darknet Observatory — preview acceptance

20 October 2026. This supersedes the unresolved performance gate in the earlier Singularity preview report for the specifically measured runs below. Production has not been deployed or changed.

## Changes

- Deadline-based 60 FPS pacing replaces a fixed 15.5 ms skip threshold. The old threshold systematically undershot 60 FPS on some refresh rates (for example 144 Hz). The new pacer is tested at 60, 100, 120, 144, 165 and 240 Hz. It does not manufacture frames when the browser supplies one callback per second.
- Continuous GPU-animated green carrier packets and slow node halos fill the source network. They are labelled decorative; they are not presented as actual messages. Pink packets and an arrival pulse correspond to changed source timestamps. Pause/reduced motion freezes both carrier motion and effect time.
- Green/black operator panels, existing violet/white black-hole art, a public-wire status panel and a clickable collector observation/storage matrix increase information density. All displayed status/counter values come from the existing public source. “Darknet” is a visual theme, not a connection to dark-web services.
- AUTO/HIGH/ULTRA controls choose device DPR, DPR 1.5 or DPR 2, with a strict 8-million-pixel drawing-buffer cap and adaptive scale. Existing buffer-size callers retain their original DPR 1.5 default. ULTRA at 5120×1440 produced a 5656×1414 scene buffer; DOM text remains at native CSS resolution.
- A fullscreen button uses the standard Fullscreen API with a visible F11 fallback. No browser security settings, OS display settings or global energy settings were modified.

## Data boundary

The same read-only SHADOW heartbeat and authority/freshness gates are used. No order, agent experiment, scheduler, paid service or backend write was started. Public source refresh remains once per minute per visible page. Source connections denote record membership and explicit ALPHA asset reference. Graph motion does not establish causality or profitable activity. The latest observed response had 15 source records and 14 relations.

## Frame pacing investigation

The prior session produced approximately 1000 ms frame intervals even for the unchanged old renderer. The current WebGL-free frame probe measured about 240 RAF callbacks per second at a mean 4.16 ms interval. The new renderer now measures approximately 60 actual drawn frames per second. The fixed-threshold pacing bug is independently demonstrated and corrected; that bug alone does not explain the previous entire 1–2 FPS episode. The exact earlier embedding/presentation condition has not been conclusively identified, so this report does not attribute all improvement to one cause or claim that background/occluded browsers always render at full speed.

Raw results are in `performance/darknet-measurements.json` and `performance/darknet-frame-probe.json`. Each main run is approximately 15 seconds at the iframe's actual CSS viewport, with real RAF-driven drawing. Studio fit scaling is presentation only and is not a physical-monitor fullscreen measurement. CPU figures time JavaScript work; GPU figures use WebGL timer queries, not device utilization percentages.

| CSS viewport | FPS | Frame p95 | JS / frame | GPU draw | Scene buffer |
|---|---:|---:|---:|---:|---|
| 5120x1440 | 59.99 | 16.70 ms | 0.637 ms | 0.701 ms | 5656x1414 |
| 3840x1080 | 60.04 | 16.70 ms | 0.617 ms | 0.406 ms | 4002x1160 |
| 3440x1440 | 59.99 | 16.70 ms | 0.579 ms | 0.520 ms | 3428x1460 |
| 1920x1080 | 60.36 | 16.70 ms | 0.583 ms | 0.462 ms | 2566x1390 |

All runs used ULTRA, quality scale 1, 15 source nodes and 14 relations. Eleven tests passed. The source freshness/authority contract is unchanged.


## Checks

Model/authority/freshness tests remain in place. Pacing tests cover the high-refresh regression, pause/resume deadline reset, and the fact that one callback per second cannot count as 60 FPS. The Ultra buffer test covers opt-in DPR 2, the unchanged legacy default, and the total pixel limit. JavaScript syntax and the visible source-to-inspector flow were checked. Long thermal runs, weak physical GPUs and exhaustive cross-browser/accessibility acceptance are not claimed.

## Physical monitor handoff

Verified through the preview's own Fullscreen API and visible telemetry on the user's monitor. The final page reported fullscreen=true, CSS viewport 4096x1152, screen CSS 4096x1152 and DPR 1.25. This corresponds to the previously verified physical 5120x1440 monitor at 125% scaling. The final DOM measured page width 4096 and height 1152, with no page overflow; lists/panels retain intentional internal scrolling.

The final approximately 15.47-second physical fullscreen run measured **59.996 FPS**, frame p95 16.90 ms, JavaScript 0.515 ms/frame and GPU drawing 0.883 ms. ULTRA remained at quality scale 1; the scene buffer was 4432x1004. Both physical-window runs are retained in `performance/darknet-fullscreen-measurements.json`.

The native Chrome launcher did not obtain a window; the successful handoff used the in-app preview's standard fullscreen support instead. The local preview was left fullscreen and marked as the deliverable.

Only local preview is authorized. No remote push or production publication forms part of this task.
