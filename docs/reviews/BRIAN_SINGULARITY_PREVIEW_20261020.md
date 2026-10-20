# BRIAN Singularity — cinematic preview 03

Isolated local preview, 20 October 2026. Theme reference: https://www.instagram.com/reel/DchOQ3aIgy6/ . The reference was inspected in the browser; its video, music and images were not downloaded or embedded. All scene art is generated procedurally by the new renderer.

Open the interface at http://127.0.0.1:8800/cinematic-preview/v3.html or the resolution studio at http://127.0.0.1:8800/cinematic-preview/singularity.html . The existing local HTTP server serves these files. The v1/v2 prototypes and production files are unchanged. There has been no publication, remote push, backend modification, experiment start or execution request.

## Visual change

The original globe has been replaced by an animated black-hole composition: opaque event-horizon sphere, bright photon rim, layered accretion sheet, polar jets, procedural nebula, and 54,000 GPU-animated particles. Particles orbit at radius-dependent speeds. The disk has independent rotating spiral filaments and concentric bands. A separate four-pass pipeline draws the scene into a bounded render target, performs quarter-resolution horizontal/vertical bloom, and composites it with exposure, gamma and vignette. DOM text and clickable record labels are outside the bloom passes.

The palette is now midnight blue, violet, icy white and cyan, with colored module icons and brighter source relationships. This is an artistic interpretation, not a general-relativistic simulation. Continuous motion is explicitly decorative and does not imply a new trade, new message or increased market activity. Changed source timestamps create separately identifiable packets on existing record relationships.

## Data and controls

The same fail-closed ObservatoryModel and public read-only SHADOW heartbeat are used. The tested source produced 15 actual records and 14 explicit relations. Freshness colors, missing IDs, source JSON paths, raw JSON, ALPHA evidence, shadow equity and world counters remain available. No prices, paper results, simulated trading profits or news titles were invented. The public source still does not provide OHLC or liquidity depth.

Scene pause freezes the particle/disk/nebula clock. Zoom, focus, reset, source filters, raw inspection and the three workspaces are retained. Reduced-motion behavior remains enabled through the existing preference check. Replay imports remain labelled and disable auto refresh. Automatic refresh is a visible, controllable GET every 60 seconds while the page is visible, without credentials or mutations.

## Validation and unresolved performance gate

Both new rendering JavaScript files pass `node --check`. The seven existing model/authority/freshness/buffer tests pass. Source selection was checked against an actual ALPHA JSON record. Shader compilation and browser console were checked; no error was reported in the inspected sessions. This does not establish full cross-device or accessibility acceptance.

**The updated design has not passed the 60 FPS acceptance gate.** During this session the in-app browser repeatedly presented approximately 1–2 frames per second, with frame gaps near 1000 ms. A control tab running the unchanged v2 engine also showed approximately 1–2 FPS under the same session conditions, despite that engine's earlier approximately 60 FPS measurements. This comparison rules out attributing the whole slowdown to the new effect alone, but does not prove a single root cause. The page reported visible and focused; therefore the slowdown is not simply labelled as document-hidden behavior.

GPU queries were initially too sparse to yield samples in a 15-second slow-frame window; v3 now samples every ten rendered frames. No previous v2 FPS figure is claimed for v3. Any final captured v3 measurement is stored in `performance/singularity-v3-measurement.json`. Foreground physical-browser/fullscreen acceptance and all four resolutions at sustained 60 FPS remain unverified. Native GPU work is not replaced with a timer loop to manufacture a higher FPS count.

Automatic resolution reduction is gated by a minimum sample count and evidence of drawing pressure, so isolated one-second presentation gaps with low JS/GPU work do not automatically destroy fidelity. Device-pixel ratio and drawing-buffer limits remain bounded. Extra scene render-target and bloom buffers are included in the displayed lower-bound buffer estimate; textures, driver overhead, geometry and other applications are not presented as measured app VRAM.

The preview is reviewable for composition and actual source mapping. It must remain a preview until the frame-pacing issue is resolved and foreground acceptance measurements pass. No production deployment is authorized by this delivery.
