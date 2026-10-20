# Event journey and data freshness — isolated preview

The preferred electron artwork, central arrival wave and black/silver theme are retained.

The journey strip shows the selected real source record (or the latest timestamped record when the envelope is selected), its JSON path and time, the public response envelope, and the local notification arrival time. Clicking the record/envelope opens the existing inspector. Initial observations are labeled as initial; they do not invent a journey or historical arrival. Arrival is recorded only after the existing 2.8-second current-record notification reaches the core.

Connections are limited to envelope membership, the explicit alpha.asset_id reference, or unique exact matches from *_id fields to record IDs present in the same response. These do not establish a causal news/decision/treasury chain. The strip states that limit.

Data-health rules are shared by the sidebar, DOM electrons and GPU carrier attributes:
- Current envelope + current module: moving decorative carrier allowed.
- Stale module, missing/old envelope, failed module or disconnected transport: carrier hidden; electron motion frozen and artwork dimmed.
- Replay: no live module carriers.
- Existing source-state checks run every 10 seconds; network refresh remains once per minute.

Health refers to record freshness and reported status, not proven service availability or trade profitability. No new source endpoints, credentials or production changes are introduced.

Six health/journey tests pass (including offline/stale/replay suppression, explicit membership, exact unique ID matching and empty state). Fullscreen benchmark: 60.0304 FPS, 6108 × 1374 drawing buffer, p95 16.9 ms, mean JS 2.257 ms, GPU draw 1.782 ms. Renderer reports no shader errors.


Live browser freshness/recovery verification (no fabricated records): automatic fetch was temporarily paused on the real 22:56:02 response. At 22:58:16, 0 flows were current and 15 were stale/stopped; the news electron rotation remained exactly unchanged between two observations. Automatic refresh was restored. The real 22:58:01 response returned, all 15 flows recovered, and the news journey recorded local core-notification arrival at 22:58:25. No WebGL errors were reported. Automatic refresh is enabled for the delivered preview.
