# Run 4 — weekly listing windows (code 4c62754)

- 2026-10-21 08:48–11:58 UTC, local operator machine, public Polymarket APIs only, 54,025 requests.
- Listing complete (weekly chunks; no truncated window), unlike run 1 and run 3.
- 16,062 markets / 2,600 events, horizon 30 h before resolution, window 2026-01-01 … 2026-10-01
  (after the paper's data cutoff).
- Slope 1.162, event-clustered bootstrap 95 % [1.114, 1.216] → **PASS** per the pre-registered rule
  (interval above 1): political markets are underconfident (prices compressed toward 50 %) out of sample.
- Smaller than the paper's 1.45 (not directly comparable: different estimator/snapshot, see result.json).
- shadow_only, trade_signal = false. A trading use needs its own pre-registered, cost-aware test.
