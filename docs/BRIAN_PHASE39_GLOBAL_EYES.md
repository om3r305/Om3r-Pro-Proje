# Brian Phase 3.9 — Global Eyes Expansion

Phase 3.9 expands the Phase 3.8 Sensor Mesh with additional independent information families while the Phase 3.7 frozen seven-day control remains untouched.

## Live public/no-key providers

- Binance USD-M public market data: funding/premium, open-interest history and taker buy/sell imbalance. These are research sensors only; no account, credential or order endpoint is present.
- GDELT DOC 2.0: global news discovery. Article headlines are preserved as point-in-time evidence. Any directional headline pressure is explicitly low-authority and is not treated as source truth.
- ECB official EUR reference-rate history: daily FX context for EURUSD, EURGBP, EURJPY and EURCHF. Daily data remains daily; it is never interpolated into fake intraday ticks.

## Fail-closed provider sockets

The following domains remain unavailable until a compliant provider is deliberately provisioned out-of-band:

- X / social psychology
- Reddit social psychology
- on-chain wallet/entity flows
- licensed equities/ETF intraday feeds
- licensed gold/silver feeds

No scraper or invented value is substituted when those providers are unavailable.

## Collector health

`brian_collector_runs` is an append-only RLS-protected health ledger. Every collector records success/degraded/failed/skipped state, observed/stored counts and degraded-source/error metadata. This is intended to expose silent collector failure rather than letting Brian believe an absent eye is healthy.

## Scientific status

All Phase 3.9 observations are `PROSPECTIVE_DEVELOPMENT_SHADOW`. They may help diagnose opportunities and train a later checkpoint, but they are not a pristine final holdout and cannot trigger live promotion. Learning and live execution remain disabled.
