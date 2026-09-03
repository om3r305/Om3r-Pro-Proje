# Brian 2026 Phase 3.3 — Broad Multi-Asset Historical Curriculum

Status: **PREREGISTERED FOUNDATION / SHADOW_RESEARCH_ONLY**

This phase broadens Brian's training world without weakening point-in-time discipline.
It is a training-data curriculum, not final evidence and not a recommendation universe.

## Hard boundary

Historical development remains strictly before `2026-01-01T00:00:00Z`.
The contaminated 2026 historical window is never reused as a pristine holdout or training extension.

## Crypto curriculum

The seed universe contains 30 explicit Binance Spot USDT symbols, including majors and selected fan tokens.
Every month is discovered independently from the official Binance Vision archive.

Rules:

- official `data.binance.vision` monthly 1m archives only,
- official `.CHECKSUM` verification before import,
- pre-listing and source-missing months remain missing,
- no interpolation of missing exchange history,
- no synthetic history before an instrument existed,
- the older locked Phase 2.x symbol allow-list is not silently changed.

## Cross-market curriculum

### ECB exchange rates

Provider: ECB Data Portal SDMX service.

- public, no API key,
- native daily frequency retained,
- explicit ECB working-day calendar,
- `includeHistory` can be used when historical revisions are required,
- no conversion to fabricated 5m bars.

### FRED / ALFRED macro and rates

Seed series:

- DFF — Effective Federal Funds Rate,
- DGS10 — 10-Year Treasury Constant Maturity Rate,
- CPIAUCSL — CPI,
- UNRATE — unemployment rate.

The production adapter must use FRED/ALFRED real-time/vintage semantics. A present-day revised value must never replace the value that was actually available at a historical decision time.

The API key is injected at runtime only. It is never committed to GitHub or stored in a dataset manifest.

### EIA energy

Seed series:

- PET.RWTC.D — WTI spot,
- PET.RBRTE.D — Brent spot.

EIA API v2 or official bulk data may be used. API keys, when required, are runtime-only. Native daily frequency is retained.

### Equity and gold

Broad tradable equity history and tradable gold history remain `LICENSED_REQUIRED` until a source with acceptable provenance, adjustment methodology, corporate-action treatment and licensing is selected.

Unverified web scrapers are intentionally not used to fill this gap.

## Point-in-time contract

Every non-market observation used by Brian must carry at least:

- canonical series identity,
- observation time,
- actual or conservatively reconstructed availability time,
- value,
- vintage/revision identity when applicable.

`asof_value()` exposes only rows already available by the decision timestamp. It rejects mixed series and the contaminated historical window.

If exact intraday release time is not supported by the source, the ingestion layer must use a conservative availability convention and disclose its precision. It must not pretend date-only metadata is second-level information.

## No fake frequency

Daily, weekly, monthly and event series remain on their native clocks. They may be joined to faster market data with causal as-of joins, but their values are not duplicated into invented OHLC candles and are never treated as newly observed every five minutes.

## Activation gate for the 100k curriculum

The full 100,000-life run remains blocked until:

1. enough verified multi-asset history is materialized,
2. source gaps/listing dates/calendars are audited,
3. PIT joins pass leakage tests,
4. a learner/update rule is preregistered,
5. small deterministic smoke curricula pass reproducibility and causal-boundary checks.

Synthetic/replay experience remains `TRAINING_ONLY`; it cannot by itself promote Brian or prove profitability.
