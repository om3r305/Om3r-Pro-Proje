# Brian Phase 4.0 — Intrabar Reaction Eye

## Why this exists

The manual XRP referee case exposed a timing blind spot: the five-minute price-structure eyes could correctly recognize a move after it was already mature while missing a large part of the path inside the still-open five-minute bar. Phase 4.0 adds a separate prospective intrabar shadow layer so fast acceleration, volume expansion, liquidity sweeps/reclaims and taker imbalance can be observed earlier.

This is **not** a retroactive strategy fitted to the XRP chart. The XRP case motivated the class of failure, but the preregistered rules are generic, normalized by each asset's recent one-minute volatility, liquidity/activity quality and observed spread. The new layer starts only from deployment time and has no historical backfill.

## Isolation from Phase 3.7

The frozen Phase 3.7 NATIVE and PROFIT policies, thresholds, learner state, $500 live-shadow books, and five-minute collector remain unchanged. Phase 4.0 writes separate sensor observations, reaction events and virtual micro-book ticks. It cannot place exchange orders, cannot promote itself, and cannot change Brian's learner.

## Coverage and cadence

Every minute the intrabar eye scans the current top 50 Binance USDT radar candidates plus BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT and XRPUSDT as always-on core symbols. It consumes only public Binance Spot endpoints:

- current and recent **1-minute klines**, including the causal partial current minute;
- latest **aggregate trades** for short-horizon price velocity;
- current **bookTicker** bid/ask for mid price and spread;
- the latest already-recorded Brian universe snapshot for asset selection and confidence context.

The collector is rate-guarded and fail-closed. Missing aggregate-trade data degrades that source rather than fabricating a signal.

## Five independent micro eyes

1. `velocity-micro` — live trade-price velocity over the available 5–60 second window, normalized by recent one-minute realized volatility.
2. `volume-burst-micro` — partial one-minute quote-volume pace plus directional body acceleration relative to the previous 20 one-minute bars.
3. `breakout-micro` — pre-close one-minute structure break with a spread/cost/volatility-aware buffer, so Brian does not have to wait for the five-minute candle close.
4. `reclaim-micro` — same-minute or prior-minute liquidity sweep followed by a reclaim/rejection, designed to observe dip/reversal paths rather than only bar endpoints.
5. `taker-flow-micro` — partial one-minute taker-buy share and volume participation.

A single family cannot create an actionable shadow event. At least two independent groups must align and the aggregate score must clear the preregistered consensus floor.

## Anti-chase protection

Earlier detection must not simply turn Brian into a late momentum buyer. The runtime therefore measures five-minute extension in realized-volatility units. When an otherwise eligible move is already at least 3.5 sigma extended and fresh velocity is stale/conflicting, taker flow conflicts, or the most recent 30-second impulse is materially decelerating, the event is recorded as `VETOED_LATE_CHASE` instead of opening the consensus virtual micro-book.

This distinction is important: **recognizing a move is not the same thing as having a tradable entry**.

## Prospective evaluation

Each specialist keeps its own cost-aware virtual micro-book only when it is active or closing an existing virtual position. A separate `intrabar-consensus` micro-book measures the combined layer. Trading-cost accounting uses the same 10 bps fee, 1 bps slippage and observed half-spread per side convention used elsewhere in Brian's development shadow work.

Reaction events explicitly store support groups, conflicts, observed mid/spread, estimated round-trip cost, extension sigma and whether the entry was vetoed as late. These records are append-only and are intended for forward manual-referee and missed-opportunity analysis.

No claim of edge should be made from the motivating XRP example or from a handful of early events. The new layer must earn its value prospectively by showing that earlier alerts improve cost-adjusted outcomes without simply increasing churn or buying exhausted spikes.
