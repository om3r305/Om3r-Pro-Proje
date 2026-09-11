# Brian DIP V8.4.1 — Brian Authority

Status: DESIGN / SHADOW ONLY

## Product intent

Brian is the only strategic decision-maker. The platform supplies market facts and executes Brian's SHADOW decision; it does not act as a second trader by stacking independent strategy vetoes.

Brian reads numerical candle/market data (the chart itself): sealed OHLC candles, swing highs/lows, trend/structure, momentum, flow/book context and trading cost. These are inputs, not independent blockers.

No model can know the exact future bottom or top. The measurable objective is to enter close to a confirmed/probable dip before the recovery is mature, and exit close to a probable top / before meaningful reversal.

## One decision interface

Every evaluation produces exactly one Brian decision:

- `LONG`
- `SHORT`
- `WAIT`

When LONG/SHORT, Brian also owns:

- entry intent / timing
- target
- invalidation / stop
- capital allocation fraction (0..1 of the user-selected SHADOW test capital)
- confidence
- short natural-language reason

## Market interpretation

Brian may use all available market facts, but they are evidence rather than hard strategy gates:

- sealed 1m / 5m / 15m / 1h / 4h candles
- recent range position: near low / middle / near high
- swing/pivot structure, BOS/CHOCH/sweep/failed break
- momentum / recovery / exhaustion
- fast and slow order flow and book pressure
- spread, fees and funding
- structural level ladder L1/L2/L3...

The target planner exposes the level ladder to Brian. L1 is an obstacle, not a mandatory take-profit. Brian can choose a farther structural target when its chart thesis supports continuation.

## Soft evidence, not blockers

These existing Package-1 vetoes become Brian inputs / warnings and DO NOT independently forbid a trade:

- `TARGET_BELOW_COST`
- `ECONOMIC_RR_TOO_LOW`
- `WAIT_RETEST`
- `ENTRY_TOO_LATE`
- `DIRECTION_REFEREE_REJECT`
- `COUNTER_STRUCTURE`
- `OPPOSING_MULTI_FLOW`
- `RAW_CONVICTION_LOW`

Cost remains visible to Brian and is included in expected net outcome. It is not a second strategy engine.

Setup-specific confirmation is interpreted by Brian. A global three-close/retest rule must not turn an `EARLY_REVERSAL` into a late-entry strategy.

## Hard technical rails only

The following can still reject execution because they are execution/integrity constraints, not a competing trading opinion:

- SHADOW ONLY; `live_execution=false`; `browser_execution=false`
- stale / missing / internally inconsistent market data
- invalid geometry (e.g. LONG stop above entry)
- exchange quantity/minimum constraints
- open-position/accounting consistency
- duplicate execution of the same immutable thesis/episode
- lease / state-version / database commit integrity
- insufficient cash for the requested 1x SHADOW allocation

No real exchange order route is permitted.

## Capital

The dashboard restores editable starting SHADOW capital. Typical values such as 500 or 1000 USDT are user-selectable; previous sessions remain append-only history.

Brian chooses how much of that selected test capital to allocate per trade. V8.4.1 is 1x SHADOW; allocation cannot exceed available SHADOW cash. There is no fixed Package-1 8% strategy ceiling in Brian-Authority mode.

## Evaluation

Measure rather than assume:

- entry distance from subsequent local bottom for LONG (or local top for SHORT)
- exit distance from subsequent local top for LONG (or local bottom for SHORT)
- MFE / MAE
- net P&L after modeled costs
- target-before-invalidation
- missed opportunity where Brian WAITed
- false entry where Brian entered before continued adverse move

Preserve V8.4 Package-1 history as the rule-heavy baseline. V8.4.1 starts a new release/session/family before collecting authority-mode evidence.
