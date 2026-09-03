# Brian 2026 Phase 3.3 — Market Gym & World Model

Status: **PREREGISTERED FOUNDATION / SHADOW_RESEARCH_ONLY**

## Goal

Give Brian repeatable, causal training experience without pretending that replayed or synthetic worlds are evidence of future profitability.

The target is not “replay one BTC chart 100,000 times.” That would mostly create memorization. The target is a curriculum across assets, regimes, synchronized historical blocks and adversarial stress worlds.

## Scientific boundary

- Synthetic/replayed episodes are `TRAINING_ONLY`.
- Scenario strength is not a forecast probability.
- A profitable synthetic episode is not evidence of live profitability.
- No synthetic result may promote a model or become pristine final-holdout evidence.
- The contaminated 2026 historical holdout remains `INVALID_CONTAMINATED` and is forbidden as development history.
- Final evidence must eventually come from genuinely unseen real point-in-time data.
- No authenticated exchange execution exists in this phase.

## Market Gym

The gym is a multi-asset, unlevered virtual portfolio.

Default account:

- starting equity: `$500`
- max gross exposure: `100%`
- max single-asset weight: `35%`
- leverage: none
- explicit fee, spread and slippage costs
- long and short positions are sandbox-only signed target weights
- ruin threshold: `1%` of starting equity

A decision made after observing frame `t` is applied no earlier than frame `t+1` open. The environment then marks the new allocation from `t+1` open to `t+1` close. Existing exposure also experiences the real close-to-next-open gap before the rebalance.

If a held/target asset is unavailable in the next frame, the episode terminates with `DATA_GAP`. The gym never forward-fills a closed/missing market in order to fabricate a tradable price.

After ruin, a new episode begins only through an explicit `reset()`, restoring exactly `$500` and clearing portfolio state. The failed episode remains in experience memory.

## World modes

### 1. REAL_REPLAY

A causal window from real historical development data. Source timestamps must be before the 2026 development cutoff.

### 2. BLOCK_BOOTSTRAP

Synchronized multi-asset historical transitions are sampled in blocks. Every asset uses the same source block, preserving observed cross-asset co-movement inside the block.

The generator reconstructs a new price path from source gap/intrabar OHLC ratios instead of concatenating raw price levels. This avoids artificial block-boundary price jumps caused merely by different nominal price levels.

### 3. STRESS_BOOTSTRAP

Uses the same synchronized source-block recipe but expands return/OHLC geometry with a preregistered stress multiplier. It is an adversarial training world, not a forecast of what will occur.

Future increments may add regime-balanced and event-conditioned worlds, but they must be specified before their evaluation outcomes are reviewed.

## Multi-asset scope

Phase 3.3 is provider-neutral. Each asset must declare:

- canonical id
- symbol
- asset class
- venue
- quote currency
- timezone
- trading calendar
- native frequency
- source/provenance
- whether it is a proxy and, if so, what it proxies

Intended curriculum families include:

- crypto spot: BTC, ETH and other liquid majors/alts
- fan tokens
- equities and broad indices/ETFs
- gold, oil and other commodity series or explicitly disclosed proxies
- FX
- rates
- macro regime series

Different clocks must not be silently mixed. Daily macro/equity observations will not be fabricated into 5-minute data. Cross-market episodes use explicit alignment/intersection rules or later point-in-time as-of joins designed for the relevant frequency.

## Experience without data bloat

The default memory stores one compact summary per episode:

- world id / mode / generator seed
- source dataset id
- policy version
- starting and ending equity
- return and max drawdown
- turnover and costs
- ruin / terminal reason
- lesson tags

Full step traces are bounded and retained only for:

- ruin episodes
- severe drawdowns
- a small deterministic audit sample

This allows tens or hundreds of thousands of episodes without persisting every transition.

## Long-horizon “foresight”

Brian must not claim to know a single deterministic market state 5–10 years ahead. Long-horizon intelligence will be expressed as distributions and scenario families:

- regime transition paths
- inflation/rates/growth combinations
- liquidity expansion/contraction
- correlation breaks
- volatility/tail shocks
- asset-class relative-strength scenarios
- event-conditioned branches

The useful output is: what future states are plausible, what evidence would move their weight, what portfolio theses survive them, and what invalidates those theses.

## Curriculum roadmap

1. exact real historical replay
2. regime-diverse historical replay
3. synchronized block-bootstrap worlds
4. stress/adversarial worlds
5. event/social/on-chain conditioned worlds
6. prospective worlds built from Brian’s own Supabase PIT memory
7. only then compare learned policies on genuinely unseen real data

Repeating the same dataset does not manufacture new scientific evidence.

## First implementation gates

- next-open/next-frame execution invariant
- no leverage by default
- exact `$500` reset after a failed life
- no fake forward-fill across missing markets
- deterministic world generation by seed + episode index
- synchronized source blocks across assets
- 2026 development-history rejection
- compact bounded trace retention
- synthetic/replay experience explicitly forbidden from final evidence
- no broker/exchange execution surface
