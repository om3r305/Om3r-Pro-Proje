# Pre-registration — opening-range breakout (ORB), prospective shadow test

- Study id: `orb-shadow-v1` · rules `orb-shadow.v1`
- Rules fingerprint: `15a033b97d3499368cea11391431d973c9513aaaecd7f76b041240a105981fd7`
  (`orbRulesFingerprint()` in `supabase/functions/_shared/orb_shadow.ts`)
- Registered 2026-10-26 before the US open (13:30 UTC) of the first prospective session. Operator:
  "full authority, integrate if free" (2026-10-26).
- Idea source: the opening-range rules of the MIT-licensed `Jakeschincariol/day-trader-skill`
  (trader-entry / trader-exit). Re-implemented; no code copied. That repo's own worked example
  (TSLA, 2026-10-19) called its replay "close to a coin flip" — this test exists to find out.
- Shadow only: hypothetical trades on public Yahoo 5-minute bars. No order, account or key exists.

## Universe (frozen)

Equities AAPL AMZN GOOGL META MSFT NVDA TSLA CVX XOM JPM; ETFs SPY QQQ IWM SMH XLE XLF USO GLD TLT
EWG (the US-listed instruments Brian already tracks). Round-trip cost: equity 18 bps, ETF 12 bps (the
multiasset engine's own cost model).

## Rules (frozen, see the module header for the exact definitions)

- Opening range = 09:30–09:44:59 New York (three 5-minute bars).
- Trigger = first 5-minute close outside the range from the 09:45 bar to the 15:50 bar; entry at that
  close; stop = range midpoint; flatten at the 15:55 bar close.
- Variants: A `ORB_1R` (target 1R), B `ORB_CLOSE` (no target), C `ORB_IN_PLAY_1R` (A, only when the
  opening-range volume is ≥ 2× the median of the previous 10 sessions').
- Conservative fills: a bar touching stop and target exits at the stop; a gap through the stop exits
  at the bar open; targets are never filled better than the target.

## Decision (frozen)

- Only `period = 'PROSPECTIVE'` rows (sessions from 2026-10-26) count. RETRO rows (the month before)
  are shown for orientation only.
- Evaluated after 40 prospective sessions (about 2026-12-21) or 2027-01-31, whichever is first; at
  least 60 trades for a variant, else `INSUFFICIENT_DATA`.
- Statistic: mean net bps per trade; 95 % interval from 2,000 bootstrap resamples clustered by session
  day; Bonferroni across the three variants (98.33 % interval).
- PASS only if the interval's lower bound is > 0. A failed variant is not re-tuned on the same data.
- On PASS the next step is a proposal to the operator, not an automatic treasury change.
