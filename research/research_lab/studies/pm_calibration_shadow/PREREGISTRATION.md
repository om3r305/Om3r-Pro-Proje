# Pre-registration — Polymarket political calibration, prospective shadow test

- Study id: `pm-calibration-shadow-v1` · rules `pm-calibration-shadow.v1`
- Rules fingerprint: `c95ec8535fa741bdde181d87cd88dd99e0a7395cccc1abd184208fb4e5fc81d1`
  (`pmCalibrationRulesFingerprint()` in `supabase/functions/_shared/pm_calibration_shadow.ts`)
- Registered 2026-10-21, before any entry was captured. Operator approval: 2026-10-21 ("4 de başlat").
- Shadow only: hypothetical entries priced at the observed best ask. No order, wallet or key exists.

## Question

Run-4 (`pm-political-slope-oos-v1:run-4-weekly`, PASS) found political markets underconfident out of
sample: outcome ≈ sigmoid(0.1356 + 1.1625 · logit(price 30 h before resolution)), slope interval
[1.114, 1.216]. **Does buying the side that this frozen map calls cheap make money after paying the
ask and the fee, on markets that had not resolved when the map was fixed?**

## Data (collected prospectively by `brian-pm-calibration-shadow` on CORE, hourly)

- Universe: Gamma markets, `closed=false`, end date in [now + 29 h, now + 31 h]; the paper's
  Politics regex (generated TS port, parity 16,078/16,078 titles with the Python original);
  outcomes exactly `["Yes","No"]`; both CLOB token ids present.
- One entry per market: the first capture inside the window (unique `market_id`).
- Books: CLOB `/book` for the YES and NO tokens at capture time. Fee rate from the market's own
  `feeSchedule` (`feesEnabled=false` → 0); unknown fee → no entry.
- Outcome: Gamma `/markets/{id}` once closed, counted only if `closed=true`, UMA `resolved`, final
  prices exactly one 1 and one 0 (`normalizeGammaResolution`). Anything else stays unresolved.

## Entry rule (frozen)

- mid = (YES best bid + YES best ask) / 2; requires both sides and YES spread ≤ 0.10.
- q = sigmoid(0.1356427546398957 + 1.1624733790827435 · logit(clip(mid, 0.01, 0.99))).
- A side is usable if its best ask ∈ [0.02, 0.98] with ≥ 5 shares at that price.
- edge_YES = q − ask_YES − fee(ask_YES); edge_NO = (1 − q) − ask_NO − fee(ask_NO),
  fee(p) = feeRate · p · (1 − p) per share.
- Take the larger edge if it is ≥ 0.02; otherwise record SKIP with the reason.
- Return per $1 = payout / (ask + fee) − 1, payout 1 if the bought side won, else 0.

## Decision (frozen)

- Evaluated when ≥ 300 resolved trades exist, or on 2027-01-31, whichever is first.
- Statistic: mean return per $1 across resolved trades; 95 % interval from 2,000 event-clustered
  bootstrap resamples (seed 20261021; events = Gamma event id).
- **PASS** if the lower bound > 0 · **FAIL** if the upper bound < 0 · **UNVERIFIED** otherwise ·
  **INSUFFICIENT_DATA** if < 100 resolved trades at the deadline.
- Also reported (not decisive): calibration slope on all captured markets (traded or not), share of
  skips by reason, mean edge vs realised return.

## Known differences from run-4 (accepted before data)

- Run-4 priced at the last trade ≤ 30 h before min(end, closed); here the live mid 29–31 h before
  the *scheduled* end. Markets that close early are still captured at their scheduled time.
- Run-4 fitted on prices; trading pays the ask. The ask is used for every hypothetical fill, with
  no size beyond the best level (≥ 5 shares required there).
- A PASS is evidence for a shadow edge only. Any real-money use needs a separate operator decision.
