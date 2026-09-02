# Brian 2026 Phase 2.8 — Expert Market Narrative / Chart Reasoner

Status: **SHADOW_RESEARCH_ONLY**

Phase 2.8 does not claim human experience, guaranteed profitability, or live-trading readiness. The engineering target is to reproduce the *decision discipline* of an experienced discretionary trader using deterministic, point-in-time-safe market evidence and explicit abstention.

## Scientific boundary

- Development data: 2020-01-01 inclusive through 2026-01-01 exclusive.
- 2026 remains `INVALID_CONTAMINATED` and `evaluation_allowed=false`.
- No authenticated exchange API, credentials, `create_order`, live execution, self-modification, or automatic promotion.
- Historical order book/bid-ask is unavailable and is never fabricated.
- Phase 2.8 thresholds are fixed in code before development results are reviewed.
- Scenario strengths are ranking scores, **not calibrated probabilities**.
- Negative evidence is retained unchanged.

## Expert reasoning architecture

`expert_reasoner.py` combines six explicit roles:

1. **Structure expert** — confirmed BOS/CHOCH, liquidity sweeps, failed breaks, retests and support/resistance.
2. **Trend expert** — completed 5m/15m/1h structure plus EMA slope.
3. **Momentum expert** — dip/rally quality, RSI, acceleration and PIT-safe confirmed divergence.
4. **Volume expert** — relative-volume and exhaustion proxies only; no fake order flow.
5. **Mean-reversion expert** — z-score/Bollinger location conditioned on confirmed structure.
6. **Risk critic** — vetoes incomplete higher-timeframe context, severe conflicts, extreme expansion and entries crowded into a confirmed opposing level.

A deterministic **head trader** combines expert opinions. A trade action requires:

- an objective preregistered setup,
- sufficient directional edge,
- sufficient confidence,
- sufficient expert agreement,
- at least two supporting directional experts,
- and no active risk veto.

Otherwise Brian returns `WAIT`.

## Objective setup library

- `LIQUIDITY_SWEEP_REVERSAL`
- `FAILED_BREAK_REVERSAL`
- `BREAKOUT_RETEST_CONTINUATION`
- `PULLBACK_CONTINUATION`
- `RANGE_REJECTION`
- `TREND_EXHAUSTION`
- `NO_CLEAR_SETUP`

## Scenario reasoning

Every decision contains three explicit cases:

- `bull_case`
- `bear_case`
- `no_trade_case`

The narrative includes activation evidence, contradictions, the current regime, expert agreement and an objective invalidation level when one exists. The no-trade case is first-class evidence rather than a fallback error state.

## Locked ablations

The development run compares the full reasoner against preregistered ablations:

- no risk critic,
- single-timeframe only,
- no volume context,
- no divergence context.

These comparisons are diagnostic. They are not used to retune the locked test outcomes.

## Evaluation

The Phase 2.8 experiment reuses the locked Phase 2.5 folds/cost model and Phase 2.7 PIT-safe features. It reports:

- LOW / BASE / STRESS portfolio results,
- directional accuracy only on acted directional samples,
- coverage / WAIT rate,
- confidence and expert agreement,
- setup-level reports,
- regime reports,
- source-gap sensitivity with forced-flat boundaries,
- causal expanding walk-forward yearly robustness,
- deterministic narrative examples selected without outcome/PnL information,
- an evidence gate that can only produce `DEVELOPMENT_CANDIDATE` or `INSUFFICIENT_EVIDENCE`; never a final champion.

## Cloud execution

`phase28-development` is a fixed cloud mode. `.github/BRIAN_PHASE28_RUN_REQUEST` triggers only the fixed 2020–2025 shadow development run on `brian-2026`. Arbitrary dates and 2026 access are not exposed by that trigger.

The development artifact includes the Phase 2.7 reference experiment so Phase 2.8 cannot silently erase or rewrite prior negative evidence.
