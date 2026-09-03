# Brian Phase 3.1 — Edge Attribution & Capital Efficiency

Status: SHADOW_RESEARCH_ONLY

Final holdout: INVALID_CONTAMINATED. 2026 must not be accessed or evaluated.

## Why this phase exists

Phase 2.8 produced limited coverage and insufficient evidence. Phase 2.9 produced zero locked-fold coverage because the hard league consensus never cleared its fixed edge/confidence/agreement gates. OOD and hard-drift rates were very low, so Phase 3.1 treats league aggregation/opportunity selection as the development question rather than weakening OOD safety.

This is explicitly **POST_DIAGNOSTIC_DEVELOPMENT_ONLY; NOT PRISTINE OOS**. The 2020–2025 development evidence has already influenced architecture choices and cannot be described as a pristine final holdout.

## Preregistered design

- Same fixed 2020-01-01 through 2026-01-01 exclusive public Binance Spot BTCUSDT development range.
- Same three locked development folds.
- Validation is split chronologically into calibration-validation and policy-validation with a 12-bar embargo.
- Model fit uses train only.
- Probability calibration, challenger reliability weights, and drift threshold use calibration-validation only.
- Opportunity threshold and capital fraction use policy-validation only.
- Test outcomes never select thresholds, weights, score quantiles, or capital fractions.
- Individual Expert Reasoner, Logistic, and Gradient Boosting challengers are reported separately for edge attribution.
- Hard action agreement is replaced by a soft evidence score. Disagreement reduces score continuously; it is not an automatic veto.
- OOD and hard concept drift remain hard vetoes.
- Opportunity quantile grid is fixed at 0.75, 0.85, 0.90, 0.95, 0.975.
- Capital benchmark is fixed at $500. This is a research benchmark, not a $30/day target.
- Candidate capital fractions are fixed at 5%, 10%, 15%, 20%, 25% of current equity with no leverage.
- Capital remains at 0% unless policy-validation has positive net PnL, at least 20 entries, PF >= 1.05, max drawdown <= 10%, and bootstrap risk-of-ruin <= 5%.
- Risk-of-ruin uses deterministic bootstrap sampling with fixed seeds and a 30% peak-to-trough ruin definition over 200 trades.
- Position sizing is forbidden from creating a candidate when the underlying signal policy lacks edge.
- Transaction fees, simulated spread, slippage, hard stop, take profit, max hold, cooldown, and single-position lifecycle remain active.
- No historical order book is fabricated.
- No exchange credentials, authenticated endpoints, live execution, automatic promotion, or self-modifying code.

## Interpretation

A positive Phase 3.1 result is development evidence only. It does not prove profitability. A negative result is retained as evidence and must not be rescued by test-driven threshold changes.
