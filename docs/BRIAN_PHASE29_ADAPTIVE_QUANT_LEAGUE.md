# Brian 2026 — Phase 2.9 Adaptive Quant League

Phase 2.9 is a preregistered **development-only, shadow-only** research phase. It combines useful design ideas seen in mature quant systems without copying external strategy code or weakening Brian's scientific gates.

## Scope

- **Market Familiarity / OOD Gate**: robust train-only reference distribution. A market state sufficiently far from the train distribution is treated as unknown and the primary policy must WAIT.
- **Causal Concept Drift Monitor**: train reference, rolling past-only feature window, alert threshold locked on the preceding validation partition. Test outcomes are never used to set the drift threshold.
- **Model Zoo / Challenger League**:
  - Phase 2.8 Expert Reasoner
  - calibrated Logistic Regression challenger
  - calibrated Gradient Boosting challenger
- **Validation-only tournament weights**: challengers fit on train, calibrate and receive weights only from validation, then remain locked throughout test evaluation.
- **Head Trader**: combines challenger edge/confidence with validation weights, familiarity, concept drift and agreement. OOD and hard drift can veto a directional action.
- **Ablations**: full league, no OOD gate, no drift gate. Ablations are diagnostic only and cannot replace the preregistered primary policy after test evidence is observed.

## Scientific protocol

- Dataset: Binance Spot BTCUSDT public/verified history only.
- Development range: `2020-01-01` through `2026-01-01` exclusive.
- Existing locked Phase 2.5 folds are reused.
- 30-minute target horizon remains partition-contained.
- Train-only preprocessing/model fitting.
- Validation-only calibration, drift threshold and challenger weights.
- Locked test evaluation under LOW / BASE / STRESS simulated costs.
- Source-gap sensitivity refits on clean development observations.
- Expanding yearly robustness uses only train/validation observations before each test year.
- Missing/unavailable features remain unavailable; they are not silently coerced to zero.
- No historical spot order book or bid/ask is fabricated.
- Logical experiment identity excludes runtime timestamp so scientific identity is reproducible for the same dataset/code/preregistration.

## Hard restrictions

- `2026` remains `INVALID_CONTAMINATED` and `evaluation_allowed=false`.
- `NO PRISTINE FINAL HOLDOUT EVALUATED` remains mandatory.
- `SHADOW_RESEARCH_ONLY` remains mandatory.
- No authenticated exchange API, credentials, real order placement or live promotion.
- No automatic champion promotion and no self-modifying trading code.
- A development candidate is evidence for further research only; it is not proof of profitability.

## Primary purpose

The goal is not to maximize the number of trades or force a positive backtest. The goal is to test whether Brian improves by **knowing when the market is unfamiliar, recognizing distribution shift, and combining independent challengers only with validation-supported trust**.
