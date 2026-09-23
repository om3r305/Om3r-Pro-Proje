# Brian upstream provenance

This document records external open-source behaviors studied for Brian. It is not a claim that upstream projects prove trading profitability.

## Phase 41 — Robustness Lab

- Brian file: `brian2026/phase41_robustness_lab.py`
- Behavior adopted: trade-order Monte Carlo, path robustness through resampling, and entry-rule significance testing before a research candidate can advance.
- Reference project: Jesse (`jesse-ai/jesse`, MIT).
- Reference revision inspected: `019c7092812dc2cc6bcb8a2f1300b7ac800f83fc`.
- Upstream evidence inspected: Jesse Research API/README exposes Monte Carlo and Rule Significance Testing as first-class research operations.
- Brian implementation: independent implementation. No Jesse source code is copied.
- Brian-specific safety: outputs remain `shadow_only=True`, `live_execution=False`, and `automatic_promotion=False`.

## Phase 42 — Autonomous Alpha Lab

- Brian file: `brian2026/phase42_autonomous_alpha_lab.py`
- Behavior adopted: hypothesis -> experiment specification/implementation -> run -> feedback -> trace.
- Reference project: Microsoft RD-Agent (`microsoft/RD-Agent`, MIT).
- Reference revision inspected: `4834df2b6f3e417e5d0512dfe4fd24a8ac338b05`.
- Upstream evidence inspected:
  - `rdagent/app/qlib_rd_loop/quant.py`: proposal, factor/model implementation, runner, feedback.
  - `rdagent/components/workflow/rd_loop.py`: propose -> coding -> running -> feedback -> record lifecycle.
- Quant evaluation reference: Microsoft Qlib (`microsoft/qlib`, MIT), revision `be725493eb1a6bbb42bf11b37aa7669f59610ff1`.
- Brian implementation:
  - built-in generator may inspect train-partition statistics only;
  - factor compilation is deterministic and content-addressed;
  - baseline and candidate use identical locked walk-forward folds;
  - preprocessing/model fit remains train-only;
  - calibration and policy thresholds remain validation-only;
  - locked test is consumed independently for baseline and candidate;
  - a passing experiment becomes only `RESEARCH_CHALLENGER_CANDIDATE`.
- No candidate may modify Brian configuration, place orders, or promote itself.

## License boundary

- MIT/Apache-2.0 projects may be studied or adapted subject to their license/notice requirements.
- GPL projects such as Freqtrade are treated as behavioral references only unless a future legal/license review explicitly approves code reuse.
- A feature name is never treated as sufficient provenance: the concrete upstream behavior and Brian acceptance tests must be recorded.


## Phase 43 — Evidence-Grounded Analysts + Regime-Aware Specialist Router

- Brian files:
  - `brian2026/phase43_grounded_analysts.py`
  - additive `selected_experts` routing support in `brian2026/expert_reasoner.py`
- Reference project: TradingAgents (`TauricResearch/TradingAgents`, Apache-2.0).
- Reference revision inspected: `2d17df8da1536c121e4d7395ac5a5dcec9e96d6f` (v0.5.0 merge).
- Upstream behavior inspected:
  - `tradingagents/agents/analysts/sentiment_analyst.py`: pre-fetch news + StockTwits + Reddit before invoking the model; inject structured source blocks; disable open-ended tool use after prefetch; produce structured output; explicitly flag unavailable/small samples.
  - `tradingagents/agents/analysts/market_analyst.py`: select at most 8 complementary indicators, avoid redundancy, fetch market data before indicators, and verify exact market/indicator claims against a verified snapshot.
  - `tradingagents/agents/managers/research_manager.py` and `portfolio_manager.py`: synthesize conflicting analyst evidence but do not treat conflict alone as an action; thin evidence may remain Hold/WAIT.
- Brian adaptation:
  - freezes point-in-time `SensorObservation` inputs into a structured packet before analyst compilation;
  - rejects future observations and marks stale/unavailable sources explicitly;
  - forbids GDELT discovery-only rows from becoming directional source truth;
  - every directional analyst claim must reference known, fresh, directional evidence IDs;
  - correlated rows from one independent group count once;
  - source-level typed claims are compiled deterministically from prefetched evidence;
  - regime routing selects no more than 8 complementary feature inputs and a bounded expert subset;
  - existing `reason_market` behavior is unchanged unless `selected_experts` is explicitly supplied;
  - Phase 43 remains shadow-only and cannot self-promote or execute.
- Brian deliberately does not copy TradingAgents' prompts or LangChain orchestration. The behavior is reimplemented against Brian's existing evidence, sensor, and expert contracts.


## Phase 44 — Conviction Portfolio Brain + Hard Risk Clamps

- Brian file: `brian2026/phase44_portfolio_brain.py`
- Reference project: ai-hedge-fund (`virattt/ai-hedge-fund`).
- Reference revision inspected: `7d897a002c263f106201154d877a4bcf74efae03`.
- Upstream behavior inspected:
  - `hedge_fund/portfolio/construction.py`: per-asset conviction is a model-weighted mean; abstained signals are excluded from both numerator and denominator; an explicit non-abstained zero is a real neutral vote; optional market-neutral mode cross-sectionally demeans convictions; nonzero books normalize to a requested gross target.
  - `hedge_fund/risk/limits.py`: hard deterministic risk stage applies per-position caps first, then a proportional portfolio gross cap; risk may only shrink positions and never redistributes removed exposure.
  - Upstream tests explicitly verify weighted means, abstention semantics, market-neutral demeaning, per-position clamps, gross clamps, shorts, idempotence, and that released capital remains cash.
- Brian adaptation:
  - Phase 43 grounded analyst claims translate into signed conviction signals while preserving evidence lineage;
  - portfolio blending is pure deterministic arithmetic;
  - hard limits are structurally independent of analyst/model requests;
  - risk clamps cannot increase any requested absolute position;
  - capital removed by a clamp remains unallocated cash;
  - output is a shadow-only book plan and is not wired to the production Treasury execution path.
- This phase copies the documented arithmetic behavior, not branding or persona agents.


## Phase 45 — Execution Contract + Shadow Position Accounting

- Brian file: `brian2026/phase45_execution_contract.py`
- Reference project: Hummingbot (`hummingbot/hummingbot`, Apache-2.0).
- Reference revision inspected: `9af100d6822da7d2d0291a906c730ef172284ee2` (master v2.17 sync).
- Upstream behavior inspected:
  - `strategy_v2/models/executor_actions.py`: controllers emit explicit Create / Stop / Store executor actions rather than placing exchange orders directly.
  - `strategy_v2/executors/executor_orchestrator.py`: orchestrator owns executor lifecycle and position-hold accounting independently of controller logic.
  - `strategy_v2/executors/position_executor/data_types.py`: position executor config validates side, amount, leverage, entry-price and triple-barrier constraints; stop-loss/time-limit exits must be market-capable.
  - `strategy_v2/models/base.py`: executor lifecycle is NOT_STARTED -> RUNNING -> SHUTTING_DOWN -> TERMINATED.
  - Hummingbot PositionHold incrementally realizes PnL when exposure is reduced, preserves average entry on partial reduction, and resets/changes average entry when a fill flips the net position.
- Brian adaptation:
  - intelligence emits immutable, time-bounded `TradeIntent` objects with evidence lineage;
  - execution config is a separate object and Phase 45 hard-forces connector_name=`shadow`;
  - explicit Create / Stop / Store actions drive a separate shadow executor lifecycle;
  - incremental fills support weighted-average entry, partial reductions, realized PnL, position flips, cumulative fees/volume, unrealized PnL, and duplicate-fill idempotency;
  - Phase 44 portfolio weights can be translated into evidence-lineaged TradeIntents;
  - there is deliberately no exchange transport, API key surface, or live order function in Phase 45.
- A later adapter may map this contract to Hummingbot itself; Phase 45 does not reimplement Hummingbot connectors.


## Phase 46 — Tiered Execution Simulator + Latency/Fill Models

- Brian file: `brian2026/phase46_execution_simulator.py`
- Reference project: NautilusTrader (`nautechsystems/nautilus_trader`, LGPL-3.0).
- Reference revision inspected: `fb2b45e330853d1cc598cf1961aab05fc5ccb287`.
- Upstream behavior inspected:
  - `crates/execution/src/models/fill.rs`: seeded probabilistic limit-fill decisions, seeded slippage decisions, best-price / one-tick / tiered-liquidity fill models, and size-aware synthetic depth.
  - `crates/execution/src/models/latency.rs`: base latency is added to operation-specific insert/update/delete latency.
  - `crates/execution/src/matching_engine/inflight.rs`: submitted client-order ids stay inflight until venue receipt; the first receipt releases duplicate submits idempotently.
  - `crates/execution/src/matching_engine/config.rs`: matching behavior explicitly separates execution, liquidity consumption, queue-position and acknowledgement concerns.
- Brian clean-room implementation:
  - no Nautilus source code is copied;
  - static latency keeps separate base + insert/update/delete components;
  - probabilistic limit-fill and one-tick adverse slippage are seeded and reproducible;
  - tiered bid/ask levels are walked in price priority and may produce partial fills;
  - execution chooses the first point-in-time book snapshot available after simulated venue arrival;
  - intent-level maximum slippage can veto a projected fill before it is applied;
  - inflight receipt state is idempotent under duplicate submits;
  - all receipts remain shadow-only with no exchange transport.
- License boundary: NautilusTrader is used strictly as a behavioral reference for this phase; implementation is independent.


## Phase 49 — Shadow/Paper Parity + Micro-Live Eligibility Gate

- Brian file: `brian2026/phase49_promotion_gate.py`
- Reference project: NautilusTrader (`nautechsystems/nautilus_trader`, used as a behavioral reference only).
- Reference revision inspected: `2c5364a5ca3ea2a68f51aa6e886aa6a7d6fc58e4`.
- Upstream behavior inspected:
  - `docs/concepts/architecture.md`: Backtest, Sandbox and Live share a common kernel/core; Sandbox uses real-time data with simulated execution, while Live uses live venue connections.
  - `docs/concepts/execution/index.md`: execution outcomes distinguish local failures, definitive venue results and unknown live outcomes; unknown outcomes remain in flight until stream/poll/query/reconciliation resolves them.
  - live execution state is reconciled from order, fill, position and account reports instead of assuming local state is authoritative.
- Brian adaptation:
  - promotion evaluates real-time shadow versus paper/sandbox observations by intent id;
  - direction parity, acknowledgement rate, reconciliation completeness, fill-fraction parity and execution-price drift are separate gates;
  - unresolved ambiguous outcomes fail the gate instead of being treated as fills or losses;
  - Phase 42 research, Phase 41 robustness and Phase 48 causality must all pass before paper parity can matter;
  - passing every gate yields only `MICRO_LIVE_ELIGIBLE`;
  - `MICRO_LIVE_ELIGIBLE` does not enable an exchange adapter, allocate capital, activate automatically or submit an order;
  - explicit later authorization remains mandatory.
- No Nautilus source code is copied in Phase 49.


## Phase 50 — Execution Reconciliation

- Brian file: `brian2026/phase50_execution_reconciliation.py`
- Reference project: NautilusTrader (`nautechsystems/nautilus_trader`, behavioral reference only).
- Reference revision inspected: `2c5364a5ca3ea2a68f51aa6e886aa6a7d6fc58e4`.
- Upstream behavior inspected:
  - `docs/concepts/execution/reconciliation.md`: explicit position reports are authoritative; missing reports are not evidence of flat; unresolved explicit reports fail closed.
  - startup reconciliation applies orders/fills before position validation and requires in-scope venue positions to match local state within quantity tolerance.
  - open positions without a usable reported entry average cannot be safely reconstructed from an empty cache.
  - matching quantity does not excuse a reported entry-average mismatch.
  - direction reversals are recovered by closing cached exposure then opening the authoritative side.
  - unknown command outcomes remain unresolved until stream/poll/query/reconciliation provides evidence.
  - bounded/incomplete historical report sets may still be usable when an explicit authoritative position report resolves the tracked exposure.
- Brian adaptation:
  - local and venue position identities are compared explicitly by account + asset;
  - a missing venue report yields `NO_AUTHORITATIVE_POSITION_REPORT`, never synthetic flat;
  - safe quantity mismatches produce auditable synthetic recovery proposals but remain `RECOVERY_REQUIRED` until a subsequent reconciliation confirms the new state;
  - synthetic recovery never invents historical realized PnL;
  - duplicate fill ids and unresolved command outcomes block readiness;
  - batch readiness is fail-closed and remains shadow-only.
- No Nautilus source code is copied in Phase 50.


## Phase 51 — PnL / Decision Attribution Ledger

- Brian file: `brian2026/phase51_pnl_attribution.py`
- Reference projects:
  - Hummingbot (`hummingbot/hummingbot`, Apache-2.0), revision `9af100d6822da7d2d0291a906c730ef172284ee2`.
  - NautilusTrader execution/portfolio event model as a behavioral reference.
- Upstream behavior inspected:
  - Hummingbot `PositionHold` and `PerformanceReport` keep realized PnL, unrealized PnL, cumulative fees, traded quote volume and close-type counts as explicit accounting fields.
  - Nautilus portfolio state is derived from execution/order/position events rather than from narrative analyst output.
- Brian adaptation:
  - every closed trade retains the originating `TradeIntent` and its evidence lineage;
  - gross PnL is decomposed exactly into market move from the decision reference plus entry-execution effect;
  - fees remain a separate explicit deduction and the arithmetic must reconcile exactly;
  - portfolio-risk clamps are recorded as weight released to cash, not invented as realized PnL;
  - regime, analyst and evidence views are association summaries only;
  - evidence-linked PnL is deliberately non-additive and never presented as a causal dollar split among signals.
- Phase 51 is shadow accounting only and introduces no execution transport.


## Phase 52 — Shrunk Covariance Portfolio Risk Overlay

- Brian file: `brian2026/phase52_covariance_risk.py`
- Reference project: Microsoft Qlib (`microsoft/qlib`, MIT).
- Reference revision inspected: `be725493eb1a6bbb42bf11b37aa7669f59610ff1`.
- Upstream behavior inspected:
  - `qlib/model/riskmodel/base.py`: risk models estimate covariance from aligned observations after centering, with explicit missing-data behavior.
  - `qlib/model/riskmodel/shrink.py`: sample covariance may be shrunk toward a constant-variance target; Qlib supports Ledoit-Wolf shrinkage using the documented `phi/gamma/T` estimator.
  - `qlib/contrib/strategy/optimizer/optimizer.py`: portfolio risk is measured through the covariance matrix using `w' S w`; inverse-volatility, minimum-variance, mean-variance and risk-parity allocations are built on that matrix.
- Brian clean-room adaptation:
  - aligned per-asset return histories are centered and converted to sample covariance;
  - the constant-variance shrink target and Ledoit-Wolf shrink parameter are independently implemented from the public estimator equations;
  - Brian keeps Phase 44 conviction signs/relative requests intact;
  - covariance risk is an overlay only: if predicted period volatility exceeds the preregistered limit, all weights are scaled down proportionally;
  - the overlay can never increase a position, flip direction or redistribute released risk into another asset;
  - marginal and normalized risk contributions plus pairwise correlation are emitted for audit;
  - released gross exposure remains cash.
- Phase 52 is shadow-only and does not alter the production Treasury path.


## Phase 53 — Turnover-Constrained Rebalance Planner

- Brian file: `brian2026/phase53_turnover_rebalance.py`
- Reference project: Microsoft Qlib (`microsoft/qlib`, MIT).
- Reference revision inspected: `be725493eb1a6bbb42bf11b37aa7669f59610ff1`.
- Upstream behavior inspected:
  - `qlib/contrib/strategy/optimizer/optimizer.py` constrains portfolio turnover with an L1 budget of the form `|w - w0| <= delta`.
  - Qlib's optimizer treats turnover as a first-class portfolio constraint rather than letting every new target immediately churn the full book.
- Brian clean-room adaptation:
  - current-to-target absolute weight change is measured as L1 turnover;
  - if ordinary additions/rotations exceed the budget, the remaining trade vector is scaled proportionally toward the approved target;
  - no planned weight may overshoot its target path;
  - hard exposure reductions are applied first and, by default, cannot be blocked by a transaction-cost/turnover budget;
  - an opposite-side reversal must close existing exposure before any new opposite exposure is opened;
  - if mandatory risk reduction alone exceeds the turnover limit, the receipt records that the limit was exceeded only for risk reduction;
  - output remains a shadow rebalance plan with no order transport.
- The hard-risk bypass is a Brian safety extension; it is not claimed to be Qlib's optimizer behavior.


## Phase 54 — Integrated Grounded Shadow Decision Pipeline

- Brian file: `brian2026/phase54_integrated_shadow_decision.py`
- This phase introduces no new upstream algorithm. It composes the already provenance-tracked contracts from:
  - Phase 43 / TradingAgents: point-in-time grounded evidence and regime-aware specialist routing.
  - Phase 44 / ai-hedge-fund: conviction blending and independent hard risk clamps.
  - Phase 52 / Qlib: shrunk covariance portfolio-risk overlay.
  - Phase 53 / Qlib: L1 turnover-constrained rebalance planning.
- Integration behavior:
  - each asset is analyzed through the real Phase 43 implementation, not a duplicate simplified analyst path;
  - grounded claims are converted through the real Phase 44 signal contract;
  - the resulting portfolio book is passed unchanged into the real Phase 52 covariance overlay;
  - covariance-approved target weights are passed into the real Phase 53 turnover planner;
  - deterministic pipeline identity includes the actual intermediate receipts;
  - missing grounded directional evidence fails to `WAIT_NO_GROUNDED_SIGNALS` and preserves the current book rather than liquidating because a source was unavailable;
  - an asset intentionally absent from an otherwise valid new target is reduced through the rebalance stage;
  - no execution/order transport is exposed in Phase 54.
- Phase 54 exists specifically to prevent "feature islands": it verifies that the proven mechanisms operate through one typed end-to-end shadow decision path.


## Phase 55 — Delta-Safe Rebalance Execution Intents

- Brian file: `brian2026/phase55_rebalance_execution_intents.py`
- Reference project: NautilusTrader (`nautechsystems/nautilus_trader`, behavioral reference only).
- Reference revision inspected: `2c5364a5ca3ea2a68f51aa6e886aa6a7d6fc58e4`.
- Upstream behavior inspected:
  - `crates/risk/src/engine/mod.rs::is_reducing_submission`: a reducing submission must be reduce-only, positive-sized, tied to the same instrument and identified open position, have the opposite order side, and must not exceed the open position quantity.
  - Nautilus `TradingState::Reducing` permits only valid exposure-reducing submissions; ordinary new-risk modifications/submissions are blocked by the risk engine.
- Brian clean-room adaptation:
  - execution is compiled from Phase 53 `planned_delta`, never from final target weight; this prevents over-ordering an already-open position;
  - same-side exposure reductions compile to explicit reduce-only intents and do not require alpha evidence merely to lower risk;
  - new/increased risk compiles to Phase 45 `TradeIntent` and requires grounded evidence, expected edge and confidence;
  - a direction reversal is always split into a reduce-only close-to-flat leg plus a pending opposite-side open;
  - the opposite-side leg cannot auto-release and requires an authoritative, reconciliation-complete flat-position receipt;
  - the pending reversal expires by TTL instead of opening stale new risk;
  - all Phase 55 instructions remain shadow-only with no exchange transport.
- The implementation operates at portfolio-weight/delta level because the real quantity conversion remains the execution adapter's responsibility.


## Phase 56 — Fail-Closed Pre-Trade Risk Engine

- Brian file: `brian2026/phase56_pretrade_risk_engine.py`
- Reference project: NautilusTrader (`nautechsystems/nautilus_trader`, behavioral reference only).
- Reference revision inspected: `2c5364a5ca3ea2a68f51aa6e886aa6a7d6fc58e4`.
- Upstream behavior inspected:
  - Nautilus `RiskEngine` sits between strategy intent and execution and independently validates order/instrument/account constraints.
  - `TradingState::Halted` denies new submissions.
  - `TradingState::Reducing` accepts only a valid reduce-only submission tied to an identified open position; order side must oppose the position and quantity must not exceed open quantity.
  - ACTIVE submissions still pass quantity/notional/account checks, including instrument min/max notional and configured `max_notional_per_order`.
- Brian clean-room adaptation:
  - Phase 56 has no bypass configuration;
  - new/increasing-risk `TradeIntent` requires ACTIVE state, available cash and all notional gates;
  - Phase 55 `RiskReductionIntent` may pass in ACTIVE or REDUCING, but must identify the current open direction, oppose it, remain within exposure, and never flip the position;
  - HALTED denies both new-risk and reduce submissions in this simplified submission-only boundary;
  - every review emits an auditable ALLOW/DENY receipt with individual checks and reasons;
  - risk receipts remain shadow-only and do not route orders themselves.
- Cancel/query behavior and venue-specific whole-position-exit exemptions are outside Phase 56 scope; they are not silently approximated.


## Phase 57 — Integrated Shadow Execution Cycle

- Brian file: `brian2026/phase57_shadow_execution_cycle.py`
- This phase introduces no new upstream algorithm; it composes the already provenance-tracked execution contracts:
  - Phase 55 delta-safe/reduce-only execution intent compilation;
  - Phase 56 independent pre-trade risk review;
  - Phase 45 executor-action contract;
  - Phase 46 latency/order-book/fill simulation.
- Integration behavior:
  - each Phase 55 instruction is independently reviewed by the real Phase 56 RiskEngine contract before simulation;
  - ALLOWed new-risk legs reserve their full requested cash inside the cycle before fill simulation, preventing later same-cycle orders from spending the same dollars;
  - simulated reduce-only fills never create spendable cash for another order because simulation is not authoritative account reconciliation;
  - new-risk orders use the real Phase 45 `CreateExecutorAction` and Phase 46 `simulate_create_action`;
  - reduce-only instructions use Phase 46 market-book simulation directly with the opposite position side;
  - reversal cycles execute only the reduce-only close leg; the opposite-side open remains pending behind the Phase 55 authoritative-flat barrier;
  - HALTED/REDUCING behavior is inherited from Phase 56 rather than recreated in the integration layer;
  - account state is explicitly marked unmutated because simulated fills are evidence, not venue reconciliation.
- Phase 57 remains fully shadow-only and cannot contact an exchange.


## Phase 58 — CPCV / PBO / Deflated Sharpe Advanced Overfit Audit

- Brian file: `brian2026/phase58_advanced_overfit_audit.py`
- Behavioral references:
  - Hudson & Thames mlfinlab CPCV/PurgedKFold public implementation lineage, inspected through historical fork `forensiclab/mlfinlab` revision `c87e19c59ad7169550d301cbddce7b89d3928d74`. License metadata is not sufficiently clear for reuse, therefore **behavior/formulas only; no source copied**.
  - `tulinette/backtest-overfitting-lab` revision `1a4440f4e28569327daf57909914a6ded2843270` (MIT) for an independently implemented CSCV-PBO and Deflated Sharpe reference plus published-paper citations.
- Upstream behavior inspected:
  - CPCV divides chronological samples into contiguous blocks, evaluates combinations of test blocks, purges train observations whose information intervals overlap test-label intervals, and applies a forward embargo. CPCV(6,2) yields 15 combinations and 5 backtest paths.
  - CSCV-PBO selects the best in-sample Sharpe strategy for each symmetric split, ranks that same strategy out-of-sample, converts the relative OOS rank to a logit, and defines PBO as the fraction at/below the OOS median.
  - Deflated Sharpe uses the Bailey/López de Prado expected-maximum-Sharpe benchmark across tried variants, then evaluates the selected strategy with the non-normality-adjusted probabilistic Sharpe ratio.
- Brian clean-room adaptation:
  - CPCV accepts explicit sample information intervals and purges any overlap before training; embargo is explicit and auditable per split;
  - PBO accepts a time x strategy-variant return matrix, supports deterministic split subsampling, and emits IS/OOS winner diagnostics;
  - DSR records best variant, naive PSR, selection-bias benchmark, skew, raw kurtosis and deflated probability;
  - an advanced robustness policy can require minimum CPCV paths, maximum PBO and minimum DSR probability;
  - a passing result is only `ADVANCED_ROBUSTNESS_CANDIDATE`: research-only, no auto-promotion and no execution.
- Phase 58 does not replace Phase 41; it adds multiple-testing/selection-bias diagnostics that Monte Carlo path shuffling alone cannot detect.


## Phase 59 — Hardened Promotion Gate + Pristine Final Validation Barrier

- Brian file: `brian2026/phase59_hardened_promotion_gate.py`
- This phase introduces no new trading algorithm. It composes:
  - Phase 49 preliminary shadow/paper/causality eligibility;
  - Phase 58 CPCV/PBO/Deflated-Sharpe advanced overfit controls;
  - Phase 50 authoritative execution-state reconciliation;
  - Brian's pre-existing scientific rule that contaminated/previously used data cannot be called a pristine final holdout.
- Hardening behavior:
  - Phase 49 `MICRO_LIVE_ELIGIBLE` is explicitly treated as **preliminary**, not final approval;
  - a one-shot final-validation receipt must identify a SHA-256-sealed dataset, be sealed before evaluation, be evaluated exactly once, never be used for tuning, and never be marked contaminated;
  - research/overfit failures block before execution-state review;
  - unresolved execution reconciliation blocks even when research has passed;
  - absence of pristine final validation produces `FINAL_VALIDATION_REQUIRED`;
  - passing every automated gate yields only `HUMAN_REVIEW_READY`.
- Even `HUMAN_REVIEW_READY` has:
  - `human_authorization_required=True`;
  - `exchange_adapter_enabled=False`;
  - `capital_authorized=False`;
  - `automatic_activation=False`;
  - `live_execution=False`.
- This makes the current absence of a pristine final holdout an explicit blocker instead of allowing a paper/backtest pass to be misrepresented as live readiness.


## Phase 60 — Append-Only Shadow State Ledger + Cycle Continuity

- Brian file: `brian2026/phase60_shadow_state_ledger.py`
- Reference project: NautilusTrader (`nautechsystems/nautilus_trader`, behavioral reference only).
- Reference revision inspected: `e1a67a5ba1c3bad68a3b6d2c8122b38c308c2844`.
- Upstream behavior inspected:
  - `docs/concepts/cache.md`: trading state (accounts, orders and positions) is retained in the central cache; when a backing database is configured, supported state can be restored after restart before execution reconciliation.
  - the live node restores persisted cache state and rebuilds derived indexes before connecting clients/reconciling execution state.
  - `docs/concepts/execution/reconciliation.md`: retaining execution events/state reduces reliance on short venue-history windows and gives reconciliation enough order/position context to interpret current state.
  - Nautilus event-store replay consumes durable sequence order and supports snapshot anchors plus replay of the subsequent event tail; replay divergence is surfaced instead of silently skipped.
- Brian clean-room adaptation:
  - a content-addressed `ShadowAccountState` is the only account/position/cash head used across shadow cycles;
  - all ledger transitions are append-only, sequential and hash-chained;
  - a Phase 57 simulation cycle is persisted as execution evidence but **cannot mutate account state**;
  - a second cycle cannot start while the previous cycle is unresolved, preventing overlapping decisions from reading different implicit account states;
  - pure simulation cycles can be closed while explicitly preserving the same authoritative state;
  - state changes require a Phase 50 reconciliation report with `ready=True`, every required check passing, and an explicit `RECONCILED_PAPER` account snapshot whose reconciliation hash exactly matches the report;
  - committed state must cover every tracked reconciliation asset, keep the same account identity and never move time backwards;
  - duplicate cycle/commit submissions are idempotent only when the content hash is identical; conflicting history is rejected;
  - integrity replay verifies sequence, previous-transition hashes, state continuity and pending-cycle continuity.
- Phase 60 deliberately does not treat simulated fills or simulated sale proceeds as authoritative cash/position updates.
- No live exchange state, broker transport or automatic promotion is introduced.


## Phase 61 — Stateful Paper Venue + Reconciliation Loop

- Brian file: `brian2026/phase61_stateful_paper_venue.py`
- Reference project: NautilusTrader sandbox execution (`nautechsystems/nautilus_trader`, LGPL-3.0; behavioral reference only, no source copied).
- Reference revision inspected: `e1a67a5ba1c3bad68a3b6d2c8122b38c308c2844`.
- Upstream behavior inspected:
  - `crates/adapters/sandbox/src/execution.rs`: sandbox execution keeps starting balances, matching-engine state, inflight order state and account identity while consuming live/simulated market data without sending real venue orders.
  - sandbox order flow emits ordinary order/fill/account events and supports report generation so the execution engine can reconcile paper state through the same conceptual execution boundary.
  - `docs/concepts/backtesting/trade-execution.md`: simulated fills are quantity-bounded by available execution evidence/liquidity, partial fills are valid outcomes, and sandbox paper trading shares the matching-engine execution semantics.
  - developer guidance requires account state and reconciliation to be established before trusting order flow.
- Brian clean-room adaptation:
  - Phase 57 execution receipts become persistent **paper** orders/fills only when explicitly applied to Phase 61;
  - cycle application is content-addressed and idempotent; reusing a cycle id with different evidence is rejected;
  - paper fills update quote-cash, signed net quantity, weighted average entry and realized PnL;
  - partial fills update only the filled quantity;
  - new BUY risk can be rejected when cash cannot fund fill + fees;
  - reduce-only BUY exits are never blocked merely because paper cash is insufficient, preserving the safety rule that risk reduction must remain possible;
  - risk-denied and local slippage-veto outcomes never become paper venue fills;
  - explicit flat `VenuePositionReport` rows are generated for tracked assets instead of treating missing reports as flat;
  - the paper venue can reconcile its reports against a caller-maintained local mirror through the real Phase 50 contract;
  - only a successful Phase 50 reconciliation can be converted into a `RECONCILED_PAPER` Phase 60 state snapshot;
  - mark-to-market equity requires explicit marks for every open paper position and short-sale cash is conservatively capped when exposed as next-cycle available cash.
- Phase 61 is a local paper venue only. It has no API-key transport, exchange connector, live order path or automatic capital authorization.
