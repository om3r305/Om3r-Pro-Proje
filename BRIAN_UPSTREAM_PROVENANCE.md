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


## Phase 62 — Automatic Shadow/Paper Parity Evidence Bridge

- Brian file: `brian2026/phase62_paper_parity_evidence.py`
- This phase introduces no new external algorithm. It composes the already provenance-tracked contracts from Phase 49, Phase 50, Phase 57 and Phase 61.
- Integration behavior:
  - Phase 57 shadow execution items are paired one-to-one with Phase 61 paper outcomes from the same content-addressed cycle;
  - a paper receipt whose cycle hash does not match the originating shadow cycle is rejected;
  - Phase 56 risk-denied legs and Phase 46 local slippage-veto legs are excluded because they were never submitted to the paper venue;
  - submitted legs preserve shadow direction/fill fraction, paper acknowledgement/fill fraction/fill price, and Phase 50 per-asset reconciliation completion in one `ShadowPaperObservation`;
  - a definitive paper rejection records paper direction `0` and acknowledgement failure instead of pretending the shadow order reached the venue;
  - missing/unresolved reconciliation marks the observation ambiguous and therefore fails the strict Phase 49 parity policy;
  - cycle evidence is append-only and content-addressed; identical repeats are idempotent while conflicting reuse of a cycle id is rejected;
  - the ledger evaluates by calling the real Phase 49 `evaluate_shadow_paper_parity` implementation rather than recreating parity thresholds.
- Phase 62 remains shadow/paper evidence only and exposes no execution transport.


## Phase 63 — Crash-Safe Paper/Shadow Checkpoint + Recovery

- Brian file: `brian2026/phase63_crash_recovery.py`
- Reference project: NautilusTrader cache persistence and event-store replay (`nautechsystems/nautilus_trader`, behavioral reference only).
- Reference revision inspected: `e1a67a5ba1c3bad68a3b6d2c8122b38c308c2844`.
- Upstream behavior inspected:
  - persisted cache state can be restored before execution reconciliation;
  - execution/event state is retained in durable sequence order;
  - snapshot anchors are content-hashed and recovery replays the tail after the snapshot;
  - replay divergence, missing payloads and invalid sequencing are surfaced as recovery errors rather than silently accepted.
- Brian clean-room adaptation:
  - Phase 61 exports a content-addressed paper checkpoint containing ordered fills, applied-cycle receipts, cash, state version and final positions;
  - paper restore **replays fills from starting cash** and recomputes positions instead of trusting serialized cash/position fields;
  - every fill id and cycle receipt id is recomputed and verified;
  - cycle state-version ordering, cash-before/cash-after continuity, fill ownership and fill ordering are validated;
  - orphan, reordered, duplicated or content-modified fills fail recovery;
  - Phase 60 ledger manifests are restored by rebuilding content-addressed states and transitions, verifying ledger hash, genesis payload hash, transition ids, sequence order, previous-transition links and state continuity;
  - recovery rebuilds the cycle idempotency indexes and pending-cycle state, so a process crash does not permit a duplicate cycle to execute again;
  - a runtime checkpoint verifies the paper venue and shadow ledger use the same account identity;
  - the critical crash window where a paper fill exists but Phase 50 reconciliation/Phase 60 commit has not yet completed is preserved as a pending cycle and can be reconciled after restart.
- Phase 63 recovery is entirely local shadow/paper state recovery; no live adapter, API key or real-money execution surface is introduced.


## Phase 64 — Independent Local Execution Event Projector

- Brian file: `brian2026/phase64_local_execution_projector.py`
- Reference project: NautilusTrader ExecutionEngine/cache event processing (`nautechsystems/nautilus_trader`, behavioral reference only).
- Reference revision inspected: `e1a67a5ba1c3bad68a3b6d2c8122b38c308c2844`.
- Upstream behavior inspected:
  - `docs/concepts/events/index.md`: the ExecutionEngine processes each `OrderFilled`, updates or creates the cached position, then emits corresponding position lifecycle events.
  - execution and reconciliation read the engine-maintained cache as local state; venue position reports remain an independent external/reconciliation input.
  - event/replay ordering is explicit so a missing or reordered fill can produce a detectable local/venue divergence rather than silently copying venue state into cache.
- Brian clean-room adaptation:
  - Phase 64 never reads Phase 61 paper positions or cash;
  - it consumes only immutable `PaperCycleReceipt` order outcomes and the referenced `PaperFill` events;
  - every cycle receipt and fill id is content-hash verified independently;
  - the complete cycle is validated before any local projection state changes, so a missing fill event cannot leave a half-applied local cache;
  - filled/partial-filled outcomes must reconcile their referenced fill quantity, weighted-average price and fill fraction;
  - risk-denied, local-veto, acknowledged-no-fill and venue-rejected outcomes create order history but no position mutation;
  - fills update an independent net position projection with average entry, realized PnL and source-fill lineage;
  - duplicate cycles are idempotent only when receipt evidence is identical;
  - Phase 50 can now compare Phase 64 local positions against Phase 61 venue reports, making reconciliation an actual independent-state check rather than a venue-state copy.
- Phase 64 is local shadow/paper event projection only and contains no exchange transport.


## Phase 65 — Event-Sourced Local Cache Recovery

- Brian file: `brian2026/phase65_event_sourced_local_recovery.py`
- Reference project: NautilusTrader ExecutionEngine/event-store/cache recovery behavior (`nautechsystems/nautilus_trader`, behavioral reference only).
- Reference revision inspected: `e1a67a5ba1c3bad68a3b6d2c8122b38c308c2844`.
- Upstream behavior inspected:
  - durable execution events can be replayed to rebuild local execution/cache state after restart;
  - order fills are the source events from which cached positions are updated rather than trusting a separate copied venue position snapshot;
  - restored local state is still reconciled against venue reports after recovery.
- Brian clean-room adaptation:
  - the Phase 63 paper checkpoint is validated first by deterministic paper-venue replay;
  - Phase 64 local execution state is then rebuilt from zero by replaying every durable Phase 61 cycle receipt and referenced fill event;
  - no serialized local position snapshot is required or trusted;
  - recovery records how many cycles/fills were available and replayed plus the resulting local projection hash;
  - a full recovered runtime contains the independently restored Phase 60 ledger, Phase 61 venue and Phase 64 local projector;
  - before the recovered runtime is returned, Phase 50 must reconcile the rebuilt local positions against the independently restored paper venue reports;
  - the pending-cycle crash window is preserved: a paper fill can already exist while the Phase 60 ledger still marks the cycle pending, and local cache replay catches up without silently committing the ledger;
  - a forensic partial-replay helper intentionally truncates local event replay so Phase 50 can prove that missing event history produces a reconciliation failure.
- Phase 65 is restart/recovery logic for local shadow/paper execution state only; it exposes no live execution transport.


## Phase 66 — Durable Shadow Cycle Write-Ahead Journal

- Brian file: `brian2026/phase66_durable_cycle_journal.py`
- Reference project: NautilusTrader event-store/cache replay behavior (`nautechsystems/nautilus_trader`, behavioral reference only).
- Reference revision inspected: `e1a67a5ba1c3bad68a3b6d2c8122b38c308c2844`.
- Upstream behavior inspected:
  - durable execution state is sequenced before replay/reconciliation;
  - replay works from persisted event/state payloads, not only opaque identifiers;
  - sequence/hash divergence is treated as recovery failure instead of silently skipping missing execution history.
- Brian clean-room adaptation:
  - the complete Phase 57 `ShadowExecutionCycle` body is content-addressed and journaled **before** paper/local/reconciliation side effects;
  - journal stages are append-only: `CYCLE_CREATED -> PAPER_APPLIED -> LOCAL_PROJECTED -> RECONCILED -> COMMITTED`, with explicit `RECONCILIATION_REQUIRED` and terminal `ABORTED` branches;
  - every entry is globally sequenced and chained to the previous entry id;
  - Phase 61 paper receipts are independently content-id verified before the journal advances;
  - reconciliation artifacts remain shadow/paper only; live reconciliation is rejected;
  - repeated identical stage artifacts are idempotent, while conflicting reuse of a cycle/stage is rejected;
  - the journal manifest persists the **full cycle body**, not merely its hash, so a crash after cycle creation still leaves enough evidence to retry the same paper cycle;
  - restore reconstructs Phase 57 nested risk/execution/pending-reversal receipts, recomputes entry ids, recomputes the journal manifest hash, verifies legal stage transitions and re-validates stored cycle hashes.
- Phase 66 is a write-ahead recovery journal only. It does not submit live orders or authorize capital.


## Phase 66 — Recovery-Aware Shadow/Paper Runtime Coordinator

- Brian file: `brian2026/phase66_runtime_coordinator.py`
- This phase introduces no new external trading algorithm. It composes the already provenance-tracked Phase 50, 57, 60, 61, 63, 64 and 65 contracts into one fail-closed runtime state machine.
- Integration behavior:
  - a cycle is first recorded in the append-only Phase 60 ledger;
  - the exact same cycle is then applied idempotently to the Phase 61 paper venue;
  - immutable Phase 61 order/fill events are projected independently through Phase 64;
  - Phase 50 reconciles the independent local projection against explicit paper-venue reports;
  - only a ready reconciliation plus complete mark-to-market inputs may produce a new Phase 60 authoritative state;
  - missing mark prices return `MARKS_REQUIRED` and keep the cycle pending without changing the ledger head;
  - local/venue divergence returns `RECONCILIATION_BLOCKED` and keeps the cycle pending;
  - while a cycle is pending, any different new cycle is rejected, preventing overlapping decisions from spending or sizing from inconsistent account state;
  - replaying the same pending cycle is safe because Phase 60, 61 and 64 are content-addressed/idempotent;
  - Phase 63 checkpoints can be restored through Phase 65, rebuilding the local projector from durable events;
  - after restart, a pending cycle can be reconciled and committed without needing the original `ShadowExecutionCycle` payload, because the durable paper events plus Phase 60 pending identity are sufficient for state completion;
  - construction rejects split-brain account identities across ledger, paper venue and local projector.
- Phase 66 remains shadow/paper-only. It has no exchange transport, API-key surface, capital authorization or automatic live activation.


## Phase 67 — Write-Ahead Durable Runtime Orchestration

- Brian file: `brian2026/phase67_durable_runtime_orchestrator.py`
- This phase introduces no new trading algorithm. It integrates the Phase 66 durable cycle journal with the already proven Phase 60–66 shadow/paper runtime contracts.
- Write-ahead behavior:
  - the complete Phase 57 `ShadowExecutionCycle` body is content-addressed in `DurableCycleJournal` **before** any ledger, paper-venue or local-projection side effect;
  - only one non-terminal journal cycle may exist at a time;
  - Phase 60 pending identity and the active journal cycle must agree whenever a ledger pending cycle exists;
  - stage order is enforced as `CYCLE_CREATED -> PAPER_APPLIED -> LOCAL_PROJECTED -> RECONCILED -> COMMITTED` (with explicit reconciliation-required/abort branches already defined by the journal).
- Crash recovery behavior:
  - crash after write-ahead but before Phase 60 append: the full original cycle body is restored from the journal and can be activated without regenerating the decision;
  - crash after Phase 60 append but before paper execution: the same journaled cycle is applied idempotently;
  - crash after Phase 61 paper side effects but before the journal stage write: the restored paper receipt proves the side effect and journal metadata is advanced forward;
  - crash after paper fill but before local projection persistence: Phase 65 event replay rebuilds Phase 64 local state from the durable paper events, and the recovered projection receipt proves `LOCAL_PROJECTED`;
  - recovered local/venue state must pass the real Phase 50 reconciliation before journal state can become `RECONCILED`;
  - missing mark data leaves both Phase 60 and Phase 67 at a pending/reconciled boundary rather than inventing equity;
  - crash after Phase 60 authoritative commit but before the journal `COMMITTED` write is repaired only when the Phase 60 `RECONCILED_COMMIT` transition proves the committed state id;
  - recovery advances journal metadata only from independently restored artifacts; it never infers a side effect from stage order alone.
- Duplicate retries cannot duplicate paper fills or local position projection because the journal, Phase 60 ledger, Phase 61 paper venue and Phase 64 projector are all content-addressed/idempotent.
- Phase 67 remains shadow/paper-only with no exchange transport, API-key surface, capital authorization or automatic live activation.


## Phase 68 — Operational Risk Governor / Circuit Breaker

- Brian file: `brian2026/phase68_operational_risk_governor.py`
- Behavioral references:
  - Freqtrade `freqtrade/freqtrade` revision `06ef422285a11ed293d1990e2bcdc964ed94aac4` (GPL-3.0; **behavior-only clean-room reference, no source copied**).
  - NautilusTrader `nautechsystems/nautilus_trader` revision `e1a67a5ba1c3bad68a3b6d2c8122b38c308c2844` (LGPL-3.0; **behavior-only clean-room reference, no source copied**).
- Upstream behavior inspected:
  - Freqtrade `MaxDrawdown` applies a global trading lock when the configured lookback drawdown is **strictly greater than** its allowed threshold.
  - Freqtrade `StoplossGuard` counts qualifying stop-loss/liquidation exits below a required-profit threshold inside a lookback window and applies a timed trading lock when the configured count is reached.
  - Freqtrade `CooldownPeriod` blocks pair re-entry for a configured period after a recent closed trade.
  - NautilusTrader exposes explicit risk-engine trading states and publishes trading-state changes; `HALTED` blocks new submit/modify flow while `REDUCING` is enforced as a risk-reduction-only boundary.
- Brian clean-room adaptation:
  - rolling peak-to-trough equity drawdown and window-loss limits can escalate the runtime to `HALTED`;
  - qualifying stop-loss bursts produce a time-bounded `REDUCING` state instead of opening new risk;
  - recently closed assets receive per-asset re-entry cooldowns without unnecessarily halting unrelated assets;
  - consecutive execution failures and repeated reconciliation failures escalate to `REDUCING`;
  - stale market data, unknown/ambiguous order outcomes and explicit manual halt escalate to `HALTED`;
  - `HALTED` is latched: clearing the triggering condition is insufficient by itself, and a manual release is required before new risk can resume;
  - manual release cannot override a severe trigger which is still active;
  - future-dated health/trade/equity observations are excluded from current-window decisions and a future market-data timestamp is rejected;
  - every decision emits a deterministic, content-addressed operational-risk receipt with the measured inputs/reasons and blocked-asset set;
  - the receipt converts directly to the real Phase 56 `PreTradeRiskPolicy`, so `REDUCING` rejects new risk while valid reduce-only intent remains possible, and `HALTED` blocks submissions according to the existing Phase 56 contract.
- Phase 68 changes no live adapter and cannot authorize capital or exchange execution.


## Phase 69 — Operationally Governed Shadow Execution

- Brian file: `brian2026/phase69_governed_shadow_execution.py`
- This phase introduces no new external algorithm. It composes the Phase 68 operational-risk receipt with the real Phase 56 pre-trade RiskEngine and Phase 57 execution simulator.
- Supporting contract hardening:
  - Phase 56 `PreTradeRiskPolicy` now carries an optional per-asset `block_new_risk` flag;
  - that flag is evaluated only for new/increasing-risk `TradeIntent`; it does not block a valid `RiskReductionIntent`;
  - Phase 57 accepts optional per-asset Phase 56 policies while preserving its prior global trading-state/default-limits behavior when none are supplied;
  - Phase 68 exposes `pretrade_policy_for_asset`, translating its cooldown set into the Phase 56 new-risk lock while retaining the global ACTIVE/REDUCING/HALTED state.
- Integration behavior:
  - every asset present in a Phase 55 rebalance plan receives a Phase 56 policy derived from the same content-addressed Phase 68 receipt;
  - a per-asset cooldown blocks only that asset's OPEN/INCREASE leg, so unrelated assets are not globally frozen;
  - cooldown does not consume/reserve cash for the denied leg, preserving cash for independently allowed assets in the same cycle;
  - a cooled-down asset may still REDUCE/CLOSE while the global state is ACTIVE or REDUCING;
  - global `REDUCING` denies all new/increased risk while still permitting valid exposure reduction;
  - global `HALTED` follows the existing Phase 56 submission boundary and denies both new-risk and reduce submissions;
  - execution still runs through the real Phase 57 cash reservation, Phase 45 executor action and Phase 46 order-book/latency fill simulation rather than a parallel shortcut;
  - the result records the Phase 68 receipt id, per-asset policy fingerprint, blocked-new-risk asset set and resulting Phase 57 cycle id.
- Phase 69 remains fully shadow-only with no broker/exchange transport or live capital authorization.


## Phase 70 — Transactional Durable Runtime Store

- Brian files:
  - `brian2026/phase70_durable_runtime_store.py`
  - `supabase/migrations/202609230720_brian_phase70_durable_runtime_store.sql`
- This phase introduces no trading algorithm. It turns the Phase 63–67 crash/recovery contracts into a real Postgres/Supabase persistence boundary.
- Proven internal concurrency patterns reused:
  - `202609030009_brian_collector_lease.sql`: owner-token lease acquisition/renewal/release and expired-owner recovery;
  - `202609111675_brian_treasury_atomic_cas_guard.sql`: transaction-scoped advisory serialization and stale-parent compare-and-swap;
  - `202609111695_brian_ocean_atomic_control_guard.sql`: database-time refresh after lock acquisition and serialized control transitions.
- Current Supabase database-function guidance was rechecked before implementation:
  - data-intensive atomic logic belongs inside Postgres functions/RPC;
  - `security invoker` is preferred generally, while genuine privileged `security definer` functions must pin `search_path` and have explicit EXECUTE grants;
  - functions are executable by PUBLIC by default unless privileges are revoked.
- Phase 70 adaptation:
  - one mutable operational head row per runtime holds current checkpoint version, lease owner, fencing token and current checkpoint anchors;
  - immutable checkpoint history, full cycle bodies, journal entries and runtime events are stored in separate append-only tables;
  - all append-only tables reject UPDATE/DELETE and have no direct anon/authenticated/service-role table mutation grant;
  - the service role may mutate runtime persistence only through explicitly granted RPC functions;
  - lease acquisition is serialized per runtime and expired takeover increments a fencing token;
  - renew/commit require exact owner token + fencing token + unexpired lease;
  - release rotates the owner token so an already-in-flight stale renewal cannot resurrect ownership;
  - checkpoint commit is serialized and uses expected-version CAS; concurrent commits from the same version cannot both advance the head;
  - exact checkpoint retries are idempotent, including historical retries, without rolling the authoritative head backwards;
  - a reused checkpoint id with different payload is treated as an integrity violation;
  - incoming durable journals must contain the entire previously persisted prefix; truncation or mutation of an existing sequence fails closed;
  - full Phase 57 cycle bodies are persisted once and conflicting cycle body/hash reuse is rejected;
  - DB heads expose checkpoint id, journal hash, Phase 60 head-state id and pending-cycle id as independent restore anchors;
  - Python `DurableRuntimeStore` is transport-agnostic (RPC callable injection) so Brian does not gain a hard dependency on supabase-py;
  - loaded JSON is reconstructed through Phase 63/67 content-hash validation and then cross-checked against all database head anchors before it can be returned to the runtime.
- Real Postgres 16 CI exercises fresh-acquire races, concurrent checkpoint CAS, expired-owner fencing, release/renew safety, journal prefix extension/conflict, exact and historical retries, checkpoint-id collision, and direct service-role mutation denial.
- The migration is committed to GitHub only in this PR; it has not been rolled out to a live Supabase project.
- Phase 70 remains shadow/paper-only and adds no exchange transport or capital authorization.


## Phase 71 — Persisted Durable Runtime Supervisor

- Brian file: `brian2026/phase71_persisted_runtime_supervisor.py`
- This phase introduces no new external algorithm. It composes the Phase 67 write-ahead runtime with the Phase 70 transactional persistence/fencing boundary.
- Persistence sequencing:
  - acquiring a fresh runtime lease persists an initial validated checkpoint before work begins;
  - every new Phase 57 cycle is first added to the Phase 66/67 full-body write-ahead journal;
  - that write-ahead checkpoint is committed through Phase 70 CAS **before** `advance_pending` may execute paper/reconciliation work;
  - after Phase 67 advances, a second checkpoint persists the resulting journal/venue/ledger state.
- Crash semantics:
  - if the process dies after the first DB commit but before/during paper execution, the authoritative DB checkpoint still contains the full original cycle body at `CYCLE_CREATED`;
  - restart restores that checkpoint and deterministically replays the same cycle rather than regenerating a decision;
  - because the current Phase 61 venue is paper-local, any in-memory work performed after the durable write-ahead but before the second checkpoint has no external exchange side effect and is safe to replay.
- Stale-worker behavior:
  - `CAS_CONFLICT`, historical-checkpoint mismatch or lease/fencing loss invalidates the supervisor instance immediately;
  - an invalid supervisor cannot continue advancing its mutated local runtime and must reload from the Phase 70 authoritative head;
  - lease renewal verifies that the fencing token is unchanged and that the database runtime version has not advanced outside the supervisor;
  - a `DUPLICATE_CURRENT` checkpoint is accepted as the safe equivalent of a commit whose network response was lost, while `DUPLICATE_HISTORICAL` is treated as stale local state.
- Durable-head precedence:
  - if a checkpoint already exists, caller-provided initial memory is ignored and the database checkpoint is restored through Phase 63/67 validation;
  - local lease-version snapshots are advanced with every accepted checkpoint commit so heartbeat/diagnostic state stays aligned with the durable head.
- Phase 71 remains shadow/paper-only and adds no external execution side effect.


## Phase 72 — Durable Operational Risk Ledger + Restart Locks

- Brian file: `brian2026/phase72_operational_risk_ledger.py`
- This phase introduces no new external protection algorithm. It hardens Phase 68 so operational safety state survives process restart.
- Phase 68 receipt hardening:
  - each operational-risk receipt can re-verify its own content hash;
  - `HALTED` and `halt_latched` are structurally required to agree;
  - temporary stop-loss, execution-failure and reconciliation-failure lock expiries are carried in the receipt;
  - per-asset cooldown expiry timestamps are persisted, not only the blocked-asset names;
  - blocked assets must exactly match the persisted cooldown map;
  - governor restoration from a verified receipt restores halt latch plus all unexpired temporary lock state.
- Recovery semantics:
  - a stop-loss REDUCING lock survives restart until its original expiry even when the caller does not replay old closed-trade rows;
  - per-asset cooldowns survive restart until their exact expiry and then clear automatically;
  - an execution-failure REDUCING lock survives restart conservatively, but a fresh EXECUTION_SUCCESS may clear it early;
  - a reconciliation-failure REDUCING lock behaves the same with a fresh RECONCILIATION_SUCCESS;
  - severe HALTED state remains latched across restart and still requires explicit manual release after the severe trigger itself has cleared.
- Phase 72 ledger behavior:
  - the exact `OperationalRiskPolicy` is content-addressed and sealed into the ledger;
  - receipts form a sequential append-only hash chain;
  - each new receipt must name the ledger's current trading state as its `previous_state`;
  - distinct receipts must advance time, preventing ambiguous same-timestamp state forks;
  - identical receipt retries are idempotent;
  - manifest restore rebuilds every receipt/entry, policy hash, current state, latch state and overall ledger hash before a governor can be recreated.
- Phase 72 remains shadow/paper risk state only. It does not create orders or enable live execution.


## Phase 73 — Transactional Operational-Risk Store

- Brian files:
  - `brian2026/phase73_operational_risk_store.py`
  - `supabase/migrations/202609230745_brian_phase73_operational_risk_store.sql`
- This phase introduces no new risk algorithm. It persists the Phase 72 operational-risk ledger under the **same Phase 70 runtime lease and fencing token** used by durable execution state.
- Database behavior:
  - one mutable operational-risk head per runtime tracks risk version, ledger hash, policy hash, head entry, current state and HALT latch;
  - immutable risk snapshots, receipt/entry history and audit events are append-only;
  - direct anon/authenticated/service-role mutation of risk tables is revoked; service-role writes go through explicit RPC only;
  - risk commits acquire the same transaction advisory lock key as Phase 70 execution checkpoints;
  - exact owner token + current fencing token + unexpired runtime lease are required;
  - risk history has its own expected-version CAS so two concurrent same-version decisions cannot both advance the risk head;
  - expired runtime takeover makes the old worker's risk commit return `LEASE_LOST`;
  - exact ledger retries are idempotent; historical retries cannot roll the head backwards;
  - a reused ledger hash with a changed manifest is an integrity violation;
  - every incoming risk manifest must preserve the exact persisted entry prefix;
  - database validation also enforces receipt state continuity, strictly increasing distinct receipt timestamps, shadow-only receipts, and HALT/latch consistency.
- Python store behavior:
  - the transport remains RPC-callable based and does not introduce a hard Supabase client dependency;
  - before commit, the ledger is round-tripped through the real Phase 72 restore validator;
  - on load, the DB manifest is rebuilt through Phase 72 and independently cross-checked against DB ledger hash, policy hash, head entry, current state and HALT latch.
- Real Postgres 16 CI covers concurrent risk-CAS races, fencing takeover, prefix extension/rewrite rejection, state/time-chain rejection, historical retry and direct service-role mutation denial.
- The Phase 73 migration is GitHub-only in this draft PR and has not been deployed to a live Supabase project.
- Phase 73 remains shadow/paper operational state only.


## Phase 74 — Persisted Risk-to-Cycle Authorization Binding

- Brian files:
  - `brian2026/phase74_governed_cycle_binding.py`
  - `supabase/migrations/202609230815_brian_phase74_governed_cycle_binding.sql`
- This phase introduces no new trading logic. It closes the identity gap between the Phase69 governed execution result and the independently persisted Phase72/73 operational-risk head.
- Database behavior:
  - binding uses the same Phase70 runtime advisory lock, lease owner and fencing token;
  - current runtime version must equal the version used to generate/submit the governed cycle;
  - current Phase73 risk version, ledger hash and **head receipt id** must exactly match the risk evidence named by the Phase69 governed result;
  - the immutable binding stores cycle id, governed result id, policy fingerprint, risk version/hash/receipt, runtime version and fencing token;
  - exact retries are idempotent and return the full immutable evidence anchors;
  - same cycle id with changed governed/risk/policy evidence is an integrity error;
  - stale runtime, stale risk or lost lease fail closed before any paper execution path;
  - direct service-role mutation of binding history is denied.
- Python behavior:
  - `GovernedCycleBindingStore` verifies the supplied risk ledger belongs to the runtime and has a persisted head receipt;
  - the Phase69 governed result must name that exact persisted head receipt before an RPC is attempted;
  - successful/duplicate database receipts must echo every immutable evidence anchor, otherwise the client rejects the response;
  - loaded bindings revalidate content-hash shapes and shadow/live boundary flags.
- Real Postgres 16 CI covers current-head binding, exact duplicate, concurrent duplicate race, risk-version advance, runtime-version advance, stale fencing takeover, same-cycle evidence conflict and direct mutation denial.
- Phase74 remains shadow/paper-only.

## Phase 75 — Atomic Governed Authorization + Durable Write-Ahead

- Brian files:
  - `brian2026/phase75_atomic_governed_writeahead.py`
  - `supabase/migrations/202609230845_brian_phase75_atomic_governed_writeahead.sql`
- This phase removes the transaction gap between Phase74 authorization and Phase71 durable `CYCLE_CREATED` persistence.
- Atomic database behavior:
  - the full Phase57 governed cycle must already be present in the submitted Phase67 journal checkpoint;
  - the latest journal stage for that cycle must be exactly `CYCLE_CREATED`, proving no paper/local side effect has occurred yet;
  - the write-ahead checkpoint must still have no Phase60 pending cycle, matching the Phase71 pre-execution boundary;
  - one transaction/advisory lock validates runtime lease/fence/version, locks the current Phase73 risk head, verifies risk version/hash/head receipt, and then calls the already-tested Phase70 checkpoint commit;
  - because Phase70 and Phase73 use the same runtime advisory lock, the risk head cannot advance between authorization validation and write-ahead commit;
  - the immutable authorization row is inserted only after the Phase70 checkpoint commit succeeds;
  - any Phase70 failure raises inside the outer transaction, rolling back the authorization as well;
  - exact lost-response retry is idempotent and can return `DUPLICATE_CURRENT`; a historical duplicate is surfaced separately and cannot be treated as current local state;
  - authorization stores both runtime version before and after plus the exact write-ahead checkpoint id.
- Python supervisor behavior:
  - `PersistedGovernedRuntimeSupervisor` loads the current persisted Phase73 risk head before attempting authorization;
  - it may mutate only the local write-ahead journal before the atomic RPC; paper venue and local projector side effects remain untouched until authorization succeeds;
  - any failed atomic authorization invalidates the local supervisor because its in-memory journal may now differ from the authoritative DB head;
  - after a successful atomic commit, Phase71 accepts the externally committed checkpoint only when the returned checkpoint id exactly matches its current in-memory checkpoint and the version advanced;
  - only then may the ordinary Phase67/71 `advance_pending` path execute and persist paper/reconciliation results.
- Real Postgres 16 CI covers successful atomic commit, lost-response retry, concurrent exact authorization, stale risk, stale runtime, inner Phase70 rollback, same-cycle evidence conflict and direct service-role mutation denial.
- Phase75 remains shadow/paper-only and adds no exchange/broker transport.


## Phase 76 — Durable Shadow Execution Outbox / Submit Boundary

- Brian files:
  - `brian2026/phase76_shadow_execution_outbox.py`
  - `supabase/migrations/202609230915_brian_phase76_shadow_execution_outbox.sql`
- This phase introduces no trading algorithm. It defines a durable **submission boundary** after Phase75 authorization but before Phase61 paper execution.
- Database behavior:
  - an immutable dispatch can exist only for an existing Phase75 governed authorization;
  - dispatch submission uses the same Phase70 runtime advisory lock, lease owner and fencing token;
  - the runtime head must still be exactly the Phase75 authorization's write-ahead checkpoint/version;
  - the current Phase73 risk version/hash/head-receipt must still exactly match the Phase75 authorization;
  - if risk changed after authorization but before submit, `RISK_VERSION_CONFLICT` is returned and no dispatch row is created;
  - if runtime state advanced after authorization but before submit, `RUNTIME_VERSION_CONFLICT` is returned and no dispatch row is created;
  - missing authorization, lost lease or changed dispatch identity fail closed;
  - exact submit retries are idempotent; concurrent identical submits serialize to one `SUBMITTED` and one duplicate result;
  - the durable dispatch copies the governed-result id, policy fingerprint, authorization checkpoint/version and risk anchors from the immutable Phase75 authorization rather than trusting caller-supplied duplicates;
  - direct service-role mutation of dispatch history is denied.
- Python behavior:
  - the dispatch id is deterministically content-addressed from the Phase75 authorization anchors;
  - `ShadowExecutionOutboxStore` submits only runtime id/owner/fence/cycle/dispatch id; successful DB responses must echo all immutable authorization anchors;
  - `PersistedDispatchedRuntimeSupervisor` cannot call paper/local execution until a durable dispatch exists;
  - a risk change before submit is treated as a normal pre-paper safety abort: the Phase66 journal moves `CYCLE_CREATED -> ABORTED` and that terminal checkpoint is persisted without any Phase61/64 side effect;
  - lease/runtime conflicts invalidate the local supervisor and require reload;
  - a historical duplicate dispatch is never treated as current local state.
- Semantics:
  - before `SUBMITTED`, a changed risk head can still veto the cycle with zero paper side effect;
  - after `SUBMITTED`, a later HALT belongs to the post-submission cancel/kill-switch lifecycle, providing a clean boundary for the next phase.
- Real Postgres 16 CI covers successful submit, exact and concurrent duplicate submit, missing authorization, risk-head advance, runtime-head advance, dispatch-id conflict and direct mutation denial.
- Phase76 remains shadow/paper-only; it does not transmit an order to an exchange or broker.


## Phase 77 — Fenced Execution Claim + Pre-Execution Kill-Switch Lifecycle

- Brian files:
  - `brian2026/phase77_execution_claim_lifecycle.py`
  - `supabase/migrations/202609230945_brian_phase77_execution_claim_lifecycle.sql`
- This phase introduces no new alpha/trading algorithm. It extends the Phase70 lease/fencing and Phase68 risk-state semantics to the worker-consumption boundary after a Phase76 durable dispatch.
- Claim ownership:
  - a `SUBMITTED` dispatch cannot advance into Phase67/71 paper execution until a worker owns a DB-persisted claim;
  - claims have an independent worker token, monotonically increasing claim fencing token and TTL;
  - a second worker is blocked while an existing claim is live;
  - an expired claim can be taken over with a higher claim fence;
  - claim renewal requires exact runtime lease/fence + worker token + claim fence + an unexpired current claim.
- Fresh pre-execution risk recheck:
  - the claim RPC reads the current persisted Phase73 head receipt and the immutable Phase57 cycle body under the same runtime advisory lock;
  - current `HALTED` cancels a fresh `CYCLE_CREATED` dispatch before paper execution;
  - current `REDUCING` cancels a fresh cycle only when it still contains an allowed non-reduce-only leg;
  - current `ACTIVE` with newly blocked/cooldown assets cancels only when the cycle contains allowed new risk for one of those assets;
  - reduce-only work remains claimable under REDUCING or per-asset cooldown;
  - missing/malformed risk state fails closed.
- Crash/resume semantics:
  - if the current durable journal is already beyond `CYCLE_CREATED`, an expired-worker takeover is `resume_only`; it does not erase or re-run an already-started side effect and instead relies on the Phase67 idempotent recovery/reconciliation path;
  - if the durable journal is already `COMMITTED`, a claim request recovers terminal `COMPLETED` state without executing again;
  - if the journal is `ABORTED`, the claim state becomes terminal `CANCELLED_BEFORE_EXECUTION`.
- Completion semantics:
  - a worker cannot mark its claim complete merely because local work returned;
  - completion requires the exact worker/claim fence and a current durable runtime checkpoint whose journal stage for that cycle is `COMMITTED`;
  - completion retries are idempotent;
  - if completion metadata is lost after the durable runtime commit, a later claim request reconstructs `COMPLETED` from the committed journal rather than re-executing.
- Python supervisor behavior:
  - Phase76 is split into explicit authorize+submit and advance-submitted boundaries;
  - `PersistedClaimedRuntimeSupervisor` claims first, advances only when claim ownership is proven, persists a risk cancellation through Phase75's ABORTED path, and calls claim completion only after durable runtime commit.
- Real Postgres 16 CI covers competing workers, exact re-claim, expired takeover/fencing, HALTED/REDUCING/cooldown cancellation, reduce-only exceptions, resume-only recovery, claim renewal, completion gating/retry, committed recovery and direct mutation denial.
- Phase77 remains shadow/paper-only and does not transmit orders to an external venue.


## Phase 78 — Double-Checked Execution Kill-Switch / Cancel Requests

- Brian files:
  - `brian2026/phase78_execution_kill_switch.py`
  - `supabase/migrations/202609231015_brian_phase78_execution_kill_switch.sql`
- This phase introduces no new alpha or execution model. It closes the risk-race window after Phase77 claim ownership is established.
- Database behavior:
  - every kill-switch check requires the current Phase70 runtime lease/fence and the exact Phase77 worker/claim fencing token;
  - the current persisted Phase73 risk head is read under the same runtime advisory lock and evaluated against the immutable Phase57 cycle body;
  - at `CYCLE_CREATED`, current HALTED / REDUCING-with-new-risk / affected ACTIVE-cooldown conditions convert the claim to terminal `CANCELLED_BEFORE_EXECUTION` with zero paper side effect;
  - after the durable journal proves execution already started, the same incompatible risk state creates an append-only `AFTER_START` cancel request and returns `CANCEL_REQUESTED` while allowing Phase67 idempotent recovery/reconciliation to continue;
  - reduce-only cycles remain permitted under REDUCING or asset cooldown;
  - healthy already-started work returns `RESUME_ONLY`;
  - COMMITTED and ABORTED journal states are terminal and returned before any new risk decision;
  - wrong/expired worker claim, lost runtime lease or unavailable risk state fail closed;
  - cancel-request history is append-only and direct service-role mutation is denied.
- Python behavior:
  - Phase77 is split into explicit claim acquisition and claimed execution advance, with optional deferred claim completion;
  - `PersistedKillSwitchRuntimeSupervisor` performs one DB risk check immediately before Phase67/71 advance and a second check immediately after it;
  - pre-execution cancellation is persisted through the existing Phase75 ABORTED write-ahead path;
  - a post-start cancellation is surfaced as a durable cancel-request outcome without discarding/restarting paper side effects;
  - claim completion is attempted only after the durable runtime is COMMITTED; completion metadata loss cannot cause re-execution because Phase77 can recover terminal completion from the journal.
- Real Postgres 16 CI covers healthy proceed, claim-to-HALT/REDUCING/cooldown races, reduce-only exceptions, post-start cancel requests, healthy resume-only, wrong worker/fence, committed terminal recovery and direct mutation denial.
- Phase78 remains shadow/paper-only. A future real-venue adapter would map AFTER_START cancel requests to venue-specific cancel/flatten behavior rather than bypassing reconciliation.


## Phase 79 — Atomic Execution-Start Point-of-No-Return

- Brian files:
  - `brian2026/phase79_atomic_execution_start.py`
  - `brian2026/sql/phase79_atomic_execution_start.sql` (**draft SQL contract, not an official Supabase migration yet**)
- This phase introduces no new alpha, execution-price or risk algorithm. It closes the remaining transaction window between the existing Phase78 pre-execution risk decision and the first paper/recovery side-effect boundary.
- Composition rather than policy duplication:
  - Phase78 remains the single owner of current-risk semantics (`PROCEED`, `RESUME_ONLY`, `CANCELLED_BEFORE_EXECUTION`, `CANCEL_REQUESTED`, terminal states);
  - Phase79 acquires the same Phase70 runtime transaction advisory lock and invokes the real Phase78 kill-switch RPC inside that same transaction;
  - only a Phase78 decision which permits progress can create an immutable Phase79 `STARTED` row;
  - therefore a Phase73 risk commit cannot slip between the final Phase78 veto decision and the durable point-of-no-return.
- Claim/fencing behavior:
  - an exact current Phase77 `CLAIMED` owner, worker token, claim fencing token, runtime lease owner/fence and unexpired TTL are required **before** checking for an existing STARTED row;
  - this ordering prevents a non-owner from exploiting idempotent `STARTED_ALREADY` recovery;
  - exact concurrent retries serialize to one `STARTED` and one `STARTED_ALREADY`;
  - stale worker token or claim fence returns `CLAIM_LOST` even when STARTED already exists.
- Point-of-no-return semantics:
  - at fresh `CYCLE_CREATED`, an incompatible current risk head still yields Phase78 `CANCELLED_BEFORE_EXECUTION`; no STARTED record is created and Phase61/64 work must not begin;
  - healthy fresh work records `STARTED` with runtime version/fence, claim fence, current risk version/receipt/state and journal stage;
  - already-started recovery (`RESUME_ONLY`) records `STARTED_RESUME`;
  - if Phase78 already has an `AFTER_START` `CANCEL_REQUESTED` for durable in-progress work, Phase79 records `STARTED_RESUME` plus the cancel evidence instead of fabricating a rollback;
  - once STARTED exists, later risk deterioration remains a Phase78 post-start cancel/recovery signal and cannot erase/relabel the immutable start evidence.
- Python orchestration:
  - `PersistedAtomicStartedRuntimeSupervisor` runs Phase77 authorize/submit/claim, then requires Phase79 STARTED before calling `advance_claimed`;
  - after advance it reuses the existing Phase78 kill-switch for post-start safety and the existing Phase77 completion contract for durable COMMITTED completion;
  - a post-STARTED response attempting `CANCELLED_BEFORE_EXECUTION` is treated as an evidence contradiction and fails closed rather than rewriting history.
- Real Postgres 16 CI covers exact/concurrent start, wrong-worker and wrong-claim-fence duplicate attacks, claim-to-start risk changes (HALTED/REDUCING/cooldown), reduce-only exceptions, resume-with-cancel evidence, post-start HALT, read-back anchors and direct service-role mutation denial.
- Supabase rollout note:
  - the SQL is intentionally stored outside `supabase/migrations` while this PR remains draft/undeployed;
  - before any rollout it must be converted into an official migration using `supabase migration new`, then rerun through the same full Python/Postgres gates.
- Phase79 remains shadow/paper-only and does not transmit an order to any exchange or broker.


## Phase 80 — Claim-Fenced Authoritative Runtime Checkpoint

- Brian files:
  - `brian2026/phase80_claim_fenced_checkpoint.py`
  - `brian2026/sql/phase80_claim_fenced_checkpoint_commit.sql`
- This phase introduces no new trading/alpha policy. It closes the authority gap after Phase79 `STARTED`: replay-safe paper/projector work may advance in memory, but that work cannot become the authoritative Phase70 runtime checkpoint unless the same Phase77 worker claim is still current.
- Database behavior:
  - commit requires the current Phase70 runtime lease owner/fence and the exact current Phase77 worker token + claim fencing token;
  - an immutable Phase79 STARTED record must exist for the dispatch before any post-STARTED checkpoint can become authoritative;
  - submitted checkpoints must prove a post-STARTED journal stage (`PAPER_APPLIED`, `LOCAL_PROJECTED`, `RECONCILIATION_REQUIRED`, `RECONCILED`, `COMMITTED`, or `ABORTED`);
  - Phase80 serializes on the same runtime advisory lock and delegates the actual durable checkpoint append/CAS validation to the already-tested Phase70 commit RPC;
  - claim takeover/expiry makes the old worker return `CLAIM_LOST` before it can advance the durable runtime head;
  - runtime version drift returns `RUNTIME_VERSION_CONFLICT`;
  - an exact lost-response retry is accepted as `DUPLICATE_CURRENT` only while the same worker/claim fence still owns execution and the current authoritative checkpoint id+payload exactly match the submitted checkpoint;
  - if the original commit succeeded but the claim was subsequently taken over, the stale worker's retry is rejected as `CLAIM_LOST`, not misclassified as an idempotent success;
  - claim-commit audit history is append-only and has no direct service-role table write grant.
- Python supervisor behavior:
  - Phase71 exposes an explicit non-authoritative in-memory advance path for replay-safe paper/projector state;
  - Phase80 runs that in-memory advance only after Phase79 STARTED, then attempts the claim-fenced authoritative DB commit;
  - any failed authoritative commit invalidates the mutated local supervisor so the worker cannot continue from a state the database did not accept;
  - successful commit receipts must echo runtime/cycle/dispatch/checkpoint and both runtime + claim fencing tokens before the local supervisor accepts the external commit;
  - only after the authoritative commit does the supervisor run the Phase78 post-start risk check and Phase77 completion path.
- Real Postgres 16 CI covers successful authoritative commit, exact lost-response retry, concurrent identical commit, stale-worker claim takeover, stale retry after takeover, missing STARTED boundary, runtime CAS conflict and direct audit-table mutation denial.
- Phase80 remains shadow/paper-only. Its SQL is still a draft contract outside `supabase/migrations`; at rollout freeze it must be converted using `supabase migration new` and rerun through the full Postgres suite before deployment.


## Phase 81 — Durable Post-Cancel Recovery Directive

- Brian files:
  - `brian2026/phase81_cancel_recovery_directive.py`
  - `brian2026/sql/phase81_cancel_recovery_directive.sql`
- This phase introduces no new alpha/trading signal. It turns a Phase78 `AFTER_START` cancel request into an immutable recovery obligation only after the already-started original cycle has an authoritative Phase60/67 `COMMITTED` state.
- Recovery semantics:
  - recovery never blindly flattens the account;
  - for each original cycle asset that actually carried ALLOWed non-reduce-only risk, Phase81 compares the authoritative Phase60 pre-cycle state with the current authoritative post-cycle head;
  - a safe automatic leg targets the exact pre-cycle position weight, preserving pre-existing exposure while removing only the additional exposure attributable to the cancelled cycle;
  - every automatic leg is structurally reduce-only: order direction opposes current exposure, target magnitude cannot exceed current magnitude, and the target cannot flip direction;
  - `ASSET_COOLDOWN` recovery is restricted to assets named in the exact cancel-risk receipt's blocked-asset evidence;
  - if the position path would require a direction flip or exposure increase to recreate the pre-cycle state, automatic recovery is refused as `MANUAL_REVIEW`;
  - if exposure is already at or below the pre-cycle level, the durable result is `NO_RECOVERY_REQUIRED`.
- Risk-state behavior:
  - Phase56's existing invariant is preserved: `HALTED` forbids submissions, including reductions;
  - therefore a valid rollback obligation under current `HALTED` risk becomes `WAIT_RISK_RELEASE`, never an auto-submit;
  - under current `ACTIVE` or `REDUCING`, structurally safe legs become `READY_REDUCE_ONLY`.
- Authority / concurrency behavior:
  - preparation uses the same Phase70 runtime advisory lock and validates current runtime lease/fence and exact expected runtime version;
  - the original Phase79 STARTED evidence must exist;
  - the current journal must prove the original cycle is `COMMITTED`;
  - the Phase60 `CYCLE_PROPOSED.before_state_id` and `RECONCILED_COMMIT.after_state_id` are used as rollback lineage;
  - if the current Phase60 head no longer equals the original cycle's commit state, preparation returns `HEAD_MOVED` instead of applying rollback assumptions to later account history;
  - exact concurrent/retry preparation is idempotent and returns the same immutable directive;
  - directive and event tables have RLS enabled, no direct anon/authenticated/service-role table grants, and append-only mutation guards.
- Python supervisor behavior:
  - Phase81 wraps the Phase80 claim-fenced authoritative execution result;
  - recovery is probed only after a STARTED cycle has an authoritative Phase80 checkpoint commit;
  - `WAIT_ORIGINAL_COMMIT` remains a non-terminal recovery obligation state while reconciliation finishes;
  - lease/version/head/evidence failures invalidate the local runtime copy and fail closed;
  - prepared outcomes are surfaced explicitly as `RECOVERY_READY`, `RECOVERY_WAIT_RISK_RELEASE`, `RECOVERY_MANUAL_REVIEW`, or `NO_RECOVERY_REQUIRED`.
- Real Postgres 16 CI covers pre-existing exposure preservation, HALTED wait behavior, asset-specific cooldown rollback, already-reduced/no-op recovery, sign-flip manual review, original-commit gating, head-moved rejection, no-cancel behavior, runtime-version conflict, idempotent retry, concurrent prepare and direct table-mutation denial.
- Phase81 remains shadow/paper-only. Its SQL is still a draft contract outside `supabase/migrations`; at rollout freeze it must be converted using `supabase migration new` and rerun through the full Postgres suite before deployment.


## Phase 82 — Recovery Claim Fencing

- Brian files:
  - `brian2026/phase82_recovery_claim_fencing.py`
  - `brian2026/sql/phase82_recovery_claim_fencing.sql`
- Phase82 adds no alpha, signal, or live-execution behavior. It makes a Phase81 recovery obligation single-owner work before any later recovery side effect is allowed.
- Claim semantics:
  - recovery ownership has its own worker token and monotonically increasing `claim_fencing_token`, independent of the original Phase77 execution claim;
  - the current Phase70 runtime lease/fence, runtime version and Phase60 head state must still match the immutable Phase81 directive;
  - `NO_RECOVERY_REQUIRED` and `MANUAL_REVIEW` are terminal/non-claimable outcomes;
  - current `HALTED` risk returns `WAIT_RISK_RELEASE` and creates no execution-ready recovery claim, preserving Phase56's rule that HALTED forbids submissions;
  - current `ACTIVE` or `REDUCING` risk may claim structurally safe Phase81 reduce-only recovery legs;
  - first owner receives claim fence 1; an exact same-worker retry returns `ALREADY_OWNED` with the same fence;
  - a second worker is `BLOCKED_ACTIVE` while the first claim is live;
  - after expiry, takeover returns `EXPIRED_RECOVERY` and increments the recovery claim fence so the stale worker cannot remain authoritative.
- Renewal semantics:
  - only the current worker/fence may renew;
  - runtime/head movement causes renewal loss;
  - if current risk becomes `HALTED`, renewal is refused as `RENEWAL_BLOCKED_RISK` instead of extending recovery execution authority;
  - ACTIVE/REDUCING may renew the same claim fence.
- Security:
  - recovery claim/event tables have RLS enabled and no direct anon/authenticated/service-role table grants;
  - privileged RPC functions explicitly revoke EXECUTE from PUBLIC/anon/authenticated and grant only service_role;
  - event history is append-only.
- Python behavior:
  - RPC responses are checked for exact runtime/cycle/runtime-fence/worker/claim-fence anchors;
  - a claimed recovery under HALTED risk is rejected as an invalid contract;
  - lease loss invalidates the local runtime and raises the lease error; head/directive/risk/evidence drift fails closed as stale runtime;
  - `MANUAL_REVIEW` and `NO_RECOVERY_REQUIRED` are never automatically claimed.
- Real Postgres 16 CI covers single-owner claim, concurrent two-worker exclusivity, same-worker idempotency, expired takeover with fence increment, stale-worker renewal rejection, HALTED initial wait, HALTED-after-claim renewal block, runtime/head movement, terminal non-claimable directives and direct table-mutation denial.
- Phase82 remains shadow/paper-only and performs no recovery execution side effect. Its SQL is still a draft contract outside `supabase/migrations`; at rollout freeze it must be converted using `supabase migration new` and rerun through the full Postgres suite before deployment.


## Phase 83 — Atomic Recovery STARTED Boundary

- Brian files:
  - `brian2026/phase83_atomic_recovery_start.py`
  - `brian2026/sql/phase83_atomic_recovery_start.sql`
- Phase83 introduces no new alpha and performs no paper/exchange side effect. It is the durable point-of-no-return marker immediately before later recovery work may begin.
- Atomic start semantics:
  - the current Phase70 runtime lease/fence is revalidated under the same runtime advisory lock used by the durable runtime;
  - the immutable Phase81 recovery directive must exist and contain non-empty recovery legs;
  - every recovery leg is revalidated as strictly reduce-only (positive reduction, opposing order direction, non-increasing target magnitude and internally consistent reduction amount);
  - the current Phase82 recovery worker/claim fencing token must still own an unexpired claim;
  - before the first STARTED row, runtime version and Phase60 head must still equal the Phase81 directive anchors;
  - current operational risk is re-read under the same lock; `HALTED` returns `WAIT_RISK_RELEASE` and no STARTED row, while `ACTIVE`/`REDUCING` may cross the boundary;
  - the immutable STARTED record copies exact recovery legs plus runtime/head/risk evidence.
- Idempotency / recovery:
  - exact lost-response retry by the same current claim returns `STARTED_ALREADY`;
  - an expired Phase82 claim may be taken over with a higher claim fence; if immutable STARTED evidence already exists, the valid takeover returns `STARTED_RESUME` and does not create a second STARTED row;
  - wrong worker, stale claim fence or expired claim returns `CLAIM_LOST`;
  - a first start after runtime/head drift returns `HEAD_MOVED`;
  - malformed recovery evidence fails closed as `EVIDENCE_INVALID`.
- Python behavior:
  - database STARTED responses must echo runtime/cycle/dispatch/cancel-risk/runtime-fence/recovery-claim-fence anchors;
  - started recovery legs must exactly match the Phase82 claimed directive legs;
  - lease/head/directive/risk/evidence failures fail closed; claim loss requires acquiring a fresh recovery claim without falsely marking the runtime stale;
  - HALTED remains a non-terminal wait and cannot cross STARTED.
- Real Postgres 16 CI covers first STARTED, exact retry, concurrent duplicate race, wrong worker/fence, HALTED after claim, runtime/head movement, malformed reduce-only evidence, expired-claim takeover/resume and direct STARTED-history mutation denial.
- Phase83 remains shadow/paper-only. Its SQL is draft/undeployed outside `supabase/migrations`; at rollout freeze it must be converted using `supabase migration new` and the full Postgres suite rerun.


## Phase 84 — Recovery Execution + Claim-Fenced Durable Checkpoint

- Brian files:
  - `brian2026/phase84_recovery_execution_checkpoint.py`
  - `brian2026/sql/phase84_recovery_execution_checkpoint.sql`
- Phase84 performs the first actual post-cancel recovery paper execution, but still remains strictly shadow/paper-only and never crosses into live execution.
- Execution path:
  - Phase83 STARTED remains mandatory;
  - the current Phase82 recovery claim is renewed before recovery simulation and current risk is re-read;
  - recovery is compiled only into Phase55 `RiskReductionIntent` objects;
  - Phase56 independently revalidates every leg as reduce-only under current `ACTIVE`/`REDUCING` risk;
  - Phase57 produces the execution cycle using explicit execution-market snapshots;
  - any denied leg, pending reversal, non-reduce-only receipt, missing execution receipt, non-FILLED simulation or side/target drift aborts Phase84 before the paper venue is touched;
  - the recovery cycle's full CYCLE_CREATED body is committed first through a recovery-claim-fenced Phase70 checkpoint;
  - the recovery claim stores `recovery_cycle_id`, progress runtime version, progress Phase60 head and progress checkpoint id so a crash/restart can resume authorized recovery progress without mistaking it for unrelated head movement;
  - the claim is renewed and current risk is rechecked again after durable write-ahead and immediately before Phase61 paper application;
  - a HALTED risk at that boundary leaves the durable recovery cycle at write-ahead only and returns a wait state with no paper side effect.
- Durable paper/reconciliation path:
  - Phase61 applies the already-validated reduce-only recovery cycle;
  - the existing local projection + Phase50 reconciliation + Phase60 authoritative ledger + Phase67 journal flow is reused unchanged;
  - intermediate `PAPER_APPLIED`, `LOCAL_PROJECTED`, `RECONCILIATION_REQUIRED`, `RECONCILED` or `ABORTED` progress can be checkpointed under the same current recovery claim fence;
  - the recovery claim becomes `COMPLETED` atomically only when the recovery cycle's journal stage is `COMMITTED` and the Phase60 head equals that recovery cycle's `RECONCILED_COMMIT.after_state_id`; Phase84 now stops at `RECOVERY_COMMITTED_PENDING_AUDIT` and leaves the recovery claim open for Phase85's authoritative quantity/exposure audit;
  - exact lost-response retry of the final atomic transaction returns `DUPLICATE_CURRENT`.
- Phase82 hardening introduced with Phase84:
  - recovery claims now carry nullable durable-progress anchors;
  - initial claims still anchor to the immutable Phase81 source runtime/head;
  - once Phase84 advances the runtime, claim/renew validation anchors to the last Phase84 progress checkpoint instead;
  - unrelated runtime/head movement still fails closed;
  - claim existence is stored in an explicit boolean rather than relying on PL/pgSQL `FOUND` after later risk queries.
- SQL validation:
  - recovery cycle assets must map one-to-one to Phase83 immutable recovery legs;
  - duplicate assets are rejected;
  - every item must have an allowed reduce-only receipt, a non-null projected target matching the Phase83 leg, and a fully FILLED execution receipt on the correct side;
  - malformed/missing JSON fields fail closed rather than passing through PostgreSQL three-valued NULL logic.
- Real Postgres 16 CI covers durable recovery write-ahead, progress-anchor resume through Phase82 renewal, final atomic claim completion, final lost-response retry, invalid/non-reduce-only recovery rejection, concurrent exact write-ahead commit, stale-worker rejection and direct event-table mutation denial.
- Phase84 SQL remains draft/undeployed outside `supabase/migrations`. At rollout freeze it must be converted with `supabase migration new` and the entire real-Postgres suite rerun before any deployment.


## Phase 85 — Authoritative Recovery Completion Audit

- Brian files:
  - `brian2026/phase85_recovery_completion_audit.py`
  - `brian2026/sql/phase85_recovery_completion_audit.sql`
- Phase85 is the only boundary that may turn a Phase84 terminal recovery checkpoint into a `COMPLETED` recovery claim.
- Evidence chain:
  - current Phase70 runtime lease/fence must still own the runtime;
  - the current runtime version/head/checkpoint must exactly equal the Phase84 recovery progress anchors stored on the Phase82 claim;
  - the recovery journal must end in `COMMITTED`;
  - an immutable Phase84 `RECOVERY_COMMITTED_PENDING_AUDIT` event must exist for the exact recovery cycle/checkpoint;
  - Phase83 STARTED recovery legs, Phase63 final paper checkpoint and Phase60 final authoritative state are all required.
- Quantity reconstruction:
  - Phase85 reads the final Phase63 `final_positions` quantity for each recovery asset;
  - it sums only fills whose `cycle_id` equals the exact recovery cycle and reconstructs pre-recovery quantity as `final_quantity - recovery_delta_quantity`;
  - the reconstructed pre-recovery paper direction must match the Phase83 current direction;
  - recovery fills must oppose existing exposure;
  - recovery fill quantity may not exceed the pre-recovery paper position;
  - final paper position may be flat or preserve the original direction, but may never flip;
  - final paper absolute quantity must be strictly smaller than reconstructed pre-recovery quantity.
- Phase60 cross-check:
  - the final authoritative position weight may be absent/zero or preserve the original direction;
  - an opposite-sign final authoritative weight fails certification even if raw paper quantity looked safe.
- Weight target handling:
  - exact target-weight equality is recorded as evidence but is not the safety gate because fees, slippage and mark-to-market equity can legitimately alter final portfolio-weight percentages;
  - the hard completion gate is actual paper quantity reduction without direction flip, backed by a fully reconciled Phase60 state.
- Completion:
  - passing audits are stored in append-only `brian_shadow_recovery_completion_certificates`;
  - the claim becomes `COMPLETED` atomically in the same transaction as certificate creation;
  - `completion_ref` remains the exact terminal Phase84 checkpoint id, so an exact Phase84 lost-response retry after certification is still recognized;
  - failed audits leave the claim open and emit immutable failure evidence instead of declaring success.
- Real Postgres 16 CI covers valid long reduction, valid short reduction, flattening, over-reduction/direction flip, missing recovery fills, Phase60 sign mismatch, checkpoint/head drift, claim-progress drift, exact duplicate certification, concurrent certification and direct certificate-history mutation denial.
- Phase85 remains shadow/paper-only. Its SQL is draft/undeployed outside `supabase/migrations`; at rollout freeze it must be converted using `supabase migration new` and the entire real-Postgres suite rerun before deployment.


## Phase 86 — Unresolved-Recovery Admission Interlock

- Brian files:
  - `brian2026/phase86_recovery_admission_interlock.py`
  - `brian2026/sql/phase86_recovery_admission_interlock.sql`
- Phase86 adds no alpha or execution strategy. It closes the race in which a new normal governed cycle could be authorized or dispatched while a Phase78 `AFTER_START` recovery obligation is still unresolved.
- Barrier source / resolution:
  - the database-authoritative barrier begins as soon as an `AFTER_START` cancel request exists;
  - `BEFORE_EXECUTION` cancels do not create a recovery admission barrier;
  - the barrier remains closed through `READY_REDUCE_ONLY`, `WAIT_RISK_RELEASE`, `MANUAL_REVIEW`, missing/not-yet-prepared Phase81 recovery, Phase84 pending-audit recovery, and Phase85 audit failure;
  - the barrier opens only when an immutable Phase85 recovery-completion certificate exists for the original cycle, or Phase81 proves `NO_RECOVERY_REQUIRED`.
- Phase75 authorization interlock:
  - the existing Phase75 SQL implementation is renamed once to a private base RPC;
  - the public/service-role Phase75 RPC name becomes a Phase86 wrapper under the same runtime advisory transaction lock;
  - a new candidate authorization is rejected as `RECOVERY_BARRIER` before any Phase75/Phase70 write-ahead mutation;
  - exact already-persisted authorization retries are delegated to the original implementation so lost-response idempotency remains readable;
  - lease/version validation precedence remains owned by the original Phase75 implementation.
- Phase76 dispatch interlock:
  - the existing Phase76 submit implementation is likewise retained as a private base RPC behind a Phase86 wrapper;
  - an authorization created before the barrier but not yet dispatched is rejected as `RECOVERY_BARRIER` on first dispatch;
  - Python immediately marks that still-pre-paper CYCLE_CREATED candidate `ABORTED` through Phase75 and persists the abort checkpoint, so it cannot remain an active journal cycle;
  - an already-persisted dispatch remains readable as an exact duplicate retry after a later barrier;
  - no paper/local side effect occurs for a dispatch blocked by the recovery barrier.
- Phase81 foreign-active-cycle hardening:
  - before freezing an immutable recovery directive, Phase81 now checks the durable journal for any other non-terminal cycle;
  - a foreign `CYCLE_CREATED` candidate is quarantined by marking it `ABORTED`, advancing the authoritative runtime version, then retrying Phase81 against that new version/head;
  - a foreign cycle that has already crossed into `PAPER_APPLIED` or a later side-effected stage is never auto-aborted; recovery remains in `FOREIGN_CYCLE_ACTIVE` wait until that cycle is reconciled/settled safely;
  - this prevents a pre-authorized candidate from invalidating the Phase81 source runtime/head immediately after the recovery directive is created.
- Security / idempotency:
  - the original Phase75/76 base functions are not executable by public/anon/authenticated/service_role after wrapping;
  - only the wrapper functions retain service-role execute permission;
  - recovery-admission events are append-only, RLS-enabled and have no direct service-role table write grant;
  - exact authorization/dispatch duplicates remain idempotent and do not get hidden by a later barrier.
- Python observability:
  - `RecoveryAdmissionInterlockStore` exposes the DB-authoritative `OPEN` vs `RECOVERY_BARRIER` state with exact original-cycle/cancel-receipt lineage validation;
  - Phase75 and Phase76 client contracts explicitly reject any impossible barrier response that claims authorization/submission succeeded.
- Real Postgres 16 CI covers open/before-execution admission, unresolved AFTER_START barrier, NO_RECOVERY_REQUIRED/certificate resolution, MANUAL_REVIEW persistence, new authorization blocking without runtime advance, Phase75 duplicate retry, pre-authorized dispatch blocking, Phase76 duplicate retry, reopened authorization, concurrent blocked authorizations, Phase81 foreign CYCLE_CREATED refusal and direct admission-event mutation denial.
- Phase86 remains shadow/paper-only. Its SQL is draft/undeployed outside `supabase/migrations`; at rollout freeze its wrapper/rename operations must be converted into the official ordered migration set and the full real-Postgres suite rerun before deployment.


## Phase 87 — Durable Recovery Backlog / Restart Resume View

- Brian files:
  - `brian2026/phase87_recovery_restart_resume.py`
  - `brian2026/sql/phase87_recovery_restart_resume.sql`
- Phase87 adds no alpha, signal or live execution. It makes unresolved recovery work discoverable after process restart without waiting for the original governed signal to reappear.
- Database backlog semantics:
  - only unresolved Phase78 `AFTER_START` cancels are considered;
  - a Phase85 completion certificate removes the item from the backlog;
  - a Phase81 `NO_RECOVERY_REQUIRED` directive also resolves the item;
  - work is selected deterministically by oldest cancel request, then risk version/receipt;
  - the returned row is anchored to the current Phase70 runtime version/checkpoint/head and exposes exact Phase81 directive, Phase82 claim, Phase83 STARTED, Phase84 recovery progress and terminal-event evidence.
- Restart work-state classification:
  - `NEEDS_DIRECTIVE`: cancel exists but no durable Phase81 directive;
  - `MANUAL_REVIEW`: Phase81 refused automatic recovery;
  - `NEEDS_CLAIM` / `CLAIM_EXPIRED`: recovery ownership must be acquired or taken over;
  - `NEEDS_START`: claim exists but Phase83 STARTED evidence does not;
  - `STARTED_NEEDS_EXECUTION`: Phase83 STARTED exists but no durable recovery cycle exists;
  - `RECOVERY_PROGRESS`: Phase84 recovery cycle exists and must be resumed from its durable journal stage;
  - `NEEDS_AUDIT`: recovery journal is COMMITTED and exact immutable Phase84 `RECOVERY_COMMITTED_PENDING_AUDIT` evidence exists, so Phase85 certification is the next boundary;
  - `COMPLETED_WITHOUT_CERTIFICATE`: fail-closed visibility for an inconsistent/legacy claim state that must never be silently treated as resolved;
  - `IDLE`: no unresolved recovery work remains.
- Python contract:
  - validates all runtime/cycle/dispatch/cancel/checkpoint/head hashes and positive versions/fences;
  - rejects impossible work-state combinations (for example progress without STARTED evidence or audit without terminal Phase84 evidence);
  - maps every DB state to one explicit operational action: prepare, acquire/take over claim, mark STARTED, execute/resume, audit, manual review, fail closed, or none.
- Real Postgres 16 CI covers idle, directive preparation backlog, claim acquisition, expired takeover, STARTED transition, recovery progress, terminal audit handoff, manual review visibility, NO_RECOVERY_REQUIRED resolution, completion-certificate resolution, inconsistent completed-without-certificate visibility, deterministic oldest-work ordering and reader permissions.
- Phase87 remains shadow/paper-only and read-oriented. Its SQL is draft/undeployed outside `supabase/migrations`; at rollout freeze it must be converted with `supabase migration new` and rerun through the complete Postgres suite before deployment.


## Phase 88 — Restart Recovery Orchestrator

- Brian file:
  - `brian2026/phase88_recovery_restart_orchestrator.py`
- Phase88 adds no alpha, order strategy, SQL persistence path or live execution. It is the process-restart orchestrator that consumes one Phase87 DB-authoritative backlog item and reuses the existing Phase81→85 recovery boundaries instead of creating a parallel recovery engine.
- Authority anchoring:
  - before any action, Phase87 runtime version, checkpoint id and Phase60 head state must exactly match the local Phase71 persisted runtime supervisor;
  - any mismatch invalidates the local runtime and requires authoritative reload;
  - one call processes at most the oldest Phase87 backlog item, preventing an unbounded restart loop from monopolizing the worker.
- Routing:
  - `IDLE` performs no recovery action;
  - `MANUAL_REVIEW` stays manual and is never automatically claimed;
  - `COMPLETED_WITHOUT_CERTIFICATE` fails closed as inconsistent legacy/state evidence;
  - `NEEDS_AUDIT` goes directly to the existing Phase85 certification boundary without reacquiring or replaying recovery execution;
  - `NEEDS_DIRECTIVE` uses Phase81 with the current runtime lease/version; pre-paper foreign `CYCLE_CREATED` work may be quarantined only when an explicit existing Phase75-compatible aborter is supplied, otherwise it remains a wait state;
  - all executable states acquire/take over the existing Phase82 claim, cross the existing idempotent Phase83 STARTED boundary, and call Phase84's direct `execute_started_recovery` resume core;
  - a terminal Phase84 result is immediately passed to Phase85 certification.
- Safety / idempotency:
  - another worker's unexpired Phase82 claim returns a wait outcome and is never stolen;
  - expired claims use Phase82's existing fencing-token takeover semantics;
  - Phase83 `STARTED_ALREADY` / `STARTED_RESUME` and Phase84 durable journal replay semantics are reused;
  - lease loss, runtime/head/evidence drift and audit invariant failures retain the existing fail-closed behavior;
  - after a successful Phase85 certificate, Phase88 re-reads Phase87 and requires the exact certified backlog item to disappear; another distinct unresolved item may remain for a later invocation.
- Red-team unit coverage includes stale Phase87 anchors, no-recovery resolution, manual review, active-owner blocking, expired takeover/resume, terminal Phase84→Phase85 completion, direct NEEDS_AUDIT certification, certificate/backlog inconsistency, completed-without-certificate fail-closed behavior, audit failure evidence and optional foreign pre-paper quarantine.
- Phase88 remains hard shadow/paper-only and introduces no deployment migration of its own.


## Phase 89 — Bounded Recovery Startup Gate

- Brian file:
  - `brian2026/phase89_recovery_startup_gate.py`
- Phase89 adds no alpha, SQL or live execution. It is the bounded startup/worker gate that runs Phase88 recovery work before normal shadow work may resume.
- Bounded drain semantics:
  - one invocation processes at most a caller-specified positive number of Phase87 backlog items;
  - only terminal resolution outcomes (`RECOVERY_COMPLETED`, `NO_RECOVERY_REQUIRED`) may continue to another item in the same invocation;
  - `WAIT_*`, `MANUAL_REVIEW_REQUIRED`, reconciliation/mark dependencies, foreign-cycle waits and any future non-terminal outcome stop immediately instead of spinning on an external dependency;
  - `IDLE` stops the drain without consuming the item budget.
- Final admission semantics:
  - after the bounded Phase88 work, Phase89 always re-reads the existing Phase86 DB admission barrier;
  - normal work is released only when Phase86 is `OPEN` AND the worker's final outcome is terminal/IDLE;
  - a new AFTER_START cancel appearing between Phase87's IDLE read and the final Phase86 read is classified as `RECOVERY_BARRIER_APPEARED` and keeps normal work closed;
  - an OPEN Phase86 read never overrides a non-terminal Phase88 result from the same invocation;
  - exhausting the item budget while Phase86 remains blocked returns `RECOVERY_BUDGET_EXHAUSTED`, never READY.
- Red-team unit coverage includes IDLE/open release, multiple resolved backlog items, wait/manual stop behavior, the Phase87-IDLE→Phase86-barrier race, bounded backlog exhaustion, OPEN-admission/nonterminal disagreement and invalid budget rejection.
- Phase89 remains hard shadow/paper-only and introduces no migration of its own.


## Phase 90 — Recovery Runtime Assembly

- Brian file:
  - `brian2026/phase90_recovery_runtime_assembly.py`
- Phase90 adds no alpha, SQL, scheduler or live execution. It centralizes construction of the Phase81–89 restart-recovery stack so deployment workers cannot accidentally cross-wire persistence/authority components.
- Assembly semantics:
  - one caller-supplied RPC transport is shared by the Phase81 directive store, Phase82 claim store, Phase83 STARTED store, Phase84 checkpoint store, Phase85 audit store, Phase86 admission reader and Phase87 backlog reader;
  - one valid Phase71 persisted runtime supervisor is shared by Phase84 and Phase88;
  - the exact Phase82 claim store is reused by Phase84 direct recovery execution and Phase88 restart orchestration;
  - Phase88 is wired to the exact Phase81/82/83/84/85/87 components created by the assembly;
  - Phase89 is wired to that exact Phase88 orchestrator and Phase86 admission store;
  - an optional existing Phase75-compatible foreign-cycle aborter may be supplied only to the Phase88 restart orchestrator.
- Fail-closed construction:
  - non-callable RPC transports are rejected before stack construction;
  - stale runtime supervisors are rejected;
  - runtime supervisors must expose a non-empty runtime id;
  - `RecoveryRuntimeStack` validates identity-level wiring across Phase84, Phase88 and Phase89 and rejects cross-wired runtime/claim/store authorities.
- Unit coverage proves shared RPC identity, shared runtime authority, exact store/supervisor identity, optional quarantine wiring, stale-runtime rejection, invalid transport/runtime rejection and cross-wired Phase84 rejection.
- Phase90 remains hard shadow/paper-only and introduces no deployment migration of its own.


## Phase 91 — Supabase Recovery RPC Transport

- Brian file:
  - `brian2026/phase91_supabase_rpc_transport.py`
- Phase91 adds no alpha, trading logic, SQL or live execution. It provides the fail-closed backend transport used to call the exact Phase70 bootstrap/lease plus Phase81–87 recovery Postgres RPC surface through Supabase PostgREST.
- Current Supabase API-key migration rules were re-verified before implementation:
  - backend workers prefer `SUPABASE_SECRET_KEY` / `sb_secret_...`;
  - hosted `SUPABASE_SECRET_KEYS` JSON with the `default` key is also supported;
  - legacy `SUPABASE_SERVICE_ROLE_KEY` remains migration-compatible;
  - publishable keys are rejected for the privileged recovery worker;
  - modern secret keys are sent only in the `apikey` header, never as `Authorization: Bearer`.
- Transport security:
  - remote project URLs must use HTTPS; plain HTTP is accepted only for localhost;
  - secret keys are never placed in URLs, exception messages or diagnostics;
  - redirects are not followed, preventing privileged key forwarding to another origin;
  - only the five known Phase70 runtime bootstrap/lease RPCs plus the eight known Phase81–87 recovery RPCs are callable;
  - responses must be 2xx JSON objects; malformed JSON, arrays/scalars, 204s and database/API errors fail closed;
  - transport/timeouts are surfaced without blind HTTP retries because replay is delegated to the durable Phase81–89 idempotency/restart protocol.
- Environment configuration supports explicit positive connect/read/write/pool timeouts and rejects malformed/zero values.
- Red-team tests cover secret-key-only headers, legacy migration compatibility, modern-key precedence, Edge secret-key JSON, publishable-key rejection, HTTPS enforcement, RPC allowlisting, sanitized database errors, redirect blocking, no blind timeout retry, malformed response rejection and timeout configuration.
- Phase91 remains hard shadow/paper-only and introduces no migration of its own.


## Phase 92 — Lease-Owned Recovery Worker Session

- Brian files:
  - `brian2026/phase92_recovery_worker_session.py`
  - Phase71 startup lease cleanup hardening in `brian2026/phase71_persisted_runtime_supervisor.py`
- Phase92 adds no alpha, SQL, market strategy or live execution. It is the backend session/bootstrap boundary that connects the real Phase91 Supabase transport to Phase70 durable ownership, Phase71 runtime restoration, Phase90 stack assembly and the Phase89 startup recovery gate.
- Runtime ownership:
  - one Phase70 lease is acquired for the configured runtime id with a per-process owner token by default;
  - lease TTL is bounded to 10–300 seconds;
  - the Phase71 durable checkpoint is restored before recovery assembly;
  - a fresh runtime is never invented implicitly: callers must explicitly supply `initial_runtime` when the durable database head does not yet exist;
  - the same RPC transport is reused for Phase70 and all recovery stores.
- Resource lifecycle:
  - Phase71 now best-effort releases a newly acquired owner+fence lease if checkpoint load, bootstrap validation or first checkpoint persistence fails;
  - Phase92 releases the supervisor lease if stack assembly fails after supervisor construction;
  - owned Phase91 HTTP transport is closed on every bootstrap failure;
  - session `close()` is idempotent and releases the owner+fence-gated lease before closing the HTTP transport;
  - the one-shot helper wraps the complete startup gate in the same cleanup boundary, so gate exceptions do not strand a lease.
- Environment bootstrap:
  - `BRIAN_RUNTIME_ID` is mandatory;
  - `BRIAN_RECOVERY_OWNER_TOKEN` is optional; when absent a unique per-process token is generated so independent workers never silently share one fencing identity;
  - `BRIAN_RUNTIME_LEASE_SECONDS` defaults to 60 and is restricted to 10–300.
- Red-team coverage includes normal gate delegation/cleanup, idempotent close, gate exceptions, release errors with transport cleanup, stack-assembly failure cleanup, unsafe lease bounds, missing runtime id, generated process owner identity, one-shot failure cleanup, plus Phase71-specific lease release on missing initial runtime, checkpoint-load failure and bootstrap commit conflict.
- Phase92 remains hard shadow/paper-only and introduces no migration of its own.


## Phase 93 — Machine-Readable Recovery Worker Entrypoint

- Brian file:
  - `brian2026/phase93_recovery_worker_entrypoint.py`
- Phase93 adds no alpha, SQL, scheduler or live execution. It is the executable one-shot backend entrypoint above the Phase92 lease-owned session.
- Invocation / input contract:
  - runnable as `python -m brian2026.phase93_recovery_worker_entrypoint`;
  - recovery market/risk/mark evidence is supplied as one JSON object through stdin or an explicit `--input` file;
  - supported input keys are exactly `markets`, `risk_limits_by_asset` and `marks`; unknown fields fail closed;
  - market snapshots are converted into the existing Phase46 order-book model and must preserve non-crossed bid/ask ordering;
  - risk limits are converted into the existing Phase56 instrument limits; marks must be finite/positive;
  - empty input is permitted only as an empty evidence set, so an IDLE probe can succeed while any actual recovery that needs market/mark evidence still fails through the existing Phase56/57/84 contracts.
- Worker controls:
  - `max_items` is bounded to 1–32;
  - recovery claim TTL is bounded to 10–300 seconds;
  - recovery intent TTL is bounded to 10–900 seconds;
  - an explicit recovery worker token may be supplied, otherwise a process-unique Phase93 token is generated;
  - no initial Phase70 runtime is implicitly invented; the Phase92/71 bootstrap still requires an existing durable head unless the caller explicitly supplies an initial runtime through the lower-level API.
- Machine output:
  - success/nonterminal results emit exactly one bounded JSON status line containing runtime id, Phase89 status, admission status/barrier identity, processed-item count and step outcomes;
  - exit 0 means `READY_FOR_NORMAL_WORK`;
  - exit 20 means recovery remains blocked/nonterminal;
  - exit 21 means manual review is required;
  - exit 22 means the bounded recovery budget was exhausted;
  - exit 30 means invocation/input/config parsing failed;
  - exit 40 means lease/transport/runtime/durable recovery execution failed.
- Error safety:
  - CLI parser errors are converted to JSON exit-30 responses instead of unstructured `argparse` termination;
  - runtime `ValueError` and other post-bootstrap failures remain worker errors rather than being mislabeled as input errors;
  - modern, legacy and hosted Supabase secret values are redacted from worker diagnostics before stderr JSON is emitted;
  - Phase92 remains responsible for lease + HTTP cleanup on every normal/exceptional exit.
- Red-team tests cover exact market/risk/mark parsing, malformed/crossed books, invalid marks, one-line READY output, block/manual/budget exit-code separation, safe empty-TTY IDLE probes, invalid JSON/bounds, runtime-vs-input error classification, secret redaction, generated worker-token identity, file-vs-stdin precedence and machine-readable unknown-argument errors.
- Phase93 remains hard shadow/paper-only and introduces no migration of its own.


## Phase 94 — Public Binance Spot Recovery Evidence

- Brian file:
  - `brian2026/phase94_binance_spot_recovery_evidence.py`
- Phase94 adds no alpha, SQL, private exchange API, account access or live execution. It is a read-only public Binance Spot market-evidence adapter for the Phase56/57/84 shadow recovery path.
- Upstream contract:
  - only public market-data GET routes are allowlisted: `/api/v3/depth` and `/api/v3/exchangeInfo`;
  - the market-data-only `data-api.binance.vision` host is preferred, with the existing public Binance REST hosts used only as bounded failover for transport/5xx availability failures;
  - no API key, Authorization header, signed request, account route or order route is accepted or constructed;
  - HTTP redirects are not followed;
  - HTTP 429/418 fails immediately instead of hopping across hosts, preserving Binance rate-limit/backoff semantics.
- Asset / instrument evidence:
  - support is intentionally restricted to uppercase Binance Spot USDT symbols used by the current crypto recovery model;
  - exchangeInfo must contain exactly the requested symbol, status `TRADING`, and Spot trading must be allowed;
  - `PRICE_FILTER.tickSize` is mandatory;
  - current `NOTIONAL.minNotional` is preferred, with legacy `MIN_NOTIONAL.minNotional` accepted as compatibility fallback;
  - the resulting values are converted into the existing Phase56 `InstrumentRiskLimits` and Phase57 `ExecutionMarketInput`, not a new risk/execution model.
- Point-in-time depth evidence:
  - bids/prices/quantities must be finite/positive, bids strictly descending and asks strictly ascending;
  - crossed/locked books fail closed;
  - the configurable top-of-book spread ceiling defaults to 30 bps;
  - the response receive timestamp becomes the existing Phase46 `OrderBookSnapshot` timestamp;
  - mid price becomes the shadow recovery mark/reference price;
  - Binance `lastUpdateId` is retained as unsigned source evidence.
- Operational bounds:
  - requested assets are deduplicated and deterministically sorted;
  - empty asset input is a valid no-op evidence bundle;
  - asset count is bounded (default 8, maximum 32);
  - depth limits are restricted to Binance-supported bounded values;
  - the provider owns/closes its HTTP client only when it created that client.
- Red-team unit coverage includes exact safe-route/header behavior, NOTIONAL precedence and MIN_NOTIONAL compatibility, deterministic asset handling, forbidden order routes, 429/418 no-host-hop behavior, bounded 5xx failover, redirect rejection, symbol/status/Spot/filter validation, malformed/crossed/out-of-order/wide books, depth/update-id validation, asset/depth budgets and all-host failure.
- Phase94 remains hard shadow/paper-only. It deliberately does not wire itself into Phase93 automatically; that integration is a separate boundary so restart states that do not need market evidence do not make unnecessary external calls.


## Phase 95 — Causal Auto-Evidence Recovery Worker

- Brian file:
  - `brian2026/phase95_auto_binance_recovery_worker.py`
- Phase95 adds no alpha, SQL, scheduler or live execution. It binds the Phase94 public Binance Spot evidence provider to the existing Phase87→89 restart-recovery path without allowing market evidence from one backlog item to leak into another.
- One-item authority boundary:
  - each Phase95 invocation processes at most one Phase87 recovery identity through Phase89 with `max_items=1`;
  - if another unresolved recovery remains, a later invocation must acquire a fresh evidence set for that item's own immutable Phase81 legs;
  - this intentionally trades a small amount of extra public market-data I/O for strict evidence/authority isolation.
- Lazy evidence acquisition:
  - Phase87 `IDLE`, `NEEDS_AUDIT`, `MANUAL_REVIEW` and other non-execution states do not construct or call the Binance provider;
  - execution-capable states preflight the existing Phase81 directive under the same Phase92 lease;
  - Phase81 `NO_RECOVERY_REQUIRED` / `MANUAL_REVIEW` or non-prepared wait states do not fetch market data;
  - only prepared `READY_REDUCE_ONLY` / `WAIT_RISK_RELEASE` directives with immutable recovery legs trigger Phase94 evidence collection;
  - the requested Binance symbols are exactly the sorted Phase81 recovery-leg asset ids and the returned evidence set must match exactly.
- Causality:
  - Phase95 freezes one recovery decision timestamp before any external Binance request;
  - every Phase94 snapshot used by the existing Phase46/57 simulator must be observed at or after that decision time;
  - the decision timestamp is passed unchanged into Phase89/88/84 as the recovery execution observation time, preventing post-fetch timestamps from being backdated into the decision.
- External-I/O race protection:
  - after Binance evidence collection, Phase95 re-reads the Phase87 backlog before any recovery gate execution;
  - a legitimate Phase81 preflight state transition such as `NEEDS_DIRECTIVE → NEEDS_CLAIM` is allowed only when original cycle, dispatch and cancel-risk receipt identity remain unchanged;
  - backlog identity change, runtime-version drift, checkpoint drift or Phase60 head drift invalidates the local Phase71 supervisor and fails closed;
  - provider exceptions close the provider context and never enter Phase89.
- Lifecycle:
  - the env one-shot helper owns no new lease logic; it uses the Phase92 session context so Supabase transport/runtime lease cleanup still occurs on evidence-provider or recovery failures.
- Red-team unit coverage includes zero-I/O IDLE/audit paths, exact leg-asset collection, causal decision-before-snapshot enforcement, NEEDS_DIRECTIVE durable transition tolerance, NO_RECOVERY_REQUIRED no-I/O behavior, evidence asset mismatch, backlog identity/runtime/head races, provider cleanup/failure, non-finite clocks and Phase92 session cleanup on provider failure.
- Phase95 remains hard shadow/paper-only and introduces no migration of its own.


## Phase 96 — Machine Auto-Recovery Entrypoint

- Brian file:
  - `brian2026/phase96_auto_recovery_entrypoint.py`
- Phase96 adds no alpha, SQL, scheduler or live execution. It exposes the Phase95 one-item causal auto-evidence worker as a machine-readable backend command with no market JSON input requirement.
- Invocation / controls:
  - runnable as `python -m brian2026.phase96_auto_recovery_entrypoint`;
  - Phase92 still owns the Supabase runtime lease/session lifecycle and Phase95 still limits each invocation to one Phase87 backlog identity;
  - recovery claim TTL is bounded to 10–300 seconds and recovery intent TTL to 10–900 seconds;
  - Binance depth size is restricted to the Phase94 allowlist;
  - public-market timeout is bounded to 0.5–20 seconds, spread ceiling to 0.1–500 bps, and requested recovery assets to 1–32;
  - CLI values override the corresponding `BRIAN_RECOVERY_*` environment settings;
  - an explicit worker token may be supplied, otherwise a process-unique `phase96-...` token is generated.
- Public evidence construction:
  - Phase96 constructs a lazy Phase94 provider factory with the bounded market settings and the same injected clock used by Phase95;
  - Phase95 decides whether evidence is needed at all, so IDLE/audit/manual/wait paths do not make unnecessary public Binance calls;
  - raw order-book snapshots are never emitted in the machine status payload; output contains only bounded evidence asset ids/timestamps and recovery/admission metadata.
- Exit/status contract:
  - exit 0: Phase89 says `READY_FOR_NORMAL_WORK`;
  - exit 20: recovery remains blocked/non-terminal;
  - exit 21: manual review is required;
  - exit 22: one-item recovery budget was consumed while another recovery barrier remains;
  - exit 30: CLI/environment configuration failed validation;
  - exit 40: lease/Supabase/public-market/runtime/recovery processing failed.
- Error safety:
  - unknown CLI arguments and invalid bounds are returned as one JSON `INPUT_ERROR`, not unstructured argparse termination;
  - runtime failures remain `WORKER_ERROR`;
  - configured modern, legacy and hosted Supabase secrets plus generic modern `sb_secret_*` forms are redacted from stderr diagnostics.
- Red-team unit coverage includes exact CLI/env forwarding, Phase94 provider configuration, all exit-code classes, bounded evidence metadata output, invalid argument/bound enforcement, secret redaction, generated worker identity and CLI-over-env precedence.
- Phase96 remains hard shadow/paper-only and introduces no migration of its own.

## Phase 97 — Bounded Auto-Recovery Drain

- Brian file:
  - `brian2026/phase97_bounded_auto_recovery_drain.py`
- Phase97 adds no alpha, SQL, scheduler or live execution. It turns the Phase95 one-item causal worker into a bounded multi-item startup drain while preserving one fresh market-evidence decision per recovery identity.
- Authority and evidence isolation:
  - one Phase92 session and one Phase70 lease remain owned across the bounded drain;
  - each backlog identity is still processed by a separate Phase95 invocation with its own decision timestamp and Phase94 provider call when market evidence is actually required;
  - market evidence from one recovery item is never reused for the next item;
  - the same recovery worker identity is reused only as process ownership, while durable original-cycle/claim/start/checkpoint identities remain item-specific.
- Retry/stop semantics:
  - Phase89 `RECOVERY_BUDGET_EXHAUSTED`, `RECOVERY_BACKLOG_REMAINS` and the IDLE→barrier race `RECOVERY_BARRIER_APPEARED` may consume another bounded Phase95 attempt;
  - WAIT/manual/reconciliation/other non-terminal outcomes stop immediately instead of spinning on an external dependency;
  - total Phase95 attempts and total processed recovery items are bounded to 1–32.
- Final handoff hardening:
  - a Phase89 `READY_FOR_NORMAL_WORK` result is re-read through the exact Phase86 admission store before Phase97 returns READY;
  - if a new AFTER_START barrier appears after the Phase89 ready read, normal work remains closed and Phase97 spends another bounded fresh-evidence attempt when budget remains;
  - Phase86's database authorization/dispatch wrappers remain the final normal-cycle interlock for any barrier that appears after Phase97 returns.
- Fail-closed wiring:
  - closed sessions, stale Phase71 supervisors, missing Phase86 admission readers and cross-runtime Phase95 receipts are rejected;
  - Phase97 remains hard shadow/paper-only and introduces no migration of its own.
- Red-team unit coverage includes immediate IDLE release, multi-item drain on one session, fresh second attempt after the ready→barrier race, non-terminal stop behavior, bounded exhaustion, cross-runtime rejection, stale/closed sessions, invalid budgets and missing Phase86 handoff authority.

## Phase 98 — Bounded Auto-Recovery Machine Entrypoint

- Brian file:
  - `brian2026/phase98_bounded_auto_recovery_entrypoint.py`
- Phase98 adds no alpha, SQL, scheduler or live execution. It exposes Phase97 as a single machine-readable startup command that opens one Phase92 session, keeps one Phase70 lease for the complete bounded drain and closes both lease/transport on every exit.
- Invocation / bounded controls:
  - runnable as `python -m brian2026.phase98_bounded_auto_recovery_entrypoint`;
  - recovery drain budget is configurable through `--max-items` / `BRIAN_RECOVERY_MAX_ITEMS` and bounded to 1–32;
  - claim TTL, recovery intent TTL, Binance depth, spread ceiling, timeout and asset budget preserve the existing Phase94/96 bounds;
  - CLI values override environment values and a process-unique `phase98-...` worker identity is generated when none is supplied.
- Runtime handoff:
  - Phase92 opens/restores the authoritative durable runtime once;
  - Phase97 invokes Phase95 separately per backlog identity, so market evidence remains causal and item-specific while lease ownership stays continuous;
  - Phase97's final Phase86 re-read must be OPEN before exit 0 is emitted;
  - Phase86's database authorization/dispatch wrappers remain the final barrier against a recovery obligation that appears after the command returns.
- Machine output:
  - exactly one bounded JSON status line is emitted on normal completion;
  - output includes attempt count, processed count, final Phase86 admission, per-attempt recovery status/outcomes and bounded evidence asset/timestamp metadata;
  - raw depth books and order-book snapshots are not emitted;
  - exit 0 = READY, 20 = blocked/non-terminal, 21 = manual review, 22 = bounded drain exhausted, 30 = input/config error, 40 = runtime/transport/evidence/recovery worker failure.
- Error/resource safety:
  - CLI parser failures are machine-readable JSON;
  - modern, legacy and hosted Supabase secrets are redacted from worker diagnostics;
  - the Phase92 context manager always closes the runtime lease/session when Phase97 returns or raises.
- Red-team unit coverage includes single-session READY, env/CLI precedence, provider bounds, blocked/manual/budget exit classes, bounded evidence-only output, invalid configuration rejection, secret redaction with session cleanup and generated process worker identity.
- Phase98 remains hard shadow/paper-only and introduces no migration of its own.

## Phase 99 — Recovery-Guarded Normal Shadow Handoff

- Brian file:
  - `brian2026/phase99_recovery_guarded_shadow_handoff.py`
- Phase99 adds no alpha, SQL, scheduler or live execution. It is the same-session authority bridge from a successful Phase97 recovery drain into the existing Phase73/75–80 governed shadow execution path.
- Normal shadow stack assembly:
  - one caller-supplied RPC transport is shared by the Phase73 operational-risk store, Phase75 atomic write-ahead store, Phase76 outbox, Phase77 execution claims, Phase78 kill-switch, Phase79 STARTED store and Phase80 claim-fenced checkpoint store;
  - the exact Phase71 runtime supervisor owned by the Phase92 recovery session is reused by the Phase75→80 stack;
  - identity-level validation rejects cross-wired risk, outbox, claim, STARTED, checkpoint or kill-switch authorities.
- Recovery handoff admission:
  - a Phase99 handoff cannot be created unless the supplied Phase97 receipt belongs to the same runtime and is `READY_FOR_NORMAL_WORK` with final Phase86 admission `OPEN`;
  - closed Phase92 sessions and stale Phase71 supervisors fail closed;
  - the exact Phase86 admission reader already assembled in the Phase90 recovery stack is reused by the handoff.
- Per-cycle safety:
  - before every governed shadow cycle, Phase99 re-reads Phase86 under the still-owned Phase70 lease/session;
  - a current recovery barrier blocks before the normal Phase75–80 execution path is called;
  - if a barrier appears after the Phase99 pre-read, the existing Phase86 transactional wrappers around Phase75 authorization and Phase76 dispatch remain the final database interlock;
  - downstream execution failures propagate without emitting a false successful handoff receipt.
- Shadow boundary:
  - non-shadow or live-enabled governed inputs are rejected before admission/execution;
  - claim TTL is bounded to 10–300 seconds and worker/source/timestamp inputs are validated;
  - successful receipts require both an OPEN admission and a hard shadow-only Phase80 execution result.
- Red-team unit coverage includes exact Phase73/75–80 authority assembly, same-runtime READY handoff, pre-execution Phase86 blocking, non-ready recovery rejection, cross-runtime/cross-authority rejection, stale/closed sessions, missing admission authority, live/non-shadow rejection, input bounds and propagation of transactional barrier races.
- Phase99 remains hard shadow/paper-only and introduces no migration of its own.

## Phase 100 — Recovery-First Shadow Worker Session

- Brian file:
  - `brian2026/phase100_recovery_first_shadow_worker.py`
- Phase100 adds no alpha, SQL, scheduler or live execution. It is the lifecycle object that keeps Phase97 recovery and Phase99 normal shadow execution under one Phase92 session instead of closing/reacquiring runtime authority between startup and work.
- Lifecycle / authority:
  - one Phase92 session owns/restores the Phase70/71 runtime authority;
  - Phase97 is run first on that exact session;
  - Phase99 is constructed only after Phase97 returns `READY_FOR_NORMAL_WORK`;
  - normal governed shadow cycles are rejected until that handoff exists;
  - once recovery has released normal work, the same startup gate is not rerun inside the worker session.
- Blocked recovery behavior:
  - a non-ready Phase97 result publishes a non-ready Phase100 startup receipt and creates no Phase99 handoff;
  - the caller may retry the recovery gate later on the same still-owned session, allowing fresh evidence and barrier state to be evaluated without changing runtime authority;
  - cross-runtime Phase97 receipts fail closed.
- Normal work:
  - `process_governed_cycle` delegates only through the Phase99 handoff, so every cycle retains Phase86 pre-admission plus the transactional Phase75/76 recovery interlock;
  - the worker never bypasses Phase77 claim fencing, Phase79 STARTED or Phase80 claim-fenced checkpoint commit.
- Resource ownership:
  - `from_env` opens a Phase92 session and owns its cleanup;
  - context-manager/close paths release the owned Phase70 lease and transport through Phase92;
  - wrapper construction failures close an already-opened owned session;
  - externally supplied sessions are not silently closed by the wrapper.
- Fail-closed state:
  - closed sessions and stale Phase71 supervisors are rejected;
  - handoff-construction failure never publishes a false ready startup state;
  - processing after close or before successful recovery release is rejected.
- Red-team unit coverage includes blocked startup, blocked→ready retry on one session, duplicate ready-gate rejection, cross-runtime recovery, stale/closed authority, owned-session cleanup, external-session ownership, constructor-failure cleanup, handoff failure and post-close execution rejection.
- Phase100 remains hard shadow/paper-only and introduces no migration of its own.

## Phase 101 — Integrated Decision → Governed Shadow Runtime

- Brian file:
  - `brian2026/phase101_integrated_decision_shadow_runtime.py`
- Phase101 adds no new alpha model, SQL, scheduler, exchange transport or live execution. It connects the existing Phase54 evidence/portfolio decision producer to the persisted Phase68/73 risk authority, Phase69 governed execution compiler and the Phase100 recovery-first worker.
- Decision authority:
  - only hard shadow Phase54 decisions are accepted; `live_execution` and `automatic_promotion` remain forbidden;
  - `WAIT_NO_GROUNDED_SIGNALS` and `HOLD_CURRENT_BOOK` are true no-op outcomes: no persisted risk read, no governed cycle and no durable execution side effect;
  - `REBALANCE_PLANNED` is the only decision status that may enter execution.
- Grounded execution metadata:
  - execution confidence is derived from the absolute Phase44 blended conviction already embedded in the Phase54 portfolio book;
  - execution evidence ids are derived only from Phase43 validated `support_evidence_ids`;
  - callers cannot inject a separate confidence/evidence lineage at the Phase101 boundary;
  - expected edge in basis points remains an explicit input because Phase54 does not contain a bps return estimate; Phase101 does not fabricate one.
- Authoritative account binding:
  - the Phase54 `current_weights` must match the exact Phase60 authoritative head owned by the same Phase100/92 runtime;
  - supplied equity and available cash must match that Phase60 head;
  - a decision older than the authoritative account head is rejected;
  - this prevents a current signal from being compiled against stale caller-side position/cash state.
- Persisted risk binding:
  - Phase101 loads and validates the current Phase73 operational-risk ledger head;
  - the exact persisted Phase68 receipt is supplied to Phase69;
  - the Phase69 result must echo that receipt id;
  - Phase75 later reloads the persisted risk head atomically at write-ahead authorization, so a risk-head change after Phase101 compilation still fails closed.
- Market / commit preflight:
  - marks must be finite and positive;
  - before durable work begins, marks must cover current non-zero paper positions plus every candidate cycle asset;
  - the execution observation time cannot precede the Phase54 decision;
  - a rebalance that compiles to zero executable items returns `NO_EXECUTABLE_INSTRUCTIONS` without creating durable execution state.
- Runtime path:
  - executable governed cycles are delegated only through Phase100 → Phase99 → Phase75/76/77/79/80;
  - Phase99 still performs a fresh Phase86 admission read and Phase86's transactional authorization/dispatch wrappers remain the final recovery interlock;
  - Phase77 claim fencing, Phase79 STARTED and Phase80 claim-fenced checkpoint authority are not bypassed.
- Red-team coverage includes no-op WAIT/HOLD behavior, persisted risk-head identity/state validation, Phase54-derived confidence/evidence lineage, exact Phase60 weights/equity/cash binding, stale-decision rejection, mark coverage, risk-head race propagation, Phase69 risk-id drift, zero-item no-op behavior, shadow/live boundaries and execution input bounds.
- Phase101 remains hard shadow/paper-only and introduces no migration of its own.

## Phase 102 — Grounded Decision Worker Cycle

- Brian file:
  - `brian2026/phase102_grounded_decision_worker_cycle.py`
- Phase102 adds no new alpha model, SQL, scheduler, exchange transport or live execution. It closes the normal-shadow composition gap by running the existing Phase43→44→52→53→54 decision producer from prefetched evidence and then passing the resulting decision into Phase101 under the already-recovered Phase100 runtime.
- Authoritative account inputs:
  - callers do not provide current portfolio weights, equity or available cash;
  - all three are read from the exact Phase60 head owned by the Phase100/92 runtime;
  - the Phase60 `state_id` is captured before Phase54 and re-read after Phase54 completes;
  - if the account head changes while the decision is being built, Phase101 is never called.
- Decision snapshot integrity:
  - the requested decision timestamp cannot precede the authoritative Phase60 head;
  - the returned Phase54 timestamp must equal the requested snapshot time;
  - the returned Phase54 current weights must equal the captured Phase60 weights;
  - malformed pipeline ids, live/non-shadow decisions and automatic promotion fail closed.
- Real decision path:
  - `run_integrated_shadow_decision` remains the default producer, preserving the real Phase43 grounded analyst → Phase44 portfolio → Phase52 covariance → Phase53 turnover contracts;
  - Phase102 forwards the completed Phase54 decision to Phase101, which derives grounded execution confidence/evidence, binds the current persisted Phase73 risk head and invokes Phase69;
  - Phase101/100/99 then retain the Phase86 recovery gate and Phase75/76/77/79/80 durable execution authority.
- Inputs that are still explicit:
  - expected edge in basis points remains explicit because Phase54 does not contain a bps-return estimate and Phase102 does not synthesize one;
  - execution market snapshots, instrument risk limits and authoritative marks remain explicit point-in-time inputs.
- Red-team coverage includes exact Phase60 weight/equity/cash forwarding, caller inability to override account state, account-head mutation during Phase54, stale timestamps, Phase54 output timestamp/weight/pipeline drift, live/auto-promotion rejection, Phase101 WAIT/HOLD propagation, cross-worker composition, missing/invalid Phase60 head and worker closure.
- Phase102 remains hard shadow/paper-only and introduces no migration of its own.

## Phase 103 — Prospective Post-Cutoff Grounded Runtime

- Brian files:
  - `brian2026/phase103_prospective_grounded_runtime.py`
  - additive injection points in `brian2026/expert_reasoner.py`, `phase43_grounded_analysts.py`, and `phase54_integrated_shadow_decision.py`
- Phase103 closes a critical runtime gap without weakening the frozen research lane:
  - the original `reason_market()` still rejects data at/after the 2026 development cutoff;
  - a separate `reason_market_prospective()` entrypoint permits only the explicit prospective-shadow lane;
  - Phase43 and Phase54 gained dependency-injection hooks, while their defaults preserve the original frozen pre-cutoff behavior.
- Prospective evidence contract:
  - decision timestamp must be post-cutoff;
  - every SensorObservation must remain `PROSPECTIVE_DEVELOPMENT_SHADOW`, `shadow_only=true`, `live_execution=false`;
  - pre-cutoff observations cannot be recycled into the prospective runtime;
  - future-dated observations fail closed;
  - automatic promotion remains forbidden.
- Phase102 now uses the prospective Phase54 runner by default, so the recovery-first current-data worker can actually consume post-cutoff observations without opening the historical contamination boundary.
- Red-team coverage proves both lanes simultaneously:
  - frozen/default reasoner still rejects post-cutoff input;
  - prospective Phase43/54 accept valid current shadow observations;
  - pre-cutoff reuse, future evidence and pre-cutoff prospective decisions are rejected;
  - the default Phase54 lane remains frozen.
- Phase103 adds no exchange transport, SQL, scheduler or live execution.

## Phase 104 — Recovery-First Grounded Backend Worker

- Brian file:
  - `brian2026/phase104_recovery_first_grounded_worker.py`
- Phase104 is the backend lifecycle bridge promised after Phase102:
  - Phase100 recovery is always evaluated before intelligence/market prefetch;
  - if recovery remains blocked, no prefetch provider is called and no Phase102 decision cycle is built;
  - once recovery is ready on the same session, exactly one immutable prefetched bundle is consumed and passed through Phase102.
- Prefetch contract:
  - `PrefetchedGroundedCycle` freezes asset inputs, returns, model weights, Phase54 config, expected-edge estimates, execution markets, risk limits, authoritative marks and timestamps before Phase43 reasoning begins;
  - the bundle itself is hard shadow-only;
  - the provider can later be backed by Supabase or another collector without giving Phase43 open-ended I/O access.
- Worker identity separation:
  - recovery claim identity and normal Phase77 execution claim identity are separate inputs;
  - both claim TTLs remain bounded by their downstream contracts;
  - an already-ready Phase100 worker does not rerun recovery unnecessarily.
- Resource / failure behavior:
  - `from_env` owns and closes its Phase100 worker/session;
  - externally supplied workers are not silently closed;
  - prefetch errors, invalid bundle types and cycle failures propagate without a false success receipt;
  - constructor failures close any worker opened by `from_env`.
- Red-team coverage includes blocked-before-prefetch ordering, ready recovery → prefetch → Phase102 ordering, already-ready reuse, separate claim identities, invalid prefetch, prefetch failure, owned/external cleanup and post-close rejection.
- Phase104 remains scheduler-neutral and hard shadow/paper-only. A concrete Supabase prefetch adapter is intentionally a later layer because expected edge must be evidence-backed rather than synthesized from confidence.

## Phase 105 — Lagged Prospective Expected Edge

- Brian file:
  - `brian2026/phase105_lagged_prospective_edge.py`
- Phase105 ports the already-proven ALPHA Evolution expected-edge challenger semantics into the Python recovery-first runtime instead of inventing edge from confidence.
- Proven upstream basis:
  - `supabase/functions/_shared/evolution_alpha_intelligence.ts`;
  - bounded reliability shrinks toward 0.50, mature groups require at least 100 resolved samples, at least two mature independent groups are required, gross directional edge is capped, dispersion/maturity create an uncertainty penalty, freshness creates decay, and decision-time round-trip cost is subtracted.
- Causal guarantees:
  - reliability snapshots generated/windowed after the decision timestamp are rejected;
  - future source observations contaminate the estimate;
  - missing cost is `COST_UNAVAILABLE`;
  - insufficient mature evidence is `INSUFFICIENT_LAGGED_EVIDENCE`;
  - only PIT-clear positive net edge above the configured margin becomes `ALLOW_EDGE`.
- `eligible_expected_edge_bps_by_asset` exposes only eligible *net* edge values. Missing/denied assets stay absent on purpose rather than receiving a synthetic default.
- Phase105 remains hard shadow-only and cannot auto-promote.

## Phase 106 — Decision-Bound Lagged Edge Gate

- Brian file:
  - `brian2026/phase106_decision_bound_lagged_edge.py`
- Phase106 binds Phase105 to the completed Phase54 decision:
  - only assets that actually require new/increasing risk or an opposite-side reversal open require edge;
  - only Phase43 support groups aligned with the planned new-risk direction may contribute;
  - current Phase44 conviction is used only as the bounded evidence-score field; it is not converted into basis points;
  - future cost evidence and missing/misaligned support groups fail closed.
- Caller-provided expected-edge injection is forbidden in the Phase106 runtime wrapper.
- The resulting eligible edge map and blocked-new-risk set are forwarded to Phase101/69/55.
- Phase55/69/101 gained additive default-off support for `blocked_new_risk_assets`:
  - a blocked OPEN/INCREASE is skipped;
  - a blocked reversal still permits the existing position to close to flat but creates no pending opposite-side open;
  - same-side reductions/closures are never blocked merely because new-risk edge is unavailable.
- This preserves the system's existing “risk reduction must not depend on alpha” invariant.

## Phase 107 — Edge-Bound Recovery-First Worker

- Brian file:
  - `brian2026/phase107_edge_bound_recovery_worker.py`
- Phase107 connects the Phase105/106 edge gate into the real Phase100→102 worker lifecycle:
  - recovery runs first;
  - if recovery is blocked, no intelligence/market prefetch occurs;
  - once READY, one typed prefetch bundle provides lagged edge contexts, not arbitrary edge numbers;
  - Phase107 builds Phase101, wraps it with Phase106, then runs Phase102 with an intentionally empty raw edge map.
- Recovery claim identity and normal Phase77 execution claim identity remain separate.
- Owned Phase100 workers are closed by the Phase107 context manager; externally supplied workers remain externally owned.

## Phase 108 — Read-Only Supabase Lagged Edge Reader

- Brian file:
  - `brian2026/phase108_supabase_lagged_edge_reader.py`
- Phase108 provides the first concrete backend data adapter for Phase105/106, using GET-only PostgREST reads and a strict two-table allowlist:
  - `brian_sensor_reliability_shadow_snapshots`;
  - `brian_dynamic_cost_quotes`.
- Reliability selection mirrors the existing ALPHA challenger:
  - choose one latest common reliability window whose `window_end` and `generated_at` both existed by decision time;
  - then load only requested independent groups from that exact window/horizon;
  - escaped/future rows, wrong evidence class, live-enabled rows, wrong horizon, or ambiguous duplicate group rows fail closed.
- Cost selection:
  - latest fillable non-`UNAVAILABLE` quote at/before decision time;
  - configurable maximum cost age;
  - stale cost becomes unavailable rather than fabricated.
- The transport has no mutation, RPC, exchange, or order surface; secrets are header-only and sanitized from diagnostics.

## Phase 109 — Point-in-Time Edge Prefetch Builder

- Brian file:
  - `brian2026/phase109_pit_edge_prefetch_builder.py`
- Phase109 assembles a Phase107 bundle while sealing another leakage gap: covariance return history.
- `PointInTimeReturnSeries` requires:
  - explicit asset identity;
  - finite returns;
  - observation start/end timestamps;
  - non-empty source lineage;
  - `observed_until <= Phase54 decision_timestamp`.
- The builder also validates every Phase54 SensorObservation as post-cutoff prospective shadow evidence and rejects future observations or asset-identity drift before any edge-reader I/O.
- Independent-group requirements are derived directly from the prefetched observations and sent to Phase108; callers may map local asset ids to the persisted cost-quote asset namespace explicitly.
- The reader must return exactly one edge context per decision asset.
- Phase109 introduces no scheduler, SQL mutation, live execution, or automatic promotion.

## Phase 110 — Grounded Supabase Market Prefetch

- Brian file:
  - `brian2026/phase110_supabase_grounded_market_prefetch.py`
- Phase110 replaces caller-built Phase54 market/evidence inputs with a bounded, read-only Supabase adapter.
- Sensor evidence:
  - GET-only `brian_sensor_observations`;
  - only `PROSPECTIVE_DEVELOPMENT_SHADOW`, `shadow_only=true`, `live_execution=false`;
  - future/pre-cutoff observations are rejected;
  - only the newest row per logical eye is retained, and conflicting equal-time eye states fail closed;
  - the persisted observation id is retained in provenance even though the immutable Python SensorObservation has its own content identity.
- Market history:
  - the original direct reader supports `brian_micro_book_ticks` for crypto and `brian_multiasset_market_marks` for non-crypto;
  - return histories use the common closed-bucket intersection across all requested assets;
  - there is no forward-fill, padding or synthetic zero return;
  - the still-open decision-time bucket may be a mark but cannot enter covariance history.
- Expert snapshot construction is deliberately sparse and deterministic:
  - current structure state from grounded price-structure evidence;
  - point-in-time return, acceleration, EMA slope and price-series z-score;
  - RSI, volume, divergence, support/resistance and other unavailable features are not invented.
- Phase110 now also exposes typed `GroundedPricePoint` and `load_with_price_points()` so a proven external public price-history adapter can reuse the same sensor parsing/alignment/feature logic without adding an open network surface to Phase43.

## Phase 111 — Crypto Edge-Bound Prefetch Provider

- Brian file:
  - `brian2026/phase111_crypto_edge_bound_prefetch.py`
- Phase111 is the zero-argument normal-work prefetch provider consumed by Phase107:
  - canonical `crypto:BTCUSDT` ids remain unchanged through decision/risk state;
  - only at the public Binance execution-data boundary are they mapped to `BTCUSDT`;
  - Phase108 supplies lagged PIT expected-edge context;
  - Phase94 supplies fresh public depth/exchange-info evidence for paper execution marks, tick size and minimum notional;
  - Phase109 seals the final point-in-time bundle.
- Non-crypto execution is intentionally not synthesized: Phase111 fails closed until a proven venue depth/risk-rule adapter exists.
- Bundle identity covers evidence, returns, edge context, execution evidence **and decision/execution policy** (model weights, Phase54 config, max slippage and TTL), so a policy change cannot reuse the same bundle identity.
- The default crypto market-history reader is Phase113 rather than sparse signal-triggered micro-book history.

## Phase 112 — Edge-Bound Crypto Shadow Service

- Brian file:
  - `brian2026/phase112_edge_bound_crypto_shadow_service.py`
- Phase112 owns the Phase107 + Phase111 lifecycle:
  - it does not prefetch itself; it only hands the Phase111 callable to Phase107, preserving recovery-before-prefetch ordering;
  - owned normal-prefetch resources close before the recovery/runtime session;
  - externally supplied components remain externally owned.
- Phase112 adds no scheduler, SQL migration, live execution or promotion path.

## Phase 113 — Completed Public Binance Kline Prefetch

- Brian file:
  - `brian2026/phase113_binance_grounded_market_prefetch.py`
- Phase113 closes the sparse-return-history problem for crypto covariance:
  - only public Binance `/api/v3/klines` is used;
  - only completed 5-minute candles whose close time is at/before the frozen decision timestamp are admitted;
  - a currently-open candle is discarded even if the venue returns it;
  - duplicate close times, invalid OHLC ordering, malformed timestamps and insufficient history fail closed;
  - 418/429 is surfaced as a rate-limit condition; bounded 5xx failures may fail over only across the existing public Binance host allowlist.
- Each admitted close becomes a typed Phase110 `GroundedPricePoint` with a deterministic content/source hash.
- Phase110 remains responsible for grounded Supabase sensors, common bucket alignment, causal returns and expert snapshot construction.
- No Binance API key, signed endpoint, account API, private data or order endpoint exists in this path.

## Phase 114 — Crypto Shadow Machine Entrypoint

- Brian file:
  - `brian2026/phase114_crypto_shadow_machine_entrypoint.py`
- Phase114 makes the Phase112 service machine-invokable for one bounded cycle without inventing strategy policy:
  - policy JSON is mandatory and explicitly supplies asset ids, model weights, Phase54 portfolio/covariance/turnover policy and execution edge/slippage/TTL limits;
  - unknown fields and invalid types fail closed;
  - only canonical `crypto:*USDT` assets are accepted by this entrypoint;
  - recovery and normal worker claim tokens are required to be distinct;
  - configured Supabase secrets are redacted from bounded machine errors.
- Output is bounded JSON containing recovery state, bundle identity, decision pipeline/status and shadow execution receipt identity. A blocked recovery path emits no fabricated decision/execution object.
- Phase114 is still one-shot and scheduler-neutral. It does not merge, deploy SQL, place a live order, or enable automatic promotion.



## Phase 115 — Crypto Shadow Readiness Gate

- Brian file:
  - `brian2026/phase115_crypto_shadow_readiness_gate.py`
- Phase115 is a read-only, pre-scheduler gate for the Phase114 machine. It does **not** bootstrap state or mutate recovery/risk/runtime records.
- CORE readiness requires:
  - an existing validated Phase70 durable runtime checkpoint;
  - an existing validated Phase73 operational-risk ledger;
  - Phase86 recovery admission `OPEN`;
  - successful grounded Phase113/110 market prefetch;
  - fresh public Phase94 Binance depth/exchange-info execution evidence.
- NEW_RISK readiness additionally requires:
  - at least one horizon-fresh available sensor group per asset;
  - covariance return count satisfying the configured Phase52 minimum;
  - reliability snapshots fresh enough for the hourly reliability producer;
  - at least Phase105's preregistered maturity threshold: two independent groups with >=100 samples each;
  - a fresh fillable decision-time dynamic cost.
- The report has three machine states:
  - `READY_FOR_EDGE_BOUND_SHADOW`: core + new-risk evidence are ready;
  - `SAFE_FAIL_CLOSED_ONLY`: core runtime is safe to invoke but new/increasing risk will remain fail-closed;
  - `NOT_READY`: runtime/recovery/market execution authority itself is not ready.
- Persisted-state reads reuse the Phase70/73/86 typed validators. The dedicated readiness RPC transport allowlists only the three existing read functions and exposes no lease, commit, claim, cancel, dispatch or order RPC.
- Reports are deterministic/content-addressed and remain `read_only=true`, `shadow_only=true`, `live_execution=false`.

## Phase 116 — Crypto Shadow Readiness Machine Entrypoint

- Brian file:
  - `brian2026/phase116_crypto_shadow_readiness_entrypoint.py`
- Phase116 turns Phase115 into a bounded one-shot machine probe before any scheduler is enabled.
- It reuses the explicit Phase114 policy parser instead of inventing strategy defaults, requires a runtime id, and emits bounded JSON with machine-visible exit codes:
  - 0 = fully edge-bound shadow ready;
  - 10 = safe fail-closed only;
  - 20 = not ready;
  - 30 = input error;
  - 40 = readiness-check error.
- Generic and scoped Supabase secrets are scrubbed from diagnostics.
- Phase116 performs no runtime bootstrap, database mutation, schedule creation, exchange order or promotion.

## Split Supabase Source Topology — Runtime Readiness Hardening

A read-only audit of the two active Brian Supabase projects showed that the current data plane is intentionally split, so Phases91/108/110 were hardened to model that split explicitly instead of assuming one `SUPABASE_URL`.

- `BRIAN_SENSOR_SUPABASE_*`: current sensor observations; intended for the realtime project.
- `BRIAN_EDGE_SUPABASE_*`: lagged reliability snapshots; intended for the market-intelligence project.
- `BRIAN_COST_SUPABASE_*`: current dynamic execution-cost quotes; intended for the realtime project.
- `BRIAN_RUNTIME_SUPABASE_*`: Phase70/73/86 durable runtime authority once that migration set is deliberately deployed.
- Every scoped source may fall back to the legacy generic `SUPABASE_*` configuration for backward compatibility, but partial scoped configurations fail closed.
- Phase108 can use independent reliability and cost project URLs/keys without creating a generic cross-project mutation surface.
- Phase110 now filters PostgREST sensor reads to supported horizons/families at query time and maps the concrete live derivative families `taker_flow`, `open_interest` and `funding_crowding` into the existing grounded `derivatives` source-kind boundary.
- Unsupported live families/horizons (for example current EVENT_DRIVEN/DAILY rows not accepted by Phase43's current horizon contract) no longer poison an otherwise valid crypto sensor prefetch.
- No live database migration was applied as part of this audit.


## Phase 117 — Readiness-Guarded Crypto Shadow Runner

- Brian file:
  - `brian2026/phase117_readiness_guarded_crypto_shadow.py`
- Phase117 composes Phase115 and Phase112 without adding a scheduler:
  - Phase115 runs first using the exact requested runtime id and Phase114 policy;
  - if readiness is `SAFE_FAIL_CLOSED_ONLY` or `NOT_READY`, Phase112 is **not even constructed**;
  - only `READY_FOR_EDGE_BOUND_SHADOW` is allowed to construct the recovery-first worker service and attempt one shadow cycle.
- The runtime id supplied to readiness is injected into the worker environment and an identity mismatch after worker startup is treated as a machine failure.
- A green readiness probe is not treated as an authorization that can bypass later controls: Phase100/107 still run recovery first, Phase101 reloads persisted risk, and later durable gates may re-block execution if state changes after preflight.
- Recovery and normal execution claim tokens remain distinct.
- Phase117 is one-shot, shadow-only and scheduler-neutral. It performs no live-order action and creates no periodic job.
