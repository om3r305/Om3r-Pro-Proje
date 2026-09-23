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
