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
