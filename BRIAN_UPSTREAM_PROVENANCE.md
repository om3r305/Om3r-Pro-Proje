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
