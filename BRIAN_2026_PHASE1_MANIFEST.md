# Brian 2026 Phase 1 — Migration Manifest

## Added
- `brian2026/` shadow-first adaptive research core
- specialist committee + meta trader with explicit WAIT
- episodic memory + specialist reliability
- independent risk governor
- deterministic counterfactual replay lab (fees/slippage included)
- bounded loss-cluster researcher
- objective promotion gate (never auto-live)
- legacy shadow bridge linking old trade opens/closes to Brian outcomes
- smoke demo and unit tests

## Changed
- `core/bot.py`: fixed `__future__` SyntaxError; schedules previously orphaned 15-minute helper jobs; integrates Brian 2026 in shadow-only mode and links outcomes.
- `config_live.json`: adds `brian2026` shadow config and promotion/risk defaults.
- `.gitignore`: ignores Brian runtime state and local secret variants.

## Verification
- 361 Python source files compiled successfully.
- 4 Brian 2026 unit tests passed.
- `Proje1.core.bot` imports successfully.
- Brian synthetic smoke decision executes in shadow mode.

## Safety / rollout rule
Phase 1 does not grant Brian authority to place, veto, rewrite, or promote live-money execution. It observes the legacy trader, learns from linked outcomes, replays experiments, and produces champion candidates only.
