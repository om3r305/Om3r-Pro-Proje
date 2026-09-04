# Brian ALPHA v2 — Agent Mission Contract

This document is the shared implementation/review contract for Codex and external Claude review on PR #49 / issue #50.

## Scope

Branch: `feat/brian-alpha-decision-compiler-v2`
Base: `brian-2026`
Primary issue: #50
Primary PR: #49

## Hard boundaries

- Frozen Phase 3.7 is read-only and must not be modified.
- SHADOW ONLY: no authenticated exchange order surface, no Binance order endpoint, no HMAC signing, no live execution.
- Preserve `shadow_only=true` and `live_execution=false` as hard invariants.
- No Sep-4 hindsight threshold tuning.
- Emergent Mover remains attention-only.
- Official macro remains context-only with zero direction vote.
- GDELT remains discovery-only and cannot vote direction in ALPHA.

## Must-fix before production SHADOW

1. Exclude GDELT/news-attention discovery rows from ALPHA directional evidence.
2. Pin frozen Phase 3.7 evidence to `phase37-prospective-live-20260903` with causal `observed_at <= decision time` semantics.
3. Prevent multiple signals from one intrabar tape/lineage from satisfying the 2-independent-group gate by themselves.
4. Wire synchronized visible L2 depth into the runtime cost compiler; degraded top-of-book is fallback only.
5. Auditor must fail closed when cost is unavailable; never silently substitute 0 bps.
6. Resolution tolerance may extend past a horizon, but excursion/MFE/MAE must not include post-horizon prices; short-side semantics must be explicit.
7. Align WAIT/VETO shadow-state semantics with Phase 3.7 comparison semantics; do not blindly close on WAIT.
8. Prefer per-asset WAIT/VETO receipts over killing the whole compiler cycle when one asset/source is malformed or unavailable.
9. Lock Edge Function invocation for production cron use; no anonymous audit-log writes.
10. CI must reject live-order/authenticated Binance surfaces in ALPHA code.

## Required adversarial tests

- `news_gdelt + market group` cannot OPEN.
- Two same-lineage intrabar signals cannot alone satisfy independence.
- Wrong Phase 3.7 experiment id cannot vote.
- Future Phase 3.7 tick cannot vote/compare.
- Thin L2 => `fillable=false` => `VETO/INSUFFICIENT_VISIBLE_DEPTH`.
- Missing cost never becomes 0 bps in auditor.
- Horizon +120s resolution does not contaminate excursion with post-horizon prices.
- `OPEN_SHORT` favorable/adverse excursion semantics are correct.
- Poison evidence cannot kill unrelated assets.
- book/context outage yields per-asset WAIT/VETO where possible.
- OPEN → WAIT/VETO state/comparator semantics are consistent.
- same-time/out-of-order state remains causal.
- forbidden `/api/v3/order`, `X-MBX-APIKEY`, HMAC/signed execution, `live_execution=true` patterns break CI.

## Definition of done

- Brian ALPHA v2 CI green.
- Brian 2026 CI green.
- New red-team tests green.
- PR remains SHADOW only.
- No production migration/function/cron rollout until issue #50 is complete and reviewed.
