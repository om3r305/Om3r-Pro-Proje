# DIP target-level observations v1

Observation-only addition to the existing candidate result. It does not change the structure fingerprint, occurrence/episode hash, target, invalidation, entry eligibility, veto or sizing. No migration or new writer/table.

Only the selected target is annotated, using the existing four target timeframes. Pivot and equal-pair provenance is retained. History begins after pivot confirmation (both pivots for a pair). Evidence timestamps identify sealed bar starts, not an invented intrabar order. States summarize whether a level was untouched, reached, wicked through and reclaimed, or closed through. CLOSE_CROSSED does not imply permanent invalidity; retest validity is not inferred. Pair matching uses the existing reader's decision-time ATR tolerance; confirmed_at records constituent pivot confirmation, not proof the pair qualified under every earlier ATR.

The wrapper copies evidence instead of mutating decision inputs. Malformed optional data returns the original candidate. The selected-target summary is capped at eight records, with a conservative 11 KB decision-envelope guard; diagnostics are omitted if they would risk the existing SQL size limit. Rejected candidates remain rejected.

Validation: 12 synthetic behavior tests passed locally with --no-check; decision.ts and its dependencies passed deno check. CI includes the new tests with type checking. These tests do not establish profitability or historical live impact.

Deployment status: prepared for review; not deployed by this package. Before rollout, confirm CI and deploy the candidate's new local module with the worker's existing dependencies. After rollout, compare recorded selected-target status versus vetoes without changing trading policy. ETHUSDT, SHADOW ONLY; Phase 3.7, Control Center and main Brian memory untouched.
