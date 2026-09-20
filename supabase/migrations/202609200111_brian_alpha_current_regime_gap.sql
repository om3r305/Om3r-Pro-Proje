insert into public.brian_evolution_gap_snapshots(
  snapshot_id,gap_id,observed_at,capability_id,domain,severity,reason,suggested_action,
  evidence_refs,metadata,evidence_class,shadow_only,live_execution
)
select
  md5('alpha-current-regime-short-horizon-edge|'||now()::text),
  'alpha-current-regime-short-horizon-edge',
  now(),
  'ALPHA_COST_ADJUSTED_SHORT_HORIZON_EDGE',
  'ALPHA',
  case when coalesce((h->>'after_cost_favorable_pct')::numeric,0) < 20 then 'HIGH' else 'MEDIUM' end,
  format(
    'Current-regime 5m after-cost favorable rate is %s%% across %s outcomes; current overall after-cost rate is %s%% and average round-trip cost is %s bps.',
    coalesce(h->>'after_cost_favorable_pct','0'),
    coalesce(h->>'samples','0'),
    current_after_cost_hit_pct,
    current_avg_cost_bps
  ),
  'Keep the 5-minute ALPHA edge in SHADOW/research. Require at least 100 current-regime 5m outcomes plus a sustained after-cost edge before any promotion; use the 15m/60m split to learn whether the horizon, entry timing, or cost gate is the limiting factor.',
  array[
    'alpha-regime:'||regime_started_at::text,
    'current-regime-outcomes:'||current_samples::text,
    '5m-outcomes:'||coalesce(h->>'samples','0'),
    '5m-after-cost-pct:'||coalesce(h->>'after_cost_favorable_pct','0')
  ]::text[],
  jsonb_build_object(
    'source','brian_alpha_development_metrics_v1',
    'regime_started_at',regime_started_at,
    'current_quality_pct',selected_quality_pct,
    'current_evidence_pct',selected_evidence_pct,
    'horizon_seconds',300,
    'current_horizons',current_horizons,
    'auto_promote',false,
    'direct_alpha_influence',false
  ),
  'PROSPECTIVE_EVOLUTION_SHADOW',
  true,
  false
from public.brian_alpha_development_metrics_v1() m
cross join lateral (
  select value as h
  from jsonb_array_elements(m.current_horizons)
  where (value->>'seconds')::int=300
  limit 1
) x
where not exists (
  select 1
  from public.brian_evolution_gap_snapshots g
  where g.gap_id='alpha-current-regime-short-horizon-edge'
    and g.observed_at > now()-interval '6 hours'
);
