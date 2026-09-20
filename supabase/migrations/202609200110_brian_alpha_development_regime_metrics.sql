-- ALPHA development anatomy: current regime is measured separately from legacy history.
insert into public.brian_evolution_runtime_config(config_key,config_value,updated_at)
values('brian_alpha_current_regime_started_at','2026-09-19 23:14:05.147+00',now())
on conflict(config_key) do nothing;

create index if not exists brian_alpha_decisions_action_observed_idx
  on public.brian_alpha_decisions(action, observed_at desc);

create index if not exists brian_alpha_outcomes_decision_resolved_idx
  on public.brian_alpha_decision_outcomes(decision_id, resolved_at desc);

CREATE OR REPLACE FUNCTION public.brian_alpha_development_metrics_v1()
 RETURNS TABLE(regime_started_at timestamp with time zone, current_samples bigint, current_cost_samples bigint, current_gross_hit_pct numeric, current_after_cost_hit_pct numeric, current_avg_cost_bps numeric, current_avg_signed_move_bps numeric, current_evidence_pct numeric, current_horizons jsonb, legacy_samples bigint, legacy_cost_samples bigint, legacy_gross_hit_pct numeric, legacy_after_cost_hit_pct numeric, lifetime_samples bigint, lifetime_decisions bigint, lifetime_evidence_pct numeric, selected_quality_pct numeric, selected_evidence_pct numeric, quality_mode text)
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
with cfg as (
  select coalesce(
    (
      select nullif(config_value,'')::timestamptz
      from public.brian_evolution_runtime_config
      where config_key='brian_alpha_current_regime_started_at'
      limit 1
    ),
    now()
  ) as regime_started_at
),
base as (
  select
    d.decision_id,
    d.observed_at as decision_at,
    o.horizon_seconds,
    o.direction_adjusted_return,
    d.estimated_round_trip_cost_bps
  from public.brian_alpha_decision_outcomes o
  join public.brian_alpha_decisions d on d.decision_id=o.decision_id
  where d.action in ('OPEN_LONG','OPEN_SHORT')
),
current_rows as (
  select b.*
  from base b,cfg
  where b.decision_at >= cfg.regime_started_at
),
legacy_rows as (
  select b.*
  from base b,cfg
  where b.decision_at < cfg.regime_started_at
),
current_agg as (
  select
    count(*)::bigint as samples,
    count(*) filter(where estimated_round_trip_cost_bps is not null and estimated_round_trip_cost_bps>=0)::bigint as cost_samples,
    100.0*count(*) filter(where direction_adjusted_return>0)/nullif(count(*),0) as gross_hit_pct,
    100.0*count(*) filter(
      where estimated_round_trip_cost_bps is not null
        and estimated_round_trip_cost_bps>=0
        and direction_adjusted_return*10000 > estimated_round_trip_cost_bps
    )/nullif(count(*) filter(where estimated_round_trip_cost_bps is not null and estimated_round_trip_cost_bps>=0),0) as after_cost_hit_pct,
    avg(estimated_round_trip_cost_bps) filter(where estimated_round_trip_cost_bps is not null and estimated_round_trip_cost_bps>=0) as avg_cost_bps,
    avg(direction_adjusted_return*10000) as avg_signed_move_bps
  from current_rows
),
legacy_agg as (
  select
    count(*)::bigint as samples,
    count(*) filter(where estimated_round_trip_cost_bps is not null and estimated_round_trip_cost_bps>=0)::bigint as cost_samples,
    100.0*count(*) filter(where direction_adjusted_return>0)/nullif(count(*),0) as gross_hit_pct,
    100.0*count(*) filter(
      where estimated_round_trip_cost_bps is not null
        and estimated_round_trip_cost_bps>=0
        and direction_adjusted_return*10000 > estimated_round_trip_cost_bps
    )/nullif(count(*) filter(where estimated_round_trip_cost_bps is not null and estimated_round_trip_cost_bps>=0),0) as after_cost_hit_pct
  from legacy_rows
),
life as (
  select count(*)::bigint as samples,count(distinct decision_id)::bigint as decisions from base
),
horizon as (
  select coalesce(jsonb_agg(jsonb_build_object(
    'seconds',horizon_seconds,
    'samples',n,
    'gross_hit_pct',round(gross_hit_pct::numeric,1),
    'after_cost_favorable_pct',round(after_cost_hit_pct::numeric,1)
  ) order by horizon_seconds),'[]'::jsonb) as rows
  from (
    select
      horizon_seconds,
      count(*)::bigint as n,
      100.0*count(*) filter(where direction_adjusted_return>0)/nullif(count(*),0) as gross_hit_pct,
      100.0*count(*) filter(
        where estimated_round_trip_cost_bps is not null
          and estimated_round_trip_cost_bps>=0
          and direction_adjusted_return*10000 > estimated_round_trip_cost_bps
      )/nullif(count(*) filter(where estimated_round_trip_cost_bps is not null and estimated_round_trip_cost_bps>=0),0) as after_cost_hit_pct
    from current_rows
    group by horizon_seconds
  ) x
),
scored as (
  select
    c.*,
    l.samples as legacy_samples,
    l.cost_samples as legacy_cost_samples,
    l.gross_hit_pct as legacy_gross_hit_pct,
    l.after_cost_hit_pct as legacy_after_cost_hit_pct,
    life.samples as lifetime_samples,
    life.decisions as lifetime_decisions,
    least(100.0,100.0*ln(1+greatest(c.samples,0)::numeric)/ln(301::numeric)) as current_evidence_pct,
    least(100.0,100.0*ln(1+greatest(life.samples,0)::numeric)/ln(3001::numeric)) as lifetime_evidence_pct,
    case
      when c.cost_samples >= 30 then c.after_cost_hit_pct
      when c.cost_samples > 0 and l.cost_samples > 0 then
        (
          coalesce(c.after_cost_hit_pct,0)*c.cost_samples
          + coalesce(l.after_cost_hit_pct,0)*greatest(0,30-c.cost_samples)
        ) / 30.0
      when l.cost_samples > 0 then l.after_cost_hit_pct
      else 0
    end as selected_quality_pct,
    case
      when c.cost_samples >= 30 then 'CURRENT_REGIME'
      when c.cost_samples > 0 then 'CURRENT_REGIME_SHRUNK'
      else 'LEGACY_FALLBACK'
    end as quality_mode
  from current_agg c
  cross join legacy_agg l
  cross join life
)
select
  cfg.regime_started_at,
  s.samples,
  s.cost_samples,
  round(coalesce(s.gross_hit_pct,0)::numeric,2),
  round(coalesce(s.after_cost_hit_pct,0)::numeric,2),
  round(coalesce(s.avg_cost_bps,0)::numeric,3),
  round(coalesce(s.avg_signed_move_bps,0)::numeric,3),
  round(coalesce(s.current_evidence_pct,0)::numeric,2),
  horizon.rows,
  s.legacy_samples,
  s.legacy_cost_samples,
  round(coalesce(s.legacy_gross_hit_pct,0)::numeric,2),
  round(coalesce(s.legacy_after_cost_hit_pct,0)::numeric,2),
  s.lifetime_samples,
  s.lifetime_decisions,
  round(coalesce(s.lifetime_evidence_pct,0)::numeric,2),
  round(coalesce(s.selected_quality_pct,0)::numeric,2),
  round(coalesce(s.current_evidence_pct,0)::numeric,2),
  s.quality_mode
from scored s
cross join cfg
cross join horizon;
$function$
;

revoke all on function public.brian_alpha_development_metrics_v1() from public, anon, authenticated;
grant execute on function public.brian_alpha_development_metrics_v1() to service_role;
