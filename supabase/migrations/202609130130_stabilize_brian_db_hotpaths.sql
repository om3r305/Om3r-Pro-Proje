-- Brian DB hot-path stabilization. No business/trading logic changes.

create index if not exists brian_alpha_outcomes_reliability_window_v2_idx
  on public.brian_alpha_decision_outcomes (resolved_at desc, horizon_seconds, decision_id)
  include (outcome_id, gross_return);

create index if not exists brian_collector_runs_started_at_idx
  on public.brian_collector_runs (started_at desc);

create index if not exists brian_collector_runs_collector_status_started_idx
  on public.brian_collector_runs (collector_id, status, started_at desc);

create index if not exists brian_collector_lease_events_observed_at_idx
  on public.brian_collector_lease_events (observed_at);

create index if not exists brian_dynamic_cost_quotes_observed_at_idx
  on public.brian_dynamic_cost_quotes (observed_at);

create index if not exists brian_alpha_shadow_position_events_event_ts_idx
  on public.brian_alpha_shadow_position_events (event_ts);

create or replace function public.brian_refresh_sensor_reliability_shadow(
  p_as_of timestamptz default now(),
  p_window interval default interval '24 hours'
) returns integer
language plpgsql
security definer
set search_path to 'pg_catalog','public'
as $function$
declare
  v_window_end timestamptz := date_trunc('hour', coalesce(p_as_of, now()));
  v_window_start timestamptz;
  v_inserted integer := 0;
begin
  if p_window < interval '1 hour' or p_window > interval '30 days' then
    raise exception 'BRIAN_RELIABILITY_WINDOW_OUT_OF_RANGE';
  end if;

  if not pg_try_advisory_xact_lock(hashtextextended('brian_refresh_sensor_reliability_shadow', 0)) then
    return 0;
  end if;
  perform set_config('statement_timeout','45000',true);
  perform set_config('work_mem','32MB',true);

  v_window_start := v_window_end - p_window;

  with outcome_window as materialized (
    select outcome_id, decision_id, horizon_seconds, gross_return, resolved_at
    from public.brian_alpha_decision_outcomes
    where resolved_at > v_window_start
      and resolved_at <= v_window_end
      and horizon_seconds in (300,900,3600)
      and gross_return is not null
  ), expanded as materialized (
    select
      d.decision_id,
      d.observed_at as decision_at,
      d.estimated_round_trip_cost_bps,
      s.observation_id,
      s.independent_group,
      s.sensor_family,
      s.horizon as sensor_horizon,
      s.direction as sensor_direction,
      s.observed_at as sensor_at,
      o.outcome_id,
      o.horizon_seconds as outcome_horizon_seconds,
      o.gross_return,
      o.resolved_at
    from outcome_window o
    join public.brian_alpha_decisions d on d.decision_id=o.decision_id
    cross join lateral unnest(d.source_observation_ids) src(observation_id)
    join public.brian_sensor_observations s on s.observation_id=src.observation_id
    where d.observed_at <= v_window_end
      and s.observed_at <= d.observed_at
      and o.resolved_at >= d.observed_at
      and s.available=true
      and s.direction<>0
      and s.independent_group<>'news_gdelt'
  ), dedup as materialized (
    select distinct on (observation_id, outcome_horizon_seconds)
      decision_id, decision_at, estimated_round_trip_cost_bps,
      observation_id, independent_group, sensor_family, sensor_horizon,
      sensor_direction, sensor_at, outcome_id, outcome_horizon_seconds,
      gross_return, resolved_at
    from expanded
    order by observation_id, outcome_horizon_seconds, decision_at asc, decision_id asc
  ), agg as (
    select
      independent_group,
      sensor_family,
      sensor_horizon,
      outcome_horizon_seconds,
      count(*)::integer as sample_count,
      count(*) filter (where sensor_direction*gross_return>0)::integer as hit_count,
      count(*) filter (where sensor_direction*gross_return<0)::integer as miss_count,
      count(*) filter (where sensor_direction*gross_return=0)::integer as flat_count,
      count(*) filter (where estimated_round_trip_cost_bps is not null)::integer as cost_covered_count,
      avg(sensor_direction*gross_return*10000.0) as avg_signed_bps,
      percentile_cont(0.5) within group (order by sensor_direction*gross_return*10000.0) as median_signed_bps,
      avg(sensor_direction*gross_return*10000.0-estimated_round_trip_cost_bps)
        filter (where estimated_round_trip_cost_bps is not null) as avg_cost_adjusted_signed_bps,
      avg(abs(gross_return)*10000.0) as avg_abs_market_move_bps
    from dedup
    group by independent_group,sensor_family,sensor_horizon,outcome_horizon_seconds
  )
  insert into public.brian_sensor_reliability_shadow_snapshots (
    snapshot_id,window_start,window_end,generated_at,
    independent_group,sensor_family,sensor_horizon,outcome_horizon_seconds,
    sample_count,hit_count,miss_count,flat_count,cost_covered_count,
    hit_rate,bayesian_hit_rate_beta10_10,
    avg_signed_bps,median_signed_bps,avg_cost_adjusted_signed_bps,avg_abs_market_move_bps,
    evidence_class,shadow_only,live_execution,metadata
  )
  select
    md5(concat_ws('|','brian.sensor-reliability-shadow.v1',v_window_start::text,v_window_end::text,
      independent_group,sensor_family,sensor_horizon,outcome_horizon_seconds::text)),
    v_window_start,v_window_end,now(),
    independent_group,sensor_family,sensor_horizon,outcome_horizon_seconds,
    sample_count,hit_count,miss_count,flat_count,cost_covered_count,
    case when hit_count+miss_count>0 then hit_count::double precision/(hit_count+miss_count)::double precision else null end,
    case when hit_count+miss_count>0 then (hit_count+10.0)/(hit_count+miss_count+20.0) else null end,
    avg_signed_bps,median_signed_bps,avg_cost_adjusted_signed_bps,avg_abs_market_move_bps,
    'PROSPECTIVE_DEVELOPMENT_SHADOW',true,false,
    jsonb_build_object(
      'schema_version','brian.sensor-reliability-shadow.v1',
      'role','measurement_only_no_alpha_weight_change',
      'dedupe','observation_id_x_outcome_horizon_first_causal_decision_per_snapshot',
      'beta_prior_alpha',10,'beta_prior_beta',10,
      'source','alpha_decision_outcomes_x_alpha_decisions_x_sensor_observations'
    )
  from agg
  on conflict (snapshot_id) do nothing;

  get diagnostics v_inserted=row_count;
  return v_inserted;
end;
$function$;

create or replace function public.brian_resolve_sensor_reliability_prospective_calibration(
  p_as_of timestamptz default now(),
  p_lookback interval default interval '12 hours',
  p_limit integer default 500
) returns integer
language plpgsql
security definer
set search_path to 'pg_catalog','public'
as $function$
declare
  v_inserted integer := 0;
  v_as_of timestamptz := coalesce(p_as_of,now());
begin
  if p_lookback < interval '1 hour' or p_lookback > interval '48 hours' then
    raise exception 'BRIAN_RELIABILITY_CALIBRATION_LOOKBACK_OUT_OF_RANGE';
  end if;

  if not pg_try_advisory_xact_lock(hashtextextended('brian_resolve_sensor_reliability_prospective_calibration',0)) then
    return 0;
  end if;
  perform set_config('statement_timeout','45000',true);
  perform set_config('work_mem','32MB',true);

  with recent_outcomes as materialized (
    select outcome_id,decision_id,horizon_seconds,gross_return,resolved_at
    from public.brian_alpha_decision_outcomes
    where resolved_at > v_as_of-p_lookback
      and resolved_at <= v_as_of
      and horizon_seconds in (300,900,3600)
      and gross_return is not null
  ), feature_rows as materialized (
    select
      f.decision_id,f.decision_observed_at,f.snapshot_window_end,f.snapshot_generated_at,
      feat,
      o.outcome_id,o.horizon_seconds as outcome_horizon_seconds,o.gross_return,o.resolved_at
    from recent_outcomes o
    join public.brian_alpha_reliability_shadow_features f on f.decision_id=o.decision_id
    cross join lateral jsonb_array_elements(f.features) feat
    left join public.brian_sensor_reliability_prospective_calibration existing
      on existing.observation_id=feat->>'observation_id'
     and existing.outcome_horizon_seconds=o.horizon_seconds
    where existing.calibration_id is null
      and coalesce((feat->>'matched_sensor_observation')::boolean,false) is true
      and (feat->>'sensor_direction')::smallint in (-1,1)
      and o.resolved_at >= f.decision_observed_at
  ), expanded as materialized (
    select
      fr.decision_id,fr.decision_observed_at,fr.snapshot_window_end,fr.snapshot_generated_at,
      fr.feat->>'observation_id' as observation_id,
      fr.feat->>'independent_group' as independent_group,
      fr.feat->>'sensor_family' as sensor_family,
      fr.feat->>'sensor_horizon' as sensor_horizon,
      (fr.feat->>'sensor_direction')::smallint as sensor_direction,
      fr.outcome_id,fr.outcome_horizon_seconds,fr.gross_return,fr.resolved_at,
      rs.snapshot_id,
      rs.sample_count as prior_sample_count,
      rs.hit_rate as prior_hit_rate,
      rs.bayesian_hit_rate_beta10_10 as prior_bayesian_hit_rate_beta10_10,
      rs.avg_signed_bps as prior_avg_signed_bps,
      rs.median_signed_bps as prior_median_signed_bps,
      rs.avg_cost_adjusted_signed_bps as prior_avg_cost_adjusted_signed_bps,
      rs.avg_abs_market_move_bps as prior_avg_abs_market_move_bps
    from feature_rows fr
    cross join lateral jsonb_array_elements(coalesce(fr.feat->'snapshot_metrics','[]'::jsonb)) metric
    join public.brian_sensor_reliability_shadow_snapshots rs on rs.snapshot_id=metric->>'snapshot_id'
    where rs.outcome_horizon_seconds=fr.outcome_horizon_seconds
      and rs.window_end=fr.snapshot_window_end
      and rs.generated_at<=fr.decision_observed_at
      and rs.window_end<=fr.decision_observed_at
  ), bounded as materialized (
    select distinct on (observation_id,outcome_horizon_seconds) *
    from expanded
    order by observation_id,outcome_horizon_seconds,decision_observed_at asc,decision_id asc
    limit greatest(1,least(coalesce(p_limit,500),2000))
  )
  insert into public.brian_sensor_reliability_prospective_calibration (
    calibration_id,observation_id,decision_id,outcome_id,snapshot_id,
    independent_group,sensor_family,sensor_horizon,outcome_horizon_seconds,
    snapshot_window_end,snapshot_generated_at,decision_observed_at,resolved_at,
    prior_sample_count,prior_hit_rate,prior_bayesian_hit_rate_beta10_10,
    prior_avg_signed_bps,prior_median_signed_bps,prior_avg_cost_adjusted_signed_bps,prior_avg_abs_market_move_bps,
    sensor_direction,gross_return,realized_sensor_signed_bps,realized_hit,
    evidence_class,shadow_only,live_execution,metadata
  )
  select
    md5(concat_ws('|','brian.sensor-reliability-prospective-calibration.v1',observation_id,outcome_horizon_seconds::text)),
    observation_id,decision_id,outcome_id,snapshot_id,
    independent_group,sensor_family,sensor_horizon,outcome_horizon_seconds,
    snapshot_window_end,snapshot_generated_at,decision_observed_at,resolved_at,
    prior_sample_count,prior_hit_rate,prior_bayesian_hit_rate_beta10_10,
    prior_avg_signed_bps,prior_median_signed_bps,prior_avg_cost_adjusted_signed_bps,prior_avg_abs_market_move_bps,
    sensor_direction,gross_return,sensor_direction*gross_return*10000.0,
    case when sensor_direction*gross_return>0 then true when sensor_direction*gross_return<0 then false else null end,
    'PROSPECTIVE_DEVELOPMENT_SHADOW',true,false,
    jsonb_build_object(
      'schema_version','brian.sensor-reliability-prospective-calibration.v1',
      'role','prospective_calibration_measurement_only',
      'dedupe','observation_id_x_outcome_horizon_earliest_frozen_decision',
      'reliability_mutation_enabled',false,
      'alpha_action_mutation_enabled',false,
      'source','alpha_reliability_shadow_features_x_future_alpha_outcomes'
    )
  from bounded
  on conflict (observation_id,outcome_horizon_seconds) do nothing;

  get diagnostics v_inserted=row_count;
  return v_inserted;
end;
$function$;

do $block$
declare v_job bigint;
begin
  select jobid into v_job from cron.job where jobname='brian-sensor-reliability-calibration-5m' limit 1;
  if v_job is not null then
    perform cron.alter_job(v_job, schedule => '39 * * * *', command => $$SELECT public.brian_resolve_sensor_reliability_prospective_calibration(now(), interval '12 hours', 500);$$);
  end if;

  select jobid into v_job from cron.job where jobname='brian-compact-retention-hourly' limit 1;
  if v_job is not null then
    perform cron.alter_job(v_job, schedule => '43 */6 * * *');
  end if;
end;
$block$;

analyze public.brian_alpha_decision_outcomes;
analyze public.brian_collector_runs;
analyze public.brian_collector_lease_events;
analyze public.brian_dynamic_cost_quotes;
