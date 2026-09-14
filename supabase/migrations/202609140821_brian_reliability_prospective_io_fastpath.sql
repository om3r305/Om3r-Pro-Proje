-- Keep the same 12h prospective-calibration semantics and 500-row batch contract,
-- but avoid thousands of random calibration-key probes and repeated snapshot fan-out.
create or replace function public.brian_resolve_sensor_reliability_prospective_calibration(
  p_as_of timestamptz default now(),
  p_lookback interval default interval '12 hours',
  p_limit integer default 500
)
returns integer
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
  perform set_config('work_mem','64MB',true);

  with recent_outcomes as materialized (
    select outcome_id,decision_id,horizon_seconds,gross_return,resolved_at
    from public.brian_alpha_decision_outcomes
    where resolved_at > v_as_of-p_lookback
      and resolved_at <= v_as_of
      and horizon_seconds in (300,900,3600)
      and gross_return is not null
  ), existing_keys as materialized (
    select observation_id,outcome_horizon_seconds
    from public.brian_sensor_reliability_prospective_calibration
  ), feature_rows as materialized (
    select
      f.decision_id,f.decision_observed_at,f.snapshot_window_end,f.snapshot_generated_at,
      feat,
      o.outcome_id,o.horizon_seconds as outcome_horizon_seconds,o.gross_return,o.resolved_at
    from recent_outcomes o
    join public.brian_alpha_reliability_shadow_features f on f.decision_id=o.decision_id
    cross join lateral jsonb_array_elements(f.features) feat
    left join existing_keys existing
      on existing.observation_id=feat->>'observation_id'
     and existing.outcome_horizon_seconds=o.horizon_seconds
    where existing.observation_id is null
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
    cross join lateral (
      select metric
      from jsonb_array_elements(coalesce(fr.feat->'snapshot_metrics','[]'::jsonb)) metric
      where (metric->>'outcome_horizon_seconds')::integer=fr.outcome_horizon_seconds
      limit 1
    ) m
    join public.brian_sensor_reliability_shadow_snapshots rs
      on rs.snapshot_id=m.metric->>'snapshot_id'
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
      'io_fastpath','materialized_existing_key_hash_and_single_horizon_snapshot_lookup',
      'source','alpha_reliability_shadow_features_x_future_alpha_outcomes'
    )
  from bounded
  on conflict (observation_id,outcome_horizon_seconds) do nothing;

  get diagnostics v_inserted=row_count;
  return v_inserted;
end;
$function$;
