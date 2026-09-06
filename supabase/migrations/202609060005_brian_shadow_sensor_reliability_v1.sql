-- Brian 2026: Shadow Sensor Reliability Learner v1.
--
-- This is OBSERVATION-ONLY learning infrastructure. It measures how each causal
-- sensor observation performed at 5m / 15m / 60m outcome horizons, but it does
-- NOT update brian_sensor_observations.reliability, ALPHA scores, thresholds,
-- Phase 3.7, sizing, or any execution path.
--
-- Reused evidence is de-duplicated: each observation_id x outcome_horizon is
-- counted once per rolling snapshot, through the earliest causal ALPHA decision
-- that referenced it inside the snapshot evidence set.

CREATE TABLE IF NOT EXISTS public.brian_sensor_reliability_shadow_snapshots (
  snapshot_id text PRIMARY KEY,
  window_start timestamptz NOT NULL,
  window_end timestamptz NOT NULL,
  generated_at timestamptz NOT NULL DEFAULT now(),
  independent_group text NOT NULL,
  sensor_family text NOT NULL,
  sensor_horizon text NOT NULL,
  outcome_horizon_seconds integer NOT NULL CHECK (outcome_horizon_seconds IN (300, 900, 3600)),
  sample_count integer NOT NULL CHECK (sample_count >= 0),
  hit_count integer NOT NULL CHECK (hit_count >= 0),
  miss_count integer NOT NULL CHECK (miss_count >= 0),
  flat_count integer NOT NULL CHECK (flat_count >= 0),
  cost_covered_count integer NOT NULL CHECK (cost_covered_count >= 0),
  hit_rate double precision,
  bayesian_hit_rate_beta10_10 double precision,
  avg_signed_bps double precision,
  median_signed_bps double precision,
  avg_cost_adjusted_signed_bps double precision,
  avg_abs_market_move_bps double precision,
  evidence_class text NOT NULL DEFAULT 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean NOT NULL DEFAULT true CHECK (shadow_only),
  live_execution boolean NOT NULL DEFAULT false CHECK (NOT live_execution),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CHECK (window_end > window_start),
  CHECK (hit_rate IS NULL OR (hit_rate >= 0 AND hit_rate <= 1)),
  CHECK (bayesian_hit_rate_beta10_10 IS NULL OR (bayesian_hit_rate_beta10_10 >= 0 AND bayesian_hit_rate_beta10_10 <= 1))
);

ALTER TABLE public.brian_sensor_reliability_shadow_snapshots ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS brian_sensor_reliability_shadow_latest_idx
  ON public.brian_sensor_reliability_shadow_snapshots (window_end DESC);

CREATE INDEX IF NOT EXISTS brian_sensor_reliability_shadow_group_idx
  ON public.brian_sensor_reliability_shadow_snapshots
     (independent_group, sensor_horizon, outcome_horizon_seconds, window_end DESC);

DROP TRIGGER IF EXISTS brian_sensor_reliability_shadow_snapshots_append_only
  ON public.brian_sensor_reliability_shadow_snapshots;
CREATE TRIGGER brian_sensor_reliability_shadow_snapshots_append_only
  BEFORE UPDATE OR DELETE ON public.brian_sensor_reliability_shadow_snapshots
  FOR EACH ROW EXECUTE FUNCTION public.brian_reject_mutation();

CREATE OR REPLACE FUNCTION public.brian_refresh_sensor_reliability_shadow(
  p_as_of timestamptz DEFAULT now(),
  p_window interval DEFAULT interval '24 hours'
)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
  v_window_end timestamptz := date_trunc('hour', coalesce(p_as_of, now()));
  v_window_start timestamptz;
  v_inserted integer := 0;
BEGIN
  IF p_window < interval '1 hour' OR p_window > interval '30 days' THEN
    RAISE EXCEPTION 'BRIAN_RELIABILITY_WINDOW_OUT_OF_RANGE';
  END IF;

  v_window_start := v_window_end - p_window;

  WITH ranked AS (
    SELECT
      d.decision_id,
      d.observed_at AS decision_at,
      d.estimated_round_trip_cost_bps,
      s.observation_id,
      s.independent_group,
      s.sensor_family,
      s.horizon AS sensor_horizon,
      s.direction AS sensor_direction,
      s.observed_at AS sensor_at,
      o.outcome_id,
      o.horizon_seconds AS outcome_horizon_seconds,
      o.gross_return,
      o.resolved_at,
      row_number() OVER (
        PARTITION BY s.observation_id, o.horizon_seconds
        ORDER BY d.observed_at ASC, d.decision_id ASC
      ) AS rn
    FROM public.brian_alpha_decision_outcomes o
    JOIN public.brian_alpha_decisions d
      ON d.decision_id = o.decision_id
    CROSS JOIN LATERAL unnest(d.source_observation_ids) AS src(observation_id)
    JOIN public.brian_sensor_observations s
      ON s.observation_id = src.observation_id
    WHERE o.resolved_at > v_window_start
      AND o.resolved_at <= v_window_end
      AND o.horizon_seconds IN (300, 900, 3600)
      AND o.gross_return IS NOT NULL
      AND d.observed_at <= v_window_end
      AND s.observed_at <= d.observed_at
      AND o.resolved_at >= d.observed_at
      AND s.available = true
      AND s.direction <> 0
      AND s.independent_group <> 'news_gdelt'
  ), dedup AS (
    SELECT * FROM ranked WHERE rn = 1
  ), agg AS (
    SELECT
      independent_group,
      sensor_family,
      sensor_horizon,
      outcome_horizon_seconds,
      count(*)::integer AS sample_count,
      count(*) FILTER (WHERE sensor_direction * gross_return > 0)::integer AS hit_count,
      count(*) FILTER (WHERE sensor_direction * gross_return < 0)::integer AS miss_count,
      count(*) FILTER (WHERE sensor_direction * gross_return = 0)::integer AS flat_count,
      count(*) FILTER (WHERE estimated_round_trip_cost_bps IS NOT NULL)::integer AS cost_covered_count,
      avg(sensor_direction * gross_return * 10000.0) AS avg_signed_bps,
      percentile_cont(0.5) WITHIN GROUP (ORDER BY sensor_direction * gross_return * 10000.0) AS median_signed_bps,
      avg(sensor_direction * gross_return * 10000.0 - estimated_round_trip_cost_bps)
        FILTER (WHERE estimated_round_trip_cost_bps IS NOT NULL) AS avg_cost_adjusted_signed_bps,
      avg(abs(gross_return) * 10000.0) AS avg_abs_market_move_bps
    FROM dedup
    GROUP BY independent_group, sensor_family, sensor_horizon, outcome_horizon_seconds
  )
  INSERT INTO public.brian_sensor_reliability_shadow_snapshots (
    snapshot_id, window_start, window_end, generated_at,
    independent_group, sensor_family, sensor_horizon, outcome_horizon_seconds,
    sample_count, hit_count, miss_count, flat_count, cost_covered_count,
    hit_rate, bayesian_hit_rate_beta10_10,
    avg_signed_bps, median_signed_bps, avg_cost_adjusted_signed_bps, avg_abs_market_move_bps,
    evidence_class, shadow_only, live_execution, metadata
  )
  SELECT
    md5(concat_ws('|',
      'brian.sensor-reliability-shadow.v1',
      v_window_start::text,
      v_window_end::text,
      independent_group,
      sensor_family,
      sensor_horizon,
      outcome_horizon_seconds::text
    )),
    v_window_start,
    v_window_end,
    now(),
    independent_group,
    sensor_family,
    sensor_horizon,
    outcome_horizon_seconds,
    sample_count,
    hit_count,
    miss_count,
    flat_count,
    cost_covered_count,
    CASE
      WHEN hit_count + miss_count > 0
      THEN hit_count::double precision / (hit_count + miss_count)::double precision
      ELSE NULL
    END,
    CASE
      WHEN hit_count + miss_count > 0
      THEN (hit_count + 10.0) / (hit_count + miss_count + 20.0)
      ELSE NULL
    END,
    avg_signed_bps,
    median_signed_bps,
    avg_cost_adjusted_signed_bps,
    avg_abs_market_move_bps,
    'PROSPECTIVE_DEVELOPMENT_SHADOW',
    true,
    false,
    jsonb_build_object(
      'schema_version', 'brian.sensor-reliability-shadow.v1',
      'role', 'measurement_only_no_alpha_weight_change',
      'dedupe', 'observation_id_x_outcome_horizon_first_causal_decision_per_snapshot',
      'beta_prior_alpha', 10,
      'beta_prior_beta', 10,
      'source', 'alpha_decision_outcomes_x_alpha_decisions_x_sensor_observations'
    )
  FROM agg
  ON CONFLICT (snapshot_id) DO NOTHING;

  GET DIAGNOSTICS v_inserted = ROW_COUNT;
  RETURN v_inserted;
END;
$$;

REVOKE ALL ON FUNCTION public.brian_refresh_sensor_reliability_shadow(timestamptz, interval) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.brian_refresh_sensor_reliability_shadow(timestamptz, interval) FROM anon, authenticated;
GRANT EXECUTE ON FUNCTION public.brian_refresh_sensor_reliability_shadow(timestamptz, interval) TO service_role;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM cron.job WHERE jobname = 'brian-sensor-reliability-shadow-hourly'
  ) THEN
    PERFORM cron.schedule(
      'brian-sensor-reliability-shadow-hourly',
      '12 * * * *',
      $cron$SELECT public.brian_refresh_sensor_reliability_shadow(now(), interval '24 hours');$cron$
    );
  END IF;
END;
$$;
