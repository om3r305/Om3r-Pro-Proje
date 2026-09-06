-- Brian 2026: Prospective Sensor Reliability Calibration v1.
--
-- Resolves previously frozen, decision-time reliability features against outcomes
-- that arrived later. This is evaluation-only infrastructure. It does not update
-- sensor reliability, ALPHA scores, thresholds, Phase 3.7, sizing, or execution.
--
-- Repeated reuse of one sensor observation is de-duplicated: each
-- observation_id x outcome_horizon is calibrated once, through the earliest
-- frozen causal ALPHA decision available in the recovery window.

CREATE TABLE IF NOT EXISTS public.brian_sensor_reliability_prospective_calibration (
  calibration_id text PRIMARY KEY,
  observation_id text NOT NULL REFERENCES public.brian_sensor_observations(observation_id),
  decision_id text NOT NULL REFERENCES public.brian_alpha_decisions(decision_id),
  outcome_id text NOT NULL REFERENCES public.brian_alpha_decision_outcomes(outcome_id),
  snapshot_id text NOT NULL REFERENCES public.brian_sensor_reliability_shadow_snapshots(snapshot_id),
  independent_group text NOT NULL,
  sensor_family text NOT NULL,
  sensor_horizon text NOT NULL,
  outcome_horizon_seconds integer NOT NULL CHECK (outcome_horizon_seconds IN (300, 900, 3600)),
  snapshot_window_end timestamptz NOT NULL,
  snapshot_generated_at timestamptz NOT NULL,
  decision_observed_at timestamptz NOT NULL,
  resolved_at timestamptz NOT NULL,
  prior_sample_count integer NOT NULL CHECK (prior_sample_count >= 0),
  prior_hit_rate double precision,
  prior_bayesian_hit_rate_beta10_10 double precision,
  prior_avg_signed_bps double precision,
  prior_median_signed_bps double precision,
  prior_avg_cost_adjusted_signed_bps double precision,
  prior_avg_abs_market_move_bps double precision,
  sensor_direction smallint NOT NULL CHECK (sensor_direction IN (-1, 1)),
  gross_return double precision NOT NULL,
  realized_sensor_signed_bps double precision NOT NULL,
  realized_hit boolean,
  evidence_class text NOT NULL DEFAULT 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean NOT NULL DEFAULT true CHECK (shadow_only),
  live_execution boolean NOT NULL DEFAULT false CHECK (NOT live_execution),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (observation_id, outcome_horizon_seconds),
  CHECK (snapshot_window_end <= decision_observed_at),
  CHECK (snapshot_generated_at <= decision_observed_at),
  CHECK (resolved_at >= decision_observed_at),
  CHECK (prior_hit_rate IS NULL OR (prior_hit_rate >= 0 AND prior_hit_rate <= 1)),
  CHECK (prior_bayesian_hit_rate_beta10_10 IS NULL OR (prior_bayesian_hit_rate_beta10_10 >= 0 AND prior_bayesian_hit_rate_beta10_10 <= 1))
);

ALTER TABLE public.brian_sensor_reliability_prospective_calibration ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS brian_sensor_reliability_calibration_resolved_idx
  ON public.brian_sensor_reliability_prospective_calibration (resolved_at DESC);

CREATE INDEX IF NOT EXISTS brian_sensor_reliability_calibration_group_idx
  ON public.brian_sensor_reliability_prospective_calibration
     (independent_group, sensor_horizon, outcome_horizon_seconds, resolved_at DESC);

DROP TRIGGER IF EXISTS brian_sensor_reliability_prospective_calibration_append_only
  ON public.brian_sensor_reliability_prospective_calibration;
CREATE TRIGGER brian_sensor_reliability_prospective_calibration_append_only
  BEFORE UPDATE OR DELETE ON public.brian_sensor_reliability_prospective_calibration
  FOR EACH ROW EXECUTE FUNCTION public.brian_reject_mutation();

CREATE OR REPLACE FUNCTION public.brian_resolve_sensor_reliability_prospective_calibration(
  p_as_of timestamptz DEFAULT now(),
  p_lookback interval DEFAULT interval '12 hours',
  p_limit integer DEFAULT 2000
)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
  v_inserted integer := 0;
BEGIN
  IF p_lookback < interval '1 hour' OR p_lookback > interval '48 hours' THEN
    RAISE EXCEPTION 'BRIAN_RELIABILITY_CALIBRATION_LOOKBACK_OUT_OF_RANGE';
  END IF;

  WITH expanded AS (
    SELECT
      f.decision_id,
      f.decision_observed_at,
      f.snapshot_window_end,
      f.snapshot_generated_at,
      feat->>'observation_id' AS observation_id,
      feat->>'independent_group' AS independent_group,
      feat->>'sensor_family' AS sensor_family,
      feat->>'sensor_horizon' AS sensor_horizon,
      (feat->>'sensor_direction')::smallint AS sensor_direction,
      o.outcome_id,
      o.horizon_seconds AS outcome_horizon_seconds,
      o.gross_return,
      o.resolved_at,
      rs.snapshot_id,
      rs.sample_count AS prior_sample_count,
      rs.hit_rate AS prior_hit_rate,
      rs.bayesian_hit_rate_beta10_10 AS prior_bayesian_hit_rate_beta10_10,
      rs.avg_signed_bps AS prior_avg_signed_bps,
      rs.median_signed_bps AS prior_median_signed_bps,
      rs.avg_cost_adjusted_signed_bps AS prior_avg_cost_adjusted_signed_bps,
      rs.avg_abs_market_move_bps AS prior_avg_abs_market_move_bps,
      row_number() OVER (
        PARTITION BY feat->>'observation_id', o.horizon_seconds
        ORDER BY f.decision_observed_at ASC, f.decision_id ASC
      ) AS rn
    FROM public.brian_alpha_reliability_shadow_features f
    JOIN public.brian_alpha_decision_outcomes o
      ON o.decision_id = f.decision_id
    CROSS JOIN LATERAL jsonb_array_elements(f.features) feat
    CROSS JOIN LATERAL jsonb_array_elements(coalesce(feat->'snapshot_metrics', '[]'::jsonb)) metric
    JOIN public.brian_sensor_reliability_shadow_snapshots rs
      ON rs.snapshot_id = metric->>'snapshot_id'
    WHERE o.resolved_at > coalesce(p_as_of, now()) - p_lookback
      AND o.resolved_at <= coalesce(p_as_of, now())
      AND o.horizon_seconds IN (300, 900, 3600)
      AND o.gross_return IS NOT NULL
      AND (feat->>'matched_sensor_observation')::boolean IS TRUE
      AND (feat->>'sensor_direction')::smallint IN (-1, 1)
      AND rs.outcome_horizon_seconds = o.horizon_seconds
      AND rs.window_end = f.snapshot_window_end
      AND rs.generated_at <= f.decision_observed_at
      AND rs.window_end <= f.decision_observed_at
      AND o.resolved_at >= f.decision_observed_at
  ), dedup AS (
    SELECT *
    FROM expanded
    WHERE rn = 1
  ), bounded AS (
    SELECT d.*
    FROM dedup d
    LEFT JOIN public.brian_sensor_reliability_prospective_calibration existing
      ON existing.observation_id = d.observation_id
     AND existing.outcome_horizon_seconds = d.outcome_horizon_seconds
    WHERE existing.calibration_id IS NULL
    ORDER BY d.resolved_at ASC, d.observation_id ASC, d.outcome_horizon_seconds ASC
    LIMIT greatest(1, least(coalesce(p_limit, 2000), 10000))
  )
  INSERT INTO public.brian_sensor_reliability_prospective_calibration (
    calibration_id,
    observation_id,
    decision_id,
    outcome_id,
    snapshot_id,
    independent_group,
    sensor_family,
    sensor_horizon,
    outcome_horizon_seconds,
    snapshot_window_end,
    snapshot_generated_at,
    decision_observed_at,
    resolved_at,
    prior_sample_count,
    prior_hit_rate,
    prior_bayesian_hit_rate_beta10_10,
    prior_avg_signed_bps,
    prior_median_signed_bps,
    prior_avg_cost_adjusted_signed_bps,
    prior_avg_abs_market_move_bps,
    sensor_direction,
    gross_return,
    realized_sensor_signed_bps,
    realized_hit,
    evidence_class,
    shadow_only,
    live_execution,
    metadata
  )
  SELECT
    md5(concat_ws('|', 'brian.sensor-reliability-prospective-calibration.v1', observation_id, outcome_horizon_seconds::text)),
    observation_id,
    decision_id,
    outcome_id,
    snapshot_id,
    independent_group,
    sensor_family,
    sensor_horizon,
    outcome_horizon_seconds,
    snapshot_window_end,
    snapshot_generated_at,
    decision_observed_at,
    resolved_at,
    prior_sample_count,
    prior_hit_rate,
    prior_bayesian_hit_rate_beta10_10,
    prior_avg_signed_bps,
    prior_median_signed_bps,
    prior_avg_cost_adjusted_signed_bps,
    prior_avg_abs_market_move_bps,
    sensor_direction,
    gross_return,
    sensor_direction * gross_return * 10000.0,
    CASE
      WHEN sensor_direction * gross_return > 0 THEN true
      WHEN sensor_direction * gross_return < 0 THEN false
      ELSE NULL
    END,
    'PROSPECTIVE_DEVELOPMENT_SHADOW',
    true,
    false,
    jsonb_build_object(
      'schema_version', 'brian.sensor-reliability-prospective-calibration.v1',
      'role', 'prospective_calibration_measurement_only',
      'dedupe', 'observation_id_x_outcome_horizon_earliest_frozen_decision',
      'reliability_mutation_enabled', false,
      'alpha_action_mutation_enabled', false,
      'source', 'alpha_reliability_shadow_features_x_future_alpha_outcomes'
    )
  FROM bounded
  ON CONFLICT (observation_id, outcome_horizon_seconds) DO NOTHING;

  GET DIAGNOSTICS v_inserted = ROW_COUNT;
  RETURN v_inserted;
END;
$$;

REVOKE ALL ON FUNCTION public.brian_resolve_sensor_reliability_prospective_calibration(timestamptz, interval, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.brian_resolve_sensor_reliability_prospective_calibration(timestamptz, interval, integer) FROM anon, authenticated;
GRANT EXECUTE ON FUNCTION public.brian_resolve_sensor_reliability_prospective_calibration(timestamptz, interval, integer) TO service_role;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM cron.job WHERE jobname = 'brian-sensor-reliability-calibration-5m'
  ) THEN
    PERFORM cron.schedule(
      'brian-sensor-reliability-calibration-5m',
      '4-59/5 * * * *',
      $cron$SELECT public.brian_resolve_sensor_reliability_prospective_calibration(now(), interval '12 hours', 2000);$cron$
    );
  END IF;
END;
$$;
