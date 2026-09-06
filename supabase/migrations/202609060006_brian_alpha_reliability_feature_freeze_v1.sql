-- Brian 2026: prospective ALPHA reliability feature freeze v1.
--
-- This migration freezes only information that was already available at each
-- ALPHA decision timestamp. It never reads decision outcomes and it does not
-- modify ALPHA scores, thresholds, sensor reliability, Phase 3.7, sizing, or
-- execution behavior. The resulting rows are for later prospective evaluation.

CREATE TABLE IF NOT EXISTS public.brian_alpha_reliability_shadow_features (
  assessment_id text PRIMARY KEY,
  decision_id text NOT NULL UNIQUE REFERENCES public.brian_alpha_decisions(decision_id),
  decision_observed_at timestamptz NOT NULL,
  captured_at timestamptz NOT NULL DEFAULT now(),
  asset_id text NOT NULL,
  action text NOT NULL CHECK (action IN ('OPEN_LONG', 'OPEN_SHORT', 'WAIT', 'VETO')),
  decision_direction smallint NOT NULL CHECK (decision_direction IN (-1, 0, 1)),
  decision_evidence_score double precision NOT NULL CHECK (decision_evidence_score >= 0 AND decision_evidence_score <= 1),
  decision_estimated_round_trip_cost_bps double precision,
  snapshot_window_end timestamptz NOT NULL,
  snapshot_generated_at timestamptz NOT NULL,
  source_observation_count integer NOT NULL CHECK (source_observation_count >= 0),
  matched_source_observation_count integer NOT NULL CHECK (matched_source_observation_count >= 0),
  feature_covered_source_count integer NOT NULL CHECK (feature_covered_source_count >= 0),
  features jsonb NOT NULL DEFAULT '[]'::jsonb,
  evidence_class text NOT NULL DEFAULT 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean NOT NULL DEFAULT true CHECK (shadow_only),
  live_execution boolean NOT NULL DEFAULT false CHECK (NOT live_execution),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CHECK (snapshot_window_end <= decision_observed_at),
  CHECK (snapshot_generated_at <= decision_observed_at),
  CHECK (matched_source_observation_count <= source_observation_count),
  CHECK (feature_covered_source_count <= matched_source_observation_count)
);

ALTER TABLE public.brian_alpha_reliability_shadow_features ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS brian_alpha_reliability_shadow_features_time_idx
  ON public.brian_alpha_reliability_shadow_features (decision_observed_at DESC);

CREATE INDEX IF NOT EXISTS brian_alpha_reliability_shadow_features_snapshot_idx
  ON public.brian_alpha_reliability_shadow_features (snapshot_window_end DESC, decision_observed_at DESC);

DROP TRIGGER IF EXISTS brian_alpha_reliability_shadow_features_append_only
  ON public.brian_alpha_reliability_shadow_features;
CREATE TRIGGER brian_alpha_reliability_shadow_features_append_only
  BEFORE UPDATE OR DELETE ON public.brian_alpha_reliability_shadow_features
  FOR EACH ROW EXECUTE FUNCTION public.brian_reject_mutation();

CREATE OR REPLACE FUNCTION public.brian_capture_alpha_reliability_shadow_features(
  p_as_of timestamptz DEFAULT now(),
  p_lookback interval DEFAULT interval '15 minutes',
  p_limit integer DEFAULT 500
)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
  v_inserted integer := 0;
BEGIN
  IF p_lookback < interval '1 minute' OR p_lookback > interval '2 hours' THEN
    RAISE EXCEPTION 'BRIAN_ALPHA_RELIABILITY_FEATURE_LOOKBACK_OUT_OF_RANGE';
  END IF;

  WITH candidates AS (
    SELECT
      d.decision_id,
      d.observed_at AS decision_observed_at,
      d.asset_id,
      d.action,
      d.direction AS decision_direction,
      d.evidence_score AS decision_evidence_score,
      d.estimated_round_trip_cost_bps AS decision_estimated_round_trip_cost_bps,
      d.source_observation_ids,
      snap.window_end AS snapshot_window_end,
      snap.generated_at AS snapshot_generated_at
    FROM public.brian_alpha_decisions d
    JOIN LATERAL (
      SELECT
        r.window_end,
        max(r.generated_at) AS generated_at
      FROM public.brian_sensor_reliability_shadow_snapshots r
      WHERE r.window_end <= d.observed_at
        AND r.generated_at <= d.observed_at
      GROUP BY r.window_end
      ORDER BY r.window_end DESC
      LIMIT 1
    ) snap ON true
    LEFT JOIN public.brian_alpha_reliability_shadow_features existing
      ON existing.decision_id = d.decision_id
    WHERE d.observed_at > coalesce(p_as_of, now()) - p_lookback
      AND d.observed_at <= coalesce(p_as_of, now())
      AND coalesce(cardinality(d.source_observation_ids), 0) > 0
      AND existing.decision_id IS NULL
    ORDER BY d.observed_at ASC, d.decision_id ASC
    LIMIT greatest(1, least(coalesce(p_limit, 500), 2000))
  ), packed AS (
    SELECT
      c.*,
      feature_pack.source_observation_count,
      feature_pack.matched_source_observation_count,
      feature_pack.feature_covered_source_count,
      feature_pack.features
    FROM candidates c
    CROSS JOIN LATERAL (
      SELECT
        count(*)::integer AS source_observation_count,
        count(*) FILTER (WHERE s.observation_id IS NOT NULL)::integer AS matched_source_observation_count,
        count(*) FILTER (
          WHERE s.observation_id IS NOT NULL
            AND EXISTS (
              SELECT 1
              FROM public.brian_sensor_reliability_shadow_snapshots rs_exists
              WHERE rs_exists.window_end = c.snapshot_window_end
                AND rs_exists.generated_at <= c.decision_observed_at
                AND rs_exists.independent_group = s.independent_group
                AND rs_exists.sensor_family = s.sensor_family
                AND rs_exists.sensor_horizon = s.horizon
            )
        )::integer AS feature_covered_source_count,
        jsonb_agg(
          jsonb_build_object(
            'observation_id', src.observation_id,
            'matched_sensor_observation', s.observation_id IS NOT NULL,
            'independent_group', s.independent_group,
            'compiler_canonical_group', CASE
              WHEN s.independent_group IN ('micro_velocity', 'micro_volume', 'micro_breakout', 'micro_reclaim', 'micro_taker_flow')
                THEN 'intrabar_tape'
              ELSE s.independent_group
            END,
            'sensor_family', s.sensor_family,
            'sensor_horizon', s.horizon,
            'sensor_direction', s.direction,
            'strength', s.strength,
            'confidence', s.confidence,
            'fixed_reliability_at_decision', s.reliability,
            'sensor_observed_at', s.observed_at,
            'snapshot_metrics', coalesce((
              SELECT jsonb_agg(
                jsonb_build_object(
                  'snapshot_id', rs.snapshot_id,
                  'outcome_horizon_seconds', rs.outcome_horizon_seconds,
                  'sample_count', rs.sample_count,
                  'hit_rate', rs.hit_rate,
                  'bayesian_hit_rate_beta10_10', rs.bayesian_hit_rate_beta10_10,
                  'avg_signed_bps', rs.avg_signed_bps,
                  'median_signed_bps', rs.median_signed_bps,
                  'avg_cost_adjusted_signed_bps', rs.avg_cost_adjusted_signed_bps,
                  'avg_abs_market_move_bps', rs.avg_abs_market_move_bps
                )
                ORDER BY rs.outcome_horizon_seconds
              )
              FROM public.brian_sensor_reliability_shadow_snapshots rs
              WHERE rs.window_end = c.snapshot_window_end
                AND rs.generated_at <= c.decision_observed_at
                AND rs.independent_group = s.independent_group
                AND rs.sensor_family = s.sensor_family
                AND rs.sensor_horizon = s.horizon
            ), '[]'::jsonb)
          )
          ORDER BY src.observation_id
        ) AS features
      FROM unnest(c.source_observation_ids) AS src(observation_id)
      LEFT JOIN public.brian_sensor_observations s
        ON s.observation_id = src.observation_id
    ) feature_pack
  )
  INSERT INTO public.brian_alpha_reliability_shadow_features (
    assessment_id,
    decision_id,
    decision_observed_at,
    captured_at,
    asset_id,
    action,
    decision_direction,
    decision_evidence_score,
    decision_estimated_round_trip_cost_bps,
    snapshot_window_end,
    snapshot_generated_at,
    source_observation_count,
    matched_source_observation_count,
    feature_covered_source_count,
    features,
    evidence_class,
    shadow_only,
    live_execution,
    metadata
  )
  SELECT
    md5(concat_ws('|', 'brian.alpha-reliability-feature-freeze.v1', decision_id, snapshot_window_end::text)),
    decision_id,
    decision_observed_at,
    now(),
    asset_id,
    action,
    decision_direction,
    decision_evidence_score,
    decision_estimated_round_trip_cost_bps,
    snapshot_window_end,
    snapshot_generated_at,
    source_observation_count,
    matched_source_observation_count,
    feature_covered_source_count,
    features,
    'PROSPECTIVE_DEVELOPMENT_SHADOW',
    true,
    false,
    jsonb_build_object(
      'schema_version', 'brian.alpha-reliability-feature-freeze.v1',
      'role', 'prospective_feature_freeze_no_alpha_effect',
      'outcomes_consulted', false,
      'causal_availability_rule', 'snapshot_generated_at_lte_decision_observed_at',
      'alpha_reliability_mutation_enabled', false
    )
  FROM packed
  ON CONFLICT (decision_id) DO NOTHING;

  GET DIAGNOSTICS v_inserted = ROW_COUNT;
  RETURN v_inserted;
END;
$$;

REVOKE ALL ON FUNCTION public.brian_capture_alpha_reliability_shadow_features(timestamptz, interval, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.brian_capture_alpha_reliability_shadow_features(timestamptz, interval, integer) FROM anon, authenticated;
GRANT EXECUTE ON FUNCTION public.brian_capture_alpha_reliability_shadow_features(timestamptz, interval, integer) TO service_role;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM cron.job WHERE jobname = 'brian-alpha-reliability-feature-freeze-1m'
  ) THEN
    PERFORM cron.schedule(
      'brian-alpha-reliability-feature-freeze-1m',
      '* * * * *',
      $cron$SELECT public.brian_capture_alpha_reliability_shadow_features(now(), interval '15 minutes', 500);$cron$
    );
  END IF;
END;
$$;
