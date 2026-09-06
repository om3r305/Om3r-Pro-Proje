-- Brian 2026 production stabilization: exact latest micro state per eye.
-- Uses the existing (eye_id, observed_at DESC) index; intentionally adds no new index.
-- SHADOW-only infrastructure change. No Phase 3.7 or execution semantics change.

CREATE OR REPLACE FUNCTION public.brian_latest_micro_book_ticks(p_eye_ids text[])
RETURNS TABLE (
  eye_id text,
  starting_equity numeric,
  equity_after numeric,
  peak_equity_after numeric,
  max_drawdown_pct_after double precision,
  target_direction smallint,
  observed_mid_price numeric,
  observed_at timestamptz
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT latest.eye_id,
         latest.starting_equity,
         latest.equity_after,
         latest.peak_equity_after,
         latest.max_drawdown_pct_after,
         latest.target_direction,
         latest.observed_mid_price,
         latest.observed_at
  FROM unnest(coalesce(p_eye_ids, ARRAY[]::text[])) AS requested(eye_id)
  CROSS JOIN LATERAL (
    SELECT t.eye_id,
           t.starting_equity,
           t.equity_after,
           t.peak_equity_after,
           t.max_drawdown_pct_after,
           t.target_direction,
           t.observed_mid_price,
           t.observed_at
    FROM public.brian_micro_book_ticks AS t
    WHERE t.eye_id = requested.eye_id
    ORDER BY t.observed_at DESC
    LIMIT 1
  ) AS latest;
$$;

REVOKE ALL ON FUNCTION public.brian_latest_micro_book_ticks(text[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.brian_latest_micro_book_ticks(text[]) FROM anon, authenticated;
GRANT EXECUTE ON FUNCTION public.brian_latest_micro_book_ticks(text[]) TO service_role;
