-- Brian ALPHA v2 auditor catch-up queue.
-- Keeps audit reads bounded and avoids repeatedly scanning already-resolved decisions.
-- SHADOW ONLY. No strategy threshold or Phase 3.7 changes.

create index if not exists brian_alpha_decisions_observed_at_idx
  on public.brian_alpha_decisions (observed_at asc);

create or replace function public.brian_alpha_pending_audit_decisions(
  p_now timestamptz default now(),
  p_lookback interval default interval '12 hours',
  p_limit integer default 120
)
returns table (
  decision_id text,
  observed_at timestamptz,
  asset_id text,
  observed_reference_price numeric,
  action text,
  direction smallint,
  evidence_score double precision,
  estimated_round_trip_cost_bps double precision,
  source_observation_ids text[]
)
language sql
security definer
set search_path = pg_catalog, public
as $$
  select
    d.decision_id,
    d.observed_at,
    d.asset_id,
    d.observed_reference_price,
    d.action,
    d.direction,
    d.evidence_score,
    d.estimated_round_trip_cost_bps,
    d.source_observation_ids
  from public.brian_alpha_decisions d
  where d.observed_at >= p_now - greatest(p_lookback, interval '1 hour')
    and d.observed_at <= p_now - interval '5 minutes'
    and exists (
      select 1
      from (values (300), (900), (3600)) as h(horizon_seconds)
      where d.observed_at + make_interval(secs => h.horizon_seconds) <= p_now
        and not exists (
          select 1
          from public.brian_alpha_decision_outcomes o
          where o.decision_id = d.decision_id
            and o.horizon_seconds = h.horizon_seconds
        )
    )
  order by d.observed_at asc, d.decision_id asc
  limit greatest(1, least(coalesce(p_limit, 120), 500));
$$;

revoke all on function public.brian_alpha_pending_audit_decisions(timestamptz, interval, integer) from public;
revoke all on function public.brian_alpha_pending_audit_decisions(timestamptz, interval, integer) from anon;
revoke all on function public.brian_alpha_pending_audit_decisions(timestamptz, interval, integer) from authenticated;
grant execute on function public.brian_alpha_pending_audit_decisions(timestamptz, interval, integer) to service_role;

comment on function public.brian_alpha_pending_audit_decisions(timestamptz, interval, integer)
is 'Service-role-only bounded queue of ALPHA decisions with at least one due unresolved 5m/15m/60m audit horizon.';
