-- Keep the live ALPHA auditor bounded to the caller-requested recovery window.
-- This prevents the hot 5m/15m/60m outcome lane from re-scanning the full 36h debt
-- on every run. A caller may still request up to 36h explicitly for recovery work.
create index if not exists brian_alpha_decisions_terminal_price_idx
on public.brian_alpha_decisions(asset_id, observed_at)
where observed_reference_price is not null;

create index if not exists brian_intrabar_terminal_price_idx
on public.brian_intrabar_reaction_events(asset_id, observed_at)
where observed_mid_price is not null;

-- Brian ALPHA auditor terminal-resolvability queue hardening.
-- SHADOW ONLY. No order, capital, sizing, or live-execution surface.
--
-- Once a preregistered horizon is more than +120 seconds past its target,
-- the strict auditor can only resolve it when a terminal market point exists
-- inside [target, target+120s]. Permanently sparse decisions previously stayed
-- "pending" forever and repeatedly consumed the bounded queue, starving newer
-- resolvable 60m evidence. Keep a due item pending during the +120s grace window,
-- then retain it only when causal terminal evidence actually exists.

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
  with cfg as materialized (
    select
      greatest(1, least(coalesce(p_limit, 120), 500))::int as lim,
      greatest(
        1,
        ceil(greatest(1, least(coalesce(p_limit, 120), 500)) / 3.0)
      )::int as recent_quota,
      greatest(
        1,
        ceil(greatest(1, least(coalesce(p_limit, 120), 500)) / 6.0)
      )::int as lane_quota,
      p_now - least(greatest(coalesce(p_lookback, interval '6 hours'), interval '2 hours'), interval '36 hours') as oldest
  ),
  eligible as materialized (
    select
      d.decision_id,
      d.observed_at,
      (
        d.observed_at + interval '5 minutes' <= p_now
        and not exists (
          select 1
          from public.brian_alpha_decision_outcomes o
          where o.decision_id = d.decision_id
            and o.horizon_seconds = 300
        )
        and (
          p_now <= d.observed_at + interval '7 minutes'
          or exists (
            select 1
            from public.brian_alpha_decisions px
            where px.asset_id = d.asset_id
              and px.observed_reference_price is not null
              and px.observed_at >= d.observed_at + interval '5 minutes'
              and px.observed_at <= d.observed_at + interval '7 minutes'
          )
          or exists (
            select 1
            from public.brian_intrabar_reaction_events ix
            where ix.asset_id = d.asset_id
              and ix.observed_mid_price is not null
              and ix.observed_at >= d.observed_at + interval '5 minutes'
              and ix.observed_at <= d.observed_at + interval '7 minutes'
          )
        )
      ) as missing_300,
      (
        d.observed_at + interval '15 minutes' <= p_now
        and not exists (
          select 1
          from public.brian_alpha_decision_outcomes o
          where o.decision_id = d.decision_id
            and o.horizon_seconds = 900
        )
        and (
          p_now <= d.observed_at + interval '17 minutes'
          or exists (
            select 1
            from public.brian_alpha_decisions px
            where px.asset_id = d.asset_id
              and px.observed_reference_price is not null
              and px.observed_at >= d.observed_at + interval '15 minutes'
              and px.observed_at <= d.observed_at + interval '17 minutes'
          )
          or exists (
            select 1
            from public.brian_intrabar_reaction_events ix
            where ix.asset_id = d.asset_id
              and ix.observed_mid_price is not null
              and ix.observed_at >= d.observed_at + interval '15 minutes'
              and ix.observed_at <= d.observed_at + interval '17 minutes'
          )
        )
      ) as missing_900,
      (
        d.observed_at + interval '60 minutes' <= p_now
        and not exists (
          select 1
          from public.brian_alpha_decision_outcomes o
          where o.decision_id = d.decision_id
            and o.horizon_seconds = 3600
        )
        and (
          p_now <= d.observed_at + interval '62 minutes'
          or exists (
            select 1
            from public.brian_alpha_decisions px
            where px.asset_id = d.asset_id
              and px.observed_reference_price is not null
              and px.observed_at >= d.observed_at + interval '60 minutes'
              and px.observed_at <= d.observed_at + interval '62 minutes'
          )
          or exists (
            select 1
            from public.brian_intrabar_reaction_events ix
            where ix.asset_id = d.asset_id
              and ix.observed_mid_price is not null
              and ix.observed_at >= d.observed_at + interval '60 minutes'
              and ix.observed_at <= d.observed_at + interval '62 minutes'
          )
        )
      ) as missing_3600
    from public.brian_alpha_decisions d, cfg
    where d.observed_at >= cfg.oldest
      and d.observed_at <= p_now - interval '5 minutes'
  ),
  recent as materialized (
    select e.*
    from eligible e, cfg
    where e.missing_300 or e.missing_900 or e.missing_3600
    order by e.observed_at desc, e.decision_id desc
    limit (select recent_quota from cfg)
  ),
  q60 as materialized (
    select e.*
    from eligible e, cfg
    where e.missing_3600
      and not exists (
        select 1 from recent r where r.decision_id = e.decision_id
      )
    order by e.observed_at asc, e.decision_id asc
    limit (select lane_quota from cfg)
  ),
  q15 as materialized (
    select e.*
    from eligible e, cfg
    where e.missing_900
      and not exists (
        select 1 from recent r where r.decision_id = e.decision_id
      )
      and not exists (
        select 1 from q60 q where q.decision_id = e.decision_id
      )
    order by e.observed_at asc, e.decision_id asc
    limit (select lane_quota from cfg)
  ),
  q5 as materialized (
    select e.*
    from eligible e, cfg
    where e.missing_300
      and not exists (
        select 1 from recent r where r.decision_id = e.decision_id
      )
      and not exists (
        select 1 from q60 q where q.decision_id = e.decision_id
      )
      and not exists (
        select 1 from q15 q where q.decision_id = e.decision_id
      )
    order by e.observed_at asc, e.decision_id asc
    limit (select lane_quota from cfg)
  ),
  selected as materialized (
    select * from recent
    union all select * from q60
    union all select * from q15
    union all select * from q5
  ),
  fill as (
    select e.*
    from eligible e, cfg
    where (e.missing_300 or e.missing_900 or e.missing_3600)
      and not exists (
        select 1 from selected s where s.decision_id = e.decision_id
      )
    order by
      case
        when e.missing_3600 then 0
        when e.missing_900 then 1
        else 2
      end,
      e.observed_at asc,
      e.decision_id asc
    limit greatest(
      0,
      (select lim from cfg) - (select count(*) from selected)
    )
  ),
  picked as (
    select * from selected
    union all
    select * from fill
  )
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
  from picked p
  join public.brian_alpha_decisions d
    on d.decision_id = p.decision_id
  order by p.observed_at desc, p.decision_id desc
  limit (select lim from cfg);
$$;

revoke all on function public.brian_alpha_pending_audit_decisions(
  timestamptz, interval, integer
) from public, anon, authenticated;
grant execute on function public.brian_alpha_pending_audit_decisions(
  timestamptz, interval, integer
) to service_role;

comment on function public.brian_alpha_pending_audit_decisions(
  timestamptz, interval, integer
) is
'Service-role-only bounded ALPHA audit queue. After each horizon +120s terminal tolerance, permanently sparse rows stop consuming queue capacity unless an alpha/intrabar terminal point exists; current due rows retain the grace window. Shadow measurement only.';
