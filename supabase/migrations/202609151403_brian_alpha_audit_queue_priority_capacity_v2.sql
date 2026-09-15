-- Keep prospective outcome evidence from starving under growing ALPHA decision volume.
-- OPEN decisions are evidence-critical, oldest debt drains first, and the queue retains a 36h recovery window.
create or replace function public.brian_alpha_pending_audit_decisions(
  p_now timestamptz default now(),
  p_lookback interval default interval '12 hours',
  p_limit integer default 120
)
returns table(
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
as $function$
  with cfg as (
    select
      greatest(1, least(coalesce(p_limit, 120), 500))::int as lim,
      greatest(1, ceil(greatest(1, least(coalesce(p_limit, 120), 500)) / 3.0))::int as quota
  ), eligible as (
    select
      d.decision_id,
      d.observed_at,
      d.asset_id,
      d.observed_reference_price,
      d.action,
      d.direction,
      d.evidence_score,
      d.estimated_round_trip_cost_bps,
      d.source_observation_ids,
      case when d.action in ('OPEN_LONG','OPEN_SHORT') then 0 else 1 end as action_priority,
      (d.observed_at + interval '5 minutes' <= p_now and not exists (
        select 1 from public.brian_alpha_decision_outcomes o where o.decision_id=d.decision_id and o.horizon_seconds=300
      )) as missing_300,
      (d.observed_at + interval '15 minutes' <= p_now and not exists (
        select 1 from public.brian_alpha_decision_outcomes o where o.decision_id=d.decision_id and o.horizon_seconds=900
      )) as missing_900,
      (d.observed_at + interval '60 minutes' <= p_now and not exists (
        select 1 from public.brian_alpha_decision_outcomes o where o.decision_id=d.decision_id and o.horizon_seconds=3600
      )) as missing_3600
    from public.brian_alpha_decisions d
    where d.observed_at >= p_now - greatest(coalesce(p_lookback, interval '12 hours'), interval '36 hours')
      and d.observed_at <= p_now - interval '5 minutes'
  ), q60 as (
    select e.* from eligible e, cfg where e.missing_3600
    order by e.action_priority asc, e.observed_at asc, e.decision_id asc
    limit (select quota from cfg)
  ), q15 as (
    select e.* from eligible e, cfg
    where e.missing_900 and not exists (select 1 from q60 q where q.decision_id=e.decision_id)
    order by e.action_priority asc, e.observed_at asc, e.decision_id asc
    limit (select quota from cfg)
  ), q5 as (
    select e.* from eligible e, cfg
    where e.missing_300
      and not exists (select 1 from q60 q where q.decision_id=e.decision_id)
      and not exists (select 1 from q15 q where q.decision_id=e.decision_id)
    order by e.action_priority asc, e.observed_at asc, e.decision_id asc
    limit (select quota from cfg)
  ), selected as (
    select * from q60 union all select * from q15 union all select * from q5
  ), fill as (
    select e.* from eligible e, cfg
    where (e.missing_300 or e.missing_900 or e.missing_3600)
      and not exists (select 1 from selected s where s.decision_id=e.decision_id)
    order by e.action_priority asc,
      case when e.missing_3600 then 0 when e.missing_900 then 1 else 2 end,
      e.observed_at asc, e.decision_id asc
    limit greatest(0, (select lim from cfg) - (select count(*) from selected))
  )
  select x.decision_id,x.observed_at,x.asset_id,x.observed_reference_price,x.action,x.direction,x.evidence_score,x.estimated_round_trip_cost_bps,x.source_observation_ids
  from (select * from selected union all select * from fill) x
  limit (select lim from cfg);
$function$;

select cron.alter_job(18, schedule := '6,21,36,51 * * * *');
