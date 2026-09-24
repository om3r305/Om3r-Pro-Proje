-- Brian ALPHA prospective outcome freshness recovery.
-- SHADOW ONLY. No live execution, capital, order, or promotion surface.
--
-- Problem repaired:
--   * the live auditor job had drifted to one run/hour although its bounded
--     queue can safely run every five minutes through the existing aux
--     backpressure dispatcher;
--   * oldest-first backlog selection could keep resolved_at chronically stale;
--   * action-priority ordering could bias calibration outcomes toward OPEN
--     decisions.
--
-- Contract:
--   * reserve 1/3 of each bounded batch for newest due decisions so current
--     prospective calibration stays fresh;
--   * reserve three 1/6 lanes for oldest missing 60m/15m/5m debt;
--   * use remaining capacity for oldest unresolved debt;
--   * never order by decision action, avoiding action-conditioned outcome bias;
--   * keep the worker batch bounded to [1,500], default 120;
--   * restore the auditor scheduler to every five minutes through
--     brian_private.enqueue_aux_service(), which already enforces inflight and
--     queue backpressure.

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
      p_now - greatest(coalesce(p_lookback, interval '12 hours'), interval '36 hours') as oldest
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
      ) as missing_300,
      (
        d.observed_at + interval '15 minutes' <= p_now
        and not exists (
          select 1
          from public.brian_alpha_decision_outcomes o
          where o.decision_id = d.decision_id
            and o.horizon_seconds = 900
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
) from public;
revoke all on function public.brian_alpha_pending_audit_decisions(
  timestamptz, interval, integer
) from anon;
revoke all on function public.brian_alpha_pending_audit_decisions(
  timestamptz, interval, integer
) from authenticated;
grant execute on function public.brian_alpha_pending_audit_decisions(
  timestamptz, interval, integer
) to service_role;

comment on function public.brian_alpha_pending_audit_decisions(
  timestamptz, interval, integer
) is
'Service-role-only bounded ALPHA audit queue: one-third recent freshness plus horizon-balanced oldest backlog, with no action-conditioned ordering.';

do $$
declare
  r record;
begin
  for r in
    select jobid
    from cron.job
    where jobname in (
      'brian-missed-opportunity-auditor-5m',
      'brian-missed-opportunity-auditor-v3-5m'
    )
  loop
    perform cron.unschedule(r.jobid);
  end loop;
end
$$;

select cron.schedule(
  'brian-missed-opportunity-auditor-v3-5m',
  '2-59/5 * * * *',
  $cron$
    select brian_private.enqueue_aux_service('missed_auditor');
  $cron$
);

-- DB-recovery throttling also drifted the two read/measurement-only reliability
-- jobs away from their original repository cadence. Restore those declared
-- cadences now that the outcome writer is again bounded and backpressured.
do $$
declare
  v_jobid bigint;
begin
  select jobid into v_jobid
  from cron.job
  where jobname = 'brian-sensor-reliability-shadow-hourly'
  limit 1;

  if v_jobid is not null then
    perform cron.alter_job(
      v_jobid,
      '12 * * * *',
      $cron$select public.brian_refresh_sensor_reliability_shadow(now(), interval '24 hours');$cron$,
      null,
      null,
      true
    );
  else
    perform cron.schedule(
      'brian-sensor-reliability-shadow-hourly',
      '12 * * * *',
      $cron$select public.brian_refresh_sensor_reliability_shadow(now(), interval '24 hours');$cron$
    );
  end if;

  select jobid into v_jobid
  from cron.job
  where jobname = 'brian-sensor-reliability-calibration-5m'
  limit 1;

  if v_jobid is not null then
    perform cron.alter_job(
      v_jobid,
      '4-59/5 * * * *',
      $cron$select public.brian_resolve_sensor_reliability_prospective_calibration(now(), interval '12 hours', 2000);$cron$,
      null,
      null,
      true
    );
  else
    perform cron.schedule(
      'brian-sensor-reliability-calibration-5m',
      '4-59/5 * * * *',
      $cron$select public.brian_resolve_sensor_reliability_prospective_calibration(now(), interval '12 hours', 2000);$cron$
    );
  end if;
end
$$;

