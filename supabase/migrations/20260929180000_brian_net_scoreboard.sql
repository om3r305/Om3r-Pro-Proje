-- Target: brian-market-intelligence. One honest scoreboard: every OPEN decision
-- scored net of the contemporaneous round-trip cost recorded by the auditor
-- (metadata.estimated_round_trip_cost_bps). Nothing is promoted toward capital
-- unless its 30-day row shows net_bps > 0 with t_net >= 2.
create or replace view public.brian_scoreboard_net_daily with (security_invoker=true) as
select
  date_trunc('day',o.observed_at)::date as day,
  o.horizon_seconds,
  o.metadata->>'original_action' as action,
  count(*) as decisions,
  round(avg((o.metadata->>'estimated_round_trip_cost_bps')::float8)::numeric,2) as cost_bps,
  round(avg(o.direction_adjusted_return*1e4)::numeric,2) as gross_bps,
  round(avg(o.direction_adjusted_return*1e4-(o.metadata->>'estimated_round_trip_cost_bps')::float8)::numeric,2) as net_bps,
  round(avg(((o.direction_adjusted_return*1e4-(o.metadata->>'estimated_round_trip_cost_bps')::float8)>0)::int)::numeric,3) as win_rate
from public.brian_alpha_decision_outcomes o
where o.classification like 'ACTION%'
  and o.metadata ? 'estimated_round_trip_cost_bps'
group by 1,2,3;

create or replace view public.brian_scoreboard_net_30d with (security_invoker=true) as
with s as (
  select o.horizon_seconds, o.metadata->>'original_action' as action,
         o.direction_adjusted_return*1e4 as gross_bps,
         (o.metadata->>'estimated_round_trip_cost_bps')::float8 as cost_bps
  from public.brian_alpha_decision_outcomes o
  where o.classification like 'ACTION%'
    and o.metadata ? 'estimated_round_trip_cost_bps'
    and o.observed_at > now()-interval '30 days'
)
select horizon_seconds, action, count(*) as decisions,
  round(avg(cost_bps)::numeric,2) as cost_bps,
  round(avg(gross_bps)::numeric,2) as gross_bps,
  round(avg(gross_bps-cost_bps)::numeric,2) as net_bps,
  round((avg(gross_bps-cost_bps)/nullif(stddev(gross_bps-cost_bps),0)*sqrt(count(*)))::numeric,2) as t_net,
  round(avg(((gross_bps-cost_bps)>0)::int)::numeric,3) as win_rate,
  case when avg(gross_bps-cost_bps)>0 and avg(gross_bps-cost_bps)/nullif(stddev(gross_bps-cost_bps),0)*sqrt(count(*))>=2
       then 'EDGE' when avg(gross_bps)>0 then 'GROSS_ONLY' else 'NO_EDGE' end as verdict
from s group by 1,2;

revoke all on public.brian_scoreboard_net_daily, public.brian_scoreboard_net_30d from anon, authenticated;
grant select on public.brian_scoreboard_net_daily, public.brian_scoreboard_net_30d to service_role;
