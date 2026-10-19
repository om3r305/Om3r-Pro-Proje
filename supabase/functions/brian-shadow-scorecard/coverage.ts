// Read-only diagnostics, never a strategy promotion or a price backfill.
export const COHORT = {
  id: 'alpha-coverage-20261019-1300z-v1',
  starts_at: '2026-10-19T13:00:00Z',
  ends_at: '2026-10-19T14:00:00Z',
  final_horizon_at: '2026-10-19T15:02:00Z',
  maximum_decisions: 1000,
  horizons_seconds: [300, 900, 3600],
  terminal_tolerance_seconds: 120,
  scope: 'All shadow ALPHA decisions observed in [start,end), including WAIT/VETO; first 1000 if bounded',
  criteria: 'Report every missing category by asset/horizon. Coverage is not profitability. No automatic promotion.',
} as const;

// $1 = snapshot time, $2/$3 = the frozen prospective window. Never accept client-supplied windows.
export const OUTCOME_COVERAGE_SQL = `
with cfg as (select $1::timestamptz at, $2::timestamptz starts_at, $3::timestamptz ends_at),
rolling_seed as materialized (
 select d.*, row_number() over(order by observed_at desc,decision_id) ordinal
 from public.brian_alpha_decisions d,cfg
 where observed_at>=cfg.at-interval '6 hours' and observed_at<=cfg.at and created_at<=cfg.at
 and shadow_only and not live_execution order by observed_at desc,decision_id limit 301
), cohort_seed as materialized (
 select d.*, row_number() over(order by observed_at,decision_id) ordinal
 from public.brian_alpha_decisions d,cfg
 where observed_at>=cfg.starts_at and observed_at<cfg.ends_at and observed_at<=cfg.at and created_at<=cfg.at
 and shadow_only and not live_execution order by observed_at,decision_id limit 1001
), d as materialized (
 select 'rolling'::text sample, r.* from rolling_seed r where ordinal<=300
 union all select 'prospective'::text sample,c.* from cohort_seed c where ordinal<=1000
), evidence as materialized (
 select d.sample,d.decision_id,d.asset_id,d.compiler_version,d.action,d.observed_at,d.observed_reference_price,
 h.seconds,cfg.at, o.outcome_id,o.resolved_at,o.reference_price,o.resolved_price,o.direction_adjusted_return,
 case when length(o.metadata->>'estimated_round_trip_cost_bps')<=80
 and (o.metadata->>'estimated_round_trip_cost_bps') ~ '^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]{1,3})?$'
 then case when (o.metadata->>'estimated_round_trip_cost_bps')::numeric<='1.7976931348623157e308'::numeric
   then (o.metadata->>'estimated_round_trip_cost_bps')::numeric else null end else null end outcome_cost,
 case when d.observed_at+make_interval(secs=>h.seconds)>cfg.at then false else (
 exists(select 1 from public.brian_alpha_decisions p where p.asset_id=d.asset_id
 and p.observed_reference_price>0 and p.observed_reference_price::text not in ('NaN','Infinity','-Infinity')
 and p.shadow_only and not p.live_execution and p.created_at<=cfg.at
 and p.observed_at between d.observed_at+make_interval(secs=>h.seconds) and least(cfg.at,d.observed_at+make_interval(secs=>h.seconds+120)))
 or exists(select 1 from public.brian_intrabar_reaction_events p where p.asset_id=d.asset_id
 and p.observed_mid_price>0 and p.observed_mid_price::text not in ('NaN','Infinity','-Infinity')
 and p.shadow_only and not p.live_execution and p.created_at<=cfg.at
 and p.observed_at between d.observed_at+make_interval(secs=>h.seconds) and least(cfg.at,d.observed_at+make_interval(secs=>h.seconds+120)))
 or exists(select 1 from public.brian_audit_price_samples p where p.asset_id=d.asset_id
 and p.observed_mid_price>0 and p.observed_mid_price::text not in ('NaN','Infinity','-Infinity')
 and p.shadow_only and not p.live_execution and p.captured_at<=cfg.at
 and p.observed_at between d.observed_at+make_interval(secs=>h.seconds) and least(cfg.at,d.observed_at+make_interval(secs=>h.seconds+120)))
 ) end terminal_present
 from d cross join (values(300),(900),(3600)) h(seconds) cross join cfg
 left join public.brian_alpha_decision_outcomes o on o.decision_id=d.decision_id and o.horizon_seconds=h.seconds
 and o.created_at<=cfg.at and o.resolved_at<=cfg.at and o.shadow_only and not o.live_execution
), tagged as (
 select *,case
 when observed_at+make_interval(secs=>seconds)>at then 'NOT_MATURE'
 when observed_reference_price is null or observed_reference_price<=0 or observed_reference_price::text in ('NaN','Infinity','-Infinity') then 'MISSING_ENTRY'
 when outcome_id is not null and (resolved_at<observed_at+make_interval(secs=>seconds)
   or resolved_at>observed_at+make_interval(secs=>seconds+120)
   or reference_price is null or reference_price<=0 or reference_price::text in ('NaN','Infinity','-Infinity')
   or reference_price<>observed_reference_price
   or resolved_price is null or resolved_price<=0 or resolved_price::text in ('NaN','Infinity','-Infinity')
   or direction_adjusted_return is null or direction_adjusted_return::text in ('NaN','Infinity','-Infinity')) then 'INVALID_OUTCOME'
 when outcome_id is not null then 'RESOLVED'
 when terminal_present then 'AWAITING_AUDITOR'
 when observed_at+make_interval(secs=>seconds+120)>=at then 'WINDOW_OPEN'
 else 'MISSING_TERMINAL' end status from evidence
), grouped as (
 select sample,asset_id,seconds,
 count(*)::int sampled,count(*) filter(where status<>'NOT_MATURE')::int mature,
 count(*) filter(where status='NOT_MATURE')::int not_mature,
 count(*) filter(where status='RESOLVED')::int resolved,
 count(*) filter(where status='MISSING_ENTRY')::int missing_entry,
 count(*) filter(where status='INVALID_OUTCOME')::int invalid_outcome,
 count(*) filter(where status='AWAITING_AUDITOR')::int awaiting_auditor,
 count(*) filter(where status='WINDOW_OPEN')::int window_open,
 count(*) filter(where status='MISSING_TERMINAL')::int missing_terminal,
 count(*) filter(where action in ('OPEN_LONG','OPEN_SHORT') and status='RESOLVED' and outcome_cost is null)::int missing_cost,
 count(*) filter(where action in ('OPEN_LONG','OPEN_SHORT') and status='RESOLVED' and outcome_cost is not null)::int cost_known_directional
 from tagged group by sample,asset_id,seconds
)
select jsonb_build_object('as_of',(select at from cfg),
 'rolling_truncated',(select count(*)>300 from rolling_seed),
 'prospective_truncated',(select count(*)>1000 from cohort_seed),
 'rows',coalesce((select jsonb_agg(to_jsonb(g) order by sample,seconds,asset_id) from grouped g),'[]'::jsonb),
 'compiler_versions',coalesce((select jsonb_agg(to_jsonb(v)) from
   (select sample,compiler_version,count(*)::int decisions from d group by sample,compiler_version order by sample,compiler_version) v),'[]'::jsonb)
) coverage`;
