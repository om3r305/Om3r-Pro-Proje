export const EDGE_PROTOCOL = {
 id:'alpha-net-probe-20261019-v1',starts_at:'2026-10-19T16:00:00Z',ends_at:'2026-11-18T16:00:00Z',final_at:'2026-11-18T16:17:00Z',
 horizon_seconds:900,terminal_tolerance_seconds:120,slot_seconds:1080,entry_window_seconds:60,
 selection:'First shadow decision per asset in the first 60 seconds of each fixed 18-minute slot; all actions retained',
 benchmark:'Cash = 0 basis points; modeled decision returns, not portfolio P&L',minimum_directional_samples:100,minimum_day_blocks:20,
 proof_status:'NOT_PROVEN',automatic_promotion:false,modeled_costs_only:true,
} as const;
// $1 snapshot; $2/$3 frozen prospective dates. The diagnostic window uses fixed UTC slots too.
export const NET_VALIDATION_SQL=`
with cfg as(select $1::timestamptz at,$2::timestamptz starts_at,$3::timestamptz ends_at),
scopes as(select 'diagnostic'::text sample,at-interval '6 hours' lo,at hi,'2026-10-19T00:00:00Z'::timestamptz anchor from cfg
 union all select 'prospective',starts_at,least(at,ends_at),starts_at from cfg),
seed as materialized(
 select distinct on(s.sample,d.asset_id,floor(extract(epoch from(d.observed_at-s.anchor))/1080))
 s.sample,d.decision_id,d.asset_id,d.compiler_version,d.observed_at,d.action,d.observed_reference_price
 from scopes s join public.brian_alpha_decisions d on d.observed_at>=s.lo and d.observed_at<s.hi cross join cfg
 where d.created_at<=cfg.at and d.shadow_only and not d.live_execution and d.asset_id ~ '^crypto:[A-Z0-9]+USDT$'
 and mod(extract(epoch from(d.observed_at-s.anchor)),1080)>=0 and mod(extract(epoch from(d.observed_at-s.anchor)),1080)<60
 order by s.sample,d.asset_id,floor(extract(epoch from(d.observed_at-s.anchor))/1080),d.observed_at,d.decision_id limit 20001
), d as materialized(select * from seed order by sample,asset_id,observed_at,decision_id limit 20000),
measured as(
 select d.*,cfg.at,o.outcome_id,o.resolved_at,o.reference_price,o.resolved_price,o.direction_adjusted_return,
 case when length(o.metadata->>'estimated_round_trip_cost_bps')<=80 and (o.metadata->>'estimated_round_trip_cost_bps') ~ '^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]{1,3})?$'
 then case when (o.metadata->>'estimated_round_trip_cost_bps')::numeric<='1.7976931348623157e308'::numeric then (o.metadata->>'estimated_round_trip_cost_bps')::numeric end end cost
 from d cross join cfg left join public.brian_alpha_decision_outcomes o on o.decision_id=d.decision_id and o.horizon_seconds=900
 and o.created_at<=cfg.at and o.resolved_at<=cfg.at and o.shadow_only and not o.live_execution
), tagged as(
 select *,case when action not in('OPEN_LONG','OPEN_SHORT') then 'CASH_DECISION'
 when observed_at+interval '15 minutes'>at then 'NOT_MATURE'
 when observed_reference_price is null or observed_reference_price<=0 or observed_reference_price::text in('NaN','Infinity','-Infinity') then 'MISSING_ENTRY'
 when outcome_id is null then 'UNRESOLVED'
 when resolved_at<observed_at+interval '15 minutes' or resolved_at>observed_at+interval '17 minutes'
 or reference_price is null or reference_price<>observed_reference_price or resolved_price is null or resolved_price<=0 or resolved_price::text in('NaN','Infinity','-Infinity')
 or direction_adjusted_return is null or direction_adjusted_return::text in('NaN','Infinity','-Infinity') then 'INVALID_OUTCOME'
 when cost is null then 'MISSING_COST' else 'COST_KNOWN' end status from measured
), daily as(
 select sample,(observed_at at time zone 'UTC')::date as day_bucket,count(*)::int n,
 avg(direction_adjusted_return*10000-cost) mean_net_bps
 from tagged where status='COST_KNOWN' group by sample,(observed_at at time zone 'UTC')::date
), grouped as(
 select s.sample,count(t.decision_id)::int selected,count(*) filter(where t.status='CASH_DECISION')::int cash_decisions,
 count(*) filter(where t.action in('OPEN_LONG','OPEN_SHORT'))::int directional,
 count(*) filter(where t.status='NOT_MATURE')::int not_mature,
 count(*) filter(where t.status='MISSING_ENTRY')::int missing_entry,
 count(*) filter(where t.status='UNRESOLVED')::int unresolved,
 count(*) filter(where t.status='INVALID_OUTCOME')::int invalid_outcome,
 count(*) filter(where t.status='MISSING_COST')::int missing_cost,
 count(*) filter(where t.status='COST_KNOWN')::int measured,
 avg(direction_adjusted_return*10000-cost) filter(where t.status='COST_KNOWN') mean_net_bps,
 count(distinct t.compiler_version)::int compiler_versions,
 coalesce((select jsonb_agg(to_jsonb(z) order by z.day_bucket) from daily z where z.sample=s.sample),'[]'::jsonb) day_blocks
 from scopes s left join tagged t on t.sample=s.sample group by s.sample
)
select jsonb_build_object('as_of',(select at from cfg),'truncated',(select count(*)>20000 from seed),'rows',(select jsonb_agg(to_jsonb(g) order by sample) from grouped g)) validation`;
type Measurement={sample:string;selected:number;directional:number;measured:number;not_mature:number;missing_entry:number;unresolved:number;invalid_outcome:number;missing_cost:number;mean_net_bps:number|string|null;compiler_versions:number;day_blocks:unknown[]};
export function assessAdvantage(value:{as_of:string;truncated:boolean;rows:Measurement[]}) {
 return {...value,status:'AVAILABLE',protocol:EDGE_PROTOCOL,proof_status:'NOT_PROVEN',rows:value.rows.map(r=>{
  const mean=r.mean_net_bps===null?null:Number(r.mean_net_bps),issues:string[]=[];
  if(value.truncated)issues.push('BOUNDED_SAMPLE_PARTIAL');
  if(r.measured<100)issues.push('INSUFFICIENT_DIRECTIONAL_SAMPLE');
  if(r.day_blocks.length<20)issues.push('INSUFFICIENT_DAY_BLOCKS');
  if(r.directional!==r.measured)issues.push('DIRECTIONAL_COVERAGE_INCOMPLETE');
  if(r.compiler_versions!==1)issues.push('MIXED_OR_UNKNOWN_POLICY_VERSIONS');
  if(r.sample!=='prospective')issues.push('ROLLING_DIAGNOSTIC_NOT_PROSPECTIVE_PROOF');
  if(Date.parse(value.as_of)<Date.parse(EDGE_PROTOCOL.final_at))issues.push('FIXED_PROTOCOL_NOT_FINISHED');
  issues.push('MODELED_COSTS_NOT_EXECUTED_TRADES','INDEPENDENT_REPLICATION_REQUIRED');
  return {...r,mean_net_bps:Number.isFinite(mean)?mean:null,interpretation:mean===null||!Number.isFinite(mean)?'NO_MEASUREMENT':mean<=0?'NON_POSITIVE_NET_MEASURED':'POSITIVE_SAMPLE_NOT_PROOF',issues,proof_status:'NOT_PROVEN'};
 }),automatic_promotion:false};
}
