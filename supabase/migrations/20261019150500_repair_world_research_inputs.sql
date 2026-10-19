create or replace function public.brian_world_research_inputs_v1() returns jsonb
language sql stable security invoker set search_path=pg_catalog,public as $fn$
with cfg as(select statement_timestamp() at),
source_reviews as materialized(
 select z.source_id,max(r.captured_at) last_source_review
 from public.brian_world_evidence_reviews r join public.brian_intel_events z on z.event_id=r.event_id group by z.source_id
), endpoints as materialized(
 select distinct source_id,canonical_domain from public.brian_source_endpoints_v2
 where tier='T1_OFFICIAL_PRIMARY' and lifecycle_state='ACTIVE' and access_mode='PUBLIC_NO_KEY'
), selected as materialized(
 select e.*,s.canonical_domain,r.last_source_review from endpoints s cross join cfg
 left join source_reviews r on r.source_id=s.source_id
 cross join lateral(
  select e.event_id,e.source_id,e.claim,e.provenance_uri,e.published_at,e.first_observed_at
  from public.brian_intel_events e where e.source_id=s.source_id
  and e.first_observed_at between cfg.at-interval '24 hours' and cfg.at and e.created_at<=cfg.at
  and e.trust_class='OFFICIAL_PRIMARY'
  and not exists(select 1 from public.brian_world_evidence_reviews v where v.event_id=e.event_id
    and v.captured_at>cfg.at-case when v.status='FAILED' then interval '1 hour' else interval '24 hours' end)
  order by e.first_observed_at,e.event_id limit 1
 ) e
 order by r.last_source_review nulls first,e.first_observed_at,e.event_id limit 2
), recent_pool as materialized(
 select event_id,source_id,claim,provenance_uri,published_at,first_observed_at,created_at
 from public.brian_intel_events,cfg where first_observed_at between cfg.at-interval '48 hours' and cfg.at and created_at<=cfg.at
 order by first_observed_at desc,event_id limit 2001
), bounded_pool as materialized(select * from recent_pool order by first_observed_at desc,event_id limit 2000)
, events as(
 select x.*,2000 counter_pool_scan_limit,(select count(*)>2000 from recent_pool) counter_pool_scan_truncated,coalesce((select jsonb_agg(to_jsonb(z)) from (
   select event_id,source_id,claim,provenance_uri,published_at,first_observed_at
   from bounded_pool p,cfg where p.event_id<>x.event_id
    and p.first_observed_at between x.first_observed_at-interval '24 hours' and cfg.at and p.created_at<=cfg.at
    and (x.source_id='official:sec:edgar_current' or p.source_id<>'official:sec:edgar_current')
    and exists(select 1 from regexp_split_to_table(lower(x.claim),'[^a-z0-9]+') w
      where length(w)>3 and w not in('this','that','with','from','have','will','their','about','after','before','press','release','statement')
        and lower(p.claim) like '%'||w||'%')
   order by p.first_observed_at desc,p.event_id limit 100) z),'[]'::jsonb) counter_pool,
 coalesce((select jsonb_agg(to_jsonb(q)) from(
  select a.asset_id,(select coalesce(jsonb_agg(to_jsonb(p) order by p.observed_at),'[]'::jsonb) from(
   select observed_at,captured_at,price,source from(
    select observed_at,created_at captured_at,observed_reference_price price,'ALPHA_REFERENCE'::text source from public.brian_alpha_decisions
      where asset_id=a.asset_id and shadow_only and not live_execution
    union all select observed_at,created_at,observed_mid_price,'INTRABAR' from public.brian_intrabar_reaction_events
      where asset_id=a.asset_id and shadow_only and not live_execution
    union all select observed_at,captured_at,observed_mid_price,'INDEPENDENT_AUDIT' from public.brian_audit_price_samples
      where asset_id=a.asset_id and shadow_only and not live_execution
   ) prices,cfg where price>0 and price::text not in('NaN','Infinity','-Infinity') and captured_at<=cfg.at and observed_at<=cfg.at
   and (observed_at between x.first_observed_at-interval '17 minutes' and x.first_observed_at+interval '17 minutes'
     or observed_at between coalesce(x.published_at,x.first_observed_at)-interval '17 minutes' and coalesce(x.published_at,x.first_observed_at)+interval '2 minutes')
   order by observed_at desc limit 300
  ) p) points from(values('crypto:BTCUSDT'),('crypto:ETHUSDT')) a(asset_id)
 ) q),'[]'::jsonb) probes from selected x
)
select jsonb_build_object('as_of',(select at from cfg),'events',coalesce((select jsonb_agg(to_jsonb(e)) from events e),'[]'::jsonb))
$fn$;
revoke all on function public.brian_world_research_inputs_v1() from public,anon,authenticated;
grant execute on function public.brian_world_research_inputs_v1() to service_role;
