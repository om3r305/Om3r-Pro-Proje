create table public.brian_world_evidence_reviews (
 review_id text primary key,event_id text not null references public.brian_intel_events(event_id),
 assessed_at timestamptz not null,captured_at timestamptz not null default clock_timestamp(),
 version text not null,status text not null check(status in ('READ','INDEX_ONLY','FAILED')),
 document_uri text,content_sha256 text,review jsonb not null check(jsonb_typeof(review)='object'),
 shadow_only boolean not null default true check(shadow_only),live_execution boolean not null default false check(not live_execution),
 check(assessed_at<=captured_at)
);
create index brian_world_evidence_reviews_event_time on public.brian_world_evidence_reviews(event_id,captured_at desc);
create index brian_world_evidence_reviews_time on public.brian_world_evidence_reviews(captured_at desc);
alter table public.brian_world_evidence_reviews enable row level security;
revoke all on public.brian_world_evidence_reviews from public,anon,authenticated;
grant select,insert on public.brian_world_evidence_reviews to service_role;
create trigger brian_world_evidence_reviews_append_only before update or delete on public.brian_world_evidence_reviews
 for each row execute function public.brian_reject_mutation();

create function public.brian_world_research_inputs_v1() returns jsonb
language sql stable security invoker set search_path=pg_catalog,public as $fn$
with cfg as(select statement_timestamp() at),
candidate as materialized(
 select e.event_id,e.source_id,e.claim,e.provenance_uri,e.published_at,e.first_observed_at,s.canonical_domain,
 row_number() over(partition by e.source_id order by e.first_observed_at,e.event_id) source_rank,
 (select max(r.captured_at) from public.brian_world_evidence_reviews r join public.brian_intel_events z on z.event_id=r.event_id where z.source_id=e.source_id) last_source_review
 from public.brian_intel_events e join public.brian_source_endpoints_v2 s on s.source_id=e.source_id cross join cfg
 where e.first_observed_at between cfg.at-interval '24 hours' and cfg.at and e.created_at<=cfg.at
 and e.trust_class='OFFICIAL_PRIMARY' and s.tier='T1_OFFICIAL_PRIMARY' and s.lifecycle_state='ACTIVE' and s.access_mode='PUBLIC_NO_KEY'
 and not exists(select 1 from public.brian_world_evidence_reviews r where r.event_id=e.event_id
   and r.captured_at>cfg.at-case when r.status='FAILED' then interval '1 hour' else interval '24 hours' end)
), selected as materialized(
 select * from candidate where source_rank=1 order by last_source_review nulls first,first_observed_at limit 2
), events as(
 select x.*,coalesce((select jsonb_agg(to_jsonb(z)) from (
   select event_id,source_id,claim,provenance_uri,published_at,first_observed_at
   from public.brian_intel_events p,cfg where p.event_id<>x.event_id
    and p.first_observed_at between x.first_observed_at-interval '24 hours' and cfg.at and p.created_at<=cfg.at
    and (x.source_id='official:sec:edgar_current' or p.source_id<>'official:sec:edgar_current')
    and exists(select 1 from regexp_split_to_table(lower(x.claim),'[^a-z0-9]+') w
      where length(w)>3 and w not in('this','that','with','from','have','will','their','about','after','before','press','release','statement')
        and lower(p.claim) like '%'||w||'%')
   order by p.first_observed_at desc,p.event_id limit 100) z),'[]'::jsonb) counter_pool,
 coalesce((select jsonb_agg(to_jsonb(q)) from(
  select a.asset_id,(select coalesce(jsonb_agg(to_jsonb(p) order by p.observed_at),'[]'::jsonb) from(
   select * from(
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
