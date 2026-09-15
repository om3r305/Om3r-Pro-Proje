create or replace view public.brian_world_source_latest
with (security_invoker = true)
as
select distinct on (source_id)
  candidate_id,
  source_id,
  discovered_at,
  canonical_uri,
  provider,
  authority_class,
  access_mode,
  stage
from public.brian_world_source_candidates
order by source_id, discovered_at desc, candidate_id desc;

create or replace view public.brian_world_source_assessment_latest
with (security_invoker = true)
as
select distinct on (source_id)
  source_id,
  assessed_at,
  trust_score,
  eligible_for_research
from public.brian_world_source_assessments
order by source_id, assessed_at desc;

revoke all on public.brian_world_source_latest from anon, authenticated;
revoke all on public.brian_world_source_assessment_latest from anon, authenticated;
grant select on public.brian_world_source_latest to service_role;
grant select on public.brian_world_source_assessment_latest to service_role;

comment on view public.brian_world_source_latest is 'Latest unique world-source candidate per source_id for Brian service-role status APIs.';
comment on view public.brian_world_source_assessment_latest is 'Latest unique world-source assessment per source_id for Brian service-role status APIs.';
