create or replace function brian_private.sync_source_v2_score_to_world()
returns trigger
language plpgsql
security definer
set search_path = public, brian_private, pg_temp
as $$
declare
  e public.brian_source_endpoints_v2%rowtype;
  aid text;
begin
  select * into e from public.brian_source_endpoints_v2 where endpoint_id=new.endpoint_id;
  if not found then return new; end if;

  update public.brian_world_source_candidates
  set stage = case when new.eligible_for_research then 'RESEARCHING' else 'VERIFYING' end,
      metadata = coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
        'source_arch_version','V2','endpoint_id',e.endpoint_id,'tier',e.tier,'category',e.category,
        'region',e.region,'official_origin',e.official_origin,
        'origin_verification_pending',not new.eligible_for_research,'decision_evidence_locked',true,
        'health_score',new.health_score,'composite_score',new.composite_score,'score_sample_count',new.sample_count
      )
  where candidate_id='source-arch-v2:'||e.endpoint_id;

  aid := 'source-v2:' || new.score_id;
  insert into public.brian_world_source_assessments(
    assessment_id,source_id,assessed_at,authority_score,freshness_score,manipulation_penalty,
    corroboration_penalty,access_penalty,trust_score,eligible_for_research,
    eligible_for_decision_evidence,reasons,metadata,evidence_class,shadow_only,live_execution
  ) values (
    aid,e.source_id,new.assessed_at,new.authority_score,new.health_score,new.manipulation_risk,
    case when e.corroboration_required then 0.10 else 0 end,
    case when e.access_mode='PUBLIC_NO_KEY' then 0 else 0.25 end,new.composite_score,
    new.eligible_for_research,false,new.reasons,
    new.metadata || jsonb_build_object('endpoint_id',e.endpoint_id,'tier',e.tier,'category',e.category,'region',e.region,'decision_evidence_locked',true),
    'PROSPECTIVE_EVOLUTION_SHADOW',true,false
  ) on conflict(assessment_id) do nothing;
  return new;
end;
$$;

drop trigger if exists brian_source_v2_score_world_sync on public.brian_source_scores_v2;
create trigger brian_source_v2_score_world_sync
after insert on public.brian_source_scores_v2
for each row execute function brian_private.sync_source_v2_score_to_world();
