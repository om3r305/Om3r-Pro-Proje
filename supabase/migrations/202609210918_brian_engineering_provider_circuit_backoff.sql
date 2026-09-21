CREATE OR REPLACE FUNCTION brian_private.enqueue_provider_resume_from_blocked_event()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'brian_private', 'extensions', 'pg_temp'
AS $function$
declare
  run_row public.brian_evolution_engineering_runs%rowtype;
  src public.brian_evolution_codegen_requests%rowtype;
  resume_request_id text;
  resume_after timestamptz := now()+interval '30 minutes';
  resume_iteration integer := 1;
  checkpoint_sha text := null;
  next_metadata jsonb;
begin
  if new.event_kind <> 'BLOCKED'
     or new.phase <> 'BLOCKED'
     or coalesce(new.payload->>'provider_exhausted','false') <> 'true' then
    return new;
  end if;

  select * into run_row
  from public.brian_evolution_engineering_runs
  where run_id=new.run_id;
  if not found then return new; end if;

  select * into src
  from public.brian_evolution_codegen_requests
  where request_id=run_row.request_id;
  if not found then return new; end if;

  select coalesce(max((r.metadata->>'provider_resume_iteration')::integer),0)+1
  into resume_iteration
  from public.brian_evolution_codegen_requests r
  where r.hypothesis_id=run_row.hypothesis_id
    and coalesce(r.metadata->>'provider_resume','false')='true'
    and coalesce(r.metadata->>'provider_resume_iteration','') ~ '^[0-9]+$';

  resume_after := now() + (interval '30 minutes' * power(2, least(greatest(resume_iteration-1,0),3)));
  checkpoint_sha := coalesce(run_row.commit_sha,nullif(src.metadata->>'failed_candidate_sha',''));
  resume_request_id := encode(digest(
    'brian-engineer-provider-resume|'||run_row.run_id::text||'|'||resume_iteration::text,
    'sha256'
  ),'hex');

  next_metadata := (coalesce(src.metadata,'{}'::jsonb) - 'auto_retry' - 'created_by') ||
    jsonb_build_object(
      'priority',1000,
      'provider_resume',true,
      'provider_resume_iteration',resume_iteration,
      'provider_resume_of_run_id',run_row.run_id::text,
      'retry_of_run_id',run_row.run_id::text,
      'retry_of_request_id',run_row.request_id,
      'provider_resume_after',resume_after,
      'provider_failure_stage',coalesce(new.payload->>'failure_stage','UNKNOWN'),
      'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
      'created_by','brian-engineering-provider-continuity',
      'production_grade_required',true,
      'must_resume_existing_candidate',checkpoint_sha is not null,
      'failed_candidate_sha',checkpoint_sha,
      'provider_failure_state',coalesce(new.payload->'provider_state','{}'::jsonb)
    );

  insert into public.brian_evolution_codegen_requests(
    request_id,candidate_id,hypothesis_id,requested_at,parent_commit,branch_name,
    changed_paths,objective,constraints,success_criteria,evidence_refs,
    contamination_declaration,external_generator_required,required_human_review,
    metadata,evidence_class,shadow_only,live_execution,autonomous_apply_allowed
  ) values (
    resume_request_id,
    'code|engineer-provider-resume|'||substr(resume_request_id,1,20),
    run_row.hypothesis_id,
    now(),
    run_row.base_sha,
    'evolution-candidate/provider-resume-'||substr(resume_request_id,1,12),
    src.changed_paths,
    src.objective||E'\n\nProvider continuity: resume the same engineering objective/checkpoint after an AI-provider outage; do not redesign merely because the provider changed.',
    src.constraints,
    src.success_criteria,
    array_append(coalesce(src.evidence_refs,'{}'::text[]),'engineering-run:'||run_row.run_id::text||':provider-exhausted'),
    coalesce(src.contamination_declaration,'Point-in-time evidence only; provider outage is infrastructure metadata, not performance evidence.'),
    true,true,next_metadata,'PROSPECTIVE_EVOLUTION_SHADOW',true,false,false
  ) on conflict (request_id) do nothing;

  update public.brian_evolution_engineering_control
  set metadata=coalesce(metadata,'{}'::jsonb)||jsonb_build_object(
        'provider_continuity_enabled',true,
        'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
        'provider_resume_claim_limit_24h',coalesce(metadata->>'provider_resume_claim_limit_24h','3'),
        'provider_circuit_open_until',resume_after,
        'provider_circuit_reason','ALL_AI_PROVIDERS_UNAVAILABLE',
        'provider_circuit_last_run_id',run_row.run_id::text,
        'provider_circuit_updated_at',now()
      ),
      updated_at=now()
  where control_id='default';

  return new;
end;
$function$
;
