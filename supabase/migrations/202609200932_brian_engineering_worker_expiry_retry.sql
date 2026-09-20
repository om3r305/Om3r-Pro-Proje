
CREATE OR REPLACE FUNCTION brian_private.reap_expired_engineering_runs()
RETURNS integer
LANGUAGE plpgsql
SET search_path TO 'pg_catalog', 'public', 'brian_private'
AS $function$
declare
  r public.brian_evolution_engineering_runs%rowtype;
  req public.brian_evolution_codegen_requests%rowtype;
  recovered integer := 0;
  worker_retry_count integer := 0;
begin
  -- Use the same mutex as claim_engineering_task before changing occupancy.
  perform 1
  from public.brian_evolution_engineering_control
  where control_id='default'
  for update;

  for r in
    select *
    from public.brian_evolution_engineering_runs
    where status='RUNNING'
      and phase in ('CLAIMED','UNDERSTAND','PLAN','CODE','COMPILE','TEST','REPLAY','REVIEW','PR','PREVIEW','MEASURE')
      and worker_id like 'github-actions:%'
      and created_at < now()-interval '65 minutes'
      and updated_at < now()-interval '65 minutes'
    order by created_at
    for update skip locked
  loop
    perform public.record_engineering_event(
      r.run_id,
      'WORKER_EXPIRED',
      'BLOCKED',
      false,
      null,
      jsonb_build_object(
        'error','GitHub worker expired without terminal event; stale queue occupancy released',
        'previous_phase',r.phase,
        'last_progress_at',r.updated_at,
        'recovery_policy','PRE_APPROVAL_65_MINUTES',
        'evidence','No progress beyond GitHub job maximum runtime plus 20 minute grace'
      )
    );

    select *
    into req
    from public.brian_evolution_codegen_requests
    where request_id=r.request_id;

    if found
      and req.shadow_only
      and not req.live_execution
      and req.required_human_review
      and not req.autonomous_apply_allowed
    then
      -- First stale recovery remains unchanged.
      if coalesce(req.metadata->>'stale_recovery_attempt','0')='0' then
        insert into public.brian_evolution_codegen_requests(
          request_id,candidate_id,hypothesis_id,requested_at,contamination_declaration,
          parent_commit,branch_name,changed_paths,objective,constraints,success_criteria,
          evidence_refs,required_human_review,external_generator_required,
          shadow_only,live_execution,autonomous_apply_allowed,metadata
        )
        values(
          'stale-recovery-'||r.run_id::text,
          req.candidate_id,
          req.hypothesis_id,
          now(),
          req.contamination_declaration,
          req.parent_commit,
          'evolution-candidate/stale-recovery-'||r.run_id::text,
          req.changed_paths,
          req.objective,
          req.constraints,
          req.success_criteria,
          req.evidence_refs,
          true,true,true,false,false,
          req.metadata || jsonb_strip_nulls(jsonb_build_object(
            'stale_recovery_attempt',1,
            'stale_recovery_of_run',r.run_id,
            'provider_resume',true,
            'provider_resume_after',now(),
            'failed_candidate_sha',coalesce(r.commit_sha,req.metadata->>'failed_candidate_sha'),
            'must_resume_existing_candidate',coalesce(r.commit_sha,req.metadata->>'failed_candidate_sha') is not null
          ))
        )
        on conflict(request_id) do nothing;

      -- A provider-resume run that itself expires before producing a commit gets
      -- one infrastructure-only retry. This does not consume a functional retry
      -- and is bounded to one attempt across the lineage.
      elsif coalesce(req.metadata->>'provider_resume','false')='true'
        and r.commit_sha is null
      then
        begin
          worker_retry_count := coalesce(nullif(req.metadata->>'worker_expiry_retry_count','')::integer,0);
        exception when others then
          worker_retry_count := 0;
        end;

        if worker_retry_count < 1 then
          insert into public.brian_evolution_codegen_requests(
            request_id,candidate_id,hypothesis_id,requested_at,contamination_declaration,
            parent_commit,branch_name,changed_paths,objective,constraints,success_criteria,
            evidence_refs,required_human_review,external_generator_required,
            shadow_only,live_execution,autonomous_apply_allowed,metadata
          )
          values(
            'infra-recovery-'||r.run_id::text,
            req.candidate_id,
            req.hypothesis_id,
            now(),
            req.contamination_declaration,
            req.parent_commit,
            'evolution-candidate/infra-recovery-'||r.run_id::text,
            req.changed_paths,
            req.objective,
            req.constraints,
            req.success_criteria,
            req.evidence_refs,
            true,true,true,false,false,
            req.metadata || jsonb_strip_nulls(jsonb_build_object(
              'provider_resume',true,
              'provider_resume_after',now(),
              'worker_expiry_retry',true,
              'worker_expiry_retry_count',worker_retry_count+1,
              'retry_of_run_id',r.run_id,
              'retry_of_request_id',req.request_id,
              'created_by','brian-engineering-infra-recovery',
              'failed_candidate_sha',coalesce(r.commit_sha,req.metadata->>'failed_candidate_sha',req.parent_commit),
              'must_resume_existing_candidate',true
            ))
          )
          on conflict(request_id) do nothing;
        end if;
      end if;
    end if;

    recovered := recovered + 1;
  end loop;

  return recovered;
end;
$function$;
