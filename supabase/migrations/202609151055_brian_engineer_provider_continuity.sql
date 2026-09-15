-- Brian Engineer provider continuity. DIP is intentionally untouched.

create or replace function brian_private.engineering_request_is_retry(p_metadata jsonb)
returns boolean
language sql
immutable
set search_path to 'public','brian_private','pg_temp'
as $$
  select case
    when coalesce(p_metadata->>'provider_resume','false') = 'true' then false
    else
      coalesce(p_metadata->>'auto_retry','false') = 'true'
      or coalesce(p_metadata,'{}'::jsonb) ?| array[
        'retry_of_request_id','retry_of_run_id','prior_request_id','prior_failed_run_id',
        'failed_run_id','retry_reason','compile_retry','test_retry','retry_base_sha'
      ]
      or coalesce(p_metadata->>'retry_iteration','') ~ '^[0-9]+$'
  end;
$$;

alter function brian_private.claim_engineering_task(text,text,text)
  rename to claim_engineering_task_fair_queue_v2;

create function brian_private.claim_engineering_task(
  p_worker_id text,
  p_base_sha text,
  p_request_id text default null
)
returns jsonb
language plpgsql
security definer
set search_path to 'public','brian_private','pg_temp'
as $$
declare
  cfg public.brian_evolution_engineering_control%rowtype;
  req public.brian_evolution_codegen_requests%rowtype;
  created_run public.brian_evolution_engineering_runs%rowtype;
  active_count integer := 0;
  provider_resume_count_24h integer := 0;
  provider_resume_limit_24h integer := 8;
  circuit_until timestamptz := null;
begin
  if coalesce(trim(p_worker_id),'') = '' then raise exception 'worker_id required'; end if;
  if p_base_sha !~ '^[0-9a-fA-F]{40}$' then raise exception 'base_sha must be a full git SHA'; end if;

  if p_request_id is not null then
    return brian_private.claim_engineering_task_fair_queue_v2(p_worker_id,p_base_sha,p_request_id);
  end if;

  select * into cfg
  from public.brian_evolution_engineering_control
  where control_id='default'
  for update;
  if not found then raise exception 'engineering control row missing'; end if;
  if cfg.require_human_approval is not true then raise exception 'human approval invariant disabled'; end if;
  if cfg.autonomous_claim_enabled is not true then return null; end if;

  if coalesce(cfg.metadata->>'provider_resume_claim_limit_24h','') ~ '^[0-9]+$' then
    provider_resume_limit_24h := greatest(1,least(24,(cfg.metadata->>'provider_resume_claim_limit_24h')::integer));
  end if;

  if nullif(cfg.metadata->>'provider_circuit_open_until','') is not null then
    begin
      circuit_until := (cfg.metadata->>'provider_circuit_open_until')::timestamptz;
    exception when others then
      circuit_until := null;
    end;
  end if;

  if circuit_until is not null and circuit_until > now() then return null; end if;

  select count(*) into active_count
  from public.brian_evolution_engineering_runs
  where status in ('RUNNING','WAITING')
    and phase not in ('HUMAN_APPROVAL','COMPLETE','BLOCKED','ROLLBACK');
  if active_count >= cfg.max_concurrent_runs then return null; end if;

  select count(*) into provider_resume_count_24h
  from public.brian_evolution_engineering_runs er
  join public.brian_evolution_codegen_requests r on r.request_id=er.request_id
  where er.created_at >= now()-interval '24 hours'
    and er.metadata->>'claim_mode'='AUTONOMOUS'
    and coalesce(r.metadata->>'provider_resume','false')='true';

  if provider_resume_count_24h < provider_resume_limit_24h then
    select r.* into req
    from public.brian_evolution_codegen_requests r
    left join public.brian_evolution_engineering_runs er on er.request_id=r.request_id
    where er.request_id is null
      and coalesce(r.metadata->>'provider_resume','false')='true'
      and r.required_human_review is true
      and r.shadow_only is true
      and r.live_execution is false
      and r.autonomous_apply_allowed is false
      and coalesce((r.metadata->>'provider_resume_after')::timestamptz,now()) <= now()
    order by
      case when jsonb_typeof(r.metadata->'priority')='number' then (r.metadata->>'priority')::numeric else 0 end desc,
      r.requested_at asc,
      r.request_id asc
    for update of r skip locked
    limit 1;

    if found then
      insert into public.brian_evolution_engineering_runs(
        request_id,candidate_id,hypothesis_id,worker_id,base_branch,base_sha,
        source_parent_sha,branch_name,previous_good_sha,metadata
      ) values (
        req.request_id,req.candidate_id,req.hypothesis_id,p_worker_id,cfg.base_branch,p_base_sha,
        req.parent_commit,'brian-engineer/'||substr(req.request_id,1,16),p_base_sha,
        jsonb_build_object(
          'requested_at',req.requested_at,'objective',req.objective,'constraints',req.constraints,
          'success_criteria',req.success_criteria,'evidence_refs',req.evidence_refs,
          'requested_changed_paths',req.changed_paths,'requested_branch_name',req.branch_name,
          'request_metadata',req.metadata,'source_parent_stale',req.parent_commit<>p_base_sha,
          'queue_tie_break','provider-resume-first','transport_retry_safe',true,
          'claim_mode','AUTONOMOUS','request_class','PROVIDER_RESUME',
          'provider_resume_claim_limit_24h',provider_resume_limit_24h,
          'provider_resume_count_24h_before_claim',provider_resume_count_24h,
          'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
          'external_content_policy','UNTRUSTED_DATA_NEVER_INSTRUCTIONS',
          'human_approval_required',true,'dip_isolated',true
        )
      ) returning * into created_run;

      insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
      values (
        created_run.run_id,'TASK_CLAIMED','CLAIMED',true,p_base_sha,
        jsonb_build_object(
          'request_id',req.request_id,'worker_id',p_worker_id,'source_parent_sha',req.parent_commit,
          'canonical_base_sha',p_base_sha,'claim_mode','AUTONOMOUS','request_class','PROVIDER_RESUME',
          'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
          'provider_resume_count_24h_before_claim',provider_resume_count_24h,
          'provider_resume_limit_24h',provider_resume_limit_24h,'dip_isolated',true
        )
      );

      return jsonb_build_object(
        'run_id',created_run.run_id,'branch_name',created_run.branch_name,
        'base_branch',created_run.base_branch,'base_sha',created_run.base_sha,
        'task',jsonb_build_object(
          'request_id',req.request_id,'candidate_id',req.candidate_id,'hypothesis_id',req.hypothesis_id,
          'source_parent_sha',req.parent_commit,'requested_branch_name',req.branch_name,
          'changed_paths',req.changed_paths,'objective',req.objective,'constraints',req.constraints,
          'success_criteria',req.success_criteria,'evidence_refs',req.evidence_refs,'metadata',req.metadata
        )
      );
    end if;
  end if;

  return brian_private.claim_engineering_task_fair_queue_v2(p_worker_id,p_base_sha,null);
end;
$$;

create or replace function public.claim_engineering_task(
  p_worker_id text,
  p_base_sha text,
  p_request_id text default null
)
returns jsonb
language sql
security definer
set search_path to 'public','brian_private','pg_temp'
as $$
  select brian_private.claim_engineering_task(p_worker_id,p_base_sha,p_request_id);
$$;

create or replace function brian_private.enqueue_provider_resume_from_blocked_event()
returns trigger
language plpgsql
security definer
set search_path to 'public','brian_private','extensions','pg_temp'
as $$
declare
  run_row public.brian_evolution_engineering_runs%rowtype;
  src public.brian_evolution_codegen_requests%rowtype;
  resume_request_id text;
  resume_after timestamptz := now()+interval '2 minutes';
  resume_iteration integer := 1;
  checkpoint_sha text := null;
  next_metadata jsonb;
begin
  if new.event_kind <> 'BLOCKED' or new.phase <> 'BLOCKED'
     or coalesce(new.payload->>'provider_exhausted','false') <> 'true' then return new; end if;

  select * into run_row from public.brian_evolution_engineering_runs where run_id=new.run_id;
  if not found then return new; end if;
  select * into src from public.brian_evolution_codegen_requests where request_id=run_row.request_id;
  if not found then return new; end if;

  select coalesce(max((r.metadata->>'provider_resume_iteration')::integer),0)+1
  into resume_iteration
  from public.brian_evolution_codegen_requests r
  where r.hypothesis_id=run_row.hypothesis_id
    and coalesce(r.metadata->>'provider_resume','false')='true'
    and coalesce(r.metadata->>'provider_resume_iteration','') ~ '^[0-9]+$';

  checkpoint_sha := coalesce(run_row.commit_sha,nullif(src.metadata->>'failed_candidate_sha',''));
  resume_request_id := encode(digest('brian-engineer-provider-resume|'||run_row.run_id::text||'|'||resume_iteration::text,'sha256'),'hex');

  next_metadata := (coalesce(src.metadata,'{}'::jsonb) - 'auto_retry' - 'created_by') ||
    jsonb_build_object(
      'priority',1000,'provider_resume',true,'provider_resume_iteration',resume_iteration,
      'provider_resume_of_run_id',run_row.run_id::text,'retry_of_run_id',run_row.run_id::text,
      'retry_of_request_id',run_row.request_id,'provider_resume_after',resume_after,
      'provider_failure_stage',coalesce(new.payload->>'failure_stage','UNKNOWN'),
      'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
      'created_by','brian-engineering-provider-continuity','production_grade_required',true,
      'must_resume_existing_candidate',checkpoint_sha is not null,'failed_candidate_sha',checkpoint_sha,
      'provider_failure_state',coalesce(new.payload->'provider_state','{}'::jsonb)
    );

  insert into public.brian_evolution_codegen_requests(
    request_id,candidate_id,hypothesis_id,requested_at,parent_commit,branch_name,
    changed_paths,objective,constraints,success_criteria,evidence_refs,contamination_declaration,
    external_generator_required,required_human_review,metadata,evidence_class,shadow_only,
    live_execution,autonomous_apply_allowed
  ) values (
    resume_request_id,'code|engineer-provider-resume|'||substr(resume_request_id,1,20),run_row.hypothesis_id,
    now(),run_row.base_sha,'evolution-candidate/provider-resume-'||substr(resume_request_id,1,12),
    src.changed_paths,
    src.objective||E'\n\nProvider continuity: resume the same engineering objective/checkpoint after an AI-provider outage; do not redesign merely because the provider changed.',
    src.constraints,src.success_criteria,
    array_append(coalesce(src.evidence_refs,'{}'::text[]),'engineering-run:'||run_row.run_id::text||':provider-exhausted'),
    coalesce(src.contamination_declaration,'Point-in-time evidence only; provider outage is infrastructure metadata, not performance evidence.'),
    true,true,next_metadata,'PROSPECTIVE_EVOLUTION_SHADOW',true,false,false
  ) on conflict (request_id) do nothing;

  update public.brian_evolution_engineering_control
  set metadata=coalesce(metadata,'{}'::jsonb)||jsonb_build_object(
        'provider_continuity_enabled',true,
        'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
        'provider_resume_claim_limit_24h',coalesce(metadata->>'provider_resume_claim_limit_24h','8'),
        'provider_circuit_open_until',resume_after,
        'provider_circuit_reason','ALL_AI_PROVIDERS_UNAVAILABLE',
        'provider_circuit_last_run_id',run_row.run_id::text,
        'provider_circuit_updated_at',now()
      ),updated_at=now()
  where control_id='default';
  return new;
end;
$$;

drop trigger if exists brian_engineering_provider_continuity on public.brian_evolution_engineering_events;
create trigger brian_engineering_provider_continuity
after insert on public.brian_evolution_engineering_events
for each row execute function brian_private.enqueue_provider_resume_from_blocked_event();

update public.brian_evolution_engineering_control
set metadata=coalesce(metadata,'{}'::jsonb)||jsonb_build_object(
      'provider_continuity_enabled',true,
      'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
      'provider_resume_claim_limit_24h',8,
      'provider_resume_cooldown_minutes',2,
      'dip_protected',true
    ),updated_at=now()
where control_id='default';
