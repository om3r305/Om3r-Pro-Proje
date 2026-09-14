begin;

-- Brian may discover and engineer its own SHADOW candidates, but autonomy is
-- deliberately budgeted and cannot bypass the existing human-approval gate.
update public.brian_evolution_engineering_control
set autonomous_claim_enabled = true,
    max_concurrent_runs = 1,
    require_human_approval = true,
    metadata = coalesce(metadata, '{}'::jsonb) || jsonb_build_object(
      'autonomy_profile', 'WORLD_ENGINEER_BOUNDED_V1',
      'autonomous_claim_limit_24h', 4,
      'world_to_engineering_enabled', true,
      'world_source_trust_floor', 0.72,
      'world_source_policy', 'OFFICIAL_PRIMARY_PUBLIC_NO_KEY_METADATA_ONLY',
      'external_content_policy', 'UNTRUSTED_DATA_NEVER_INSTRUCTIONS',
      'protected_scope', 'DIP_ISOLATED',
      'canonical_apply', 'HUMAN_APPROVAL_ONLY',
      'credit_guard', 'MAX_4_AUTONOMOUS_CLAIMS_PER_ROLLING_24H'
    ),
    updated_at = now()
where control_id = 'default';

create or replace function brian_private.claim_engineering_task(
  p_worker_id text,
  p_base_sha text,
  p_request_id text default null
) returns jsonb
language plpgsql
security definer
set search_path = public, brian_private, pg_temp
as $$
declare
  cfg public.brian_evolution_engineering_control%rowtype;
  req public.brian_evolution_codegen_requests%rowtype;
  created_run public.brian_evolution_engineering_runs%rowtype;
  active_count integer;
  autonomous_count_24h integer := 0;
  autonomous_limit_24h integer := 4;
begin
  if coalesce(trim(p_worker_id),'') = '' then raise exception 'worker_id required'; end if;
  if p_base_sha !~ '^[0-9a-fA-F]{40}$' then raise exception 'base_sha must be a full git SHA'; end if;

  select * into cfg
  from public.brian_evolution_engineering_control
  where control_id='default'
  for update;
  if not found then raise exception 'engineering control row missing'; end if;
  if cfg.require_human_approval is not true then raise exception 'human approval invariant disabled'; end if;
  if p_request_id is null and cfg.autonomous_claim_enabled is not true then return null; end if;

  if coalesce(cfg.metadata->>'autonomous_claim_limit_24h','') ~ '^[0-9]+$' then
    autonomous_limit_24h := greatest(1, least(24, (cfg.metadata->>'autonomous_claim_limit_24h')::integer));
  end if;

  -- Transport retries from the same GitHub run/attempt must return the already
  -- claimed run instead of claiming a second task or consuming more budget.
  select er.* into created_run
  from public.brian_evolution_engineering_runs er
  where er.worker_id=p_worker_id
    and er.base_sha=p_base_sha
    and er.status in ('RUNNING','WAITING')
    and er.phase not in ('COMPLETE','BLOCKED','ROLLBACK')
    and (p_request_id is null or er.request_id=p_request_id)
  order by er.created_at desc
  limit 1;

  if found then
    select r.* into req
    from public.brian_evolution_codegen_requests r
    where r.request_id=created_run.request_id;
    if not found then raise exception 'claimed engineering request missing'; end if;

    return jsonb_build_object(
      'run_id',created_run.run_id,
      'branch_name',created_run.branch_name,
      'base_branch',created_run.base_branch,
      'base_sha',created_run.base_sha,
      'task',jsonb_build_object(
        'request_id',req.request_id,'candidate_id',req.candidate_id,'hypothesis_id',req.hypothesis_id,
        'source_parent_sha',req.parent_commit,'requested_branch_name',req.branch_name,'changed_paths',req.changed_paths,
        'objective',req.objective,'constraints',req.constraints,'success_criteria',req.success_criteria,
        'evidence_refs',req.evidence_refs,'metadata',req.metadata
      )
    );
  end if;

  -- Manual request_id runs remain available to the owner. Automatic runs are
  -- capped over a rolling 24-hour window so the 15-minute scheduler cannot burn
  -- credits just because the queue is large.
  if p_request_id is null then
    select count(*) into autonomous_count_24h
    from public.brian_evolution_engineering_runs
    where created_at >= now() - interval '24 hours'
      and metadata->>'claim_mode' = 'AUTONOMOUS';
    if autonomous_count_24h >= autonomous_limit_24h then return null; end if;
  end if;

  select count(*) into active_count
  from public.brian_evolution_engineering_runs
  where status in ('RUNNING','WAITING')
    and phase not in ('HUMAN_APPROVAL','COMPLETE','BLOCKED','ROLLBACK');
  if active_count >= cfg.max_concurrent_runs then return null; end if;

  select r.* into req
  from public.brian_evolution_codegen_requests r
  join public.brian_evolution_code_candidates c on c.candidate_id=r.candidate_id
  left join public.brian_evolution_engineering_runs er on er.request_id=r.request_id
  where er.request_id is null
    and r.required_human_review is true
    and r.shadow_only is true
    and r.live_execution is false
    and r.autonomous_apply_allowed is false
    and c.shadow_only is true
    and c.live_execution is false
    and c.autonomous_apply_allowed is false
    and (p_request_id is null or r.request_id=p_request_id)
    -- Stable world-source adapters do not consume another autonomous model run
    -- after one version of the same hypothesis already reached independent review.
    and not (
      coalesce(c.metadata->>'parent_rotation_policy','') = 'STABLE_ONCE'
      and exists (
        select 1
        from public.brian_evolution_engineering_runs prior
        where prior.hypothesis_id=r.hypothesis_id
          and (
            prior.review_passed is true
            or prior.phase in ('PR','PREVIEW','MEASURE','HUMAN_APPROVAL','DEPLOY','MONITOR','COMPLETE')
          )
      )
    )
    and not exists (
      select 1
      from public.brian_evolution_codegen_requests newer
      where newer.hypothesis_id=r.hypothesis_id
        and (
          newer.requested_at > r.requested_at
          or (newer.requested_at = r.requested_at and newer.created_at > r.created_at)
          or (
            newer.requested_at = r.requested_at
            and newer.created_at = r.created_at
            and newer.request_id > r.request_id
          )
        )
    )
  order by coalesce((r.metadata->>'priority')::numeric,0) desc,
           r.requested_at desc,
           r.created_at desc,
           r.request_id desc
  for update of r skip locked
  limit 1;

  if not found then return null; end if;

  insert into public.brian_evolution_engineering_runs(
    request_id,candidate_id,hypothesis_id,worker_id,base_branch,base_sha,source_parent_sha,branch_name,previous_good_sha,metadata
  ) values (
    req.request_id,req.candidate_id,req.hypothesis_id,p_worker_id,cfg.base_branch,p_base_sha,req.parent_commit,
    'brian-engineer/' || substr(req.request_id,1,16),p_base_sha,
    jsonb_build_object(
      'requested_at',req.requested_at,
      'objective',req.objective,
      'constraints',req.constraints,
      'success_criteria',req.success_criteria,
      'evidence_refs',req.evidence_refs,
      'requested_changed_paths',req.changed_paths,
      'requested_branch_name',req.branch_name,
      'request_metadata',req.metadata,
      'source_parent_stale',req.parent_commit<>p_base_sha,
      'queue_tie_break','requested_at,created_at,request_id',
      'transport_retry_safe',true,
      'claim_mode',case when p_request_id is null then 'AUTONOMOUS' else 'MANUAL_REQUEST' end,
      'autonomous_claim_limit_24h',autonomous_limit_24h,
      'external_content_policy','UNTRUSTED_DATA_NEVER_INSTRUCTIONS',
      'human_approval_required',true,
      'dip_isolated',true
    )
  ) returning * into created_run;

  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (created_run.run_id,'TASK_CLAIMED','CLAIMED',true,p_base_sha,
    jsonb_build_object(
      'request_id',req.request_id,
      'worker_id',p_worker_id,
      'source_parent_sha',req.parent_commit,
      'canonical_base_sha',p_base_sha,
      'claim_mode',case when p_request_id is null then 'AUTONOMOUS' else 'MANUAL_REQUEST' end,
      'autonomous_count_24h_before_claim',autonomous_count_24h,
      'autonomous_limit_24h',autonomous_limit_24h
    ));

  return jsonb_build_object(
    'run_id',created_run.run_id,
    'branch_name',created_run.branch_name,
    'base_branch',created_run.base_branch,
    'base_sha',created_run.base_sha,
    'task',jsonb_build_object(
      'request_id',req.request_id,'candidate_id',req.candidate_id,'hypothesis_id',req.hypothesis_id,
      'source_parent_sha',req.parent_commit,'requested_branch_name',req.branch_name,'changed_paths',req.changed_paths,
      'objective',req.objective,'constraints',req.constraints,'success_criteria',req.success_criteria,
      'evidence_refs',req.evidence_refs,'metadata',req.metadata
    )
  );
end;
$$;

revoke all on function brian_private.claim_engineering_task(text,text,text) from public;
grant execute on function brian_private.claim_engineering_task(text,text,text) to service_role;

comment on function brian_private.claim_engineering_task(text,text,text) is
  'Claims bounded SHADOW engineering work. Autonomous claims are credit-capped; human approval and DIP isolation remain invariant.';

commit;
