begin;

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

  -- Transport retries from the same GitHub run/attempt must return the already
  -- claimed run instead of claiming a second task or losing the first claim.
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

  select count(*) into active_count
  from public.brian_evolution_engineering_runs
  where status in ('RUNNING','WAITING')
    and phase not in ('HUMAN_APPROVAL','COMPLETE','BLOCKED','ROLLBACK');
  if active_count >= cfg.max_concurrent_runs then return null; end if;

  select r.* into req
  from public.brian_evolution_codegen_requests r
  left join public.brian_evolution_engineering_runs er on er.request_id=r.request_id
  where er.request_id is null
    and r.required_human_review is true
    and r.shadow_only is true
    and r.live_execution is false
    and r.autonomous_apply_allowed is false
    and (p_request_id is null or r.request_id=p_request_id)
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
      'transport_retry_safe',true
    )
  ) returning * into created_run;

  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (created_run.run_id,'TASK_CLAIMED','CLAIMED',true,p_base_sha,
    jsonb_build_object('request_id',req.request_id,'worker_id',p_worker_id,'source_parent_sha',req.parent_commit,'canonical_base_sha',p_base_sha));

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

commit;
