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

  select * into cfg from public.brian_evolution_engineering_control where control_id='default' for update;
  if not found then raise exception 'engineering control row missing'; end if;
  if cfg.require_human_approval is not true then raise exception 'human approval invariant disabled'; end if;
  if p_request_id is null and cfg.autonomous_claim_enabled is not true then return null; end if;

  -- Exact transport retries from the same GitHub run/attempt are idempotent.
  -- This closes the ambiguity where the DB commit succeeds but the gateway response is lost.
  if p_request_id is not null then
    select r.* into req
    from public.brian_evolution_codegen_requests r
    where r.request_id=p_request_id;

    if found then
      select er.* into created_run
      from public.brian_evolution_engineering_runs er
      where er.request_id=req.request_id
        and er.worker_id=p_worker_id
        and er.base_sha=p_base_sha
        and er.status in ('RUNNING','WAITING')
        and er.phase not in ('COMPLETE','BLOCKED','ROLLBACK')
      limit 1;

      if found then
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
    end if;
  end if;

  select count(*) into active_count
  from public.brian_evolution_engineering_runs
  where status in ('RUNNING','WAITING') and phase not in ('HUMAN_APPROVAL','COMPLETE','BLOCKED','ROLLBACK');
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
      select 1 from public.brian_evolution_codegen_requests newer
      where newer.hypothesis_id=r.hypothesis_id and newer.requested_at>r.requested_at
    )
  order by coalesce((r.metadata->>'priority')::numeric,0) desc, r.requested_at desc
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
      'source_parent_stale',req.parent_commit<>p_base_sha
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

create or replace function brian_private.record_engineering_event(
  p_run_id uuid,
  p_event_kind text,
  p_phase text,
  p_passed boolean,
  p_commit_sha text,
  p_payload jsonb default '{}'::jsonb
) returns void
language plpgsql
security definer
set search_path = public, brian_private, pg_temp
as $$
declare
  r public.brian_evolution_engineering_runs%rowtype;
  allowed boolean := false;
  payload jsonb := coalesce(p_payload,'{}'::jsonb);
  normalized_event_kind text := left(p_event_kind,120);
begin
  select * into r from public.brian_evolution_engineering_runs where run_id=p_run_id for update;
  if not found then raise exception 'engineering run not found'; end if;
  if p_phase not in ('CLAIMED','UNDERSTAND','PLAN','CODE','COMPILE','TEST','REPLAY','REVIEW','PR','PREVIEW','MEASURE','HUMAN_APPROVAL','DEPLOY','MONITOR','ROLLBACK','COMPLETE','BLOCKED') then
    raise exception 'invalid engineering phase';
  end if;
  if coalesce(trim(p_event_kind),'')='' then raise exception 'event kind required'; end if;

  -- Exact event retries are idempotent. Check before terminal/transition guards so
  -- a response-loss retry cannot turn a successfully recorded gate into a false failure.
  if exists (
    select 1
    from public.brian_evolution_engineering_events e
    where e.run_id=p_run_id
      and e.event_kind=normalized_event_kind
      and e.phase=p_phase
      and e.passed is not distinct from p_passed
      and e.commit_sha is not distinct from p_commit_sha
      and e.payload=payload
  ) then
    return;
  end if;

  if r.phase in ('COMPLETE','BLOCKED','ROLLBACK') then raise exception 'engineering run is terminal'; end if;

  if p_phase='BLOCKED' then
    allowed := true;
  elsif p_passed is not true then
    raise exception 'non-terminal gate events must pass; use BLOCKED for failures';
  elsif r.phase='CLAIMED' and p_phase='UNDERSTAND' then allowed := true;
  elsif r.phase='UNDERSTAND' and p_phase='PLAN' then allowed := true;
  elsif r.phase='PLAN' and p_phase='CODE' then allowed := true;
  elsif r.phase='CODE' and p_phase='COMPILE' then allowed := true;
  elsif r.phase='COMPILE' and p_phase='TEST' and r.compile_passed is true then allowed := true;
  elsif r.phase='TEST' and p_phase='REPLAY' and p_event_kind='REPLAY' and r.tests_passed is true then allowed := true;
  elsif r.phase='REPLAY' and p_phase='REPLAY' and p_event_kind='STRESS' and r.replay_passed is true then allowed := true;
  elsif r.phase='REPLAY' and p_phase='REVIEW' and r.replay_passed is true and r.stress_passed is true then allowed := true;
  elsif r.phase='REVIEW' and p_phase='PR' and r.review_passed is true then allowed := true;
  elsif r.phase='PR' and p_phase='PREVIEW' then allowed := true;
  elsif r.phase='HUMAN_APPROVAL' and p_phase='DEPLOY' and r.human_approval_status='APPROVED' and r.measurement_passed is true then allowed := true;
  elsif r.phase='DEPLOY' and p_phase='MONITOR' then allowed := true;
  elsif r.phase='MONITOR' and p_phase='COMPLETE' then allowed := true;
  elsif r.phase in ('HUMAN_APPROVAL','DEPLOY','MONITOR') and p_phase='ROLLBACK' then allowed := true;
  end if;

  if not allowed then raise exception 'invalid engineering transition % -> % (%).',r.phase,p_phase,p_event_kind; end if;

  if p_phase not in ('DEPLOY','MONITOR','ROLLBACK','COMPLETE','BLOCKED') and p_commit_sha is not null then
    if p_commit_sha !~ '^[0-9a-fA-F]{40}$' then raise exception 'candidate commit SHA invalid'; end if;
    if r.commit_sha is not null and r.commit_sha<>p_commit_sha then raise exception 'candidate commit SHA changed after evidence started'; end if;
  end if;

  if p_phase='DEPLOY' then
    if p_commit_sha is null or p_commit_sha !~ '^[0-9a-fA-F]{40}$' then raise exception 'deployed commit SHA invalid'; end if;
  elsif p_phase in ('MONITOR','COMPLETE') then
    if r.deployed_sha is null then raise exception 'deployed SHA missing before monitor/complete'; end if;
    if p_commit_sha is distinct from r.deployed_sha then raise exception 'monitor/complete SHA does not match deployed SHA'; end if;
  elsif p_phase='ROLLBACK' and p_commit_sha is not null and p_commit_sha !~ '^[0-9a-fA-F]{40}$' then
    raise exception 'rollback commit SHA invalid';
  end if;

  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (p_run_id,normalized_event_kind,p_phase,p_passed,p_commit_sha,payload);

  update public.brian_evolution_engineering_runs set
    phase=p_phase,
    status=case
      when p_phase='BLOCKED' then 'BLOCKED'
      when p_phase='ROLLBACK' then 'ROLLED_BACK'
      when p_phase='COMPLETE' then 'COMPLETE'
      when p_phase='HUMAN_APPROVAL' then 'WAITING'
      else 'RUNNING'
    end,
    commit_sha=case when p_phase in ('UNDERSTAND','PLAN','CODE','COMPILE','TEST','REPLAY','REVIEW','PR','PREVIEW') then coalesce(commit_sha,p_commit_sha) else commit_sha end,
    compile_passed=case when p_phase='COMPILE' then true else compile_passed end,
    tests_passed=case when p_phase='TEST' then true else tests_passed end,
    replay_passed=case when p_phase='REPLAY' and p_event_kind='REPLAY' then true else replay_passed end,
    stress_passed=case when p_phase='REPLAY' and p_event_kind='STRESS' then true else stress_passed end,
    review_passed=case when p_phase='REVIEW' then true else review_passed end,
    review_result=case when p_phase='REVIEW' then payload else review_result end,
    pr_url=case when p_phase='PR' then nullif(payload->>'pr_url','') else pr_url end,
    pr_number=case when p_phase='PR' and (payload->>'pr_number') ~ '^[0-9]+$' then (payload->>'pr_number')::bigint else pr_number end,
    preview_passed=case when p_phase='PREVIEW' then true else preview_passed end,
    preview_url=case when p_phase='PREVIEW' then nullif(payload->>'preview_url','') else preview_url end,
    deployed_sha=case when p_phase='DEPLOY' then p_commit_sha else deployed_sha end,
    monitor_status=case when p_phase='MONITOR' then coalesce(payload->>'status','HEALTHY') else monitor_status end,
    rollback_sha=case when p_phase='ROLLBACK' then p_commit_sha else rollback_sha end,
    failure_reason=case when p_phase='BLOCKED' then left(coalesce(payload->>'error',p_event_kind),1200) else failure_reason end,
    check_results=check_results || jsonb_build_object(lower(p_event_kind),payload),
    updated_at=now()
  where run_id=p_run_id;
end;
$$;

revoke all on function brian_private.claim_engineering_task(text,text,text) from public;
revoke all on function brian_private.record_engineering_event(uuid,text,text,boolean,text,jsonb) from public;
grant execute on function brian_private.claim_engineering_task(text,text,text) to service_role;
grant execute on function brian_private.record_engineering_event(uuid,text,text,boolean,text,jsonb) to service_role;

comment on function brian_private.claim_engineering_task(text,text,text) is 'Claims one fail-closed engineering request. Exact same-worker transport retries return the already-created active run.';
comment on function brian_private.record_engineering_event(uuid,text,text,boolean,text,jsonb) is 'Records monotonic engineering gates. Exact duplicate transport retries are idempotent.';

commit;
