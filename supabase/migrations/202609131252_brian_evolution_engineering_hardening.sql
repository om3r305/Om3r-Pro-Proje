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
begin
  select * into r from public.brian_evolution_engineering_runs where run_id=p_run_id for update;
  if not found then raise exception 'engineering run not found'; end if;
  if r.phase in ('COMPLETE','BLOCKED','ROLLBACK') then raise exception 'engineering run is terminal'; end if;
  if p_phase not in ('CLAIMED','UNDERSTAND','PLAN','CODE','COMPILE','TEST','REPLAY','REVIEW','PR','PREVIEW','MEASURE','HUMAN_APPROVAL','DEPLOY','MONITOR','ROLLBACK','COMPLETE','BLOCKED') then
    raise exception 'invalid engineering phase';
  end if;
  if coalesce(trim(p_event_kind),'')='' then raise exception 'event kind required'; end if;

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

  if p_phase not in ('DEPLOY','MONITOR','ROLLBACK','BLOCKED') and p_commit_sha is not null then
    if p_commit_sha !~ '^[0-9a-fA-F]{40}$' then raise exception 'candidate commit SHA invalid'; end if;
    if r.commit_sha is not null and r.commit_sha<>p_commit_sha then raise exception 'candidate commit SHA changed after evidence started'; end if;
  end if;

  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (p_run_id,left(p_event_kind,120),p_phase,p_passed,p_commit_sha,payload);

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

create or replace function brian_private.measure_engineering_run(
  p_run_id uuid,
  p_commit_sha text,
  p_payload jsonb
) returns jsonb
language plpgsql
security definer
set search_path = public, brian_private, pg_temp
as $$
declare
  r public.brian_evolution_engineering_runs%rowtype;
  payload jsonb := coalesce(p_payload,'{}'::jsonb);
begin
  select * into r from public.brian_evolution_engineering_runs where run_id=p_run_id for update;
  if not found then raise exception 'engineering run not found'; end if;
  if r.phase<>'PREVIEW' or r.status<>'RUNNING' then raise exception 'measurement requires PREVIEW/RUNNING state'; end if;
  if r.commit_sha is null or r.commit_sha<>p_commit_sha then raise exception 'measurement commit does not match reviewed candidate'; end if;
  if r.compile_passed is not true or r.tests_passed is not true or r.replay_passed is not true or r.stress_passed is not true or r.review_passed is not true or r.preview_passed is not true then
    raise exception 'measurement prerequisites incomplete';
  end if;
  if coalesce((payload->>'patch_bytes')::bigint,-1)<0 then raise exception 'measurement patch_bytes missing'; end if;
  if coalesce((payload->>'changed_files')::int,0)<1 then raise exception 'measurement changed_files missing'; end if;
  if coalesce((payload->>'test_files')::int,0)<1 then raise exception 'measurement test_files missing'; end if;
  if payload->>'exact_commit_sha' is distinct from p_commit_sha then raise exception 'measurement is not bound to exact commit'; end if;
  if payload->>'protected_scope_clear' is distinct from 'true' then raise exception 'measurement protected-scope evidence missing'; end if;

  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (p_run_id,'ENGINEERING_MEASURE','MEASURE',true,p_commit_sha,payload);
  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (p_run_id,'HUMAN_APPROVAL_REQUIRED','HUMAN_APPROVAL',null,p_commit_sha,jsonb_build_object('measurement_kind','EXACT_COMMIT_ENGINEERING'));

  update public.brian_evolution_engineering_runs set
    phase='HUMAN_APPROVAL',status='WAITING',measurement_passed=true,measurement_result=payload,updated_at=now()
  where run_id=p_run_id;

  return jsonb_build_object('run_id',p_run_id,'status','WAITING_HUMAN_APPROVAL','commit_sha',p_commit_sha);
end;
$$;

create or replace function brian_private.approve_engineering_run(
  p_run_id uuid,
  p_commit_sha text,
  p_actor text
) returns jsonb
language plpgsql
security definer
set search_path = public, brian_private, pg_temp
as $$
declare
  r public.brian_evolution_engineering_runs%rowtype;
begin
  select * into r from public.brian_evolution_engineering_runs where run_id=p_run_id for update;
  if not found then raise exception 'engineering run not found'; end if;
  if r.phase<>'HUMAN_APPROVAL' or r.status<>'WAITING' or r.measurement_passed is not true then raise exception 'run is not ready for human approval'; end if;
  if r.commit_sha is null or r.commit_sha<>p_commit_sha then raise exception 'approved SHA is not the measured SHA'; end if;
  if coalesce(trim(p_actor),'')='' or p_actor ~* '\[bot\]$' then raise exception 'human actor required'; end if;
  if r.human_approval_status='REJECTED' then raise exception 'run was rejected'; end if;

  update public.brian_evolution_engineering_runs set
    human_approval_status='APPROVED',human_approved_by=p_actor,human_approved_at=now(),updated_at=now()
  where run_id=p_run_id;
  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (p_run_id,'HUMAN_APPROVED','HUMAN_APPROVAL',true,p_commit_sha,jsonb_build_object('actor',p_actor));
  return jsonb_build_object('run_id',p_run_id,'status','APPROVED','actor',p_actor);
end;
$$;

create or replace function public.measure_engineering_run(p_run_id uuid,p_commit_sha text,p_payload jsonb)
returns jsonb language sql security definer set search_path=public,brian_private,pg_temp as $$
  select brian_private.measure_engineering_run(p_run_id,p_commit_sha,p_payload);
$$;
create or replace function public.approve_engineering_run(p_run_id uuid,p_commit_sha text,p_actor text)
returns jsonb language sql security definer set search_path=public,brian_private,pg_temp as $$
  select brian_private.approve_engineering_run(p_run_id,p_commit_sha,p_actor);
$$;

revoke all on function public.measure_engineering_run(uuid,text,jsonb) from public,anon,authenticated;
revoke all on function public.approve_engineering_run(uuid,text,text) from public,anon,authenticated;
grant execute on function public.measure_engineering_run(uuid,text,jsonb) to service_role;
grant execute on function public.approve_engineering_run(uuid,text,text) to service_role;

comment on function brian_private.measure_engineering_run(uuid,text,jsonb) is 'Binds engineering measurement to the exact reviewed commit. Behavioral/canonical promotion remains a separate Evolution science gate.';

commit;
