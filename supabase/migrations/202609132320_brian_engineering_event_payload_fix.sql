begin;

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
  v_payload jsonb := coalesce(p_payload,'{}'::jsonb);
  normalized_event_kind text := left(p_event_kind,120);
begin
  select * into r from public.brian_evolution_engineering_runs where run_id=p_run_id for update;
  if not found then raise exception 'engineering run not found'; end if;
  if p_phase not in ('CLAIMED','UNDERSTAND','PLAN','CODE','COMPILE','TEST','REPLAY','REVIEW','PR','PREVIEW','MEASURE','HUMAN_APPROVAL','DEPLOY','MONITOR','ROLLBACK','COMPLETE','BLOCKED') then
    raise exception 'invalid engineering phase';
  end if;
  if coalesce(trim(p_event_kind),'')='' then raise exception 'event kind required'; end if;

  if exists (
    select 1 from public.brian_evolution_engineering_events e
    where e.run_id=p_run_id
      and e.event_kind=normalized_event_kind
      and e.phase=p_phase
      and e.passed is not distinct from p_passed
      and e.commit_sha is not distinct from p_commit_sha
      and e.payload=v_payload
  ) then return; end if;

  if r.phase in ('COMPLETE','BLOCKED','ROLLBACK') then raise exception 'engineering run is terminal'; end if;

  if p_phase='BLOCKED' then allowed := true;
  elsif p_passed is not true then raise exception 'non-terminal gate events must pass; use BLOCKED for failures';
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
  values (p_run_id,normalized_event_kind,p_phase,p_passed,p_commit_sha,v_payload);

  update public.brian_evolution_engineering_runs set
    phase=p_phase,
    status=case when p_phase='BLOCKED' then 'BLOCKED' when p_phase='ROLLBACK' then 'ROLLED_BACK' when p_phase='COMPLETE' then 'COMPLETE' when p_phase='HUMAN_APPROVAL' then 'WAITING' else 'RUNNING' end,
    commit_sha=case when p_phase in ('UNDERSTAND','PLAN','CODE','COMPILE','TEST','REPLAY','REVIEW','PR','PREVIEW') then coalesce(commit_sha,p_commit_sha) else commit_sha end,
    compile_passed=case when p_phase='COMPILE' then true else compile_passed end,
    tests_passed=case when p_phase='TEST' then true else tests_passed end,
    replay_passed=case when p_phase='REPLAY' and p_event_kind='REPLAY' then true else replay_passed end,
    stress_passed=case when p_phase='REPLAY' and p_event_kind='STRESS' then true else stress_passed end,
    review_passed=case when p_phase='REVIEW' then true else review_passed end,
    review_result=case when p_phase='REVIEW' then v_payload else review_result end,
    pr_url=case when p_phase='PR' then nullif(v_payload->>'pr_url','') else pr_url end,
    pr_number=case when p_phase='PR' and (v_payload->>'pr_number') ~ '^[0-9]+$' then (v_payload->>'pr_number')::bigint else pr_number end,
    preview_passed=case when p_phase='PREVIEW' then true else preview_passed end,
    preview_url=case when p_phase='PREVIEW' then nullif(v_payload->>'preview_url','') else preview_url end,
    deployed_sha=case when p_phase='DEPLOY' then p_commit_sha else deployed_sha end,
    monitor_status=case when p_phase='MONITOR' then coalesce(v_payload->>'status','HEALTHY') else monitor_status end,
    rollback_sha=case when p_phase='ROLLBACK' then p_commit_sha else rollback_sha end,
    failure_reason=case when p_phase='BLOCKED' then left(coalesce(v_payload->>'error',p_event_kind),1200) else failure_reason end,
    check_results=check_results || jsonb_build_object(lower(p_event_kind),v_payload),
    updated_at=now()
  where run_id=p_run_id;
end;
$$;

commit;
