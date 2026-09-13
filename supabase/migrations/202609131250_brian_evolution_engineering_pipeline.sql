begin;

create schema if not exists brian_private;

create table if not exists public.brian_evolution_engineering_control (
  control_id text primary key default 'default',
  autonomous_claim_enabled boolean not null default false,
  base_branch text not null default 'brian-2026',
  max_concurrent_runs integer not null default 1 check (max_concurrent_runs between 1 and 3),
  require_human_approval boolean not null default true,
  monitor_minutes integer not null default 30 check (monitor_minutes between 10 and 240),
  updated_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb
);

insert into public.brian_evolution_engineering_control(control_id)
values ('default')
on conflict (control_id) do nothing;

create table if not exists public.brian_evolution_engineering_runs (
  run_id uuid primary key default gen_random_uuid(),
  request_id text not null unique references public.brian_evolution_codegen_requests(request_id) on delete restrict,
  candidate_id text not null,
  hypothesis_id text not null,
  worker_id text not null,
  phase text not null default 'CLAIMED' check (phase in (
    'CLAIMED','UNDERSTAND','PLAN','CODE','COMPILE','TEST','REPLAY','REVIEW','PR','PREVIEW','MEASURE','HUMAN_APPROVAL','DEPLOY','MONITOR','ROLLBACK','COMPLETE','BLOCKED'
  )),
  status text not null default 'RUNNING' check (status in ('RUNNING','WAITING','PASSED','FAILED','BLOCKED','ROLLED_BACK','COMPLETE')),
  base_branch text not null default 'brian-2026',
  base_sha text not null,
  source_parent_sha text,
  branch_name text unique,
  commit_sha text,
  pr_number bigint,
  pr_url text,
  preview_url text,
  previous_good_sha text,
  deployed_sha text,
  rollback_sha text,
  compile_passed boolean,
  tests_passed boolean,
  replay_passed boolean,
  stress_passed boolean,
  review_passed boolean,
  preview_passed boolean,
  measurement_passed boolean,
  human_approval_status text not null default 'PENDING' check (human_approval_status in ('PENDING','APPROVED','REJECTED')),
  human_approved_by text,
  human_approved_at timestamptz,
  monitor_status text,
  failure_reason text,
  check_results jsonb not null default '{}'::jsonb,
  review_result jsonb not null default '{}'::jsonb,
  measurement_result jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false),
  autonomous_apply_allowed boolean not null default false check (autonomous_apply_allowed = false),
  claimed_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists brian_evolution_engineering_runs_phase_idx
  on public.brian_evolution_engineering_runs(status, phase, updated_at desc);

create table if not exists public.brian_evolution_engineering_events (
  event_id bigint generated always as identity primary key,
  run_id uuid not null references public.brian_evolution_engineering_runs(run_id) on delete cascade,
  observed_at timestamptz not null default now(),
  event_kind text not null,
  phase text not null,
  passed boolean,
  commit_sha text,
  payload jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'ENGINEERING_PIPELINE',
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false)
);

create index if not exists brian_evolution_engineering_events_run_idx
  on public.brian_evolution_engineering_events(run_id, observed_at desc);

alter table public.brian_evolution_engineering_control enable row level security;
alter table public.brian_evolution_engineering_runs enable row level security;
alter table public.brian_evolution_engineering_events enable row level security;

revoke all on public.brian_evolution_engineering_control from anon, authenticated;
revoke all on public.brian_evolution_engineering_runs from anon, authenticated;
revoke all on public.brian_evolution_engineering_events from anon, authenticated;

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
  if p_request_id is null and cfg.autonomous_claim_enabled is not true then return null; end if;

  select count(*) into active_count
  from public.brian_evolution_engineering_runs
  where status in ('RUNNING','WAITING') and phase not in ('HUMAN_APPROVAL','COMPLETE','BLOCKED');
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
  order by coalesce((r.metadata->>'priority')::numeric,0) desc, r.requested_at asc
  for update of r skip locked
  limit 1;

  if not found then return null; end if;

  insert into public.brian_evolution_engineering_runs(
    request_id,candidate_id,hypothesis_id,worker_id,base_branch,base_sha,source_parent_sha,branch_name,previous_good_sha,metadata
  ) values (
    req.request_id,req.candidate_id,req.hypothesis_id,p_worker_id,cfg.base_branch,p_base_sha,req.parent_commit,
    'brian-engineer/' || substr(req.request_id,1,16),p_base_sha,
    jsonb_build_object('requested_at',req.requested_at,'objective',req.objective,'constraints',req.constraints,'success_criteria',req.success_criteria,'evidence_refs',req.evidence_refs,'request_metadata',req.metadata)
  ) returning * into created_run;

  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (created_run.run_id,'TASK_CLAIMED','CLAIMED',true,p_base_sha,jsonb_build_object('request_id',req.request_id,'worker_id',p_worker_id));

  return jsonb_build_object(
    'run_id',created_run.run_id,
    'branch_name',created_run.branch_name,
    'base_branch',created_run.base_branch,
    'base_sha',created_run.base_sha,
    'task',jsonb_build_object(
      'request_id',req.request_id,'candidate_id',req.candidate_id,'hypothesis_id',req.hypothesis_id,
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
begin
  if p_phase not in ('CLAIMED','UNDERSTAND','PLAN','CODE','COMPILE','TEST','REPLAY','REVIEW','PR','PREVIEW','MEASURE','HUMAN_APPROVAL','DEPLOY','MONITOR','ROLLBACK','COMPLETE','BLOCKED') then
    raise exception 'invalid engineering phase';
  end if;
  insert into public.brian_evolution_engineering_events(run_id,event_kind,phase,passed,commit_sha,payload)
  values (p_run_id,left(p_event_kind,120),p_phase,p_passed,p_commit_sha,coalesce(p_payload,'{}'::jsonb));
  update public.brian_evolution_engineering_runs set
    phase=p_phase,
    status=case when p_passed is false then 'FAILED' when p_phase='HUMAN_APPROVAL' then 'WAITING' when p_phase='COMPLETE' then 'COMPLETE' else status end,
    commit_sha=coalesce(p_commit_sha,commit_sha),
    compile_passed=case when p_phase='COMPILE' then p_passed else compile_passed end,
    tests_passed=case when p_phase='TEST' then p_passed else tests_passed end,
    replay_passed=case when p_phase='REPLAY' then p_passed else replay_passed end,
    stress_passed=case when p_event_kind='STRESS' then p_passed else stress_passed end,
    review_passed=case when p_phase='REVIEW' then p_passed else review_passed end,
    preview_passed=case when p_phase='PREVIEW' then p_passed else preview_passed end,
    measurement_passed=case when p_phase='MEASURE' then p_passed else measurement_passed end,
    failure_reason=case when p_passed is false then left(coalesce(p_payload->>'error',p_event_kind),1200) else failure_reason end,
    check_results=check_results || jsonb_build_object(lower(p_event_kind),coalesce(p_payload,'{}'::jsonb)),
    updated_at=now()
  where run_id=p_run_id;
  if not found then raise exception 'engineering run not found'; end if;
end;
$$;

revoke all on function brian_private.claim_engineering_task(text,text,text) from public;
revoke all on function brian_private.record_engineering_event(uuid,text,text,boolean,text,jsonb) from public;
grant execute on function brian_private.claim_engineering_task(text,text,text) to service_role;
grant execute on function brian_private.record_engineering_event(uuid,text,text,boolean,text,jsonb) to service_role;

comment on table public.brian_evolution_engineering_runs is 'Fail-closed Brian engineering lifecycle. Production integration always requires human approval.';

commit;
