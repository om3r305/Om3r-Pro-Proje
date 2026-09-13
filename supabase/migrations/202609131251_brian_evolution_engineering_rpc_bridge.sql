begin;

create or replace function public.claim_engineering_task(
  p_worker_id text,
  p_base_sha text,
  p_request_id text default null
) returns jsonb
language sql
security definer
set search_path = public, brian_private, pg_temp
as $$
  select brian_private.claim_engineering_task(p_worker_id,p_base_sha,p_request_id);
$$;

create or replace function public.record_engineering_event(
  p_run_id uuid,
  p_event_kind text,
  p_phase text,
  p_passed boolean,
  p_commit_sha text,
  p_payload jsonb default '{}'::jsonb
) returns void
language sql
security definer
set search_path = public, brian_private, pg_temp
as $$
  select brian_private.record_engineering_event(p_run_id,p_event_kind,p_phase,p_passed,p_commit_sha,p_payload);
$$;

revoke all on function public.claim_engineering_task(text,text,text) from public, anon, authenticated;
revoke all on function public.record_engineering_event(uuid,text,text,boolean,text,jsonb) from public, anon, authenticated;
grant execute on function public.claim_engineering_task(text,text,text) to service_role;
grant execute on function public.record_engineering_event(uuid,text,text,boolean,text,jsonb) to service_role;

commit;
