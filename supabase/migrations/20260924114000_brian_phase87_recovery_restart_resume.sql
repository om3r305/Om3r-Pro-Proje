-- Brian Phase87 official migration: durable recovery backlog/restart resume.
-- Promoted from the reviewed Phase87 draft contract into official
-- migration lineage on 2026-09-24. This repository change does NOT deploy the
-- migration to any live Supabase project. Runtime remains shadow/paper-only.

create or replace function public.brian_read_next_shadow_recovery_work(
  p_runtime_id text
) returns jsonb
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  with head as (
    select
      h.version as runtime_version,
      h.checkpoint_id,
      h.head_state_id,
      h.checkpoint_payload
    from public.brian_shadow_runtime_heads h
    where h.runtime_id = p_runtime_id
  ),
  unresolved as (
    select
      c.runtime_id,
      c.dispatch_id,
      c.cycle_id as original_cycle_id,
      c.risk_version as cancel_risk_version,
      c.risk_receipt_id as cancel_risk_receipt_id,
      c.reason as cancel_reason,
      c.requested_at
    from public.brian_shadow_execution_cancel_requests c
    where c.runtime_id = p_runtime_id
      and c.phase = 'AFTER_START'
      and not exists (
        select 1
        from public.brian_shadow_recovery_completion_certificates cert
        where cert.runtime_id = c.runtime_id
          and cert.original_cycle_id = c.cycle_id
      )
      and not exists (
        select 1
        from public.brian_shadow_cancel_recovery_directives d0
        where d0.runtime_id = c.runtime_id
          and d0.cycle_id = c.cycle_id
          and d0.recovery_status = 'NO_RECOVERY_REQUIRED'
      )
    order by c.requested_at asc, c.risk_version asc, c.risk_receipt_id asc
    limit 1
  ),
  work as (
    select
      u.*,
      h.runtime_version,
      h.checkpoint_id as runtime_checkpoint_id,
      h.head_state_id as runtime_head_state_id,
      h.checkpoint_payload,
      d.recovery_status,
      d.source_runtime_version as directive_runtime_version,
      d.current_state_id as directive_state_id,
      d.prepared_at as directive_prepared_at,
      rc.status as claim_status,
      rc.worker_token as claim_worker_token,
      rc.claim_fencing_token,
      rc.claim_until,
      rc.recovery_cycle_id,
      rc.progress_runtime_version,
      rc.progress_head_state_id,
      rc.progress_checkpoint_id,
      rs.started_at,
      case
        when rc.recovery_cycle_id is null then null
        else (
          select e.value->>'stage'
          from jsonb_array_elements(
            coalesce(h.checkpoint_payload->'journal_manifest'->'entries','[]'::jsonb)
          ) with ordinality e(value,ord)
          where e.value->>'cycle_id' = rc.recovery_cycle_id
          order by e.ord desc
          limit 1
        )
      end as recovery_journal_stage,
      exists(
        select 1
        from public.brian_shadow_recovery_commit_events ce
        where ce.runtime_id = u.runtime_id
          and ce.dispatch_id = u.dispatch_id
          and ce.original_cycle_id = u.original_cycle_id
          and ce.recovery_cycle_id = rc.recovery_cycle_id
          and ce.checkpoint_id = rc.progress_checkpoint_id
          and ce.event = 'RECOVERY_COMMITTED_PENDING_AUDIT'
      ) as phase84_terminal_event
    from unresolved u
    left join head h on true
    left join lateral (
      select d1.*
      from public.brian_shadow_cancel_recovery_directives d1
      where d1.runtime_id=u.runtime_id
        and d1.cycle_id=u.original_cycle_id
      order by d1.prepared_at asc, d1.cancel_risk_version asc
      limit 1
    ) d on true
    left join public.brian_shadow_recovery_claims rc
      on rc.runtime_id=u.runtime_id
     and rc.dispatch_id=u.dispatch_id
     and rc.cancel_risk_receipt_id=u.cancel_risk_receipt_id
    left join public.brian_shadow_recovery_starts rs
      on rs.runtime_id=u.runtime_id
     and rs.dispatch_id=u.dispatch_id
     and rs.cancel_risk_receipt_id=u.cancel_risk_receipt_id
  )
  select case
    when w.original_cycle_id is null then jsonb_build_object(
      'has_work',false,
      'status','IDLE',
      'runtime_id',p_runtime_id,
      'work_state','IDLE'
    )
    else jsonb_build_object(
      'has_work',true,
      'status','WORK',
      'runtime_id',p_runtime_id,
      'original_cycle_id',w.original_cycle_id,
      'dispatch_id',w.dispatch_id,
      'cancel_risk_version',w.cancel_risk_version,
      'cancel_risk_receipt_id',w.cancel_risk_receipt_id,
      'cancel_reason',w.cancel_reason,
      'requested_at',w.requested_at,
      'runtime_version',w.runtime_version,
      'runtime_checkpoint_id',w.runtime_checkpoint_id,
      'runtime_head_state_id',w.runtime_head_state_id,
      'directive_exists',w.recovery_status is not null,
      'recovery_status',w.recovery_status,
      'directive_runtime_version',w.directive_runtime_version,
      'directive_state_id',w.directive_state_id,
      'directive_prepared_at',w.directive_prepared_at,
      'claim_status',w.claim_status,
      'claim_worker_token',w.claim_worker_token,
      'claim_fencing_token',w.claim_fencing_token,
      'claim_until',w.claim_until,
      'recovery_cycle_id',w.recovery_cycle_id,
      'progress_runtime_version',w.progress_runtime_version,
      'progress_head_state_id',w.progress_head_state_id,
      'progress_checkpoint_id',w.progress_checkpoint_id,
      'started',w.started_at is not null,
      'started_at',w.started_at,
      'recovery_journal_stage',w.recovery_journal_stage,
      'phase84_terminal_event',w.phase84_terminal_event,
      'work_state',
        case
          when w.recovery_status is null then 'NEEDS_DIRECTIVE'
          when w.recovery_status = 'MANUAL_REVIEW' then 'MANUAL_REVIEW'
          when w.claim_status = 'COMPLETED' then 'COMPLETED_WITHOUT_CERTIFICATE'
          when w.recovery_cycle_id is not null
               and w.recovery_journal_stage = 'COMMITTED'
               and w.phase84_terminal_event
            then 'NEEDS_AUDIT'
          when w.claim_status is null then 'NEEDS_CLAIM'
          when w.claim_status = 'CLAIMED'
               and w.claim_until <= clock_timestamp()
            then 'CLAIM_EXPIRED'
          when w.started_at is null then 'NEEDS_START'
          when w.recovery_cycle_id is null then 'STARTED_NEEDS_EXECUTION'
          else 'RECOVERY_PROGRESS'
        end
    )
  end
  from (select 1) seed
  left join work w on true;
$$;

revoke all on function public.brian_read_next_shadow_recovery_work(text)
  from public, anon, authenticated;
grant execute on function public.brian_read_next_shadow_recovery_work(text)
  to service_role;
