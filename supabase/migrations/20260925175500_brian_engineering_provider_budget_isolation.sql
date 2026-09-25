-- Separate provider-resume traffic from the normal autonomous engineering budget.
-- Also honor the explicit pre-UNDERSTAND provider/infrastructure backoff at both
-- preflight and claim time. Human approval, DIP isolation, shadow-only, and all
-- downstream evidence gates remain unchanged.

CREATE OR REPLACE FUNCTION brian_private.claim_engineering_task(p_worker_id text, p_base_sha text, p_request_id text DEFAULT NULL::text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'brian_private', 'pg_temp'
AS $function$
declare
  cfg public.brian_evolution_engineering_control%rowtype;
  req public.brian_evolution_codegen_requests%rowtype;
  created_run public.brian_evolution_engineering_runs%rowtype;
  active_count integer := 0;
  provider_resume_count_24h integer := 0;
  provider_resume_limit_24h integer := 8;
  circuit_until timestamptz := null;
  provider_backoff_until timestamptz := null;
begin
  if coalesce(trim(p_worker_id),'') = '' then raise exception 'worker_id required'; end if;
  if p_base_sha !~ '^[0-9a-fA-F]{40}$' then raise exception 'base_sha must be a full git SHA'; end if;

  -- Explicit/manual requests keep the established behavior and safety gates.
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

  -- When all providers were unavailable, do not burn fresh functional tasks.
  if circuit_until is not null and circuit_until > now() then
    return null;
  end if;

  begin
    provider_backoff_until := nullif(cfg.metadata->>'engineering_provider_backoff_until','')::timestamptz;
  exception when others then
    provider_backoff_until := null;
  end;
  if provider_backoff_until is not null and provider_backoff_until > now() then
    return null;
  end if;

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
          'requested_at',req.requested_at,
          'objective',req.objective,
          'constraints',req.constraints,
          'success_criteria',req.success_criteria,
          'evidence_refs',req.evidence_refs,
          'requested_changed_paths',req.changed_paths,
          'requested_branch_name',req.branch_name,
          'request_metadata',req.metadata,
          'source_parent_stale',req.parent_commit<>p_base_sha,
          'queue_tie_break','provider-resume-first',
          'transport_retry_safe',true,
          'claim_mode','AUTONOMOUS',
          'request_class','PROVIDER_RESUME',
          'provider_resume_claim_limit_24h',provider_resume_limit_24h,
          'provider_resume_count_24h_before_claim',provider_resume_count_24h,
          'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
          'external_content_policy','UNTRUSTED_DATA_NEVER_INSTRUCTIONS',
          'human_approval_required',true,
          'dip_isolated',true
        )
      ) returning * into created_run;

      insert into public.brian_evolution_engineering_events(
        run_id,event_kind,phase,passed,commit_sha,payload
      ) values (
        created_run.run_id,'TASK_CLAIMED','CLAIMED',true,p_base_sha,
        jsonb_build_object(
          'request_id',req.request_id,
          'worker_id',p_worker_id,
          'source_parent_sha',req.parent_commit,
          'canonical_base_sha',p_base_sha,
          'claim_mode','AUTONOMOUS',
          'request_class','PROVIDER_RESUME',
          'provider_continuity_policy','HOSTED_COPILOT_GITHUB_MODELS_BYOK_V1',
          'provider_resume_count_24h_before_claim',provider_resume_count_24h,
          'provider_resume_limit_24h',provider_resume_limit_24h,
          'dip_isolated',true
        )
      );

      return jsonb_build_object(
        'run_id',created_run.run_id,
        'branch_name',created_run.branch_name,
        'base_branch',created_run.base_branch,
        'base_sha',created_run.base_sha,
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
$function$
;

CREATE OR REPLACE FUNCTION brian_private.claim_engineering_task_fair_queue_v2(p_worker_id text, p_base_sha text, p_request_id text DEFAULT NULL::text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'brian_private', 'pg_temp'
AS $function$
declare
  cfg public.brian_evolution_engineering_control%rowtype;
  req public.brian_evolution_codegen_requests%rowtype;
  created_run public.brian_evolution_engineering_runs%rowtype;
  active_count integer;
  autonomous_count_24h integer := 0;
  autonomous_fresh_count_24h integer := 0;
  autonomous_retry_count_24h integer := 0;
  autonomous_limit_24h integer := 4;
  autonomous_fresh_reserve_24h integer := 2;
  autonomous_retry_limit_24h integer := 2;
  autonomous_retry_per_hypothesis_24h integer := 2;
  autonomous_stalled_cooldown_hours integer := 12;
  world_trust_floor numeric := 0.72;
  provider_backoff_until timestamptz := null;
  request_is_retry boolean := false;
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
  if coalesce(cfg.metadata->>'autonomous_fresh_reserve_24h','') ~ '^[0-9]+$' then
    autonomous_fresh_reserve_24h := greatest(1, least(autonomous_limit_24h, (cfg.metadata->>'autonomous_fresh_reserve_24h')::integer));
  end if;
  if coalesce(cfg.metadata->>'autonomous_retry_claim_limit_24h','') ~ '^[0-9]+$' then
    autonomous_retry_limit_24h := greatest(0, least(autonomous_limit_24h, (cfg.metadata->>'autonomous_retry_claim_limit_24h')::integer));
  end if;
  if coalesce(cfg.metadata->>'autonomous_retry_per_hypothesis_24h','') ~ '^[0-9]+$' then
    autonomous_retry_per_hypothesis_24h := greatest(1, least(autonomous_limit_24h, (cfg.metadata->>'autonomous_retry_per_hypothesis_24h')::integer));
  end if;
  if coalesce(cfg.metadata->>'autonomous_stalled_cooldown_hours','') ~ '^[0-9]+$' then
    autonomous_stalled_cooldown_hours := greatest(1, least(72, (cfg.metadata->>'autonomous_stalled_cooldown_hours')::integer));
  end if;
  if coalesce(cfg.metadata->>'world_source_trust_floor','') ~ '^(0([.][0-9]+)?|1([.]0+)?)
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

  if p_request_id is null then
    select
      count(*),
      count(*) filter (where not brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))),
      count(*) filter (where brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb)))
    into autonomous_count_24h, autonomous_fresh_count_24h, autonomous_retry_count_24h
    from public.brian_evolution_engineering_runs er
    join public.brian_evolution_codegen_requests r on r.request_id=er.request_id
    where er.created_at >= now() - interval '24 hours'
      and er.metadata->>'claim_mode' = 'AUTONOMOUS'
      and coalesce(r.metadata->>'provider_resume','false') <> 'true';

    -- Normal ceiling remains 4/24h. If legacy retry spam already consumed the ceiling,
    -- allow only enough fresh claims to restore the reserved fresh-work slots.
    if autonomous_count_24h >= autonomous_limit_24h
       and autonomous_fresh_count_24h >= autonomous_fresh_reserve_24h then
      return null;
    end if;
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
    and (p_request_id is not null or coalesce(r.metadata->>'provider_resume','false') <> 'true')
    and case
      when p_request_id is not null then true
      when coalesce(r.metadata->>'world_engineering','false') = 'true' then
        coalesce(cfg.metadata->>'world_to_engineering_enabled','false') = 'true'
        and brian_private.world_engineering_request_is_current(r.metadata, world_trust_floor)
      when exists (
        select 1 from unnest(coalesce(r.evidence_refs,'{}'::text[])) as legacy_ref
        where legacy_ref like 'world_source:%'
      ) then false
      else true
    end
    and not (
      p_request_id is null
      and coalesce(r.metadata->>'world_engineering','false') = 'true'
      and coalesce(r.metadata->>'parent_rotation_policy','') = 'STABLE_ONCE'
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
    and (
      p_request_id is not null
      or (
        not brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))
        and autonomous_fresh_count_24h < autonomous_limit_24h
        and (
          autonomous_count_24h < autonomous_limit_24h
          or autonomous_fresh_count_24h < autonomous_fresh_reserve_24h
        )
      )
      or (
        brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))
        and autonomous_count_24h < autonomous_limit_24h
        and autonomous_retry_count_24h < autonomous_retry_limit_24h
        and (
          select count(*)
          from public.brian_evolution_engineering_runs prior_run
          join public.brian_evolution_codegen_requests prior_req on prior_req.request_id=prior_run.request_id
          where prior_run.hypothesis_id=r.hypothesis_id
            and prior_run.created_at >= now() - interval '24 hours'
            and prior_run.metadata->>'claim_mode'='AUTONOMOUS'
            and brian_private.engineering_request_is_retry(coalesce(prior_req.metadata,'{}'::jsonb))
        ) < autonomous_retry_per_hypothesis_24h
        and not (
          coalesce(r.metadata->>'failed_candidate_sha','') <> ''
          and (
            select count(*)
            from public.brian_evolution_engineering_runs stalled_run
            join public.brian_evolution_codegen_requests stalled_req on stalled_req.request_id=stalled_run.request_id
            where stalled_run.hypothesis_id=r.hypothesis_id
              and stalled_run.status='BLOCKED'
              and stalled_run.commit_sha is null
              and stalled_run.updated_at >= now() - make_interval(hours => autonomous_stalled_cooldown_hours)
              and coalesce(stalled_req.metadata->>'failed_candidate_sha','')=coalesce(r.metadata->>'failed_candidate_sha','')
              and brian_private.engineering_request_is_retry(coalesce(stalled_req.metadata,'{}'::jsonb))
          ) >= 2
        )
      )
    )
  order by
    case
      when p_request_id is null
       and autonomous_fresh_count_24h < autonomous_fresh_reserve_24h
       and brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb)) then 1
      else 0
    end asc,
    case
      when jsonb_typeof(r.metadata->'priority') = 'number' then (r.metadata->>'priority')::numeric
      else 0
    end desc,
    r.requested_at desc,
    r.created_at desc,
    r.request_id desc
  for update of r skip locked
  limit 1;

  if not found then return null; end if;

  request_is_retry := brian_private.engineering_request_is_retry(coalesce(req.metadata,'{}'::jsonb));

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
      'queue_tie_break','fresh-reserve,retry-bounds,priority,requested_at,created_at,request_id',
      'transport_retry_safe',true,
      'claim_mode',case when p_request_id is null then 'AUTONOMOUS' else 'MANUAL_REQUEST' end,
      'request_class',case when request_is_retry then 'RETRY' else 'FRESH' end,
      'autonomous_claim_limit_24h',autonomous_limit_24h,
      'autonomous_fresh_reserve_24h',autonomous_fresh_reserve_24h,
      'autonomous_retry_claim_limit_24h',autonomous_retry_limit_24h,
      'autonomous_retry_per_hypothesis_24h',autonomous_retry_per_hypothesis_24h,
      'autonomous_stalled_cooldown_hours',autonomous_stalled_cooldown_hours,
      'anti_stuck_policy','FAIR_QUEUE_STALLED_NO_DELTA_V2',
      'world_claim_revalidated',case when p_request_id is null and coalesce(req.metadata->>'world_engineering','false')='true' then true else null end,
      'world_source_trust_floor',world_trust_floor,
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
      'request_class',case when request_is_retry then 'RETRY' else 'FRESH' end,
      'anti_stuck_policy','FAIR_QUEUE_STALLED_NO_DELTA_V2',
      'world_engineering',coalesce(req.metadata->>'world_engineering','false')='true',
      'world_claim_revalidated',p_request_id is null and coalesce(req.metadata->>'world_engineering','false')='true',
      'autonomous_count_24h_before_claim',autonomous_count_24h,
      'autonomous_fresh_count_24h_before_claim',autonomous_fresh_count_24h,
      'autonomous_retry_count_24h_before_claim',autonomous_retry_count_24h,
      'autonomous_limit_24h',autonomous_limit_24h,
      'autonomous_fresh_reserve_24h',autonomous_fresh_reserve_24h,
      'autonomous_retry_limit_24h',autonomous_retry_limit_24h
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
$function$
 then
    world_trust_floor := (cfg.metadata->>'world_source_trust_floor')::numeric;
  end if;

  begin
    provider_backoff_until := nullif(cfg.metadata->>'engineering_provider_backoff_until','')::timestamptz;
  exception when others then
    provider_backoff_until := null;
  end;
  if p_request_id is null and provider_backoff_until is not null and provider_backoff_until > now() then
    return null;
  end if;

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

  if p_request_id is null then
    select
      count(*),
      count(*) filter (where not brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))),
      count(*) filter (where brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb)))
    into autonomous_count_24h, autonomous_fresh_count_24h, autonomous_retry_count_24h
    from public.brian_evolution_engineering_runs er
    join public.brian_evolution_codegen_requests r on r.request_id=er.request_id
    where er.created_at >= now() - interval '24 hours'
      and er.metadata->>'claim_mode' = 'AUTONOMOUS';

    -- Normal ceiling remains 4/24h. If legacy retry spam already consumed the ceiling,
    -- allow only enough fresh claims to restore the reserved fresh-work slots.
    if autonomous_count_24h >= autonomous_limit_24h
       and autonomous_fresh_count_24h >= autonomous_fresh_reserve_24h then
      return null;
    end if;
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
    and (p_request_id is not null or coalesce(r.metadata->>'provider_resume','false') <> 'true')
    and case
      when p_request_id is not null then true
      when coalesce(r.metadata->>'world_engineering','false') = 'true' then
        coalesce(cfg.metadata->>'world_to_engineering_enabled','false') = 'true'
        and brian_private.world_engineering_request_is_current(r.metadata, world_trust_floor)
      when exists (
        select 1 from unnest(coalesce(r.evidence_refs,'{}'::text[])) as legacy_ref
        where legacy_ref like 'world_source:%'
      ) then false
      else true
    end
    and not (
      p_request_id is null
      and coalesce(r.metadata->>'world_engineering','false') = 'true'
      and coalesce(r.metadata->>'parent_rotation_policy','') = 'STABLE_ONCE'
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
    and (
      p_request_id is not null
      or (
        not brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))
        and autonomous_fresh_count_24h < autonomous_limit_24h
        and (
          autonomous_count_24h < autonomous_limit_24h
          or autonomous_fresh_count_24h < autonomous_fresh_reserve_24h
        )
      )
      or (
        brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))
        and autonomous_count_24h < autonomous_limit_24h
        and autonomous_retry_count_24h < autonomous_retry_limit_24h
        and (
          select count(*)
          from public.brian_evolution_engineering_runs prior_run
          join public.brian_evolution_codegen_requests prior_req on prior_req.request_id=prior_run.request_id
          where prior_run.hypothesis_id=r.hypothesis_id
            and prior_run.created_at >= now() - interval '24 hours'
            and prior_run.metadata->>'claim_mode'='AUTONOMOUS'
            and brian_private.engineering_request_is_retry(coalesce(prior_req.metadata,'{}'::jsonb))
        ) < autonomous_retry_per_hypothesis_24h
        and not (
          coalesce(r.metadata->>'failed_candidate_sha','') <> ''
          and (
            select count(*)
            from public.brian_evolution_engineering_runs stalled_run
            join public.brian_evolution_codegen_requests stalled_req on stalled_req.request_id=stalled_run.request_id
            where stalled_run.hypothesis_id=r.hypothesis_id
              and stalled_run.status='BLOCKED'
              and stalled_run.commit_sha is null
              and stalled_run.updated_at >= now() - make_interval(hours => autonomous_stalled_cooldown_hours)
              and coalesce(stalled_req.metadata->>'failed_candidate_sha','')=coalesce(r.metadata->>'failed_candidate_sha','')
              and brian_private.engineering_request_is_retry(coalesce(stalled_req.metadata,'{}'::jsonb))
          ) >= 2
        )
      )
    )
  order by
    case
      when p_request_id is null
       and autonomous_fresh_count_24h < autonomous_fresh_reserve_24h
       and brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb)) then 1
      else 0
    end asc,
    case
      when jsonb_typeof(r.metadata->'priority') = 'number' then (r.metadata->>'priority')::numeric
      else 0
    end desc,
    r.requested_at desc,
    r.created_at desc,
    r.request_id desc
  for update of r skip locked
  limit 1;

  if not found then return null; end if;

  request_is_retry := brian_private.engineering_request_is_retry(coalesce(req.metadata,'{}'::jsonb));

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
      'queue_tie_break','fresh-reserve,retry-bounds,priority,requested_at,created_at,request_id',
      'transport_retry_safe',true,
      'claim_mode',case when p_request_id is null then 'AUTONOMOUS' else 'MANUAL_REQUEST' end,
      'request_class',case when request_is_retry then 'RETRY' else 'FRESH' end,
      'autonomous_claim_limit_24h',autonomous_limit_24h,
      'autonomous_fresh_reserve_24h',autonomous_fresh_reserve_24h,
      'autonomous_retry_claim_limit_24h',autonomous_retry_limit_24h,
      'autonomous_retry_per_hypothesis_24h',autonomous_retry_per_hypothesis_24h,
      'autonomous_stalled_cooldown_hours',autonomous_stalled_cooldown_hours,
      'anti_stuck_policy','FAIR_QUEUE_STALLED_NO_DELTA_V2',
      'world_claim_revalidated',case when p_request_id is null and coalesce(req.metadata->>'world_engineering','false')='true' then true else null end,
      'world_source_trust_floor',world_trust_floor,
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
      'request_class',case when request_is_retry then 'RETRY' else 'FRESH' end,
      'anti_stuck_policy','FAIR_QUEUE_STALLED_NO_DELTA_V2',
      'world_engineering',coalesce(req.metadata->>'world_engineering','false')='true',
      'world_claim_revalidated',p_request_id is null and coalesce(req.metadata->>'world_engineering','false')='true',
      'autonomous_count_24h_before_claim',autonomous_count_24h,
      'autonomous_fresh_count_24h_before_claim',autonomous_fresh_count_24h,
      'autonomous_retry_count_24h_before_claim',autonomous_retry_count_24h,
      'autonomous_limit_24h',autonomous_limit_24h,
      'autonomous_fresh_reserve_24h',autonomous_fresh_reserve_24h,
      'autonomous_retry_limit_24h',autonomous_retry_limit_24h
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
$function$
;

CREATE OR REPLACE FUNCTION public.brian_engineering_preflight_v1()
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'brian_private', 'pg_temp'
AS $function$
declare
  cfg public.brian_evolution_engineering_control%rowtype;
  active_count integer := 0;
  provider_resume_count_24h integer := 0;
  provider_resume_limit_24h integer := 3;
  autonomous_count_24h integer := 0;
  autonomous_fresh_count_24h integer := 0;
  autonomous_retry_count_24h integer := 0;
  autonomous_limit_24h integer := 4;
  autonomous_fresh_reserve_24h integer := 2;
  autonomous_retry_limit_24h integer := 2;
  circuit_until timestamptz := null;
  provider_backoff_until timestamptz := null;
  world_trust_floor numeric := 0.72;
  pending_provider integer := 0;
  pending_fresh integer := 0;
  pending_retry integer := 0;
begin
  select * into cfg
  from public.brian_evolution_engineering_control
  where control_id='default';

  if not found then
    return jsonb_build_object('should_run',false,'reason','CONTROL_MISSING');
  end if;
  if cfg.require_human_approval is not true then
    return jsonb_build_object('should_run',false,'reason','HUMAN_APPROVAL_INVARIANT_OFF');
  end if;
  if cfg.autonomous_claim_enabled is not true then
    return jsonb_build_object('should_run',false,'reason','AUTONOMOUS_CLAIM_DISABLED');
  end if;

  begin
    circuit_until := nullif(cfg.metadata->>'provider_circuit_open_until','')::timestamptz;
  exception when others then
    circuit_until := null;
  end;
  if circuit_until is not null and circuit_until > now() then
    return jsonb_build_object(
      'should_run',false,'reason','PROVIDER_CIRCUIT_OPEN','circuit_until',circuit_until
    );
  end if;

  begin
    provider_backoff_until := nullif(cfg.metadata->>'engineering_provider_backoff_until','')::timestamptz;
  exception when others then
    provider_backoff_until := null;
  end;
  if provider_backoff_until is not null and provider_backoff_until > now() then
    return jsonb_build_object(
      'should_run',false,'reason','PROVIDER_BACKOFF_ACTIVE','backoff_until',provider_backoff_until
    );
  end if;

  select count(*) into active_count
  from public.brian_evolution_engineering_runs
  where status in ('RUNNING','WAITING')
    and phase not in ('HUMAN_APPROVAL','COMPLETE','BLOCKED','ROLLBACK');
  if active_count >= cfg.max_concurrent_runs then
    return jsonb_build_object('should_run',false,'reason','CONCURRENCY_FULL','active_count',active_count);
  end if;

  if coalesce(cfg.metadata->>'provider_resume_claim_limit_24h','') ~ '^[0-9]+$' then
    provider_resume_limit_24h := greatest(1,least(24,(cfg.metadata->>'provider_resume_claim_limit_24h')::integer));
  end if;
  if coalesce(cfg.metadata->>'autonomous_claim_limit_24h','') ~ '^[0-9]+$' then
    autonomous_limit_24h := greatest(1,least(24,(cfg.metadata->>'autonomous_claim_limit_24h')::integer));
  end if;
  if coalesce(cfg.metadata->>'autonomous_fresh_reserve_24h','') ~ '^[0-9]+$' then
    autonomous_fresh_reserve_24h := greatest(1,least(autonomous_limit_24h,(cfg.metadata->>'autonomous_fresh_reserve_24h')::integer));
  end if;
  if coalesce(cfg.metadata->>'autonomous_retry_claim_limit_24h','') ~ '^[0-9]+$' then
    autonomous_retry_limit_24h := greatest(0,least(autonomous_limit_24h,(cfg.metadata->>'autonomous_retry_claim_limit_24h')::integer));
  end if;
  if coalesce(cfg.metadata->>'world_source_trust_floor','') ~ '^(0([.][0-9]+)?|1([.]0+)?)$' then
    world_trust_floor := (cfg.metadata->>'world_source_trust_floor')::numeric;
  end if;

  select count(*) into provider_resume_count_24h
  from public.brian_evolution_engineering_runs er
  join public.brian_evolution_codegen_requests r on r.request_id=er.request_id
  where er.created_at >= now()-interval '24 hours'
    and er.metadata->>'claim_mode'='AUTONOMOUS'
    and coalesce(r.metadata->>'provider_resume','false')='true';

  if provider_resume_count_24h < provider_resume_limit_24h then
    select count(*) into pending_provider
    from public.brian_evolution_codegen_requests r
    where coalesce(r.metadata->>'provider_resume','false')='true'
      and r.required_human_review is true
      and r.shadow_only is true
      and r.live_execution is false
      and r.autonomous_apply_allowed is false
      and coalesce((r.metadata->>'provider_resume_after')::timestamptz,now()) <= now()
      and not exists (
        select 1 from public.brian_evolution_engineering_runs er where er.request_id=r.request_id
      );
  end if;

  select
    count(*),
    count(*) filter (where not brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))),
    count(*) filter (where brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb)))
  into autonomous_count_24h,autonomous_fresh_count_24h,autonomous_retry_count_24h
  from public.brian_evolution_engineering_runs er
  join public.brian_evolution_codegen_requests r on r.request_id=er.request_id
  where er.created_at >= now()-interval '24 hours'
    and er.metadata->>'claim_mode'='AUTONOMOUS'
    and coalesce(r.metadata->>'provider_resume','false') <> 'true';

  select count(*) filter (
           where not brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))
         ),
         count(*) filter (
           where brian_private.engineering_request_is_retry(coalesce(r.metadata,'{}'::jsonb))
         )
  into pending_fresh,pending_retry
  from public.brian_evolution_codegen_requests r
  where coalesce(r.metadata->>'provider_resume','false') <> 'true'
    and r.required_human_review is true
    and r.shadow_only is true
    and r.live_execution is false
    and r.autonomous_apply_allowed is false
    and not exists (
      select 1 from public.brian_evolution_engineering_runs er where er.request_id=r.request_id
    )
    and case
      when coalesce(r.metadata->>'world_engineering','false')='true' then
        coalesce(cfg.metadata->>'world_to_engineering_enabled','false')='true'
        and brian_private.world_engineering_request_is_current(r.metadata,world_trust_floor)
      when exists (
        select 1 from unnest(coalesce(r.evidence_refs,'{}'::text[])) as legacy_ref
        where legacy_ref like 'world_source:%'
      ) then false
      else true
    end
    and not exists (
      select 1
      from public.brian_evolution_codegen_requests newer
      where newer.hypothesis_id=r.hypothesis_id
        and (
          newer.requested_at > r.requested_at
          or (newer.requested_at=r.requested_at and newer.created_at>r.created_at)
          or (newer.requested_at=r.requested_at and newer.created_at=r.created_at and newer.request_id>r.request_id)
        )
    );

  pending_fresh := case
    when autonomous_fresh_count_24h < autonomous_limit_24h
     and (autonomous_count_24h < autonomous_limit_24h or autonomous_fresh_count_24h < autonomous_fresh_reserve_24h)
    then pending_fresh else 0 end;

  pending_retry := case
    when autonomous_count_24h < autonomous_limit_24h
     and autonomous_retry_count_24h < autonomous_retry_limit_24h
    then pending_retry else 0 end;

  return jsonb_build_object(
    'should_run',(pending_provider+pending_fresh+pending_retry)>0,
    'reason',case when (pending_provider+pending_fresh+pending_retry)>0 then 'WORK_AVAILABLE' else 'QUEUE_IDLE_OR_LIMITED' end,
    'pending_provider_resume',pending_provider,
    'pending_fresh',pending_fresh,
    'pending_retry',pending_retry,
    'provider_resume_count_24h',provider_resume_count_24h,
    'provider_resume_limit_24h',provider_resume_limit_24h,
    'autonomous_count_24h',autonomous_count_24h,
    'active_count',active_count
  );
end;
$function$
;
