-- Brian Evolution OS explicit runtime activation control.
-- This migration NEVER activates Evolution jobs by itself.
-- MAIN/ALPHA only; DIP schedules are not inspected, changed or controlled here.

create or replace function brian_private.activate_evolution_os()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  core_jobs jsonb;
  discovery_job bigint;
  researcher_job bigint;
  lab_jobs jsonb;
  codegen_job bigint;
  alpha_edge_job bigint;
  treasury_job bigint;
  ocean_job bigint;
begin
  -- Each scheduler validates the shared Vault runtime contract. Because this function runs
  -- in one transaction, any failure rolls the entire activation back instead of leaving a
  -- half-enabled Evolution stack.
  core_jobs := brian_private.schedule_evolution_os();
  discovery_job := brian_private.schedule_world_discovery_eye();
  researcher_job := brian_private.schedule_evolution_researcher();
  lab_jobs := brian_private.schedule_evolution_lab();
  codegen_job := brian_private.schedule_evolution_template_generator();
  alpha_edge_job := brian_private.schedule_evolution_alpha_edge_challenger();
  treasury_job := brian_private.schedule_evolution_treasury();
  ocean_job := brian_private.schedule_evolution_ocean_worker();

  return jsonb_build_object(
    'status', 'ACTIVATED',
    'core', core_jobs,
    'world_discovery_job', discovery_job,
    'researcher_job', researcher_job,
    'lab', lab_jobs,
    'template_generator_job', codegen_job,
    'alpha_edge_job', alpha_edge_job,
    'treasury_job', treasury_job,
    'ocean_job', ocean_job,
    'shadow_only', true,
    'live_execution', false,
    'dip_controlled', false
  );
end;
$$;

create or replace function brian_private.deactivate_evolution_os()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  job record;
  removed integer := 0;
begin
  for job in
    select jobid, jobname
    from cron.job
    where jobname in (
      'brian-evolution-orchestrator-10m',
      'brian-world-brain-5m',
      'brian-world-discovery-eye-10m',
      'brian-evolution-researcher-30m',
      'brian-evolution-sandbox-30m',
      'brian-evolution-experiment-runner-hourly',
      'brian-evolution-promotion-council-hourly',
      'brian-evolution-template-generator-30m',
      'brian-evolution-alpha-edge-2m',
      'brian-evolution-treasury-1m',
      'brian-evolution-ocean-5m'
    )
  loop
    perform cron.unschedule(job.jobid);
    removed := removed + 1;
  end loop;

  return jsonb_build_object(
    'status', 'DEACTIVATED',
    'jobs_removed', removed,
    'shadow_only', true,
    'live_execution', false,
    'dip_controlled', false
  );
end;
$$;

create or replace function brian_private.evolution_activation_status()
returns jsonb
language sql
security definer
set search_path = pg_catalog, public
as $$
  select jsonb_build_object(
    'active_job_count', count(*),
    'job_names', coalesce(jsonb_agg(jobname order by jobname), '[]'::jsonb),
    'expected_job_count', 11,
    'fully_active', count(*) = 11,
    'shadow_only', true,
    'live_execution', false,
    'dip_controlled', false
  )
  from cron.job
  where jobname in (
    'brian-evolution-orchestrator-10m',
    'brian-world-brain-5m',
    'brian-world-discovery-eye-10m',
    'brian-evolution-researcher-30m',
    'brian-evolution-sandbox-30m',
    'brian-evolution-experiment-runner-hourly',
    'brian-evolution-promotion-council-hourly',
    'brian-evolution-template-generator-30m',
    'brian-evolution-alpha-edge-2m',
    'brian-evolution-treasury-1m',
    'brian-evolution-ocean-5m'
  );
$$;

revoke all on function brian_private.activate_evolution_os() from public, anon, authenticated, service_role;
revoke all on function brian_private.deactivate_evolution_os() from public, anon, authenticated, service_role;
revoke all on function brian_private.evolution_activation_status() from public, anon, authenticated, service_role;
grant execute on function brian_private.activate_evolution_os() to postgres;
grant execute on function brian_private.deactivate_evolution_os() to postgres;
grant execute on function brian_private.evolution_activation_status() to postgres;

-- Intentionally no automatic call to activate_evolution_os().
-- Rollout must explicitly deploy migrations + functions + UI, run smoke checks, then activate.
