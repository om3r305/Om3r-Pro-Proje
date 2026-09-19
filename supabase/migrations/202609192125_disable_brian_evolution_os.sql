-- Brian Evolution OS operator disablement.
-- Self-development / autonomous innovation layer is intentionally disabled.
-- Treasury, Ocean, World and ALPHA are not disabled by this migration.

do $$
declare r record;
begin
  for r in
    select jobid
    from cron.job
    where jobname in (
      'brian-evolution-alpha-edge-2m',
      'brian-evolution-experiment-runner-hourly',
      'brian-evolution-orchestrator-10m',
      'brian-evolution-promotion-council-hourly',
      'brian-evolution-researcher-30m',
      'brian-evolution-sandbox-30m',
      'brian-evolution-template-generator-30m'
    )
  loop
    perform cron.alter_job(r.jobid, null, null, null, null, false);
  end loop;
end $$;

update public.brian_system_job_registry
set operator_managed=false
where job_name in (
  'brian-evolution-alpha-edge-2m',
  'brian-evolution-experiment-runner-hourly',
  'brian-evolution-orchestrator-10m',
  'brian-evolution-promotion-council-hourly',
  'brian-evolution-researcher-30m',
  'brian-evolution-sandbox-30m',
  'brian-evolution-template-generator-30m'
);

insert into public.brian_evolution_runtime_config(config_key,config_value,updated_at)
values('brian_evolution_os_enabled','false',clock_timestamp())
on conflict(config_key) do update
set config_value='false',updated_at=clock_timestamp();

insert into brian_private.core_dispatch_state(
  service_id,last_attempt_at,last_response_at,last_http_status,attempt_count
)
values
  ('evolution','2100-01-01 00:00:00+00','2100-01-01 00:00:00+00',299,0),
  ('researcher','2100-01-01 00:00:00+00','2100-01-01 00:00:00+00',299,0),
  ('sandbox','2100-01-01 00:00:00+00','2100-01-01 00:00:00+00',299,0)
on conflict(service_id) do update set
  last_attempt_at=excluded.last_attempt_at,
  last_response_at=excluded.last_response_at,
  last_http_status=excluded.last_http_status;
