-- Corrective migration after operator clarification.
-- Restore Brian Evolution OS. Cloudflare Worker decommissioning is handled separately.

update public.brian_system_job_registry
set operator_managed=true
where job_name in (
  'brian-evolution-alpha-edge-2m',
  'brian-evolution-experiment-runner-hourly',
  'brian-evolution-orchestrator-10m',
  'brian-evolution-promotion-council-hourly',
  'brian-evolution-researcher-30m',
  'brian-evolution-sandbox-30m',
  'brian-evolution-template-generator-30m'
);

select cron.alter_job(jobid,null,null,null,null,
  case
    when jobname in (
      'brian-evolution-alpha-edge-2m',
      'brian-evolution-experiment-runner-hourly',
      'brian-evolution-promotion-council-hourly',
      'brian-evolution-template-generator-30m'
    ) then true else false end
)
from cron.job
where jobname in (
  'brian-evolution-alpha-edge-2m',
  'brian-evolution-experiment-runner-hourly',
  'brian-evolution-orchestrator-10m',
  'brian-evolution-promotion-council-hourly',
  'brian-evolution-researcher-30m',
  'brian-evolution-sandbox-30m',
  'brian-evolution-template-generator-30m'
);

insert into public.brian_evolution_runtime_config(config_key,config_value,updated_at)
values('brian_evolution_os_enabled','true',clock_timestamp())
on conflict(config_key) do update
set config_value='true',updated_at=clock_timestamp();

update brian_private.core_dispatch_state
set last_attempt_at=clock_timestamp()-interval '1 hour',
    last_response_at=clock_timestamp()-interval '1 hour',
    last_http_status=null
where service_id in ('evolution','researcher','sandbox');
