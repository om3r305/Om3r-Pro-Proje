-- Activate V8.4.4 Cycle Forecast worker and stop superseded trade workers.
-- Keep foresight/chart support jobs untouched. SHADOW ONLY.

update cron.job
set active=false
where jobname in ('brian-dip-v842-long-worker-10s','brian-dip-v843-profit-worker-10s');

do $do$
begin
  if not exists(select 1 from cron.job where jobname='brian-dip-v844-cycle-worker-10s') then
    perform cron.schedule(
      'brian-dip-v844-cycle-worker-10s',
      '10 seconds',
      $cmd$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-dip-v844-cycle-worker' from vault.decrypted_secrets where name='brian_project_url' limit 1),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1)
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 50000
      );
      $cmd$
    );
  else
    update cron.job set active=true,schedule='10 seconds' where jobname='brian-dip-v844-cycle-worker-10s';
  end if;
end $do$;
