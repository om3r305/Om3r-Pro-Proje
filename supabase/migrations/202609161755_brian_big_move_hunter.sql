-- Brian Big Move Hunter: event/opportunity bridge. MAIN only; DIP remains untouched.
DO $$
DECLARE jid bigint;
BEGIN
  SELECT jobid INTO jid FROM cron.job WHERE jobname='brian-big-move-hunter-2m' LIMIT 1;
  IF jid IS NOT NULL THEN PERFORM cron.unschedule(jid); END IF;
  PERFORM cron.schedule(
    'brian-big-move-hunter-2m',
    '*/2 * * * *',
    $job$
    select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-big-move-hunter' from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object(
        'Content-Type','application/json',
        'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
        'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
        'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1)
      ),
      body := '{}'::jsonb,
      timeout_milliseconds := 50000
    );
    $job$
  );
END $$;
