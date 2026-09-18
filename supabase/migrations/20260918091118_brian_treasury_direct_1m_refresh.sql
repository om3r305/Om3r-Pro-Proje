
select cron.alter_job(
  48,
  schedule := '* * * * *',
  active := true
);
