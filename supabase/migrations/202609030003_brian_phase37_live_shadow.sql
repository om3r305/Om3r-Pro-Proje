-- Brian Phase 3.7 prospective live shadow experiment.
-- Starts forward from deployment time; historical backfill and live exchange execution are forbidden.

create table if not exists public.brian_live_shadow_experiments (
  experiment_id text primary key,
  started_at timestamptz not null,
  evidence_class text not null check (evidence_class = 'PROSPECTIVE_DEVELOPMENT_SHADOW'),
  checkpoint_raw_state_id text not null,
  checkpoint_portable_fingerprint text not null,
  checkpoint_source_run_id text not null,
  symbols text[] not null,
  timeframe text not null check (timeframe = '5m'),
  starting_equity numeric not null check (starting_equity > 0),
  policies text[] not null,
  config jsonb not null,
  shadow_only boolean not null default true check (shadow_only),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_live_shadow_ticks (
  tick_id text primary key,
  experiment_id text not null references public.brian_live_shadow_experiments(experiment_id),
  policy_kind text not null check (policy_kind in ('NATIVE','PROFIT')),
  observed_at timestamptz not null,
  feature_close_at timestamptz not null,
  raw_capture_id text not null references public.brian_raw_captures(capture_id),
  equity_before_mark numeric not null check (equity_before_mark >= 0),
  period_pnl numeric not null,
  equity_after_mark numeric not null check (equity_after_mark >= 0),
  trading_cost numeric not null check (trading_cost >= 0),
  equity_after_costs numeric not null check (equity_after_costs >= 0),
  peak_equity_after numeric not null check (peak_equity_after >= equity_after_costs),
  drawdown_pct double precision not null check (drawdown_pct between 0 and 100),
  max_drawdown_pct_after double precision not null check (max_drawdown_pct_after between 0 and 100),
  turnover_notional numeric not null check (turnover_notional >= 0),
  prior_weights jsonb not null,
  drifted_weights jsonb not null,
  target_weights jsonb not null,
  observed_mid_prices jsonb not null,
  observed_spread_bps jsonb not null,
  feature_hash text not null,
  diagnostics jsonb not null,
  evidence_class text not null check (evidence_class = 'PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  created_at timestamptz not null default now(),
  constraint brian_live_tick_causal check (observed_at > feature_close_at),
  unique (experiment_id, policy_kind, observed_at)
);

create index if not exists brian_live_shadow_ticks_policy_time_idx
  on public.brian_live_shadow_ticks(experiment_id, policy_kind, observed_at desc);

alter table public.brian_live_shadow_experiments enable row level security;
alter table public.brian_live_shadow_ticks enable row level security;

revoke all on public.brian_live_shadow_experiments from anon, authenticated;
revoke all on public.brian_live_shadow_ticks from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_live_shadow_experiments from service_role;
revoke update, delete, truncate, references, trigger on public.brian_live_shadow_ticks from service_role;
grant select, insert on public.brian_live_shadow_experiments to service_role;
grant select, insert on public.brian_live_shadow_ticks to service_role;

drop trigger if exists brian_live_shadow_experiments_append_only on public.brian_live_shadow_experiments;
create trigger brian_live_shadow_experiments_append_only before update or delete on public.brian_live_shadow_experiments
  for each row execute function public.brian_reject_mutation();
drop trigger if exists brian_live_shadow_ticks_append_only on public.brian_live_shadow_ticks;
create trigger brian_live_shadow_ticks_append_only before update or delete on public.brian_live_shadow_ticks
  for each row execute function public.brian_reject_mutation();

create or replace function brian_private.schedule_live_shadow_collector()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  existing_job bigint;
  scheduled_job bigint;
begin
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_project_url') then
    raise exception 'BRIAN_RUNTIME: Vault secret brian_project_url is missing';
  end if;
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_anon_jwt') then
    raise exception 'BRIAN_RUNTIME: Vault secret brian_anon_jwt is missing';
  end if;
  for existing_job in select jobid from cron.job where jobname = 'brian-live-shadow-5m' loop
    perform cron.unschedule(existing_job);
  end loop;
  scheduled_job := cron.schedule(
    'brian-live-shadow-5m',
    '1-59/5 * * * *',
    $cron$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-live-shadow' from vault.decrypted_secrets where name='brian_project_url' limit 1),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1)
        ),
        body := '{}'::jsonb
      );
    $cron$
  );
  return scheduled_job;
end;
$$;

revoke all on function brian_private.schedule_live_shadow_collector() from public;
revoke all on function brian_private.schedule_live_shadow_collector() from anon, authenticated, service_role;
grant execute on function brian_private.schedule_live_shadow_collector() to postgres;
