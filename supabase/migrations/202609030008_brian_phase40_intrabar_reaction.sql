-- Brian Phase 4.0 intrabar reaction eye.
-- Additive prospective shadow only: the frozen Phase 3.7 live-shadow experiment is not mutated.

create table if not exists public.brian_intrabar_reaction_experiments (
  experiment_id text primary key,
  started_at timestamptz not null default now(),
  schema_version text not null,
  config jsonb not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

insert into public.brian_intrabar_reaction_experiments (
  experiment_id, schema_version, config, evidence_class, shadow_only, live_execution
) values (
  'phase40-intrabar-reaction-v1',
  'brian.phase40-intrabar-reaction.v1',
  jsonb_build_object(
    'cadence_seconds', 60,
    'scan_top_n', 50,
    'core_symbols', jsonb_build_array('BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT'),
    'horizon', 'MICRO_1_5M',
    'templates', jsonb_build_array('velocity-micro','volume-burst-micro','breakout-micro','reclaim-micro','taker-flow-micro'),
    'min_support_groups', 2,
    'min_consensus_score', 0.18,
    'overextension_sigma', 3.5,
    'fee_bps', 10.0,
    'slippage_bps', 1.0,
    'learning_enabled', false,
    'historical_backfill_allowed', false,
    'phase37_mutation_allowed', false,
    'automatic_promotion', false
  ),
  'PROSPECTIVE_DEVELOPMENT_SHADOW', true, false
) on conflict (experiment_id) do nothing;

create table if not exists public.brian_intrabar_reaction_events (
  event_id text primary key,
  experiment_id text not null references public.brian_intrabar_reaction_experiments(experiment_id),
  observed_at timestamptz not null,
  asset_id text not null,
  direction smallint not null check (direction between -1 and 1),
  score double precision not null check (score between 0 and 1),
  support_groups text[] not null default '{}',
  conflict_groups text[] not null default '{}',
  source_observation_ids text[] not null default '{}',
  observed_mid_price numeric not null check (observed_mid_price > 0),
  observed_spread_bps double precision not null check (observed_spread_bps >= 0),
  estimated_round_trip_cost_bps double precision not null check (estimated_round_trip_cost_bps >= 0),
  extension_sigma double precision not null check (extension_sigma >= 0),
  late_chase boolean not null,
  status text not null check (status in ('WATCH','ACTIONABLE_SHADOW','VETOED_LATE_CHASE')),
  reason text not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint brian_intrabar_action_requires_direction check (status <> 'ACTIONABLE_SHADOW' or direction <> 0),
  constraint brian_intrabar_action_requires_independence check (status <> 'ACTIONABLE_SHADOW' or cardinality(support_groups) >= 2),
  constraint brian_intrabar_late_chase_not_actionable check (not late_chase or status <> 'ACTIONABLE_SHADOW')
);

create index if not exists brian_intrabar_event_asset_time_idx
  on public.brian_intrabar_reaction_events(asset_id, observed_at desc);
create index if not exists brian_intrabar_event_status_time_idx
  on public.brian_intrabar_reaction_events(status, observed_at desc);

alter table public.brian_intrabar_reaction_experiments enable row level security;
alter table public.brian_intrabar_reaction_events enable row level security;
revoke all on public.brian_intrabar_reaction_experiments from anon, authenticated;
revoke all on public.brian_intrabar_reaction_events from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_intrabar_reaction_experiments from service_role;
revoke update, delete, truncate, references, trigger on public.brian_intrabar_reaction_events from service_role;
grant select, insert on public.brian_intrabar_reaction_experiments to service_role;
grant select, insert on public.brian_intrabar_reaction_events to service_role;

drop trigger if exists brian_intrabar_reaction_experiments_append_only on public.brian_intrabar_reaction_experiments;
create trigger brian_intrabar_reaction_experiments_append_only
before update or delete on public.brian_intrabar_reaction_experiments
for each row execute function public.brian_reject_mutation();

drop trigger if exists brian_intrabar_reaction_events_append_only on public.brian_intrabar_reaction_events;
create trigger brian_intrabar_reaction_events_append_only
before update or delete on public.brian_intrabar_reaction_events
for each row execute function public.brian_reject_mutation();

create or replace function brian_private.schedule_intrabar_reaction_eye()
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

  for existing_job in select jobid from cron.job where jobname = 'brian-intrabar-eye-1m' loop
    perform cron.unschedule(existing_job);
  end loop;

  scheduled_job := cron.schedule(
    'brian-intrabar-eye-1m',
    '* * * * *',
    $cron$select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-intrabar-eye' from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object(
        'Content-Type','application/json',
        'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
        'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1)
      ),
      body := '{}'::jsonb,
      timeout_milliseconds := 40000
    );$cron$
  );
  return scheduled_job;
end;
$$;

revoke all on function brian_private.schedule_intrabar_reaction_eye() from public, anon, authenticated, service_role;
grant execute on function brian_private.schedule_intrabar_reaction_eye() to postgres;
