-- Brian Evolution OS Layer 6: Ocean prospective SHADOW run persistence.
-- GitHub-only until explicit rollout. MAIN/ALPHA only; no live-money execution surfaces.

create table if not exists public.brian_ocean_run_commands (
  command_id text primary key,
  run_id text not null,
  command text not null check (command in ('START','STOP')),
  requested_at timestamptz not null,
  duration_hours integer check (
    (command='START' and duration_hours in (24,48)) or
    (command='STOP' and duration_hours is null)
  ),
  reason text,
  requested_by text not null default 'dashboard',
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW' check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  unique(run_id,command)
);

create table if not exists public.brian_ocean_run_checkpoints (
  checkpoint_id text primary key,
  run_id text not null,
  observed_at timestamptz not null,
  treasury_snapshot_id text,
  treasury_equity_usd numeric,
  treasury_cash_usd numeric,
  treasury_deployment_usd numeric,
  treasury_open_positions integer check (treasury_open_positions is null or treasury_open_positions >= 0),
  collector_runs_window integer not null default 0 check (collector_runs_window >= 0),
  collector_failures_window integer not null default 0 check (collector_failures_window >= 0),
  collector_degraded_window integer not null default 0 check (collector_degraded_window >= 0),
  payload jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW' check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_ocean_run_reports (
  report_id text primary key,
  run_id text not null unique,
  started_at timestamptz not null,
  ended_at timestamptz not null check (ended_at >= started_at),
  duration_hours double precision not null check (duration_hours >= 0 and duration_hours <= 48.01),
  summary jsonb not null check (jsonb_typeof(summary)='object'),
  evidence_refs text[] not null default '{}',
  report_version text not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW' check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create index if not exists brian_ocean_commands_time_idx on public.brian_ocean_run_commands(requested_at desc,created_at desc);
create index if not exists brian_ocean_commands_run_idx on public.brian_ocean_run_commands(run_id,requested_at asc);
create index if not exists brian_ocean_checkpoints_run_time_idx on public.brian_ocean_run_checkpoints(run_id,observed_at desc);
create index if not exists brian_ocean_reports_end_idx on public.brian_ocean_run_reports(ended_at desc);

alter table public.brian_ocean_run_commands enable row level security;
alter table public.brian_ocean_run_checkpoints enable row level security;
alter table public.brian_ocean_run_reports enable row level security;

revoke all on public.brian_ocean_run_commands from anon,authenticated;
revoke all on public.brian_ocean_run_checkpoints from anon,authenticated;
revoke all on public.brian_ocean_run_reports from anon,authenticated;

revoke update,delete,truncate,references,trigger on public.brian_ocean_run_commands from service_role;
revoke update,delete,truncate,references,trigger on public.brian_ocean_run_checkpoints from service_role;
revoke update,delete,truncate,references,trigger on public.brian_ocean_run_reports from service_role;

grant select,insert on public.brian_ocean_run_commands to service_role;
grant select,insert on public.brian_ocean_run_checkpoints to service_role;
grant select,insert on public.brian_ocean_run_reports to service_role;

drop trigger if exists brian_ocean_run_commands_append_only on public.brian_ocean_run_commands;
create trigger brian_ocean_run_commands_append_only before update or delete on public.brian_ocean_run_commands
for each row execute function public.brian_reject_mutation();

drop trigger if exists brian_ocean_run_checkpoints_append_only on public.brian_ocean_run_checkpoints;
create trigger brian_ocean_run_checkpoints_append_only before update or delete on public.brian_ocean_run_checkpoints
for each row execute function public.brian_reject_mutation();

drop trigger if exists brian_ocean_run_reports_append_only on public.brian_ocean_run_reports;
create trigger brian_ocean_run_reports_append_only before update or delete on public.brian_ocean_run_reports
for each row execute function public.brian_reject_mutation();
