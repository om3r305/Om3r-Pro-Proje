-- Evolution runtime-owned canonical parent pointer. MAIN/ALPHA only; DIP untouched.
-- Service-role workers may read it, but only database-owner rollout tooling may mutate it.

create table if not exists public.brian_evolution_runtime_config (
  config_key text primary key,
  config_value text not null,
  updated_at timestamptz not null default now(),
  constraint brian_evolution_runtime_parent_sha_guard check (
    config_key <> 'canonical_parent_commit' or config_value ~ '^[0-9a-fA-F]{40}$'
  )
);

alter table public.brian_evolution_runtime_config enable row level security;
revoke all on public.brian_evolution_runtime_config from anon, authenticated;
revoke insert, update, delete, truncate, references, trigger on public.brian_evolution_runtime_config from service_role;
grant select on public.brian_evolution_runtime_config to service_role;
