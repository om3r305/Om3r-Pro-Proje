-- Multi-agent (Claude + Grok) shadow assessments, one row per decided candidate
-- (brian2026/multi_agent/live.py). Append-only; client roles have no access; never an order:
-- execution_allowed is constrained to false. Duplicate candidates are ignored by the primary key.

create table if not exists public.brian_multi_agent_decisions (
  candidate_id text primary key,
  decision_time timestamptz not null,
  asset text not null,
  asset_class text not null,
  decision text not null check (decision in
    ('BUY_CANDIDATE', 'SELL_CANDIDATE', 'WAIT', 'REJECT', 'INSUFFICIENT_DATA', 'NO_TRADE')),
  expected_net_edge_bps double precision,
  provider_cost_usd double precision not null default 0 check (provider_cost_usd >= 0),
  debate_rounds smallint not null check (debate_rounds between 0 and 2),
  mock boolean not null default false,
  schema_version text not null,
  record jsonb not null check (
    jsonb_typeof(record) = 'object'
    and record->>'execution_allowed' = 'false'
    and record->>'shadow_only' = 'true'
  ),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);
create index if not exists brian_multi_agent_decisions_time_idx
  on public.brian_multi_agent_decisions (decision_time desc);

drop trigger if exists brian_multi_agent_decisions_append_only on public.brian_multi_agent_decisions;
create trigger brian_multi_agent_decisions_append_only before update or delete on public.brian_multi_agent_decisions
  for each row execute function public.brian_reject_mutation();
alter table public.brian_multi_agent_decisions enable row level security;
revoke all on public.brian_multi_agent_decisions from public, anon, authenticated;
grant select, insert on public.brian_multi_agent_decisions to service_role;
