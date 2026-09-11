-- Brian Evolution OS Layer 5: unified SHADOW Treasury persistence.
-- GitHub-only until explicit rollout. MAIN/ALPHA only. DIP is outside scope.
-- No exchange/order/withdrawal surface is created by this migration.

create table if not exists public.brian_treasury_shadow_snapshots (
  snapshot_id text primary key,
  cycle_id text not null unique,
  observed_at timestamptz not null,
  previous_snapshot_id text references public.brian_treasury_shadow_snapshots(snapshot_id),
  starting_equity_usd numeric not null check (starting_equity_usd > 0),
  cash_usd numeric not null check (cash_usd >= 0),
  equity_usd numeric not null check (equity_usd > 0),
  realized_pnl_usd numeric not null,
  cumulative_costs_usd numeric not null check (cumulative_costs_usd >= 0),
  deployment_usd numeric not null check (deployment_usd >= 0),
  deployment_pct double precision not null check (deployment_pct >= 0 and deployment_pct <= 1.000001),
  cash_reserve_pct double precision not null check (cash_reserve_pct >= 0 and cash_reserve_pct <= 1.000001),
  positions jsonb not null default '[]'::jsonb check (jsonb_typeof(positions)='array'),
  action_count integer not null default 0 check (action_count >= 0),
  promotion_gate_open boolean not null default false,
  promotion_gate_ref text,
  promotion_gate_reason text not null,
  treasury_version text not null,
  gate_version text not null,
  blocked_reasons text[] not null default '{}',
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_treasury_shadow_actions (
  action_id text primary key,
  cycle_id text not null references public.brian_treasury_shadow_snapshots(cycle_id),
  observed_at timestamptz not null,
  kind text not null check (kind in ('OPEN','EXIT')),
  asset_id text not null,
  direction smallint not null check (direction in (-1,1)),
  capital_usd numeric not null check (capital_usd > 0),
  reference_price numeric not null check (reference_price > 0),
  cost_usd numeric not null check (cost_usd >= 0),
  expected_net_edge_bps double precision not null,
  source_decision_id text not null,
  reason text not null,
  position_id text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create index if not exists brian_treasury_snapshot_time_idx
  on public.brian_treasury_shadow_snapshots(observed_at desc, created_at desc);
create index if not exists brian_treasury_action_time_idx
  on public.brian_treasury_shadow_actions(observed_at desc, created_at desc);
create index if not exists brian_treasury_action_asset_time_idx
  on public.brian_treasury_shadow_actions(asset_id, observed_at desc);

alter table public.brian_treasury_shadow_snapshots enable row level security;
alter table public.brian_treasury_shadow_actions enable row level security;

revoke all on public.brian_treasury_shadow_snapshots from anon, authenticated;
revoke all on public.brian_treasury_shadow_actions from anon, authenticated;
revoke insert, update, delete, truncate, references, trigger on public.brian_treasury_shadow_snapshots from service_role;
revoke insert, update, delete, truncate, references, trigger on public.brian_treasury_shadow_actions from service_role;
grant select on public.brian_treasury_shadow_snapshots to service_role;
grant select on public.brian_treasury_shadow_actions to service_role;

drop trigger if exists brian_treasury_shadow_snapshots_append_only on public.brian_treasury_shadow_snapshots;
create trigger brian_treasury_shadow_snapshots_append_only
before update or delete on public.brian_treasury_shadow_snapshots
for each row execute function public.brian_reject_mutation();

drop trigger if exists brian_treasury_shadow_actions_append_only on public.brian_treasury_shadow_actions;
create trigger brian_treasury_shadow_actions_append_only
before update or delete on public.brian_treasury_shadow_actions
for each row execute function public.brian_reject_mutation();

create or replace view public.brian_treasury_shadow_latest
with (security_invoker = true) as
select *
from public.brian_treasury_shadow_snapshots
order by observed_at desc, created_at desc
limit 1;

revoke all on public.brian_treasury_shadow_latest from anon, authenticated;
grant select on public.brian_treasury_shadow_latest to service_role;

-- Atomic append-only commit: one snapshot and all actions are persisted in one transaction.
create or replace function public.brian_commit_treasury_shadow_cycle(p_snapshot jsonb, p_actions jsonb)
returns text
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_cycle_id text;
  v_snapshot_id text;
  v_action jsonb;
  v_action_id text;
  v_actions jsonb;
begin
  if jsonb_typeof(p_snapshot) <> 'object' then
    raise exception 'TREASURY_COMMIT: snapshot must be an object';
  end if;
  v_actions := coalesce(p_actions, '[]'::jsonb);
  if jsonb_typeof(v_actions) <> 'array' then
    raise exception 'TREASURY_COMMIT: actions must be an array';
  end if;

  v_cycle_id := nullif(trim(p_snapshot->>'cycle_id'), '');
  v_snapshot_id := nullif(trim(p_snapshot->>'snapshot_id'), '');
  if v_cycle_id is null or v_snapshot_id is null then
    raise exception 'TREASURY_COMMIT: cycle_id and snapshot_id are required';
  end if;

  -- Idempotent retry: a completed cycle is never duplicated.
  if exists(select 1 from public.brian_treasury_shadow_snapshots where cycle_id=v_cycle_id) then
    return v_cycle_id;
  end if;

  insert into public.brian_treasury_shadow_snapshots(
    snapshot_id, cycle_id, observed_at, previous_snapshot_id,
    starting_equity_usd, cash_usd, equity_usd, realized_pnl_usd, cumulative_costs_usd,
    deployment_usd, deployment_pct, cash_reserve_pct, positions, action_count,
    promotion_gate_open, promotion_gate_ref, promotion_gate_reason,
    treasury_version, gate_version, blocked_reasons, metadata,
    evidence_class, shadow_only, live_execution
  ) values (
    v_snapshot_id,
    v_cycle_id,
    (p_snapshot->>'observed_at')::timestamptz,
    nullif(p_snapshot->>'previous_snapshot_id',''),
    (p_snapshot->>'starting_equity_usd')::numeric,
    (p_snapshot->>'cash_usd')::numeric,
    (p_snapshot->>'equity_usd')::numeric,
    (p_snapshot->>'realized_pnl_usd')::numeric,
    (p_snapshot->>'cumulative_costs_usd')::numeric,
    (p_snapshot->>'deployment_usd')::numeric,
    (p_snapshot->>'deployment_pct')::double precision,
    (p_snapshot->>'cash_reserve_pct')::double precision,
    coalesce(p_snapshot->'positions','[]'::jsonb),
    jsonb_array_length(v_actions),
    coalesce((p_snapshot->>'promotion_gate_open')::boolean,false),
    nullif(p_snapshot->>'promotion_gate_ref',''),
    coalesce(p_snapshot->>'promotion_gate_reason','unspecified'),
    p_snapshot->>'treasury_version',
    p_snapshot->>'gate_version',
    coalesce(array(select jsonb_array_elements_text(coalesce(p_snapshot->'blocked_reasons','[]'::jsonb))), '{}'),
    coalesce(p_snapshot->'metadata','{}'::jsonb),
    'PROSPECTIVE_EVOLUTION_SHADOW', true, false
  );

  for v_action in select value from jsonb_array_elements(v_actions) loop
    if jsonb_typeof(v_action) <> 'object' then
      raise exception 'TREASURY_COMMIT: action must be an object';
    end if;
    v_action_id := nullif(trim(v_action->>'action_id'), '');
    if v_action_id is null then
      raise exception 'TREASURY_COMMIT: action_id required';
    end if;
    insert into public.brian_treasury_shadow_actions(
      action_id, cycle_id, observed_at, kind, asset_id, direction,
      capital_usd, reference_price, cost_usd, expected_net_edge_bps,
      source_decision_id, reason, position_id, metadata,
      evidence_class, shadow_only, live_execution
    ) values (
      v_action_id,
      v_cycle_id,
      (v_action->>'observed_at')::timestamptz,
      v_action->>'kind',
      v_action->>'asset_id',
      (v_action->>'direction')::smallint,
      (v_action->>'capital_usd')::numeric,
      (v_action->>'reference_price')::numeric,
      (v_action->>'cost_usd')::numeric,
      (v_action->>'expected_net_edge_bps')::double precision,
      v_action->>'source_decision_id',
      v_action->>'reason',
      nullif(v_action->>'position_id',''),
      coalesce(v_action->'metadata','{}'::jsonb),
      'PROSPECTIVE_EVOLUTION_SHADOW', true, false
    );
  end loop;

  return v_cycle_id;
end;
$$;

revoke all on function public.brian_commit_treasury_shadow_cycle(jsonb,jsonb) from public, anon, authenticated;
grant execute on function public.brian_commit_treasury_shadow_cycle(jsonb,jsonb) to service_role;
