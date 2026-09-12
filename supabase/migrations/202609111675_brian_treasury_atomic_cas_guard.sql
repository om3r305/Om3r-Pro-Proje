-- Brian Evolution OS Layer 5 hardening: serialize Treasury commits and reject stale forks.
-- GitHub-only until explicit rollout. MAIN/ALPHA only. DIP is outside scope.
--
-- The worker already uses a collector lease, but the database is the final source of truth.
-- This replacement makes the append-only Treasury a compare-and-swap chain even if two
-- service-role callers race, a retry arrives late, or an operator invokes the RPC manually.

create or replace function public.brian_commit_treasury_shadow_cycle(p_snapshot jsonb, p_actions jsonb)
returns text
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_cycle_id text;
  v_snapshot_id text;
  v_previous_snapshot_id text;
  v_latest_snapshot_id text;
  v_latest_observed_at timestamptz;
  v_new_observed_at timestamptz;
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
  v_previous_snapshot_id := nullif(trim(p_snapshot->>'previous_snapshot_id'), '');
  v_new_observed_at := (p_snapshot->>'observed_at')::timestamptz;
  if v_cycle_id is null or v_snapshot_id is null then
    raise exception 'TREASURY_COMMIT: cycle_id and snapshot_id are required';
  end if;
  if v_new_observed_at is null then
    raise exception 'TREASURY_COMMIT: observed_at is required';
  end if;

  -- Serialize the Treasury chain at the database boundary. Collector leases are useful
  -- operationally, but correctness must not depend on the caller keeping a lease.
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('brian_evolution_treasury_chain_v1'));

  -- Exact retry after a successful commit is idempotent, even if newer cycles now exist.
  if exists(select 1 from public.brian_treasury_shadow_snapshots where cycle_id=v_cycle_id) then
    return v_cycle_id;
  end if;

  select snapshot_id, observed_at
    into v_latest_snapshot_id, v_latest_observed_at
  from public.brian_treasury_shadow_snapshots
  order by observed_at desc, created_at desc
  limit 1;

  if v_latest_snapshot_id is null then
    if v_previous_snapshot_id is not null then
      raise exception 'TREASURY_COMMIT_STALE_PARENT: genesis expected null parent, got %', v_previous_snapshot_id;
    end if;
  else
    if v_previous_snapshot_id is distinct from v_latest_snapshot_id then
      raise exception 'TREASURY_COMMIT_STALE_PARENT: expected %, got %', v_latest_snapshot_id, coalesce(v_previous_snapshot_id, '<null>');
    end if;
    if v_new_observed_at <= v_latest_observed_at then
      raise exception 'TREASURY_COMMIT_NON_MONOTONIC_TIME: new % must be after latest %', v_new_observed_at, v_latest_observed_at;
    end if;
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
    v_new_observed_at,
    v_previous_snapshot_id,
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
