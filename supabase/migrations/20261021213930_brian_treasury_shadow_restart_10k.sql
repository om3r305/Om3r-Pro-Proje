-- Operator request (2026-10-21): restart the SHADOW treasury from $10,000 to see what the current
-- system does from a clean slate. Nothing is deleted: all earlier snapshots/actions stay; a new
-- genesis snapshot is appended and the engine continues from the latest snapshot. Only allowed
-- while the shadow book is flat, so no open position is orphaned. Shadow only, no real money.
do $$
declare
  prev public.brian_treasury_shadow_snapshots;
  t timestamptz := clock_timestamp();
begin
  select * into prev from public.brian_treasury_shadow_snapshots order by observed_at desc, created_at desc limit 1;
  if prev.snapshot_id is null or jsonb_array_length(prev.positions) <> 0 then
    raise exception 'treasury restart refused: shadow book not flat';
  end if;
  insert into public.brian_treasury_shadow_snapshots (
    snapshot_id, cycle_id, observed_at, previous_snapshot_id, starting_equity_usd, cash_usd, equity_usd,
    realized_pnl_usd, cumulative_costs_usd, deployment_usd, deployment_pct, cash_reserve_pct, positions,
    action_count, promotion_gate_open, promotion_gate_ref, promotion_gate_reason, treasury_version,
    gate_version, blocked_reasons, metadata, evidence_class, shadow_only, live_execution)
  values (
    encode(extensions.digest('operator-restart-10k|' || t::text, 'sha256'), 'hex'),
    'operator-restart-10k-20261021', t, prev.snapshot_id, 10000, 10000, 10000,
    0, 0, 0, 0, 1, '[]'::jsonb,
    0, prev.promotion_gate_open, prev.promotion_gate_ref, prev.promotion_gate_reason, prev.treasury_version,
    prev.gate_version, prev.blocked_reasons,
    jsonb_build_object('operator_restart', true, 'reason', 'clean slate requested by operator 2026-10-21',
      'previous_snapshot_id', prev.snapshot_id, 'previous_starting_equity_usd', prev.starting_equity_usd,
      'previous_equity_usd', prev.equity_usd, 'previous_realized_pnl_usd', prev.realized_pnl_usd,
      'previous_cumulative_costs_usd', prev.cumulative_costs_usd),
    'PROSPECTIVE_EVOLUTION_SHADOW', true, false);
end $$;

-- Period 4: measurement from the $10,000 restart (period 3 began on the old $5,000 session and
-- will show SESSION_CHANGED; it stays recorded as is).
-- NOTE: superseded by period 5 (20261021213955) because this capture ran inside the same
-- transaction and still saw the old snapshot.
insert into public.brian_shadow_protocols (protocol_id, engine_id, starts_at, ends_at, rules)
select 'forward-20261021-v4-treasury-10k', 'treasury', now(), now() + interval '14 days', v1.rules
from public.brian_shadow_protocols v1
where v1.protocol_id = 'forward-20260922-v1-treasury'
on conflict (protocol_id) do nothing;

select public.brian_capture_shadow_observations();
