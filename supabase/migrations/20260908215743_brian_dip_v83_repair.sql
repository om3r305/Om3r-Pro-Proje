-- DIP only. Preserve all learning and ledger history. No session or balance reset.
create index if not exists brian_dip_v8_ledger_entered_episode_idx
  on public.brian_dip_v8_ledger(episode_id) where event_kind in ('BUY','SHORT_OPEN');

CREATE OR REPLACE FUNCTION public.brian_dip_v82_calibration(p_setup text, p_direction text, p_regime text)
 RETURNS TABLE(hit boolean)
 LANGUAGE sql
 STABLE
 SET search_path TO 'pg_catalog', 'public'
AS $function$
  select firsts.hit
  from (
    select distinct on (episode_id) episode_id, hit, decision_at, resolved_at, resolution
    from public.brian_dip_v8_decisions
    where setup=p_setup and direction=p_direction and regime=p_regime
      and venue='SHADOW_PERP'
      and policy_version='dip-v8-dual-20260908.2'
      and metric_version='target-before-invalidation-v8.2'
      and evidence->>'decision_revision'='dip-v8-dual-integrity-20260908.5'
    order by episode_id,decision_at,occurrence_id
  ) firsts
  where resolved_at is not null and firsts.resolution->>'resolver_version'='dip-v8-path-20260908.5'
  order by resolved_at desc
  limit 500
$function$
;

CREATE OR REPLACE FUNCTION public.brian_dip_v82_commit(p_session_id text, p_expected_version bigint, p_owner_token text, p_commit_id text, p_runtime jsonb, p_snapshot jsonb, p_decision jsonb, p_events jsonb)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'pg_catalog', 'public'
AS $function$
declare
  old_state public.brian_dip_v8_runtime%rowtype;
  latest public.brian_dip_session_events%rowtype;
  start_row public.brian_dip_session_events%rowtype;
  lease_row public.brian_collector_leases%rowtype;
  payload_hash text;
  e jsonb;
  d public.brian_dip_v8_decisions%rowtype;
  opens integer:=0; closes integer:=0; wins integer:=0; losses integer:=0;
  delta_cash numeric:=0; delta_realized numeric:=0;
  pos jsonb; old_pos jsonb;
  side text; kind text; lev integer; margin numeric; gross numeric; fee_close numeric; expected_net numeric;
  margin_cap numeric; amb_rate numeric; funding numeric; settlement jsonb; funding_time numeric; previous_funding_time numeric; funding_sum numeric;
begin
  if p_commit_id is null or length(p_commit_id)>160
     or jsonb_typeof(p_events) is distinct from 'array'
     or jsonb_array_length(p_events)>1
     or octet_length(p_runtime::text)>100000
     or octet_length(p_snapshot::text)>150000
     or octet_length(coalesce(p_decision,'null'::jsonb)::text)>15000
  then raise exception 'INVALID_V82_COMMIT'; end if;

  perform pg_advisory_xact_lock(hashtextextended('brian-aggressive-dip-control',0));
  select * into old_state from public.brian_dip_v8_runtime where session_id=p_session_id for update;
  if not found then raise exception 'V82_STATE_MISSING_RECONCILE'; end if;

  payload_hash=md5(jsonb_build_array(p_runtime,p_snapshot,p_decision,p_events)::text);
  if old_state.last_commit_id=p_commit_id then
    if old_state.last_commit_hash<>payload_hash then raise exception 'V82_IDEMPOTENCY_PAYLOAD_CONFLICT'; end if;
    return jsonb_build_object('status','ALREADY_COMMITTED','state_version',old_state.state_version);
  end if;
  if old_state.state_version<>p_expected_version then raise exception 'V82_STATE_VERSION_CONFLICT'; end if;

  select * into latest from public.brian_dip_session_events order by requested_at desc,event_id desc limit 1;
  if not found or latest.session_id is distinct from p_session_id then raise exception 'STALE_V82_SESSION'; end if;
  if latest.event_kind='PAUSE' then
    select * into start_row from public.brian_dip_session_events where session_id=p_session_id and event_kind='START' order by requested_at asc,event_id asc limit 1;
    if not found then raise exception 'V82_START_MISSING'; end if;
    latest.config := start_row.config;
    latest.starting_equity := start_row.starting_equity;
    latest.trade_notional := start_row.trade_notional;
  end if;
  if latest.config->>'policy_version' is distinct from 'dip-v8-dual-20260908.2'
     or latest.config->>'engine_version' is distinct from 'brian-dip-chart-reader-v8-dual'
     or latest.config->>'shadow_only' is distinct from 'true'
     or latest.config->>'live_execution' is distinct from 'false'
     or latest.config->'symbols' is distinct from '["ETHUSDT"]'::jsonb
     or latest.config->>'browser_execution' is distinct from 'false'
     or latest.config->>'server_authoritative' is distinct from 'true'
     or latest.config->>'allow_shadow_short' is distinct from 'true'
     or coalesce((latest.config->>'max_shadow_leverage')::integer,0)<>2
  then raise exception 'INVALID_V82_SESSION_CONTRACT'; end if;

  select * into lease_row from public.brian_collector_leases where collector_id='brian-dip-dual-shadow-worker-v82' for share;
  if lease_row.owner_token is distinct from p_owner_token or lease_row.lease_until is null or lease_row.lease_until<=clock_timestamp() then raise exception 'V82_LEASE_LOST'; end if;

  if p_snapshot->>'session_id' is distinct from p_session_id
     or p_snapshot#>>'{state,serverRuntime,policy_version}' is distinct from 'dip-v8-dual-20260908.2'
     or p_snapshot#>>'{state,serverRuntime,shadow_only}' is distinct from 'true'
     or p_snapshot#>>'{state,serverRuntime,live_execution}' is distinct from 'false'
     or p_snapshot#>>'{state,serverRuntime,dual_direction}' is distinct from 'true'
  then raise exception 'V82_SNAPSHOT_CONTRACT'; end if;

  if (p_runtime->>'start')::numeric is distinct from (old_state.runtime->>'start')::numeric
     or (p_runtime->>'marketCursor')::bigint < (old_state.runtime->>'marketCursor')::bigint
  then raise exception 'V82_STATE_RESET_OR_TIME_REGRESSION'; end if;
  if jsonb_typeof(p_runtime) is distinct from 'object' or not (p_runtime ? 'pos') or exists (
    select 1 from unnest(array['start','cash','realized','trades','wins','losses','marketCursor']) k
    where jsonb_typeof(p_runtime->k) is distinct from 'number'
  ) then raise exception 'V82_INVALID_RUNTIME_FIELDS'; end if;
  if p_snapshot#>'{state,v8}' is distinct from p_runtime
     or (p_snapshot->>'cash')::numeric is distinct from (p_runtime->>'cash')::numeric
     or (p_snapshot->>'realized_pnl')::numeric is distinct from (p_runtime->>'realized')::numeric
     or (p_snapshot->>'trade_count')::integer is distinct from (p_runtime->>'trades')::integer
     or (p_snapshot->>'win_count')::integer is distinct from (p_runtime->>'wins')::integer
     or (p_snapshot->>'loss_count')::integer is distinct from (p_runtime->>'losses')::integer
  then raise exception 'V82_SNAPSHOT_ACCOUNT_MISMATCH'; end if;

  pos=p_runtime->'pos'; old_pos=old_state.runtime->'pos';
  for e in select value from jsonb_array_elements(p_events) loop
    kind=e->>'event_kind';
    if kind not in ('BUY','SELL','SHORT_OPEN','SHORT_CLOSE')
       or e->>'occurrence_id' is null
       or e->>'transition_id' is distinct from ('v82-'||lower(kind)||'-'||(e->>'occurrence_id'))
       or e#>>'{metadata,server_v8}' is distinct from 'true'
       or exists (select 1 from unnest(array['quantity','notional','fees','realized_pnl','cash_after']) k where jsonb_typeof(e->k) is distinct from 'number')
    then raise exception 'INVALID_V82_EVENT'; end if;

    if kind in ('BUY','SHORT_OPEN') then
      opens=opens+1;
      if latest.event_kind<>'START' or latest.config->>'execution_mode' is distinct from 'SHADOW_PAPER' then raise exception 'V82_ENTRY_DISABLED'; end if;
      if (p_snapshot->>'observed_at')::timestamptz<clock_timestamp()-interval '30 seconds'
         or (p_snapshot->>'observed_at')::timestamptz>clock_timestamp()+interval '2 seconds'
      then raise exception 'V82_STALE_ENTRY'; end if;
      if jsonb_typeof(pos) is distinct from 'object' then raise exception 'V82_POSITION_MISSING'; end if;
      margin=(pos->>'margin')::numeric;
      delta_cash=delta_cash-margin-(e->>'fees')::numeric;
    else
      closes=closes+1;
      if old_pos='null'::jsonb then raise exception 'V82_CLOSE_WITHOUT_POSITION'; end if;
      side=old_pos->>'side';
      fee_close=(e->>'fees')::numeric-(old_pos->>'fees_open')::numeric;
      if fee_close<0 then raise exception 'V82_NEGATIVE_CLOSE_FEE'; end if;
      gross=case when side='LONG'
        then ((e->>'exit_price')::numeric-(old_pos->>'entry')::numeric)*(old_pos->>'qty')::numeric
        else ((old_pos->>'entry')::numeric-(e->>'exit_price')::numeric)*(old_pos->>'qty')::numeric end;
      funding=coalesce((e->>'funding_cashflow')::numeric,0);
      funding_sum=0;
      previous_funding_time=extract(epoch from (old_pos->>'opened_at')::timestamptz)*1000;
      if jsonb_typeof(coalesce(e#>'{metadata,funding_settlements}','[]'::jsonb)) is distinct from 'array' then raise exception 'V83_FUNDING_EVIDENCE'; end if;
      for settlement in select value from jsonb_array_elements(coalesce(e#>'{metadata,funding_settlements}','[]'::jsonb)) loop
        funding_time=(settlement->>'fundingTime')::numeric;
        if funding_time is null or funding_time<=previous_funding_time
           or funding_time>coalesce((e#>>'{metadata,resolution,eventAt}')::numeric,-1)
           or coalesce((settlement->>'markPrice')::numeric,0)<=0
           or jsonb_typeof(settlement->'fundingRate') is distinct from 'number'
        then raise exception 'V83_FUNDING_EVIDENCE'; end if;
        previous_funding_time=funding_time;
        funding_sum=funding_sum+(case when side='LONG' then -1 else 1 end)*(old_pos->>'qty')::numeric*(settlement->>'markPrice')::numeric*(settlement->>'fundingRate')::numeric;
      end loop;
      if abs(funding-funding_sum)>0.0000001 then raise exception 'V83_FUNDING_SUM_MISMATCH'; end if;
      expected_net=gross-(e->>'fees')::numeric+funding;
      if abs((e->>'realized_pnl')::numeric-expected_net)>0.0000001 then raise exception 'V82_EXIT_PNL_MISMATCH'; end if;
      delta_cash=delta_cash+(old_pos->>'margin')::numeric+gross-fee_close+funding;
      delta_realized=delta_realized+expected_net;
      if expected_net>0 then wins=wins+1; elsif expected_net<0 then losses=losses+1; end if;
    end if;
  end loop;

  if (old_pos='null'::jsonb and pos<>'null'::jsonb and opens<>1)
     or (old_pos<>'null'::jsonb and pos='null'::jsonb and closes<>1)
     or (opens=1 and old_pos<>'null'::jsonb)
     or (closes=1 and old_pos='null'::jsonb)
     or (closes=1 and pos<>'null'::jsonb)
     or (opens=1 and pos='null'::jsonb)
  then raise exception 'V82_ILLEGAL_POSITION_TRANSITION'; end if;

  if old_pos<>'null'::jsonb and pos<>'null'::jsonb
     and (old_pos-array['checked_until','market_price','funding_accrued','funding_settlements']) is distinct from (pos-array['checked_until','market_price','funding_accrued','funding_settlements'])
  then raise exception 'V82_OPEN_PLAN_IMMUTABLE'; end if;

  if opens=1 then
    if jsonb_typeof(pos) is distinct from 'object' or exists (
      select 1 from unnest(array['entry','stop','target','qty','notional','fees_open','margin','leverage','actual_fraction','gross_fraction']) k
      where jsonb_typeof(pos->k) is distinct from 'number'
    ) then raise exception 'V82_INVALID_POSITION_FIELDS'; end if;
    side=pos->>'side'; lev=(pos->>'leverage')::integer; margin=(pos->>'margin')::numeric;
    if side not in ('LONG','SHORT') or pos->>'venue' is distinct from 'SHADOW_PERP' or lev not in (1,2)
       or (pos->>'qty')::numeric<=0 or (pos->>'fees_open')::numeric<0 or margin<=0
       or abs((pos->>'notional')::numeric-(pos->>'qty')::numeric*(pos->>'entry')::numeric)>0.0000001
       or abs((pos->>'notional')::numeric-margin*lev)>0.0000001
       or (side='LONG' and not ((pos->>'stop')::numeric<(pos->>'entry')::numeric and (pos->>'entry')::numeric<(pos->>'target')::numeric))
       or (side='SHORT' and not ((pos->>'target')::numeric<(pos->>'entry')::numeric and (pos->>'entry')::numeric<(pos->>'stop')::numeric))
    then raise exception 'V82_POSITION_GEOMETRY_OR_MARGIN'; end if;
  end if;

  if abs((p_runtime->>'cash')::numeric-((old_state.runtime->>'cash')::numeric+delta_cash))>0.0000001
     or abs((p_runtime->>'realized')::numeric-((old_state.runtime->>'realized')::numeric+delta_realized))>0.0000001
     or (p_runtime->>'trades')::integer<>(old_state.runtime->>'trades')::integer+closes
     or (p_runtime->>'wins')::integer<>(old_state.runtime->>'wins')::integer+wins
     or (p_runtime->>'losses')::integer<>(old_state.runtime->>'losses')::integer+losses
  then raise exception 'V82_ACCOUNTING_MISMATCH'; end if;
  if (p_runtime->>'cash')::numeric<0 then raise exception 'V82_NEGATIVE_CASH'; end if;

  if p_decision is not null and p_decision<>'null'::jsonb then
    d=jsonb_populate_record(null::public.brian_dip_v8_decisions,p_decision);
    if d.session_id<>p_session_id or d.policy_version<>'dip-v8-dual-20260908.2'
       or d.metric_version<>'target-before-invalidation-v8.2' or d.venue<>'SHADOW_PERP'
    then raise exception 'INVALID_V82_DECISION'; end if;
    insert into public.brian_dip_v8_decisions(occurrence_id,session_id,episode_id,symbol,decision_at,due_at,setup,direction,regime,venue,entry_price,target_price,invalidation_price,raw_conviction,calibrated_probability,calibration_samples,policy_version,metric_version,evidence,shadow_only,live_execution)
    values(d.occurrence_id,d.session_id,d.episode_id,d.symbol,d.decision_at,d.due_at,d.setup,d.direction,d.regime,d.venue,d.entry_price,d.target_price,d.invalidation_price,d.raw_conviction,d.calibrated_probability,d.calibration_samples,d.policy_version,d.metric_version,d.evidence,true,false);
  end if;

  for e in select value from jsonb_array_elements(p_events) loop
    kind=e->>'event_kind';
    if kind in ('BUY','SHORT_OPEN') then
      select * into d from public.brian_dip_v8_decisions where occurrence_id=e->>'occurrence_id';
      if not found or d.session_id<>p_session_id or d.venue<>'SHADOW_PERP'
         or (kind='BUY' and (d.direction<>'UP' or pos->>'side'<>'LONG'))
         or (kind='SHORT_OPEN' and (d.direction<>'DOWN' or pos->>'side'<>'SHORT'))
         or (e->>'notional')::numeric is distinct from (pos->>'notional')::numeric
         or (e->>'quantity')::numeric is distinct from (pos->>'qty')::numeric
         or (e->>'fees')::numeric is distinct from (pos->>'fees_open')::numeric
         or e->>'occurrence_id' is distinct from pos->>'thesis_id'
      then raise exception 'V82_ENTRY_EVIDENCE_MISMATCH'; end if;
      if d.evidence->>'decision_revision' is distinct from 'dip-v8-dual-integrity-20260908.5'
         or e->>'episode_id' is distinct from d.episode_id
         or pos->>'episode_id' is distinct from d.episode_id
         or (pos->>'entry')::numeric is distinct from d.entry_price
         or (pos->>'target')::numeric is distinct from d.target_price
         or (pos->>'stop')::numeric is distinct from d.invalidation_price
         or exists(select 1 from jsonb_array_elements_text(d.evidence->'veto') v where v<>'CALIBRATING')
      then raise exception 'V83_ENTRY_PLAN_OR_REVISION_MISMATCH'; end if;
      -- Held under the same transaction/control lock as the atomic ledger insert.
      -- Historical duplicates are retained; a new entry cannot repeat any recorded episode.
      if exists(select 1 from public.brian_dip_v8_ledger where episode_id=d.episode_id and event_kind in ('BUY','SHORT_OPEN'))
      then raise exception 'V83_EPISODE_ALREADY_TRADED'; end if;
      margin_cap=case when d.calibration_samples<40 or d.calibrated_probability is null then 0.08 else 0.20 end;
      if (pos->>'notional')::numeric>least(latest.trade_notional,(old_state.runtime->>'cash')::numeric*margin_cap)+0.0000001
      then raise exception 'V83_NOTIONAL_CAP'; end if;
      if (pos->>'margin')::numeric>(old_state.runtime->>'cash')::numeric*margin_cap+0.0000001 then raise exception 'V82_MARGIN_CAP'; end if;
      if abs((pos->>'entry')::numeric-(pos->>'stop')::numeric)*(pos->>'qty')::numeric + (pos->>'notional')::numeric*(2*coalesce((pos->>'fee_bps')::numeric,0)+2*coalesce((pos->>'slippage_bps')::numeric,0)+coalesce((pos->>'spread_bps')::numeric,0))/10000 > (old_state.runtime->>'cash')::numeric*0.005+0.0000001 then raise exception 'V83_RISK_BUDGET'; end if;
      lev=(pos->>'leverage')::integer;
      if lev=2 then
        amb_rate=coalesce((d.evidence#>>'{calibration,ambiguous}')::numeric,0)/greatest(1,d.calibration_samples+coalesce((d.evidence#>>'{calibration,ambiguous}')::numeric,0));
        if d.calibration_samples<40 or coalesce(d.calibrated_probability,0)<0.68
           or coalesce((d.evidence#>>'{calibration,lower}')::numeric,0)<0.55
           or coalesce(d.raw_conviction,0)<0.72
           or coalesce((d.evidence->>'rr')::numeric,0)<2.5
           or abs(coalesce((d.evidence#>>'{flow,ofi}')::numeric,0))<0.25
           or abs(d.target_price-d.entry_price)/d.entry_price*10000 < 3.5*coalesce((d.evidence->>'cost_bps')::numeric,999999)
           or amb_rate>0.10
        then raise exception 'V82_2X_EDGE_GATE_FAILED'; end if;
      end if;
    else
      if e->>'occurrence_id' is distinct from old_pos->>'thesis_id'
         or (e->>'quantity')::numeric is distinct from (old_pos->>'qty')::numeric
         or (e->>'exit_price')::numeric is null
         or (kind='SELL' and old_pos->>'side'<>'LONG')
         or (kind='SHORT_CLOSE' and old_pos->>'side'<>'SHORT')
      then raise exception 'V82_EXIT_ACCOUNT_MISMATCH'; end if;
    end if;

    insert into public.brian_dip_v8_ledger(transition_id,session_id,occurrence_id,episode_id,event_kind,payload,account_after)
    values(e->>'transition_id',p_session_id,e->>'occurrence_id',e->>'episode_id',kind,e,jsonb_build_object('cash',p_runtime->'cash','realized',p_runtime->'realized','trades',p_runtime->'trades','wins',p_runtime->'wins','losses',p_runtime->'losses','pos',p_runtime->'pos'));
    insert into public.brian_dip_events(event_id,session_id,observed_at,event_kind,symbol,price,entry_price,exit_price,quantity,notional,fees,realized_pnl,cash_after,equity_after,metadata,shadow_only,live_execution)
    values(e->>'transition_id',p_session_id,clock_timestamp(),kind,'ETHUSDT',(e->>'price')::numeric,(e->>'entry_price')::numeric,(e->>'exit_price')::numeric,(e->>'quantity')::numeric,(e->>'notional')::numeric,(e->>'fees')::numeric,(e->>'realized_pnl')::numeric,(e->>'cash_after')::numeric,(e->>'equity_after')::numeric,e->'metadata',true,false);
  end loop;

  if p_runtime->>'lastSnapshotHour' is distinct from old_state.runtime->>'lastSnapshotHour' then
    insert into public.brian_dip_snapshots(snapshot_id,session_id,observed_at,cash,equity,realized_pnl,unrealized_pnl,trade_count,win_count,loss_count,state,shadow_only,live_execution)
    values(p_snapshot->>'snapshot_id',p_session_id,(p_snapshot->>'observed_at')::timestamptz,(p_snapshot->>'cash')::numeric,(p_snapshot->>'equity')::numeric,(p_snapshot->>'realized_pnl')::numeric,(p_snapshot->>'unrealized_pnl')::numeric,(p_snapshot->>'trade_count')::integer,(p_snapshot->>'win_count')::integer,(p_snapshot->>'loss_count')::integer,p_snapshot->'state',true,false);
  end if;
  if lease_row.lease_until<=clock_timestamp() then raise exception 'V82_LEASE_EXPIRED_DURING_COMMIT'; end if;
  update public.brian_dip_v8_runtime set runtime=p_runtime,snapshot=p_snapshot,state_version=state_version+1,last_commit_id=p_commit_id,last_commit_hash=payload_hash,updated_at=clock_timestamp() where session_id=p_session_id;
  return jsonb_build_object('status','COMMITTED','state_version',p_expected_version+1);
end $function$
;

revoke all on function public.brian_dip_v82_commit(text,bigint,text,text,jsonb,jsonb,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.brian_dip_v82_commit(text,bigint,text,text,jsonb,jsonb,jsonb,jsonb) to service_role;
revoke all on function public.brian_dip_v82_calibration(text,text,text) from public,anon,authenticated;
grant execute on function public.brian_dip_v82_calibration(text,text,text) to service_role;
