-- Brian DIP V8.2 dual-direction SHADOW rollout.
-- LONG + SHORT paper lifecycle, dual-policy calibration and guarded 2x accounting.
-- No live exchange execution is enabled by this migration.

alter table public.brian_dip_v8_decisions drop constraint if exists brian_dip_v8_decisions_venue_check;
alter table public.brian_dip_v8_decisions
  add constraint brian_dip_v8_decisions_venue_check check (venue in ('SPOT','SHADOW_PERP'));

alter table public.brian_dip_v8_ledger drop constraint if exists brian_dip_v8_ledger_event_kind_check;
alter table public.brian_dip_v8_ledger
  add constraint brian_dip_v8_ledger_event_kind_check
  check (event_kind in ('SESSION_START','BUY','SELL','SHORT_OPEN','SHORT_CLOSE'));

create or replace function public.brian_dip_v82_calibration(p_setup text,p_direction text,p_regime text)
returns table(hit boolean)
language sql stable
set search_path to 'pg_catalog','public'
as $$
  select firsts.hit
  from (
    select distinct on (episode_id) episode_id,hit,decision_at,resolved_at
    from public.brian_dip_v8_decisions
    where setup=p_setup and direction=p_direction and regime=p_regime
      and venue='SHADOW_PERP'
      and policy_version='dip-v8-dual-20260908.2'
      and metric_version='target-before-invalidation-v8.2'
    order by episode_id,decision_at,occurrence_id
  ) firsts
  where resolved_at is not null
  order by resolved_at desc
  limit 500
$$;

create or replace function public.brian_dip_v8_calibration(p_setup text,p_direction text,p_regime text)
returns table(hit boolean)
language sql stable
set search_path to 'pg_catalog','public'
as $$
  with active_policy as (
    select coalesce(config->>'policy_version','dip-v8-integrity-20260908.1') as pv
    from public.brian_dip_session_events
    order by requested_at desc,event_id desc
    limit 1
  ), cfg as (
    select pv,
      case when pv='dip-v8-dual-20260908.2' then 'SHADOW_PERP' else 'SPOT' end as venue,
      case when pv='dip-v8-dual-20260908.2' then 'target-before-invalidation-v8.2' else 'target-before-invalidation-v8.1' end as metric
    from active_policy
  )
  select firsts.hit
  from (
    select distinct on (d.episode_id) d.episode_id,d.hit,d.decision_at,d.resolved_at
    from public.brian_dip_v8_decisions d,cfg
    where d.setup=p_setup and d.direction=p_direction and d.regime=p_regime
      and d.venue=cfg.venue and d.policy_version=cfg.pv and d.metric_version=cfg.metric
    order by d.episode_id,d.decision_at,d.occurrence_id
  ) firsts
  where firsts.resolved_at is not null
  order by firsts.resolved_at desc
  limit 500
$$;

create or replace function public.brian_dip_v8_initialize_session()
returns trigger
language plpgsql
set search_path to 'pg_catalog','public'
as $$
declare r jsonb; pv text;
begin
  if new.event_kind='START' and exists(
    select 1 from public.brian_dip_v8_runtime
    where session_id<>new.session_id and runtime->'pos'<>'null'::jsonb
  ) then raise exception 'OPEN_V8_POSITION_BLOCKS_RESTART'; end if;
  if new.event_kind<>'START' then return new; end if;
  pv:=new.config->>'policy_version';
  if pv not in ('dip-v8-integrity-20260908.1','dip-v8-dual-20260908.2') or pv is null then return new; end if;
  if new.config->'symbols' is distinct from '["ETHUSDT"]'::jsonb
     or new.config->>'shadow_only' is distinct from 'true'
     or new.config->>'live_execution' is distinct from 'false'
     or new.config->>'browser_execution' is distinct from 'false'
     or new.config->>'server_authoritative' is distinct from 'true'
     or coalesce(new.config->>'execution_mode','') not in ('OBSERVE','SHADOW_PAPER')
  then raise exception 'INVALID_V8_SHADOW_SESSION'; end if;
  if pv='dip-v8-integrity-20260908.1' then
    if new.config->>'engine_version' is distinct from 'brian-dip-chart-reader-v8'
       or new.config->>'allow_shadow_short' is distinct from 'false'
       or coalesce((new.config->>'max_shadow_leverage')::numeric,1)<>1
    then raise exception 'INVALID_V8_LEGACY_SESSION'; end if;
  else
    if new.config->>'engine_version' is distinct from 'brian-dip-chart-reader-v8-dual'
       or new.config->>'allow_shadow_short' is distinct from 'true'
       or coalesce((new.config->>'max_shadow_leverage')::numeric,0)<>2
    then raise exception 'INVALID_V82_DUAL_SESSION'; end if;
  end if;
  if exists(select 1 from public.brian_dip_v8_runtime where session_id=new.session_id) then return new; end if;
  if exists(select 1 from public.brian_dip_session_events where session_id=new.session_id) then raise exception 'V8_STATE_MISSING_RECONCILE'; end if;
  r=jsonb_build_object('start',new.starting_equity,'cash',new.starting_equity,'realized',0,'trades',0,'wins',0,'losses',0,'pos',null,'lastLock',null,'latestThesis',null,'lastOccurrence',null,'lastSnapshotHour',null,'marketCursor',0,'lastClosedAt',null);
  insert into public.brian_dip_v8_runtime(session_id,runtime) values(new.session_id,r);
  insert into public.brian_dip_v8_ledger(transition_id,session_id,event_kind,payload,account_after)
  values('v8-start-'||new.session_id,new.session_id,'SESSION_START',jsonb_build_object('starting_equity',new.starting_equity,'policy_version',pv),r);
  return new;
end $$;

create or replace function public.brian_dip_v82_commit(
  p_session_id text,p_expected_version bigint,p_owner_token text,p_commit_id text,
  p_runtime jsonb,p_snapshot jsonb,p_decision jsonb,p_events jsonb
) returns jsonb
language plpgsql
set search_path to 'pg_catalog','public'
as $$
declare
  old_state public.brian_dip_v8_runtime%rowtype;
  latest public.brian_dip_session_events%rowtype;
  lease_row public.brian_collector_leases%rowtype;
  payload_hash text; e jsonb; d public.brian_dip_v8_decisions%rowtype;
  opens integer:=0; closes integer:=0; wins integer:=0; losses integer:=0;
  delta_cash numeric:=0; delta_realized numeric:=0;
  pos jsonb; old_pos jsonb; side text; kind text; lev integer; margin numeric;
  gross numeric; fee_close numeric; expected_net numeric; margin_cap numeric; amb_rate numeric;
begin
  if p_commit_id is null or length(p_commit_id)>160
     or jsonb_typeof(p_events) is distinct from 'array' or jsonb_array_length(p_events)>1
     or octet_length(p_runtime::text)>100000 or octet_length(p_snapshot::text)>150000
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
  if latest.session_id is distinct from p_session_id then raise exception 'STALE_V82_SESSION'; end if;
  if latest.config->>'policy_version' is distinct from 'dip-v8-dual-20260908.2'
     or latest.config->>'engine_version' is distinct from 'brian-dip-chart-reader-v8-dual'
     or latest.config->>'shadow_only' is distinct from 'true'
     or latest.config->>'live_execution' is distinct from 'false'
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
     or (p_runtime->>'marketCursor')::bigint<(old_state.runtime->>'marketCursor')::bigint
  then raise exception 'V82_STATE_RESET_OR_TIME_REGRESSION'; end if;
  if jsonb_typeof(p_runtime) is distinct from 'object' or not(p_runtime?'pos') or exists(
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
       or exists(select 1 from unnest(array['quantity','notional','fees','realized_pnl','cash_after']) k where jsonb_typeof(e->k) is distinct from 'number')
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
      expected_net=gross-(e->>'fees')::numeric;
      if abs((e->>'realized_pnl')::numeric-expected_net)>0.0000001 then raise exception 'V82_EXIT_PNL_MISMATCH'; end if;
      delta_cash=delta_cash+(old_pos->>'margin')::numeric+gross-fee_close;
      delta_realized=delta_realized+expected_net;
      if expected_net>0 then wins=wins+1; elsif expected_net<0 then losses=losses+1; end if;
    end if;
  end loop;
  if (old_pos='null'::jsonb and pos<>'null'::jsonb and opens<>1)
     or (old_pos<>'null'::jsonb and pos='null'::jsonb and closes<>1)
     or (opens=1 and old_pos<>'null'::jsonb) or (closes=1 and old_pos='null'::jsonb)
     or (closes=1 and pos<>'null'::jsonb) or (opens=1 and pos='null'::jsonb)
  then raise exception 'V82_ILLEGAL_POSITION_TRANSITION'; end if;
  if old_pos<>'null'::jsonb and pos<>'null'::jsonb
     and (old_pos-array['checked_until','market_price']) is distinct from (pos-array['checked_until','market_price'])
  then raise exception 'V82_OPEN_PLAN_IMMUTABLE'; end if;
  if opens=1 then
    if jsonb_typeof(pos) is distinct from 'object' or exists(
      select 1 from unnest(array['entry','stop','target','qty','notional','fees_open','margin','leverage','actual_fraction','gross_fraction']) k
      where jsonb_typeof(pos->k) is distinct from 'number'
    ) then raise exception 'V82_INVALID_POSITION_FIELDS'; end if;
    side=pos->>'side'; lev=(pos->>'leverage')::integer; margin=(pos->>'margin')::numeric;
    if side not in ('LONG','SHORT') or pos->>'venue' is distinct from 'SHADOW_PERP' or lev not in (1,2)
       or (pos->>'qty')::numeric<=0 or (pos->>'fees_open')::numeric<0 or margin<=0
       or abs((pos->>'notional')::numeric-(pos->>'qty')::numeric*(pos->>'entry')::numeric)>0.0000001
       or abs((pos->>'notional')::numeric-margin*lev)>0.0000001
       or (side='LONG' and not((pos->>'stop')::numeric<(pos->>'entry')::numeric and (pos->>'entry')::numeric<(pos->>'target')::numeric))
       or (side='SHORT' and not((pos->>'target')::numeric<(pos->>'entry')::numeric and (pos->>'entry')::numeric<(pos->>'stop')::numeric))
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
      margin_cap=case when d.calibration_samples<40 then 0.08 else 0.20 end;
      if (pos->>'margin')::numeric>(old_state.runtime->>'cash')::numeric*margin_cap+0.0000001 then raise exception 'V82_MARGIN_CAP'; end if;
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
  update public.brian_dip_v8_runtime
  set runtime=p_runtime,snapshot=p_snapshot,state_version=state_version+1,last_commit_id=p_commit_id,last_commit_hash=payload_hash,updated_at=clock_timestamp()
  where session_id=p_session_id;
  return jsonb_build_object('status','COMMITTED','state_version',p_expected_version+1);
end $$;

do $$
begin
  if to_regprocedure('public.brian_dip_v8_commit_legacy(text,bigint,text,text,jsonb,jsonb,jsonb,jsonb)') is null then
    alter function public.brian_dip_v8_commit(text,bigint,text,text,jsonb,jsonb,jsonb,jsonb)
      rename to brian_dip_v8_commit_legacy;
  end if;
end $$;

create or replace function public.brian_dip_v8_commit(
  p_session_id text,p_expected_version bigint,p_owner_token text,p_commit_id text,
  p_runtime jsonb,p_snapshot jsonb,p_decision jsonb,p_events jsonb
) returns jsonb
language plpgsql
set search_path to 'pg_catalog','public'
as $$
declare pv text;
begin
  select config->>'policy_version' into pv
  from public.brian_dip_session_events
  order by requested_at desc,event_id desc
  limit 1;
  if pv='dip-v8-dual-20260908.2' then
    return public.brian_dip_v82_commit(p_session_id,p_expected_version,p_owner_token,p_commit_id,p_runtime,p_snapshot,p_decision,p_events);
  end if;
  return public.brian_dip_v8_commit_legacy(p_session_id,p_expected_version,p_owner_token,p_commit_id,p_runtime,p_snapshot,p_decision,p_events);
end $$;
