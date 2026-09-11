-- Brian DIP V8.4.1 Authority release foundation.
-- SHADOW ONLY. Additive release registration; existing V8.4 evidence/history is preserved.
-- No cron is created here. New release remains PREPARED until a separately reviewed seal.

insert into public.brian_dip_v84_releases(release_id,status,logic_hash,strategy_manifest_hash,calibration_family_id,db_contract_version,manifest)
values(
  'dip-v841-brian-authority-20260911.1','PREPARED','UNSEALED_GITHUB_ONLY',
  '1b9bc41c4edd2f0b9896be316843bde4cd27b67793d4d3912fb3450fe47ee7f1',
  'dip-v841-brian-authority-family-1','brian-dip-v84-db-2-authority',
  '{"symbol":"ETHUSDT","shadow_only":true,"live_execution":false,"browser_execution":false,"server_authoritative":true,"max_shadow_leverage":1,"decision_authority":"BRIAN","target_policy":"BRIAN_STRUCTURAL_LADDER","fixed_notional_fraction_cap":false,"live_l2":false}'::jsonb
) on conflict (release_id) do nothing;

-- Lease storage is shared, but the lease key maps to an explicit release so Package-1 and Authority cannot impersonate each other.
create or replace function public.brian_dip_v84_acquire_lease(p_lease_key text,p_owner_id text,p_lease_seconds integer)
returns table(acquired boolean,lease_generation bigint)
language plpgsql set search_path='pg_catalog','public' as $$
declare r public.brian_dip_v84_worker_leases%rowtype; now_at timestamptz:=clock_timestamp(); wanted_release text;
begin
  if coalesce(length(p_lease_key),0)=0 or coalesce(length(p_owner_id),0)=0 or p_lease_seconds not between 5 and 120 then raise exception 'V84_INVALID_LEASE_REQUEST'; end if;
  wanted_release:=case p_lease_key
    when 'brian-dip-v84-worker' then 'dip-v84-package1-evidence-20260910.1'
    when 'brian-dip-v84-authority-worker' then 'dip-v841-brian-authority-20260911.1'
    when 'brian-dip-v84-foresight' then coalesce((select config->>'release_id' from public.brian_dip_v84_session_events order by requested_at desc,event_id desc limit 1),'dip-v84-package1-evidence-20260910.1')
    else null end;
  if wanted_release is null then raise exception 'V84_UNKNOWN_LEASE_KEY'; end if;
  insert into public.brian_dip_v84_worker_leases(lease_key) values(p_lease_key) on conflict do nothing;
  select * into r from public.brian_dip_v84_worker_leases where lease_key=p_lease_key for update;
  if r.owner_id is null or r.expires_at is null or r.expires_at<=now_at then
    update public.brian_dip_v84_worker_leases set owner_id=p_owner_id,lease_generation=r.lease_generation+1,acquired_at=now_at,heartbeat_at=now_at,expires_at=now_at+make_interval(secs=>p_lease_seconds),release_id=wanted_release where lease_key=p_lease_key returning * into r;
    return query select true,r.lease_generation;return;
  end if;
  return query select false,r.lease_generation;
end $$;

create or replace function public.brian_dip_v84_renew_lease(p_lease_key text,p_owner_id text,p_lease_generation bigint,p_lease_seconds integer)
returns boolean language plpgsql set search_path='pg_catalog','public' as $$
declare n integer; wanted_release text;
begin
  if p_lease_seconds not between 5 and 120 then return false; end if;
  wanted_release:=case p_lease_key
    when 'brian-dip-v84-worker' then 'dip-v84-package1-evidence-20260910.1'
    when 'brian-dip-v84-authority-worker' then 'dip-v841-brian-authority-20260911.1'
    when 'brian-dip-v84-foresight' then coalesce((select config->>'release_id' from public.brian_dip_v84_session_events order by requested_at desc,event_id desc limit 1),'dip-v84-package1-evidence-20260910.1')
    else null end;
  if wanted_release is null then return false; end if;
  update public.brian_dip_v84_worker_leases set heartbeat_at=clock_timestamp(),expires_at=clock_timestamp()+make_interval(secs=>p_lease_seconds)
  where lease_key=p_lease_key and owner_id=p_owner_id and lease_generation=p_lease_generation and expires_at>clock_timestamp() and release_id=wanted_release;
  get diagnostics n=row_count;return n=1;
end $$;

-- START accepts either sealed Package-1 or sealed Authority, but always enforces the common SHADOW contract and exact release identity.
create or replace function public.brian_dip_v84_initialize_session() returns trigger
language plpgsql set search_path='pg_catalog','public' as $$
declare r jsonb; rel public.brian_dip_v84_releases%rowtype; engine text;
begin
  if new.event_kind<>'START' then return new; end if;
  select * into rel from public.brian_dip_v84_releases where release_id=new.config->>'release_id';
  if not found or rel.status<>'SEALED' then raise exception 'V84_RELEASE_NOT_SEALED'; end if;
  engine:=new.config->>'engine_version';
  if new.config->>'release_id' is distinct from rel.release_id
     or new.config->>'logic_hash' is distinct from rel.logic_hash
     or new.config->>'strategy_manifest_hash' is distinct from rel.strategy_manifest_hash
     or new.config->>'calibration_family_id' is distinct from rel.calibration_family_id
     or new.config->>'db_contract_version' is distinct from rel.db_contract_version
     or new.config->'symbols' is distinct from '["ETHUSDT"]'::jsonb
     or new.config->>'shadow_only' is distinct from 'true'
     or new.config->>'live_execution' is distinct from 'false'
     or new.config->>'browser_execution' is distinct from 'false'
     or new.config->>'server_authoritative' is distinct from 'true'
     or new.config->>'allow_shadow_short' is distinct from 'true'
     or coalesce((new.config->>'max_shadow_leverage')::integer,0)<>1
     or coalesce(new.config->>'execution_mode','') not in ('OBSERVE','SHADOW_PAPER')
     or rel.manifest->>'shadow_only' is distinct from 'true'
     or rel.manifest->>'live_execution' is distinct from 'false'
     or rel.manifest->>'browser_execution' is distinct from 'false'
     or coalesce((rel.manifest->>'max_shadow_leverage')::integer,0)<>1
  then raise exception 'V84_INVALID_SESSION_CONTRACT'; end if;
  if rel.release_id='dip-v84-package1-evidence-20260910.1' then
    if engine is distinct from 'brian-dip-v84' or new.config->>'policy_version' is distinct from 'dip-v84-l1-evidence-20260910.1' or new.config->>'decision_revision' is distinct from 'dip-v84-l1-occurrence-20260910.1' or new.config->>'metric_version' is distinct from 'target-before-invalidation-v84.1' then raise exception 'V84_PACKAGE1_VERSION_CONTRACT'; end if;
  elsif rel.release_id='dip-v841-brian-authority-20260911.1' then
    if engine is distinct from 'brian-dip-v841-authority' or new.config->>'policy_version' is distinct from 'dip-v841-brian-authority-20260911.1' or new.config->>'decision_revision' is distinct from 'dip-v841-authority-occurrence-20260911.1' or new.config->>'metric_version' is distinct from 'target-before-invalidation-v841.1' or new.config->>'decision_authority' is distinct from 'BRIAN' or new.config->>'worker_lease_key' is distinct from 'brian-dip-v84-authority-worker' then raise exception 'V841_AUTHORITY_VERSION_CONTRACT'; end if;
  else raise exception 'V84_UNKNOWN_RELEASE'; end if;
  if exists(select 1 from public.brian_dip_v84_runtime where session_id=new.session_id) then return new; end if;
  r=jsonb_build_object('start',new.starting_equity,'cash',new.starting_equity,'realized',0,'trades',0,'wins',0,'losses',0,'pos',null,'latestThesis',null,'lastOccurrence',null,'lastSnapshotHour',null,'marketCursor',0,'lastClosedAt',null);
  insert into public.brian_dip_v84_runtime(session_id,runtime) values(new.session_id,r);
  insert into public.brian_dip_v84_ledger(transition_id,session_id,event_kind,release_id,calibration_family_id,payload,account_after)
  values('v84-start-'||new.session_id,new.session_id,'SESSION_START',rel.release_id,rel.calibration_family_id,jsonb_build_object('starting_equity',new.starting_equity,'decision_authority',new.config->>'decision_authority'),r);
  return new;
end $$;

-- Dual-release atomic commit. Strategic policy lives in the worker; this function only enforces identity, SHADOW safety, transitions and accounting.
create or replace function public.brian_dip_v84_commit(
  p_session_id text,p_expected_version bigint,p_owner_id text,p_lease_generation bigint,p_commit_id text,
  p_runtime jsonb,p_snapshot jsonb,p_decision jsonb,p_events jsonb
) returns jsonb
language plpgsql set search_path='pg_catalog','public' as $$
declare
  old_state public.brian_dip_v84_runtime%rowtype; latest public.brian_dip_v84_session_events%rowtype; start_row public.brian_dip_v84_session_events%rowtype; lease_row public.brian_dip_v84_worker_leases%rowtype; rel public.brian_dip_v84_releases%rowtype;
  cfg jsonb; lease_key text; payload_hash text; e jsonb; d public.brian_dip_v84_decisions%rowtype; pos jsonb; old_pos jsonb; kind text; side text; margin numeric; fee_close numeric; gross numeric; funding numeric; expected_net numeric; delta_cash numeric:=0; delta_realized numeric:=0; opens integer:=0; closes integer:=0; wins integer:=0; losses integer:=0; ambiguous_loss boolean;
begin
  if p_commit_id is null or length(p_commit_id)>160 or jsonb_typeof(p_events) is distinct from 'array' or jsonb_array_length(p_events)>1 or octet_length(p_runtime::text)>120000 or octet_length(p_snapshot::text)>180000 or octet_length(coalesce(p_decision,'null'::jsonb)::text)>60000 then raise exception 'V84_INVALID_COMMIT'; end if;
  perform pg_advisory_xact_lock(hashtextextended('brian-dip-v84-control',0));
  select * into old_state from public.brian_dip_v84_runtime where session_id=p_session_id for update;if not found then raise exception 'V84_STATE_MISSING_RECONCILE';end if;
  payload_hash=md5(jsonb_build_array(p_runtime,p_snapshot,p_decision,p_events)::text);
  if old_state.last_commit_id=p_commit_id then if old_state.last_commit_hash<>payload_hash then raise exception 'V84_IDEMPOTENCY_PAYLOAD_CONFLICT';end if;return jsonb_build_object('status','ALREADY_COMMITTED','state_version',old_state.state_version);end if;
  if old_state.state_version<>p_expected_version then raise exception 'V84_STATE_VERSION_CONFLICT';end if;
  select * into latest from public.brian_dip_v84_session_events order by requested_at desc,event_id desc limit 1;if not found or latest.session_id is distinct from p_session_id then raise exception 'V84_STALE_SESSION';end if;
  if latest.event_kind='PAUSE' then select * into start_row from public.brian_dip_v84_session_events where session_id=p_session_id and event_kind='START' order by requested_at,event_id limit 1;if not found then raise exception 'V84_START_MISSING';end if;cfg:=start_row.config;else cfg:=latest.config;end if;
  select * into rel from public.brian_dip_v84_releases where release_id=cfg->>'release_id';
  if not found or rel.status<>'SEALED' or cfg->>'logic_hash' is distinct from rel.logic_hash or cfg->>'strategy_manifest_hash' is distinct from rel.strategy_manifest_hash or cfg->>'calibration_family_id' is distinct from rel.calibration_family_id or cfg->>'db_contract_version' is distinct from rel.db_contract_version or cfg->>'shadow_only' is distinct from 'true' or cfg->>'live_execution' is distinct from 'false' or cfg->>'browser_execution' is distinct from 'false' or cfg->>'server_authoritative' is distinct from 'true' or coalesce((cfg->>'max_shadow_leverage')::integer,0)<>1 then raise exception 'V84_SESSION_CONTRACT_MISMATCH';end if;
  lease_key:=coalesce(cfg->>'worker_lease_key','brian-dip-v84-worker');
  select * into lease_row from public.brian_dip_v84_worker_leases where lease_key=lease_key for share;
  if not found then raise exception 'V84_LEASE_MISSING';end if;
  if lease_row.owner_id is distinct from p_owner_id or lease_row.lease_generation is distinct from p_lease_generation or lease_row.expires_at is null or lease_row.expires_at<=clock_timestamp() or lease_row.release_id is distinct from rel.release_id then raise exception 'V84_LEASE_FENCE_LOST';end if;
  if p_snapshot->>'session_id' is distinct from p_session_id or p_snapshot#>>'{state,serverRuntime,release_id}' is distinct from rel.release_id or p_snapshot#>>'{state,serverRuntime,shadow_only}' is distinct from 'true' or p_snapshot#>>'{state,serverRuntime,live_execution}' is distinct from 'false' then raise exception 'V84_SNAPSHOT_CONTRACT';end if;
  if jsonb_typeof(p_runtime) is distinct from 'object' or not(p_runtime?'pos') or (p_runtime->>'start')::numeric is distinct from (old_state.runtime->>'start')::numeric or (p_runtime->>'marketCursor')::bigint<(old_state.runtime->>'marketCursor')::bigint then raise exception 'V84_RUNTIME_CONTRACT';end if;
  pos=p_runtime->'pos';old_pos=old_state.runtime->'pos';
  for e in select value from jsonb_array_elements(p_events) loop
    kind=e->>'event_kind';if kind not in ('BUY','SELL','SHORT_OPEN','SHORT_CLOSE') or e#>>'{metadata,server_v84}' is distinct from 'true' or e#>>'{metadata,release_id}' is distinct from rel.release_id then raise exception 'V84_INVALID_EVENT';end if;
    if kind in ('BUY','SHORT_OPEN') then
      opens=opens+1;if latest.event_kind<>'START' or cfg->>'execution_mode' is distinct from 'SHADOW_PAPER' then raise exception 'V84_ENTRY_DISABLED';end if;if jsonb_typeof(pos) is distinct from 'object' or (pos->>'leverage')::integer<>1 or pos->>'release_id' is distinct from rel.release_id then raise exception 'V84_POSITION_CONTRACT';end if;margin=(pos->>'margin')::numeric;delta_cash=delta_cash-margin-(e->>'fees')::numeric;
    else
      closes=closes+1;if old_pos='null'::jsonb then raise exception 'V84_CLOSE_WITHOUT_POSITION';end if;side=old_pos->>'side';fee_close=(e->>'fees')::numeric-(old_pos->>'fees_open')::numeric;if fee_close<0 then raise exception 'V84_NEGATIVE_CLOSE_FEE';end if;gross=case when side='LONG' then ((e->>'exit_price')::numeric-(old_pos->>'entry')::numeric)*(old_pos->>'qty')::numeric else ((old_pos->>'entry')::numeric-(e->>'exit_price')::numeric)*(old_pos->>'qty')::numeric end;funding=coalesce((e->>'funding_cashflow')::numeric,0);expected_net=gross-(e->>'fees')::numeric+funding;if abs((e->>'realized_pnl')::numeric-expected_net)>0.0000001 then raise exception 'V84_EXIT_PNL_MISMATCH';end if;delta_cash=delta_cash+(old_pos->>'margin')::numeric+gross-fee_close+funding;delta_realized=delta_realized+expected_net;if expected_net>0 then wins=wins+1;elsif expected_net<0 then losses=losses+1;end if;
    end if;
  end loop;
  if opens>1 or closes>1 or (old_pos='null'::jsonb and pos<>'null'::jsonb and opens<>1) or (old_pos<>'null'::jsonb and pos='null'::jsonb and closes<>1) or (opens=1 and old_pos<>'null'::jsonb) or (closes=1 and old_pos='null'::jsonb) then raise exception 'V84_ILLEGAL_POSITION_TRANSITION';end if;
  if old_pos<>'null'::jsonb and pos<>'null'::jsonb and (old_pos-array['checked_until','market_price','funding_accrued','funding_settlements']) is distinct from (pos-array['checked_until','market_price','funding_accrued','funding_settlements']) then raise exception 'V84_OPEN_PLAN_IMMUTABLE';end if;
  if abs((p_runtime->>'cash')::numeric-((old_state.runtime->>'cash')::numeric+delta_cash))>0.0000001 or abs((p_runtime->>'realized')::numeric-((old_state.runtime->>'realized')::numeric+delta_realized))>0.0000001 or (p_runtime->>'trades')::integer<>(old_state.runtime->>'trades')::integer+closes or (p_runtime->>'wins')::integer<>(old_state.runtime->>'wins')::integer+wins or (p_runtime->>'losses')::integer<>(old_state.runtime->>'losses')::integer+losses or (p_runtime->>'cash')::numeric<0 then raise exception 'V84_ACCOUNTING_MISMATCH';end if;
  if p_decision is not null and p_decision<>'null'::jsonb then
    d=jsonb_populate_record(null::public.brian_dip_v84_decisions,p_decision);if d.session_id<>p_session_id or d.release_id<>rel.release_id or d.logic_hash<>rel.logic_hash or d.strategy_manifest_hash<>rel.strategy_manifest_hash or d.calibration_family_id<>rel.calibration_family_id or d.db_contract_version<>rel.db_contract_version or d.shadow_only is distinct from true or d.live_execution is distinct from false then raise exception 'V84_INVALID_DECISION';end if;
    insert into public.brian_dip_v84_decisions select d.*;
  end if;
  for e in select value from jsonb_array_elements(p_events) loop
    insert into public.brian_dip_v84_ledger(transition_id,session_id,occurrence_id,episode_id,event_kind,release_id,calibration_family_id,payload,account_after)
    values(e->>'transition_id',p_session_id,e->>'occurrence_id',e->>'episode_id',e->>'event_kind',rel.release_id,rel.calibration_family_id,e,p_runtime);
    if e->>'event_kind' in ('SELL','SHORT_CLOSE') then
      ambiguous_loss=coalesce((e#>>'{metadata,execution_ambiguous_loss}')::boolean,false);
      insert into public.brian_dip_v84_execution_outcomes(outcome_id,session_id,occurrence_id,episode_id,setup,direction,regime,release_id,calibration_family_id,closed_at,resolution_reason,is_win,ambiguous_conservative_loss,realized_pnl)
      values('outcome-'||(e->>'transition_id'),p_session_id,e->>'occurrence_id',e->>'episode_id',old_pos->>'setup',case when old_pos->>'side'='LONG' then 'UP' else 'DOWN' end,old_pos->>'regime',rel.release_id,rel.calibration_family_id,clock_timestamp(),coalesce(e#>>'{metadata,exit_reason}','UNKNOWN'),case when ambiguous_loss then false else (e->>'realized_pnl')::numeric>0 end,ambiguous_loss,(e->>'realized_pnl')::numeric);
    end if;
  end loop;
  update public.brian_dip_v84_runtime set runtime=p_runtime,snapshot=p_snapshot,state_version=state_version+1,last_commit_id=p_commit_id,last_commit_hash=payload_hash,updated_at=clock_timestamp() where session_id=p_session_id;
  return jsonb_build_object('status','COMMITTED','state_version',old_state.state_version+1,'release_id',rel.release_id);
end $$;

revoke all on function public.brian_dip_v84_acquire_lease(text,text,integer) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_acquire_lease(text,text,integer) to service_role;
revoke all on function public.brian_dip_v84_renew_lease(text,text,bigint,integer) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_renew_lease(text,text,bigint,integer) to service_role;
revoke all on function public.brian_dip_v84_initialize_session() from public,anon,authenticated;
grant execute on function public.brian_dip_v84_initialize_session() to service_role;
revoke all on function public.brian_dip_v84_commit(text,bigint,text,bigint,text,jsonb,jsonb,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_commit(text,bigint,text,bigint,text,jsonb,jsonb,jsonb,jsonb) to service_role;
