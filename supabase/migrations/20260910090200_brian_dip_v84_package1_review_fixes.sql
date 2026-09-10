-- V8.4 review fixes: synchronize the PREPARED manifest contract and atomic commit.
-- V8.3 is untouched. This migration does not seal a release and creates no cron.

DO $$
DECLARE n integer;
BEGIN
  UPDATE public.brian_dip_v84_releases
  SET strategy_manifest_hash='9a574469a3df381e56b0993d1f1525ec3a243970b92f8e867b99e07295c06f1e'
  WHERE release_id='dip-v84-package1-evidence-20260910.1'
    AND status='PREPARED'
    AND logic_hash='UNSEALED_GITHUB_ONLY'
    AND strategy_manifest_hash IN (
      '1b4938913e189d04d0b325727e0318b786c4c6fa4734d575f8b278a1d0696fb0',
      '9a574469a3df381e56b0993d1f1525ec3a243970b92f8e867b99e07295c06f1e'
    );
  GET DIAGNOSTICS n=ROW_COUNT;
  IF n<>1 THEN RAISE EXCEPTION 'V84_MANIFEST_REPAIR_CONFLICT'; END IF;
END $$;

CREATE OR REPLACE FUNCTION public.brian_dip_v84_commit(
  p_session_id text,p_expected_version bigint,p_owner_id text,p_lease_generation bigint,p_commit_id text,
  p_runtime jsonb,p_snapshot jsonb,p_decision jsonb,p_events jsonb
) RETURNS jsonb
LANGUAGE plpgsql SET search_path='pg_catalog','public' AS $$
DECLARE
  old_state public.brian_dip_v84_runtime%rowtype; latest public.brian_dip_v84_session_events%rowtype; start_row public.brian_dip_v84_session_events%rowtype; lease_row public.brian_dip_v84_worker_leases%rowtype;
  payload_hash text; e jsonb; d public.brian_dip_v84_decisions%rowtype; pos jsonb; old_pos jsonb; kind text; side text; margin numeric; fee_close numeric; gross numeric; funding numeric; expected_net numeric; delta_cash numeric:=0; delta_realized numeric:=0; opens integer:=0; closes integer:=0; wins integer:=0; losses integer:=0; ambiguous_loss boolean;
BEGIN
  IF p_commit_id IS NULL OR length(p_commit_id)>160 OR jsonb_typeof(p_events) IS DISTINCT FROM 'array' OR jsonb_array_length(p_events)>1 OR octet_length(p_runtime::text)>120000 OR octet_length(p_snapshot::text)>180000 OR octet_length(coalesce(p_decision,'null'::jsonb)::text)>30000 THEN RAISE EXCEPTION 'V84_INVALID_COMMIT'; END IF;
  PERFORM pg_advisory_xact_lock(hashtextextended('brian-dip-v84-control',0));
  SELECT * INTO old_state FROM public.brian_dip_v84_runtime WHERE session_id=p_session_id FOR UPDATE; IF NOT FOUND THEN RAISE EXCEPTION 'V84_STATE_MISSING_RECONCILE'; END IF;
  payload_hash=md5(jsonb_build_array(p_runtime,p_snapshot,p_decision,p_events)::text);
  IF old_state.last_commit_id=p_commit_id THEN IF old_state.last_commit_hash<>payload_hash THEN RAISE EXCEPTION 'V84_IDEMPOTENCY_PAYLOAD_CONFLICT'; END IF; RETURN jsonb_build_object('status','ALREADY_COMMITTED','state_version',old_state.state_version); END IF;
  IF old_state.state_version<>p_expected_version THEN RAISE EXCEPTION 'V84_STATE_VERSION_CONFLICT'; END IF;
  SELECT * INTO latest FROM public.brian_dip_v84_session_events ORDER BY requested_at DESC,event_id DESC LIMIT 1; IF NOT FOUND OR latest.session_id IS DISTINCT FROM p_session_id THEN RAISE EXCEPTION 'V84_STALE_SESSION'; END IF;
  IF latest.event_kind='PAUSE' THEN SELECT * INTO start_row FROM public.brian_dip_v84_session_events WHERE session_id=p_session_id AND event_kind='START' ORDER BY requested_at,event_id LIMIT 1; IF NOT FOUND THEN RAISE EXCEPTION 'V84_START_MISSING'; END IF; latest.config:=start_row.config;latest.starting_equity:=start_row.starting_equity;latest.trade_notional:=start_row.trade_notional; END IF;
  IF latest.config->>'release_id' IS DISTINCT FROM 'dip-v84-package1-evidence-20260910.1'
     OR latest.config->>'strategy_manifest_hash' IS DISTINCT FROM '9a574469a3df381e56b0993d1f1525ec3a243970b92f8e867b99e07295c06f1e'
     OR latest.config->>'calibration_family_id' IS DISTINCT FROM 'dip-v84-package1-evidence-family-1'
     OR latest.config->>'shadow_only' IS DISTINCT FROM 'true'
     OR latest.config->>'live_execution' IS DISTINCT FROM 'false'
     OR latest.config->>'browser_execution' IS DISTINCT FROM 'false'
     OR latest.config->>'server_authoritative' IS DISTINCT FROM 'true'
     OR coalesce((latest.config->>'max_shadow_leverage')::integer,0)<>1
  THEN RAISE EXCEPTION 'V84_SESSION_CONTRACT_MISMATCH'; END IF;
  SELECT * INTO lease_row FROM public.brian_dip_v84_worker_leases WHERE lease_key='brian-dip-v84-worker' FOR SHARE;
  IF lease_row.owner_id IS DISTINCT FROM p_owner_id OR lease_row.lease_generation IS DISTINCT FROM p_lease_generation OR lease_row.expires_at IS NULL OR lease_row.expires_at<=clock_timestamp() OR lease_row.release_id IS DISTINCT FROM 'dip-v84-package1-evidence-20260910.1' THEN RAISE EXCEPTION 'V84_LEASE_FENCE_LOST'; END IF;
  IF p_snapshot->>'session_id' IS DISTINCT FROM p_session_id OR p_snapshot#>>'{state,serverRuntime,release_id}' IS DISTINCT FROM 'dip-v84-package1-evidence-20260910.1' OR p_snapshot#>>'{state,serverRuntime,shadow_only}' IS DISTINCT FROM 'true' OR p_snapshot#>>'{state,serverRuntime,live_execution}' IS DISTINCT FROM 'false' THEN RAISE EXCEPTION 'V84_SNAPSHOT_CONTRACT'; END IF;
  IF jsonb_typeof(p_runtime) IS DISTINCT FROM 'object' OR NOT(p_runtime?'pos') OR (p_runtime->>'start')::numeric IS DISTINCT FROM (old_state.runtime->>'start')::numeric OR (p_runtime->>'marketCursor')::bigint<(old_state.runtime->>'marketCursor')::bigint THEN RAISE EXCEPTION 'V84_RUNTIME_CONTRACT'; END IF;
  pos=p_runtime->'pos';old_pos=old_state.runtime->'pos';
  FOR e IN SELECT value FROM jsonb_array_elements(p_events) LOOP
    kind=e->>'event_kind'; IF kind NOT IN ('BUY','SELL','SHORT_OPEN','SHORT_CLOSE') OR e#>>'{metadata,server_v84}' IS DISTINCT FROM 'true' OR e#>>'{metadata,release_id}' IS DISTINCT FROM 'dip-v84-package1-evidence-20260910.1' THEN RAISE EXCEPTION 'V84_INVALID_EVENT'; END IF;
    IF kind IN ('BUY','SHORT_OPEN') THEN
      opens=opens+1; IF latest.event_kind<>'START' OR latest.config->>'execution_mode' IS DISTINCT FROM 'SHADOW_PAPER' THEN RAISE EXCEPTION 'V84_ENTRY_DISABLED'; END IF; IF jsonb_typeof(pos) IS DISTINCT FROM 'object' OR (pos->>'leverage')::integer<>1 OR pos->>'release_id' IS DISTINCT FROM 'dip-v84-package1-evidence-20260910.1' THEN RAISE EXCEPTION 'V84_POSITION_CONTRACT'; END IF; margin=(pos->>'margin')::numeric;delta_cash=delta_cash-margin-(e->>'fees')::numeric;
    ELSE
      closes=closes+1; IF old_pos='null'::jsonb THEN RAISE EXCEPTION 'V84_CLOSE_WITHOUT_POSITION'; END IF; side=old_pos->>'side';fee_close=(e->>'fees')::numeric-(old_pos->>'fees_open')::numeric; IF fee_close<0 THEN RAISE EXCEPTION 'V84_NEGATIVE_CLOSE_FEE'; END IF; gross=CASE WHEN side='LONG' THEN ((e->>'exit_price')::numeric-(old_pos->>'entry')::numeric)*(old_pos->>'qty')::numeric ELSE ((old_pos->>'entry')::numeric-(e->>'exit_price')::numeric)*(old_pos->>'qty')::numeric END;funding=coalesce((e->>'funding_cashflow')::numeric,0);expected_net=gross-(e->>'fees')::numeric+funding; IF abs((e->>'realized_pnl')::numeric-expected_net)>0.0000001 THEN RAISE EXCEPTION 'V84_EXIT_PNL_MISMATCH'; END IF; delta_cash=delta_cash+(old_pos->>'margin')::numeric+gross-fee_close+funding;delta_realized=delta_realized+expected_net; IF expected_net>0 THEN wins=wins+1; ELSIF expected_net<0 THEN losses=losses+1; END IF;
    END IF;
  END LOOP;
  IF opens>1 OR closes>1 OR (old_pos='null'::jsonb AND pos<>'null'::jsonb AND opens<>1) OR (old_pos<>'null'::jsonb AND pos='null'::jsonb AND closes<>1) OR (opens=1 AND old_pos<>'null'::jsonb) OR (closes=1 AND old_pos='null'::jsonb) THEN RAISE EXCEPTION 'V84_ILLEGAL_POSITION_TRANSITION'; END IF;
  IF old_pos<>'null'::jsonb AND pos<>'null'::jsonb AND (old_pos-array['checked_until','market_price','funding_accrued','funding_settlements']) IS DISTINCT FROM (pos-array['checked_until','market_price','funding_accrued','funding_settlements']) THEN RAISE EXCEPTION 'V84_OPEN_PLAN_IMMUTABLE'; END IF;
  IF abs((p_runtime->>'cash')::numeric-((old_state.runtime->>'cash')::numeric+delta_cash))>0.0000001 OR abs((p_runtime->>'realized')::numeric-((old_state.runtime->>'realized')::numeric+delta_realized))>0.0000001 OR (p_runtime->>'trades')::integer<>(old_state.runtime->>'trades')::integer+closes OR (p_runtime->>'wins')::integer<>(old_state.runtime->>'wins')::integer+wins OR (p_runtime->>'losses')::integer<>(old_state.runtime->>'losses')::integer+losses OR (p_runtime->>'cash')::numeric<0 THEN RAISE EXCEPTION 'V84_ACCOUNTING_MISMATCH'; END IF;
  IF p_decision IS NOT NULL AND p_decision<>'null'::jsonb THEN
    d=jsonb_populate_record(null::public.brian_dip_v84_decisions,p_decision); IF d.session_id<>p_session_id OR d.release_id<>'dip-v84-package1-evidence-20260910.1' OR d.strategy_manifest_hash<>'9a574469a3df381e56b0993d1f1525ec3a243970b92f8e867b99e07295c06f1e' OR d.calibration_family_id<>'dip-v84-package1-evidence-family-1' OR d.shadow_only IS DISTINCT FROM true OR d.live_execution IS DISTINCT FROM false THEN RAISE EXCEPTION 'V84_INVALID_DECISION'; END IF;
    INSERT INTO public.brian_dip_v84_decisions SELECT d.*;
  END IF;
  FOR e IN SELECT value FROM jsonb_array_elements(p_events) LOOP
    INSERT INTO public.brian_dip_v84_ledger(transition_id,session_id,occurrence_id,episode_id,event_kind,release_id,calibration_family_id,payload,account_after)
    VALUES(e->>'transition_id',p_session_id,e->>'occurrence_id',e->>'episode_id',e->>'event_kind','dip-v84-package1-evidence-20260910.1','dip-v84-package1-evidence-family-1',e,p_runtime);
    IF e->>'event_kind' IN ('SELL','SHORT_CLOSE') THEN
      ambiguous_loss=coalesce((e#>>'{metadata,execution_ambiguous_loss}')::boolean,false);
      INSERT INTO public.brian_dip_v84_execution_outcomes(outcome_id,session_id,occurrence_id,episode_id,setup,direction,regime,release_id,calibration_family_id,closed_at,resolution_reason,is_win,ambiguous_conservative_loss,realized_pnl)
      VALUES('outcome-'||(e->>'transition_id'),p_session_id,e->>'occurrence_id',e->>'episode_id',old_pos->>'setup',CASE WHEN old_pos->>'side'='LONG' THEN 'UP' ELSE 'DOWN' END,old_pos->>'regime','dip-v84-package1-evidence-20260910.1','dip-v84-package1-evidence-family-1',clock_timestamp(),coalesce(e#>>'{metadata,exit_reason}','UNKNOWN'),CASE WHEN ambiguous_loss THEN false ELSE (e->>'realized_pnl')::numeric>0 END,ambiguous_loss,(e->>'realized_pnl')::numeric);
    END IF;
  END LOOP;
  UPDATE public.brian_dip_v84_runtime SET runtime=p_runtime,snapshot=p_snapshot,state_version=state_version+1,last_commit_id=p_commit_id,last_commit_hash=payload_hash,updated_at=clock_timestamp() WHERE session_id=p_session_id;
  RETURN jsonb_build_object('status','COMMITTED','state_version',old_state.state_version+1);
END $$;

REVOKE ALL ON FUNCTION public.brian_dip_v84_commit(text,bigint,text,bigint,text,jsonb,jsonb,jsonb,jsonb) FROM public,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.brian_dip_v84_commit(text,bigint,text,bigint,text,jsonb,jsonb,jsonb,jsonb) TO service_role;

-- Still PREPARED / UNSEALED. Activation remains a separately reviewed migration.