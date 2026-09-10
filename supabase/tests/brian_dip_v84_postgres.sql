\set ON_ERROR_STOP on

-- Effective migration state must use the canonical TypeScript manifest hash.
DO $$
DECLARE h text;
BEGIN
  SELECT strategy_manifest_hash INTO h FROM public.brian_dip_v84_releases WHERE release_id='dip-v84-package1-evidence-20260910.1';
  IF h IS DISTINCT FROM '9a574469a3df381e56b0993d1f1525ec3a243970b92f8e867b99e07295c06f1e' THEN
    RAISE EXCEPTION 'TEST_MANIFEST_HASH_MISMATCH:%',h;
  END IF;
END $$;

-- PREPARED means START is impossible.
DO $$
BEGIN
  BEGIN
    INSERT INTO public.brian_dip_v84_session_events(session_id,event_kind,starting_equity,trade_notional,config)
    VALUES('ci-prepared','START',500,100,'{"release_id":"dip-v84-package1-evidence-20260910.1"}'::jsonb);
    RAISE EXCEPTION 'TEST_EXPECTED_PREPARED_START_FAILURE';
  EXCEPTION WHEN OTHERS THEN
    IF position('V84_RELEASE_NOT_SEALED' in SQLERRM)=0 THEN RAISE; END IF;
  END;
END $$;

-- Wrong manifest cannot seal the release.
DO $$
BEGIN
  BEGIN
    PERFORM public.brian_dip_v84_seal_release('dip-v84-package1-evidence-20260910.1','wrong-manifest',repeat('a',64));
    RAISE EXCEPTION 'TEST_EXPECTED_BAD_SEAL_FAILURE';
  EXCEPTION WHEN OTHERS THEN
    IF position('V84_RELEASE_SEAL_CONFLICT' in SQLERRM)=0 THEN RAISE; END IF;
  END;
END $$;

SELECT public.brian_dip_v84_seal_release(
  'dip-v84-package1-evidence-20260910.1',
  '9a574469a3df381e56b0993d1f1525ec3a243970b92f8e867b99e07295c06f1e',
  repeat('a',64)
);

-- Valid sealed SHADOW session initializes runtime + start ledger atomically through the trigger.
INSERT INTO public.brian_dip_v84_session_events(session_id,event_kind,starting_equity,trade_notional,config)
VALUES(
  'ci-session','START',500,100,
  jsonb_build_object(
    'release_id','dip-v84-package1-evidence-20260910.1',
    'logic_hash',repeat('a',64),
    'strategy_manifest_hash','9a574469a3df381e56b0993d1f1525ec3a243970b92f8e867b99e07295c06f1e',
    'calibration_family_id','dip-v84-package1-evidence-family-1',
    'db_contract_version','brian-dip-v84-db-1',
    'policy_version','dip-v84-l1-evidence-20260910.1',
    'engine_version','brian-dip-v84',
    'decision_revision','dip-v84-l1-occurrence-20260910.1',
    'metric_version','target-before-invalidation-v84.1',
    'symbols',jsonb_build_array('ETHUSDT'),
    'shadow_only',true,
    'live_execution',false,
    'browser_execution',false,
    'server_authoritative',true,
    'allow_shadow_short',true,
    'max_shadow_leverage',1,
    'execution_mode','SHADOW_PAPER'
  )
);

DO $$
DECLARE c integer;
BEGIN
  SELECT count(*) INTO c FROM public.brian_dip_v84_runtime WHERE session_id='ci-session';
  IF c<>1 THEN RAISE EXCEPTION 'TEST_RUNTIME_NOT_INITIALIZED'; END IF;
  SELECT count(*) INTO c FROM public.brian_dip_v84_ledger WHERE session_id='ci-session' AND event_kind='SESSION_START';
  IF c<>1 THEN RAISE EXCEPTION 'TEST_START_LEDGER_NOT_INITIALIZED'; END IF;
END $$;

-- Lease generation fences stale workers.
DO $$
DECLARE a boolean; g bigint;
BEGIN
  SELECT acquired,lease_generation INTO a,g FROM public.brian_dip_v84_acquire_lease('brian-dip-v84-worker','ci-owner',55);
  IF NOT a OR g<>1 THEN RAISE EXCEPTION 'TEST_LEASE_ACQUIRE_FAILED:%/%',a,g; END IF;
  IF public.brian_dip_v84_renew_lease('brian-dip-v84-worker','ci-owner',g+1,55) THEN RAISE EXCEPTION 'TEST_STALE_LEASE_RENEWED'; END IF;
END $$;

-- State-version mismatch and lease-generation mismatch must both fail closed.
DO $$
DECLARE rt jsonb; snap jsonb;
BEGIN
  SELECT runtime INTO rt FROM public.brian_dip_v84_runtime WHERE session_id='ci-session';
  snap=jsonb_build_object('session_id','ci-session','state',jsonb_build_object('serverRuntime',jsonb_build_object('release_id','dip-v84-package1-evidence-20260910.1','shadow_only',true,'live_execution',false)));
  BEGIN
    PERFORM public.brian_dip_v84_commit('ci-session',99,'ci-owner',1,'ci-stale-version',rt,snap,null,'[]'::jsonb);
    RAISE EXCEPTION 'TEST_EXPECTED_STATE_VERSION_FAILURE';
  EXCEPTION WHEN OTHERS THEN
    IF position('V84_STATE_VERSION_CONFLICT' in SQLERRM)=0 THEN RAISE; END IF;
  END;
END $$;

DO $$
DECLARE rt jsonb; snap jsonb; result jsonb;
BEGIN
  SELECT runtime INTO rt FROM public.brian_dip_v84_runtime WHERE session_id='ci-session';
  snap=jsonb_build_object('session_id','ci-session','state',jsonb_build_object('serverRuntime',jsonb_build_object('release_id','dip-v84-package1-evidence-20260910.1','shadow_only',true,'live_execution',false)));
  result=public.brian_dip_v84_commit('ci-session',0,'ci-owner',1,'ci-valid-noop',rt,snap,null,'[]'::jsonb);
  IF result->>'status'<>'COMMITTED' OR (result->>'state_version')::integer<>1 THEN RAISE EXCEPTION 'TEST_VALID_COMMIT_FAILED:%',result; END IF;
  BEGIN
    PERFORM public.brian_dip_v84_commit('ci-session',1,'ci-owner',999,'ci-stale-lease',rt,snap,null,'[]'::jsonb);
    RAISE EXCEPTION 'TEST_EXPECTED_LEASE_FENCE_FAILURE';
  EXCEPTION WHEN OTHERS THEN
    IF position('V84_LEASE_FENCE_LOST' in SQLERRM)=0 THEN RAISE; END IF;
  END;
END $$;

-- Database-level one-entry-per-episode invariant must reject a second opening event.
DO $$
BEGIN
  INSERT INTO public.brian_dip_v84_ledger(transition_id,session_id,occurrence_id,episode_id,event_kind,release_id,calibration_family_id,payload,account_after)
  VALUES('ci-open-1','ci-session','ci-occ-1','ci-episode','BUY','dip-v84-package1-evidence-20260910.1','dip-v84-package1-evidence-family-1','{}','{}');
  BEGIN
    INSERT INTO public.brian_dip_v84_ledger(transition_id,session_id,occurrence_id,episode_id,event_kind,release_id,calibration_family_id,payload,account_after)
    VALUES('ci-open-2','ci-session','ci-occ-2','ci-episode','SHORT_OPEN','dip-v84-package1-evidence-20260910.1','dip-v84-package1-evidence-family-1','{}','{}');
    RAISE EXCEPTION 'TEST_EXPECTED_DUPLICATE_EPISODE_FAILURE';
  EXCEPTION WHEN unique_violation THEN NULL;
  END;
END $$;

-- Empty valid execution family is COLD, never UNAVAILABLE and never fake-WARM.
DO $$
DECLARE r record;
BEGIN
  SELECT * INTO r FROM public.brian_dip_v84_execution_calibration(
    'dip-v84-package1-evidence-20260910.1','dip-v84-package1-evidence-family-1','SWEEP_RECLAIM','UP','RANGE'
  );
  IF r.state<>'COLD_NEW_FAMILY' OR r.samples<>0 OR r.p IS NOT NULL OR r.lower95 IS NOT NULL THEN
    RAISE EXCEPTION 'TEST_BAD_COLD_CALIBRATION:%/%',r.state,r.samples;
  END IF;
END $$;

-- Final installed commit function must contain the canonical hash and not the superseded one.
DO $$
DECLARE d text;
BEGIN
  SELECT pg_get_functiondef('public.brian_dip_v84_commit(text,bigint,text,bigint,text,jsonb,jsonb,jsonb,jsonb)'::regprocedure) INTO d;
  IF position('9a574469a3df381e56b0993d1f1525ec3a243970b92f8e867b99e07295c06f1e' in d)=0 THEN RAISE EXCEPTION 'TEST_COMMIT_MISSING_CANONICAL_HASH'; END IF;
  IF position('1b4938913e189d04d0b325727e0318b786c4c6fa4734d575f8b278a1d0696fb0' in d)>0 THEN RAISE EXCEPTION 'TEST_COMMIT_STILL_HAS_OLD_HASH'; END IF;
END $$;
