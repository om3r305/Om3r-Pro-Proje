-- Brian DIP V8.4.4 Cycle Forecast + Harvest release.
-- SHADOW ONLY: live execution=false, browser execution=false, SHORT entries=false, max 1x.
-- V8.4.3 remains immutable for before/after evidence comparison.

insert into public.brian_dip_v84_releases(
  release_id,status,logic_hash,strategy_manifest_hash,calibration_family_id,db_contract_version,manifest,sealed_at
) values (
  'dip-v844-cycle-forecast-20260911.1','SEALED','git:e210589642dbac13e65570a6d729360c7d656b54',
  'b41bcf3d3ff6f5359937894f03d7165bdef3b425bc9bc6ff7e8abb6669b89449',
  'dip-v844-cycle-forecast-family-1','brian-dip-v84-db-2-authority',
  '{"symbol":"ETHUSDT","shadow_only":true,"live_execution":false,"browser_execution":false,"short_entries":false,"max_shadow_leverage":1,"server_authoritative":true,"decision_authority":"BRIAN_LONG_ONLY_CYCLE_FORECAST_HARVEST","target_policy":"BRIAN_FORECAST_PRIMARY_WITH_STRETCH_TELEMETRY","forecast_policy":"NEAR_TERM_REACHABLE_DESTINATION_FIRST","far_pivots":"STRETCH_ONLY","raw_bearish_immediate":false,"severe_bearish_immediate":true,"post_exit_rebase_required":true,"minimum_entry_quality":0.46,"minimum_forecast_net_edge_bps":5}'::jsonb,
  clock_timestamp()
) on conflict (release_id) do nothing;

do $do$
declare d text; n text;
begin
  select pg_get_functiondef('public.brian_dip_v84_acquire_lease(text,text,integer)'::regprocedure) into d;
  n:=replace(d,
    $old$    when 'brian-dip-v843-profit-protect-worker' then 'dip-v843-profit-protect-20260911.1'$old$,
    $new$    when 'brian-dip-v843-profit-protect-worker' then 'dip-v843-profit-protect-20260911.1'
    when 'brian-dip-v844-cycle-worker' then 'dip-v844-cycle-forecast-20260911.1'$new$);
  if n=d then raise exception 'V844_ACQUIRE_PATCH_ANCHOR_MISSING'; end if;
  execute n;
end $do$;

do $do$
declare d text; n text;
begin
  select pg_get_functiondef('public.brian_dip_v84_renew_lease(text,text,bigint,integer)'::regprocedure) into d;
  n:=replace(d,
    $old$    when 'brian-dip-v843-profit-protect-worker' then 'dip-v843-profit-protect-20260911.1'$old$,
    $new$    when 'brian-dip-v843-profit-protect-worker' then 'dip-v843-profit-protect-20260911.1'
    when 'brian-dip-v844-cycle-worker' then 'dip-v844-cycle-forecast-20260911.1'$new$);
  if n=d then raise exception 'V844_RENEW_PATCH_ANCHOR_MISSING'; end if;
  execute n;
end $do$;

do $do$
declare d text; n text;
begin
  select pg_get_functiondef('public.brian_dip_v84_initialize_session()'::regprocedure) into d;
  n:=replace(d,
$old$  else
    raise exception 'V84_UNKNOWN_RELEASE';
  end if;$old$,
$new$  elsif rel.release_id='dip-v844-cycle-forecast-20260911.1' then
    if new.config->>'allow_shadow_short' is distinct from 'false'
       or engine is distinct from 'brian-dip-v844-cycle-forecast'
       or new.config->>'policy_version' is distinct from 'dip-v844-cycle-forecast-20260911.1'
       or new.config->>'decision_revision' is distinct from 'dip-v844-cycle-occurrence-20260911.1'
       or new.config->>'metric_version' is distinct from 'forecast-before-entry-v844.1'
       or new.config->>'decision_authority' is distinct from 'BRIAN'
       or new.config->>'worker_lease_key' is distinct from 'brian-dip-v844-cycle-worker'
       or rel.manifest->>'short_entries' is distinct from 'false'
       or rel.manifest->>'raw_bearish_immediate' is distinct from 'false'
       or rel.manifest->>'post_exit_rebase_required' is distinct from 'true'
    then raise exception 'V844_CYCLE_FORECAST_VERSION_CONTRACT'; end if;
  else
    raise exception 'V84_UNKNOWN_RELEASE';
  end if;$new$);
  if n=d then raise exception 'V844_INITIALIZE_PATCH_ANCHOR_MISSING'; end if;
  execute n;
end $do$;

do $do$
declare d text; n text;
begin
  select pg_get_functiondef('public.brian_dip_v84_commit(text,bigint,text,bigint,text,jsonb,jsonb,jsonb,jsonb)'::regprocedure) into d;
  n:=replace(d,
$old$  elsif rel.release_id='dip-v843-profit-protect-20260911.1' then
    if rel.strategy_manifest_hash is distinct from 'd302c5aa4cf048e27236177674ad02bd9c825f08a2cae8b4a39b9f9b2627a25e' or cfg->>'allow_shadow_short' is distinct from 'false' or rel.manifest->>'short_entries' is distinct from 'false' or rel.manifest->>'strong_sell_immediate' is distinct from 'true' then raise exception 'V843_PROFIT_PROTECT_CANONICAL_HASH_MISMATCH';end if;
  else raise exception 'V84_UNKNOWN_RELEASE';end if;$old$,
$new$  elsif rel.release_id='dip-v843-profit-protect-20260911.1' then
    if rel.strategy_manifest_hash is distinct from 'd302c5aa4cf048e27236177674ad02bd9c825f08a2cae8b4a39b9f9b2627a25e' or cfg->>'allow_shadow_short' is distinct from 'false' or rel.manifest->>'short_entries' is distinct from 'false' or rel.manifest->>'strong_sell_immediate' is distinct from 'true' then raise exception 'V843_PROFIT_PROTECT_CANONICAL_HASH_MISMATCH';end if;
  elsif rel.release_id='dip-v844-cycle-forecast-20260911.1' then
    if rel.strategy_manifest_hash is distinct from 'b41bcf3d3ff6f5359937894f03d7165bdef3b425bc9bc6ff7e8abb6669b89449' or cfg->>'allow_shadow_short' is distinct from 'false' or rel.manifest->>'short_entries' is distinct from 'false' or rel.manifest->>'raw_bearish_immediate' is distinct from 'false' or rel.manifest->>'post_exit_rebase_required' is distinct from 'true' then raise exception 'V844_CYCLE_FORECAST_CANONICAL_HASH_MISMATCH';end if;
  else raise exception 'V84_UNKNOWN_RELEASE';end if;$new$);
  if n=d then raise exception 'V844_COMMIT_PATCH_ANCHOR_MISSING'; end if;
  execute n;
end $do$;

-- Fail closed if an unexpected row already occupied the immutable release id.
do $do$
declare r public.brian_dip_v84_releases%rowtype;
begin
  select * into r from public.brian_dip_v84_releases where release_id='dip-v844-cycle-forecast-20260911.1';
  if not found
     or r.status<>'SEALED'
     or r.logic_hash<>'git:e210589642dbac13e65570a6d729360c7d656b54'
     or r.strategy_manifest_hash<>'b41bcf3d3ff6f5359937894f03d7165bdef3b425bc9bc6ff7e8abb6669b89449'
     or r.calibration_family_id<>'dip-v844-cycle-forecast-family-1'
     or r.db_contract_version<>'brian-dip-v84-db-2-authority'
  then raise exception 'V844_RELEASE_REGISTRY_MISMATCH'; end if;
end $do$;
