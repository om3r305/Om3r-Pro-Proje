-- Brian Source Architecture V2 runtime hardening.
-- Fix only source-registry plumbing/endpoints. Decision evidence remains locked.

-- Correct documented Treasury DTS API version.
update public.brian_source_endpoints_v2
set endpoint_url='https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object('endpoint_fix','DTS_V1_DOCUMENTED_2026_09_15')
where endpoint_id='treasury_dts';

-- CFTC's RSS path returns 403 from the Edge runtime. Keep the same official origin,
-- but health-check the public Press Releases surface instead of bypassing access controls.
update public.brian_source_endpoints_v2
set endpoint_url='https://www.cftc.gov/PressRoom/PressReleases',
    endpoint_kind='HTML',
    expected_content='html',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object('adapter','HTML_DIFF_PENDING','endpoint_fix','OFFICIAL_PRESS_HTML_FALLBACK')
where endpoint_id='cftc_press_rss';

-- Council of the EU publishes an official machine-readable press-release RSS feed.
update public.brian_source_endpoints_v2
set endpoint_url='https://www.consilium.europa.eu/en/rss/pressreleases.ashx',
    endpoint_kind='RSS',
    expected_content='xml',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object('adapter','GENERIC_FEED','endpoint_fix','OFFICIAL_RSS')
where endpoint_id='consilium_press';

-- Current Destatis English press hub.
update public.brian_source_endpoints_v2
set endpoint_url='https://www.destatis.de/EN/Press/press_node_2.html',
    endpoint_kind='HTML',
    expected_content='html',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object('adapter','HTML_DIFF_PENDING','endpoint_fix','CURRENT_EN_PRESS_HUB')
where endpoint_id='destatis_press';

-- BaFin changed its public information architecture. Use its current official home/news surface;
-- a narrower feed can replace this later if BaFin publishes a stable machine endpoint.
update public.brian_source_endpoints_v2
set endpoint_url='https://bafin.de/',
    endpoint_kind='HTML',
    expected_content='html',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object('adapter','HTML_DIFF_PENDING','endpoint_fix','CURRENT_OFFICIAL_SURFACE')
where endpoint_id='bafin_press';

-- PBOC English legacy index is stale/404. Observe the current official Chinese primary page.
update public.brian_source_endpoints_v2
set endpoint_url='https://www.pbc.gov.cn/?isapp=true',
    endpoint_kind='HTML',
    expected_content='html',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object('adapter','HTML_DIFF_PENDING','language','zh','endpoint_fix','CURRENT_CN_PRIMARY')
where endpoint_id='pbc_news';

-- Do not bypass Upbit's 403-protected notice web surface.
-- Replace it with the official public Korean market-state API and diff pair/warning changes.
update public.brian_source_endpoints_v2
set lifecycle_state='DISABLED',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object('disabled_reason','WEB_NOTICE_403_USE_OFFICIAL_MARKET_API')
where endpoint_id='upbit_notices';

insert into public.brian_source_endpoints_v2(
  endpoint_id,source_id,organization,canonical_domain,endpoint_url,endpoint_kind,tier,category,region,
  authority_class,access_mode,lifecycle_state,polling_seconds,priority,expected_content,collector_owner,
  official_origin,corroboration_required,manipulation_risk,direct_decision_allowed,shadow_only,live_execution,metadata
) values (
  'upbit_market_list','official:upbit:market_list','Upbit Korea','upbit.com',
  'https://api.upbit.com/v1/market/all?is_details=true','JSON_API','T0_RAW_TELEMETRY','CRYPTO_EXCHANGE_MARKETS','KR',
  'OFFICIAL_PRIMARY','PUBLIC_NO_KEY','VERIFYING',60,100,'json',null,true,true,0.04,false,true,false,
  '{"adapter":"JSON_DIFF_PENDING","purpose":"listing_delisting_warning_state","source_arch_version":"V2","decision_evidence_locked":true}'::jsonb
) on conflict(endpoint_id) do update set
  endpoint_url=excluded.endpoint_url,
  endpoint_kind=excluded.endpoint_kind,
  tier=excluded.tier,
  category=excluded.category,
  lifecycle_state=case when public.brian_source_endpoints_v2.lifecycle_state='REJECTED' then 'REJECTED' else 'VERIFYING' end,
  polling_seconds=excluded.polling_seconds,
  priority=excluded.priority,
  expected_content=excluded.expected_content,
  updated_at=now(),
  metadata=public.brian_source_endpoints_v2.metadata || excluded.metadata;

insert into public.brian_world_source_candidates(
  candidate_id,source_id,discovered_at,canonical_uri,provider,source_kind,authority_class,access_mode,stage,
  freshness_seconds,corroboration_required,manipulation_risk,rationale,metadata,evidence_class,shadow_only,live_execution
) values (
  'source-arch-v2:upbit_market_list','official:upbit:market_list',now(),
  'https://api.upbit.com/v1/market/all?is_details=true','SOURCE_ARCH_V2','OFFICIAL_EXCHANGE_MARKET_STATE',
  'OFFICIAL_PRIMARY','PUBLIC_NO_KEY','VERIFYING',60,true,0.04,
  'Official Upbit Korea market-list/warning state; used as data-only listing/delisting early telemetry.',
  '{"source_arch_version":"V2","endpoint_id":"upbit_market_list","tier":"T0_RAW_TELEMETRY","category":"CRYPTO_EXCHANGE_MARKETS","region":"KR","official_origin":true,"origin_verification_pending":true,"decision_evidence_locked":true}'::jsonb,
  'PROSPECTIVE_EVOLUTION_SHADOW',true,false
) on conflict(candidate_id) do nothing;

-- Idempotent cron. Endpoint-specific polling_seconds still controls actual external requests;
-- this scheduler only asks the registry for the oldest due batch.
do $$
declare r record;
begin
  for r in select jobid from cron.job where jobname='brian-source-registry-v2-2m' loop
    perform cron.unschedule(r.jobid);
  end loop;
end $$;

select cron.schedule(
  'brian-source-registry-v2-2m',
  '*/2 * * * *',
  $cron$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-source-registry-v2'
      from vault.decrypted_secrets where name='brian_project_url' limit 1
    ),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 55000
  );
  $cron$
);
