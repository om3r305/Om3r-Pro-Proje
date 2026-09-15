-- Remaining Source Architecture V2 endpoint repairs.
-- Never bypass provider access controls; inaccessible first-party surfaces remain explicit/fail-closed.

update public.brian_source_endpoints_v2
set endpoint_url='https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&count=40&output=atom',
    polling_seconds=60,
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
      'endpoint_fix','EDGAR_SMALLER_ATOM_WINDOW',
      'normal_polling_seconds',120,
      'temporary_validation_polling_seconds',60
    )
where endpoint_id='sec_edgar_current';

update public.brian_source_endpoints_v2
set canonical_domain='commission.europa.eu',
    endpoint_url='https://commission.europa.eu/news-and-media_en',
    endpoint_kind='HTML',
    expected_content='html',
    polling_seconds=60,
    lifecycle_state='VERIFYING',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
      'adapter','HTML_DIFF_PENDING',
      'endpoint_fix','CURRENT_COMMISSION_NEWS_SURFACE',
      'normal_polling_seconds',600,
      'temporary_validation_polling_seconds',60
    )
where endpoint_id='eu_press_corner';

update public.brian_source_endpoints_v2
set canonical_domain='mofcom.gov.cn',
    endpoint_url='https://www.mofcom.gov.cn/zcfb/',
    endpoint_kind='HTML',
    expected_content='html',
    polling_seconds=60,
    lifecycle_state='VERIFYING',
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
      'adapter','HTML_DIFF_PENDING',
      'language','zh',
      'endpoint_fix','CURRENT_CN_POLICY_RELEASES',
      'normal_polling_seconds',600,
      'temporary_validation_polling_seconds',60
    )
where endpoint_id='mofcom_news';

-- CFTC blocks the Edge runtime on both RSS and public press HTML. Do not bypass or spoof around it.
-- Keep the source registered for architecture/coverage visibility, but remove it from the public-no-key probe queue.
update public.brian_source_endpoints_v2
set access_mode='UNAVAILABLE',
    lifecycle_state='DEGRADED',
    polling_seconds=300,
    updated_at=now(),
    metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
      'endpoint_fix','EDGE_403_NO_BYPASS',
      'coverage_fallbacks',jsonb_build_array('federal_register_public_inspection','world_discovery'),
      'manual_recheck_required',true,
      'decision_evidence_locked',true
    )
where endpoint_id='cftc_press_rss';
