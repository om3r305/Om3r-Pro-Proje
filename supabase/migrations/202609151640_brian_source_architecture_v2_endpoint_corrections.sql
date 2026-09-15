update public.brian_source_endpoints_v2
set endpoint_url='https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance',
    lifecycle_state='VERIFYING', updated_at=now(),
    metadata=metadata||jsonb_build_object('endpoint_corrected_at',now(),'endpoint_correction','Fiscal Data DTS uses v1 accounting/dts path')
where endpoint_id='treasury_dts';

update public.brian_source_endpoints_v2
set lifecycle_state='DISABLED', updated_at=now(),
    metadata=metadata||jsonb_build_object('disabled_reason','Official Upbit notice HTML returns 403 to server collectors; replaced by authenticated official announcement WebSocket')
where endpoint_id='upbit_notices';

insert into public.brian_source_endpoints_v2(
 endpoint_id,source_id,organization,canonical_domain,endpoint_url,endpoint_kind,tier,category,region,
 authority_class,access_mode,lifecycle_state,polling_seconds,priority,expected_content,collector_owner,
 official_origin,corroboration_required,manipulation_risk,direct_decision_allowed,shadow_only,live_execution,metadata
) values (
 'upbit_announcement_ws','official:upbit:announcements','Upbit Announcement WebSocket','api.upbit.com',
 'wss://api.upbit.com/websocket/v1/private','WEBSOCKET','T1_OFFICIAL_PRIMARY','CRYPTO_EXCHANGE','KR',
 'OFFICIAL_PRIMARY','API_KEY_REQUIRED','VERIFYING',60,100,'json',null,true,true,0.04,false,true,false,
 jsonb_build_object('source_arch_version','V2','adapter','AUTHENTICATED_WEBSOCKET_PENDING','auth_required',true,'type','announcement','categories',jsonb_build_array('trade'),'include_body',false,'decision_evidence_locked',true)
) on conflict(endpoint_id) do update set
 endpoint_url=excluded.endpoint_url,endpoint_kind=excluded.endpoint_kind,access_mode=excluded.access_mode,
 lifecycle_state=excluded.lifecycle_state,metadata=public.brian_source_endpoints_v2.metadata||excluded.metadata,updated_at=now();