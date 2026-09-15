-- Restore normal production polling after the bounded Source Architecture V2 validation window.
update public.brian_source_endpoints_v2
set polling_seconds = case endpoint_id
  when 'sec_edgar_current' then 120
  when 'bafin_press' then 900
  when 'bls_cpi' then 600
  when 'consilium_press' then 600
  when 'destatis_press' then 900
  when 'ecb_press' then 600
  when 'eia_weekly_petroleum' then 900
  when 'eu_press_corner' then 600
  when 'mofcom_news' then 600
  when 'pbc_news' then 600
  when 'treasury_dts' then 3600
  else polling_seconds
end,
updated_at=now(),
metadata=coalesce(metadata,'{}'::jsonb) || jsonb_build_object('validation_window_complete_at',now(),'runtime_validated','REGISTRY_V4')
where endpoint_id in (
  'sec_edgar_current','bafin_press','bls_cpi','consilium_press','destatis_press','ecb_press',
  'eia_weekly_petroleum','eu_press_corner','mofcom_news','pbc_news','treasury_dts'
);

-- Upbit market-state telemetry intentionally remains at 60 seconds.
update public.brian_source_endpoints_v2
set polling_seconds=60, updated_at=now()
where endpoint_id='upbit_market_list';
