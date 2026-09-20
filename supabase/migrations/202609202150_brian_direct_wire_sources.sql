
insert into public.brian_source_endpoints_v2(
  endpoint_id,source_id,organization,canonical_domain,endpoint_url,endpoint_kind,tier,category,
  region,authority_class,access_mode,lifecycle_state,polling_seconds,priority,expected_content,
  collector_owner,official_origin,corroboration_required,manipulation_risk,direct_decision_allowed,
  shadow_only,live_execution,metadata,updated_at
) values
(
  'fed_speeches_rss','official:fed:board:speeches','Federal Reserve Board','federalreserve.gov',
  'https://www.federalreserve.gov/feeds/speeches.xml','RSS','T1_OFFICIAL_PRIMARY','CENTRAL_BANK_SPEECH',
  'US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY','ACTIVE',60,100,'Federal Reserve Board speeches RSS',
  'brian-realtime-official-eye',true,false,0.01,false,true,false,
  '{"adapter":"REALTIME_OFFICIAL_EYE","critical_macro_watch":true,"speech_feed":true}'::jsonb,now()
),
(
  'minneapolisfed_policy_html','official:fed:minneapolis:speeches','Federal Reserve Bank of Minneapolis','minneapolisfed.org',
  'https://www.minneapolisfed.org/topic/policy','HTML','T1_OFFICIAL_PRIMARY','CENTRAL_BANK_SPEECH',
  'US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY','ACTIVE',60,100,'Minneapolis Fed policy and speech links',
  'brian-realtime-official-eye',true,false,0.01,false,true,false,
  '{"adapter":"REALTIME_OFFICIAL_EYE","html_path_prefix":"/speeches/","critical_macro_watch":true,"regional_fed":true}'::jsonb,now()
),
(
  'wire_yahoo_finance_rss','wire:yahoo-finance','Yahoo Finance','finance.yahoo.com',
  'https://finance.yahoo.com/news/rssindex','RSS','T2_INSTITUTIONAL','MARKET_WIRE',
  'GLOBAL','INDEPENDENT_PROFESSIONAL','PUBLIC_NO_KEY','ACTIVE',120,100,'Yahoo Finance market news RSS',
  'brian-direct-wire-eye',false,true,0.08,false,true,false,
  '{"adapter":"DIRECT_WIRE_EYE","independent_from_bing_google":true,"publisher_filter":true}'::jsonb,now()
),
(
  'wire_bloomberg_markets_rss','wire:bloomberg-markets','Bloomberg Markets','feeds.bloomberg.com',
  'https://feeds.bloomberg.com/markets/news.rss','RSS','T2_INSTITUTIONAL','MARKET_WIRE',
  'GLOBAL','INDEPENDENT_PROFESSIONAL','PUBLIC_NO_KEY','ACTIVE',120,100,'Bloomberg Markets RSS',
  'brian-direct-wire-eye',false,true,0.05,false,true,false,
  '{"adapter":"DIRECT_WIRE_EYE","independent_from_bing_google":true}'::jsonb,now()
),
(
  'wire_cnbc_us_rss','wire:cnbc-us','CNBC','cnbc.com',
  'https://www.cnbc.com/id/100003114/device/rss/rss.html','RSS','T2_INSTITUTIONAL','MARKET_WIRE',
  'US','INDEPENDENT_PROFESSIONAL','PUBLIC_NO_KEY','ACTIVE',120,95,'CNBC US top news RSS',
  'brian-direct-wire-eye',false,true,0.06,false,true,false,
  '{"adapter":"DIRECT_WIRE_EYE","independent_from_bing_google":true}'::jsonb,now()
),
(
  'wire_aljazeera_rss','wire:aljazeera','Al Jazeera','aljazeera.com',
  'https://www.aljazeera.com/xml/rss/all.xml','RSS','T2_INSTITUTIONAL','GEOPOLITICS_WIRE',
  'GLOBAL','INDEPENDENT_PROFESSIONAL','PUBLIC_NO_KEY','ACTIVE',120,90,'Al Jazeera breaking world RSS',
  'brian-direct-wire-eye',false,true,0.10,false,true,false,
  '{"adapter":"DIRECT_WIRE_EYE","independent_from_bing_google":true,"geopolitics_only":true}'::jsonb,now()
)
on conflict(endpoint_id) do update set
  source_id=excluded.source_id,
  organization=excluded.organization,
  canonical_domain=excluded.canonical_domain,
  endpoint_url=excluded.endpoint_url,
  endpoint_kind=excluded.endpoint_kind,
  tier=excluded.tier,
  category=excluded.category,
  region=excluded.region,
  authority_class=excluded.authority_class,
  access_mode=excluded.access_mode,
  lifecycle_state=excluded.lifecycle_state,
  polling_seconds=excluded.polling_seconds,
  priority=excluded.priority,
  expected_content=excluded.expected_content,
  collector_owner=excluded.collector_owner,
  official_origin=excluded.official_origin,
  corroboration_required=excluded.corroboration_required,
  manipulation_risk=excluded.manipulation_risk,
  direct_decision_allowed=excluded.direct_decision_allowed,
  shadow_only=excluded.shadow_only,
  live_execution=excluded.live_execution,
  metadata=excluded.metadata,
  updated_at=now();
