-- Brian Source Architecture V2
-- Endpoint-first registry, health and measured source scoring.
-- Discovery is data-only; no source is permitted to become live decision evidence here.

create table if not exists public.brian_source_endpoints_v2 (
  endpoint_id text primary key,
  source_id text not null,
  organization text not null,
  canonical_domain text not null,
  endpoint_url text not null,
  endpoint_kind text not null check (endpoint_kind in ('RSS','ATOM','JSON_API','HTML','DATASET','STATUSPAGE_ATOM','WEBSOCKET')),
  tier text not null check (tier in ('T0_RAW_TELEMETRY','T1_OFFICIAL_PRIMARY','T2_INSTITUTIONAL','T3_TOP_TIER_NEWS','T4_SPECIALIST','T5_DISCOVERY')),
  category text not null,
  region text not null default 'GLOBAL',
  authority_class text not null check (authority_class in ('OFFICIAL_PRIMARY','INDEPENDENT_PROFESSIONAL','COMMUNITY','UNKNOWN')),
  access_mode text not null check (access_mode in ('PUBLIC_NO_KEY','API_KEY_REQUIRED','LICENSED_REQUIRED','UNAVAILABLE')),
  lifecycle_state text not null default 'VERIFYING' check (lifecycle_state in ('VERIFYING','ACTIVE','DEGRADED','DISABLED','REJECTED')),
  polling_seconds integer not null default 900 check (polling_seconds >= 60),
  priority smallint not null default 50 check (priority between 0 and 100),
  expected_content text,
  collector_owner text,
  official_origin boolean not null default false,
  corroboration_required boolean not null default true,
  manipulation_risk double precision not null default 0.10 check (manipulation_risk between 0 and 1),
  direct_decision_allowed boolean not null default false check (not direct_decision_allowed),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(source_id, endpoint_url)
);

create table if not exists public.brian_source_endpoint_health_v2 (
  health_id text primary key,
  endpoint_id text not null references public.brian_source_endpoints_v2(endpoint_id) on delete cascade,
  observed_at timestamptz not null,
  reachable boolean not null,
  parseable boolean not null,
  origin_match boolean not null,
  http_status integer,
  latency_ms integer,
  content_type text,
  payload_bytes integer,
  content_hash text,
  change_detected boolean not null default false,
  error_class text,
  error_message text,
  metadata jsonb not null default '{}'::jsonb,
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create index if not exists brian_source_endpoint_health_v2_endpoint_time_idx
  on public.brian_source_endpoint_health_v2(endpoint_id, observed_at desc);

create table if not exists public.brian_source_scores_v2 (
  score_id text primary key,
  endpoint_id text not null references public.brian_source_endpoints_v2(endpoint_id) on delete cascade,
  assessed_at timestamptz not null,
  authority_score double precision not null check (authority_score between 0 and 1),
  lead_time_score double precision not null check (lead_time_score between 0 and 1),
  originality_score double precision not null check (originality_score between 0 and 1),
  market_relevance_score double precision not null check (market_relevance_score between 0 and 1),
  manipulation_risk double precision not null check (manipulation_risk between 0 and 1),
  historical_precision_score double precision not null check (historical_precision_score between 0 and 1),
  health_score double precision not null check (health_score between 0 and 1),
  composite_score double precision not null check (composite_score between 0 and 1),
  sample_count integer not null default 0 check (sample_count >= 0),
  eligible_for_research boolean not null default false,
  eligible_for_decision_evidence boolean not null default false check (not eligible_for_decision_evidence),
  reasons text[] not null default '{}'::text[],
  metadata jsonb not null default '{}'::jsonb,
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create index if not exists brian_source_scores_v2_endpoint_time_idx
  on public.brian_source_scores_v2(endpoint_id, assessed_at desc);

create or replace view public.brian_source_registry_status_v2 as
select
  e.*,
  h.observed_at as last_health_at,
  h.reachable as last_reachable,
  h.parseable as last_parseable,
  h.origin_match as last_origin_match,
  h.http_status as last_http_status,
  h.latency_ms as last_latency_ms,
  h.content_type as last_content_type,
  h.change_detected as last_change_detected,
  h.error_class as last_error_class,
  h.error_message as last_error_message,
  s.assessed_at as last_assessed_at,
  s.authority_score,
  s.lead_time_score,
  s.originality_score,
  s.market_relevance_score,
  s.historical_precision_score,
  s.health_score,
  s.composite_score,
  s.sample_count,
  s.eligible_for_research,
  s.eligible_for_decision_evidence
from public.brian_source_endpoints_v2 e
left join lateral (
  select * from public.brian_source_endpoint_health_v2 h0
  where h0.endpoint_id=e.endpoint_id order by h0.observed_at desc limit 1
) h on true
left join lateral (
  select * from public.brian_source_scores_v2 s0
  where s0.endpoint_id=e.endpoint_id order by s0.assessed_at desc limit 1
) s on true;

revoke all on public.brian_source_endpoints_v2, public.brian_source_endpoint_health_v2, public.brian_source_scores_v2 from anon, authenticated;
revoke all on public.brian_source_registry_status_v2 from anon, authenticated;
grant all on public.brian_source_endpoints_v2, public.brian_source_endpoint_health_v2, public.brian_source_scores_v2 to service_role;
grant select on public.brian_source_registry_status_v2 to service_role;

with seed(endpoint_id,source_id,organization,canonical_domain,endpoint_url,endpoint_kind,tier,category,region,authority_class,access_mode,polling_seconds,priority,expected_content,collector_owner,official_origin,corroboration_required,manipulation_risk,metadata) as (
values
('sec_edgar_current','official:sec:edgar_current','SEC EDGAR','sec.gov','https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&count=100&output=atom','ATOM','T1_OFFICIAL_PRIMARY','CORPORATE_FILINGS','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',120,100,'xml',null,true,true,0.03,'{"adapter":"GENERIC_FEED","forms":["8-K","10-Q","10-K","4","13D","13G","S-4","6-K","19b-4"],"user_agent_required":true}'::jsonb),
('sec_press_rss','official:sec:press','SEC Newsroom','sec.gov','https://www.sec.gov/news/pressreleases.rss','RSS','T1_OFFICIAL_PRIMARY','FINANCIAL_REGULATION','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',300,95,'xml',null,true,true,0.03,'{"adapter":"GENERIC_FEED","user_agent_required":true}'::jsonb),
('fed_monetary','official:fed:monetary','Federal Reserve Board','federalreserve.gov','https://www.federalreserve.gov/feeds/press_monetary.xml','RSS','T1_OFFICIAL_PRIMARY','CENTRAL_BANK','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',300,100,'xml','brian-official-macro-eye',true,true,0.02,'{"adapter":"OWNED_EXISTING"}'::jsonb),
('fed_h41','official:fed:h41','Federal Reserve Board H.4.1','federalreserve.gov','https://www.federalreserve.gov/releases/h41/','HTML','T1_OFFICIAL_PRIMARY','SYSTEMIC_LIQUIDITY','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',1800,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('nyfed_rates','official:nyfed:markets','Federal Reserve Bank of New York','newyorkfed.org','https://markets.newyorkfed.org/api/rates/all/latest.json','JSON_API','T1_OFFICIAL_PRIMARY','FUNDING_LIQUIDITY','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,95,'json',null,true,true,0.02,'{"adapter":"JSON_PENDING"}'::jsonb),
('ofac_recent_actions','official:ofac:recent_actions','OFAC','ofac.treasury.gov','https://ofac.treasury.gov/recent-actions','HTML','T1_OFFICIAL_PRIMARY','SANCTIONS','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',300,100,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('federal_register_public_inspection','official:federal_register:public_inspection','Federal Register Public Inspection','federalregister.gov','https://www.federalregister.gov/api/v1/public-inspection-documents/current.json','JSON_API','T1_OFFICIAL_PRIMARY','REGULATION','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',300,100,'json',null,true,true,0.02,'{"adapter":"JSON_PENDING"}'::jsonb),
('bls_employment','official:bls:employment','BLS Employment Situation','bls.gov','https://www.bls.gov/feed/empsit.rss','RSS','T1_OFFICIAL_PRIMARY','MACRO_EMPLOYMENT','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,95,'xml','brian-official-macro-eye',true,true,0.02,'{"adapter":"OWNED_EXISTING"}'::jsonb),
('bls_cpi','official:bls:cpi','BLS CPI','bls.gov','https://www.bls.gov/feed/cpi.rss','RSS','T1_OFFICIAL_PRIMARY','MACRO_INFLATION','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,100,'xml','brian-official-macro-eye',true,true,0.02,'{"adapter":"OWNED_EXISTING"}'::jsonb),
('bls_jolts','official:bls:jolts','BLS JOLTS','bls.gov','https://www.bls.gov/feed/jolts.rss','RSS','T1_OFFICIAL_PRIMARY','MACRO_EMPLOYMENT','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,80,'xml','brian-official-macro-eye',true,true,0.02,'{"adapter":"OWNED_EXISTING"}'::jsonb),
('bea_schedule','official:bea:schedule','Bureau of Economic Analysis','bea.gov','https://www.bea.gov/news/schedule','HTML','T1_OFFICIAL_PRIMARY','MACRO_GDP_PCE','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('treasury_press','official:ustreasury:press','U.S. Treasury','home.treasury.gov','https://home.treasury.gov/news/press-releases','HTML','T1_OFFICIAL_PRIMARY','FISCAL_POLICY','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('treasury_dts','official:ustreasury:dts','U.S. Treasury Fiscal Data','fiscaldata.treasury.gov','https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/dts/operating_cash_balance','JSON_API','T1_OFFICIAL_PRIMARY','FISCAL_LIQUIDITY','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',3600,85,'json',null,true,true,0.02,'{"adapter":"JSON_PENDING"}'::jsonb),
('eia_weekly_petroleum','official:eia:wpsr','U.S. EIA','eia.gov','https://www.eia.gov/petroleum/supply/weekly/','HTML','T1_OFFICIAL_PRIMARY','ENERGY_OIL','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('cftc_press_rss','official:cftc:press','CFTC','cftc.gov','https://www.cftc.gov/RSS/RSSGP/rssgp.xml','RSS','T1_OFFICIAL_PRIMARY','DERIVATIVES_REGULATION','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',300,95,'xml',null,true,true,0.02,'{"adapter":"GENERIC_FEED"}'::jsonb),
('ecb_press','official:ecb:press','ECB','ecb.europa.eu','https://www.ecb.europa.eu/rss/press.html','RSS','T1_OFFICIAL_PRIMARY','CENTRAL_BANK','EU','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,100,'xml','brian-official-macro-eye',true,true,0.02,'{"adapter":"OWNED_EXISTING"}'::jsonb),
('boj_whatsnew_rss','official:boj:whatsnew','Bank of Japan','boj.or.jp','https://www.boj.or.jp/en/rss/whatsnew.xml','RSS','T1_OFFICIAL_PRIMARY','CENTRAL_BANK','JP','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',300,95,'xml',null,true,true,0.02,'{"adapter":"GENERIC_FEED"}'::jsonb),
('bundesbank_general_rss','official:bundesbank:general','Deutsche Bundesbank','bundesbank.de','https://www.bundesbank.de/service/rss/de/633290/feed.rss','RSS','T1_OFFICIAL_PRIMARY','CENTRAL_BANK','DE','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,90,'xml',null,true,true,0.02,'{"adapter":"GENERIC_FEED","language":"de"}'::jsonb),
('boe_news','official:boe:news','Bank of England','bankofengland.co.uk','https://www.bankofengland.co.uk/news','HTML','T1_OFFICIAL_PRIMARY','CENTRAL_BANK','UK','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('pbc_news','official:pbc:news','People''s Bank of China','pbc.gov.cn','https://www.pbc.gov.cn/en/3688110/index.html','HTML','T1_OFFICIAL_PRIMARY','CENTRAL_BANK','CN','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,95,'html',null,true,true,0.04,'{"adapter":"HTML_DIFF_PENDING","language":"zh_en"}'::jsonb),
('eurostat_indicators','official:eurostat:indicators','Eurostat','ec.europa.eu','https://ec.europa.eu/eurostat/web/main/news/euro-indicators','HTML','T1_OFFICIAL_PRIMARY','MACRO_EU','EU','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,90,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('destatis_press','official:destatis:press','Destatis','destatis.de','https://www.destatis.de/EN/Press/press.html','HTML','T1_OFFICIAL_PRIMARY','MACRO_GERMANY','DE','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,90,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('esma_news','official:esma:news','ESMA','esma.europa.eu','https://www.esma.europa.eu/press-news/esma-news','HTML','T1_OFFICIAL_PRIMARY','FINANCIAL_REGULATION','EU','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,90,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('bafin_press','official:bafin:press','BaFin','bafin.de','https://www.bafin.de/EN/PublikationenDaten/Pressemitteilungen/pressemitteilungen_node_en.html','HTML','T1_OFFICIAL_PRIMARY','FINANCIAL_REGULATION','DE','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,90,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('fca_news','official:fca:news','FCA','fca.org.uk','https://www.fca.org.uk/news','HTML','T1_OFFICIAL_PRIMARY','FINANCIAL_REGULATION','UK','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,90,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('eu_press_corner','official:eucommission:press','European Commission Press Corner','ec.europa.eu','https://ec.europa.eu/commission/presscorner/home/en','HTML','T1_OFFICIAL_PRIMARY','EU_POLICY','EU','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('consilium_press','official:eucouncil:press','Council of the EU','consilium.europa.eu','https://www.consilium.europa.eu/en/press/press-releases/','HTML','T1_OFFICIAL_PRIMARY','GEOPOLITICS_SANCTIONS','EU','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('ustr_press','official:ustr:press','USTR','ustr.gov','https://ustr.gov/about-us/policy-offices/press-office/press-releases','HTML','T1_OFFICIAL_PRIMARY','TRADE_POLICY','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('bis_export_control','official:usbis:export_controls','U.S. Bureau of Industry and Security','bis.gov','https://www.bis.gov/','HTML','T1_OFFICIAL_PRIMARY','EXPORT_CONTROLS','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,95,'html',null,true,true,0.02,'{"adapter":"HTML_DIFF_PENDING","canonical_entity":"US_COMMERCE_BIS","not_bis_org":true}'::jsonb),
('mofcom_news','official:mofcom:news','China MOFCOM','mofcom.gov.cn','https://english.mofcom.gov.cn/','HTML','T1_OFFICIAL_PRIMARY','TRADE_POLICY','CN','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',600,95,'html',null,true,true,0.04,'{"adapter":"HTML_DIFF_PENDING","language":"zh_en"}'::jsonb),
('binance_announcements','official:binance:announcements','Binance Announcements','binance.com','https://www.binance.com/en/support/announcement','HTML','T1_OFFICIAL_PRIMARY','CRYPTO_EXCHANGE','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',120,100,'html',null,true,true,0.08,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('coinbase_status_atom','official:coinbase:status','Coinbase Status','status.coinbase.com','https://status.coinbase.com/history.atom','STATUSPAGE_ATOM','T1_OFFICIAL_PRIMARY','CRYPTO_EXCHANGE_STATUS','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',120,95,'xml',null,true,true,0.04,'{"adapter":"GENERIC_FEED"}'::jsonb),
('kraken_status_atom','official:kraken:status','Kraken Status','status.kraken.com','https://status.kraken.com/history.atom','STATUSPAGE_ATOM','T1_OFFICIAL_PRIMARY','CRYPTO_EXCHANGE_STATUS','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',120,95,'xml',null,true,true,0.04,'{"adapter":"GENERIC_FEED"}'::jsonb),
('upbit_notices','official:upbit:notices','Upbit Notices','upbit.com','https://upbit.com/service_center/notice','HTML','T1_OFFICIAL_PRIMARY','CRYPTO_EXCHANGE','KR','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',120,100,'html',null,true,true,0.08,'{"adapter":"HTML_DIFF_PENDING","language":"ko"}'::jsonb),
('tether_transparency','official:tether:transparency','Tether Transparency','tether.to','https://tether.to/en/transparency/','HTML','T1_OFFICIAL_PRIMARY','STABLECOIN','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,90,'html',null,true,true,0.10,'{"adapter":"HTML_DIFF_PENDING","issuer_claim_requires_onchain_corroboration":true}'::jsonb),
('circle_transparency','official:circle:transparency','Circle Transparency','circle.com','https://www.circle.com/transparency','HTML','T1_OFFICIAL_PRIMARY','STABLECOIN','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,90,'html',null,true,true,0.08,'{"adapter":"HTML_DIFF_PENDING","issuer_claim_requires_onchain_corroboration":true}'::jsonb),
('deribit_options_api','telemetry:deribit:options','Deribit Public API','deribit.com','https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option','JSON_API','T0_RAW_TELEMETRY','CRYPTO_OPTIONS','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',120,95,'json',null,true,true,0.04,'{"adapter":"JSON_PENDING"}'::jsonb),
('tsmc_monthly_revenue','official:tsmc:monthly_revenue','TSMC Investor Relations','investor.tsmc.com','https://investor.tsmc.com/english/monthly-revenue','HTML','T1_OFFICIAL_PRIMARY','SEMICONDUCTORS','TW','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',3600,90,'html',null,true,true,0.03,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('asml_press','official:asml:press','ASML Press Releases','asml.com','https://www.asml.com/en/news/press-releases','HTML','T1_OFFICIAL_PRIMARY','SEMICONDUCTORS','EU','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',1800,90,'html',null,true,true,0.03,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('nvidia_news','official:nvidia:news','NVIDIA Newsroom','nvidia.com','https://nvidianews.nvidia.com/','HTML','T1_OFFICIAL_PRIMARY','AI_SEMICONDUCTORS','US','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',900,90,'html',null,true,true,0.04,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('ukmto_advisories','official:ukmto:advisories','UKMTO Advisories','ukmto.org','https://www.ukmto.org/ukmto-products/advisories','HTML','T1_OFFICIAL_PRIMARY','MARITIME_SECURITY','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',300,95,'html',null,true,true,0.03,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('usgs_quakes','telemetry:usgs:quakes','USGS Earthquakes','usgs.gov','https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson','JSON_API','T0_RAW_TELEMETRY','NATURAL_HAZARD','GLOBAL','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',120,90,'json',null,true,true,0.01,'{"adapter":"JSON_PENDING"}'::jsonb),
('gdacs_rss','institutional:gdacs:rss','GDACS','gdacs.org','https://www.gdacs.org/xml/rss.xml','RSS','T2_INSTITUTIONAL','NATURAL_HAZARD','GLOBAL','INDEPENDENT_PROFESSIONAL','PUBLIC_NO_KEY',300,80,'xml',null,false,true,0.05,'{"adapter":"GENERIC_FEED"}'::jsonb),
('gassco_umm','official:gassco:umm','Gassco UMM','gassco.no','https://umm.gassco.no/','HTML','T1_OFFICIAL_PRIMARY','EUROPE_GAS','EU','OFFICIAL_PRIMARY','PUBLIC_NO_KEY',300,85,'html',null,true,true,0.03,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb),
('entsog_transparency','institutional:entsog:transparency','ENTSOG Transparency','entsog.eu','https://transparency.entsog.eu/','HTML','T2_INSTITUTIONAL','EUROPE_GAS','EU','INDEPENDENT_PROFESSIONAL','PUBLIC_NO_KEY',600,85,'html',null,false,true,0.03,'{"adapter":"HTML_DIFF_PENDING"}'::jsonb)
)
insert into public.brian_source_endpoints_v2 as e(
 endpoint_id,source_id,organization,canonical_domain,endpoint_url,endpoint_kind,tier,category,region,authority_class,access_mode,polling_seconds,priority,expected_content,collector_owner,official_origin,corroboration_required,manipulation_risk,metadata
)
select * from seed
on conflict(endpoint_id) do update set
 source_id=excluded.source_id, organization=excluded.organization, canonical_domain=excluded.canonical_domain,
 endpoint_url=excluded.endpoint_url, endpoint_kind=excluded.endpoint_kind, tier=excluded.tier,
 category=excluded.category, region=excluded.region, authority_class=excluded.authority_class,
 access_mode=excluded.access_mode, polling_seconds=excluded.polling_seconds, priority=excluded.priority,
 expected_content=excluded.expected_content, collector_owner=excluded.collector_owner,
 official_origin=excluded.official_origin, corroboration_required=excluded.corroboration_required,
 manipulation_risk=excluded.manipulation_risk, metadata=e.metadata || excluded.metadata, updated_at=now();

-- Mirror endpoint identities into the existing World source library, but keep every new source VERIFYING.
insert into public.brian_world_source_candidates(
 candidate_id, source_id, discovered_at, canonical_uri, provider, source_kind, authority_class,
 access_mode, stage, freshness_seconds, corroboration_required, manipulation_risk, rationale,
 metadata, evidence_class, shadow_only, live_execution
)
select
 'source-arch-v2:'||endpoint_id,
 source_id,
 now(),
 endpoint_url,
 organization,
 'SOURCE_ARCH_V2_'||tier,
 authority_class,
 access_mode,
 'VERIFYING',
 polling_seconds * 3,
 corroboration_required,
 manipulation_risk,
 'Brian Source Architecture V2 endpoint; verification and measured scoring required before research eligibility.',
 jsonb_build_object(
   'source_arch_version','V2',
   'endpoint_id',endpoint_id,
   'tier',tier,
   'category',category,
   'region',region,
   'official_origin',official_origin,
   'decision_evidence_locked',true,
   'origin_verification_pending',true,
   'collector_owner',collector_owner
 ) || metadata,
 'PROSPECTIVE_EVOLUTION_SHADOW', true, false
from public.brian_source_endpoints_v2
on conflict(candidate_id) do update set
 canonical_uri=excluded.canonical_uri,
 provider=excluded.provider,
 source_kind=excluded.source_kind,
 authority_class=excluded.authority_class,
 access_mode=excluded.access_mode,
 freshness_seconds=excluded.freshness_seconds,
 corroboration_required=excluded.corroboration_required,
 manipulation_risk=excluded.manipulation_risk,
 rationale=excluded.rationale,
 metadata=public.brian_world_source_candidates.metadata || excluded.metadata;
