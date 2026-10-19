-- The general Yahoo Finance RSS endpoint returns 404. This is a narrower,
-- explicitly labeled benchmark feed from the same provider, not all Yahoo news.
update public.brian_source_endpoints_v2
set endpoint_url='https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC%2C%5EIXIC&region=US&lang=en-US',
 expected_content='Yahoo Finance S&P 500 and Nasdaq benchmark news RSS',
 lifecycle_state='VERIFYING',updated_at=clock_timestamp(),
 metadata=metadata||jsonb_build_object('coverage_scope','SP500_NASDAQ_BENCHMARK_NEWS_NOT_ALL_YAHOO','replaces_unavailable_general_rss',true)
where endpoint_id='wire_yahoo_finance_rss' and source_id='wire:yahoo-finance'
 and endpoint_url='https://finance.yahoo.com/news/rssindex'
 and access_mode='PUBLIC_NO_KEY' and shadow_only and not live_execution;
