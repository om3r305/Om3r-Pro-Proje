-- Preserve publisher identity while recording its observed canonical RSS URL.
-- The previous endpoint redirects to this same-publisher, parsed RSS document.
update public.brian_source_endpoints_v2
set endpoint_url='https://www.bloomberg.com/feeds/markets/news.rss',
 canonical_domain='bloomberg.com',lifecycle_state='VERIFYING',updated_at=clock_timestamp(),
 metadata=metadata||jsonb_build_object('origin_repair','OBSERVED_SAME_PUBLISHER_CANONICAL_RSS')
where endpoint_id='wire_bloomberg_markets_rss'
 and endpoint_url='https://feeds.bloomberg.com/markets/news.rss'
 and canonical_domain='feeds.bloomberg.com'
 and access_mode='PUBLIC_NO_KEY' and shadow_only and not live_execution;
