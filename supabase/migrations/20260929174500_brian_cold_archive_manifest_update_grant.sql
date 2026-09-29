-- Target: brian-market-intelligence. brian-cold-archive upserts manifests
-- (on conflict archive_id) and stamps restores, both needing UPDATE; service_role
-- only had INSERT,SELECT so every run failed closed with 42501 before writing a
-- manifest. DELETE stays withheld: purges go through brian_archive_purge_manifest.
grant update on public.brian_archive_manifests to service_role;
