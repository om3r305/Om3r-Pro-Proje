// Brian Control Center core pinned to the V8 rollout merge.
// This preserves the known-good Control Center implementation while the public
// brian-control-center function layers V8 server-authoritative DIP status on top.
// Custom dashboard/cron authentication remains enforced by the imported source.
import "https://raw.githubusercontent.com/om3r305/Om3r-Pro-Proje/f1b2213e941b2f6d3e4effdecb8fa55dcd2544a2/supabase/functions/brian-control-center/index.ts";
