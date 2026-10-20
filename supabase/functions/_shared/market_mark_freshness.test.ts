import { assertEquals } from "jsr:@std/assert@1";
import { LIVE_MAX_AGE_SECONDS, markFreshness } from "./market_mark_freshness.ts";

const now = Date.parse("2026-10-20T08:53:02Z");

Deno.test("live only when the mark is fresh now, regardless of stored latency", () => {
  assertEquals(markFreshness("2026-10-20T08:52:30Z", now), { age_seconds: 32, live: true });
  // The SILVER case from 20 Oct: provider_time 08:40:24 seen at 08:53:02 -> 758 s old -> not live.
  assertEquals(markFreshness("2026-10-20T08:40:24Z", now), { age_seconds: 758, live: false });
  assertEquals(markFreshness(new Date(now - LIVE_MAX_AGE_SECONDS * 1000).toISOString(), now).live, true);
  assertEquals(markFreshness(new Date(now - (LIVE_MAX_AGE_SECONDS + 1) * 1000).toISOString(), now).live, false);
});

Deno.test("closed session, bad or future timestamps are never live", () => {
  assertEquals(markFreshness("2026-10-20T08:52:59Z", now, false).live, false);
  assertEquals(markFreshness("not a time", now), { age_seconds: null, live: false });
  assertEquals(markFreshness(null, now), { age_seconds: null, live: false });
  assertEquals(markFreshness("2026-10-20T09:10:00Z", now), { age_seconds: null, live: false });
});
