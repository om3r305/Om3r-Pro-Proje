import assert from "node:assert/strict";
import { readRuntime, runWorker } from "./worker.ts";
import {
  ENGINE_VERSION,
  initialRuntime,
  POLICY_VERSION,
  type Runtime,
} from "../_shared/dip_v8.ts";
import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
const cfg = {
  engine_version: ENGINE_VERSION,
  policy_version: POLICY_VERSION,
  symbols: ["ETHUSDT"],
  shadow_only: true,
  live_execution: false,
  browser_execution: false,
  server_authoritative: true,
  allow_shadow_short: false,
  max_shadow_leverage: 1,
  execution_mode: "SHADOW_PAPER",
};
function database(
  rt: Runtime | null,
  error: string | null = null,
  kind = "START",
) {
  const commits: Record<string, unknown>[] = [];
  const db = {
    from(table: string) {
      const result = table === "brian_dip_session_events"
        ? {
          data: { session_id: "s", event_kind: kind, config: cfg },
          error: null,
        }
        : {
          data: rt ? { runtime: rt, state_version: 0 } : null,
          error: error ? { message: error } : null,
        };
      const chain = {
        select: () => chain,
        eq: () => chain,
        order: () => chain,
        limit: () => chain,
        maybeSingle: () => Promise.resolve(result),
      };
      return chain;
    },
    rpc(name: string, args: Record<string, unknown>) {
      if (name === "brian_dip_v8_commit") commits.push(args);
      return Promise.resolve({ data: [], error: null });
    },
  } as unknown as SupabaseClient;
  return { db, commits };
}
Deno.test("runtime read failure never manufactures starting cash or a flat position", async () => {
  for (
    const [rt, error, pattern] of [[null, "connection lost", /READ_FAILED/], [
      null,
      null,
      /STATE_MISSING/,
    ], [{ cash: 1000 }, null, /INVALID_RUNTIME/]] as const
  ) {
    const { db, commits } = database(rt as Runtime | null, error);
    await assert.rejects(() => readRuntime(db, "s"), pattern);
    assert.equal(commits.length, 0);
  }
});
Deno.test("paused flat session performs neither market calls nor accounting writes", async () => {
  const { db, commits } = database(initialRuntime(1000), null, "PAUSE");
  let calls = 0;
  const result = await runWorker(
    db,
    "owner",
    () => {},
    (() => {
      calls++;
      throw Error("unexpected fetch");
    }) as typeof fetch,
  );
  assert.equal(result.status, "PAUSED");
  assert.equal(calls, 0);
  assert.equal(commits.length, 0);
});
Deno.test("upper timeframe outage does not suppress an already observable position exit", async () => {
  const rt = initialRuntime(1000),
    start = Math.floor(Date.now() / 60000) * 60000 - 180000;
  rt.cash = 919.92;
  rt.pos = {
    side: "LONG",
    position_id: "p",
    thesis_id: "t",
    episode_id: "e",
    setup: "SWEEP_RECLAIM",
    regime: "RANGE",
    entry: 100,
    qty: .8,
    notional: 80,
    target: 103,
    stop: 99,
    opened_at: new Date(start).toISOString(),
    due_at: new Date(start + 120000).toISOString(),
    fees_open: .08,
    fee_bps: 10,
    slippage_bps: 1,
    spread_bps: 2,
    venue: "SPOT",
    policy_version: POLICY_VERSION,
    checked_until: start,
    market_price: 100,
    actual_fraction: .08,
  };
  const { db, commits } = database(rt);
  const fetcher = (async (url: Request | URL | string) => {
    const u = new URL(String(url));
    if (u.pathname === "/api/v3/klines" && u.searchParams.has("startTime")) {
      return Response.json(
        [0, 1].map(
          (i) => [
            start + i * 60000,
            "100",
            "104",
            "99.5",
            "100",
            "10",
            start + (i + 1) * 60000 - 1,
          ],
        ),
      );
    }
    return new Response("unavailable", { status: 503 });
  }) as typeof fetch;
  const result = await runWorker(db, "owner", () => {}, fetcher);
  assert.equal(result.status, "DATA_UNAVAILABLE");
  assert.equal(commits.length, 1);
  assert.equal((commits[0].p_runtime as Runtime).pos, null);
  const events = commits[0].p_events as { event_kind: string }[];
  assert.deepEqual(events.map((e) => e.event_kind), ["SELL"]);
  assert.equal((commits[0].p_runtime as Runtime).trades, 1);
});
Deno.test("lease loss stops the final mutation", async () => {
  const { db, commits } = database(initialRuntime(1000));
  await assert.rejects(
    () =>
      runWorker(
        db,
        "owner",
        () => {
          throw Error("LEASE_LOST");
        },
        (async () => new Response("offline", { status: 503 })) as typeof fetch,
      ),
    /LEASE_LOST/,
  );
  assert.equal(commits.length, 0);
});
