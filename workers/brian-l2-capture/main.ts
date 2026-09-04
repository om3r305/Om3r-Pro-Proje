// Brian 2026 persistent real Binance Spot L2 collector bootstrap.
//
// SHADOW_RESEARCH_ONLY. Public market data only: no Binance API key, no account/order endpoint,
// no BUY/SELL output. Continuous capture runs as a portable Deno worker; Supabase is append-only
// storage/audit infrastructure, not the long-lived WebSocket runtime.

import { loadConfigFromEnv } from "./config.ts";
import { BrianRealL2Worker } from "./worker.ts";

async function main(): Promise<void> {
  const worker = new BrianRealL2Worker(loadConfigFromEnv());
  const stop = () => worker.stop();
  try { Deno.addSignalListener("SIGINT", stop); } catch { /* platform may not expose signal */ }
  try { Deno.addSignalListener("SIGTERM", stop); } catch { /* platform may not expose signal */ }

  try {
    await worker.run();
  } catch (error) {
    console.error("Brian real L2 capture terminated fail-closed", error);
    // Never silently restart a data-integrity/persistence failure inside the same collector
    // session. An external supervisor may start a brand-new session after the failure is visible.
    Deno.exit(1);
  }
}

if (import.meta.main) await main();
