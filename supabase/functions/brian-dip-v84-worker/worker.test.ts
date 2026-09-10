import assert from "node:assert/strict";
import { assertReleaseSealed, runWorker } from "./worker.ts";

Deno.test("unsealed GitHub release fails closed before database or market execution",async()=>{
  const db=new Proxy({} as Record<string,unknown>,{get(){throw Error("DB_MUST_NOT_BE_TOUCHED_BEFORE_SOURCE_SEAL");}});
  await assert.rejects(()=>assertReleaseSealed(db as any),/V84_RELEASE_NOT_SEALED_IN_SOURCE/);
  await assert.rejects(()=>runWorker(db as any,{owner:"x",generation:1,assertOwned:()=>{}},(async()=>{throw Error("MARKET_MUST_NOT_BE_TOUCHED");}) as typeof fetch),/V84_RELEASE_NOT_SEALED_IN_SOURCE/);
});
