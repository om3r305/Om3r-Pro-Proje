import { assertEquals, assertThrows } from "jsr:@std/assert@1.0.19";
import { isAfterQueueReset, parseQueueResetAt } from "./engineering_queue_reset.ts";

Deno.test("no reset preserves earlier queue behavior",()=>{
  assertEquals(parseQueueResetAt(undefined),null);
  assertEquals(isAfterQueueReset("2026-10-18T10:00:00Z",null),true);
});
Deno.test("at and before reset are archived, later requests are new",()=>{
  const reset=parseQueueResetAt("2026-10-19T13:00:00Z");
  assertEquals(isAfterQueueReset("2026-10-18T10:00:00Z",reset),false);
  assertEquals(isAfterQueueReset("2026-10-19T13:00:00Z",reset),false);
  assertEquals(isAfterQueueReset("2026-10-19T13:00:01Z",reset),true);
});
Deno.test("malformed reset fails closed instead of silently reopening the queue",()=>{
  assertThrows(()=>parseQueueResetAt("broken"),Error,"ENGINEERING_QUEUE_RESET_INVALID");
  assertThrows(()=>parseQueueResetAt(123),Error,"ENGINEERING_QUEUE_RESET_INVALID");
});
Deno.test("request without a valid created_at is never new after reset",()=>{
  const reset=parseQueueResetAt("2026-10-19T13:00:00Z");
  assertEquals(isAfterQueueReset(null,reset),false);
  assertEquals(isAfterQueueReset("nonsense",reset),false);
});
