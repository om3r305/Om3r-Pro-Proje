/** Parse an operator reset boundary without silently reactivating old requests. */
export function parseQueueResetAt(raw: unknown): number | null {
  if (raw === undefined || raw === null || raw === "") return null;
  if (typeof raw !== "string") throw new Error("ENGINEERING_QUEUE_RESET_INVALID");
  const time = Date.parse(raw);
  if (!Number.isFinite(time)) throw new Error("ENGINEERING_QUEUE_RESET_INVALID");
  return time;
}

/** Prior requests are archived; only records created after the reset are new work. */
export function isAfterQueueReset(createdAt: unknown, resetAt: number | null): boolean {
  if (resetAt === null) return true;
  if (typeof createdAt !== "string") return false;
  const time = Date.parse(createdAt);
  return Number.isFinite(time) && time > resetAt;
}
