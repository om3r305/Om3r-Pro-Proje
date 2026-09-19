const INTERNAL_KEY_SHA256 = "b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a";

async function sha256Hex(value: string): Promise<string> {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function constantTimeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i++) {
    diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  }
  return diff === 0;
}

export async function requireRealtimeInternal(req: Request): Promise<string> {
  const supplied = (req.headers.get("x-brian-internal-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_INTERNAL");
  const digest = await sha256Hex(supplied);
  if (!constantTimeEqual(digest, INTERNAL_KEY_SHA256)) {
    throw new Error("UNAUTHORIZED_INTERNAL");
  }
  return supplied;
}
