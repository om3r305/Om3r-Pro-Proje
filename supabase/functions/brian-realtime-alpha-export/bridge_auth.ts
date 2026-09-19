const ALLOWED_SHA256 = new Set([
  "b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a",
  "814a5df4f8d6e3b15f1b9ac19a4ea823ad69eedc52caa6ad7573fde7aa96eaab",
]);

async function sha256Hex(value: string): Promise<string> {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function requireBridgeAuth(req: Request): Promise<void> {
  const supplied = (req.headers.get("x-brian-internal-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_INTERNAL");
  const digest = await sha256Hex(supplied);
  if (!ALLOWED_SHA256.has(digest)) throw new Error("UNAUTHORIZED_INTERNAL");
}
