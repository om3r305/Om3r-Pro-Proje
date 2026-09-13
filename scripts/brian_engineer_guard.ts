import { validateEngineerChangeSet } from "../supabase/functions/_shared/evolution_engineer_guard.ts";

function fail(message: string): never {
  console.error(`BRIAN_ENGINEER_GUARD:${message}`);
  Deno.exit(1);
}

async function git(...args: string[]): Promise<string> {
  const command = new Deno.Command("git", { args, stdout: "piped", stderr: "piped" });
  const output = await command.output();
  if (!output.success) fail(new TextDecoder().decode(output.stderr).trim() || `git ${args.join(" ")} failed`);
  return new TextDecoder().decode(output.stdout).trim();
}

const base = (Deno.env.get("BRIAN_ENGINEER_BASE_SHA") ?? "").trim();
if (!/^[0-9a-f]{7,64}$/i.test(base)) fail("BRIAN_ENGINEER_BASE_SHA must be a git SHA");
const changed = (await git("diff", "--name-only", "--diff-filter=ACMR", `${base}...HEAD`)).split("\n").map((x) => x.trim()).filter(Boolean);
const patch = await git("diff", "--no-ext-diff", "--binary", `${base}...HEAD`);
const addedText = (await git("diff", "--unified=0", `${base}...HEAD`)).split("\n").filter((line) => line.startsWith("+") && !line.startsWith("+++")).join("\n");
const patchBytes = new TextEncoder().encode(patch).length;
const result = validateEngineerChangeSet({ changedPaths: changed, patchBytes, addedText });
if (!result.valid) fail(result.reasons.join(" | "));

const nonTestCode = changed.filter((path) => /\.(ts|tsx|js|jsx|py|sql)$/i.test(path) && !/(\.test\.|\/tests?\/)/i.test(path));
const tests = changed.filter((path) => /(\.test\.|\/tests?\/)/i.test(path));
if (nonTestCode.length && !tests.length) fail("source-code change requires a changed or added test");

console.log(JSON.stringify({
  status: "PASS",
  changed_paths: result.normalizedPaths,
  patch_bytes: patchBytes,
  source_files: nonTestCode.length,
  test_files: tests.length,
  protected_scope_clear: true,
}));
