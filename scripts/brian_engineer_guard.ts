import {
  isDipProtectedPath,
  validateEngineerChangeSet,
} from "../supabase/functions/_shared/evolution_engineer_guard.ts";

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

function resolveImport(fromFile: string, specifier: string, tracked: Set<string>): string | null {
  if (!specifier.startsWith(".")) return null;
  const baseParts = fromFile.replaceAll("\\", "/").split("/");
  baseParts.pop();
  for (const part of specifier.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") baseParts.pop();
    else baseParts.push(part);
  }
  const raw = baseParts.join("/");
  for (const candidate of [raw, `${raw}.ts`, `${raw}.tsx`, `${raw}.js`, `${raw}.jsx`, `${raw}/index.ts`, `${raw}/index.js`]) {
    if (tracked.has(candidate)) return candidate;
  }
  return null;
}

async function assertNoIndirectDipDependency(changedPaths: string[]): Promise<void> {
  const sharedChanges = changedPaths.filter((path) => path.startsWith("supabase/functions/_shared/") && /\.(ts|tsx|js|jsx)$/i.test(path));
  if (!sharedChanges.length) return;

  const trackedList = (await git("ls-files", "supabase/functions")).split("\n").map((x) => x.trim()).filter(Boolean);
  const sourceFiles = trackedList.filter((path) => /\.(ts|tsx|js|jsx)$/i.test(path));
  const tracked = new Set(sourceFiles);
  const reverse = new Map<string, Set<string>>();
  const importPattern = /(?:\bfrom\s*|\bimport\s*\(|\bimport\s*)["']([^"']+)["']/g;

  for (const file of sourceFiles) {
    let text = "";
    try {
      text = await Deno.readTextFile(file);
    } catch {
      continue;
    }
    importPattern.lastIndex = 0;
    for (const match of text.matchAll(importPattern)) {
      const dependency = resolveImport(file, match[1], tracked);
      if (!dependency) continue;
      const consumers = reverse.get(dependency) ?? new Set<string>();
      consumers.add(file);
      reverse.set(dependency, consumers);
    }
  }

  for (const changed of sharedChanges) {
    const queue = [changed];
    const visited = new Set<string>(queue);
    while (queue.length) {
      const dependency = queue.shift()!;
      for (const consumer of reverse.get(dependency) ?? []) {
        if (isDipProtectedPath(consumer)) {
          fail(`shared change reaches protected DIP dependency graph:${changed} -> ${consumer}`);
        }
        if (!visited.has(consumer)) {
          visited.add(consumer);
          queue.push(consumer);
        }
      }
    }
  }
}

const base = (Deno.env.get("BRIAN_ENGINEER_BASE_SHA") ?? "").trim();
if (!/^[0-9a-f]{7,64}$/i.test(base)) fail("BRIAN_ENGINEER_BASE_SHA must be a git SHA");
const changed = (await git("diff", "--name-only", "--diff-filter=ACMR", base)).split("\n").map((x) => x.trim()).filter(Boolean);
const patch = await git("diff", "--no-ext-diff", "--binary", base);
const addedText = (await git("diff", "--unified=0", base)).split("\n").filter((line) => line.startsWith("+") && !line.startsWith("+++")).join("\n");
const patchBytes = new TextEncoder().encode(patch).length;
const result = validateEngineerChangeSet({ changedPaths: changed, patchBytes, addedText });
if (!result.valid) fail(result.reasons.join(" | "));

await assertNoIndirectDipDependency(changed);

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
  indirect_dip_dependency_clear: true,
}));
