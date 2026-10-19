export const BRIAN_ENGINEER_GUARD_VERSION = "brian.engineer-guard.v1.1-operator-tests";
export const MAX_ENGINEER_CHANGED_FILES = 12;
export const MAX_ENGINEER_PATCH_BYTES = 500_000;

const CONTROL_PLANE_PREFIXES = [
  ".github/workflows/",
  ".github/agents/",
  "supabase/migrations/",
  "supabase/functions/brian-evolution-engineering-gateway/",
  "supabase/functions/_shared/evolution_engineer_guard.ts",
  "scripts/brian_engineer_",
] as const;

const OPERATOR_TEST_PATHS = new Set([
  "supabase/functions/_shared/evolution_contract.test.ts",
  "supabase/functions/_shared/evolution_core.test.ts",
  "supabase/functions/_shared/evolution_research.test.ts",
  "supabase/functions/_shared/evolution_sandbox.test.ts",
  "supabase/functions/_shared/evolution_lab.test.ts",
  "supabase/functions/_shared/evolution_codegen.test.ts",
  "supabase/functions/_shared/evolution_alpha_intelligence.test.ts",
  "supabase/functions/_shared/evolution_alpha_reliability_mapping.test.ts",
  "supabase/functions/_shared/evolution_prospective_timing.test.ts",
  "supabase/functions/_shared/evolution_engineer_guard.test.ts",
  "tests/open-world-cycle.test.cjs", "tests/open-world-research.test.cjs",
  "tests/alpha-price-path-bridge.test.cjs", "tests/alpha-audit-cost.test.cjs",
  "tests/operator-state.test.cjs", "tests/research-scheduler.test.cjs",
  "tests/derivatives-universe.test.cjs", "tests/direct-wire-fallback.test.cjs",
  "tests/main-multiasset-pipeline.test.cjs", "tests/frontier-reliability.test.cjs",
  "tests/shadow-scorecard.test.cjs", "tests/shadow-scorecard-page-scope.test.cjs",
  "tests/engineer-node-tests.test.cjs", "tests/engineering-room-model.test.cjs",
  "tests/engineer-provider-canary.test.cjs", "tests/engineer-preflight.test.cjs",
  "tests/engineer-provider-mode.test.cjs", "tests/engineer-evaluator-boundary.test.cjs",
  "tests/engineer-checkpoint.test.cjs",
  "tests/cron-auth-contract.test.cjs", "tests/world-dossier.test.cjs",
  "supabase/functions/_shared/world_dossier.test.ts",
]);

const AUTH_OR_SECRET_PATTERNS = [
  /(^|\/)(\.env|secrets?)(\.|\/|$)/i,
  /credential/i,
  /dashboard_auth/i,
  /cron_auth/i,
  /service[_-]?role/i,
];

export type EngineerGuardInput = {
  changedPaths: string[];
  patchBytes: number;
  addedText?: string;
  allowControlPlane?: boolean;
};

export type EngineerGuardResult = {
  valid: boolean;
  reasons: string[];
  normalizedPaths: string[];
};

export function normalizeEngineerPath(value: string): string {
  return String(value ?? "").trim().replaceAll("\\", "/").replace(/^\.\//, "");
}

export function isDipProtectedPath(value: string): boolean {
  const path = normalizeEngineerPath(value).toLowerCase();
  const parts = path.split("/");
  return parts.some((part) =>
    part === "dip" ||
    part.startsWith("dip-") ||
    part.startsWith("dip_") ||
    part.includes("brian-dip") ||
    part.includes("brian_dip")
  );
}

export function isEngineerControlPlanePath(value: string): boolean {
  const path = normalizeEngineerPath(value);
  return OPERATOR_TEST_PATHS.has(path) || CONTROL_PLANE_PREFIXES.some((prefix) => path.startsWith(prefix));
}

export function validateEngineerChangeSet(input: EngineerGuardInput): EngineerGuardResult {
  const reasons: string[] = [];
  const normalizedPaths = [...new Set((input.changedPaths ?? []).map(normalizeEngineerPath).filter(Boolean))];

  if (!normalizedPaths.length) reasons.push("change set is empty");
  if (normalizedPaths.length > MAX_ENGINEER_CHANGED_FILES) reasons.push(`more than ${MAX_ENGINEER_CHANGED_FILES} files changed`);
  if (!Number.isInteger(input.patchBytes) || input.patchBytes < 0) reasons.push("patchBytes must be a non-negative integer");
  if (input.patchBytes > MAX_ENGINEER_PATCH_BYTES) reasons.push(`patch exceeds ${MAX_ENGINEER_PATCH_BYTES} bytes`);

  for (const path of normalizedPaths) {
    if (path.startsWith("/") || path.includes("../") || path === "..") reasons.push(`unsafe path:${path}`);
    if (isDipProtectedPath(path)) reasons.push(`protected DIP scope:${path}`);
    if (!input.allowControlPlane && isEngineerControlPlanePath(path)) reasons.push(`engineer control-plane is immutable to autonomous runs:${path}`);
    if (AUTH_OR_SECRET_PATTERNS.some((pattern) => pattern.test(path))) reasons.push(`auth/secret scope is protected:${path}`);
  }

  const addedText = String(input.addedText ?? "");
  if (/\bbrian[_-]dip\b/i.test(addedText)) reasons.push("patch text references protected DIP identifiers");

  const enablesLiveExecution = /\blive[_-]?execution\b["']?\s*[:=]\s*true\b/i.test(addedText) ||
    /\bliveExecution\b["']?\s*[:=]\s*true\b/i.test(addedText);
  const introducesAuthenticatedExecution = /\b(place[_-]?order|withdraw|exchange[_-]?secret)\b/i.test(addedText);
  if (enablesLiveExecution || introducesAuthenticatedExecution) {
    reasons.push("patch introduces live-execution or authenticated exchange surface");
  }

  return { valid: reasons.length === 0, reasons, normalizedPaths };
}
