#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const sha = value => createHash('sha256').update(value).digest('hex');
const spec = `Implement assessEvidence(rows, decisionAt, maxAgeMs=120000) in candidate.cjs, exporting { assessEvidence } using CommonJS.
decisionAt must be a finite numeric epoch in milliseconds; maxAgeMs must be finite, numeric and nonnegative. Invalid arguments return []. rows must be an array.
Each valid row has a nonempty string asset_id, a valid ISO timestamp string observed_at, a finite numeric positive price, and a finite numeric nonnegative cost_bps. Ignore malformed rows, missing costs, future rows and rows older than maxAgeMs. Boundaries are inclusive.
Keep the newest valid row per asset_id; equal timestamps keep the first valid input. Sort output ascending by asset_id. Output only asset_id, observed_at, price, cost_bps and computed age_ms. Never mutate input. No dependencies, filesystem, network, subprocesses or dynamic imports.
Only edit candidate.cjs. Read this specification and inspect the existing source before editing. External acceptance tests run after you finish; do not claim they have passed. This is a synthetic provider capability check, not a trading strategy or production task.`;
const stub = "'use strict';\nfunction assessEvidence() { throw new Error('NOT_IMPLEMENTED'); }\nmodule.exports = { assessEvidence };\n";
function classify(text, rc) {
  if (/quota|premium request.*limit|credits.*exhaust/i.test(text)) return 'QUOTA_EXHAUSTED';
  if (/401|403|unauthori[sz]ed|authenticat|invalid.*token|login|sign in/i.test(text)) return 'AUTH';
  if (/429|rate.?limit|too many requests/i.test(text)) return 'RATE_LIMIT';
  if (rc === 124 || rc === 137 || /timed out|timeout/i.test(text)) return 'TIMEOUT';
  if (/unknown option|unknown argument|invalid.*option/i.test(text)) return 'CLI_CONFIGURATION';
  if (/model.*(unknown|not found|unavailable|unsupported)/i.test(text)) return 'MODEL_UNAVAILABLE';
  return 'PROVIDER_ERROR';
}
function args() {
  return ['-p', 'Read SPEC.md and implement the contract in candidate.cjs. Only edit candidate.cjs.', '-s',
    '--disable-builtin-mcps', '--no-custom-instructions', '--no-ask-user', '--no-auto-update',
    '--available-tools=view,grep,glob,edit,create,apply_patch', '--allow-tool=read,write(candidate.cjs)',
    '--deny-tool=shell', '--deny-tool=url'];
}
function validateWorkspace(root, originalSpec) {
  const files = fs.readdirSync(root).sort();
  if (JSON.stringify(files) !== JSON.stringify(['SPEC.md', 'candidate.cjs'])) throw Error('OUT_OF_SCOPE_WRITE');
  for (const name of files) if (!fs.lstatSync(path.join(root, name)).isFile()) throw Error('NON_REGULAR_FILE');
  if (fs.readFileSync(path.join(root, 'SPEC.md'), 'utf8') !== originalSpec) throw Error('SPEC_MUTATED');
  const source = fs.readFileSync(path.join(root, 'candidate.cjs'));
  if (source.length > 32000) throw Error('CANDIDATE_TOO_LARGE');
  if (source.toString() === stub) throw Error('NO_CODE_CHANGE');
  return source;
}
function main() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-model-canary-'));
  const reportPath = process.env.BRIAN_CANARY_REPORT;
  if (!reportPath) throw Error('BRIAN_CANARY_REPORT required');
  const started = new Date().toISOString();
  const report = { started_at: started, observed_at: null, state: 'FAILED', stage: 'BOOTSTRAP',
    requested_provider: 'COPILOT_HOSTED', requested_model: 'CLI_DEFAULT', actual_model: 'UNVERIFIED',
    cli_version: '1.0.98', attempts: 1, code_written: false, acceptance_passed: false,
    production_task_claimed: false, autonomous_worker_enabled: false, live_execution: false };
  try {
    fs.writeFileSync(path.join(root, 'SPEC.md'), spec);
    fs.writeFileSync(path.join(root, 'candidate.cjs'), stub);
    fs.chmodSync(root, 0o755);
    report.stage = 'PROVIDER';
    const env = { ...process.env };
    // Only the existing hosted Copilot credential is used. No provider fallback or paid setup.
    for (const key of Object.keys(env)) if (/^COPILOT_PROVIDER_|^BRIAN_OPENAI_|^BRIAN_ANTHROPIC_/.test(key)) delete env[key];
    delete env.COPILOT_MODEL;
    const ai = spawnSync('timeout', ['--signal=TERM', '--kill-after=10s', '180', 'copilot', ...args()],
      { cwd: root, env, encoding: 'utf8', timeout: 195000, maxBuffer: 2 * 1024 * 1024 });
    report.provider_exit_code = ai.status;
    if (ai.error || ai.status !== 0) throw Error(classify((ai.stdout || '') + (ai.stderr || ''), ai.status));
    report.stage = 'SCOPE';
    const source = validateWorkspace(root, spec);
    report.code_written = true; report.candidate_sha256 = sha(source);
    report.stage = 'ACCEPTANCE';
    // Generated code only runs inside a disposable, credential-free, network-free container.
    const evaluation = spawnSync('docker', ['run', '--rm', '--network=none', '--read-only', '--cap-drop=ALL',
      '--security-opt=no-new-privileges', '--pids-limit=64', '--memory=128m', '--cpus=1', '--user=65534:65534',
      '--mount', `type=bind,src=${root},dst=/candidate,readonly`,
      '--mount', `type=bind,src=${path.resolve(__dirname, 'brian_engineer_canary_eval.cjs')},dst=/checks/eval.cjs,readonly`,
      'node:24-alpine', 'node', '--test', '--test-reporter=tap', '/checks/eval.cjs'],
      { encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024 });
    report.acceptance_exit_code = evaluation.status;
    const passed = Number((evaluation.stdout?.match(/^# pass (\d+)\s*$/m) || [])[1]);
    report.tests_passed = Number.isFinite(passed) ? passed : 0;
    if (evaluation.error || evaluation.status !== 0 || passed !== 8) throw Error('ACCEPTANCE_FAILED');
    report.acceptance_passed = true; report.state = 'CAPABILITY_CANARY_PASSED'; report.stage = 'COMPLETE';
  } catch (error) {
    // Only fixed error identifiers are persisted; raw model output and credentials are never uploaded.
    const name = String(error.message);
    report.error = /^[A-Z_]+$/.test(name) ? name : 'LOCAL_EXECUTION_ERROR';
    process.exitCode = 1;
  } finally {
    report.observed_at = new Date().toISOString();
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
    fs.rmSync(root, { recursive: true, force: true });
  }
}
if (require.main === module) main();
module.exports = { args, classify, validateWorkspace, spec, stub };
