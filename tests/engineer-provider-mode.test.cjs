const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const { spawnSync } = require('node:child_process');
const prefix = fs.readFileSync(path.resolve(__dirname, '../scripts/brian_engineer_runner.sh'), 'utf8').split('\nrecord_event() {')[0]
  .replace('PROVIDER_STATUS_FILE="/tmp/brian-provider-status.json"', 'PROVIDER_STATUS_FILE="$TEST_ROOT/status.json"')
  .replace('PROVIDER_ATTEMPTS_FILE="/tmp/brian-provider-attempts.jsonl"', 'PROVIDER_ATTEMPTS_FILE="$TEST_ROOT/attempts.jsonl"');
test('hosted-only mode never falls through to configured BYOK after failure or quota exhaustion', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-provider-mode-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  for (const [exhausted, selected] of [[false, false], [true, false], [false, true]]) {
    const script = prefix + `
run_copilot_attempt() { echo "$2" >> "$TEST_ROOT/calls"; return 1; }
provider_summary() { :; }
HOSTED_QUOTA_EXHAUSTED=${exhausted}
SELECTED_PROVIDER=${selected ? 'COPILOT_HOSTED' : ''}
SELECTED_MODEL=${selected ? 'auto' : ''}
set +e
ai_call test /dev/null /dev/null read
exit $?
`;
    const calls = path.join(root, 'calls'); if (fs.existsSync(calls)) fs.unlinkSync(calls);
    const r = spawnSync('bash', ['-c', script], { encoding: 'utf8', env: { ...process.env, TEST_ROOT: root,
      GITHUB_STEP_SUMMARY: path.join(root, 'summary'), BRIAN_HOSTED_ONLY: 'true',
      BRIAN_OPENAI_API_KEY: 'test-only', BRIAN_ANTHROPIC_API_KEY: 'test-only' } });
    assert.equal(r.status, 75, r.stderr);
    const attempts = fs.existsSync(calls) ? fs.readFileSync(calls, 'utf8').trim().split('\n') : [];
    assert.deepEqual(attempts, exhausted ? [] : ['COPILOT_HOSTED']);
  }
});
test('code timeout uses its bounded budget and records telemetry without raw output', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-provider-timeout-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.writeFileSync(path.join(root, 'timeout'), '#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_ROOT/arguments"\necho "private failure detail" >&2\nexit 124\n', { mode: 0o755 });
  const harness = prefix + '\nset +e\nrun_copilot_attempt telemetry_mode COPILOT_HOSTED auto /dev/null "$TEST_ROOT/result" code\nexit $?\n';
  const r = spawnSync('bash', ['-c', harness], { encoding: 'utf8', env: { ...process.env,
    PATH: root + path.delimiter + process.env.PATH, TEST_ROOT: root, GITHUB_STEP_SUMMARY: path.join(root, 'summary'),
    BRIAN_HOSTED_ATTEMPT_TIMEOUT_SECONDS: '120', BRIAN_CODE_ATTEMPT_TIMEOUT_SECONDS: '360' } });
  assert.equal(r.status, 1, r.stderr);
  const receipt = JSON.parse(fs.readFileSync(path.join(root, 'attempts.jsonl'), 'utf8'));
  assert.equal(receipt.exit_code, 124); assert.equal(receipt.error_class, 'TIMEOUT');
  assert.equal(receipt.stdout_bytes, 0); assert.ok(receipt.stderr_bytes > 0);
  assert.equal(typeof receipt.worktree_changed_files, 'number');
  assert.ok(!JSON.stringify(receipt).includes('private failure detail'));
  assert.ok(fs.readFileSync(path.join(root, 'arguments'), 'utf8').split('\n').includes('360'));
  assert.ok(!fs.readFileSync(path.join(root, 'arguments'), 'utf8').split('\n').includes('--model'));
});
