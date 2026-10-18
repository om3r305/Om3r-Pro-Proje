const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const { spawnSync } = require('node:child_process');
const script = path.resolve(__dirname, '../scripts/brian_engineer_preflight.sh');
function run(t, request, response = '{"should_run":false}', fail = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-preflight-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const curl = path.join(root, 'curl');
  fs.writeFileSync(curl, `#!/bin/sh\necho call >> "$COUNTER"\nif [ "$FAIL" = true ]; then exit 1; fi\ncase "$*" in *oidc*) echo '{"value":"test"}' ;; *) printf '%s' "$RESPONSE" ;; esac\n`, { mode: 0o755 });
  const output = path.join(root, 'output'), summary = path.join(root, 'summary'), counter = path.join(root, 'counter');
  const result = spawnSync('bash', [script], { encoding: 'utf8', env: { ...process.env,
    PATH: root + path.delimiter + process.env.PATH, REQUEST_ID: request, RESPONSE: response, FAIL: String(fail),
    COUNTER: counter, GITHUB_OUTPUT: output, GITHUB_STEP_SUMMARY: summary,
    ACTIONS_ID_TOKEN_REQUEST_TOKEN: 'fake', ACTIONS_ID_TOKEN_REQUEST_URL: 'https://oidc.invalid/?test=1',
    ENGINEERING_AUDIENCE: 'test', ENGINEERING_GATEWAY: 'https://gateway.invalid/' } });
  return { ...result, output: fs.existsSync(output) ? fs.readFileSync(output, 'utf8') : '',
    calls: fs.existsSync(counter) ? fs.readFileSync(counter, 'utf8').trim().split('\n').length : 0 };
}
test('exact manual request reaches existing claim gates without opening the automatic queue', t => {
  const r = run(t, 'a'.repeat(64)); assert.equal(r.status, 0); assert.equal(r.output, 'should_run=true\n'); assert.equal(r.calls, 0);
});
test('malformed manual request is rejected before any network or task claim', t => {
  const r = run(t, 'bad request'); assert.notEqual(r.status, 0); assert.equal(r.output, ''); assert.equal(r.calls, 0);
});
test('automatic mode preserves disabled flag and honors explicit boolean availability', t => {
  for (const enabled of [false, true]) {
    const r = run(t, '', JSON.stringify({ should_run: enabled }));
    assert.equal(r.status, 0); assert.equal(r.output, `should_run=${enabled}\n`); assert.equal(r.calls, 2);
  }
});
test('automatic network and malformed-response failures never enable work', t => {
  for (const [body, fail] of [['{}', false], ['{"should_run":"true"}', false], ['', true]]) {
    const r = run(t, '', body, fail); assert.equal(r.status, 0); assert.equal(r.output, 'should_run=false\n');
  }
});
