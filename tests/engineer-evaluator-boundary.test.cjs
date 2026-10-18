const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), vm = require('node:vm');
const { stripTypeScriptTypes } = require('node:module');
const { baseline } = require('../scripts/brian_engineer_node_tests.cjs');
const source = stripTypeScriptTypes(fs.readFileSync('supabase/functions/_shared/evolution_engineer_guard.ts', 'utf8')).replace(/^export /gm, '');
const validate = vm.runInNewContext(source + ';validateEngineerChangeSet');
test('autonomous candidates cannot rewrite any operator-owned Node baseline test', () => {
  for (const file of baseline) {
    const result = validate({ changedPaths: [file], patchBytes: 1 });
    assert.equal(result.valid, false, file);
    assert.ok(result.reasons.some(r => r.includes('control-plane')), file);
  }
});
test('new candidate unit, replay and stress tests remain writable', () => {
  const result = validate({ changedPaths: [
    'supabase/functions/_shared/evolution_candidates/research.test.ts',
    'tests/evolution_engineer/replay/research.test.ts',
    'tests/evolution_engineer/stress/research.test.ts',
  ], patchBytes: 100 });
  assert.equal(result.valid, true);
});
test('the fixed Deno regression suite invoked by the runner is also immutable to candidates', () => {
  const runner = fs.readFileSync('scripts/brian_engineer_runner.sh', 'utf8');
  const paths = [...runner.matchAll(/supabase\/functions\/_shared\/evolution_[a-z_]+\.test\.ts/g)].map(x => x[0]);
  assert.ok(paths.length >= 9);
  for (const file of paths) assert.equal(validate({ changedPaths: [file], patchBytes: 1 }).valid, false, file);
});
test('explicit operator maintenance can update baseline tests', () => {
  const result = validate({ changedPaths: baseline, patchBytes: 100, allowControlPlane: true });
  // The independent max-file limit still applies; use a single maintenance change.
  assert.ok(result.reasons.some(r => r.includes('files changed')));
  assert.equal(validate({ changedPaths: [baseline[0]], patchBytes: 100, allowControlPlane: true }).valid, true);
});
