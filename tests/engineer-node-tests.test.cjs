const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtempSync, writeFileSync, rmSync, symlinkSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { selectTests, baseline } = require('../scripts/brian_engineer_node_tests.cjs');
const runner = path.resolve(__dirname, '../scripts/brian_engineer_node_tests.cjs');
function fixture(t) {
  const dir = mkdtempSync(path.join(tmpdir(), 'brian-node-check-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
}
function execute(dir, files) {
  const env = { ...process.env };
  delete env.NODE_TEST_CONTEXT;
  return spawnSync(process.execPath, ['-e',
    'const r=require(process.argv[1]);r.runTests(JSON.parse(process.argv[2]),process.argv[3]);',
    runner, JSON.stringify(files), dir], { encoding: 'utf8', timeout: 15_000, env });
}
test('selection includes unchanged baseline, added/renamed test formats, spaces, and deduplicates', () => {
  const selected = selectTests(['tests/new.test.cjs', 'tests/a b.spec.mjs', 'src/check.test.js', baseline[0], 'src/source.js']);
  assert.ok(baseline.every(file => selected.includes(file)));
  assert.ok(selected.includes('tests/a b.spec.mjs'));
  assert.ok(selected.includes('src/check.test.js'));
  assert.equal(selected.filter(file => file === baseline[0]).length, 1);
  assert.ok(!selected.includes('src/source.js'));
});
test('an actual passing test executes successfully with spaces in its filename', t => {
  const dir = fixture(t);
  writeFileSync(path.join(dir, 'pass space.test.cjs'), "require('node:test')('executed',()=>{});");
  const result = execute(dir, ['pass space.test.cjs']);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /# pass 1/);
});
test('a syntactically valid failing assertion blocks the gate', t => {
  const dir = fixture(t);
  writeFileSync(path.join(dir, 'fail.test.cjs'), "require('node:test')('regression',()=>{require('node:assert/strict').equal(1,2)});");
  const result = execute(dir, ['fail.test.cjs']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Node test failed/);
});
test('an all-skipped test file cannot count as passing evidence', t => {
  const dir = fixture(t);
  writeFileSync(path.join(dir, 'skip.test.cjs'), "require('node:test').skip('unavailable',()=>{});");
  const result = execute(dir, ['skip.test.cjs']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /No passing tests executed/);
});
test('missing and symlinked files fail closed', t => {
  const dir = fixture(t);
  writeFileSync(path.join(dir, 'real.test.cjs'), "require('node:test')('ok',()=>{});");
  symlinkSync(path.join(dir, 'real.test.cjs'), path.join(dir, 'link.test.cjs'));
  for (const file of ['missing.test.cjs', 'link.test.cjs', '../escape.test.cjs']) {
    assert.notEqual(execute(dir, [file]).status, 0);
  }
});
