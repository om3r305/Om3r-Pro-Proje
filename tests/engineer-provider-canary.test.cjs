const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { args, classify, validateWorkspace, spec, stub } = require('../scripts/brian_engineer_canary.cjs');
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-canary-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.writeFileSync(path.join(root, 'SPEC.md'), spec);
  fs.writeFileSync(path.join(root, 'candidate.cjs'), stub + '// changed');
  return root;
}
test('generation exposes only file operations and one writable target', () => {
  const options = args();
  assert.ok(options.includes('--allow-tool=read,write(candidate.cjs)'));
  assert.ok(options.includes('--deny-tool=shell'));
  assert.ok(options.includes('--deny-tool=url'));
  assert.ok(options.includes('--disable-builtin-mcps'));
  assert.ok(options.includes('--no-custom-instructions'));
  assert.ok(!options.join(' ').includes('allow-all'));
  assert.ok(!options.join(' ').includes('available-tools=bash'));
});
test('provider failures become fixed diagnostic classes', () => {
  for (const [text, rc, expected] of [['monthly quota exceeded', 1, 'QUOTA_EXHAUSTED'],
    ['401 token secret-text', 1, 'AUTH'], ['429 too many requests', 1, 'RATE_LIMIT'],
    ['', 124, 'TIMEOUT'], ['unknown option --flag', 1, 'CLI_CONFIGURATION']]) {
    assert.equal(classify(text, rc), expected);
  }
});
test('scope verification rejects unchanged code, extra files and modified specifications', t => {
  const root = fixture(t);
  assert.ok(validateWorkspace(root, spec).length > 0);
  fs.writeFileSync(path.join(root, 'candidate.cjs'), stub);
  assert.throws(() => validateWorkspace(root, spec), /NO_CODE_CHANGE/);
  fs.writeFileSync(path.join(root, 'candidate.cjs'), stub + '// changed');
  fs.writeFileSync(path.join(root, 'extra.txt'), 'extra');
  assert.throws(() => validateWorkspace(root, spec), /OUT_OF_SCOPE_WRITE/);
  fs.unlinkSync(path.join(root, 'extra.txt'));
  fs.writeFileSync(path.join(root, 'SPEC.md'), 'altered');
  assert.throws(() => validateWorkspace(root, spec), /SPEC_MUTATED/);
});
test('symlinked and oversized candidates cannot enter the evaluator', t => {
  const root = fixture(t), candidate = path.join(root, 'candidate.cjs');
  fs.unlinkSync(candidate); fs.symlinkSync(path.join(root, 'SPEC.md'), candidate);
  assert.throws(() => validateWorkspace(root, spec), /NON_REGULAR_FILE/);
  fs.unlinkSync(candidate); fs.writeFileSync(candidate, 'x'.repeat(32001));
  assert.throws(() => validateWorkspace(root, spec), /CANDIDATE_TOO_LARGE/);
});
const reference = `module.exports.assessEvidence = (rows, at, max=120000) => {
 if (!Array.isArray(rows) || !Number.isFinite(at) || !Number.isFinite(max) || max < 0) return [];
 const result=new Map();
 for(const r of rows){
  if(!r || typeof r.asset_id!=='string' || !r.asset_id.length || typeof r.observed_at!=='string' || !Number.isFinite(r.price) || r.price<=0 || !Number.isFinite(r.cost_bps) || r.cost_bps<0)continue;
  const t=Date.parse(r.observed_at),age=at-t;
  if(!Number.isFinite(t)||age<0||age>max)continue;
  if(!result.has(r.asset_id)||age<result.get(r.asset_id).age_ms)result.set(r.asset_id,{asset_id:r.asset_id,observed_at:r.observed_at,price:r.price,cost_bps:r.cost_bps,age_ms:age});
 }
 return [...result.values()].sort((a,b)=>a.asset_id<b.asset_id?-1:a.asset_id>b.asset_id?1:0);
};`;
function evaluate(root, source) {
  const candidate = path.join(root, 'candidate.cjs'); fs.writeFileSync(candidate, source);
  const env = { ...process.env, BRIAN_CANARY_CANDIDATE: candidate }; delete env.NODE_TEST_CONTEXT;
  return spawnSync(process.execPath, ['--test', '--test-reporter=tap', path.resolve(__dirname, '../scripts/brian_engineer_canary_eval.cjs')],
    { env, encoding: 'utf8', timeout: 10000 });
}
test('independent evaluator accepts a known implementation and rejects future-data contamination', t => {
  const root = fixture(t);
  const good = evaluate(root, reference);
  assert.equal(good.status, 0, good.stdout + good.stderr); assert.match(good.stdout, /# pass 8/);
  const bad = evaluate(root, reference.replace('age<0||', ''));
  assert.notEqual(bad.status, 0); assert.match(bad.stdout, /# fail [1-9]/);
});
test('independent evaluator rejects missing-cost substitution and placeholder output', t => {
  const root = fixture(t);
  assert.notEqual(evaluate(root, reference.replace('!Number.isFinite(r.cost_bps) ||', '')).status, 0);
  assert.notEqual(evaluate(root, 'module.exports.assessEvidence=()=>[];').status, 0);
});
