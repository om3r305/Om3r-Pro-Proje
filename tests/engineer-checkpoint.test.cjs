const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const { spawnSync } = require('node:child_process');
const { checkpoint } = require('../scripts/brian_engineer_checkpoint.cjs');
const runId = '11111111-1111-4111-8111-111111111111';
const sourcePath = 'supabase/functions/_shared/evolution_candidates/research.ts';
function git(root, ...args) {
  const r = spawnSync('git', args, { cwd: root, encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr); return r.stdout.trim();
}
function put(root, file, text) {
  fs.mkdirSync(path.dirname(path.join(root, file)), { recursive: true });
  fs.writeFileSync(path.join(root, file), text);
}
function repo(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-checkpoint-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  git(root, 'init', '-b', 'main'); git(root, 'config', 'user.name', 'test'); git(root, 'config', 'user.email', 'test@example.invalid');
  put(root, 'supabase/functions/_shared/evolution_engineer_guard.ts',
    fs.readFileSync('supabase/functions/_shared/evolution_engineer_guard.ts', 'utf8'));
  put(root, 'scripts/brian_engineer_checkpoint.cjs', fs.readFileSync('scripts/brian_engineer_checkpoint.cjs', 'utf8'));
  put(root, sourcePath, 'export const value=1;\n');
  git(root, 'add', '.'); git(root, 'commit', '-m', 'base');
  return { root, base: git(root, 'rev-parse', 'HEAD') };
}
test('checkpoint preserves modified, new and deleted files without moving HEAD or the caller index', t => {
  const { root, base } = repo(t);
  fs.unlinkSync(path.join(root, sourcePath));
  put(root, 'supabase/functions/_shared/evolution_candidates/new.ts', 'export const value=2;\n');
  put(root, 'docs/evolution_candidates/research.md', 'Incomplete draft');
  git(root, 'add', 'docs/evolution_candidates/research.md');
  const before = git(root, 'status', '--porcelain');
  const result = checkpoint(root, base, runId);
  assert.equal(result.state, 'INCOMPLETE_NOT_VALIDATED'); assert.equal(result.tested, false);
  assert.equal(result.changed_files, 3); assert.equal(git(root, 'rev-parse', 'HEAD'), base);
  assert.equal(git(root, 'status', '--porcelain'), before);
  assert.equal(git(root, 'rev-parse', `${result.commit_sha}^`), base);
  assert.equal(git(root, 'show', `${result.commit_sha}:docs/evolution_candidates/research.md`), 'Incomplete draft');
  assert.ok(!git(root, 'ls-tree', '-r', '--name-only', result.commit_sha).split('\n').includes(sourcePath));
  assert.match(git(root, 'show', '-s', '--format=%B', result.commit_sha), /Not compiled, tested, reviewed/);
});
test('a guarded incomplete checkpoint can be persisted and fetched separately from the canonical branch', t => {
  const { root, base } = repo(t), remote = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-checkpoint-remote-'));
  t.after(() => fs.rmSync(remote, { recursive: true, force: true }));
  git(root, 'init', '--bare', remote);
  git(root, 'remote', 'add', 'origin', remote);
  git(root, 'push', 'origin', 'HEAD:refs/heads/main');
  put(root, sourcePath, 'export const value=2;\n');
  const result = checkpoint(root, base, runId);
  git(root, 'push', 'origin', `${result.commit_sha}:refs/heads/${result.branch}`);
  assert.match(git(root, 'ls-remote', 'origin', 'refs/heads/main'), new RegExp('^' + base));
  assert.match(git(root, 'ls-remote', 'origin', `refs/heads/${result.branch}`), new RegExp('^' + result.commit_sha));
});
test('out-of-scope edits and a rewritten guard are rejected without executing candidate code', t => {
  const { root, base } = repo(t);
  put(root, 'supabase/functions/_shared/evolution_engineer_guard.ts', 'throw new Error("candidate code must not run");');
  assert.throws(() => checkpoint(root, base, runId), /CHECKPOINT_SCOPE_REJECTED/);
  assert.equal(git(root, 'rev-parse', 'HEAD'), base);
});
test('base guard rejects live execution even if only candidate paths change', t => {
  const { root, base } = repo(t);
  put(root, sourcePath, 'export const live_execution=true;\n');
  assert.throws(() => checkpoint(root, base, runId), /CHECKPOINT_GUARD_REJECTED/);
});
test('symlinks, including dangling links, cannot become checkpoint content', t => {
  const { root, base } = repo(t);
  fs.unlinkSync(path.join(root, sourcePath));
  fs.symlinkSync(path.join(root, 'not-present'), path.join(root, sourcePath));
  assert.throws(() => checkpoint(root, base, runId), /CHECKPOINT_SYMLINK_REJECTED/);
});
test('secret-looking content and oversized drafts are rejected', t => {
  const { root, base } = repo(t);
  put(root, sourcePath, 'const key="github_pat_' + 'A'.repeat(30) + '";');
  assert.throws(() => checkpoint(root, base, runId), /CHECKPOINT_SECRET_REJECTED/);
  put(root, sourcePath, 'x'.repeat(500001));
  assert.throws(() => checkpoint(root, base, runId), /CHECKPOINT_CONTENT_REJECTED/);
});
test('unchanged work is not claimed as a persisted checkpoint', t => {
  const { root, base } = repo(t);
  assert.equal(checkpoint(root, base, runId).state, 'NO_CHANGES');
});
test('checkpoint inspection does not execute configured filters or commit hooks', t => {
  const { root } = repo(t);
  put(root, '.gitattributes', '*.ts filter=trap\n');
  git(root, 'add', '.gitattributes'); git(root, 'commit', '-m', 'attributes');
  const base = git(root, 'rev-parse', 'HEAD');
  git(root, 'config', 'filter.trap.clean', 'touch filter-marker; cat');
  put(root, '.git/hooks/pre-commit', '#!/bin/sh\ntouch hook-marker\nexit 1\n');
  fs.chmodSync(path.join(root, '.git/hooks/pre-commit'), 0o755);
  put(root, sourcePath, 'export const value=3;\n');
  assert.equal(checkpoint(root, base, runId).state, 'INCOMPLETE_NOT_VALIDATED');
  assert.equal(fs.existsSync(path.join(root, 'filter-marker')), false);
  assert.equal(fs.existsSync(path.join(root, 'hook-marker')), false);
});
test('the actual failure handler pushes an incomplete branch and records its persisted state', t => {
  const { root, base } = repo(t), remote = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-checkpoint-handler-'));
  t.after(() => fs.rmSync(remote, { recursive: true, force: true }));
  git(root, 'init', '--bare', remote); git(root, 'remote', 'add', 'origin', remote);
  put(root, sourcePath, 'export const value=4;\n');
  const prefix = fs.readFileSync('scripts/brian_engineer_runner.sh', 'utf8').split('\non_error() {')[0]
    .replace('PROVIDER_STATUS_FILE="/tmp/brian-provider-status.json"', 'PROVIDER_STATUS_FILE="$TEST_ROOT/status.json"')
    .replace('PROVIDER_ATTEMPTS_FILE="/tmp/brian-provider-attempts.jsonl"', 'PROVIDER_ATTEMPTS_FILE="$TEST_ROOT/attempts.jsonl"');
  const harness = prefix + `
RUN_ID="$TEST_RUN_ID"
BASE_SHA="$TEST_BASE_SHA"
CURRENT_STAGE=CODE
record_event() { printf '%s' "$4" > "$TEST_ROOT/receipt.json"; }
record_blocked 75
`;
  const result = spawnSync('bash', ['-c', harness], { cwd: root, encoding: 'utf8', env: { ...process.env,
    TEST_ROOT: remote, TEST_RUN_ID: runId, TEST_BASE_SHA: base } });
  assert.equal(result.status, 0, result.stderr);
  const receipt = JSON.parse(fs.readFileSync(path.join(remote, 'receipt.json'), 'utf8')).incomplete_checkpoint;
  assert.equal(receipt.persisted, true); assert.equal(receipt.tested, false);
  assert.equal(receipt.state, 'INCOMPLETE_NOT_VALIDATED');
  assert.match(git(root, 'ls-remote', 'origin', `refs/heads/${receipt.branch}`), new RegExp('^' + receipt.commit_sha));
  assert.equal(git(root, 'rev-parse', 'HEAD'), base);
});
