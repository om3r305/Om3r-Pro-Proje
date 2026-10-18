#!/usr/bin/env node
'use strict';
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const { spawnSync } = require('node:child_process');
const { stripTypeScriptTypes } = require('node:module');
const vm = require('node:vm');
const prefixes = ['supabase/functions/_shared/evolution_candidates/',
  'tests/evolution_engineer/replay/', 'tests/evolution_engineer/stress/', 'docs/evolution_candidates/'];
function checkpoint(cwd, base, runId) {
  if (!/^[a-f0-9]{40}$/i.test(base) || !/^[a-f0-9-]{36}$/i.test(runId)) throw Error('INVALID_IDENTITY');
  const root = fs.realpathSync(cwd), temp = fs.mkdtempSync(path.join(os.tmpdir(), 'brian-checkpoint-'));
  const filterFlags = [];
  function git(args, input, index = false) {
    const result = spawnSync('git', ['-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false',
      '-c', 'commit.gpgsign=false', '-c', 'user.name=brian-engineer[bot]',
      '-c', 'user.email=brian-engineer@users.noreply.github.com', ...filterFlags, ...args],
      { cwd: root, input, encoding: 'utf8', timeout: 30000, maxBuffer: 2 * 1024 * 1024,
        env: { ...process.env, ...(index ? { GIT_INDEX_FILE: path.join(temp, 'index') } : {}) } });
    if (result.error || result.status !== 0) throw Error('CHECKPOINT_GIT_FAILED');
    return result.stdout;
  }
  try {
    // Disable configured conversion commands before inspecting worktree changes.
    const config = spawnSync('git', ['config', '--name-only', '--get-regexp', '^filter\\..*\\.(clean|smudge|process|required)$'],
      { cwd: root, encoding: 'utf8', timeout: 10000, maxBuffer: 65536 });
    if (config.error || ![0, 1].includes(config.status)) throw Error('CHECKPOINT_FILTER_CONFIGURATION');
    for (const key of config.stdout.split('\n').filter(Boolean)) {
      const driver = key.match(/^filter\.([A-Za-z0-9_.-]+)\.(clean|smudge|process|required)$/)?.[1];
      if (!driver) throw Error('CHECKPOINT_FILTER_CONFIGURATION');
      filterFlags.push('-c', `filter.${driver}.clean=cat`, '-c', `filter.${driver}.smudge=cat`,
        '-c', `filter.${driver}.process=`, '-c', `filter.${driver}.required=false`);
    }
    if (git(['rev-parse', 'HEAD']).trim() !== base) throw Error('HEAD_CHANGED');
    const files = [...new Set([
      ...git(['diff', '--no-ext-diff', '--name-only', '-z', base, '--']).split('\0'),
      ...git(['ls-files', '--others', '--exclude-standard', '-z']).split('\0'),
    ].filter(Boolean))];
    if (!files.length) return { state: 'NO_CHANGES', base_sha: base };
    if (files.length > 12) throw Error('CHECKPOINT_TOO_MANY_FILES');
    const snapshots = []; let totalBytes = 0;
    for (const file of files) {
      if (!prefixes.some(prefix => file.startsWith(prefix)) || !/^[A-Za-z0-9_./-]+$/.test(file) ||
          file.split('/').includes('..') || !/\.(ts|tsx|js|cjs|mjs|md|json)$/.test(file)) throw Error('CHECKPOINT_SCOPE_REJECTED');
      let cursor = root;
      const parts = file.split('/');
      for (const part of parts) {
        cursor = path.join(cursor, part);
        try {
          if (fs.lstatSync(cursor).isSymbolicLink()) throw Error('CHECKPOINT_SYMLINK_REJECTED');
        } catch (error) { if (error.code !== 'ENOENT') throw error; }
      }
      const absolute = path.join(root, file);
      if (!fs.existsSync(absolute)) { snapshots.push({ file, deleted: true }); continue; }
      if (!fs.lstatSync(absolute).isFile()) throw Error('CHECKPOINT_NON_REGULAR_FILE');
      const bytes = fs.readFileSync(absolute);
      totalBytes += bytes.length;
      if (totalBytes > 500000 || bytes.includes(0)) throw Error('CHECKPOINT_CONTENT_REJECTED');
      const text = bytes.toString('utf8');
      if (/-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:github_pat_|gh[pousr]_|sk_live_|sb_secret_)[A-Za-z0-9_]{20,}/.test(text)) throw Error('CHECKPOINT_SECRET_REJECTED');
      snapshots.push({ file, bytes, mode: fs.statSync(absolute).mode & 0o111 ? '100755' : '100644' });
    }
    // Evaluate against the operator's guard from the exact base, never candidate-modified code.
    const trusted = git(['show', `${base}:supabase/functions/_shared/evolution_engineer_guard.ts`]);
    const validate = vm.runInNewContext(stripTypeScriptTypes(trusted).replace(/^export /gm, '') + ';validateEngineerChangeSet');
    git(['read-tree', base], undefined, true);
    for (const item of snapshots) {
      if (item.deleted) git(['update-index', '--force-remove', '--', item.file], undefined, true);
      else {
        const blob = git(['hash-object', '-w', '--no-filters', '--stdin'], item.bytes).trim();
        git(['update-index', '--add', '--cacheinfo', item.mode, blob, item.file], undefined, true);
      }
    }
    const tree = git(['write-tree'], undefined, true).trim();
    const patch = git(['diff', '--no-ext-diff', '--no-textconv', '--binary', base, tree, '--']);
    const additions = git(['diff', '--no-ext-diff', '--no-textconv', '--unified=0', base, tree, '--'])
      .split('\n').filter(line => line.startsWith('+') && !line.startsWith('+++')).join('\n');
    const checked = validate({ changedPaths: files, patchBytes: Buffer.byteLength(patch), addedText: additions });
    if (!checked.valid) throw Error('CHECKPOINT_GUARD_REJECTED');
    const commit = git(['commit-tree', tree, '-p', base],
      `INCOMPLETE Brian checkpoint ${runId}\n\nNot compiled, tested, reviewed, approved or deployable.\n`).trim();
    return { state: 'INCOMPLETE_NOT_VALIDATED', commit_sha: commit, base_sha: base, changed_files: files.length,
      branch: `brian-engineer-checkpoint/${runId}`, compiled: false, tested: false, approved: false, live_execution: false };
  } finally { fs.rmSync(temp, { recursive: true, force: true }); }
}
if (require.main === module) {
  try { console.log(JSON.stringify(checkpoint(process.cwd(), process.argv[2], process.argv[3]))); }
  catch (error) { console.log(JSON.stringify({ state: 'REJECTED', reason: /^[A-Z_]+$/.test(error.message) ? error.message : 'CHECKPOINT_LOCAL_ERROR' })); process.exitCode = 1; }
}
module.exports = { checkpoint };
