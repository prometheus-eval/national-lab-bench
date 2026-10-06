"""420 fresh Opus-5.5/max evaluations, eight workers, model-level checkpoints.

Sibling of run_codex_judge.py. It imports that runner and reuses, unchanged, its
job list, paper mapping, source verification, snapshot rules, workspace staging,
checkpoint validation and summaries, plus the same separated-rubrics-evidence-v1
prompt, output schema, validator and frozen deterministic grader. Only the judge
engine differs: a Claude Code print session (claude -p) instead of a Codex thread.

Engine settings mirror the Codex run: one ephemeral session per rubric stage, the
same staged workspace as cwd, schema-enforced output, no web, read-only evidence.
Blinding is stronger than the Codex run's prompt-level blinding: Read/Grep/Glob are
limited to the stage workspace and its snapshot, and Bash runs in the OS sandbox
with those two paths write-denied, no network, and the batch root, SSD, ~/.claude
and both repositories read-denied except the workspace and snapshot themselves.

Snapshots live with the batch on the internal disk: the full set is ~216 GiB and
the SSD has too little free space for a second copy. Any stage failure or changed
frozen input pauses scheduling, and resume is explicit. Completed, validated stages
are never repeated.

Revision 2 (two subscriptions): worker slots are assigned to Claude accounts, each a
separate CLAUDE_CONFIG_DIR login with its own usage windows. A job starts on an
account only if its latest reported utilization plus the projected cost of every
stage already in flight there, plus the new job's two stages, stays under that
account's cap, so concurrency tapers instead of overshooting. An account at its cap
idles until its window resets (then it is probed and resumes by itself); a session
refused for usage limits is requeued, not failed. Judging is unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import judge_protocol as protocol
import judge_common as base
import run_codex_judge as astra

REPO = Path(__file__).resolve().parents[2]
PREFIX = 'claude_code_judge_'
JUDGE_MODEL, EFFORT = 'claude-opus-5-5', 'max'
CONCURRENCY = 8                     # revision 1 (single account); kept for the original manifest
QUOTA_PAUSE = 0.90
SLOTS_PER_ACCOUNT = 4                # 8 sessions total: throughput is quota-bound (a 5-hour window fills in
                                     # ~4 h even at 4 per account), and the laptop has 16 GiB of RAM
# Extra Claude subscriptions, one CLAUDE_CONFIG_DIR each: SOG_CLAUDE_CONFIG_DIRS=dir1,dir2
# (acct1 = the default login). Caps are fractions of the 5-hour and 7-day windows.
ACCOUNTS = tuple([dict(name='acct1', config_dir=None, caps=dict(five_hour=0.80, seven_day=0.80))] +
                 [dict(name=f'acct{i}', config_dir=d, caps=dict(five_hour=0.95, seven_day=0.95))
                  for i, d in enumerate(filter(None, os.environ.get('SOG_CLAUDE_CONFIG_DIRS', '').split(',')), 2)])
# Conservative utilization per rubric stage, from the pilot (+6 pts 5h, +1 pt 7d per two-stage evaluation).
STAGE_COST = dict(five_hour=0.035, seven_day=0.007)
MAX_QUOTA_REQUEUES = 3
EXIT_QUOTA = 75                      # EX_TEMPFAIL: the job was requeued for quota, not failed
PROBE_MIN_INTERVAL_S = 600
STAGE_TIMEOUT_S = 4 * 3600          # safety net against a hung session; a timeout pauses, it never retries
TOOLS = 'Read,Grep,Glob,Bash'
CODE_FILES = ('run_claude_code_judge.py', 'run_codex_judge.py', 'judge_common.py', 'judge_protocol.py')
read, write, digest, utc = base.read, base.write, base.digest, base.utc


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def claude_bin():
    found = shutil.which('claude')
    if not found:
        raise RuntimeError('claude CLI not found on PATH')
    return Path(found).resolve()


def claude_version(binary):
    return subprocess.run([str(binary), '--version'], capture_output=True, text=True,
                          timeout=60).stdout.strip()


def judge_env(acct=None):
    env = dict(os.environ, DISABLE_AUTOUPDATER='1', PYTHONDONTWRITEBYTECODE='1')
    # OAuth subscription runtime, as the Codex run uses its subscription; never an API key.
    for k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDECODE',
              'CLAUDE_CODE_ENTRYPOINT', 'CLAUDE_CONFIG_DIR'):
        env.pop(k, None)
    if acct and acct.get('config_dir'):
        env['CLAUDE_CONFIG_DIR'] = acct['config_dir']
    return env


class QuotaWait(Exception):
    """The account is at its cap or a session was refused for usage limits: requeue, do not fail."""


def accounts_of(m):
    return m.get('accounts') or [dict(name='acct1', config_dir=None,
                                      caps=dict(five_hour=QUOTA_PAUSE, seven_day=QUOTA_PAUSE))]


def slots_per_account(m):
    return m.get('slots_per_account', m['concurrency'])


def slot_account(m, slot):
    return accounts_of(m)[slot // slots_per_account(m)]


def fmt_ts(ts):
    return datetime.fromtimestamp(ts).strftime('%a %H:%M') if ts else '?'



def pause(root, reason):
    """Single-writer pause record; the batch is on the internal disk, so no SSD check here."""
    try:
        with (root / 'PAUSE.json').open('x') as f:
            json.dump(dict(reason=reason, utc=utc()), f)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        pass

# --------------------------------------------------------------------------- quota
def usage_from_events(events):
    """Latest subscription utilization reported in a stream-json session."""
    latest = None
    for e in events:
        if e.get('type') == 'rate_limit_event':
            latest = e.get('rate_limit_info') or latest
    return latest


def quota_reason(info):
    if not info:
        return 'Subscription quota cannot be established'
    if info.get('status') not in (None, 'allowed', 'allowed_warning'):
        return f'service reports rate-limit status {info.get("status")!r}'
    for name, w in (info.get('unifiedWindows') or {}).items():
        u = (w or {}).get('utilization')
        if u is not None and u >= QUOTA_PAUSE:
            return f'{name}: {u:.0%} used, at or above the {QUOTA_PAUSE:.0%} safeguard'
    return None


def usage_path(root, name):
    return root / f'usage_{name}.json'


def read_usage(root, name):
    p = usage_path(root, name)
    return read(p) if p.exists() else None


def record_usage(root, name, info, source):
    if not info:
        return
    with astra.wait_lock(root / f'usage_{name}.lock'):
        write(usage_path(root, name), dict(info=info, source=source, utc=utc(), epoch=time.time()))


def is_limited(info):
    if not info:
        return False
    if info.get('status') not in (None, 'allowed', 'allowed_warning'):
        return True
    return any((w or {}).get('utilization', 0) >= 1.0 for w in (info.get('unifiedWindows') or {}).values())


def account_block(root, acct, remaining_stages):
    """None if a new two-stage job fits under the account's caps, else the reason it does not."""
    rec = read_usage(root, acct['name'])
    if not rec:
        return 'no usage reading yet'
    info = rec['info']
    if info.get('status') not in (None, 'allowed', 'allowed_warning'):
        return f'service status {info.get("status")!r}'
    for window, w in (info.get('unifiedWindows') or {}).items():
        cap, u = acct['caps'].get(window), (w or {}).get('utilization')
        if cap is None or u is None:
            continue
        projected = u + (remaining_stages + 2) * STAGE_COST.get(window, STAGE_COST['five_hour'])
        if projected > cap:
            return (f'{window} {u:.0%} + {remaining_stages} stage(s) in flight -> {projected:.0%} > cap {cap:.0%}; '
                    f'resets {fmt_ts(w.get("resetsAt"))}')
    return None


def stage_backstop(root, acct):
    """Before each stage: yield (requeue) if the account is clearly past its cap."""
    rec = read_usage(root, acct['name'])
    if not rec:
        return
    info = rec['info']
    if is_limited(info):
        raise QuotaWait(f'{acct["name"]}: service reports a usage limit')
    for window, w in (info.get('unifiedWindows') or {}).items():
        cap, u = acct['caps'].get(window), (w or {}).get('utilization')
        if cap is not None and u is not None and u >= min(1.0, cap + 0.04):
            raise QuotaWait(f'{acct["name"]}: {window} at {u:.0%}, past cap {cap:.0%}')


def needs_probe(root, acct, inflight):
    """Refresh an idle account's reading once a window has reset, or when it is stale."""
    rec = read_usage(root, acct['name'])
    if rec is None:
        return True
    age = time.time() - rec.get('epoch', 0)
    if age < PROBE_MIN_INTERVAL_S:
        return False
    if any((w or {}).get('resetsAt', 0) <= time.time() for w in (rec['info'].get('unifiedWindows') or {}).values()):
        return True
    return inflight == 0 and age > 1800


def probe_quota(root, binary, acct):
    """Minimal session to obtain a fresh rate-limit reading for one account."""
    probe = root / 'preflight'
    probe.mkdir(exist_ok=True)
    out = subprocess.run([str(binary), '-p', 'Reply with the single word: ready',
        '--model', JUDGE_MODEL, '--effort', 'low', '--tools', '',
        '--no-session-persistence', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
        '--settings', json.dumps(dict(autoMemoryEnabled=False, disableAllHooks=True,
                                      disableClaudeAiConnectors=True)),
        '--output-format', 'stream-json', '--verbose'],
        cwd=probe, stdin=subprocess.DEVNULL, capture_output=True, text=True, env=judge_env(acct), timeout=600)
    events = [json.loads(l) for l in out.stdout.splitlines() if l.strip().startswith('{')]
    info = usage_from_events(events)
    write(root / f'preflight_usage_{acct["name"]}_{stamp()}.json',
          dict(account=acct['name'], info=info, returncode=out.returncode, stderr_tail=out.stderr[-2000:]))
    record_usage(root, acct['name'], info, 'probe')
    return info


# --------------------------------------------------------------------------- prepare
def prepare():
    base.mounted()
    # Identical to the Codex judge: verify all 105 identities and task provenance first.
    sources = astra.discover_sources()

    base.JUDGE_RUNS.mkdir(parents=True, exist_ok=True)
    root = base.JUDGE_RUNS / f'{PREFIX}{stamp()}'
    root.mkdir(exist_ok=False)
    for name in CODE_FILES:
        shutil.copy2(Path(__file__).parent / name, root / name)
    material_files = []
    for n in range(1, 16):
        material_files += [f'tasks/paper{n}/task_spec.md']
        material_files += [f'src/grader/eval_artifacts/paper{n}/{name}' for name in (*base.KINDS.values(), 'grading.py')]
    for rel in material_files:
        dest = root / 'frozen' / rel.removeprefix('src/grader/')   # frozen/{tasks,eval_artifacts}/paper<i>/
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
    policies = {}
    for n in range(1, 16):
        folder = root / f'frozen/eval_artifacts/paper{n}'
        policy = base.grader(folder).load_policy()
        policies[f'P{n}'] = dict(revision=policy['rubric_revision'], rubric_sha256=policy['rubric_sha256'])
        inp = root / f'inputs/P{n}'
        inp.mkdir(parents=True)
        for scope, filename in base.KINDS.items():
            rubric = read(folder / filename)
            context = policy['context_fields'] if scope == 'direction' else {}
            write(inp / f'{scope}.schema.json', protocol.schema(rubric, context))
            write(inp / f'{scope}.context_fields.json', context)
            (inp / f'{scope}.prompt.txt').write_text(protocol.prompt(scope, len(protocol.leaves(rubric)), context))
    for sid, source in sources.items():
        inp = root / 'source_inputs' / sid
        inp.mkdir(parents=True)
        shutil.copy2(source['reference'], inp / 'paper.md')
        if digest(inp / 'paper.md') != source['reference_sha256']:
            raise RuntimeError(f'Reference changed during preparation: {sid}')
    fixed = {name: digest(root / name) for name in CODE_FILES}
    for dirname in ('frozen', 'inputs', 'source_inputs'):
        fixed.update({f'{dirname}/{p}': h for p, h in base.hashes(root / dirname).items()})

    # Byte-for-byte parity with the Codex batch's judge inputs, where one exists.
    parity = dict(checked=False)
    runs = sorted(base.JUDGE_RUNS.glob(f'{astra.PREFIX}*'))
    if runs:
        other = read(runs[-1] / 'manifest.json')
        shared = {k: v for k, v in fixed.items() if k.split('/')[0] in ('frozen', 'inputs', 'source_inputs')}
        mismatched = [k for k, v in shared.items() if other['fixed_hashes'].get(k) != v]
        parity = dict(checked=True, astra_batch=str(runs[-1]), compared_files=len(shared),
                      identical=len(shared) - len(mismatched), mismatched=mismatched,
                      same_prompt_version=other.get('prompt_version') == protocol.VERSION,
                      same_jobs=[j['id'] for j in other['jobs']] == [j['id'] for j in astra.make_jobs()])
        if mismatched or not parity['same_prompt_version'] or not parity['same_jobs']:
            raise RuntimeError(f'Judge inputs differ from the Codex batch: {mismatched[:5]}')
        print(f'PARITY: {len(shared)} judge-input files identical to {runs[-1].name}', flush=True)

    jobs = astra.make_jobs()
    for job in jobs:
        cell = root / job['id']
        cell.mkdir(parents=True)
        (cell / 'verdicts').mkdir()
        write(cell / 'state.json', dict(status='pending', job=job, stages={}, updated_utc=utc()))
    binary = claude_bin()
    manifest = dict(created_utc=utc(), model=JUDGE_MODEL, judge_model=JUDGE_MODEL, effort=EFFORT,
        engine='claude-code-print', claude_bin=str(binary), claude_sha256=digest(binary),
        claude_version=claude_version(binary), tools=TOOLS,
        total_submissions=105, repeats=4, total_evaluations=420, total_rubric_sessions=840,
        concurrency=SLOTS_PER_ACCOUNT * len(ACCOUNTS), slots_per_account=SLOTS_PER_ACCOUNT,
        accounts=[dict(a) for a in ACCOUNTS], stage_cost_estimate=STAGE_COST,
        model_order=[m[0] for m in astra.MODELS], jobs=jobs, sources=sources,
        fixed_hashes=fixed, policies=policies, prompt_version=protocol.VERSION,
        python=sys.executable, quota_pause_fraction=QUOTA_PAUSE, automatic_retries=0,
        stage_timeout_s=STAGE_TIMEOUT_S, parity_with_codex_batch=parity,
        isolation=('Separate ephemeral Claude Code print sessions per stage/repeat; shared per-submission '
                   'physical snapshot, prior judgments excluded. Read/Grep/Glob limited to the stage '
                   'workspace and its snapshot; Bash in the OS sandbox, write-denied there, no network, '
                   'and batch root/SSD/~/.claude/repositories read-denied except workspace and snapshot.'),
        scope='All 105 submissions (7 agents x 15 papers). Four fresh assessments each, with the '
              'Codex judge\'s exact inputs; only the judge engine differs.',
        snapshot_exclusions=sorted(astra.SKIP_DIRS) + ['eval_results*', '*.pyc', '._*'],
        snapshot_location='with the batch')
    write(root / 'manifest.json', manifest)
    write(root / 'status.json', dict(status='prepared', completed=0, total=420, updated_utc=utc()))
    print(f'BATCH_DIRECTORY={root}', flush=True)
    return root


def verify_frozen(root, m):
    for rel, expected in m['fixed_hashes'].items():
        if digest(root / rel) != expected:
            raise RuntimeError(f'Frozen evaluation input changed: {rel}')
    if digest(Path(m['claude_bin'])) != m['claude_sha256']:
        raise RuntimeError('Claude Code binary changed; review before launching more sessions')


# --------------------------------------------------------------------------- engine
def stage_settings(root, ws, snap, config_dirs=()):
    """`snap` is the staged proposal, or the list of staged view directories (revision 3)."""
    snaps = [str(d) for d in (snap if isinstance(snap, (list, tuple)) else [snap])]
    blind = [str(root), str(base.SUBMISSIONS), str(Path.home() / '.claude'), str(REPO)]
    blind += [d for d in config_dirs if d and d not in blind]
    return dict(autoMemoryEnabled=False, disableAllHooks=True, disableClaudeAiConnectors=True,
        sandbox=dict(enabled=True, failIfUnavailable=True, autoAllowBashIfSandboxed=True,
                     allowUnsandboxedCommands=False, network=dict(allowedDomains=[]),
                     filesystem=dict(denyWrite=[str(ws), *snaps], denyRead=blind,
                                     allowRead=[str(ws), *snaps])))


def run_stage(root, m, cell, job, scope, ws, attempt, spec, prompt_text, acct=None):
    acct = acct or accounts_of(m)[0]
    if m.get('view_version') == 2:
        snaps = [d.resolve() for d in astra.view_dirs(root, job['source_id'])]
    else:
        snaps = [(root / 'snapshots' / job['source_id'] / 'proposal').resolve()]
    settings = stage_settings(root, ws.resolve(), snaps, [a.get('config_dir') for a in accounts_of(m)])
    cmd = [m['claude_bin'], '-p', prompt_text, '--model', m['judge_model'], '--effort', m['effort'],
           '--json-schema', json.dumps(spec, ensure_ascii=False),
           '--tools', m['tools'], '--permission-mode', 'dontAsk', '--permission-prompts', 'none',
           *[a for d in snaps for a in ('--add-dir', str(d))], '--no-session-persistence',
           '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
           '--settings', json.dumps(settings),
           '--output-format', 'stream-json', '--verbose']
    write(attempt / 'command.json', dict(argv=[a if len(a) < 4000 else f'<{len(a)} chars>' for a in cmd],
                                         cwd=str(ws), account=acct['name'], settings=settings))
    started = time.monotonic()
    with (attempt / 'events.jsonl').open('x') as out, (attempt / 'stderr.log').open('x') as err:
        proc = subprocess.run(cmd, cwd=ws, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                              env=judge_env(acct), timeout=STAGE_TIMEOUT_S)
    events = []
    for line in (attempt / 'events.jsonl').read_text().splitlines():
        if line.strip().startswith('{'):
            events.append(json.loads(line))
    info = usage_from_events(events)
    record_usage(root, acct['name'], info, f'{job["id"]}/{scope}')
    write(cell / 'usage_limits.json', dict(account=acct['name'], info=info, utc=utc()))
    init = next((e for e in events if e.get('type') == 'system' and e.get('subtype') == 'init'), {})
    result = next((e for e in reversed(events) if e.get('type') == 'result'), None)
    meta = dict(account=acct['name'], returncode=proc.returncode, wall_s=round(time.monotonic() - started, 1),
                init_model=init.get('model'), init_tools=init.get('tools'),
                init_mcp=init.get('mcp_servers'), claude_code_version=init.get('claude_code_version'),
                session_id=init.get('session_id'), rate_limit=info)
    if result:
        meta.update({k: result.get(k) for k in ('subtype', 'is_error', 'duration_ms', 'num_turns',
                                                'total_cost_usd', 'usage', 'modelUsage', 'permission_denials')})
    write(attempt / 'turn.json', meta)
    if proc.returncode != 0 or not result or result.get('is_error') or result.get('subtype') != 'success':
        text = (json.dumps(result or {}) + (attempt / 'stderr.log').read_text()[-4000:]).lower()
        if is_limited(info) or any(k in text for k in ('rate limit', 'rate_limit', 'usage limit', 'limit reached')):
            raise QuotaWait(f'{acct["name"]}: {scope} refused for usage limits; see {attempt.name}')
        raise RuntimeError(f'{scope}: session failed (rc={proc.returncode}, '
                           f'subtype={result and result.get("subtype")}); see {attempt.name}')
    if init.get('model') != m['judge_model'] or m['judge_model'] not in (result.get('modelUsage') or {}):
        raise RuntimeError(f'{scope}: judge model not confirmed (init={init.get("model")}, '
                           f'used={list((result.get("modelUsage") or {}))})')
    if sorted(init.get('tools') or []) != sorted(m['tools'].split(',') + ['StructuredOutput']):
        raise RuntimeError(f'{scope}: unexpected tool set {init.get("tools")}')
    verdict = result.get('structured_output')
    if verdict is None:
        raise RuntimeError(f'{scope}: no structured output')
    # Re-parse with the duplicate-key guard used by the Codex run.
    verdict = json.loads(json.dumps(verdict, ensure_ascii=False), object_pairs_hook=base.unique)
    write(attempt / 'raw_verdict.json', verdict)
    protocol.validate(verdict, spec)
    return verdict, meta


def worker(root, job_id, slot):
    base.mounted()
    m = read(root / 'manifest.json')
    job = next(j for j in m['jobs'] if j['id'] == job_id)
    acct = slot_account(m, slot)
    cell = root / job_id
    with base.lock(root / f'slot_{slot}.lock'), base.lock(cell / 'worker.lock'):
        state = read(cell / 'state.json')
        if state['status'] == 'complete':
            return
        if state['status'] != 'pending':
            raise RuntimeError('Started evaluations require explicit resume')
        try:
            v2 = m.get('view_version') == 2
            verify_frozen(root, m)
            astra.verify_protected(m['sources'][job['source_id']])
            (astra.verify_view if v2 else astra.verify_snapshot)(root, job['source_id'])
            state.update(status='running', pid=os.getpid(), account=acct['name'], started_utc=utc(), updated_utc=utc())
            write(cell / 'state.json', state)
            for scope in base.KINDS:
                base.mounted()
                inp = root / f'inputs/P{job["new_paper_id"]}'
                spec = read(inp / f'{scope}.schema.json')
                if astra.checkpoint_valid(cell, state, scope, spec):
                    continue
                if (root / 'PAUSE.json').exists():
                    state.update(status='paused', updated_utc=utc())
                    write(cell / 'state.json', state)
                    return
                stage_backstop(root, acct)
                ws = (astra.stage_view_workspace if v2 else astra.stage_workspace)(root, cell, job, scope)
                prompt_path = astra.prompt_file(root, job, scope) if v2 else inp / f'{scope}.prompt.txt'
                attempt = cell / 'attempts' / f'{scope}_{stamp()}'
                attempt.mkdir(parents=True, exist_ok=False)
                state['stages'][scope] = dict(status='running', started_utc=utc(), account=acct['name'],
                                              attempt=str(attempt.relative_to(root)), prompt=prompt_path.name)
                state['updated_utc'] = utc()
                write(cell / 'state.json', state)
                print(f'START {job_id} {scope} [{acct["name"]}]', flush=True)
                verdict, meta = run_stage(root, m, cell, job, scope, ws, attempt, spec,
                                          prompt_path.read_text(), acct)
                write(cell / f'verdicts/{scope}.json', verdict)
                state['stages'][scope].update(status='complete', duration_ms=meta.get('duration_ms'),
                    wall_s=meta['wall_s'], num_turns=meta.get('num_turns'), usage=meta.get('usage'),
                    total_cost_usd=meta.get('total_cost_usd'), session_id=meta.get('session_id'),
                    completed_utc=utc(), verdict_sha256=digest(cell / f'verdicts/{scope}.json'))
                state['updated_utc'] = utc()
                write(cell / 'state.json', state)
                print(f'COMPLETE {job_id} {scope} {meta["wall_s"]}s [{acct["name"]}]', flush=True)
            astra.verify_protected(m['sources'][job['source_id']])
            (astra.verify_view if v2 else astra.verify_snapshot)(root, job['source_id'])
            folder = root / f'frozen/eval_artifacts/paper{job["new_paper_id"]}'
            g = base.grader(folder)
            bundle = g.template()
            for scope in base.KINDS:
                verdict = read(cell / f'verdicts/{scope}.json')
                bundle[scope] = verdict['nodes']
                if scope == 'direction':
                    bundle['context'] = {k: row['value'] for k, row in verdict.get('context', {}).items()}
            write(cell / 'judgment_bundle.json', bundle)
            grade = g.grade_bundle(bundle)
            write(cell / 'grade.json', grade)
            state.update(status='complete', completed_utc=utc(), updated_utc=utc(),
                gates={k: v['status'] for k, v in grade['gates'].items()}, outcome=grade['outcome'],
                protected_files_unchanged=True, artifact_metadata_unchanged=True)
            write(cell / 'state.json', state)
        except QuotaWait as exc:
            n = state.get('quota_requeues', 0) + 1
            for s in state['stages'].values():
                if s['status'] == 'running':
                    s['status'] = 'quota_interrupted'
            if n > MAX_QUOTA_REQUEUES:
                state.update(status='failed', error=f'QuotaWait repeated {n}x: {exc}', updated_utc=utc())
                write(cell / 'state.json', state)
                pause(root, f'{job_id}: {state["error"]}')
                raise RuntimeError(state['error'])
            state.update(status='pending', quota_requeues=n, last_quota_wait=str(exc), updated_utc=utc())
            write(cell / 'state.json', state)
            print(f'REQUEUE {job_id} [{acct["name"]}]: {exc}', flush=True)
            sys.exit(EXIT_QUOTA)
        except BaseException as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}', updated_utc=utc())
            for s in state['stages'].values():
                if s['status'] == 'running':
                    s['status'] = 'failed'
            write(cell / 'state.json', state)
            pause(root, f'{job_id}: {state["error"]}')
            raise


# --------------------------------------------------------------------------- scheduling
def remaining_stages(root, job):
    stages = read(root / job['id'] / 'state.json').get('stages', {})
    return sum(stages.get(s, {}).get('status') != 'complete' for s in base.KINDS)


def write_accounts_status(root, m, active, blocks):
    rows = []
    for a in accounts_of(m):
        rec = read_usage(root, a['name']) or {}
        wins = (rec.get('info') or {}).get('unifiedWindows') or {}
        rows.append(dict(account=a['name'], caps=a['caps'],
            utilization={k: (v or {}).get('utilization') for k, v in wins.items()},
            resets={k: fmt_ts((v or {}).get('resetsAt')) for k, v in wins.items()},
            reading_utc=rec.get('utc'), in_flight=sum(1 for _, (p, j, lg, acc) in active.items() if acc == a['name']),
            blocked=blocks.get(a['name'])))
    write(root / 'accounts_status.json', dict(updated_utc=utc(), accounts=rows))


def supervise(root):
    base.mounted()
    m = read(root / 'manifest.json')
    accts, per = accounts_of(m), slots_per_account(m)
    total_slots = per * len(accts)
    binary = Path(m['claude_bin'])
    with base.lock(root / 'supervisor.lock'):
        astra.idle_slots(root, total_slots)
        verify_frozen(root, m)
        selected = m.get('view_version') == 2
        if selected:
            by_id = {j['id']: j for j in m['jobs']}
            groups = [('selection', [by_id[i] for i in m['selection']['order']])]
            views = astra.ViewPrefetcher(root, m)
        else:
            groups = [(key, [j for j in m['jobs'] if j['model_key'] == key]) for key in m['model_order']]
        active = {}                                   # slot -> (proc, job, logfile, account name)
        error = None
        try:
            for key, group in groups:
                pending = [j for j in group if read(root / j['id'] / 'state.json')['status'] == 'pending']
                while pending or active:
                    base.mounted()
                    if (root / 'PAUSE.json').exists():
                        error = read(root / 'PAUSE.json')['reason']
                    for slot, (proc, job, log, acc) in list(active.items()):
                        code = proc.poll()
                        if code is None:
                            continue
                        log.close()
                        del active[slot]
                        state = read(root / job['id'] / 'state.json')
                        if code == EXIT_QUOTA and state['status'] == 'pending':
                            pending.insert(0, job)        # requeued for quota; its account is re-gated
                            print(f'REQUEUED {job["id"]} [{acc}]', flush=True)
                        elif code or state['status'] != 'complete':
                            error = state.get('error', f'Worker stopped: {job["id"]}, status={state["status"]}, exit={code}')
                            pause(root, error)
                    blocks = {}
                    if not error:
                        for a in accts:
                            inflight = [j for (p, j, lg, acc) in active.values() if acc == a['name']]
                            if needs_probe(root, a, len(inflight)):
                                try:
                                    probe_quota(root, binary, a)
                                except Exception as exc:          # a failed probe only delays this account
                                    print(f'PROBE FAILED {a["name"]}: {exc}', flush=True)
                            blocks[a['name']] = account_block(root, a, sum(remaining_stages(root, j) for j in inflight))
                        if selected:
                            views.poll()
                            views.request(pending)
                            astra.evict_views(root, pending, [j['source_id'] for (p, j, lg, acc) in active.values()],
                                              views.building())
                        free = [n for n in range(total_slots) if n not in active]
                        for slot in free:
                            if not pending or (root / 'PAUSE.json').exists():
                                break
                            a = accts[slot // per]
                            inflight = [j for (p, j, lg, acc) in active.values() if acc == a['name']]
                            if account_block(root, a, sum(remaining_stages(root, j) for j in inflight)):
                                continue
                            if selected:
                                job = views.next_ready(pending)
                                if job is None:
                                    break
                                pending.remove(job)
                            else:
                                job = pending[0]
                                astra.snapshot(root, job['source_id'], m['sources'][job['source_id']])
                                pending.pop(0)
                            logfile = (root / job['id'] / 'worker.log').open('a')
                            proc = subprocess.Popen([sys.executable, '-B', '-u', str(root / 'run_claude_code_judge.py'),
                                'worker', '--root', str(root), '--job', job['id'], '--slot', str(slot)],
                                stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT)
                            active[slot] = (proc, job, logfile, a['name'])
                            print(f'LAUNCH pid={proc.pid} {job["id"]} [{a["name"]} slot {slot}]', flush=True)
                    write_accounts_status(root, m, active, blocks)
                    waiting = bool(pending) and not active and not error
                    reason = error or ('; '.join(f'{k}: {v}' for k, v in blocks.items() if v) if waiting else None)
                    now = [{'pid': p.pid, 'job': j['id'], 'account': acc} for p, j, lg, acc in active.values()]
                    phase = 'pausing' if error and active else 'paused' if error else 'waiting_quota' if waiting else 'running'
                    if selected:
                        astra.summarize_selected(root, m, now, phase, reason)
                    else:
                        astra.summarize(root, m, now, key, phase, reason)
                    if error and not active:
                        return
                    time.sleep(15 if active or (selected and views.building()) else 60)
                if any(read(root / j['id'] / 'state.json')['status'] != 'complete' for j in group):
                    raise RuntimeError(f'Incomplete model checkpoint: {key}')
                print(f'MODEL COMPLETE {key}', flush=True)
                if selected:
                    astra.evict_views(root, [], [])
            (astra.summarize_selected if selected else astra.summarize)(root, m, status='complete')
        except BaseException as exc:
            error = f'Supervisor: {type(exc).__name__}: {exc}'
            pause(root, error)
            for proc, _, log, _ in active.values():
                proc.wait()
                log.close()
            (astra.summarize_selected if selected else astra.summarize)(root, m, status='paused', reason=error)
            raise


def launch(root, resume=False):
    base.mounted()
    with base.lock(root / 'launch.lock'), base.lock(root / 'supervisor.lock'):
        m = read(root / 'manifest.json')
        astra.idle_slots(root, slots_per_account(m) * len(accounts_of(m)))
        verify_frozen(root, m)
        if (root / 'launch.json').exists() and not resume:
            raise RuntimeError('Already launched; use explicit resume to retain checkpoints')
        readings = {a['name']: probe_quota(root, Path(m['claude_bin']), a) for a in accounts_of(m)}
        if not any(readings.values()):
            raise RuntimeError('No account returned a usage reading; check the logins')
        if resume:
            for j in m['jobs']:
                cell = root / j['id']
                s = read(cell / 'state.json')
                if s['status'] not in ('pending', 'complete'):
                    history = cell / 'state_history'
                    history.mkdir(exist_ok=True)
                    write(history / f'{stamp()}.json', s)
                    s.update(status='pending', updated_utc=utc())
                    s.pop('error', None)
                    write(cell / 'state.json', s)
            if (root / 'PAUSE.json').exists():
                (root / 'PAUSE.json').rename(root / f'pause_history_{stamp()}.json')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with (root / 'supervisor.log').open('a') as logfile:
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(root / 'run_claude_code_judge.py'),
                'supervise', '--root', str(root)], stdin=subprocess.DEVNULL,
                stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        record = dict(pid=proc.pid, utc=utc(), resume=resume, accounts=[a['name'] for a in accounts_of(m)],
                      slots=slots_per_account(m) * len(accounts_of(m)),
                      preflight_utilization={n: {k: (v or {}).get('utilization') for k, v in
                                                 ((info or {}).get('unifiedWindows') or {}).items()}
                                             for n, info in readings.items()})
        write(root / 'launch.json', record)
        (root / 'launch_history').mkdir(exist_ok=True)
        write(root / 'launch_history' / f'{stamp()}.json', record)
        if Path('/usr/bin/caffeinate').exists():
            subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(proc.pid)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print(f'BATCH_PID={proc.pid}\nBATCH_DIRECTORY={root}', flush=True)


def revise(root, reason):
    """Install this file into an idle batch as a recorded code revision; judging inputs are untouched."""
    base.mounted()
    with base.lock(root / 'launch.lock'), base.lock(root / 'supervisor.lock'):
        m = read(root / 'manifest.json')
        astra.idle_slots(root, slots_per_account(m) * len(accounts_of(m)))
        running = [j['id'] for j in m['jobs'] if read(root / j['id'] / 'state.json')['status'] == 'running']
        if running:
            raise RuntimeError(f'Jobs still marked running: {running[:4]}')
        src = Path(__file__).resolve()
        if src == (root / 'run_claude_code_judge.py').resolve():
            raise RuntimeError('Run revise from the repository copy, not the frozen one')
        old_sha, new_sha = digest(root / 'run_claude_code_judge.py'), digest(src)
        history = root / 'code_history'
        history.mkdir(exist_ok=True)
        shutil.copy2(root / 'run_claude_code_judge.py', history / f'run_opus_batch.{old_sha[:12]}.py')
        shutil.copy2(src, root / 'run_claude_code_judge.py')
        if digest(root / 'run_claude_code_judge.py') != new_sha:
            raise RuntimeError('Revision copy mismatch')
        m['fixed_hashes']['run_claude_code_judge.py'] = new_sha
        m['accounts'] = [dict(a) for a in ACCOUNTS]
        m['slots_per_account'] = SLOTS_PER_ACCOUNT
        m['concurrency'] = SLOTS_PER_ACCOUNT * len(ACCOUNTS)
        m['stage_cost_estimate'] = STAGE_COST
        m.setdefault('code_revisions', []).append(dict(utc=utc(), file='run_claude_code_judge.py',
            old_sha256=old_sha, new_sha256=new_sha, reason=reason, judging_inputs_changed=False,
            accounts=[a['name'] for a in ACCOUNTS], slots=m['concurrency']))
        if (root / 'usage_latest.json').exists() and not usage_path(root, 'acct1').exists():
            shutil.copy2(root / 'usage_latest.json', usage_path(root, 'acct1'))
        write(root / 'manifest.json', m)
        print(f'REVISED run_claude_code_judge.py {old_sha[:12]} -> {new_sha[:12]}; accounts '
              f'{[a["name"] for a in ACCOUNTS]}, {m["concurrency"]} slots', flush=True)


def rescope_batch(root, reason):
    """Revision 3 (see run_codex_judge.py): 240-evaluation scope and full-submission views."""
    base.mounted()
    with base.lock(root / 'launch.lock'), base.lock(root / 'supervisor.lock'):
        m = read(root / 'manifest.json')
        astra.idle_slots(root, slots_per_account(m) * len(accounts_of(m)))
        running = [j['id'] for j in m['jobs'] if read(root / j['id'] / 'state.json')['status'] == 'running']
        if running:
            raise RuntimeError(f'Jobs still marked running: {running[:4]}')
        if m.get('view_version') == 2:
            raise RuntimeError('Batch already rescoped')
        installed = astra.install_code(root, m, ['run_claude_code_judge.py', 'run_codex_judge.py'])
        astra.rescope(root, m, reason, installed)


def set_slots(root, per_account, reason):
    """Change sessions per account on an idle batch (throughput knob; judging inputs untouched)."""
    base.mounted()
    with base.lock(root / 'launch.lock'), base.lock(root / 'supervisor.lock'):
        m = read(root / 'manifest.json')
        old = slots_per_account(m)
        astra.idle_slots(root, max(old, per_account) * len(accounts_of(m)))
        live = [j['id'] for j in m['jobs'] if read(root / j['id'] / 'state.json')['status'] == 'running'
                and astra.worker_alive(root / j['id'])]
        if live:
            raise RuntimeError(f'Workers still running: {live[:4]}')
        m['slots_per_account'] = per_account
        m['concurrency'] = per_account * len(accounts_of(m))
        m.setdefault('settings_history', []).append(dict(utc=utc(), slots_per_account=[old, per_account], reason=reason))
        write(root / 'manifest.json', m)
        print(f'SLOTS {old} -> {per_account} per account ({m["concurrency"]} sessions)', flush=True)


def update_code_batch(root, reason):
    base.mounted()
    with base.lock(root / 'launch.lock'), base.lock(root / 'supervisor.lock'):
        m = read(root / 'manifest.json')
        astra.idle_slots(root, slots_per_account(m) * len(accounts_of(m)))
        astra.update_code(root, m, reason, ['run_claude_code_judge.py', 'run_codex_judge.py'])


def pilot(root, job_id):
    """Run exactly one job in the foreground before the supervisor exists (usage calibration)."""
    base.mounted()
    m = read(root / 'manifest.json')
    if (root / 'launch.json').exists():
        raise RuntimeError('Batch already launched; the pilot only precedes launch')
    job = next(j for j in m['jobs'] if j['id'] == job_id)
    acct = slot_account(m, 0)
    info = probe_quota(root, Path(m['claude_bin']), acct)
    if is_limited(info):
        raise RuntimeError('Preflight quota: account is at its usage limit')
    astra.snapshot(root, job['source_id'], m['sources'][job['source_id']])
    worker(root, job_id, 0)
    astra.summarize(root, m, status='prepared')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare', 'pilot', 'launch', 'resume', 'supervise', 'worker',
                                        'status', 'pause', 'revise', 'rescope', 'update-code', 'set-slots'))
    ap.add_argument('--root', type=Path)
    ap.add_argument('--job')
    ap.add_argument('--slot', type=int)
    ap.add_argument('--reason')
    ap.add_argument('--slots', type=int)
    args = ap.parse_args()
    if args.command == 'prepare':
        prepare()
        return
    base.mounted()
    if not args.root:
        ap.error('--root is required')
    root = args.root.resolve()
    if not root.is_relative_to(base.JUDGE_RUNS) or not root.name.startswith(PREFIX):
        ap.error('Root must be a prepared Opus batch')
    if args.command == 'supervise':
        time.sleep(1)
        supervise(root)
    elif args.command in ('launch', 'resume'):
        launch(root, args.command == 'resume')
    elif args.command == 'pilot':
        if not args.job:
            ap.error('--job is required')
        pilot(root, args.job)
    elif args.command == 'worker':
        m = read(root / 'manifest.json')
        if args.job is None or args.slot is None or not 0 <= args.slot < slots_per_account(m) * len(accounts_of(m)):
            ap.error('--job and a valid --slot are required')
        worker(root, args.job, args.slot)
    elif args.command == 'set-slots':
        if not args.reason or not args.slots or args.slots < 1:
            ap.error('--slots N and --reason are required')
        set_slots(root, args.slots, args.reason)
    elif args.command in ('revise', 'rescope', 'update-code'):
        if not args.reason:
            ap.error('--reason is required')
        dict(revise=revise, rescope=rescope_batch).get(args.command, update_code_batch)(root, args.reason)
    elif args.command == 'pause':
        pause(root, 'Manual pause; in-flight stages finish and checkpoint')
    else:
        print(json.dumps(read(root / 'status.json'), indent=2))


if __name__ == '__main__':
    main()
