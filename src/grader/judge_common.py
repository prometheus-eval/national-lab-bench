"""Shared helpers for the two judges (run_codex_judge.py, run_claude_code_judge.py).

Paths come from the environment:
  SOG_SUBMISSIONS  solver workspaces, one per <agent>/paper<i>/ (default: <repo>/runs)
  SOG_JUDGE_RUNS   where judge batches are created           (default: <repo>/judge_runs)
  CODEX_BIN        the Codex CLI used by the Codex judge       (default: `codex` on PATH)
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
SUBMISSIONS = Path(os.environ.get('SOG_SUBMISSIONS', REPO / 'runs')).expanduser().resolve()
JUDGE_RUNS = Path(os.environ.get('SOG_JUDGE_RUNS', REPO / 'judge_runs')).expanduser().resolve()
CODEX = Path(os.environ.get('CODEX_BIN') or shutil.which('codex') or 'codex').resolve()
MODEL, EFFORT = 'gpt-6-astra', 'max'
KINDS = {'direction': 'direction_specific_rubric.json', 'integrity': 'integrity_check_rubric.json'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def serial(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if dataclasses.is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if hasattr(value, 'value'):
        return value.value
    raise TypeError(type(value).__name__)


def unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f'Duplicate JSON key: {k}')
        result[k] = v
    return result


def read(path):
    return json.loads(path.read_text(), object_pairs_hook=unique)


def write(path, value):
    # Generated checkpoints only, inside an exclusively allocated experiment.
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=serial)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def mounted():
    if not SUBMISSIONS.is_dir():
        raise RuntimeError(f'Submissions root not found: {SUBMISSIONS} (set SOG_SUBMISSIONS)')


@contextmanager
def lock(path):
    with path.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def limits(codex):
    raw = codex._client._request_raw('account/rateLimits/read', {})
    return {k: raw.get(k) for k in ('rateLimits', 'rateLimitsByLimitId')}


def quota_reason(response):
    windows = response.get('rateLimitsByLimitId') or {'codex': response.get('rateLimits')}
    if not any(windows.values()):
        return 'Account quota cannot be established'
    for name, r in windows.items():
        if not r:
            continue
        if r.get('spendControlReached') or r.get('rateLimitReachedType'):
            return f'{name}: service reports a spend/usage limit'
        for key in ('primary', 'secondary'):
            w = r.get(key)
            if w and w.get('usedPercent', 0) >= 90:
                return f'{name}/{key}: {w["usedPercent"]}% used, at or above the 90% safeguard'
    return None


def grader(folder):
    """The self-contained eval_artifacts/paper<i>/grading.py of one paper (src/grader/, or a batch's frozen/ copy)."""
    spec = importlib.util.spec_from_file_location(f'grading_{Path(folder).name}', Path(folder) / 'grading.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
