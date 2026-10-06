"""
Benchmark harness orchestrator — minimal baseline (`claude -p` CLI runner).

Also the shared helper module for run_served_harness.py and run_openhands_sdk.py,
which import its workspace staging, prompt, timeout, completion check
(_completion_status) and trajectory writer.

Runs a single Claude Code invocation on one benchmark task with NO
harness features: no hooks, skills, plugins, MCP servers, loops, or
subagents. The workspace is expected to already contain `task_spec.md`
and any reference materials the spec mentions (`paper.md`, `code/`,
`data/`, etc.). The runner copies `neurips.sty` in and launches
claude-code; nothing else.

Two outputs land in the workspace root:
  - Whatever the agent produces under `proposal/`
  - `ai_scientist_trajectory.json` — full untrimmed event log from the
    Claude Code stream, plus session metadata (elapsed time, model, etc.)

Usage:
  python3 run.py --workspace ./papers/paper1/                     # default: no time mention to agent
  python3 run.py --workspace ./papers/paper1/ --hours 24          # tell agent "you have 24 hours"
  python3 run.py --workspace ./papers/paper1/ --unlimited         # tell agent "you have unlimited time"
  python3 run.py --workspace ./papers/paper1/ --session-id <existing>

The default omits any time-budget language from the prompt (the subprocess
still has a safety-net timeout). `--hours N` injects the budget into the
prompt, caps the subprocess at N hours, and inlines per-paper anti-quit
content from spec_addendums/<paper_id>_time.md if present. `--unlimited`
tells the agent so explicitly and effectively disables the subprocess cap.
"""

import argparse
import json
import os
import re
import fcntl
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
DEFAULT_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", 86400))  # safety-net timeout when no --hours / --unlimited
UNLIMITED_TIMEOUT = 30 * 24 * 3600  # 30 days; effectively unlimited

REQUIRED_CLAUDE_VERSION = "2.1.215"
MODEL = "claude-opus-4-8"
EFFORT = "max"

STYLE_FILE = Path(__file__).resolve().parent / "neurips.sty"
TIME_ADDENDUM_DIR = Path(__file__).resolve().parent / "spec_addendums"
TRAJECTORY_FILENAME = "ai_scientist_trajectory.json"

# Allow `--dangerously-skip-permissions` inside containers running as root.
# Claude Code's root guard checks this undocumented env var.
os.environ.setdefault("IS_SANDBOX", "1")

# Wait (unlimited) for background subagents/workflows at `-p` exit rather than
# the ~10-minute default, so long-running experiment subagents aren't cut off.
os.environ.setdefault("CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS", "0")

# Ban Claude Code auto-memory: every run must start clean. Disables BOTH reading
# and writing of ~/.claude/projects/<repo>/memory. That dir is keyed to the git
# root, so it is otherwise shared across all papers AND persisted across runs --
# a rerun would inherit its own prior solution (method, derivations, env),
# invalidating the benchmark. Hard-set (not setdefault) so the environment can't
# turn it back on. OAuth/subscription-safe, unlike --bare.
os.environ["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"

# Completion loop: after the agent stops, if the paper isn't actually finished
# (no compiled PDF, or placeholder/pending tokens remain), --resume the same
# session to drive it to completion, up to this many resumes (then accept the
# honest partial). Set CLAUDE_MAX_RESUMES=0 to disable the loop.
MAX_RESUMES = int(os.environ.get("CLAUDE_MAX_RESUMES", "5"))

# Persistent objective injected into the SYSTEM prompt on every attempt (initial
# and every resume) via --append-system-prompt-file, so the goal survives across
# turns and compaction instead of living only in a single -p user message.
GOAL_FILE = Path(__file__).resolve().parent / "goal.md"


def check_auth():
    """
    Fail fast if Claude Code isn't authenticated. Accepts either an
    `ANTHROPIC_API_KEY` env var or a credentials file from `claude /login`.
    Best-effort: macOS users authenticated via Keychain may need to set the
    env var explicitly even if logged in.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if (Path.home() / ".claude" / ".credentials.json").exists():
        return
    # macOS: `claude /login` stores the OAuth token in the login Keychain, not
    # in ~/.claude/.credentials.json. Treat a Keychain entry as authenticated —
    # the claude subprocess reads it the same way the interactive REPL does.
    if sys.platform == "darwin":
        try:
            rc = subprocess.run(
                ["security", "find-generic-password", "-s", "Claude Code-credentials"],
                capture_output=True, check=False,
            ).returncode
            if rc == 0:
                return
        except Exception:
            pass
    sys.exit(
        "[run] Claude Code is not authenticated. Either:\n"
        "  - set ANTHROPIC_API_KEY=<your-api-key>, or\n"
        "  - run `claude /login` once to authenticate via OAuth."
    )


def check_claude_version():
    """Fail fast if `claude --version` does not match REQUIRED_CLAUDE_VERSION."""
    result = subprocess.run(
        [CLAUDE_BIN, "--version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(f"[run] `{CLAUDE_BIN} --version` failed: {result.stderr.strip()}")
    match = re.search(r"\d+\.\d+\.\d+", result.stdout)
    if not match:
        sys.exit(f"[run] Could not parse claude version from: {result.stdout.strip()!r}")
    version = match.group(0)
    if version != REQUIRED_CLAUDE_VERSION:
        sys.exit(
            f"[run] Claude Code version mismatch: required {REQUIRED_CLAUDE_VERSION}, "
            f"found {version}. Install the pinned version before running."
        )


def _set_blocking(fd):
    """Clear O_NONBLOCK on a file descriptor (best-effort)."""
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    except (OSError, ValueError):
        pass


def _safe_print(*args, **kwargs):
    """print() that survives BlockingIOError (Errno 11).

    The claude/codex child (Node/Bun) can flip the shared stdout/stderr pipe to
    non-blocking; when the agent's output is bursty and a downstream `tee` cannot
    drain it fast enough, the parent's print() then raises BlockingIOError and
    would crash the run -- killing the agent mid-task. Restore blocking mode and
    retry so the run keeps going; never let a display write abort the run.
    """
    kwargs.setdefault("flush", True)
    for _ in range(200):
        try:
            print(*args, **kwargs)
            return
        except BlockingIOError:
            _set_blocking(sys.stdout.fileno())
            _set_blocking(sys.stderr.fileno())
            time.sleep(0.05)
    # Give up on this one line rather than crash the run.


def _kill_process_group(pid):
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _short(s, n=200):
    s = s.replace("\n", " | ")
    return s if len(s) <= n else s[:n] + "..."


def _tool_summary(name, inp):
    """One-line human summary of a tool call's input."""
    if name == "Bash":
        return _short(inp.get("command", ""))
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return inp.get("file_path", "?")
    if name in ("Grep", "Glob"):
        return f"pattern={inp.get('pattern', '?')}"
    if name in ("Task", "Agent"):
        return inp.get("description", "?")
    if name == "WebFetch":
        return inp.get("url", "?")
    if name == "WebSearch":
        return inp.get("query", "?")
    return _short(json.dumps(inp))


def _format_event(evt):
    """Format one stream-json event for human display. Returns str or None."""
    t = evt.get("type")
    if t == "system":
        sub = evt.get("subtype", "system")
        if sub == "init":
            return f"[init] session={evt.get('session_id', '?')[:8]}"
        return f"[{sub}]"
    if t == "assistant":
        parts = []
        for block in evt.get("message", {}).get("content", []):
            bt = block.get("type")
            if bt == "text":
                txt = block.get("text", "").strip()
                if txt:
                    parts.append(txt)
            elif bt == "tool_use":
                name = block.get("name", "?")
                parts.append(f"→ {name}: {_tool_summary(name, block.get('input', {}))}")
        return "\n".join(parts) if parts else None
    if t == "user":
        parts = []
        for block in evt.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                marker = "✗" if block.get("is_error") else "✓"
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                parts.append(f"  {marker} {_short(str(content).strip())}")
        return "\n".join(parts) if parts else None
    if t == "result":
        success = evt.get("subtype") == "success"
        cost = evt.get("total_cost_usd")
        cost_str = f" cost=${cost:.4f}" if cost else ""
        return f"[result] success={success}{cost_str}"
    return None


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as `Hh Mm Ss`."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def build_time_section(workspace, hours=None, unlimited=False):
    """Build the time-budget prefix injected into the agent's prompt.

    Returns an empty string when neither flag is set (default: no time
    mention to the agent). With --hours N, also inlines per-paper anti-quit
    content from spec_addendums/<paper_id>_time.md when present, after
    substituting `{{hours}}` with N.
    """
    if unlimited:
        # The durable objective + survival rules now live in the system prompt
        # (goal.md via --append-system-prompt-file), so the per-turn user message
        # stays a one-liner. Nothing to prepend here.
        return ""
    if hours is None:
        return ""
    parts = [
        f"You have {hours} hours from the start of this run to final submission. "
        f"Plan experiments accordingly — submissions that finish early without "
        f"exhausting the budget often have unrealized improvements. When this "
        f"time elapses, the run will be terminated.\n\n"
        f"If the budget runs out and substantial work remains, wrap up with what "
        f"you have rather than overclaiming. Honest accounting in the appendix "
        f"(with attempts logged) is a better outcome than an inflated main paper "
        f"that the reward-hacking checks will catch."
    ]
    paper_id = Path(workspace).name
    addendum_path = TIME_ADDENDUM_DIR / f"{paper_id}_time.md"
    if addendum_path.exists():
        parts.append(addendum_path.read_text().replace("{{hours}}", str(hours)))
    return "\n\n".join(parts)


def build_prompt(workspace, hours=None, unlimited=False):
    """Compose the -p prompt: optional time prefix + 'Read task_spec.md...'."""
    parts = []
    time_section = build_time_section(workspace, hours, unlimited)
    if time_section:
        parts.append(time_section)
    parts.append("Read task_spec.md and complete the task it describes.")
    return "\n\n".join(parts)


def resolve_timeout(hours=None, unlimited=False):
    """Pick the subprocess timeout based on the time-budget flags."""
    if unlimited:
        return UNLIMITED_TIMEOUT
    if hours is not None:
        return hours * 3600
    return DEFAULT_TIMEOUT


# Distinctive placeholder / "not-done" markers a finished paper must not contain.
# Kept conservative (bracketed tokens, "runs pending", LaTeX \todo, "to be X")
# to avoid false positives on ordinary prose.
_PLACEHOLDER_RE = re.compile(
    r"\[\s*(?:PEND(?:ING)?|TODO|TBD|PLACEHOLDER|XXX|\?\?\?|FIXME|FILL[ _]?IN|RESULTS?)\s*\]"
    r"|runs?\s+pending"
    r"|\\(?:todo|pending|placeholder)\b"
    r"|\bto\s?be\s?(?:filled|determined|computed|run|added|reported)\b"
    r"|\bX{3,}\b"                       # bare unfilled macro/table value: XXX, XXXX
    r"|\bplaceholder\b"                 # e.g. "Placeholder abstract for the ..."
    r"|\bsmoke[-\s]?test\b"             # e.g. "build smoke test"
    r"|\b(?:TBD|TODO|FIXME)\b"          # bare not-done markers
    # \newcommand{\X}{--} (or {-},{?},{N/A},{nan}): an unfilled result macro,
    # the numeric analogue of {XXX}. Scoped to the macro definition so ordinary
    # en-dashes in prose ("pages 1--49") don't match.
    r"|\\newcommand\*?\s*\{\\[A-Za-z@]+\}\s*(?:\[\d+\])?\s*\{\s*(?:-{1,2}|\?{1,3}|n/?a|nan)\s*\}",
    re.IGNORECASE,
)

# \input / \include / \subfile references we must follow: report.tex is usually
# just a wrapper, and the real (often still-placeholder) content lives in the
# sections/*.tex and logs/*.tex files it pulls in. Reading only report.tex made
# the completion check blind to a whole paper of `\newcommand{..}{XXX}` macros.
_TEX_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\*?\s*\{([^}]+)\}")


def _strip_tex_comments(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("%"))


def _resolve_tex_sources(report_tex: Path, _max_files: int = 200):
    """report.tex plus every .tex it \\input/\\include-s, resolved recursively
    relative to report.tex's directory. Cycle- and count-guarded."""
    base = report_tex.parent
    seen, order, queue = set(), [], [report_tex]
    while queue and len(order) < _max_files:
        cur = queue.pop(0)
        if not cur.exists():
            continue
        try:
            rp = cur.resolve()
        except OSError:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        order.append(cur)
        try:
            body = _strip_tex_comments(cur.read_text(errors="ignore"))
        except OSError:
            continue
        for ref in _TEX_INPUT_RE.findall(body):
            cand = base / ref.strip()
            if cand.suffix != ".tex":
                cand = cand.with_suffix(".tex")
            queue.append(cand)
    return order


def _completion_status(workspace: Path):
    """Heuristic check that the paper is actually finished. Returns
    (complete: bool, reason: str). Complete = a non-empty proposal/report.pdf
    exists AND neither report.tex nor any file it \\input-s contains a
    placeholder/pending token."""
    pdf = workspace / "proposal" / "report.pdf"
    tex = workspace / "proposal" / "report.tex"
    if not pdf.exists() or pdf.stat().st_size == 0:
        return False, "proposal/report.pdf is missing or empty (the paper did not compile)"
    if not tex.exists():
        return False, "proposal/report.tex is missing"
    from collections import Counter
    counts = Counter()
    offenders = set()
    for src in _resolve_tex_sources(tex):
        body = _strip_tex_comments(src.read_text(errors="ignore"))
        for h in _PLACEHOLDER_RE.findall(body):
            counts[h.strip().replace("\n", " ")] += 1
            try:
                offenders.add(src.relative_to(workspace).as_posix())
            except ValueError:
                offenders.add(src.name)
    if counts:
        summary = ", ".join(f"'{tok}' x{n}" for tok, n in counts.most_common(6))
        where = ", ".join(sorted(offenders)[:4])
        return False, (f"the paper still contains placeholder/pending entries "
                       f"({summary}) in {where}")
    return True, "complete"


def _resume_prompt(reason: str = "") -> str:
    """Generic continuation prompt used to --resume an unfinished session. The
    durable rules live in goal.md (system prompt); this just re-engages the agent
    for another turn. `reason` is accepted for call-site compatibility but not
    interpolated, to keep the nudge generic."""
    return (
        "Your previous turn ended but the paper is not finished. Continue: wait for or "
        "poll any running experiments and fold their results in, fill every remaining "
        "value, and make sure proposal/report.pdf recompiles with no placeholder, "
        "pending, or XXX content anywhere (including the \\input-ed section and macro "
        "files). If a specific experiment genuinely cannot be completed, remove its "
        "placeholder and log the attempt in proposal/attempts_log.md. Do not stop until "
        "it is done."
    )


def run_claude(args, cwd=None, timeout=DEFAULT_TIMEOUT, trajectory_events=None):
    """
    Run the claude CLI in stream-json mode and pretty-print events live.
    On timeout (occasional Bun runtime issue), kill the whole process group.

    If `trajectory_events` is a list, every stdout line is appended to it as
    an entry: `{"t_offset": seconds_since_start, "parsed": <obj or None>,
    "raw_line": <string or None>}`. Non-JSON lines are recorded with
    `parsed=None` and the original `raw_line`; parsed JSON lines have the
    full parsed object and `raw_line=None` to avoid duplication.

    Returns a tuple `(timed_out: bool, return_code: int)`.
    """
    proc = subprocess.Popen(
        [CLAUDE_BIN] + args,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    timed_out = threading.Event()

    def _on_timeout():
        timed_out.set()
        _kill_process_group(proc.pid)

    timer = threading.Timer(timeout, _on_timeout)
    timer.start()
    t0 = time.monotonic()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            t_offset = time.monotonic() - t0
            try:
                obj = json.loads(line)
                pretty = _format_event(obj)
                if trajectory_events is not None:
                    trajectory_events.append({
                        "t_offset": round(t_offset, 3),
                        "parsed": obj,
                        "raw_line": None,
                    })
            except json.JSONDecodeError:
                pretty = line  # pass through non-JSON (warnings, etc.)
                if trajectory_events is not None:
                    trajectory_events.append({
                        "t_offset": round(t_offset, 3),
                        "parsed": None,
                        "raw_line": line,
                    })
            if pretty:
                _safe_print(pretty)
        proc.wait()
        if timed_out.is_set():
            print(f"\n[run] Timeout after {timeout}s; killed process group.", flush=True)
    finally:
        timer.cancel()
        _kill_process_group(proc.pid)

    return timed_out.is_set(), proc.returncode


def setup_workspace(workspace_path):
    """Validate the workspace contains task_spec.md and stage neurips.sty."""
    workspace = Path(workspace_path).resolve()

    if workspace.exists() and not workspace.is_dir():
        sys.exit(f"[run] --workspace is not a directory: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)

    spec = workspace / "task_spec.md"
    if not spec.exists() or spec.stat().st_size == 0:
        sys.exit(f"[run] task_spec.md missing or empty in workspace: {spec}")

    if not STYLE_FILE.exists():
        sys.exit(f"[run] Style file missing: {STYLE_FILE}")
    workspace_style = workspace / "neurips.sty"
    if STYLE_FILE != workspace_style:
        shutil.copy2(STYLE_FILE, workspace_style)

    return workspace


def save_trajectory(workspace: Path, metadata: dict, events: list) -> Path:
    """Save the full trajectory plus session metadata to
    `<workspace>/ai_scientist_trajectory.json`. Written even on timeout
    or crash so the partial trajectory is recoverable. Returns the path.
    """
    path = workspace / TRAJECTORY_FILENAME
    payload = {"metadata": metadata, "events": events}
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    except (OSError, TypeError) as e:
        # As a fallback, dump events without the indent to avoid losing data
        # if some embedded value is non-serializable; coerce via default=str.
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str))
        print(f"[run] Note: trajectory written with str-coercion fallback: {e}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Run baseline Claude Code (no harness features) on a single benchmark task."
    )
    ap.add_argument("--workspace", required=True,
                    help="Paper directory containing task_spec.md (the agent runs here).")
    ap.add_argument("--session-id", default=None,
                    help="Resume an existing Claude Code session ID.")
    time_group = ap.add_mutually_exclusive_group()
    time_group.add_argument("--hours", type=int, default=None,
                            help="Tell the agent it has N hours, and cap the subprocess at N hours. "
                                 "If a per-paper addendum exists at spec_addendums/<paper_id>_time.md, "
                                 "it is inlined into the prompt with {{hours}} substituted.")
    time_group.add_argument("--unlimited", action="store_true",
                            help="Tell the agent it has unlimited time. Subprocess effectively uncapped.")
    args = ap.parse_args()

    if args.hours is not None and args.hours <= 0:
        sys.exit("[run] --hours must be a positive integer")

    check_claude_version()
    check_auth()
    workspace = setup_workspace(args.workspace)
    session_id = args.session_id or str(uuid.uuid4())
    prompt = build_prompt(workspace, hours=args.hours, unlimited=args.unlimited)
    claude_timeout = resolve_timeout(hours=args.hours, unlimited=args.unlimited)

    if args.unlimited:
        time_label = "unlimited (agent told)"
    elif args.hours is not None:
        time_label = f"{args.hours}h (agent told)"
    else:
        time_label = f"{claude_timeout}s safety-net (agent not told)"

    print(f"[run] Claude:    {CLAUDE_BIN} {REQUIRED_CLAUDE_VERSION}", flush=True)
    print(f"[run] Model:     {MODEL} (effort={EFFORT})", flush=True)
    print(f"[run] Workspace: {workspace}", flush=True)
    print(f"[run] Session:   {session_id}", flush=True)
    print(f"[run] Time:      {time_label}", flush=True)
    print(f"[run] Timeout:   {claude_timeout}s ({_format_elapsed(claude_timeout)})", flush=True)

    print(f"\n=== Run (session {session_id}) ===", flush=True)
    trajectory_events: list = []
    start_wall = time.monotonic()
    start_iso = datetime.now(timezone.utc).isoformat()
    deadline = start_wall + claude_timeout
    timed_out = False
    return_code = None
    attempt = 0
    complete = False
    reason = ""
    try:
        while True:
            attempt += 1
            remaining = max(1, int(deadline - time.monotonic()))
            this_prompt = prompt if attempt == 1 else _resume_prompt(reason)
            # attempt 1 opens a new session (unless the user asked to resume one);
            # every subsequent attempt resumes the same session to keep context.
            if attempt == 1 and not args.session_id:
                session_flag = ["--session-id", session_id]
            else:
                session_flag = ["--resume", session_id]
            claude_args = [
                "-p", this_prompt,
                "--output-format", "stream-json", "--verbose",
                "--model", MODEL, "--effort", EFFORT,
                "--dangerously-skip-permissions",
            ] + session_flag
            # Pin the durable objective in the system prompt on every attempt.
            if GOAL_FILE.exists():
                claude_args += ["--append-system-prompt-file", str(GOAL_FILE)]

            trajectory_events.append({
                "t_offset": round(time.monotonic() - start_wall, 3),
                "parsed": {"type": "harness", "event": "attempt_start",
                           "attempt": attempt, "resume": attempt > 1,
                           "reason": (reason if attempt > 1 else None)},
                "raw_line": None,
            })
            if attempt == 1:
                print(f"\n[run] Attempt {attempt} (initial)", flush=True)
            else:
                print(f"\n[run] Attempt {attempt} (resume) — previous stop was incomplete: {reason}", flush=True)

            to, return_code = run_claude(
                claude_args, cwd=workspace, timeout=remaining,
                trajectory_events=trajectory_events,
            )
            timed_out = timed_out or to

            complete, reason = _completion_status(workspace)
            print(f"[run] After attempt {attempt}: "
                  f"{'COMPLETE' if complete else 'INCOMPLETE -- ' + reason}", flush=True)

            if complete or timed_out:
                break
            if attempt >= 1 + MAX_RESUMES:
                print(f"[run] Reached resume cap ({MAX_RESUMES}); accepting the current "
                      f"(incomplete) submission.", flush=True)
                break
            if time.monotonic() >= deadline:
                print("[run] Time budget exhausted; accepting the current submission.", flush=True)
                break
    finally:
        end_wall = time.monotonic()
        end_iso = datetime.now(timezone.utc).isoformat()
        elapsed_seconds = end_wall - start_wall
        metadata = {
            "session_id": session_id,
            "model": MODEL,
            "effort": EFFORT,
            "claude_version": REQUIRED_CLAUDE_VERSION,
            "claude_bin": CLAUDE_BIN,
            "workspace": str(workspace),
            "timeout_seconds": claude_timeout,
            "time_budget_hours": args.hours,
            "time_budget_unlimited": args.unlimited,
            "attempts": attempt,
            "max_resumes": MAX_RESUMES,
            "completed": complete,
            "completion_reason": reason,
            "timed_out": timed_out,
            "return_code": return_code,
            "start_utc": start_iso,
            "end_utc": end_iso,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_human": _format_elapsed(elapsed_seconds),
            "event_count": len(trajectory_events),
        }
        traj_path = save_trajectory(workspace, metadata, trajectory_events)

        print(f"\n[run] Elapsed:    {_format_elapsed(elapsed_seconds)} ({elapsed_seconds:.1f}s) over {attempt} attempt(s)", flush=True)
        if complete:
            print(f"[run] Status:     COMPLETE (report.pdf present, no placeholders)", flush=True)
        elif timed_out:
            print(f"[run] Status:     TIMED OUT, still incomplete ({reason})", flush=True)
        else:
            print(f"[run] Status:     stopped INCOMPLETE ({reason})", flush=True)
        print(f"[run] Trajectory: {traj_path} ({len(trajectory_events)} events)", flush=True)
        print(f"[run] Session:    {session_id}", flush=True)


if __name__ == "__main__":
    main()