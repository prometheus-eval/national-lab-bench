GOAL: Produce the complete follow-up paper described in task_spec.md — a finished
proposal/report.pdf a top venue would accept, with every experiment actually run
and every number, table, and figure filled from real results.

WORKSPACE: Work only inside your current directory. Inputs are here: task_spec.md
(your assignment) and the provided code/ and data/. Write everything you produce
under proposal/, exactly as task_spec.md specifies (paper at proposal/report.tex
compiled to proposal/report.pdf with the provided neurips.sty). Do not read or
write anything outside this directory.

HOW YOU RUN: You are running autonomously — there is no human to hand back to and
no next turn unless you finish the paper now.
- Launch every long-running experiment with the Bash tool's run_in_background=true (or
  as a background subagent) -- NOT with raw nohup/setsid/`&`, which the harness cannot
  see and may cut off. The harness tracks run_in_background jobs, keeps the session alive
  while they run, and automatically re-invokes you when they finish, so it is fine to end
  your turn to await that re-invocation. Have each job write its results to disk; when
  re-invoked, read them and fold the real numbers in.
- Never finalize the paper while an experiment is still running or any result is unfilled.
- Ship nothing with placeholder, pending, or XXX content anywhere — including the
  \input-ed sections/*.tex and logs/*.tex macro files. Fill every value from a real
  run, or drop the claim and record the attempt honestly in proposal/attempts_log.md.
