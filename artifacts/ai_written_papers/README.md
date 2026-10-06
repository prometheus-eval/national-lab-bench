# AI-written papers

The follow-up papers the agents wrote: `<agent>/paper<i>.pdf` is `proposal/report.pdf` of that agent's submission for
task P<i>, the PDF the judge read (`artifacts/judge_results/`). `<agent>` uses the solver directory names of `src/solver/run.sh`.

- 116 of the 120 submissions include a PDF. GLM-5.3 left none for P6, P10, P12 and P14 (`manifest.json` lists them as `null`).
- Every copy is identical to the judged PDF (`source_sha256`), except `fable-5/paper8.pdf`: the agent put a real person's
  name, affiliation and e-mail address on its title page, which are blacked out here (`redacted` in `manifest.json`).
- Browse them with the viewer (`python viewer/serve.py`, tab "AI-written papers").
