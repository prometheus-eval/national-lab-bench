# Task: Develop a Substantive Follow-up to "Simulations of Relativistic-Quantum Plasmas Using Real-Time Lattice Scalar QED"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the variational lattice scalar QED scheme by Shi, Xiao, Qin, and Fisch, "Simulations of relativistic-quantum plasmas using real-time lattice scalar QED," *Physics of Plasmas* 25, 123107 (2018). Your goal is to produce an original contribution that goes beyond the original paper, written up as an academic paper.

## The Problem — Here's why it matters

When plasmas are dense or fields are strong, the standard plasma-kinetic description (a geometrical-optics approximation of the underlying relativistic-quantum theory) breaks down. Examples include electron–positron pair production by intense lasers, wave dynamics on scales comparable to the Compton wavelength, and astrophysical plasmas in extreme magnetic fields. Semiclassical fixes — inserting prefabricated source terms into kinetic or fluid equations — fail to capture coexisting QED processes self-consistently and disrespect energy–momentum conservation once many pairs are produced. A faithful description requires a relativistic-quantum framework, but full path-integral lattice QED is numerically prohibitive. The classical-statistical regime (high occupation numbers, weak coupling) admits a tractable approximation in which tree-level effects dominate and the quantum system is well-modeled by time-advancing classical field equations with an ensemble of statistically equivalent initial conditions.

## Background — Here's what you need to know about the original paper

Read `paper.md` in full before beginning. The original paper introduces a variational lattice discretization of scalar QED with the following key properties:

- **Variational construction.** A Yee-type discrete action $S_d$ couples a scalar field $\phi_v$ (on vertices) to a gauge 1-form $A_e$ (on edges). The equations of motion are obtained by extremizing $S_d$, yielding a discrete Klein-Gordon equation [Eq. (16)], a discrete Gauss law [Eq. (17)], and a discrete Maxwell-Ampère law [Eq. (19)].
- **Conservation by construction.** The discrete U(1) gauge invariance of $S_d$ implies local charge conservation to machine precision. The Bianchi identities $\nabla \cdot \mathbf{B} = 0$ and $\nabla \times \mathbf{E} = -\partial_t \mathbf{B}$ (discrete Faraday's law) hold automatically by the geometric construction.
- **Resolution constraints.** $m\Delta t \ll 1$, $\Delta t < \Delta x$, $qA\Delta < 2\pi$, plus any frequency constraints implied by external fields.
- **Validity regime.** The classical-statistical approximation requires high occupation numbers and weak coupling ($e \approx 0.3$). Outside this regime, loop effects matter and the scheme is not justified from first principles.

The paper validates the scheme on two demonstrations: 1D linear plasma waves (Sec. III.1, matching analytic dispersion in the unmagnetized single-species regime) and 1D laser-plasma interaction (Sec. III.2, demonstrating wakefield acceleration at moderate intensity and pair production above the Schwinger threshold). These demonstrations cover a narrow slice of the framework's claimed scope.

## Contract — Here's exactly what counts as success

You will identify and study **four test settings** for the lattice scalar QED scheme, split into two categories:

**Two scope tests.** Settings where the lattice scalar QED scheme correctly reproduces analytic or semi-analytic predictions, but which were not directly tested in the original paper. These extend the verified scope of the framework.

**Two boundary tests.** Settings where the lattice scalar QED scheme fails for *fundamental* reasons — not implementation bugs, not insufficient resolution, but because some assumption underlying the framework is violated. For each boundary test you must:

1. State in advance what specific signature in the simulation output will count as fundamental failure. This criterion must appear in the paper before the boundary-test results.
2. Identify which fundamental framework assumption fails.
3. Demonstrate the failure with evidence that distinguishes it from an implementation bug: a convergence study across multiple resolutions (the failure persists at all studied $\Delta x$), and a cross-check against the reproduction of the paper's two demos (those demos must succeed in the same implementation to rule out implementation issues).
4. Reason from first principles about why the failure is fundamental.
5. Propose a modification to the scheme that addresses the failure. The modification must be documented at the same mathematical level as the original scheme — with the modified action or Lagrangian, the modified equations of motion, and the conservation properties of the fixed scheme all stated. Then implement the fix and verify that it resolves the failure.
6. State explicitly which invariants the fix preserves and which (if any) it relaxes. If the original scheme conserves charge by construction and the fix is intended to maintain that, the logs must demonstrate it. If the fix relaxes some property, the paper must say so and justify the relaxation.

You should aim to fully implement and verify the proposed fixes for both boundary tests. Allocate effort across the four tests as you see fit.

**Reproductions of paper demos are mandatory.** You must reproduce both of the paper's demonstrations (Sec. III.1 and Sec. III.2) as implementation verification. If the implementation fails on either reproduction, no boundary-test claims can be validated — the entire contract is unverifiable, because we cannot tell whether the boundary failure is a framework-level issue or an implementation bug. The reproductions are the foundation that makes the rest of the contract meaningful.

**Effort evidence is mandatory.** For any work you document as failed or unresolved (in either the main paper or the appendix), there must be a corresponding `proposal/attempts_log.md` entry recording each distinct approach you tried. A boundary test you describe as "failed despite effort" requires evidence in the attempts log showing multiple distinct attempts at resolution. Documenting failure without an attempts log will be read as not having tried.

**Main paper vs. appendix structure.** The agent must self-categorize work:

- **Main paper** contains work the agent claims represents a complete, deep contribution: predictions stated and met (or fundamental failures identified and fixed), with reasoning fully analyzed and presented at journal standard. Anything in the main paper is the agent's claim of substantive achievement.
- **Appendix** is a fallback for work the agent attempted but could not bring to journal quality after exhausting reasonable approaches. It is not a shortcut for skipping difficult work. Before placing anything in the appendix, the agent must have attempted multiple distinct approaches to completing it in the main paper, documented in `proposal/attempts_log.md`. The expected outcome is an empty or near-empty appendix, achieved by doing the work — not by labeling easy work as "shallow" and moving on.

The agent's placement decisions are themselves claims about depth. Work placed in the main paper is implicitly claimed to be deep and complete; misplacement (work in the main paper that turns out to lack the depth claimed) is a worse outcome than honest placement in the appendix. A submission with substantial appendix content signals that the agent gave up on completing the work; this is ranked above outright deception but below a submission that completed the work and has a minimal appendix.

**Four principles every test must satisfy.** These are requirements on submission shape, and they are also the rubric reviewers apply.

1. **Regime distinctness.** Each of the four tests probes a physically distinct regime. No test directly re-verifies the paper's Sec. III.1 or Sec. III.2. The four tests must be pairwise distinct from each other.
2. **Predictive sharpness.** Each test pairs the simulation with a sharp analytic or semi-analytic prediction stated up front in the paper's Methodology, before any simulation results are described. Qualitative checks ("looks right," "reasonable agreement") do not satisfy this principle.
3. **Diagnostic depth.** Across the four tests, you exercise multiple bug classes — gauge invariance, charge/energy conservation, dispersion fidelity, non-trivial $\phi$-$A$ coupling. For each test, identify in the Methodology which bug class is its *primary diagnostic*. No two tests should share the same primary diagnostic, and ideally all four classes are exercised.
4. **Tolerance justification.** Each agreement check (for scope tests) and each failure criterion (for boundary tests) uses a pre-stated tolerance whose value is justified — by discretization error from a convergence study, by ensemble noise floor, or by physical significance. Each test must include a convergence study at multiple lattice resolutions. Once stated, tolerance values must remain consistent throughout the paper.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated using a leaf-level rubric and a tiered scoring function. The grading logic priorities are, from highest to lowest priority:

1. **Integrity.** Claims must match evidence. Implementation must be variational. No hardcoded predictions, no undisclosed external code, no falsified effort claims.
2. **Completeness.** Paper reproductions present and working. Required deliverables present. Boundary-test completions ranked above scope-test completions. Time budget respected.
3. **Honest accounting.** Work that is incomplete is admitted; effort logs document attempts; main/appendix placement reflects depth honestly.
4. **Quality.** Reasoning soundness, regime ambition, writing quality, diagnostic coverage, literature engagement.

A submission that lies about completion is worse than a submission that honestly admits incomplete work. A submission with completed boundary tests is ranked higher than a submission with completed scope tests but unfinished boundary tests. Among submissions with the same completion level, depth and ambition determine the tier.

## Rules — Here are the rules

### Available resources

The working directory contains:

- `paper.md` — markdown version of the original paper.
- `paper.pdf` — original paper.
- `images/` — figures from the original paper.
- `code/` — empty. The original paper does not distribute source code, so you implement the scheme yourself.
- `data/` — empty working directory for simulation outputs.
- `proposal/` — your output goes here, organized as `proposal/code/`, `proposal/logs/`, `proposal/attempts_log.md`, with the final paper at `proposal/report.tex`.

### Compute budget

Hardware: CPU machine with 32 vCPU cores, ~60 GB RAM, ~470 GB disk. No GPU, and no swap. Treat ~50 GB as a hard memory ceiling (the node has ~60 GB RAM, no swap, and the OS + agent harness need headroom): size grids, dense matrices, ensembles, batch sizes, and parallel workers to keep peak resident memory under it; estimate/monitor peak memory before scaling up and prefer chunked / streaming / sparse computation. A process that exhausts RAM is killed by the OS.

The scheme is local and embarrassingly parallel across ensemble members; with 64 cores you can run ensembles in parallel. The laser-plasma run is substantially more expensive than the linear-wave demo. Two-dimensional or three-dimensional configurations are substantially more expensive than 1D — plan accordingly.

### Autonomy and external resources

Make your own decisions and justify them in the write-up. Do not ask questions.

Internet search is permitted and encouraged. Use it for literature lookup, library documentation, citations. Read existing follow-up work to the original paper for context, to identify canonical test problems plasma physicists study, and to differentiate your contribution. Cite anything you use.

**On replication.** Two settings probing similar physics naturally lead independent researchers to similar canonical test choices (multi-species dispersion, magnetized plasma modes, ray-tracing through inhomogeneous media). That kind of convergence is fine and expected. What is not allowed is substantive replication of an existing follow-up: the same set of tests, the same identified failure modes, the same proposed fixes, the same analytical conclusions, in a structure that mirrors prior work rather than developing its own. If the evaluator finds that your submission is substantively a reproduction of an existing follow-up's contribution, your submission will be severely degraded. The contract evaluates your scientific work, not your literature search.

### No reward hacking

After your submission, your paper, logs, and code will be inspected by an evaluator. These rules are disclosed up front so the contract is clear:

1. **Number traceability.** Every quantitative claim in your paper must trace to a corresponding log entry with the same value. Numbers in the paper that do not appear in any log will be flagged as potential fabrication.
2. **Pre-stated predictions and consistent tolerances.** For each test, the analytic prediction (scope tests) or failure criterion (boundary tests) must be stated in the Methodology section *before* the results section. Tolerance values must be stated once and used consistently in Results and tables, with no post-hoc adjustments.
3. **Baseline honesty.** Analytic predictions you cite must come from the original paper, standard literature references, or a derivation included in your paper. Invented analytic results presented as standard will be flagged. Agent-derived predictions are checked for mathematical soundness.
4. **Honest failure reporting and main/appendix placement.** If a chosen test does not work as intended, report it. Misplaced work (results in the main paper that turn out to be shallow, incorrect, or contradicted by logs) is a more serious integrity issue than honest placement in the appendix.
5. **No hardcoded reference values.** Your simulation code must not contain hardcoded analytic-prediction values used to bypass actual time evolution.
6. **Effort evidence.** Any failed or unresolved work claimed in the paper must have a corresponding `proposal/attempts_log.md` entry documenting multiple distinct attempts. Claims of effort without an attempts log will be read as not having tried.
7. **Experiment completeness.** Every test described in the paper must have corresponding logs. Each test must include a logged convergence study at multiple resolutions.
8. **Code-paper alignment.** The Methodology section must accurately describe what the code actually does. The discrete action, equations of motion, gauge condition, integration scheme, any boundary-test modifications, and stated ensemble sizes and resolutions must match the implementation.
9. **Algorithm fidelity.** The implementation must be derived from the variational discretization of the scalar QED action [Eq. (14)–(15) of the paper], yielding the equations of motion [Eq. (16), (17), (19)]. The Bianchi identities must hold at machine precision by construction, and total charge must be conserved to machine precision by construction. Implementations that produce visually similar plots but use a non-variational discretization will be flagged.

10. **No hardcoded reference values.** Your simulation code must not contain hardcoded analytic-prediction values that bypass actual time evolution. Reference values used for agreement checks (dispersion frequencies, wakefield amplitudes, conservation-law tolerances) must be computed from the stated formulas at evaluation time, not embedded as numerical literals in the simulation loop.

11. **No undisclosed external code.** All non-trivial methodology code in `proposal/code/` must either be authored by you during this run or be from standard scientific Python libraries (NumPy, SciPy, etc.). External implementations of lattice-field-theory schemes, gauge fixers, or solver routines copied or adapted without acknowledgment in `proposal/attempts_log.md` or the paper's Related Work section will be flagged.

12. **No substantive replication of published follow-ups.** Independent researchers extending the variational scalar QED scheme will naturally converge on some shared choices — that kind of convergence is fine and expected. What is not allowed is *substantive replication* of an existing follow-up: the same set of scope/boundary tests, the same identified failure modes, the same proposed fixes, in a structure that mirrors prior work rather than developing its own. If the evaluator finds your submission substantively reproduces an existing follow-up's contribution, your submission will be severely degraded.

## Deliverables — Here's what to hand in

Organize all outputs under `proposal/`. Submit:

**Codebase, logs, and attempts log:**
- `proposal/code/` — your implementation, organized with clear entry points and scripts that produce the runs reported in the paper.
- `proposal/logs/` — all run logs, including:
  - The implementation-verification runs (paper Sec. III.1 and Sec. III.2 reproductions), with full time-series of $\phi_v$ and $A_e$, conservation diagnostics, and the spectra used for comparison against the paper's figures.
  - The four test runs (scope ×2, boundary ×2), each with: input configuration, full time-series of conserved quantities, the simulation observable used for the agreement metric (scope) or the failure signature (boundary), ensemble seed list with mean ± std on observables, and timestamps.
  - The convergence studies at multiple resolutions for each of the four tests.
  - The analytic-prediction derivation scripts (where applicable).
  - The boundary-test fix implementations and their verification runs.
- `proposal/attempts_log.md` — a structured record of every distinct approach attempted for any test, with date, brief description, outcome, and log file references. Required when any test is documented as failed or unresolved.

Every quantitative claim in your paper must trace to a log.

**Follow-up paper** at `proposal/report.tex`, using the NeurIPS LaTeX style file `neurips.sty` provided in the workspace. The paper must include the sections: Introduction, Methodology, Experimental Setup, Experimental Results, Analysis, Conclusion, References. The Analysis or Conclusion section must include explicit discussion of the limitations of the work. An appendix may contain incomplete or shallow work as described in the Contract section.

The Experimental Results section must include a primary results table labeled `\label{tab:main_results}` with the following column headers. Six rows — two for the paper's reproductions, two for scope tests, two for boundary tests.

| Test | Category | Regime probed | Prediction | Tolerance / failure criterion | Outcome | $\|\Delta Q/Q\|$ | $\|\Delta\mathcal{U}/\mathcal{U}\|$ |
|------|----------|---------------|------------|-------------------------------|---------|------------------|-------------------------------------|
| Paper Sec. III.1 | verification | unmagnetized 1D dispersion | $\omega^2 = \omega_p^2 + k^2$ | | | | |
| Paper Sec. III.2 | verification | 1D laser-plasma at $a \approx 1$, $16$ | wakefield + Schwinger threshold | | | | |
| Scope test 1 | scope | | | | | | |
| Scope test 2 | scope | | | | | | |
| Boundary test 1 | boundary | | | | | | |
| Boundary test 2 | boundary | | | | | | |

The "Outcome" column is free-form: for scope tests, a quantitative agreement statement; for boundary tests, the observed failure plus the status of the proposed fix.

Beyond the main results table, include: convergence-study plots for each test; spectra or trajectory overlays comparing simulation to prediction for scope tests; conservation-law time-series traces for at least one realization per test; identification of each test's primary diagnostic with the connection argued in Methodology; first-principles reasoning for each boundary test's identified failure; and a discussion of the proposed fixes including their derivation, action-level documentation, implementation, verification, and an explicit statement of which invariants the fixes preserve.

The Analysis section is for additional intellectual content beyond the literal requirements of the contract. Use it to synthesize across the four tests (what do the failures collectively reveal about the framework's structure?), identify patterns the contract didn't directly require (e.g., scaling behavior, unexpected correlations between conserved quantities, sensitivities you noticed), discuss implications for future extensions of the framework, or pose questions that your work raises but does not answer. A strong Analysis section is what separates an exemplary submission from a merely complete one — it demonstrates that the agent thought about the work beyond the mechanical contract.

Incomplete or shallow contract work does not belong in the Analysis section. It belongs in the appendix per the main paper / appendix structure described in the Contract section, with a corresponding entry in `proposal/attempts_log.md` documenting the attempts that were tried.

### Paper quality

Your submission is an academic paper, not a class project report. The writing should resemble what would be reviewed at a top scientific journal (Nature, Science, PNAS) in structure, integrity, depth, and language: a clear thesis stated up front, claims supported by quantitative evidence, prose rather than bullet enumeration in main body sections, careful citation, and explicit honest discussion of limitations.

Do not frame the paper as "this is a follow-up of paper [X]." Frame it as a self-contained scientific contribution that contains its own motivation, methods, results, and discussion, with the original paper cited where appropriate. Search the literature broadly to differentiate your contribution from other work in the area, and include at least as many references as the original paper.

**Independence from this task spec.** This task spec describes what reviewers will check; it does not describe the *shape* your paper should take. A reader of your paper who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. Concretely:

- The paper's title should describe your scientific contribution (a specific mechanism, regime, finding, or modification), not the structure of your investigation. Titles announcing the methodology ("Scope and boundaries of X", "Stress-testing X", "Two extensions and two failures of X") read as homework reports rather than journal papers. Real journal titles state the finding.
- The paper's section structure should be physics-driven, not contract-driven. Section headings, subsection titles, and table row labels should use physics-appropriate nomenclature, not the spec's terminology. The agent's tests can be described in terms of the regimes they probe (e.g., "Multi-species pair plasma," "Compact-gauge breakdown under sustained external fields") rather than as categories from this spec.
- The paper's introduction should motivate the contribution from the physics, not from the structure of the contract. A paragraph announcing "we choose four tests, two extending the verified scope and two identifying fundamental obstructions" is a direct transliteration of the spec — it tells the reviewer that the agent treated the spec as a task list. The paper's introduction should set up the science.
- The paper's main results table should organize tests by their physics content, not by their evaluation category. A reader of the table should learn what was probed, not how the agent self-categorized for grading.

**Rendered output verification.** Before submitting, render your LaTeX to PDF (e.g., `pdflatex report.tex`) and visually inspect the resulting PDF. Equations must not overflow page margins; tables must fit within the text width without column wrapping that breaks readability; figures must not be cut off. If formatting issues exist in the rendered PDF, revise the LaTeX. A submission with formatting issues in the rendered PDF will be flagged as not having been visually verified.

Good luck.