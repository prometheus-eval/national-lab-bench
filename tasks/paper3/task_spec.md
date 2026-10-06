# Task: Develop a Substantive Follow-up to "Convolutional Dictionary Learning: A Comparative Review and New Algorithms"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the convolutional dictionary learning (CDL) comparative review by Garcia-Cardona and Wohlberg, "Convolutional Dictionary Learning: A Comparative Review and New Algorithms," IEEE Transactions on Computational Imaging 4(3), 366–381 (2018). Your contribution is to **build a single operator-splitting / majorization-minimization template for the CDL dictionary update that abstracts the differently-derived dictionary-update algorithms as instances of one design space, extend that template to multi-channel signals so it expresses the distinct ways channel information can be represented, and use the template to reveal something the separate algorithms do not — an un-instantiated method the design space predicts, and an equivalence the shared structure exposes.** The deliverable is the framework and what it reveals, not a new benchmark number.

## The Problem — Here's why it matters

Convolutional sparse representations approximate a signal as a sum of a few filters convolved with sparse coefficient maps, $\mathbf{s} \approx \sum_m \mathbf{d}_m * \mathbf{x}_m$. Learning the filters — **convolutional dictionary learning (CDL)** — alternates a sparse-coding step (update the coefficient maps) with a **dictionary-update** step (update the filters). The dictionary update is the hard, expensive step: its frequency-domain normal equations $(\hat X^H \hat X + \sigma I)$ decompose per frequency into systems whose data-fidelity part has **rank $K$** (the number of training images) rather than the rank-1 structure of sparse coding, so the Sherman–Morrison shortcut that makes sparse coding fast does not apply directly. Every dictionary-update algorithm is, at bottom, a different way of handling this same rank-$K$ structure.

The original paper compares several such algorithms — an ADMM update with an equality constraint (with conjugate-gradient, iterated-Sherman–Morrison, or spatial-tiling inner solvers), an ADMM consensus update (and its parallel form), a frequency-domain "3D" variant, and a FISTA (proximal-gradient) update — and reaches two observations that motivate this follow-up. First, that two methods derived independently and presented as entirely distinct are in fact **the same class** (the "3D" method is the consensus method with the data-fidelity term and constraint expressed in the DFT domain). Second, that the choice between these methods is a choice among **shared design axes** (how the rank-$K$ structure is split, in which domain, with which inner solver), which is why the paper can explain *why* one wins over another from the structure of the linear system each solves.

Two openings follow. If different-looking dictionary updates are secretly instances of one structure, that structure can be written down explicitly as a **template** whose axes generate the known methods — and whose empty cells generate new ones. And the paper studies **grayscale** images almost exclusively; real signals are **multi-channel** (color, hyperspectral), and there is more than one way to add channel information — a single shared dictionary applied per channel with per-channel coefficient maps, a multi-channel dictionary with one shared coefficient map, or per-channel maps tied across channels by a joint-sparsity coupling. These are genuinely different representations, and it is not obvious which method suits which. A template that abstracts all of them — and reveals which axis choices are equivalent, which are new, and which suit which regime — is the contribution.

## Background — Here's what you need to know about the original paper

Read `paper.md` in full before beginning (including the supplementary sections). The pieces you will hold fixed, abstract, and extend:

- **The CDL functional.** $\arg\min_{\{\mathbf{d}_m\},\{\mathbf{x}_{m,k}\}} \tfrac12 \sum_k \big\| \sum_m \mathbf{d}_m * \mathbf{x}_{m,k} - \mathbf{s}_k \big\|_2^2 + \lambda \sum_{m,k} \|\mathbf{x}_{m,k}\|_1$, subject to $\|\mathbf{d}_m\|_2 = 1$, over $K$ training images. Non-convex jointly, convex in each block, solved by alternating a sparse-coding step and a dictionary-update step with auxiliary-variable coupling (a single inner iteration of each before alternating).

- **The dictionary-update subproblem.** Fixing the coefficients, the update is $\arg\min_{\mathbf{d}} \tfrac12 \|X\mathbf{d} - \mathbf{s}\|_2^2 + \iota_{C_{PN}}(\mathbf{d})$, where $\iota_{C_{PN}}$ is the indicator of the support-plus-unit-norm constraint set (its proximal operator is a support-restriction followed by normalization). This is the object your template must abstract. In the DFT domain the normal equations $(\hat X^H \hat X + \sigma I)\hat{\mathbf{d}} = \dots$ split per frequency into $N$ systems, each diagonal plus **rank $K$**.

- **The dictionary-update algorithms your template must abstract.** Each solves the same subproblem, differing only in how it handles the rank-$K$ structure:
  1. **ADMM with equality constraint** — split $\mathbf{d}=\mathbf{g}$; solve the full rank-$K$ system, by conjugate gradient (**CG**), iterated Sherman–Morrison (**ISM**), or spatial tiling to an equivalent rank-1 problem (**Tiled**).
  2. **ADMM consensus** (**Cns**, and its parallel form **Cns-P**) — give each image its own dictionary copy so the per-image systems are rank-1 (Sherman–Morrison-solvable), coupled by a consensus averaging-plus-projection step.
  3. **Frequency-domain "3D" consensus** — treat the $K$ images as one volume; a block-circulant data-fidelity operator diagonalized by a block DFT. **The paper proves this is exactly the consensus method after that change of basis** — the seed of the unification.
  4. **FISTA** — accelerated proximal gradient; no linear solve at all, only gradient steps and the constraint prox, which is why it is advantageous for the rank-$K$ update.

  Of these, SPORCO ships runnable, dispatchable solvers only for **ISM**, **CG**, **Cns/Cns-P**, and **FISTA** (via `cbpdndl.py`'s `dmethod`); **spatial tiling** and the **"3D" consensus** are described and analyzable in the paper but have no SPORCO solver. So the special cases you *recover and validate against SPORCO* are drawn from {ISM, CG, Cns, FISTA}; tiling and "3D" belong to your design-space analysis and are natural candidates for the re-derived equivalence (the paper's own "3D = consensus in the DFT domain"), not for special-case recovery.

- **Multi-channel CDL** (the extension axis; the paper's Sec. VI and its supplementary color experiment). For $C$-channel signals there is more than one representation: a **single-channel (shared) dictionary with per-channel coefficient maps**; a **multi-channel dictionary with a single shared coefficient map** (the paper's focus, Eq. 80/87–90); and per-channel maps **coupled by an inter-channel (joint-sparsity) term**. The paper shows the multi-channel *dictionary update* is no more expensive than grayscale (the per-channel dictionaries solve with a shared per-frequency factor), and notes a **duality**: the channel count $C$ controls the rank of the sparse-coding subproblem exactly as the image count $K$ controls the rank of the dictionary update. The paper deliberately leaves multi-channel parameter-selection and a fuller multi-channel study to future work.

- **Reference implementation and test protocol.** `code/sporco/` (SPORCO, by one of the authors) is the canonical implementation: the dictionary-update solvers live in `sporco/admm/ccmod.py` (ISM/CG/Cns), `sporco/pgm/ccmod.py` (FISTA), and the masked variants in `sporco/admm/ccmodmd.py`; the CDL drivers `sporco/dictlrn/cbpdndl.py` (which dispatches all four via a `dmethod` string) and `sporco/dictlrn/prlcnscdl.py` (Cns-P). SPORCO ships color-CDL examples for the **multi-channel-dictionary** representation (`examples/scripts/cdl/cbpdndl_pgm_clr.py` and the parallel-consensus color example), but these run at $\lambda = 0.2$; since the frozen $\lambda$ is $0.1$ and the CDL functional scales with $\lambda$ (it weights the entire $\ell_1$ term), build your multi-channel-dictionary reference by running SPORCO's own `ConvBPDNDictLearn` on the multi-channel-dictionary path at $\lambda = 0.1$ — symmetric with the single-channel-dictionary reference below — or re-run the shipped example at $\lambda = 0.1$; do **not** compare a $\lambda=0.1$ instance against the shipped $\lambda=0.2$ run, which minimizes a differently-weighted functional. It does **not** ship a plain **single-channel-dictionary** color reference: its only single-channel-dictionary color example (`cbpdndl_jnt_clr.py`) adds an inter-channel joint-sparsity term and uses a different $\lambda$, so it realizes the inter-channel-correlation representation, not the plain per-channel one. For the single-channel-dictionary representation, build your own plain reference with SPORCO's `ConvBPDNDictLearn` (ordinary CBPDN, no joint-sparsity term) at $\lambda = 0.1$ — do not import the joint-sparsity example's settings, which would change the frozen $\lambda$. The primary metric across the paper is the **CDL functional value at convergence**, evaluated with unit-norm filters, on MIRFLICKR-derived images ($M=64$ filters of $8\times8$ support, $\lambda=0.1$ for grayscale; $M=64$, $C=3$ for the color experiment). Filter constraint $\|\mathbf{d}_m\|_2=1$ enforced by the prox onto the constraint set.

## Contract — Here's exactly what counts as success

Your contribution is a **unifying dictionary-update template**: one operator-splitting / majorization-minimization derivation, parameterized by a small set of explicit design axes, from which the known dictionary-update algorithms and the multi-channel representations are recovered by *choosing axis values* — not by dispatching to, or relabeling, separate hand-derived algorithms. Success has three parts, all required.

### Operationalized success thresholds

**1. Special-case recovery.** Instantiate the template — by axis choices, through one shared driver, not by copying four separate solver bodies — to reproduce the grayscale dictionary-update algorithms, validated against SPORCO on a fixed input (same seed, initial dictionary, images, $\lambda$, ADMM penalty / PGM step, iteration budget, and the inner-solver configuration taken from SPORCO's implementation for that family — for instance the CG stopping tolerance and iteration cap for the CG solver, and the step-size / momentum / backtracking policy for FISTA; with SPORCO's own settings all four families are numerically deterministic, so the tolerance below is reachable when the configuration is matched). You must recover **at least three of the four** update families, and the three must include the two exact DFT-domain solves — **iterated Sherman–Morrison** and **ADMM consensus** — plus **at least one fundamentally different solver family** (the **FISTA** proximal-gradient update, or the **conjugate-gradient** inner solver). "Recover" means: the instantiated template's CDL-functional trajectory tracks SPORCO's reference trajectory to a relative deviation $\leq 10^{-3}$ at every logged iteration, and the converged dictionary matches SPORCO's to a relative Frobenius error $\leq 10^{-3}$. Recovering the fourth family, or matching to a tighter tolerance for the exact solves, is rewarded.

**2. Multi-channel instantiation.** From the *same* template, instantiate — as concrete, runnable methods — the two multi-channel representations the paper derives: a **multi-channel dictionary with a shared coefficient map**, and a **single-channel (shared) dictionary with per-channel coefficient maps**. Each must run to completion on real color (multi-channel) images and reduce the multi-channel CDL functional to a converged value (below $0.6\times$ its initial value, and within a few percent of a matched reference run for that representation, both run at the frozen $\lambda=0.1$ — SPORCO's own `ConvBPDNDictLearn` on the multi-channel-dictionary path at $\lambda=0.1$ for the multi-channel-dictionary representation, not the shipped color examples which run at $\lambda=0.2$; and your own plain-CBPDN run at $\lambda=0.1$ for the single-channel-dictionary representation, as noted in the Background). These two representations are the core of "abstract all the different multi-channel representations." An **inter-channel-correlation** representation (per-channel maps coupled by a joint-sparsity term across channels) is rewarded, not required — its distinguishing content lives in the sparse-coding subproblem, which is held fixed here, so it does not exercise the dictionary-update template that is this contribution's object; expressing it as a template instance is credited as breadth, not as one of the required representations.

**3. Reveal.** The point of a genuine template is that it generates more than it was built from. You must demonstrate **at least one method that the template's design space contains but the existing algorithms and their published multi-channel extensions do not** — a representation × axis-choice combination that is not any method already implemented in SPORCO or in the published literature, implemented, run on data, and shown to converge. In particular, it may **not** be the inter-channel joint-sparsity color CDL that SPORCO already ships (`cbpdndl_jnt_clr.py` / `ConvBPDNJoint`), nor the masked CDL variants SPORCO ships (`ccmodmd.py` / `cbpdndl_md_clr.py`): these are already instantiated, so they earn breadth credit as recovered/analyzed cells, not as the reveal. Re-authoring an existing method's code from scratch does not make it un-instantiated. You must also **re-derive at least one equivalence** in the multi-channel setting as a consequence of axis choices (in the way the paper derives that "3D" is consensus in the DFT domain), rather than asserting it. Because a new cell has no external reference, validate it by an internal-consistency check: setting the new axis to a degenerate value must collapse it to a case that *does* match SPORCO within $10^{-3}$. This is the decisive test that the template is a genuine abstraction and not a wrapper.

The characterization — which axis choices are equivalent, which are genuinely new, and which representation/solver suits which $(C, K, \text{channel-correlation})$ regime — is the substance. The recovery match, the converged multi-channel runs, and the working reveal cell are the anchors that make it verifiable.

### Held fixed (parity constraints)

These are not part of the design space; they must remain identical to the paper so that what you abstract is the paper's method:

- **The CDL functional and the constraint.** The functional above and the support-plus-unit-norm constraint $C_{PN}$ (with its normalization prox), computed exactly as the paper defines them — no rescaling, no alternative normalization, filters unit-norm at the reported convergence point. For multi-channel, the multi-channel functional of the paper's Sec. VI.
- **The alternating-minimization structure.** Auxiliary-variable coupling with a single inner iteration of each subproblem before alternating, and the sparse-coding stage (CBPDN) at $\lambda = 0.1$. Your contribution is the dictionary-update template, not a new sparse-coding algorithm or a new coupling strategy (abstracting those too is rewarded, not required — see below).
- **The reference algorithms as recovery targets.** The special cases you claim to recover are defined by SPORCO's implementations (`ccmod.py`, `pgm/ccmod.py`, `cbpdndl.py`); your recovery is validated against those runs, not against a re-derivation you also authored.
- **The data and protocol for the anchors.** MIRFLICKR-derived images with the paper's preprocessing; the recovery checks at small scale (small $K$, $N$, $M$, a modest iteration budget) where the trajectory-matching tolerance is the grade; the multi-channel convergence runs on color images at the paper's color-experiment scale. Training-image selection by a logged seed; the same inputs for the template instance and the SPORCO reference.

Changing any of these to make an instantiation match more easily — and presenting the result as a recovered special case — is not a contribution.

### What you may design

The template itself: the master operator-splitting / MM problem, the design axes (for instance the splitting variable, the coupling/constraint form, the domain and change-of-basis, the inner solver for the diagonal-plus-low-rank system, the channel-coupling representation, and whether coupling is enforced in the linear solve or in a prox — the exact axis set is yours to define and justify), and the mapping from axis values to concrete updates. The multi-channel representation you fold in beyond the two required ones. The un-instantiated cell you fill, and the equivalence you re-derive. The regimes over which you characterize which axis choices win. State the template and its axes in a plan file **before** the validation runs.

### Anti-replication and the genuine-abstraction bar

The reviewed paper itself already carries these dictionary updates to $C$ channels: its Sec. VI derives the multi-channel-dictionary representation, names the single-channel-dictionary representation, states that all four updates apply on $C$ channels, and gives the $K$↔$C$ duality, and its supplementary color experiment runs the updates on $C=3$ color data. The authors have elaborated this further in subsequent work. Because this material is in the paper you are building on, reproducing it — including running both required representations to convergence on color — is **required baseline work, not the contribution.** **Merely extending the four updates to multiple channels and reporting that they scale is not a contribution here; it is replication of the reviewed paper's own Sec. VI.** Independent researchers converge on shared choices, and that is fine, but a submission whose contribution is "we ran the existing updates on color/hyperspectral data" — with no template that generates them from shared axes, no filled un-instantiated cell, and no new equivalence — substantively reproduces existing follow-up work and will be severely degraded. What distinguishes your contribution is the unification itself and what it reveals.

Relatedly, a template that is an **if/else dispatcher** over the four existing solvers, or a **relabeling wrapper** that puts common notation over separately-derived algorithms without a shared derivation, is not a genuine abstraction: removing a branch removes a method, and no new method can arise from a branch that was not pre-written. The reveal requirement (part 3) is what separates the two — a dispatcher cannot produce a working method from an empty cell; a genuine template can.

### Honest unfinished work

Any part you attempt but cannot complete must be documented in `proposal/attempts_log.md` with what you tried and why it did not work. A template that genuinely recovers three updates and one representation with an honest account of what did not generalize is worth more than an overclaimed four-update, all-representation template whose instances do not actually run. The main paper carries the work you claim is complete; the appendix is a fallback for attempted-but-incomplete work, not a place to relegate contract content.

## Evaluation — Here's how you'll be judged

Your submission is graded on a leaf-level rubric of **gates** (failing one precludes a high score) and **differentiators** (which separate a solid contribution from an excellent one), in priority order **integrity > completeness > honest accounting > quality**.

**Gates (all must pass):**
1. **Integrity.** Quantitative claims trace to logs; the recovery match is against actual SPORCO runs (not a hardcoded or hand-fit trajectory); the template in the paper is what the code implements; the frozen functional/constraint/coupling are respected (or any deviation is disclosed); the contribution is differentiated from the published multi-channel extension.
2. **Genuine abstraction.** The template is a single shared derivation instantiated by axis choices — not an if/else dispatcher or a relabeling wrapper — evidenced by a shared driver in the code and by the reveal cell running.
3. **Special-case recovery.** At least three of the four updates recovered (including ISM and consensus, plus one different solver family) to the stated tolerance against SPORCO.
4. **Multi-channel instantiation.** Both required representations realized from the template and run to convergence on color data.
5. **Reveal.** At least one un-instantiated cell implemented, run, and internally validated; at least one equivalence re-derived from axis choices.
6. **Honest accounting.** All recovery cells, representations, and reveal attempts reported; nothing cherry-picked.

**Differentiators (among gate-passing submissions):** the depth and correctness of the **design-axis analysis** (are the axes real and orthogonal; do known equivalences fall out as algebra); the breadth of **recovered special cases** (all four, tight tolerances) and **representations** (the inter-channel-correlation model; masked CDL); the number and interest of **filled un-instantiated cells**; the **selection map** across $(C, K, \text{correlation})$ regimes that exploits the $K$↔$C$ duality — the observation that the channel count $C$ plays for the sparse-coding subproblem the rank role the image count $K$ plays for the dictionary update (the template need only recognize and use this correspondence in the regime map; it need not abstract the sparse-coding stage); the depth of the **mechanism** (why axis choices drive cost/performance from the linear-system structure); the quality of the **failure analysis** (where the abstraction is incomplete, which cells do not work and why), the **compute-cost analysis**, the **originality and literature engagement**, and the **writing**.

A submission that overclaims — a template presented as unifying that is a dispatcher, or recovered "special cases" that do not actually match the reference — is worse than one that honestly reports a narrower but genuine template and explains its limits.

### Paper quality

This is an academic paper, not a class-project report; it should read as it would in a strong journal. Do not frame it as "a follow-up to Garcia-Cardona and Wohlberg." Frame it as a self-contained contribution: a unifying framework for convolutional dictionary updates and what it reveals for multi-channel signals, citing the original where appropriate.

**Independence from this task spec.** A reader who has not seen this spec should not be able to reverse-engineer it from your paper's structure or terminology. Do not reuse spec phrases ("special-case recovery," "the reveal," "design axis," "gate/differentiator") or transliterate its section order; use the natural terminology of sparse optimization. The title should name the scientific contribution, not the structure of the investigation. Organize the results table by method/representation content, not by evaluation category.

**Rendered output verification.** Compile the LaTeX to PDF and inspect it: equations must not overflow the margins, tables must fit, figures must not be cut off.

## Rules — Here are the rules

### Available resources

Everything you need is in the working directory.

- `paper.md` — markdown of the original paper (Garcia-Cardona & Wohlberg, 2018), including the supplementary sections (the derivations of the updates, the consensus/"3D" equivalence, the penalty-scaling analysis, and the multi-channel and color material).
- `paper.pdf` — the original paper PDF.
- `images/` — figures from the original paper (algorithm diagrams, convergence and parameter-sensitivity plots, learned dictionaries).
- `code/sporco/` — SPORCO, the reference implementation. The dictionary-update solvers (`admm/ccmod.py`, `pgm/ccmod.py`, `admm/ccmodmd.py`), the CDL drivers (`dictlrn/cbpdndl.py`, `dictlrn/prlcnscdl.py`, `dictlrn/cbpdndlmd.py`), the convolutional-representation and FFT utilities (`cnvrep.py`, `fft.py`, `linalg.py`), the CuPy/GPU mirror (`sporco/cupy/`), and runnable grayscale and color CDL examples (`examples/scripts/cdl/`). Use SPORCO as the reference to recover and validate against; do not reimplement its provided components from scratch and present that as your framework.
- `data/mirflickr25k/mirflickr25k.zip` — the MIRFLICKR image dataset used by the paper (unzip if needed). SPORCO also bundles standard color images used by its color-CDL examples, which are acceptable for the multi-channel convergence runs.

Standard scientific Python is available (NumPy, SciPy, and SPORCO itself; CuPy / PyTorch for GPU primitives if you use them). SPORCO's batch dictionary-update solvers run efficiently on CPU (FFT-bound); a GPU is available but not required for these solvers.

### Autonomy and external resources

**Internet search is permitted and encouraged.** Use it for the sparse-optimization, CDL, operator-splitting / majorization-minimization, and multi-channel / hyperspectral CSC literature; for SPORCO and library documentation; and to read work published since the original paper so you can position and differentiate your contribution. Cite anything you use.

**On replication.** Convergence with prior choices is fine and expected; substantive replication of an existing follow-up's contribution — the same representations, the same carried-over updates, the same conclusions, in a structure that mirrors prior work — is not, and will be severely degraded if found. The contract evaluates your scientific work, not your literature search.

**Working-directory constraint.** Do not read local files outside the provided working directory.

**No questions.** Make your own decisions and justify them in your write-up.

### Compute budget

- **Hardware.** 1× H100 80GB GPU, 16 vCPU, 128 GB system memory, 500 GB working disk. The dictionary-update solvers are FFT-bound convex subproblems, not deep learning; the recovery checks run at small scale and the multi-channel convergence runs at the paper's color scale, all comfortably within budget on CPU.
- **Where compute goes.** Favor breadth and correctness of the recovery/instantiation/reveal matrix over large-scale runs. Keep the recovery checks small (small $K$, $N$, $M$, a modest iteration budget) where the trajectory-matching tolerance is the grade; reserve larger color runs for the convergence tier and run only a few, with a small number of seeds.

### No reward hacking

Your submission and logs will be inspected for the following. These are disclosed up front so the contract is clear.

1. **Number traceability.** Every quantitative claim (functional values and trajectories, recovery deviations, convergence values, timings, axis/hyperparameter settings) must trace to a log entry with the same value.
2. **Pre-stated framework and consistent reporting.** The template, its design axes, the special cases and representations you will recover, and the un-instantiated cell you will fill must appear in a timestamped plan file before the validation runs; settings quoted in the paper must match the logs. Post-hoc refinement is allowed only if disclosed.
3. **Test / evaluation integrity.** Recovery deviations, convergence values, and reveal-cell results must come from actual runs at the stated configurations, not from analytical estimates or interpolation.
4. **Baseline honesty.** SPORCO reference numbers must come from your own logged SPORCO runs (or the paper with specific citations); mixing sources without distinction is selective framing.
5. **No selective reporting.** Every recovery cell, every representation, and every reveal attempt (including ones that did not work) must be reported, not only the successes.
6. **No hardcoded reference values.** The code must not contain hardcoded functional targets, SPORCO trajectories, or converged dictionaries used to fake a recovery match; references are computed by running SPORCO at runtime.
7. **Experiment completeness.** Every experiment described in the paper has corresponding logs (recovery runs and their SPORCO references, multi-channel convergence runs, reveal-cell runs and the degenerate-collapse check).
8. **Code-paper alignment.** The Methodology must accurately describe the template, its axes, and the instantiations, and must match the implementation.
9. **Genuine-abstraction integrity.** The template must be a single shared derivation and driver instantiated by axis choices, not an if/else dispatcher or a relabeling wrapper over SPORCO's separate solvers. Claiming a unifying template while the code dispatches to four separate hand-derived solvers is a misrepresentation.
10. **Recovery-matching integrity.** A claimed special-case recovery must be validated against an actual SPORCO run on identical inputs, logged, not against a hand-tuned or hardcoded trajectory, and not by loosening the frozen functional/constraint to force a match.
11. **Held-fixed disclosure.** Any deviation from the §"Held fixed" items — the functional, the constraint, the coupling/sparse-coding stage, the reference-defined recovery targets — must be disclosed transparently. Silent deviation paired with a compliance claim is a deception.
12. **No substantive replication of published methods.** If your representations, carried-over updates, and conclusions substantively replicate the published multi-channel extension of this work without acknowledgment and differentiation, that will be flagged. Acknowledged convergence is fine; silent structural replication is not.
13. **No axis or representation switching after the fact.** The design axes, recovered special cases, and representations committed in the plan file may not be silently changed after seeing results; abandoned choices go in the attempts log.
14. **Codebase inspection evidence.** `proposal/codebase_notes.md` must document what you inspected and reused in SPORCO and how your template relates to it; an empty file paired with a from-scratch reimplementation of provided components is flagged.
15. **No undisclosed external code.** Non-trivial methodology code must be authored by you during this run or come from SPORCO / a named standard library with attribution.
16. **Effort evidence when partial work is claimed.** If any contract item is described as attempted-but-incomplete, `proposal/attempts_log.md` must contain at least two documented attempts with reasoning.

## Deliverables — Here's what to hand in

Organize all outputs under `proposal/`. Submit:

**Codebase, logs, codebase notes, and attempts log.**
- `proposal/code/` — the template implementation (one shared driver + the axis-instantiation machinery), the recovery-validation harness that runs SPORCO and compares, the multi-channel instantiations, and the reveal cell.
- `proposal/code/framework_plan.{py,md,json}` — the template, its design axes, the special cases and representations to be recovered, and the un-instantiated cell to be filled, with a timestamp predating the validation runs.
- `proposal/codebase_notes.md` — what was inspected and reused in SPORCO and how the template relates to its solvers.
- `proposal/logs/` — recovery runs (template instance and the matched SPORCO reference, functional trajectories, per-filter norms, converged dictionaries, deviations); multi-channel convergence runs per representation (functional trajectory, seeds); reveal-cell runs and the degenerate-collapse consistency check; any characterization sweeps.
- `proposal/attempts_log.md` — distinct documented attempts for anything unresolved or partial.

Every quantitative claim in the paper must be reproducible from a log entry.

**Follow-up paper.** A LaTeX paper at `proposal/report.tex`, compiled to `proposal/report.pdf`, using the provided `neurips.sty`. It must contain: motivation and the framework's scientific question; the template — the master operator-splitting / MM problem, the design axes, and the mapping to concrete updates; the special-case recovery (how each update is an axis setting, and the validation against the reference); the multi-channel extension (each representation as a template instance, and its convergence); the reveal (the un-instantiated method and its validation, and the re-derived equivalence); a characterization of which axis choices are equivalent, which are new, and which suit which regime; a mechanism discussion (why the axes drive cost/performance from the linear-system structure); a failure analysis (where the abstraction is incomplete); a compute-cost discussion; engagement with related work (including the published multi-channel extension, differentiated); and limitations and conclusions. Choose section names that fit the contribution rather than transcribing this spec.

It must include a primary results table labeled `\label{tab:main_results}` reporting, for each recovered update and each representation, the method, the representation, the axis settings that instantiate it, the recovery deviation against SPORCO (where applicable), and the converged functional. A skeleton is below; the row labels are placeholders — name the methods and representations as fits the paper.

| Method | Representation | Axis settings | Recovery deviation vs reference | Converged functional |
|---|---|---|---|---|
| (recovered update) | (grayscale / multi-channel) | | | |
| (multi-channel instance) | | | | |
| (reveal cell) | | | (degenerate-collapse check) | |

Beyond the table, the results should include: the design-space grid (axes × cells, marking recovered / new / equivalent / empty); functional-vs-iteration curves overlaying each template instance on its SPORCO reference; the multi-channel convergence curves per representation; and the reveal cell's convergence with its degenerate-collapse consistency check. Additional analyses are encouraged.

Good luck.
