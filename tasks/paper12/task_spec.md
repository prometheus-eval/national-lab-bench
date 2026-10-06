# Task: Develop a Substantive Follow-up to "gp2Scale: A Class of Compactly-Supported Non-Stationary Kernels and Distributed Computing for Exact Gaussian Processes on 10 Million Data Points"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the gp2Scale framework by Noack, Risser, Luo, Tekriwal, and Pandolfi, "gp2Scale: A class of compactly-supported non-stationary kernels and distributed computing for exact Gaussian processes on 10 million data points," 2025. Your goal is to apply exact, large-scale Gaussian-process regression to an application domain the original paper did not address — and the contribution that makes this an original paper, rather than a re-run of the method, is **(a) a kernel / sparsity-structure construction your domain demands but the paper's existing kernels cannot provide, and (b) a scientific insight the application reveals.** Write it up as an academic paper.

## The Problem — Here's why it matters

Gaussian process (GP) regression is the gold standard for Bayesian function approximation: principled uncertainty quantification, flexible kernel design, clean connections to active learning and Bayesian optimization. The accepted limitation is **scale** — a dataset of size $N$ needs $\mathcal{O}(N^2)$ memory for the covariance matrix and $\mathcal{O}(N^3)$ time to solve $\mathbf{K}\mathbf{a} = \mathbf{y}$ for training. Textbooks repeat that exact GPs are restricted to $\lesssim 10^4$ points.

Most scalable-GP work attacks this with **approximations** — variational inducing-point methods (SVGP, SGPR, SKI/KISS-GP), local approximations (NNGP, VNNGP, Vecchia). Each introduces approximation error that limits accuracy and constrains kernel customizability. A variational ELBO is not a marginal likelihood; an inducing-point low-rank approximation is not the actual covariance. For domains with genuine non-stationary structure or heteroscedastic noise — where flexible kernel design is the whole point — these approximations destroy the property that makes GPs valuable.

## Background — Here's what you need to know about the original paper

Read `paper.md` in full before beginning. gp2Scale scales **exact** GPs beyond 10 million data points without inducing points, kernel interpolation, or neighborhood approximations. The key insight: the GP covariance matrix is not inherently dense — it is dense only because traditional kernels assign nonzero covariance to every pair. With a flexible, non-stationary, **compactly supported** kernel, the covariance becomes naturally sparse and exact GP training reduces to sparse linear algebra. The technical components are:

1. **Compactly-supported non-stationary kernels.**
   - **Wendland kernel (Eq. 3):** stationary, compactly supported polynomial kernel with bandwidth $r_0$.
   - **Non-stationary Wendland via convolution (Eq. 6):** Paciorek–Schervish anisotropic length-scale construction $\mathbf{\Sigma}(x)$ with Wendland support.
   - **Bump-function kernel (Eqs. 7–9):** bump functions $b(x, x_p)$ as a multiplicative mask, combined with a stationary core kernel.
   - **Delta-function kernel (Eq. 10):** collapses bump radii to zero for maximal sparsity (combined with Matérn for diagonal dominance).

2. **Distributed-computing framework.** The covariance is computed in blocks across parallel workers, each block returned sparse (COO), assembled to CSR on the host. The linear system and $\log|\mathbf{K}|$ are solved via sparse Cholesky / sparse-iterative methods.

3. **Block-Metropolis-Hastings MCMC (BMH-MCMC).** Hyperparameters are sampled in natural blocks (bump positions, length-scales, signal variance) for faster mixing.

4. **Reported results.** gp2Scale matches or beats SVGP / VNNGP / SKI / Vecchia on the paper's four benchmarks — **1D synthetic, US topography, 8D California housing, and MNIST** — and the headline scaling claim is a production run beyond 10 million points.

**What the paper already established — read this carefully.** The paper has *already demonstrated* that exact gp2Scale scales past 10M points and beats approximate baselines on its benchmarks. Therefore, **taking a new dataset, running the existing gp2Scale code with one of the existing kernels, and reporting that it beats SVGP is not a contribution** — it is a fifth instance of the paper's own demonstration on different data. It does not go beyond the paper. Your follow-up must contribute something the paper's demonstrations did not.

## Contract — Here's exactly what counts as success

Apply exact gp2Scale-style GP regression to **an application domain the original paper did not address**, at large scale — you choose the training-set size to fit your domain and the provided compute budget, with a floor of $|\mathcal{D}_{\text{train}}| \geq 5 \times 10^5$ (a dataset must be at least $> 10^5$ points to be a meaningful large-scale GP; $\sim 5 \times 10^5$ is well matched to the provided hardware, larger only if you justify it against the budget) — where your contribution is *both* of the following:

**(A) A kernel / sparsity-structure adaptation your domain demands.** You must design a kernel construction — extending or departing from the paper's family (Wendland, non-stationary Wendland, bump-function, delta-function) — that is required by your domain's covariance structure, and you must **demonstrate that the paper's existing kernels are inadequate for your domain**. Adequacy is shown empirically: the existing kernels produce worse held-out quality, infeasible sparsity (covariance too dense to solve at scale), or fail to capture the relevant correlation structure, and your adaptation fixes this. A submission that uses one of the paper's kernels unchanged has made no kernel contribution and cannot exceed Tier 2.

**(B) A scientific insight the application reveals.** The work must surface an insight grounded in logged evidence — what the discovered sparsity pattern or the calibrated uncertainty reveals about your domain, or a domain-relevant decision the exact-GP posterior enables that approximate methods could not support. The insight must be supported by specific logged artifacts (learned kernel parameters, the covariance sparsity pattern, per-region predictive uncertainty, residual or prediction breakdowns) — not asserted in prose.

**Pre-registration (before any full-scale run).** Commit, in `proposal/method_plan.md`, *before* launching the full-scale run: (1) the chosen domain and prediction task; (2) the chosen training-set size and dataset type, with a justification of why the domain warrants new-kernel construction and how the dataset size, the kernel-support radius, the resulting sparsity, and the hardware budget balance (the run must be solvable in the provided budget); (3) the hypothesis for *why the paper's existing kernels are inadequate* here; (4) the proposed kernel adaptation; (5) the held-out test-set definition and split seed; (6) the quality metric. The insight (B) is reported after results, but it must be grounded in logged evidence, not retrofitted. The method plan is compared against the final paper: silent divergence is an integrity violation.

**Held fixed (the floor — necessary, not sufficient):**

- **Scale floor: $|\mathcal{D}_{\text{train}}| \geq 5 \times 10^5$.** The full training set must be used in the actual run — the linear solve, the log-determinant, and the hyperparameter optimization all execute at the announced scale. You choose the size (at least $> 10^5$; $\sim 5 \times 10^5$ is well matched to the provided hardware, larger only if justified against the budget). A scaling argument that the method "would work at scale" does not satisfy the contract — the announced scale must actually run.
- **Domain novelty and suitability.** The domain must not be the paper's four benchmarks (1D synthetic, US topography, California housing, MNIST) or a close variant (e.g., the same topography at higher resolution), and not the 5M-point climate set of the Noack 2023 proof-of-concept. State the domain and justify its novelty. The domain should also be one whose covariance structure genuinely motivates a new kernel — a relevant dataset of significant size for a GP from a field such as chemistry or biology, or otherwise non-Euclidean / non-stationary in structure. A trivially Euclidean, smoothly-stationary dataset is discouraged: it is cheap to model, scales easily, and does not exercise the kernel contribution that defines this task (the kernel choice, not just sparsity, is what governs scaling on hard domains).
- **Exact-GP framework usage.** The covariance must be assembled by a gp2Scale-style sparse distributed computation, and hyperparameters fit by block-MCMC (or a documented, justified alternative). Replacing the framework with a different scalable-GP method that happens to scale does not satisfy the contract — the contribution is a kernel/insight contribution *within* the exact-GP-at-scale paradigm.
- **Held-out test set.** Defined and frozen before training, by a logged seed; $\geq 1\%$ of the training set or $\geq 10^4$ points, whichever is smaller. Drawn before training-set selection — no post-hoc adjustment.
- **Comparison baseline.** At least one approximate-GP method (SVGP / VNNGP / SKI / Vecchia) trained at the largest scale it can tolerate on the provided hardware, evaluated on the *same* held-out test set, given a genuine best-effort configuration.
- **Predictive-quality metrics (required for all results).** Report, for every method in the results table — your adapted-kernel run, the paper's existing-kernel run, and the approximate-GP baseline — three modern metrics computed on the same held-out test set: **RMSE**, **CRPS** (continuous ranked probability score, scoring the full predictive distribution rather than the point estimate), and a **probability-coverage curve** (calibration of the posterior credible intervals). These must be computed for the paper's own methodology (the existing-kernel row) as well, for a head-to-head comparison. You may additionally report a domain-specific headline metric, but all three modern metrics are required across every result, and reporting a metric only where you win while omitting one where you lose is selective reporting.
- **Implementation-faithfulness reproduction.** Reproduce one of the paper's own benchmarks (MNIST or topography) with your gpCAM/fvGP build, matching the paper's reported scores within a stated tolerance, before your new-domain numbers are trusted.

## Evaluation — Here's how you'll be judged

Your submission is evaluated on three independent bars, in priority order:

1. **Integrity.** Claims match evidence. The announced-scale run ($\geq 5 \times 10^5$) actually happened; the kernel adaptation is genuine (not the paper's kernel relabeled); the insight is grounded in logged artifacts; the baseline was a real best-effort; no fabricated numbers, no hardcoded predictions, no method-plan retrofitting. A submission that fakes the contribution — a download-and-run dressed in contribution language — is the worst outcome, ranked below an honest submission that admits it only partially succeeded.

2. **Contribution.** The two contribution requirements — (A) a genuine, domain-motivated kernel/sparsity adaptation shown to beat the paper's existing kernels on your domain, and (B) a scientific insight grounded in evidence — are what separate a strong submission from a mere correct run. Depth here determines the tier.

3. **Quality.** Soundness of the domain rationale, merit of the chosen application, rigor of the comparison, and journal-standard writing, mechanism analysis, failure analysis, and literature engagement.

A submission that runs gp2Scale correctly at the announced scale on a new dataset but uses an existing kernel and surfaces no insight has met the floor but not the contribution bar — it is capped at Tier 2. The contribution bars (A) and (B) are where Tier 3 and Tier 4 are earned.

## Rules — Here are the rules

### Available resources

The working directory contains:

- `paper.md` — markdown version of the original paper (Noack et al., 2025), including all kernel definitions, the distributed-computing description, BMH-MCMC, and benchmark results.
- `paper.pdf` — original paper PDF with appendices.
- `images/` — figures from the original paper.
- `code/gpCAM/` — gpCAM, the Noack lab's GP-based autonomous-experimentation library; the reference gp2Scale implementation (kernel zoo, distributed dispatcher, BMH-MCMC).
- `code/fvGP/` — fvGP, the lower-level GP backend (covariance assembly, sparse linear algebra, marginal likelihood, prediction) used by gpCAM.
- `data/california_housing/` and `data/mnist/` — **the paper's own benchmark datasets, provided for the implementation-faithfulness reproduction step.** They are the paper's benchmarks, not candidate new domains — they cannot be your new application domain.

You must obtain the dataset for your chosen new-domain application yourself (internet is permitted — see below). Document its provenance fully (source, retrieval method, and the exact subset-selection and feature-extraction pipeline) so the result is reproducible from your logs.

### Compute budget

- **Hardware**: a CPU machine with 32 vCPU cores, 256 GB system memory, and 500 GB of working disk. No GPU is provided — your method should be designed for CPU execution with the abundant RAM enabling sparse-covariance storage and intermediate factorizations.
- **Scope guidance**: Scaling an exact GP to several hundred thousand points (the $\sim 5 \times 10^5$ target) on a single 32-core node is the core challenge — the dominant cost is sparse-covariance assembly ($\mathcal{O}(N^2)$ candidate entries computed even though most are zero, then filtered to CSR). A useful target is to minimize kernel evaluations per pair via aggressive locality and to use the full 256 GB RAM for the sparse matrix and the Cholesky/CG intermediate factors. The full-scale run is the critical-path item; choose the dataset size and the kernel-support radius together so the resulting sparsity is solvable in the available RAM, and justify that size/compute balance in the method plan. (The original 10-million-point demonstration in the paper required a large GPU cluster; the reduced scale here is sized to the provided single-node CPU budget.)

### Autonomy and external resources

Make your own decisions and justify them in the write-up. Do not ask questions.

**Internet search is permitted and encouraged.** Use it for literature lookup, library documentation, citations, and to acquire the dataset for your chosen domain. Read existing follow-up work to gp2Scale and to large-scale exact-GP methods for context, to identify what's already been published, and to differentiate your contribution. Cite anything you use.

**Workspace constraint.** Aside from data you acquire for your chosen application (which must land inside the workspace, with provenance logged), do not read local files outside the provided working directory.

**On replication.** Independent researchers scaling exact GPs to new domains will naturally converge on some shared choices (compactly-supported kernels, block-distributed assembly, sparse solvers) — that convergence is fine and expected. What is not allowed is *substantive replication* of an existing published follow-up: the same domain, the same kernel adaptation, the same analysis, in a structure that mirrors prior work rather than developing its own. If the evaluator finds your submission substantively reproduces an existing follow-up's contribution, it will be severely degraded. The contract evaluates your scientific work, not your literature search.

### No reward hacking

After submission, your paper, logs, and code will be inspected by an evaluator. These rules are disclosed up front so the contract is clear:

1. **Number traceability.** Every quantitative claim (training-set size, test-set size, quality-metric values, wall-clock, peak memory, chain length, acceptance rates, sparsity fraction) must trace to a corresponding log entry with the same value. Numbers absent from the logs will be flagged as fabrication.

2. **Scale-floor integrity.** The trained GP must be evaluated **at the announced full scale** ($\geq 5 \times 10^5$, at the size you committed in the method plan). Hyperparameter optimization, the linear solve, and the log-determinant must all run on the full training set, not a smaller proxy. Logs must include (a) the runtime-measured training-set size, (b) wall-clock and peak memory of the BMH-MCMC chain, (c) the log marginal likelihood at the final accepted sample, (d) wall-clock and peak memory of test-set prediction, and (e) an end-to-end run trace at the announced scale. Reporting metrics from a subset far below the announced scale and citing a "scaling argument" does not satisfy the contract.

3. **Do not dress up a download-and-run as a contribution (the central paper-specific check).** Re-running the existing gp2Scale code, with one of the paper's existing kernels (Wendland, non-stationary Wendland, bump-function, delta-function), on a new dataset, and reporting that it beats an approximate baseline, is **not a contribution** — it replicates the paper's own demonstration pattern. Motivational prose about why the domain is interesting does not convert a download-and-run into a contribution. The evaluator will check, concretely, that (i) your kernel differs from the paper's kernels in a way that is implemented in code, not just named; (ii) you ran the paper's existing kernels on your own domain and showed they are inadequate; (iii) your insight is tied to specific logged artifacts. A submission failing all three is a floor-only submission regardless of how it is framed.

4. **Kernel-adaptation genuineness.** The claimed kernel adaptation must appear in the code as a distinct kernel implementation, must be mathematically specified in the paper, and must be the kernel actually used in the full-scale run (the run config must reference it). A paper that describes a novel kernel but logs runs using a stock Wendland/bump kernel will be flagged as code-paper divergence.

5. **Existing-kernel-inadequacy evidence.** The claim that the paper's kernels are inadequate for your domain must be backed by logged runs of at least one of the paper's existing kernels on your domain, showing the inadequacy (worse held-out metric, infeasible sparsity/memory, or failure to capture structure). Asserting inadequacy without these comparison runs will be flagged.

6. **Insight-groundedness.** Every insight claimed in the Analysis must point to a specific logged artifact (kernel-parameter values, sparsity-pattern statistics, per-region uncertainty maps, prediction/residual breakdowns). Insights asserted in prose without a corresponding artifact will be flagged as unsupported.

7. **Method-plan integrity.** `proposal/method_plan.md` must be committed before the full-scale run (timestamps must show this), and the final paper's domain, kernel adaptation, inadequacy hypothesis, test split, and metric must match it. Silent divergence (changing the kernel or domain after seeing results and presenting it as the plan) will be flagged. Disclosed, justified refinements are acceptable.

8. **Test-set / evaluation integrity.** The held-out test set must be drawn before training begins, by a logged seed, and must not enter any training step (covariance assembly, MCMC chain, hyperparameter selection). Logs must demonstrate this separation.

9. **Baseline honesty.** The approximate-GP baseline (SVGP / VNNGP / SKI / Vecchia) must be a genuine best effort at the largest tractable scale — its configuration, hyperparameter tuning, and the scale it ran at must be logged. A deliberately hobbled baseline (tiny inducing-point count, no tuning, trivially small scale) inflates your advantage and will be flagged. Baseline numbers must come from your own logged runs, the relevant library, or the original paper where comparable.

10. **No selective reporting.** Multiple seeds of the baseline and of the reproduction must be reported with mean ± std or full distribution. If you report one quality metric where you win, you must also report metrics where you do not (e.g., do not report CRPS where you win and silently omit RMSE where you lose).

11. **No hardcoded reference values.** Code must not contain hardcoded predictions, metric targets, sparsity values, or other ground-truth literals that allow shortcutting the actual gp2Scale evaluation.

12. **Experiment completeness.** Every experiment in the paper (data preparation, reproduction, existing-kernel-inadequacy runs, the full-scale run, baseline runs, prediction, metric computation) must have corresponding logs. Gaps will be flagged.

13. **Code-paper alignment.** The Methodology must accurately describe what the code does — the kernel construction, covariance-assembly distribution, MCMC block structure, prediction routine, stated scale, and worker count must match the implementation.

14. **Data-provenance integrity.** The new-domain dataset's source, retrieval, and preparation pipeline must be logged in enough detail that the dataset can be reconstructed. A dataset asserted to meet the announced scale ($\geq 5\times10^5$ points) whose preparation is undocumented, or whose size cannot be verified from the logs, will be flagged.

15. **No substantive replication of published follow-ups.** Convergence on shared technical choices is fine; substantive replication of an existing follow-up's domain + kernel + analysis presented as novel will be severely degraded.

16. **Effort evidence.** Any work you describe as failed or unresolved (kernel constructions you tried and abandoned, domains you attempted that did not reach scale, sparsity configurations that ran out of memory) must have a corresponding `proposal/attempts_log.md` entry documenting the attempts. Claims of effort without an attempts log will be read as not having tried.

## Deliverables — Here's what to hand in

Organize all outputs under `proposal/`.

**Codebase, logs, method plan, and attempts log:**
- `proposal/code/` — your implementation, including your kernel construction as a distinct, clearly-named module, with entry points for each reported run.
- `proposal/method_plan.md` — the pre-registration (domain, inadequacy hypothesis, kernel adaptation, test split, metric), committed before the full-scale run.
- `proposal/logs/` — all run logs, including:
  - **Data preparation**: source/retrieval of the new-domain dataset, the subset-selection and feature pipeline, the train/test split with seed, and the runtime-measured dataset size.
  - **Implementation-faithfulness reproduction**: your gp2Scale build reproducing one paper benchmark (MNIST or topography), $\geq 3$ seeds, with hyperparameter traces and the matched scores.
  - **Existing-kernel-inadequacy runs**: at least one of the paper's existing kernels run on your domain, demonstrating the inadequacy your adaptation fixes.
  - **The full-scale run**: the BMH-MCMC chain trace (per-block acceptance, hyperparameter trajectories, log marginal likelihood per accepted sample), the trained-hyperparameter snapshot, wall-clock, and peak memory — using your adapted kernel.
  - **Test-set prediction**: predictions (means and variances) on the held-out set, the quality metric, and any per-region breakdown.
  - **Baseline runs**: SVGP / VNNGP / SKI / Vecchia at the largest tractable scale, $\geq 3$ seeds, on the same test set.
- `proposal/attempts_log.md` — structured record of distinct approaches attempted (kernels, domains, sparsity configurations), with outcomes and log references. Required when any work is documented as failed or unresolved.

**Follow-up paper** at `proposal/report.tex`, using the provided `neurips.sty`, with sections: Introduction, Methodology, Experimental Setup, Experimental Results, Analysis, Conclusion, References. The Analysis or Conclusion must include an explicit limitations discussion.

The Experimental Results section must include a primary results table labeled `\label{tab:main_results}` with these column headers:

| Method | Kernel | $|\mathcal{D}_{\text{train}}|$ | $|\mathcal{D}_{\text{test}}|$ | RMSE | CRPS | Coverage | Wall-clock (h) | Peak memory (GB) |
|--------|--------|--------------------------------|-------------------------------|------|------|----------|----------------|-------------------|
| gp2Scale (your adapted kernel) | (your kernel) | $\geq 5\times10^5$ | | | | | | |
| gp2Scale (paper's existing kernel) | (Wendland / bump / …) | (your value) | (same) | | | | | |
| SVGP / VNNGP / SKI / Vecchia (largest tractable) | — | (your value) | (same) | | | | | |

The "gp2Scale (paper's existing kernel)" row is the existing-kernel-inadequacy comparison — it is what evidences contribution (A). Beyond the table, include: the kernel construction with its mathematical specification and the argument for why the paper's kernels are inadequate for your domain; a visualization of the learned kernel and the resulting covariance sparsity (percent non-zero); the BMH-MCMC chain-trace summary; test-set prediction visualizations; a probability-coverage (calibration) figure comparing your adapted kernel, the paper's existing kernel, and the approximate baseline; and the insight (B) tied to specific logged artifacts.

The **Analysis** section must include a mechanism subsection (why your kernel adaptation captures the domain's structure that the existing kernels miss), a failure subsection (regions of high uncertainty, kernel-support choices that produced infeasible sparsity, scaling regimes where the BMH-MCMC failed to converge), and the scientific insight (B). Incomplete or shallow contract work belongs in an appendix with corresponding `attempts_log.md` entries, not in the Analysis.

### Paper quality

Your submission is an academic paper, not a class-project report — top-journal standard in structure, integrity, depth, and language: a clear thesis up front, claims supported by quantitative evidence, prose rather than bullet enumeration in main-body sections, careful citation, and explicit honest discussion of limitations.

Do not frame the paper as "this is a follow-up of gp2Scale." Frame it as a self-contained contribution with its own motivation, methods, results, and discussion, citing the original paper where appropriate.

**Independence from this task spec.** A reader of your paper who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. The title should state your scientific contribution (your kernel/insight/domain finding), not the structure of the investigation. Section headings should be physics/domain/method-driven, not contract-driven. The introduction should motivate the work from the science, not announce "we apply gp2Scale to a new domain with a new kernel."

**Rendered output verification.** Before submitting, render the LaTeX to PDF and inspect it: equations must not overflow margins, tables must fit text width, figures must not be cut off. A submission with formatting issues in the rendered PDF will be flagged as not visually verified.

Good luck.
