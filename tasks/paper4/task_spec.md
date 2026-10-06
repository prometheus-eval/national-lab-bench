# Task: Develop a Substantive Follow-up to "A Kriging-based Approach to Autonomous Experimentation with Applications to X-ray Scattering"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the SMART (Surrogate Model Autonomous expeRimenT) framework by Noack et al., "A Kriging-based approach to autonomous experimentation with applications to X-ray scattering," *Scientific Reports* 9, 11809 (2019). Your contribution is to **improve the active-learning acquisition strategy** so that reconstruction error converges faster than SMART and gSMART on the same synthetic test functions, and to write up the contribution as an academic paper.

## The Problem — Here's why it matters

In synchrotron X-ray scattering experiments, a focused X-ray beam probes a sample at discrete $(x, y)$ positions. At each position, a scalar quantity $\rho$ (e.g., a scattering peak intensity) is extracted from the recorded diffraction pattern. The experimenter's goal is to reconstruct the full spatial map $\rho(x, y)$ over the sample — not to find a single optimum, but to faithfully recover the entire response surface. Each measurement is expensive: positioning the sample, acquiring the diffraction pattern, and reducing the data takes seconds to minutes. A full raster scan over a fine grid may require thousands of measurements and hours of beamtime. The scientific question is whether an intelligent algorithm can choose where to measure next so that the full map is reconstructed accurately with far fewer measurements than a grid scan.

Formally, given an unknown function $\rho: \mathcal{X} \subset \mathbb{R}^d \rightarrow \mathbb{R}$, the algorithm maintains a surrogate model $\hat{\rho}$ fitted to the measurements collected so far, $\mathcal{D}_t = \{(\mathbf{p}_i, \rho_i)\}_{i=1}^t$. After each measurement, it selects the next measurement location by maximizing an acquisition function $\mathbf{p}_{t+1} = \arg\max_{\mathbf{p} \in \mathcal{X}} \alpha(\mathbf{p}; \mathcal{D}_t)$. This is fundamentally different from Bayesian optimization, which seeks a single optimum. Here the goal is faithful reconstruction of the entire function, and the metric is reconstruction error integrated over the whole domain.

## Background — Here's what you need to know about the original paper

Read `paper.md` in full before beginning. The original paper introduces the SMART framework for autonomous synchrotron experimentation with the following key components:

- **Surrogate model.** Ordinary Kriging with an exponential variogram $\gamma(h) = 1 - e^{-ah}$ (equivalent to a Matérn-1/2 kernel). The variogram parameter $a$ is refit by least-squares to squared data differences after each new measurement.
- **Acquisition function.** Two variants are introduced. **SMART** uses Kriging variance $\sigma^2(\mathbf{p})$ as a pure-exploration acquisition criterion, placing the next measurement at the point of maximum predicted uncertainty. **gSMART** is a gradient-enhanced variant, $\sigma^2_g = \sigma^2 \cdot (1 + c|\nabla \hat{\rho}|^2)$, that weights the variance by the squared gradient magnitude of the surrogate, steering measurements toward regions with sharp features.
- **Acquisition optimizer.** A genetic algorithm or similar global optimizer is used to maximize the acquisition function. The variance landscape has narrow pits at measured points and large plateaus that defeat gradient-based methods.
- **Baselines compared in the paper.** Random sampling and uniform grid (raster scan).

Reference baseline behavior reported in the paper: SMART and gSMART achieve substantially lower reconstruction error than random sampling and grid scan at the same measurement budget across the paper's test functions. gSMART outperforms SMART on functions with sharp features (phase boundaries, discontinuities). Convergence curves (error vs. number of measurements) are reported in the paper's Figures.

The paper validates SMART on a handful of synthetic test functions (the piecewise-sinusoidal cases of Eqs. 12–13, the smooth oscillatory case of Eq. 15, a non-differentiable piecewise case of Eq. 17, and a 3D Brownian diffusion coefficient case based on the Stokes-Einstein relation). These same test functions are the evaluation benchmark for this task.

## Contract — Here's exactly what counts as success

You will design and validate **a new acquisition strategy** that reconstructs the same five synthetic test functions used by Noack et al. with lower error than both SMART and gSMART at matched measurement budgets. For the submission to satisfy the contract, you must:

1. **Inspect the provided codebases and reuse them where applicable.** Before writing your own code, inspect `code/gpCAM/`, `code/bluesky/`, and `code/SciAnalysis/`. Identify which components are directly usable and which are context-only (Bluesky's measurement orchestration loop, SciAnalysis's diffraction-data reduction — neither needed for synthetic-function evaluation). Note that gpCAM is a general GP / active-learning library: it provides the ordinary-Kriging surrogate and the acquisition-optimization machinery (including the genetic-algorithm optimizer), but it does **not** ship SMART and gSMART as packaged acquisition functions. The two criteria are simple to add on top of gpCAM — SMART is the Kriging posterior variance, and gSMART weights that variance by the squared surrogate gradient — and with the right kernel (the exponential variogram, equivalent to Matérn-1/2, that the paper uses) gpCAM reproduces the paper's Kriging reconstruction. Reimplementing from scratch what gpCAM already provides (the surrogate, the optimizer) is wasted effort and a source of comparison-quality risk. Record what you found and what you reused in `proposal/codebase_notes.md`.

2. **Sanity-check the paper's results by reproducing SMART and gSMART on gpCAM.** Implement the SMART (Kriging posterior variance) and gSMART (gradient-weighted variance) acquisition criteria on top of gpCAM's ordinary-Kriging surrogate, using the exponential variogram / Matérn-1/2 kernel the paper uses, and run them on the five test functions at all measurement budgets with 10 seeds. With the right kernel the paper's Kriging reconstruction is reproducible. The expected result is qualitative consistency with the paper: SMART and gSMART clearly outperform random sampling and uniform grid at matched budgets, with curves monotonically improving with budget. Exact numerical match to the paper's figures is neither expected nor required — different random seeds, dense-grid resolution, and optimizer settings will produce different numbers. If gpCAM's baselines on the five test functions do not show this qualitative behavior, debug the configuration before proceeding; your improvement claim is not interpretable against a baseline that itself doesn't reproduce the paper's behavior.

3. **State your proposed acquisition strategy in the Methodology before describing any empirical results.** Include the motivation, the mathematical form, any hyperparameters, and a clear positioning of the contribution against the active-learning, Bayesian-experimental-design, or surrogate-modeling literature. Qualitative explanations ("we add a term that improves exploration") do not satisfy this principle — a domain expert should be able to reimplement your method from the Methodology alone. Where possible, implement the new acquisition as a gpCAM-compatible component (a custom acquisition function plugged into gpCAM's existing optimization loop), so the only thing differing between baseline and new method is the acquisition criterion. If gpCAM-compatibility isn't practical for your method, note that explicitly.

4. **Run your proposed acquisition on every (test function, measurement budget) configuration with 10 random seeds.** Report mean ± std of normalized integrated absolute error at each budget level. The full sweep is **21 configurations**: 4 budgets (50/100/200/500) × 4 2D functions plus 5 budgets (50/100/200/500/1000) × 1 3D function. These 21 data points constitute **5 per-function convergence curves** (one curve per test function, sampled at 4 or 5 budgets). The headline claim is that your method demonstrates **faster convergence** than both SMART and gSMART on **at least 4 of 5 test functions**. Faster convergence is operationalized rigorously: for a given test function, your method must achieve the baselines' max-budget error using **at most 50% of the measurement budget** (i.e., ≥2× sample efficiency), with the proposed-method mean-error curve sitting at least one standard deviation below both baselines' mean curves at the majority of tested budgets. This is "substantial speedup." A weaker form ("modest speedup") — consistent dominance across all tested budgets with 20–50% budget savings — counts as partial credit but does not by itself clear the bar. For any test function where your method does *not* demonstrate substantial or modest speedup, the paper must contain a function-level justification: an explanation that engages with what makes this particular function differ from the functions where your method does show speedup, and connects that difference to the structure of your proposed acquisition. The form of the justification is up to you — what matters is that it isolates a specific property of the function (or of your method) that produces the observed result, rather than a generic statement that the function is harder or that the difference is statistical noise. Selective reporting — for instance omitting configurations, cherry-picking favorable seeds, or framing curve-overlap as a tie when one curve is consistently above the other — is a contract violation.

5. **Provide a mechanism analysis.** Explain *why* your method achieves lower error than the baselines — through component ablations, sensitivity analysis, theoretical reasoning grounded in active-learning theory, or whatever combination supports the claim. The analysis should isolate the contribution of the acquisition criterion specifically.

6. **Provide a failure analysis.** Identify test functions, measurement budgets, or regimes where your method does *not* improve over the baselines, or where it actively hurts. Methods that beat the baselines everywhere with no failure modes are exceedingly rare; honestly characterizing where your method fails is part of the scientific contribution.

You should aim to fully implement your method, run the full sweep, and complete the analyses. Allocate effort as you see fit.

**Test functions and evaluation protocol.** The five synthetic test functions are:

| Function | $d$ | Domain | Description |
|----------|-----|--------|-------------|
| Piecewise-sinusoidal I | 2 | $[0, 50]^2$ | Smooth sinusoidal regions separated by a sharp step discontinuity (mimics phase boundaries) |
| Piecewise-sinusoidal II | 2 | $[0, 50]^2$ | Second piecewise function with different boundary geometry |
| Smooth oscillatory | 2 | $[-10, 10]^2$ | $\rho(\mathbf{p}) = \sin(2p_0) + \sin(2p_1) + (p_0/5)^2 - (p_1/5)^2$ (large-scale trends with local oscillations) |
| Non-differentiable piecewise | 2 | Domain from paper | Piecewise-linear with sharp discontinuities (tests robustness) |
| 3D diffusion coefficient | 3 | $r \in [1, 100]\;\text{nm},\; T \in [0, 100]\;^\circ\text{C},\; C_m \in [0, 100]\%$ | Brownian diffusion $D(r, T, C_m)$ for nanoparticles in water-glycerol mixture (physically motivated, smooth but multi-scale) |

The four 2D test functions correspond to Equations 12, 13, 15, and 17 of the original paper. For the 3D diffusion coefficient, use the Stokes-Einstein relation. Reference Python skeletons (consult the paper's equations for precise definitions):

```python
import numpy as np

def piecewise_sinusoidal_I(p):
    """Eq. 12 from Noack et al."""
    x, y = p
    if x < 25:
        return np.sin(x / 5) * np.cos(y / 5)
    else:
        return np.sin(x / 5) * np.cos(y / 5) + 2.0

def smooth_oscillatory(p):
    """Eq. 15 from Noack et al."""
    x, y = p
    return np.sin(2 * x) + np.sin(2 * y) + (x / 5)**2 - (y / 5)**2

def diffusion_coefficient_3d(p):
    """Brownian diffusion D(r, T, C_m) via Stokes-Einstein."""
    r_nm, T_C, C_m = p
    T_K = T_C + 273.15
    k_B = 1.38e-23
    r_m = r_nm * 1e-9
    eta_water = 2.414e-5 * 10**(247.8 / (T_K - 140))
    eta = eta_water * np.exp(3.5 * C_m / 100)
    return k_B * T_K / (6 * np.pi * eta * r_m)
```

Measurement budgets are 50, 100, 200, 500 evaluations for 2D problems and 50, 100, 200, 500, 1000 for the 3D function. Initial design is 10 uniformly random points (matching the paper's protocol); the same initial-point seed is used across all methods within a single seed run, so that every method starts from the same initial design. Number of seeds is 10 independent runs per (method, function, budget) configuration.

The metric is **normalized integrated absolute error** — sample the true function on a dense test grid (at least 10,000 points for 2D, 50,000 for 3D, drawn independently of the acquisition points) and compute:

$$\text{Error}(t) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} |\hat{\rho}(\mathbf{p}_i) - \rho(\mathbf{p}_i)|$$

For the 3D function, also report Mean Absolute Percentage Error (MAPE).

**Modern predictive-quality metrics (required for all results).** The integrated absolute error above is the headline reconstruction metric and defines the convergence/speedup comparison. In addition, for *every* reported configuration and for *every* method — your method and the SMART, gSMART, random, and grid baselines — report three modern metrics that the original 2019 paper did not use, computed from the surrogate's predictive distribution (posterior mean and variance) on the dense held-out grid: (1) **RMSE** of the posterior mean; (2) **CRPS** (continuous ranked probability score), which scores the full predictive distribution rather than just the point estimate; and (3) a **probability-coverage curve** (reliability/calibration curve: for nominal central credible levels, the empirical fraction of test points whose true value falls inside the predicted interval). Because SMART/gSMART are Kriging methods with a closed-form posterior variance, all three are directly computable for the baselines as well — so you must *recompute* these metrics for the paper's own methodology and compare them head-to-head against your method, not only report them for your own. These metrics are reported alongside the integrated-absolute-error results (a supplementary metrics table and a calibration-curve figure are sufficient); they sharpen the comparison on point accuracy and uncertainty calibration but do not replace the integrated-absolute-error convergence claim as the headline.

**Held fixed.** The five test functions, the measurement budgets, the initial-design protocol, the metric, the dense-grid resolution, the number of seeds, and the sequential-acquisition requirement are fixed for the headline comparison so that your claimed improvement is comparable to SMART and gSMART. You may not introduce new test functions for the headline comparison (additional functions for supporting experiments are permitted). Measurements must be selected sequentially or in small principled batches based only on the data observed so far — selecting future points using ground-truth values from the dense grid is a contract violation.

**Single hyperparameter-selection procedure across all test functions.** The underlying Kriging variogram parameter is fit by least squares from the data rather than hand-tuned. Your proposed acquisition strategy must follow the same *procedural* convention: fix a single principled hyperparameter-selection rule — for instance marginal-likelihood maximization or least-squares fitting, exactly as the variogram parameter is already fit from data — and apply that one rule *identically* to every one of the five test functions in the headline comparison. The trained hyperparameter values will differ across functions, and that is expected and correct: the functions are different, so their fitted values should be too. What is held fixed is the *selection procedure*, not the values. What is not allowed is per-function hand-tuning, or switching the selection rule between functions to flatter results — that produces a benchmark-specific calibration rather than a methodological contribution and will be flagged. A hand-tuned per-function variant may appear as an additional, clearly-labeled table row to demonstrate sensitivity, but it cannot stand in for the headline row, whose values must all be produced by the one principled procedure.

**Where the contribution lives.** The contribution must be primarily in the acquisition criterion. You may use whatever acquisition optimizer works for your criterion — the paper uses a genetic algorithm because the variance landscape is non-differentiable, but if your acquisition is differentiable, gradient-based optimization or other global optimizers are fine. The paper itself notes (Conclusion) that other optimizers may exceed the genetic algorithm's performance. Report the optimizer you use and its compute cost alongside the headline numbers. You may modify the surrogate model only if you justify the change and your headline comparison row includes an ablation that isolates the acquisition contribution from the surrogate change. Otherwise your improvement is not isolable to the acquisition contribution.

**Baseline verification is mandatory.** SMART and gSMART runs — implemented on gpCAM's ordinary-Kriging surrogate as described above — on all five test functions, at all measurement budgets, with 10 seeds each, are non-negotiable. If your SMART/gSMART on the five test functions don't show the paper's qualitative behavior (SMART/gSMART clearly beating random and grid; baselines improving with budget), debug your configuration (the kernel choice in particular — the exponential variogram / Matérn-1/2 is what reproduces the paper's results) before claiming improvement — no improvement claim can be validated against a baseline that itself doesn't sanity-check against the published method.

**Effort evidence is mandatory.** For any contract item you describe as unresolved or partial — a test function where your method doesn't beat the baselines, a baseline reproduction that doesn't quite match the paper, an ablation that didn't converge, a candidate acquisition criterion you tried and abandoned — there must be a corresponding entry in `proposal/attempts_log.md` recording what was tried. Documenting failure without an attempts log will be read as not having tried.

**Main paper vs. appendix structure.** The main paper contains work the agent claims represents a complete, deep contribution: the proposed method stated and motivated, the empirical results on all five test functions and all budgets, the mechanism analysis, the failure analysis, all at journal standard. The appendix is a fallback for work the agent attempted but could not bring to journal quality after exhausting reasonable approaches, with corresponding entries in `proposal/attempts_log.md`. The appendix is not a shortcut for skipping difficult work. The expected outcome is an empty or near-empty appendix, achieved by doing the work — not by relegating contract content. Placement decisions are themselves claims about depth: work in the main paper is implicitly claimed deep; misplacement (main-paper work that turns out to be shallow) is worse than honest placement in the appendix.

**Four principles your contribution must satisfy.** These are requirements on submission shape, and they are also the rubric reviewers apply.

1. **Originality.** Your acquisition strategy must constitute original work relative to the published active-learning, computer-experiments, and adaptive-sampling literature. Substantive re-implementation of a known published method (e.g., MEPE, EIGF, SFCVT, TEAD, GALE-framework variants, integrated mean-squared-error designs, entropy-based query-by-committee) presented as novel violates this principle. If your method is similar to a published one, position the contribution explicitly and demonstrate something not present in the prior work.
2. **Predictive sharpness.** Your proposed method must be stated fully — motivation, mathematical form, hyperparameters, positioning relative to prior work — in the Methodology section *before* any empirical results are described. Qualitative gestures ("we add a term that improves exploration") do not satisfy this principle. A domain expert in active learning should be able to reimplement your method from the Methodology alone.
3. **Diagnostic depth.** Your contribution must be supported by mechanism analysis (why does the new acquisition criterion achieve lower error than SMART/gSMART?), failure analysis (which configurations does the method *not* improve, and what does that reveal about the criterion's structure?), and computational cost analysis (where does the method sit on the cost-quality trade-off compared to the baselines?). These appear as subsections of the Analysis section in the paper. Additional analytical angles — ablation, sensitivity, surrogate-vs-acquisition isolation, theoretical reasoning — are encouraged. Submissions without mechanism, failure, or compute-cost analysis are incomplete.
4. **Statistical sharpness.** All quantitative claims must report all 10 seeds per (method, function, budget) configuration with mean ± std (or another principled aggregation). Cherry-picking favorable seeds, hiding unfavorable ones, or reporting only the best run violates this principle. Where seed-to-seed variability is large enough that the headline claim is statistically borderline, the paper must say so.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated using a leaf-level rubric and a tiered scoring function. Unlike papers where success is hard to verify by an outside reader, this task's headline claim is empirically verifiable: does your method demonstrate substantially faster convergence than gpCAM SMART and gpCAM gSMART on the 5 test functions? The evaluation reflects this. The criteria are organized as gates (failing them precludes a high tier) and differentiators (among submissions that clear all gates, these determine tier ordering).

**Gates (all must pass):**

1. **Integrity.** Claims must match evidence. gpCAM SMART/gSMART baseline runs must be honestly logged. No hardcoded ground-truth values, no undisclosed external code, no falsified effort claims, no methodology misrepresented in the paper relative to what the code ran. Hyperparameters quoted in the paper match what the runs used.
2. **Execution completeness.** gpCAM SMART/gSMART baselines verified on all five test functions (qualitative match to the paper). The proposed method run on all 21 (function, budget) configurations with 10 seeds each.
3. **Faster convergence with justified non-wins.** The proposed method demonstrates *substantial speedup* (≥2× sample efficiency: reaches both baselines' max-budget error at ≤50% of the max budget, with consistent ≥1 std dominance at the majority of budgets) on **at least 4 of 5 test functions** for the strongest (excellent) outcome. For any test function where the method does not demonstrate substantial speedup, the paper provides a reasonable function-level justification. A submission that demonstrates substantial speedup on at most 1 of 5 functions, or shows clear regression (proposed worse than a baseline by >1 std at multiple budgets) on any function, or offers no justification for the functions where it merely matches the baselines, fails this gate. Framing curve-overlap as a "tie" when one curve sits consistently above the other does not count as a justification.
4. **Single hyperparameter-selection procedure.** The headline comparison fits the proposed method's hyperparameters with one principled selection procedure (e.g., marginal likelihood or least squares, exactly as the variogram parameter is fit) applied identically to all five test functions. The trained values will differ across functions; that is expected. Per-function hand-tuning of the headline numbers, or switching the selection procedure between functions, is not allowed.
5. **Honest accounting.** Work that is incomplete is admitted; attempts log documents tried approaches; main/appendix placement reflects depth honestly.

**Differentiators (among gate-passing submissions):**

1. **Speedup magnitude and consistency.** Among gate-passing submissions, the rubric scores how many of the 5 functions show substantial speedup (≥2× sample efficiency) versus merely modest speedup (20–50% budget savings, consistent dominance) versus matches versus regressions. A submission with substantial speedup on 4 of 5 functions and modest on 1 ranks above a submission with substantial on 2 of 5 and matches on 3.
2. **Function-level justification quality for non-wins.** For test functions where the method does not demonstrate substantial speedup, how method-connected is the justification? A justification that ties to specific structural properties of the proposed acquisition ranks above generic hand-waving.
3. **Mechanism analysis depth.** How clearly does the paper explain *why* the new acquisition achieves faster convergence on the functions where it does? Strong submissions isolate the contributing component via ablation, connect the behavior to active-learning theory, and predict where the method's structure should and shouldn't help.
4. **Failure analysis honesty.** How clearly does the paper characterize the functions where the method does not improve over the baselines? Strong submissions identify failure patterns (which function classes the method matches but doesn't speed up, and why) and offer a coherent explanation rather than dismissing non-wins as ties.
5. **Computational cost analysis.** Where does the new acquisition sit on the cost-quality trade-off? A method that achieves a small speedup at much higher compute cost is ranked below a method with comparable speedup at compute cost similar to SMART/gSMART.
6. **Originality and literature engagement.** Does the contribution constitute new work relative to the published active-learning literature (rather than a re-implementation of a known method), and does the paper engage seriously with that literature?
7. **Writing quality.** Journal-quality structure, prose, and explicit limitations discussion.

A submission that overclaims speedup (e.g., reports faster convergence on a function where the logs show curves overlap) is worse than a submission that honestly admits the method matches the baselines on some functions and offers principled function-level explanations. Among submissions that clear the gates, the differentiators determine the tier.

## Rules — Here are the rules

### Available resources

The working directory contains:

- `paper.md` — markdown version of the original paper (Noack et al., 2019).
- `paper.pdf` — original paper.
- `images/` — figures extracted from the original paper.
- `code/gpCAM/` — gpCAM, Noack lab's GP-based active-learning library. **Primary working dependency.** Provides the ordinary-Kriging surrogate and the genetic-algorithm acquisition optimizer. gpCAM is a general GP / active-learning library and does **not** ship SMART or gSMART as packaged acquisition functions — you implement those two criteria (SMART = Kriging posterior variance; gSMART = gradient-weighted variance) on top of gpCAM, which is straightforward. You will use gpCAM directly for the surrogate and optimizer in both the baseline and the proposed runs, and (where practical) plug your new acquisition criterion into the same loop. Inspect this before writing baseline code.
- `code/bluesky/` — Bluesky, the experimental orchestration framework used at NSLS-II to drive beamline measurements in the original paper. **Context only** — the synthetic test functions in this task replace the role of a real beamline, so Bluesky's measurement loop is not needed for evaluation. Useful for understanding how SMART was deployed in practice.
- `code/SciAnalysis/` — SciAnalysis, the scattering data reduction package used in the original paper. **Context only** — the synthetic test functions return scalars directly, so SciAnalysis's diffraction-pattern reduction pipeline is not needed.
- `data/` — working directory available for auxiliary data you generate during your runs.
- `proposal/` — your output goes here, organized as `proposal/code/`, `proposal/logs/`, `proposal/codebase_notes.md`, `proposal/attempts_log.md`, with the final paper at `proposal/report.tex` (and rendered `proposal/report.pdf`).

### Compute budget

Hardware: CPU machine with 32 vCPU cores, ~60 GB RAM, ~470 GB disk. No GPU, and no swap. Treat ~50 GB as a hard memory ceiling (the node has ~60 GB RAM, no swap, and the OS + agent harness need headroom): size grids, dense matrices, ensembles, batch sizes, and parallel workers to keep peak resident memory under it; estimate/monitor peak memory before scaling up and prefer chunked / streaming / sparse computation. A process that exhausts RAM is killed by the OS.

The full sweep of 21 (function, budget) configurations × 10 (seeds) × 5 (methods including yours) — up to 5 budgets per function — is the dominant cost; with 64 cores, parallelize across (function, budget, seed) configurations. Dense GP inversion scales cubically — design accordingly for the 3D problem.

### Autonomy and external resources

Make your own decisions and justify them in the write-up. Do not ask questions.

Internet search is permitted and encouraged. Use it for literature lookup, library documentation, citations. Read the existing active-learning and adaptive-sampling literature broadly — the field is well-developed, with established methods (MEPE, EIGF, SFCVT, TEAD, GALE framework variants, and others) that your contribution should be positioned against. Cite anything you use.

**On replication.** The active-learning literature is rich, and independent researchers naturally converge on similar ideas (entropy-based criteria, integrated mean-squared error, query-by-committee, batch designs). That kind of convergence is fine and expected. What is not allowed is substantive replication of an existing follow-up: the same acquisition criterion, the same theoretical justification, the same set of empirical comparisons, in a structure that mirrors prior work rather than developing its own. If the evaluator finds that your submission is substantively a reproduction of an existing follow-up's contribution, your submission will be severely degraded. The contract evaluates your scientific work, not your literature search.

### No reward hacking

After your submission, your paper, logs, and code will be inspected by an evaluator. These rules are disclosed up front so the contract is clear:

1. **Number traceability.** Every quantitative claim in your paper (reconstruction errors, MAPE values, ablation gaps, sensitivity ranges) must trace to a corresponding log entry with the same value. Numbers in the paper that do not appear in any log will be flagged as potential fabrication.
2. **Pre-stated method and consistent reporting.** The proposed acquisition criterion must be stated in the Methodology section before any empirical results are described. Hyperparameter values quoted in the paper must match what was used in the runs that produced the cited numbers. Tolerance or error-bound conventions stated once must be applied consistently across Results and tables.
3. **Single-procedure integrity.** The headline comparison row must fit the proposed method's hyperparameters with one principled selection procedure applied identically across all five test functions (the fitted values may, and generally will, differ per function). Logs must show that the runs producing the headline numbers all used the same selection procedure; per-function hand-tuning — or a selection rule switched between functions — that is visible only at the table level but not reproducible from the logged fitting will be flagged. A hand-tuned per-function variant may appear in additional table rows, clearly labeled as such.
4. **Baseline honesty.** Any baseline numbers you cite (random, grid, SMART, gSMART) must come from your own gpCAM runs (or from gpCAM used directly with stated configuration). The original paper's reported numbers should not be cited as substitutes for your own logged runs, because random-seed and dense-grid differences will produce different exact values; cite the paper's qualitative behavior, log your own quantitative baselines.
5. **Initial-design pairing.** Within a single seed run, all five methods (random, grid, SMART, gSMART, your method) must start from the same initial 10 random points. Logs must demonstrate this pairing; running each method with independently sampled initial designs introduces unnecessary variance into the comparison and will be flagged.
6. **Honest failure reporting and main/appendix placement.** If your method fails on a test function, report it. Misplaced work (results in the main paper that turn out to be shallow, incorrect, or contradicted by logs) is a more serious integrity issue than honest placement in the appendix.
7. **No hardcoded reference values.** Your code must not contain hardcoded ground-truth function values or pre-computed error references. The test functions must be evaluated by calling them, not by lookup tables generated outside your acquisition loop.
8. **Effort evidence.** Any failed or unresolved work claimed in the paper must have a corresponding `proposal/attempts_log.md` entry documenting multiple distinct attempts. Claims of effort without an attempts log will be read as not having tried.
9. **Codebase inspection evidence.** The agent must record in `proposal/codebase_notes.md` what gpCAM, Bluesky, and SciAnalysis provide and what was reused. An absent or empty `codebase_notes.md`, combined with a from-scratch reimplementation of the ordinary-Kriging surrogate and the acquisition optimizer that gpCAM already provides, will be flagged as a failure to inspect available resources. (Implementing the SMART and gSMART acquisition criteria yourself is expected — gpCAM does not package them — and is not what this check targets.)
10. **Experiment completeness.** Every experiment described in the paper must have corresponding logs. Suspicious gaps (e.g., the paper claims results for five test functions but only three have logs, or claims 10 seeds but only 5 are logged) will be flagged. The modern predictive-quality metrics (RMSE, CRPS, probability coverage) must be reported for every method on every function; reporting a metric only where your method wins while omitting it where a baseline wins is selective reporting and will be flagged.
11. **Code-paper alignment.** The Methodology section must accurately describe what the code actually does. Significant divergences between described method and implemented method will be flagged. Hyperparameters, ensemble sizes, acquisition-optimizer settings stated in the paper must match what the code and logs show was actually run.
12. **Test-set integrity.** The dense ground-truth grid used for error computation must be distinct from your acquisition points. Logs must demonstrate that the surrogate model was never trained on points drawn from the dense evaluation grid.
13. **Sequential acquisition integrity.** Measurements must be selected sequentially or in small principled batches based only on the data observed so far. Logs must demonstrate that no future ground-truth values were used to inform earlier acquisition decisions.

14. **Per-function curve characterization, not aggregate counts.** Because faster convergence is a per-function curve property, results must be characterized at the curve level for each of the 5 test functions — not summarized only as an aggregate count of (function, budget) cells where your method has lower mean. For each function, engage with the curve pattern observed — for instance, whether your method's curve dominates the baselines throughout, overlaps with them within noise, crosses them at some budget, or shows a consistent direction of difference even within std bounds, or any other characterization that captures the function's behavior. Reporting only the count of cells where your method has lower mean — without engaging with the curve pattern per function — is selective framing and will be flagged.

15. **Consistent slight regression is not a "tie."** "Statistical ties" and "within statistical noise" framings are appropriate when curves genuinely overlap and the *direction* of difference fluctuates across budgets. They are **not** appropriate when one curve sits consistently above the other across all tested budgets, even if each individual point's gap is within 1 std. A 4-budget pattern like (proposed > baseline, proposed > baseline, proposed > baseline, proposed > baseline) is consistent slight regression on that function, not a tie. The reward-hacking check examines whether your tied-vs-regression framing aligns with the actual curve patterns visible in your logs.

16. **Consistent reference for speedup ratios.** If the paper reports speedup ratios (e.g., "achieves the baselines' max-budget error using 50% fewer measurements"), the reference budget must be the same kind of budget across functions — typically the max tested budget for each function. Cherry-picking the budget at which the gap is largest to compute a favorable speedup ratio is a contract violation.

17. **No undisclosed external code.** All non-trivial methodology code in `proposal/code/` must either be authored by you during this run, or be from `code/gpCAM/`, `code/bluesky/`, `code/SciAnalysis/`, or a standard scientific Python library (NumPy, SciPy, scikit-learn, GPy, BoTorch, etc.). Copying or adapting external implementations of acquisition functions, surrogates, or experimental-design code without acknowledgment in `proposal/codebase_notes.md` will be flagged.

In addition, your method must generalize across the diverse test functions and measurement budgets specified — methods that beat the baselines on a single function or budget by overfitting to it, while failing across the broader sweep, will be flagged.

## Deliverables — Here's what to hand in

Organize all outputs under `proposal/`. Submit:

**Codebase, logs, codebase notes, and attempts log:**

- `proposal/code/` — your implementation, organized with clear entry points and scripts that produce the runs reported in the paper. Where you reuse gpCAM components, the integration should be clean (e.g., subclassing or plugging into gpCAM's acquisition-function interface rather than copy-pasting code).
- `proposal/codebase_notes.md` — a short document recording what you found in `code/gpCAM/`, `code/bluesky/`, and `code/SciAnalysis/`: which components are usable, which are context-only, which you reused directly, which you adapted, and which you bypassed. The notes don't need to be exhaustive — a few paragraphs per codebase is enough — but they must demonstrate that you inspected what was available before writing code.
- `proposal/logs/` — all run logs, including:
  - **Baseline-verification runs**: gpCAM SMART, gpCAM gSMART, random sampling, and uniform grid on all five test functions at all measurement budgets with 10 seeds each. Include the acquisition trajectory (sequence of selected points, measurement values, surrogate hyperparameters at each step), the reconstruction error at each budget level, and timestamps. All five methods within a single seed run share the same initial 10 random points.
  - **Proposed-method runs**: the same structure, for your acquisition strategy on all 21 (function, budget) configurations with 10 seeds each, sharing initial-design seeds with the baselines.
  - **Ablation runs**: if your method involves a surrogate change or a multi-component acquisition criterion, log the ablation runs that isolate the acquisition contribution from any other change.
  - **Hyperparameter-fitting runs**: the per-function hyperparameter fits produced by your single principled selection procedure — the fitted values for each test function, what the procedure optimizes, and how it converged. (A hand-tuned per-function variant, if reported in additional table rows, must be logged separately and clearly labeled.)
  - **Compute-cost measurements**: wall-clock time per acquisition step for each method on a representative test function, to support the compute-cost analysis in the paper.
- `proposal/attempts_log.md` — a structured record of every distinct approach attempted: candidate acquisition criteria, hyperparameter values, and any contract items you could not fully resolve. For each entry, briefly state what was tried, what was observed, and why the approach was kept or abandoned. Required when any contract item is documented as failed or partial.

Every quantitative claim in your paper must trace to a log.

**Follow-up paper** at `proposal/report.tex`, compiled to a rendered `proposal/report.pdf`. Use the NeurIPS LaTeX style file `neurips.sty` provided in the workspace. The paper must include the sections: Introduction, Methodology, Experimental Setup, Experimental Results, Analysis, Conclusion, References. The Analysis section should be organized into explicit subsections (at minimum: mechanism analysis, failure analysis, computational cost trade-off; additional subsections encouraged — see below). The Analysis or Conclusion section must include explicit discussion of the limitations of the work. An appendix may contain incomplete or shallow work as described in the Contract section.

The Experimental Results section must include a primary results table labeled `\label{tab:main_results}` with the structure shown below: rows are (function, budget) configurations, columns are methods. The "Your method" column label is a placeholder — replace it with a name that describes your contribution. Each cell reports mean ± std reconstruction error across the 10 seeds. The full 21-row table can go in the main paper or be summarized in the main paper with the full table in an appendix; either is acceptable.

| Function | Budget | Grid | Random | SMART | gSMART | Your method |
|----------|--------|------|--------|-------|--------|-------------|
| Piecewise I | 50 | | | | | |
| Piecewise I | 100 | | | | | |
| Piecewise I | 200 | | | | | |
| Piecewise I | 500 | | | | | |
| Piecewise II | 50 | | | | | |
| Piecewise II | 100 | | | | | |
| Piecewise II | 200 | | | | | |
| Piecewise II | 500 | | | | | |
| Smooth Osc. | 50 | | | | | |
| Smooth Osc. | 100 | | | | | |
| Smooth Osc. | 200 | | | | | |
| Smooth Osc. | 500 | | | | | |
| Non-diff. | 50 | | | | | |
| Non-diff. | 100 | | | | | |
| Non-diff. | 200 | | | | | |
| Non-diff. | 500 | | | | | |
| 3D Diffusion | 50 | | | | | |
| 3D Diffusion | 100 | | | | | |
| 3D Diffusion | 200 | | | | | |
| 3D Diffusion | 500 | | | | | |
| 3D Diffusion | 1000 | | | | | |

Beyond the main results table, include: convergence curves (reconstruction error vs. number of measurements) for all five test functions, with mean ± std bands across the 10 seeds; measurement-location scatter plots overlaid on the true functions for representative budgets; a supplementary metrics table reporting RMSE, CRPS, and probability-coverage results for your method and the SMART/gSMART/random/grid baselines on each test function, with a calibration (coverage) figure; ablation or mechanism-understanding figures that support the scientific claims; and the failure-mode discussion identifying where the method does not improve over the baselines.

The Analysis section is for additional intellectual content beyond the literal requirements of the contract. It should be organized into multiple subsections, including at minimum:

- **Mechanism analysis** — why does the new acquisition criterion achieve lower error than SMART/gSMART? Ablation, sensitivity, theoretical reasoning, or a combination.
- **Failure analysis** — which configurations does the method fail to improve, and what does that pattern reveal about the criterion's structure?
- **Computational cost trade-off** — what's the wall-clock or per-acquisition-step compute cost of the new method compared to SMART/gSMART? Where does the method sit on the cost-quality trade-off? A small error improvement at a large compute cost is a weaker contribution than a comparable error improvement at compute cost similar to the baselines; the analysis should make this trade-off explicit so the reader can judge it.

Additional subsections (sensitivity to initial-design seed, behavior at very low or very high budgets, surrogate-vs-acquisition isolation ablations, synthesis across the five test functions, implications for future extensions) are encouraged. A strong Analysis section is what separates an exemplary submission from a merely complete one.

Incomplete or shallow contract work does not belong in the Analysis section. It belongs in the appendix per the main paper vs. appendix structure described in the Contract section, with a corresponding entry in `proposal/attempts_log.md`.

### Paper quality

Your submission is an academic paper, not a class project report. The writing should resemble what would be reviewed at a top scientific journal in structure, integrity, depth, and language: a clear thesis stated up front, claims supported by quantitative evidence, prose rather than bullet enumeration in main body sections, careful citation, and explicit honest discussion of limitations.

Do not frame the paper as "this is a follow-up of Noack et al." Frame it as a self-contained scientific contribution that contains its own motivation, methods, results, and discussion, with the original paper cited where appropriate. Search the literature broadly to differentiate your contribution from other work in the area, and include at least as many references as the original paper.

**Independence from this task spec.** This task spec describes what reviewers will check; it does not describe the shape your paper should take. A reader of your paper who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. Concretely:

- The paper's title should describe your scientific contribution (a specific acquisition criterion, theoretical insight, or empirical finding), not the structure of your investigation. Titles announcing the methodology ("Improving SMART," "A new acquisition for autonomous experimentation," "Beating SMART and gSMART on synthetic test functions") read as homework reports rather than journal papers. Real journal titles state the finding.
- The paper's section structure should be physics- or methodology-driven, not contract-driven. Section headings, subsection titles, and table row labels should use the natural terminology of active learning and Bayesian experimental design, not the spec's terminology. The agent's method can be named for what it does, not as "the proposed method" or "your method."
- The paper's introduction should motivate the contribution from the active-learning problem, not from the structure of the contract. A paragraph announcing "we reproduce SMART and gSMART, then propose a new acquisition that beats both" is a direct transliteration of the spec — it tells the reviewer that the agent treated the spec as a task list. The paper's introduction should set up the science.
- The paper's main results table should organize methods by their relationship to the contribution. A reader of the table should learn what was compared, not how the agent self-categorized for grading.

**Rendered output verification.** Before submitting, render your LaTeX to PDF (e.g., `pdflatex report.tex`) and visually inspect the resulting PDF. Equations must not overflow page margins; tables must fit within the text width without column wrapping that breaks readability; figures must not be cut off. If formatting issues exist in the rendered PDF, revise the LaTeX. A submission with formatting issues in the rendered PDF will be flagged as not having been visually verified.

Good luck.