# Task: Develop a Substantive Follow-up to "Projected Hessian Learning: Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the Projected Hessian Learning (PHL) framework by Rodriguez, Smith, Matin, Lubbers, Barros, and Mendoza-Cortes, "Projected Hessian Learning: Fast curvature supervision for accurate machine-learning interatomic potentials," 2025. Your contribution is to **reduce the number of probe vectors needed per minibatch step substantially below the original PHL baseline**, while preserving the variance and convergence behavior of the stochastic gradient estimator and the downstream model accuracy of the original PHL, written up as an academic paper.

## The Problem — Here's why it matters

Machine-learning interatomic potentials (MLIPs) approximate the potential energy $E(\mathbf{R})$ of molecular configurations and are typically trained on energies and atomic forces $\mathbf{F} = -\nabla E$ from quantum-chemistry reference calculations. Adding gradient (force) supervision dramatically improves accuracy and generalization over energy-only training. The natural next step is to add **second-derivative (Hessian) supervision**, since the Hessian

$$H_{(i,\alpha),(j,\beta)} = \frac{\partial^2 E}{\partial r_i^\alpha \partial r_j^\beta}$$

determines vibrational frequencies, transition-state curvature, and reaction-pathway information that energies and forces alone cannot reliably recover. Recent benchmarks show that even high-performing universal MLIPs miss phonon observables despite strong energy/force accuracy.

The obstacles are computational: a full Hessian has $(3N)^2$ entries per configuration, so storing or training against it scales quadratically in the system size $N$, and computing it from quantum chemistry generally requires response equations beyond those used for forces. **PHL** addresses this by replacing the explicit Hessian loss

$$\mathcal{L}_H = \frac{1}{(3N)^2} \sum_{i,j=1}^{3N} (\tilde H_{ij} - H_{ij})^2$$

with the unbiased Hutchinson trace estimator

$$\hat{\mathcal{L}}_H = \frac{1}{(3N)^2} |\tilde H v - H v|^2 = v^T A v, \qquad A = B^T B,\; B = (\tilde H - H)/(3N),$$

where $v$ is a random probe vector with components satisfying $\langle v_i v_j \rangle = \delta_{ij}$ (Hutchinson uses $v_i \in \{\pm 1\}$ i.i.d.). Because the Hessian–vector product $Hv$ can be computed in cost comparable to a small constant number of gradient evaluations via forward-over-reverse autodiff (and entirely avoids forming $\tilde H$), $\hat{\mathcal{L}}_H$ supervises curvature at near-force-level cost.

PHL closes most of the gap between full-Hessian training (which is computationally prohibitive) and force-only training (which misses curvature information). PHL's one-probe-per-molecule-per-minibatch protocol is the standard, but the original paper does not investigate whether this is the right operating point on the variance-vs-cost trade-off.

The headline metric for this follow-up is **probe count per minibatch step**, not per-epoch wall-clock. The distinction matters: at PHL's testbed scale (median $N \approx 14$, per-epoch times of ~4 s for E-F, ~13 s for PHL, and ~310 s for E-F-H as reported in the original paper), wall-clock is bounded by the E-F floor of ~4 s — even an infinite probe-count reduction would yield at most a ~3.25× wall-clock improvement at this scale, because non-HVP overhead (E-F gradient computation, parameter updates, data movement) does not shrink with probe count. Probe count is the better metric for three reasons.

*First*, probe count is a property of the **estimator**, not the implementation pipeline. It tests statistical efficiency directly and is reproducible across hardware, autodiff implementations, batch sizes, and code-optimization choices.

*Second*, probe-count reduction at variance and accuracy parity tests whether the PHL estimator has untapped statistical efficiency — if probes can be cut while preserving variance, the original estimator carries redundancy that variance-reduction techniques can exploit. This is a real scientific finding about the estimator's structure, not just about training speed.

*Third*, probe-count reduction tracks wall-clock more directly at scales beyond PHL's testbed. For systems at $N \approx 50$–$200$ (proteins, transition-metal complexes, materials), HVP cost dominates the per-step compute because per-molecule HVP scales with $N$ while non-HVP overhead is roughly fixed per minibatch. At those scales, a 5× probe-count reduction approaches a 5× wall-clock reduction. The contribution matters more at scales PHL did not test.

## Background — Here's what you need to know about the original paper

The original paper introduces and validates PHL on a chemically diverse reactive-chemistry dataset. Your follow-up will build on or compete against this established baseline. The technical components of the original method are:

1. **Loss formulation** [Eqs. (15)–(16) of the paper]:
   $$\mathcal{L}_{\text{HVP}} = \mathcal{L}_E + \lambda_F\,\mathcal{L}_F + \lambda_H\,\hat{\mathcal{L}}_H,$$
   with $\lambda_F = 0.30$ and $\lambda_H = 0.09$ across all comparisons.

2. **Probe-vector distributions.** Two unbiased families are compared:
   - **One-column ($v^{1\text{Hot}}$):** $v_i = \sqrt{3N}\,\delta_{i,c}$ for a uniformly random column index $c$. This samples a single Hessian column per HVP.
   - **Hutchinson (PHL, $v^{\text{Hutch}}$):** $v_i = \pm 1$ i.i.d. (or zero-mean unit-variance Gaussian); samples a random weighted combination of all Hessian columns.

3. **Probe-vector regimes.** Two protocols are compared:
   - **Randomized-vector:** a fresh random $v$ is drawn for each molecule in each minibatch.
   - **Fixed-vector:** a single random $v$ is drawn per molecule and reused across all epochs (a controlled surrogate for the realistic case where only a single HVP per system is available from quantum chemistry).

4. **Training protocols compared.**
   - **E–F:** energies + forces only (baseline).
   - **E–F–HVP (one-column):** energies + forces + one HVP per molecule per minibatch with $v^{1\text{Hot}}$.
   - **E–F–HVP (PHL):** energies + forces + one HVP per molecule per minibatch with $v^{\text{Hutch}}$.
   - **E–F–H:** energies + forces + full Hessian (upper-bound accuracy, worst cost).

5. **Dataset.** $\omega$B97XD/6-31G(d) reactive chemistry covering reactants, products, transition states (RTP, 35,087 equilibrium geometries from 11,961 reactions), intrinsic reaction coordinates (IRC, 34,248 geometries from 600 trajectories), and normal-mode sampled geometries (NMS, 62,527 far-from-equilibrium configurations from 574 reactions). Median atom count $N \approx 14$. Files in `data/openreact_chon_efh/`.

6. **Architecture.** HIP-NN-style MLIP, batch size and training schedule documented in the authors' notebook (`code/PHL/PHL_training.ipynb`), trained for ~700–1000 epochs.

**Reference baseline behavior reported in the paper:**

- **Randomized-vector regime:** Both one-column and PHL (Hutchinson) achieve **statistically indistinguishable** energy / force / Hessian RMSEs across Test, IRC, NMS datasets, all approaching E–F–H accuracy. PHL achieves $24\times$ speed-up per epoch over E–F–H (13 s vs ~310 s, vs ~4 s for E–F).
- **Fixed-vector regime:** PHL outperforms one-column with statistically significant margins on the NMS dataset (e.g., $-28.5\%$ energy RMSE for PHL vs $-23.8\%$ for one-column, both relative to E–F; $11–15\%$ lower Hessian RMSE for PHL vs one-column across datasets).
- **Probe-vector count per minibatch:** The original paper uses **one probe vector per molecule per minibatch** in both HVP regimes. At minibatch size $B$, the per-step probe count is therefore $B$. The paper does not investigate reducing this count further while preserving estimator quality.

## Contract — Here's exactly what counts as success

Design a method that **substantially reduces the probe-vector count per minibatch step** while preserving estimator unbiasedness, gradient-variance parity, and downstream model accuracy. Write the contribution up as an academic paper using the NeurIPS LaTeX style file `neurips.sty` (provided in the workspace).

### The Contract

**Success metric (headline claim):** Your method must achieve **substantial probe-count reduction** on the **vast majority** of the evaluation datasets, defined rigorously as:

- **Substantial reduction:** Full success requires a substantial time-averaged probe-count reduction while preserving unbiasedness, gradient-variance parity, and downstream accuracy parity. Treat >=5x fewer probe-vector evaluations per minibatch step than the PHL baseline (which uses $B$ probes per minibatch at minibatch size $B$, where $B$ is the training-time minibatch size) as the full-credit target; treat >=3x and >=2x as partial-credit reductions whose value depends on accuracy, variance, mechanism analysis, and whether the paper establishes a principled limit on larger reductions. State clearly that accuracy parity is an all-cell gate across Test/IRC/NMS, not an aggregate average. The probe count is measured as a **time-averaged** count over training — if your method uses an irregular probe schedule (e.g., HVP every $K$-th minibatch with $K > 1$, or $P < B$ probes drawn per minibatch with $P$ varying by training stage), report both the per-step count and the time-averaged count; the time-averaged count is the contract metric.

- **Vast majority of datasets:** at least **2 of 3** evaluation datasets (Test, IRC, NMS) must show substantial reduction at matched accuracy and matched variance.

- **Matched accuracy:** downstream RMSEs on each (property × dataset) cell satisfy either of two accepted assessment routes against your **reimplemented PHL baseline**, using at least 5 matched random seeds: (a) a paired difference $t$-test with $p > 0.05$; or (b) a pre-stated paired equivalence or non-inferiority test that passes its stated margin and decision rule. Report which route is used and apply the stated protocol consistently, without silently switching tests after seeing which gives a favorable result. Route (a) is accepted for this benchmark but does not, by itself, establish statistical equivalence or non-inferiority. The match must hold on **all three datasets** (Test, IRC, NMS) — failure on any one dataset breaks the "matched accuracy" condition. The reimplemented PHL baseline is your own 5-seed run of the unmodified PHL method at the same compute budget your method uses (see "Two-baseline reporting" below); paired comparison is meaningful only when both methods see the same training schedule, data, and seeds.

- **Matched variance:** the gradient-estimator variance, measured at matched compute budget on the same checkpoints, minibatches, and seeds, must be no worse than **2x** that of reimplemented PHL on each of Test, IRC, and NMS. Report the estimator, normalization, number of minibatches/checkpoints, confidence interval or uncertainty estimate, and all three dataset-specific ratios in the main results or an adjacent table. Pre-state how near-boundary values such as 2.04 +/- 0.1 are classified.

A weaker form — modest probe-count reduction (≥2× and < 5×) at matched accuracy and variance on the majority of datasets — counts as partial credit but does not by itself clear the bar for substantial contribution. For any (dataset, property) combination where your method shows accuracy degradation, variance excess, or no probe-count reduction, the paper must contain a per-dataset justification: an explanation that engages with what about that dataset's distribution or property combination makes your probe-allocation strategy ineffective there, and connects the difference to the structure of your method. Selective reporting — for instance reporting only the dataset where reductions are largest, or framing per-dataset regressions as aggregate ties — is a contract violation.

**Two-baseline reporting.** Every quantitative comparison in your paper must report two PHL baselines side-by-side:

1. **Paper-as-reported PHL** — the original paper's published RMSEs and wall-clock figures cited from a specific table or figure in Rodriguez et al. 2025. These are *fixed reference values*, not your own runs.

2. **Reimplemented PHL** — your own logged 5-seed run of the unmodified PHL method using `code/PHL/`, at whatever training schedule (epochs, data fraction, batch size) your compute budget permits.

Your method's results must be compared at *matched compute* to the reimplemented PHL: same epoch budget, same data fraction, same batch size, same seeds. Matched-accuracy and matched-variance tests are done against the reimplemented baseline. The paper-as-reported baseline serves as an external anchor: if your reimplemented PHL substantially under-converges relative to paper-as-reported (e.g., 2–3× higher RMSEs), the gap must be documented in the Experimental Setup section, *and the Analysis section must address whether your method's gains over reimplemented PHL would transfer to the paper's full-convergence regime.* The two-baseline pattern is auditable: an evaluator can check whether your reimplementation tracks the paper, and whether your method is gaining at a regime that the paper's results characterize. Hidden under-convergence is a contract violation; transparent under-convergence with substantive transfer analysis is acceptable.

**Held fixed (parity constraints):**

- **Estimator unbiasedness:** $\mathbb{E}[\hat{\mathcal{L}}_H] = \mathcal{L}_H$ exactly under your probe-distribution choice. Proof must appear in the Methodology section before any empirical results.
- **Architecture, dataset, hyperparameters:** identical to the paper's setup — HIP-NN-style MLIP, the dataset in `data/openreact_chon_efh/`, the same data splits, $\lambda_F = 0.30$, $\lambda_H = 0.09$, the same Adam optimizer + scheduler from the authors' notebook. These are not part of the design space.
- **Training schedule (epochs, data fraction, batch size):** the agent may run at a reduced training schedule if compute-constrained, **subject to the two-baseline reporting requirement above**. The schedule used by your method must be *exactly* the schedule used by your reimplemented PHL — same epoch budget, same data fraction, same batch size — so the paired comparison is well-defined. The contract evaluates parity between your method and your reimplemented baseline; the paper-as-reported reference anchors the regime your reimplementation is meant to track. Running both methods at a reduced schedule with the gap to paper-as-reported visible and discussed is permitted; running at a reduced schedule and presenting parity-with-reimplementation as if it were parity-with-published-PHL is a contract violation.
- **Random seed protocol:** at least 5 random seeds for both your method and reimplemented PHL. The same seed set must be used for both methods so paired tests are valid.
- **Anti-cheating:** you cannot reduce the probe count by switching off the Hessian loss term ($\lambda_H = 0$ is just E-F training, which the paper already evaluates). You cannot reduce by silently moving Hessian supervision into offline pre-training, cached HVP labels, baseline-run side information, test-set selection, or a quantum-chemistry pre-computation. Any skipped-Hessian or skipped-molecule strategy must include an explicit compensation term and an unbiasedness proof. The probe count is measured during the SGD updates that minimize $\mathcal{L}_{\text{HVP}}$, including every HVP used to choose or execute those updates.

**Where the contribution lives.** The contribution must be primarily in the probe-allocation/probe-estimation strategy: how often probes are drawn, how many are drawn, which molecules receive them, what probe law is used, and how any omissions are reweighted. If the method changes the probe distribution (for example Gaussian to Rademacher), require an ablation against a same-probe-count baseline using that probe law so reviewers can separate zero-cost probe-law improvements from true probe-count savings. You may not modify the underlying loss formulation, model architecture, data splits, or held-fixed hyperparameters.

**Provide a mechanism analysis.** Explain *why* your allocation rule achieves the probe-count reduction it does while preserving variance and accuracy — through theoretical analysis of the variance structure, ablations of the allocation components, or whatever combination supports the claim.

**Provide a failure analysis.** Identify datasets, training stages, or properties where your method fails to meet the parity constraints. Methods that hit every constraint everywhere are exceedingly rare; honestly characterizing where your method fails is part of the scientific contribution. If your reimplemented PHL under-converges relative to paper-as-reported, this analysis must also address whether your method's gains transfer to the paper's full-convergence regime — for example, whether the variance-reduction mechanism is convergence-stage-dependent.

**Provide a computational cost analysis.** Where does your method sit on the cost–quality trade-off? A small probe reduction at substantial classical overhead (e.g., expensive allocation-decision logic per minibatch) is a weaker contribution than a comparable reduction at PHL-level cost.

**Effort evidence is mandatory.** For any contract item you describe as unresolved or partial — a dataset where your method fails the accuracy match, an allocation strategy that didn't converge, a variance-reduction approach you tried and abandoned — there must be a corresponding entry in `proposal/attempts_log.md` recording what was tried, what was observed, and why the approach was kept or abandoned. Documenting failure without an attempts log will be read as not having tried.

**Main paper vs. appendix structure.** The main paper contains work the agent claims represents a complete, deep contribution: the proposed allocation rule stated and motivated, the unbiasedness proof, the variance analysis, the empirical results across the three datasets, the mechanism / failure / compute-cost analyses, all at journal standard. The appendix is a fallback for work the agent attempted but could not bring to journal quality, with corresponding `attempts_log.md` entries. The appendix is not a shortcut for skipping difficult work. The expected outcome is an empty or near-empty appendix, achieved by doing the work — not by relegating contract content. Placement decisions are themselves claims about depth: misplacement (main-paper work that turns out to be shallow) is worse than honest placement in the appendix.

**Four principles your contribution must satisfy.** These are requirements on submission shape, and they are also the rubric reviewers apply.

1. **Originality.** Your probe-allocation strategy must constitute original work relative to the published variance-reduction, stochastic-optimization, and Hessian-supervision literature. Substantive re-implementation of a known method (e.g., importance sampling, control variates as published, adaptive probe schedules from existing variance-reduction papers) presented as novel violates this principle. If your method is similar to a published one, position the contribution explicitly and demonstrate something not present in the prior work.

2. **Predictive sharpness.** Your method — probe distribution, allocation rule, hyperparameters, unbiasedness proof, variance analysis — must be stated fully in the Methodology section *before* any empirical results are described. Qualitative gestures ("we sample fewer probes adaptively") do not satisfy this principle. A domain expert in stochastic gradient methods should be able to reimplement your method from the Methodology alone.

3. **Diagnostic depth.** Your contribution must be supported by mechanism analysis (why does the allocation rule preserve variance?), failure analysis (which datasets / training stages / properties does the rule struggle with, and what does that reveal?), and computational cost analysis (where does the method sit on the cost-reduction trade-off vs. PHL?). These appear as subsections of the Analysis section. Submissions without mechanism, failure, or compute-cost analysis are incomplete.

4. **Statistical sharpness.** All quantitative claims must report all 5+ seeds per (method, dataset) configuration with mean ± std, with the chosen paired accuracy-parity test on each (property × dataset) cell, using either accepted route in the Matched accuracy definition. Cherry-picking favorable seeds, hiding unfavorable ones, or reporting only the best run violates this principle. Where seed-to-seed variability is large enough that the headline claim is statistically borderline, the paper must say so.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated using a leaf-level rubric and a tiered scoring function organized as **gates** (failing them precludes a high tier) and **differentiators** (among submissions that clear all gates, these determine tier ordering).

**Gates (all must pass):**

1. **Integrity.** Claims must match evidence. Reimplemented PHL baseline runs honestly logged. Paper-as-reported numbers traceable to specific tables/figures in the original paper. No hardcoded ground-truth values, no undisclosed external code, no falsified effort claims, no methodology misrepresented in the paper relative to what the code ran. Hyperparameters, probe schedules, and allocation rules quoted in the paper match what the runs used.

2. **Execution completeness.** Reimplemented PHL baseline reproduced across 5+ seeds. Your method run across the same 5+ seeds. Gradient-variance audit performed at periodic checkpoints. Downstream RMSE evaluation on Test, IRC, and NMS for energy, force, and Hessian. Two-baseline reporting present in the main results table.

3. **Estimator unbiasedness and variance parity.** Provable unbiasedness for the stated probe distribution. Gradient-estimator variance no more than $2\times$ that of your reimplemented PHL on each of (Test, IRC, NMS), as measured by the audit.

4. **Probe-count reduction with justified non-wins.** Full success requires >=5x lower time-averaged probe count than the reimplemented PHL $B$-probe baseline on the required dataset coverage, while passing the global accuracy and variance gates. Methods with 2x-5x reduction, near-boundary variance, or practically negligible but statistically resolved derivative-cell changes should be scored as partial credit according to their mechanism, honesty, and per-dataset analysis rather than automatically treated as equivalent to failed implementations. Framing per-dataset regressions as aggregate ties does not count as justification.

5. **Honest accounting.** Work that is incomplete is admitted; the attempts log documents tried approaches; main/appendix placement reflects depth honestly. If reimplemented PHL under-converges relative to paper-as-reported, this is documented and the transfer question is addressed.

**Differentiators (among gate-passing submissions):**

1. **Probe-count reduction magnitude and consistency.** The rubric scores how many of the 3 datasets show substantial reduction (≥5×) versus modest reduction (2–5×) versus matches versus regressions. A submission with substantial reduction on 3 of 3 datasets ranks above one with substantial on 2 of 3 and modest on 1.

2. **Per-dataset justification quality for non-wins.** For datasets where your method does not demonstrate substantial reduction at parity, how method-connected is the justification? A justification that ties to specific structural properties of the dataset (e.g., NMS's distribution of vibrational modes, IRC's reaction-path curvature) and your method ranks above generic hand-waving.

3. **Mechanism analysis depth.** How clearly does the paper explain *why* the allocation rule preserves variance and accuracy? Strong submissions isolate the contributing component via ablation, connect the behavior to variance-reduction or stochastic-optimization theory, and predict where the rule should and shouldn't help.

4. **Failure analysis honesty.** How clearly does the paper characterize the regimes where the method does not improve over PHL? Strong submissions identify failure patterns (e.g., specific dataset structural properties, specific training stages) and offer a coherent explanation rather than dismissing non-wins. If reimplemented PHL under-converges relative to paper-as-reported, strong submissions explicitly address whether the variance-reduction mechanism is convergence-stage-dependent.

5. **Computational cost analysis.** Where does the method sit on the cost-quality trade-off? A method that achieves a 5× probe reduction at much higher classical overhead is ranked below one with comparable reduction at PHL-level cost.

6. **Originality and literature engagement.** Does the contribution constitute new work relative to the published variance-reduction, stochastic-optimization, and Hessian-supervision literature, and does the paper engage seriously with that literature? Including additional baselines from the literature in your comparison table is encouraged where relevant.

7. **Writing quality.** Journal-quality structure, prose, and explicit limitations discussion.

A submission that overclaims (e.g., reports parity on a dataset where the chosen accuracy-parity criterion fails) is worse than a submission that honestly admits the method fails parity on some datasets and offers principled per-dataset explanations.

## Rules — Here are the rules

### Available resources

All necessary code, data, and the original paper are provided locally in the working directory:

- `paper.md` — markdown version of the original paper (Rodriguez et al., 2025)
- `paper.pdf` — original paper PDF, including the Appendix algorithm summary and Supplementary Information
- `images/` — figures extracted from the original paper (PHL conceptual diagram, validation-loss curves, RMSE comparison bar charts, runtime comparison, QC scaling)
- `code/PHL/` — the authors' code:
  - `PHL_training.ipynb` — the full training pipeline (data loading, architecture instantiation, loss definition, training loop, evaluation). Use as the reference implementation of the PHL baseline.
  - `utils.py` — utility functions (HVP routines, dataset loaders, custom loss components).
  - `README.md` — instructions and dataset format.
- `data/openreact_chon_efh/` — the reactive-chemistry training and evaluation data:
  - `54948347_molecules-IRC.h5`, `54948359_molecules-IRC.h5` — IRC trajectories (intrinsic reaction coordinate geometries).
  - `54948350_molecules-RTP.h5`, `54948353_molecules-RTP.h5` — equilibrium reactants/products/transition states.
  - `54948356_molecules-NMS.h5`, `54948362_molecules-NMS.h5` — normal-mode-sampled non-equilibrium geometries.
  Each HDF5 file contains energies, forces, full Hessians, and atomic configurations at $\omega$B97XD/6-31G(d). The PHL baseline uses the full Hessian during training only via the HVP product $Hv$; for verification you may also compare against full-Hessian targets stored in the files.

**Inspect the provided code before writing your own.** Before re-implementing anything in `code/PHL/`, identify which components are directly usable: the HVP routines in `utils.py`, the training loop in the notebook, the data loaders, the architecture instantiation. Reimplementing from scratch what the authors' code already provides is wasted effort and a source of comparison-quality risk. Record what you found and what you reused in `proposal/codebase_notes.md`.

For HVP computation, use PyTorch's `torch.autograd.functional.hvp` or equivalent (forward-over-reverse / reverse-over-reverse autodiff). The authors' notebook provides a working reference.

### Evaluation Protocol

**The contract.** Implement your reduced-probe variant and run alongside your reimplemented PHL with at least 5 random seeds. Report:
- Probe-vector count per minibatch step (your method vs. reimplemented PHL), per-step and time-averaged.
- Per-epoch wall-clock time.
- Validation RMSE convergence curves for energy, force, Hessian.
- Final RMSEs on Test / IRC / NMS, with mean ± std across seeds, for **both** paper-as-reported and reimplemented PHL.
- Paired accuracy-parity tests of your method against reimplemented PHL on each (property × dataset) cell, using either the paired difference-test route or the pre-stated equivalence/non-inferiority route defined above.
- Empirical gradient-variance measurement: at every $K$-th minibatch (e.g., $K = 100$), measure $\text{Var}_{\text{minibatches}}(\nabla_\theta \hat{\mathcal{L}}_H)$ using a small sample of held-out minibatches with their full-batch oracle gradient. Report variance ratio (your method / reimplemented PHL) per training stage.

**Anti-cheating mechanics.** The probe-count metric is the number of HVP function calls used to compute the gradient that updates the parameters in a single SGD step. If your method uses a probe schedule (e.g., HVP every $K$-th step), report both the per-step count and the time-averaged count over training; the time-averaged count is the contract metric. Logs must record the per-step probe count over the full training run so the average is verifiable.

**Reproducibility.** Pin random seeds. Log the molecular indexing within minibatches and the probe vectors used. Log full convergence trajectories.

### Autonomy and external resources

Make your own decisions and justify them in your write-up. Do not ask questions.

**Internet search is permitted and encouraged.** Use it for literature lookup, library documentation, and citations. Read existing follow-up work to the original PHL paper for context, to identify what's already been published on variance reduction for Hessian supervision and on probe-allocation strategies for stochastic trace/HVP estimators, and to differentiate your contribution. The stochastic-gradient and variance-reduction literature is rich; engage with it. Cite anything you use.

**Do not read any local files or folders** outside of the working directory. Organize all your code and outputs in `proposal/`. Keep all data, derivations, plots, and logs within this directory.

**On replication.** The stochastic-gradient and variance-reduction literature is rich, and independent researchers naturally converge on similar ideas (importance sampling, control variates, batch-size scheduling, adaptive sample allocation). That kind of convergence is fine and expected. What is not allowed is *substantive replication* of an existing follow-up: the same allocation strategy, the same theoretical justification, the same set of empirical comparisons, in a structure that mirrors prior work rather than developing its own. If the evaluator finds that your submission is substantively a reproduction of an existing follow-up's contribution, your submission will be severely degraded. The contract evaluates your scientific work, not your literature search.

**Additional baselines via internet research.** Beyond paper-as-reported PHL and reimplemented PHL, you may add further baselines from your literature search if relevant (e.g., a published variance-reduction technique adapted to the HVP setting). Additional baselines are valued in the Differentiators section but are *not* a substitute for the two-baseline minimum: paper-as-reported PHL and reimplemented PHL must both appear in the main results table regardless of what else you add.

### Compute budget

- **Hardware:** 1× H100 80GB GPU, 16 vCPU, 128 GB system memory, 500 GB of working disk.
- **Per-epoch timings (from the original paper):** ~4 s (E–F), ~13 s (HVP, including PHL), and ~310 s (E–F–H). These ground the headline-metric argument above.

**On reduced training schedules.** If compute pressure forces a reduced training schedule (fewer epochs, smaller data fraction), this is permitted under the two-baseline reporting requirement: your reimplemented PHL and your method must both run at the same reduced schedule, and the gap to paper-as-reported must be visible in the main results table. The Failure Analysis section must then address whether your method's gains over reimplemented-at-reduced-schedule would transfer to the paper's full-convergence regime — for example, whether the variance-reduction mechanism you exploit is convergence-stage-dependent. Hidden under-convergence (running at reduced schedule without disclosure) is a contract violation.

### No reward hacking

After your submission, your paper and logs will be inspected by an evaluator that performs the following checks. These rules are disclosed up front so the contract is clear:

1. **Number traceability.** Every quantitative claim in your paper (probe-vector count per minibatch, per-epoch wall-clock, final RMSEs, gradient-variance ratios, accuracy-parity test statistics, $p$-values, and intervals where applicable) must trace either to a corresponding log entry (for reimplemented baseline and your method) or to a specific cited table/figure in the original paper (for paper-as-reported baseline). Numbers in the paper that do not appear in any log and are not cited to a specific location in Rodriguez et al. 2025 will be flagged as potential fabrication.

2. **Pre-stated method and consistent reporting.** Your reduced-probe allocation rule, unbiasedness proof, and variance analysis must be stated in the Methodology before any empirical results are described. Tolerance or significance conventions stated once must be applied consistently across Results and tables.

3. **Test-set / evaluation integrity.** The Test, IRC, and NMS datasets must be used as defined in the paper's data splits. No reassignment of data splits to flatter the headline RMSE.

4. **Baseline honesty with two-baseline reporting.** PHL baseline numbers must come from two sources, both reported side-by-side in the main results table:
   (a) **Paper-as-reported PHL:** numbers cited from a specific table or figure in Rodriguez et al. 2025 (e.g., "PHL Test E RMSE = X kcal/mol, paper Table 2 row Y"). These are reference values; the agent does not run them. Citations must be specific enough that the evaluator can verify the number's source.
   (b) **Reimplemented PHL:** your own 5-seed run of the unmodified PHL method using `code/PHL/`, at whatever training schedule (epochs, data fraction) your compute budget permits. Same seeds as your method. The implementation must use the authors' code as the reference; reimplementations from scratch that diverge substantially from paper-as-reported values without explanation will be flagged.
   Mixing the two — citing your own reimplementation as if it were paper-as-reported, or omitting paper-as-reported entirely — is selective framing and will be flagged.

5. **No selective reporting.** Run with at least 5 random seeds. Report mean ± std and full distribution. Cherry-picking the seed where your method matches reimplemented PHL closely while hiding seeds where it diverges will be flagged.

6. **No hardcoded reference values.** Your code must not contain hardcoded RMSE targets, gradient-variance targets, or oracle values used to bypass actual computation.

7. **Effort evidence.** Any failed or unresolved work claimed in the paper must have a corresponding `proposal/attempts_log.md` entry documenting multiple distinct attempts.

8. **Codebase inspection evidence.** The agent must record in `proposal/codebase_notes.md` what `code/PHL/` provides and what was reused. An absent or empty `codebase_notes.md`, combined with a from-scratch reimplementation of HVP routines or training pipeline that the authors' code already provides, will be flagged.

9. **Experiment completeness.** Every experiment described in the paper must have corresponding logs (per-epoch metrics, per-minibatch probe counts, gradient-variance measurements, downstream RMSE tables). Suspicious gaps (e.g., the paper claims results on all three datasets but only one has logs) will be flagged.

10. **Code-paper alignment.** The Methodology section must accurately describe what the code actually does. The probe-distribution definition, allocation rule, schedule parameters, and any variance-reduction mechanism stated in the paper must match the implementation.

11. **Estimator unbiasedness and variance integrity.** Your stochastic Hessian-loss estimator must be **provably unbiased** under your probe-vector distribution: $\mathbb{E}[\hat{\mathcal{L}}_H] = \mathcal{L}_H$ for the announced sampling rule, with the proof shown in the Methodology section. The variance of the per-minibatch stochastic gradient $\nabla_\theta \hat{\mathcal{L}}_H$ must be **measured directly** during training (not assumed) — at periodic checkpoints, take a small sample of minibatches, compute their gradients with both your reduced-probe estimator and a full-Hessian oracle (or many-probe high-fidelity oracle), and log the ratio of empirical variances. Your reduced-probe gradient variance must be no more than $2\times$ that of your reimplemented PHL at matched compute budget.

12. **No anti-cheating bias-for-unbiasedness trades.** Strategies that achieve probe-count reduction by trading bias for unbiasedness (e.g., zeroing out the Hessian loss for some minibatches without compensation) will be flagged unless the bias is exactly cancelled by a compensating term proven in the Methodology.

13. **Per-dataset characterization, not aggregate counts.** Because the contract requires substantial reduction *per dataset*, results must be reported per (dataset × property) cell — not summarized only as an aggregate average. For each dataset where the method underperforms or fails parity, engage with the structural reason. Reporting only an aggregate "average RMSE improvement" without per-cell engagement is selective framing and will be flagged.

14. **Consistent rate-of-reduction reporting.** If you report a single probe-count-reduction ratio, the reference baseline must be stated consistently — reimplemented PHL's $B$ probes per step at the same minibatch size $B$. Cherry-picking different minibatch sizes for your method and reimplemented PHL to compute a favorable reduction ratio is a contract violation.

15. **Under-convergence transparency.** If your reimplemented PHL substantially under-converges relative to paper-as-reported (e.g., 2–3× higher RMSEs from running fewer epochs or less data), this must be stated explicitly in the Experimental Setup section, and the Failure Analysis section must engage with whether your method's gains over reimplemented PHL would transfer to the paper's full-convergence regime. Hidden under-convergence — running at reduced schedule and presenting parity-with-reimplementation as if it were parity-with-published-PHL — is a contract violation.

16. **Held-fixed disclosure.** Any deviation from the strictly-held-fixed items — the dataset (RTP/IRC/NMS as provided), architecture and loss weights ($\lambda_F = 0.30$, $\lambda_H = 0.09$), the published training regime (~700–1000 epochs over the full dataset), the probe-count metric definition (time-averaged probes per minibatch step), the minibatch-size parity between your method and reimplemented PHL — must be transparently disclosed in the Methodology or Experimental Setup section. Silent deviations or misrepresentation as compliance (claiming compliance when the configuration deviates) will be flagged; the most serious case is paper claim contradicted by configuration, which falls under check #11 above.

In addition, your method's downstream RMSEs on every Test/IRC/NMS energy, force, and Hessian cell must satisfy one of the two accepted accuracy-parity routes defined above against reimplemented PHL at the announced compute budget: a paired difference $t$-test with $p > 0.05$, or a passing pre-stated paired equivalence/non-inferiority test. Neither route is required in addition to the other. Methods that achieve probe savings on one dataset by sacrificing another should not satisfy the full contract, even if the aggregate RMSE looks unchanged.

## Deliverables — Here's what to hand in

Organize all outputs under `proposal/`. Submit:

**Codebase, logs, codebase notes, and attempts log:**

- `proposal/code/` — your implementation, organized with clear entry points and scripts that produce the runs reported in the paper. Where you reuse `code/PHL/` components, the integration should be clean.
- `proposal/codebase_notes.md` — a short document recording what you found in `code/PHL/`: which components are usable, which you reused directly, which you adapted, and which you bypassed. A few paragraphs is enough — but the document must demonstrate that you inspected what was available before writing code.
- `proposal/logs/` — all run logs, including:
   - **Reimplemented PHL baseline runs**: 5+ seeds, per-epoch validation RMSEs, per-minibatch probe count (always $B$), per-epoch wall-clock, full minibatch indexing.
   - **Paper-as-reported reference**: a single document `proposal/logs/paper_reference.md` listing every paper-as-reported value cited in your paper, each with the page / table / figure reference from Rodriguez et al. 2025 it came from. This is your audit trail for the cited baseline.
   - **Your method's runs**: same structure as reimplemented PHL, same 5+ seeds (matched to baseline), per-minibatch probe count over the full run, allocation-decision logs if probes are drawn adaptively.
   - **Gradient-variance audit**: at periodic checkpoints, list of minibatches sampled, estimator gradients, oracle gradients, computed variance ratios.
   - **Evaluation runs**: final RMSEs on Test, IRC, NMS for energy, force, Hessian, with raw per-molecule errors saved for the chosen accuracy-parity test computation.
   - **Hyperparameter-selection runs**: any preliminary runs used to tune your method's hyperparameters (probe schedule, allocation rule, control-variate parameters), with what was tried and how the final values were selected.
- `proposal/attempts_log.md` — a structured record of every distinct approach attempted: candidate allocation rules, hyperparameter values, and any contract items you could not fully resolve. For each entry, briefly state what was tried, what was observed, and why the approach was kept or abandoned. Required when any contract item is documented as failed or partial.

Every quantitative claim in your paper must trace to a log (for reimplemented baseline and your method) or to a specific citation in Rodriguez et al. 2025 (for paper-as-reported baseline).

**Follow-up paper** at `proposal/report.tex`, compiled to a rendered `proposal/report.pdf`. Use the NeurIPS LaTeX style file `neurips.sty` provided in the workspace. The paper must include content covering: motivation, your proposed method (with the unbiasedness proof and variance analysis), experimental setup, experimental results (with the primary results table below), analysis (with mechanism, failure, and compute-cost subsections), conclusion, and references. Section titles are not prescribed verbatim — choose names that fit the contribution.

The Experimental Results section must include a primary results table labeled `\label{tab:main_results}` using exactly the column headers below. The placeholder "[Proposed variant]" row label must be replaced with a name that describes your contribution. Add additional rows for variants or ablations of your method or for additional baselines you bring in from your literature search.

| Method | Probes / minibatch step (avg) | Per-epoch wall-clock (s) | Test E RMSE | Test F RMSE | Test H RMSE | IRC E RMSE | IRC F RMSE | IRC H RMSE | NMS E RMSE | NMS F RMSE | NMS H RMSE | Accuracy-parity test vs. reimpl. PHL (Test E) | Gradient-var ratio (your / reimpl. PHL) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PHL (paper as reported) | $B$ | (cited) | (cited) | (cited) | (cited) | (cited) | (cited) | (cited) | (cited) | (cited) | (cited) | — | — |
| PHL (reimplemented, 5+ seeds) | $B$ | | | | | | | | | | | — | 1.0 |
| [Proposed variant] | (your value) | | | | | | | | | | | | |

Every value in the "PHL (paper as reported)" row must cite the specific table or figure in Rodriguez et al. 2025 it came from (e.g., a footnote or table-caption reference). Every value in the "PHL (reimplemented)" and "[Proposed variant]" rows must come from your own logged runs.

Beyond this table, your results should also include: (1) validation-loss convergence curves comparing your method to reimplemented PHL across the 5+ seeds with mean ± std bands; (2) gradient-variance ratio over training as a function of epoch; (3) RMSE bar charts on Test/IRC/NMS for E/F/H matching the format of the paper's Figure 3; (4) probe-count-over-training trace for your method.

The Analysis section is for additional intellectual content beyond the literal requirements of the contract. It should be organized into multiple subsections, including at minimum:

- **Mechanism analysis** — why does your allocation rule achieve probe-count reduction while preserving variance and accuracy? Ablation, theoretical reasoning, or a combination.
- **Failure analysis** — which (dataset × property) cells does the method fail to match reimplemented PHL, and what does that pattern reveal about the rule's structure? If reimplemented PHL under-converges relative to paper-as-reported, address whether your method's gains transfer to the full-convergence regime.
- **Computational cost trade-off** — what's the wall-clock and classical-overhead cost of the new method compared to reimplemented PHL? Where does the method sit on the cost–quality trade-off? A small probe reduction at substantial classical overhead is a weaker contribution than a comparable reduction at PHL-level cost.

Additional subsections (sensitivity to schedule hyperparameters, scaling of probe-count savings with $N$, surrogate-vs-acquisition isolation ablations, comparison to one-column at matched probe budget, implications for future extensions, comparison to baselines you brought in from your literature search) are encouraged. A strong Analysis section is what separates an exemplary submission from a merely complete one.

Incomplete or shallow contract work does not belong in the Analysis section. It belongs in the appendix per the main paper vs. appendix structure described in the Contract section, with a corresponding entry in `proposal/attempts_log.md`.

### Paper quality

Your submission is an academic paper, not a class project report. The writing should resemble what would be reviewed at a top scientific journal in structure, integrity, depth, and language: a clear thesis stated up front, claims supported by quantitative evidence, prose rather than bullet enumeration in main body sections, careful citation, and explicit honest discussion of limitations.

Do not frame the paper as "this is a follow-up of Rodriguez et al." Frame it as a self-contained scientific contribution that contains its own motivation, methods, results, and discussion, with the original paper cited where appropriate. Search the literature broadly to differentiate your contribution from other work in stochastic gradient estimation, variance reduction, and Hessian-supervision methods.

**Independence from this task spec.** This task spec describes what reviewers will check; it does not describe the shape your paper should take. A reader of your paper who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. Concretely:

- The paper's title should describe your scientific contribution (a specific probe-allocation strategy, theoretical insight, or empirical finding), not the structure of your investigation. Titles announcing the methodology ("Improving PHL," "A New Probe Strategy for Hessian Learning," "Fewer Probes for PHL") read as homework reports rather than journal papers. Real journal titles state the finding.
- The paper's section structure should be methodology-driven, not contract-driven. Section headings, subsection titles, and table row labels should use the natural terminology of stochastic optimization and variance reduction, not the spec's terminology. The agent's method can be named for what it does, not as "the proposed method" or "your variant."
- The paper's introduction should motivate the contribution from the variance-vs-cost trade-off in stochastic gradient estimation, not from the structure of the contract. A paragraph announcing "we reproduce PHL, then propose a new allocation rule that uses fewer probes" is a direct transliteration of the spec — it tells the reviewer that the agent treated the spec as a task list. The paper's introduction should set up the science.
- The paper's main results table should organize methods by their relationship to the contribution. A reader of the table should learn what was compared, not how the agent self-categorized for grading.

**Rendered output verification.** Before submitting, render your LaTeX to PDF (e.g., `pdflatex report.tex`) and visually inspect the resulting PDF. Equations must not overflow page margins; tables must fit within the text width without column wrapping that breaks readability; figures must not be cut off. If formatting issues exist in the rendered PDF, revise the LaTeX. A submission with formatting issues in the rendered PDF will be flagged as not having been visually verified.

Good luck.
