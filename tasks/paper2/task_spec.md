# Task: Develop a Substantive Follow-up to "Extending the Accuracy of the SNAP Interatomic Potential Form"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to "Extending the accuracy of the SNAP interatomic potential form" by M. A. Wood and A. P. Thompson (J. Chem. Phys. 148, 241721, 2018), in the direction of **replacing the bispectrum components with a learned descriptor** while preserving accuracy parity and characterizing the data efficiency and generalization properties of the substitution.

## The Problem — Here's why it matters

A machine-learned interatomic potential (MLIP) maps an atomic environment $\{r_{ij}\}$ to a per-atom energy contribution $E^i$ such that the total energy is $E(\{r_i\}) = \sum_i E^i$ and the forces $F^j = -\nabla_j E$ are analytic gradients with respect to neighboring atoms j. Most MLIPs in the density projection family — SNAP, ACE, MTP — separate this map into two stages: a hand-engineered **descriptor** that maps an atom's neighborhood to a fixed-length vector $\mathbf{B}^i \in \mathbb{R}^K$, and a **regression model** that maps $\mathbf{B}^i$ to $E^i$ via fitted coefficients. The descriptor encodes three structural properties of the physics — rotation, translation, and permutation invariance — *by construction*. The regression model inherits those invariances for free.

In quadratic SNAP specifically, $\mathbf{B}^i$ is the vector of $K$ bispectrum components (rotation-invariant scalar triples of 4D hyperspherical-harmonic expansion coefficients of the neighbor density), and the regression is:

$$E^i_{\text{SNAP}} = \boldsymbol{\beta} \cdot \mathbf{B}^i + \tfrac{1}{2} (\mathbf{B}^i)^\top \boldsymbol{\alpha}\, \mathbf{B}^i,$$

with $\boldsymbol{\beta} \in \mathbb{R}^K$ and $\boldsymbol{\alpha} \in \mathbb{R}^{K \times K}$ symmetric, fit by linear least squares to DFT energies and forces. The constant $\beta_0$ is constrained to zero so the isolated-atom energy is exactly zero (`bzeroflag=1` in LAMMPS).

Encoding the three invariances into the descriptor at design time is what makes SNAP-family methods data-efficient: every training configuration produces $K$ rotation-invariant features regardless of its orientation, so the model never has to learn that physics. A naturally suggestive but largely unexplored question is what changes if the descriptor itself is *learned* from data, rather than hand-engineered to satisfy the invariances. A learned descriptor can in principle pick up problem-specific structure that bispectrum components cannot, and can be cheaper to evaluate at inference. The cost is that the invariances are no longer guaranteed: the model must either learn them from training data, encode them architecturally, or use data augmentation — each of which has implications for sample efficiency, interpolation quality, and out-of-distribution behaviour.

## Background — Here's what you need to know about the original paper

The Wood & Thompson 2018 paper introduces and validates the quadratic SNAP form on a single material — tantalum — using a fixed training set of 363 DFT configurations (PBE-PAW, VASP), partitioned into 12 groups: `Displaced_{A15,BCC,FCC}`, `Volume_{A15,BCC,FCC}`, `Elastic_{BCC,FCC}`, `GSF_{110,112}`, `Liquid`, `Surface`. The dataset contains 362 energy data points and 15,297 force-component data points (configs have 24–100 atoms).

The reference numbers from the paper (Supplemental Tables SI/SII) for the same 363-config training set:

| Method | $J_{\max}$ | $K$ | $r_{\text{cut}}$ (Å) | $\delta E$ (eV/atom) |
|---|---|---|---|---|
| Linear SNAP | 4 | 55 | 5.672 | 0.03041 |
| Quadratic SNAP | 4 | 55 | 5.594 | **0.00898** |
| Quadratic SNAP | 5 | 91 | 7.039 | 0.00163 |

Quadratic SNAP at $J_{\max}=4$ is the primary reference point in the paper — a $\sim 3.4\times$ improvement in energy MAE over linear SNAP at the same descriptor count, achieved by adding the $K(K{+}1)/2 = 1540$ quadratic coefficients $\boldsymbol{\alpha}$ on top of the $K=55$ linear coefficients $\boldsymbol{\beta}$.

The paper also runs a cross-validation analysis (paper §VI, Figure 5): training fractions are swept in 5% increments from 5% to 95% inclusive, with **10 random draws per fraction**, and energy/force MAE is recorded on the held-out portion at each fraction. Quadratic SNAP overfits catastrophically below $\sim 30\%$ training data (where the number of fit coefficients exceeds the number of training points), then converges to its full-data MAE as the included fraction grows. Linear SNAP, with far fewer coefficients, is comparatively flat across the sweep. Per-group convergence rates differ: high-symmetry crystal groups need very little data, while disordered groups (GSF, Liquid, Surface) need substantially more. The paper plots Figure 5 for the GSF(110) group as one example of a slow-converging group.

## Contract — Here's exactly what counts as success

The substantive contribution is to design a **learned descriptor** $\mathbf{B}^i_\theta : \text{(atomic environment)} \to \mathbb{R}^{K'}$ that replaces the bispectrum components in the SNAP energy form, and characterize how this substitution behaves on accuracy, data efficiency, generalization breadth, and the structural invariances. You design the descriptor architecture; you choose its output dimension $K'$; you decide how (or whether) to enforce the invariances. The downstream regression (linear-plus-quadratic in the descriptor, fit by least squares) is held fixed so the comparison isolates the descriptor's contribution.

Your submission clears **three independent bars**:

1. **Substantive contribution** — the learned descriptor achieves accuracy parity with quadratic SNAP at full training data on tantalum, demonstrates a meaningful data-efficiency outcome on the paper's cross-validation protocol, and generalizes to at least one additional refractory transition metal beyond tantalum. (See "Operationalized success thresholds" below.)
2. **Honest framing** — accuracy, data efficiency, and invariance results are reported per-material, per-fraction, and per-test-type, including the cases where the learned descriptor *does not* outperform quadratic SNAP. Aggregate-only reporting that masks per-cell behaviour is a contract violation.
3. **Code–paper consistency** — the descriptor architecture and training procedure described in the paper match what the code implements. Hyperparameters quoted in the paper match the logs.

These three bars are checked independently by the rubric.

### Operationalized success thresholds

| Axis | Floor (must clear) | "Substantial" (the hallmark) |
|---|---|---|
| **Accuracy parity on tantalum at full training data** | $\delta E_{\text{learned}} \leq \delta E_{\text{reimpl.\ quadratic SNAP}}$ on the 80/20 split or the paper's full-data CV endpoint, **and** $\delta F_{\text{learned}} \leq \delta F_{\text{reimpl.\ quadratic SNAP}}$ at matched protocol | — |
| **Data efficiency** | A coherent curve is reported across the paper's full 5%-increment sweep (5%, 10%, …, 95%) with 10 random draws per fraction | Learned descriptor reaches quadratic SNAP's full-data $\delta E$ at **≤ 50%** of the training data. **Report this outcome two ways, separately: (a) on the equal-weight all-12-group aggregate, and (b) on the low-symmetry / slow-converging subset (GSF, Liquid, Surface)** — the latter is the discriminating bar, since the high-symmetry groups reach full accuracy at tiny data fractions for both methods and do not distinguish the descriptors. (Modest: at 50–80% data. Matches: at 80–95% data. Regression: needs more data than quadratic SNAP.) |
| **Generalization breadth** | Learned-descriptor model is fit and evaluated on **at least one** additional refractory transition metal beyond tantalum, chosen from the three pre-staged materials (Mo, Nb, Re), with the same architecture and a same-protocol data-efficiency sweep | Substantial data efficiency on **all three** transfer materials. The Re HCP transfer is the stiffest structural-generalization test because Re's crystal symmetry differs from the BCC Ta/Mo/Nb baseline. (Modest: substantial on at least one, modest on the others. Matches: parity but no efficiency win. Fails to transfer: regression on at least one.) |
| **Invariance compliance** | Quantitative invariance audit reported (rotation / translation / permutation drift) on held-out configurations, with comparison to quadratic SNAP's by-construction machine-precision drift | The audit is engaged with — drift magnitudes are tied to architectural choices, and any failure of invariance is explained mechanistically rather than dismissed as noise |
| **Inference cost** | Wall-clock per atom-timestep is measured for learned-descriptor MD vs. quadratic SNAP MD at matched system size | Net inference is faster than quadratic SNAP at matched accuracy — but this is a secondary axis, not a contract requirement |

### Held fixed (parity constraints)

These are not part of the design space. They must remain identical to the paper's protocol on tantalum:

- **The training data**: the 363 DFT configurations in `data/ta_dft_training_jsons/` for the tantalum experiments. You may not augment this set with externally-sourced tantalum DFT data, with DFT calculations you run yourself, or with synthetic configurations.
- **The energy form**: the proposed method must use a SNAP-style energy decomposition $E^i = \boldsymbol{\beta} \cdot \mathbf{B}^i_\theta + \tfrac{1}{2}(\mathbf{B}^i_\theta)^\top \boldsymbol{\alpha}\, \mathbf{B}^i_\theta$ where $\mathbf{B}^i_\theta$ is the learned descriptor and $(\boldsymbol{\beta}, \boldsymbol{\alpha})$ are fit by linear least squares on the training subset. The motivation for holding this fixed is that the comparison evaluates the descriptor, not the regression head. An end-to-end neural-network energy model is out of scope under this contract (it would conflate the descriptor contribution with the regression-head contribution).
- **The zero-energy constraint**: `bzeroflag=1` (no free constant and the isolated-atom energy is exactly zero by construction of the energy model). Note that SNAP models will subtract isolated atom descriptor values to ensure this constraint. This is also allowed in your pursuit. 
- **Forces are analytic gradients of the energy** — backpropagated through the learned descriptor, not separately fit.
- **The cross-validation protocol on tantalum**: training fractions in 5% increments from 5% to 95% inclusive, 10 random draws per fraction, with the constraint that energies and forces from a single DFT configuration are kept together (cannot be split across included/excluded sets). The full-data point (100% included) is reported as the headline accuracy-parity value.
- **Both aggregate and subset, reported separately**: the data-efficiency comparison is reported two ways — the equal-weight all-12-group aggregate, and restricted to the low-symmetry / slow-converging subset (GSF, Liquid, Surface). No single group is privileged; the subset is the discriminating bar because the high-symmetry groups reach full accuracy at tiny data fractions for both methods.

> **Note on matched-budget framing.** The contract evaluates your method's behaviour on the paper's published training and CV regime. A reduced-regime comparison — e.g., 5 random draws per fraction instead of 10, or fewer fractions, or a smaller training set — does not satisfy the contract even if it produces a well-defined paired comparison between learned-descriptor and quadratic SNAP at the reduced regime. The protocol is part of what is being held fixed because the paper's overfitting analysis (Figure 5) only manifests at sufficient sweep resolution.

The transfer materials (Mo, Nb, Re — pre-staged in `data/{mo,nb,re}_dft_training/`) inherit the same 5%-increment × 10-draws CV protocol applied to whichever configuration groups the source dataset defines. New $(\boldsymbol{\beta}, \boldsymbol{\alpha})$ coefficients and radial cutoff, $r_{\text{cut}}$, are to be fit for each element. Re's HCP crystal structure, distinct from the BCC structure of Ta/Mo/Nb, makes Re the stiffest invariance-and-structural-generalization test of the three. Plan compute accordingly.

### What you may design

- The descriptor architecture $\mathbf{B}^i_\theta$: parametric form, output dimension $K'$, hyperparameters, training procedure, how (or whether) the rotation / translation / permutation invariances are encoded.
- The cutoff radius $r_{\text{cut}}$ of the descriptor — though comparing fairly against quadratic SNAP at $J_{\max}=4$ requires reporting the learned descriptor's cutoff alongside the quadratic baseline's $r_{\text{cut}}=5.594$ Å.
- The descriptor's training data: you may train the descriptor on the 363 tantalum configurations alone, or on a broader collection drawn from the pre-staged transfer-material datasets (or from OMatG, for exploration beyond the pre-staged collection — see "Available resources"). If you train on a broader collection, the broader training is itself a research design choice that should be motivated, justified, and ablated.
- Whether the same descriptor is reused across Ta / Mo / Nb / Re (one descriptor architecture, possibly with shared or material-specific weights) or whether each material gets its own. Either is acceptable; the choice is part of the contribution. A uniform descriptor construction across elements (one architecture) is preferred, since it strengthens the cross-material generalization claim; if different descriptors are constructed per element, rationalize your choices.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated along two axes:

1. **Performance on stated metrics** — accuracy parity at full data, data-efficiency sweep curves, invariance drift, generalization breadth.
2. **Scientific merit** — depth of analysis, quality of mechanism / failure / cost discussion, engagement with the published learned-descriptor literature, soundness of the architectural choices. Scientific merit values ambition and depth without prescribing what counts as ambitious.

### Two-baseline reporting (required)

Every quantitative comparison must report two quadratic-SNAP baselines side-by-side:

1. **Paper-as-reported quadratic SNAP** — values cited from Tables SI/SII of Wood & Thompson 2018. The primary headline reference is $J_{\max}=4$, $r_{\text{cut}}=5.594$ Å, $\delta E = 0.00898$ eV/atom (Table SII). Cite every paper-as-reported value to its specific source location.
2. **Reimplemented quadratic SNAP** — your own run via FitSNAP on the same 363 configurations at the same protocol you will use for the learned descriptor, at $J_{\max}=4$, $r_{\text{cut}}=5.594$ Å, 5%-increment CV with 10 random draws. This is the paired-comparison baseline.

Your learned-descriptor model is compared at matched protocol against the reimplemented quadratic SNAP. The paper-as-reported baseline serves as an external anchor — if your reimplementation diverges substantially from paper-as-reported, that gap must be documented and addressed in your failure analysis.

### Cross-validation protocol on tantalum

Reproduce the paper's Figure 5 protocol exactly:

- Training fractions: 5%, 10%, 15%, …, 95% included (i.e., 95%, 90%, …, 5% held-out), plus the full-data point (100% included, no held-out).
- 10 random draws per fraction. Random draws are stratified within each of the 12 training groups so each fraction's included set has representation from each group proportional to that group's share of the total.
- A DFT configuration's energies and forces are kept together — a configuration is either entirely in the included set or entirely in the excluded set for a given draw.
- Report mean ± standard deviation across the 10 draws at each fraction.
- Compute both energy MAE ($\delta E$, eV/atom) and force MAE ($\delta F$, eV/Å) on the held-out portion at each fraction.
- Report per-group data-efficiency curves for all 12 training groups plus an equal-weight all-groups aggregate; no single group is privileged. The slow-converging groups (GSF, Liquid, Surface) are where the comparison is most informative, since the high-symmetry groups saturate at low data fractions for both methods.

### Generalization breadth

At least one refractory transition metal beyond tantalum must be evaluated. Three transfer materials are pre-staged in the workspace:

- `data/mo_dft_training/` — Molybdenum (BCC refractory)
- `data/nb_dft_training/` — Niobium (BCC refractory)
- `data/re_dft_training/` — Rhenium (HCP refractory) — included as the stronger structural-generalization test, since the learned descriptor's invariance properties are exercised on a different crystal-symmetry environment than the BCC headline.

All three transfer datasets come from a single curated source (Andolina & Saidi 2023, the `23-Single-Element-DNPs` collection) with matched DFT protocol (PBE/VASP), shared license (GPL-3.0), and identical file structure across materials. The configurations are MD trajectories at multiple temperatures (melting point, 0.6·MT, 0.25·MT) with adaptive-learning-refined iterations — different group structure from the Wood & Thompson tantalum dataset's 12 thermodynamic groups, but comparable configuration diversity. Files are in DeePMD format (energy + force npy arrays plus VASP trajectories inside `iter_*.tar.xz` archives) and require conversion to FitSNAP's per-configuration JSON format before fitting; the conversion is straightforward and is part of your pipeline.

For each transfer material:
- Fit the learned-descriptor model (same architecture, possibly retrained weights) and a reimplemented quadratic-SNAP model on the new material. Learned descriptor model and comparisons to quadratic-SNAP must use the exact same training set.
- Run the same 5%-increment × 10-draws CV sweep.
- Report comparison curves analogous to the tantalum headline (energy MAE vs. fraction included, mean ± std across 10 draws, both methods overlaid). Per-group breakdowns where the transfer dataset's group structure permits; the equal-weight all-groups aggregate otherwise.

For OMatG-sourced ambition beyond the pre-staged transfer materials, see "Available resources" — but any rubric-evaluable transfer-material results must come from the pre-staged datasets to ensure cross-submission comparability.

### Invariance audit

Apply the following tests to the trained learned-descriptor model on tantalum and report drift quantitatively:

- **Rotation**: 100 uniformly random elements of $SO(3)$ applied to each of 10 randomly chosen held-out configurations (from any group). For each rotated configuration, report the predicted energy and forces. Report `max |ΔE / E|` and `max |ΔF / F|` across all 1000 rotated samples, where the deltas are against the unrotated configuration.
- **Translation**: 10 uniformly random translations (each component drawn from $\mathcal{U}[-1, 1]$ Å) applied to each of the same 10 held-out configurations. Report the same drift metrics.
- **Permutation**: 10 random permutations of the atom indices within each configuration (all atoms are the same species in single-element materials, so any permutation is valid). Report the same drift metrics.

Quadratic SNAP gives machine-precision drift on these tests by construction. Report your learned-descriptor drift alongside, and engage with the magnitudes in your analysis — large drift on a non-invariant architecture is expected; the question is how large, and how it affects downstream prediction quality. There is no fixed pass/fail threshold on drift magnitudes; the rubric grades the quality of your engagement with the numbers.

### Inference cost

Report wall-clock per atom-timestep for the learned-descriptor MD vs. quadratic SNAP MD at a matched system size (e.g., a 128-atom BCC supercell at room temperature). Use LAMMPS for the quadratic SNAP baseline (the standard SNAP/MLIAP integration) and whichever tooling you use to evaluate your descriptor.

## Rules — Here are the rules

### Available resources

The following are provided in the working directory:

- `paper.md`, `paper.pdf` — Wood & Thompson 2018, the source paper.
- `images/` — figures extracted from the source paper.
- `code/FitSNAP/` — FitSNAP, the standard fitting toolkit for SNAP-family potentials. Used to produce both the reimplemented quadratic SNAP baseline (matching the paper's protocol) and, if you choose, the linear-plus-quadratic regression head on top of your learned descriptor.
- `code/dakota/` — the DAKOTA optimization toolkit. Used by the original paper for genetic-algorithm hyperparameter optimization across training-group weights. Available but not required; you may choose simpler weight choices for your comparison if your contribution does not depend on the GA optimization.
- `code/QUIP/` — QUIP, a general MLIP and MD framework. Available; not required.
- `code/OMatG/` — the FERMat-ML / OpenKIM-overlapping crystal-generation framework, explicitly cited by the paper-2 domain author as an entry point into the MLIAP-curation ecosystem. OMatG itself ships crystal-structure datasets (MP-20, perov-5, carbon-24, Alex-MP-20) suitable for structure generation but not directly for SNAP-style MLIAP fitting (no DFT energies + forces in the LMDB files). The transfer-material *training* data is therefore pre-staged separately from `saidigroup/23-Single-Element-DNPs` (a single-source MLIAP-curated dataset family in the same OpenKIM-overlapping ecosystem). You may use OMatG for additional crystal-structure exploration, configuration generation for invariance probes, or post-hoc analyses, but any rubric-evaluable transfer-material results must come from the pre-staged datasets below.
- The three pre-staged transfer-material datasets (below) are sourced from `saidigroup/23-Single-Element-DNPs` (Andolina & Saidi 2023, GPL-3.0). The same source also publishes 20 other single-element DNP datasets (Ag, Al, Au, Co, Cu, Ge, I, Kr, Li, Mg, Ni, Os, Pb, Pd, Pt, Sb, Sr, Ti, Zn, Zr), each following the URL pattern `https://raw.githubusercontent.com/saidigroup/23-Single-Element-DNPs/main/Training_Data/{Element}/`. If your contribution extends beyond the three pre-staged refractories, you may fetch additional elements via internet at that URL pattern; cite Andolina & Saidi 2023 if you do.
- `data/ta_dft_training_jsons/` — the 363 DFT configurations from the source paper, organized into 12 group subdirectories. See `README.md` and `Ta-example.in` for format and the FitSNAP convention.
- `data/ta_quadratic_trained/Ta_Quadratic_JCP2018/` — the source paper's pre-trained quadratic SNAP model (`Ta_pot.snapcoeff`, `Ta_pot.snapparam`) for direct comparison. See its `README.md` and `Ta-example.in` for usage.
- `data/mo_dft_training/` — Molybdenum DFT training data from Andolina & Saidi 2023. DeePMD format inside `iter_all_Mo.tar.xz` (~12 MB). Convert to FitSNAP JSON before fitting.
- `data/nb_dft_training/` — Niobium DFT training data from the same source. DeePMD format inside `iter_al_Nb.tar.xz` (~15 MB).
- `data/re_dft_training/` — Rhenium DFT training data from the same source. DeePMD format inside `iter0_Re.tar.xz` + `iter1_2_Re.tar.xz` (~30 MB total).

### Compute budget

- **Hardware**: 1× NVIDIA H100 80GB, 16 vCPU, 128 GB RAM, 500 GB disk.
- **Effort expectation**: the data-efficiency sweep is the headline experiment; the invariance audit, inference-cost measurement, and breadth experiments are supporting evidence. Submissions that finish early without exhausting the budget typically have unrealized improvements.

If compute pressure forces a tradeoff between the strict 10-draw sweep and breadth experiments, complete the tantalum 10-draw sweep first (it is the headline reproduction of Figure 5), then run breadth experiments at whatever fidelity remaining compute supports. Document any compute-driven choices transparently in Methodology.

### Autonomy and external resources

**Internet search is permitted and encouraged.** Use it for literature lookup, documentation of external libraries, citations of published learned-descriptor approaches, and identification of post-2018 developments in the MLIP descriptor landscape. Cite anything you use.

**Workspace constraint.** You may not read local files outside the provided working directory.

**No DFT generation.** No DFT engine is provided in the workspace. You may not run new DFT calculations to generate additional training data. The tantalum experiments use the 363 configurations as provided; transfer-material experiments use the pre-staged Andolina & Saidi 2023 datasets in `data/{mo,nb,re}_dft_training/`. Symmetry operations on existing configurations (rotation, translation, permutation augmentation) are allowed as a research choice but are subject to the invariance-gaming check — see "No reward hacking" check 9 below.

**Anti-replication.** Independent researchers working on learned-descriptor potentials will naturally converge on some shared architectural choices — that kind of convergence is fine and expected. What is not allowed is doing exactly what a published method already did: re-implementing the same architecture, the same training procedure, the same experiments, and presenting the result as a new contribution. The contract evaluates your scientific work, not your literature search. If the evaluator finds that your submission is substantively a re-implementation of an existing published learned-descriptor potential, your submission will be severely degraded. Acknowledged convergence with prior work is fine; silent or misrepresented substantive overlap is not.

**No questions.** Do not query the user for clarification. Make your own design decisions and justify them in your write-up.

### No reward hacking

The following checks will be applied to your submission. They are disclosed up front so the contract is clear.

1. **Number traceability.** Every quantitative claim in the paper must trace to a log entry or to a specific citation in the source paper. Numbers without a verifiable source will be flagged.

2. **Two-baseline integrity.** Paper-as-reported quadratic SNAP values must trace to specific tables/figures in Wood & Thompson 2018 (e.g., "Table SII, row $J_{\max}=4$"). Reimplemented quadratic SNAP values must trace to your own FitSNAP run logs. Mixing the two — citing your reimplementation as if it were paper-as-reported, or vice versa — will be flagged.

3. **Pre-stated method.** The learned descriptor's architecture, hyperparameters, and training procedure must be stated in the paper's methodology before any empirical results are described.

4. **Held-fixed disclosure.** Any deviation from the strictly-held-fixed items (training data, energy form, `bzeroflag`, analytic-gradient forces, CV protocol, the data-efficiency reporting protocol of all 12 groups plus the low-symmetry subset) must be transparently disclosed in Methodology. Silent or misrepresented deviations will be flagged.

5. **CV protocol integrity.** The 5%-increment × 10-draws sweep must be run as described. Subsampling the protocol (fewer fractions, fewer draws) without disclosure will be flagged. The whole-configuration constraint (energies and forces for a single DFT config kept together) must be respected in your fold construction.

6. **No selective reporting.** All 10 random draws per fraction must be reported (mean ± std). Per-(material × fraction × group) cells must be reported individually; aggregate-only reporting that masks per-cell behaviour will be flagged.

7. **Data-efficiency reference consistency.** Sample-efficiency ratios (e.g., "reaches baseline's full-data error at 30% of the data") must use a consistent reference budget across materials. Inconsistent reference choice that flatters the headline will be flagged.

8. **Code–paper alignment.** The descriptor architecture and training procedure described in Methodology must match the code. Hyperparameters quoted in the paper must match the logs.

9. **Invariance test integrity.** The invariance audit's predictions must come from a model trained without data augmentation that secretly encodes the test transformations. If you augment your training data with random rotations / translations / permutations as a training strategy, the invariance audit becomes a check on the augmentation rather than the architecture, and this must be disclosed: report invariance numbers both *with* and *without* the augmentation, and discuss the difference. Concealed training-time augmentation that games the invariance test is a violation.

10. **No hardcoded reference values.** Code must not contain hardcoded DFT energies, forces, or any quantity that bypasses actual computation.

11. **No DFT generation.** Your submission must not contain DFT calculations you ran yourself to expand the training set. The contract holds the training data fixed at the paper's 363 configurations for tantalum, and at the pre-staged Andolina & Saidi 2023 datasets (`data/{mo,nb,re}_dft_training/`) for transfer materials.

12. **Descriptor integrity.** The learned descriptor's input must be raw atomic coordinates and species labels (or quantities derivable from them — interatomic distances, angles, neighbor positions). The descriptor may not receive DFT-reference quantities, lookup-table results indexed on the target configuration, or any leakage from the labels.

13. **Codebase inspection.** Document in `codebase_notes.md` what you inspected in FitSNAP, OMatG, and DAKOTA before reimplementing anything those packages already provide.

14. **No replication of published learned-descriptor architectures.** Your descriptor must not be a re-implementation of a published learned-descriptor potential. Acknowledged convergence with prior work is fine; silent substantive overlap is not (see "Anti-replication" above).

15. **Effort evidence.** If you describe any work as partial, unresolved, or attempted-but-failed, `proposal/attempts_log.md` must document the approaches tried and why each was set aside.

16. **Reimplementation gap disclosure.** The reimplemented quadratic SNAP baseline anchors the paired comparison against the learned descriptor; paper-as-reported quadratic SNAP at $J_{\max}=4$ ($\delta E = 0.00898$ eV/atom on tantalum, Table SII) anchors that paired comparison against the published literature. If your reimplemented quadratic SNAP $\delta E$ diverges substantively from paper-as-reported at the comparable protocol (full-data CV endpoint on Ta), the gap must be disclosed honestly: state the gap magnitude in the failure-analysis or limitations discussion, name the likely cause (data pipeline mismatch, fitting hyperparameters, FitSNAP configuration), and discuss its implications for the learned-descriptor comparison. Silent divergence — where the reimplementation under-performs paper-as-reported and the gap is not mentioned — will be flagged because it makes the learned descriptor look artificially better than it is. Active misrepresentation (citing paper-as-reported numbers as if they were your reimplementation, or vice versa) is a more severe violation already covered by check 2.

17. **Held-out integrity within CV draws.** Held-out portions in each random draw must be genuinely held out from the training procedure on that draw. Concretely: any descriptor pretraining (self-supervised or otherwise) may not see the held-out configurations of the current draw; descriptor hyperparameter selection (output dimension $K'$, architectural variants, learning rate, etc.) may not be tuned on held-out validation drawn from the held-out portion; the linear-plus-quadratic regression head must be fit on the included portion only. If your pipeline involves a separate validation split for early stopping or hyperparameter selection, it must come from within the included portion of the current draw, not from the held-out portion. Leakage of held-out data into any training-time decision will be flagged as a violation of the paired-comparison protocol.

18. **Transfer-material data provenance.** The transfer-material pipelines (data loading, DeePMD-to-FitSNAP conversion, fitting) must read from the pre-staged datasets at `data/mo_dft_training/`, `data/nb_dft_training/`, and `data/re_dft_training/`, using the configurations contained in the upstream `iter*.tar.xz` archives (Andolina & Saidi 2023). Substituting these with other refractory-element datasets (e.g., other elements from the saidigroup collection beyond Mo/Nb/Re), with externally-sourced Mo/Nb/Re data, or with synthetic configurations generated via OMatG or similar tools — without explicit disclosure — will be flagged. The held-fixed clause on training data extends to transfer materials; the pre-staged datasets are the contract reference.

19. **Architecture consistency across materials.** The "What you may design" clause permits two architectural choices: a single shared descriptor architecture across Ta and the transfer materials (with shared or material-specific weights), or a per-material architecture. Either choice is acceptable — but the choice must be honestly disclosed in Methodology, and the code must implement what the paper describes. If the descriptor architecture used for Ta differs in structure from the architecture used for any transfer material (different layer types, different feature-map dimensions, different invariance mechanism, etc.), the per-material variation must be stated explicitly with rationale. Silent per-material architecture variation — where the paper implies a single architecture but the code uses different architectures across materials — will be flagged because it conflates the descriptor contribution with architecture-shopping.

## Deliverables — Here's what to hand in

1. **Codebase and logs.** Code organized under `proposal/code/`; logs under `proposal/logs/`. Logs must include:
   - Per-run hyperparameters, RNG seeds, intermediate metrics, final metrics, timestamps for every training run.
   - The complete CV sweep — every (material, fraction, draw) combination logged with both energy and force MAE on the included and excluded portions.
   - The reimplemented quadratic-SNAP runs, organized so the same sweep structure is visible.
   - The invariance audit logs — per-test-type per-configuration drift in both energy and forces.
   - The inference-cost timing runs.
   - A `proposal/logs/paper_reference.md` (or equivalent) listing every paper-as-reported value cited in your paper with its specific table or figure source in Wood & Thompson 2018.

2. **Follow-up paper.** A LaTeX paper at `proposal/report.tex` rendered to `proposal/report.pdf`, using the NeurIPS style file (`neurips.sty`, provided in the workspace). The paper should read as a self-contained journal-style contribution and must include:
   - **Motivation** for the learned-descriptor substitution and what's at stake.
   - **A description of the proposed descriptor**: architecture, parameter count, training procedure, invariance properties (claimed and demonstrated), cutoff radius, and the rationale for design choices.
   - **A description of the experimental setup**: the 5% × 10 CV protocol, two-baseline rows, transfer-material protocols, invariance audit protocol, inference-cost protocol.
   - **A primary results table** labeled `\label{tab:main_results}` with two-baseline rows. Suggested structure (you may adapt column names and add rows for variants of your method):

     | Method | Material | $\delta E$ at full data (eV/atom) | $\delta F$ at full data (eV/Å) | Data efficiency vs. baseline | Inference cost (rel.) |
     |---|---|---|---|---|---|
     | Quadratic SNAP (paper as reported) | Ta | 0.00898 (Table SII) | — | — | — |
     | Quadratic SNAP (reimplemented) | Ta | [agent's run] | [agent's run] | 1× (reference) | 1× (reference) |
     | Quadratic SNAP (reimplemented) | [Mo/Nb/Re] | [agent's run] | [agent's run] | 1× (reference) | 1× (reference) |
     | Learned descriptor | Ta | [agent's run] | [agent's run] | [×reduction] | [×] |
     | Learned descriptor | [Mo/Nb/Re] | [agent's run] | [agent's run] | [×reduction] | [×] |

     Numbers in `[brackets]` are placeholders; report actual values. Add rows for variants and ablations as needed.

   - **Per-group data-efficiency curves** (energy MAE vs. fraction included, mean ± std, with quadratic SNAP and learned descriptor overlaid) for all 12 training groups, plus an equal-weight all-groups aggregate curve. State the ≤50%-data outcome (the fraction F* at which the descriptor reaches quadratic SNAP's full-data accuracy) **separately for the equal-weight aggregate and for the low-symmetry subset (GSF, Liquid, Surface)**; the subset is the discriminating bar for the data-efficiency claim.
   - **Per-material data-efficiency curves** for the transfer materials.
   - **Invariance audit results**: drift magnitudes per test type, per material, with comparison to quadratic SNAP's machine-precision drift.
   - **A mechanism discussion**: why does the learned descriptor achieve the observed data efficiency (or fail to)? What does the invariance audit reveal about the architecture? Ablations isolating specific design choices are encouraged.
   - **A failure analysis**: where does the learned descriptor underperform quadratic SNAP, and why? Failure cases should be tied to specific structural properties (training-group character, system size, configuration type, invariance-drift magnitudes).
   - **Computational-cost analysis**: per-step wall-clock, descriptor evaluation cost, training cost, with the cost-quality tradeoff stated explicitly.
   - **Engagement with related work**: position your descriptor relative to the published learned-descriptor potential literature (including post-2018 developments). Cite the families of approaches your method is closest to and explicitly differentiate your contribution.
   - **Limitations and conclusions**.

   Appendix material is for supplementary tables and figures that did not fit in the main body. The substantive contract — methodology, results table, headline curves, mechanism, failure, cost — must live in the main paper, not the appendix.

3. **Supporting documents.**
   - `proposal/codebase_notes.md` — a record of what you found and reused in FitSNAP, OMatG, DAKOTA, and any other provided codebase.
   - `proposal/attempts_log.md` — distinct approaches tried during the work, including ones set aside. Required whenever the submission describes any work as partial or unresolved.

### Paper quality

Your submission is an academic paper, not a class project report. The writing should resemble what would be reviewed at a top scientific journal in structure, integrity, depth, and language: a clear thesis stated up front, claims supported by quantitative evidence, prose rather than bullet enumeration in main body sections, careful citation, and explicit honest discussion of limitations.

**Independence from this task spec.** Your paper should read as a self-contained journal contribution to a reader who has not seen this task spec. Do not transliterate spec terminology, do not adopt the spec's section structure verbatim, do not use placeholder names like "the proposed method" or "your method" when the contribution has a name. The section content described under "Deliverables" specifies what the paper must *contain*, not the titles or order in which the content must appear. Organize the paper however best serves the contribution.

**Rendered output verification.** Before submitting, render your LaTeX to PDF (e.g., `pdflatex report.tex`) and visually inspect the resulting PDF. Equations must not overflow page margins; tables must fit within the text width without column wrapping that breaks readability; figures must not be cut off. A submission with formatting issues in the rendered PDF will be flagged as not having been visually verified.

Good luck.
