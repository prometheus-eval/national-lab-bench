# Task: Develop a Substantive Follow-up to "Beyond Classical Molecular Dynamics with Layered Interatomic Potentials"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the layered Sommerfeld interatomic-potential framework by Wood, Koknat, and Thompson, "Beyond classical molecular dynamics with layered interatomic potentials," 2026. Your contribution is to **design a principled switching function for the two-state Sommerfeld potential** — replacing the paper's physically-motivated-but-unoptimized sigmoid — and show that it improves on the paper's choice in (1) accuracy of the blended potential against DFT at the held-out intermediate electronic temperature ($T_e = 0.1$ eV) and (2) molecular-dynamics stability, while preserving the paper's physical effect. The direction was selected by the paper's author (Mitchell A. Wood) in elicitation, who noted the switching function and its free parameters were chosen physically but "this was not done rigorously."

## The Problem — Here's why it matters

Classical molecular dynamics (MD) uses Born–Oppenheimer interatomic potentials that omit electronic degrees of freedom. This breaks down precisely where modern materials science needs accuracy: plasma-facing materials in fusion reactors, where a 14.1 MeV D+T fusion neutron transfers tens of keV to a primary knock-on atom (PKA), creating a thermal spike that drives the local electronic state far from the ground state. The energy that excited electrons drain from the cascade is missing from a ground-state potential.

The layered Sommerfeld framework captures this at classical-MD cost. Two machine-learned interatomic potentials are trained at different electronic-smearing temperatures — a ground-state $\mathcal{U}_{gs}$ ($T_e \approx 0$) and a high-temperature $\mathcal{U}_{ht}$ ($T_e > 0$) — and blended through an auxiliary variable $\lambda$: $\mathcal{U}_{\text{tot}} = \lambda\,\mathcal{U}_{ht} + (1-\lambda)\,\mathcal{U}_{gs}$. The auxiliary variable is the lost quantum degree of freedom, mapped to a physical driver. For radiation damage the driver is the local atomic temperature $T_i$, and $\lambda$ switches each atom between the two potentials.

The **switching function** $\lambda(T_i)$ is the load-bearing choice: it decides how much excited-state physics each atom feels. The paper adopts a sigmoid, $\lambda_i = 1/(1 + e^{-(T_i - T^*)/\sigma})$, with a switching temperature $T^*$ and width $\sigma = T^*/14$, chosen for derivative continuity and so equilibrium fluctuations do not spontaneously toggle atoms. But the paper is explicit that this form and its parameters were not rigorously optimized, and that aggressive switching (low $T^*$, long residency on $\mathcal{U}_{ht}$) destabilizes the dynamics at high recoil energy. The paper's own Supplemental Fig. 18 makes the accuracy gap concrete: at an intermediate $T_e = 0.1$ eV, the fixed blend fraction that minimizes the DFT *energy* error ($\lambda \approx 0.2$) coincides with the naive linear-in-$T_e$ value, but the fraction that minimizes the *force* error ($\lambda \approx 0.35$) does not — the potential-energy surface does not distort monotonically with $T_e$, so a single sigmoid mapping cannot be simultaneously energy- and force-accurate at the intermediate electronic state. A switching function chosen by a stated principle — one that makes the blended potential reproduce the true intermediate electronic state and keeps the dynamics stable — would put the framework on firmer footing.

## Background — Here's what you need to know about the original paper

The framework and its radiation-damage demonstration:

1. **The two potentials.** Linear ACE machine-learned potentials fit with FitSNAP: $\mathcal{U}_{gs}$ trained on DFT at $T_e = 0.001$ eV and $\mathcal{U}_{ht}$ at $T_e = 0.5$ eV (VASP, PBE, PAW W_sv_GW, 700 eV cutoff, with extra bands for elevated smearing; 9,200 structures per smearing temperature). Material-property objective functions (elastic constants, defect-formation energies, phase stability) are optimized with DAKOTA.

2. **The layered potential.** $\mathcal{U}_{\text{tot}} = \lambda\,\mathcal{U}_{ht} + (1-\lambda)\,\mathcal{U}_{gs}$, with $\lambda$ a per-atom auxiliary variable. The blend is applied to the **forces** through the LAMMPS `pair_style hybrid/scaled` mechanism (extended to per-atom, atom-style scale factors): $-\mathbf{F}_i = \lambda_i \nabla_i \mathcal{U}_{ht} + (1 - \lambda_i)\nabla_i \mathcal{U}_{gs}$. Because $\lambda$ scales forces rather than potentials, the combined force is not strictly conservative.

3. **The switching function.** For radiation damage, $\lambda_i = 1/(1 + e^{-(T_i - T^*)/\sigma})$ depends on the local atomic temperature $T_i$ (center-of-mass-velocity corrected), with $\sigma = T^*/14$. $T^*$ is treated parametrically and is **not** equal to the smearing temperature $T_e$. The paper also notes $\lambda$ can be mapped to other drivers (e.g. radiative decay $\lambda = e^{-(t-t_0)/\tau}$).

4. **Cascade protocol.** Single-crystal W; PKA recoil energies $E_{\text{PKA}} \in [1, 200]$ keV; 25 random-orientation repetitions per recoil energy; cube faces thermostatted at 1000 K; adaptive time stepping ($10^{-6}$ to $5\times10^{-1}$ fs) through the thermal spike, then 30 ps at 1000 K; surviving Frenkel pairs counted by OVITO Wigner–Seitz; peak thermal-spike volume from a surface mesh. Marinica EAM W is the classical comparison.

5. **Diagnostics.** Instantaneous excitation $\eta(t) = \sum_i \lambda_i(t)$ and its time integral $\chi$ (atoms·time) quantify how much the run samples $\mathcal{U}_{ht}$.

**Reference behavior reported in the paper (what your work must be consistent with):**

- Using $\mathcal{U}_{ht}$ **decreases** surviving Frenkel pairs and peak thermal-spike volume relative to ground-state-only dynamics across most of the recoil spectrum — the paper's headline physical effect — with a documented exception at the highest recoil energies under the most aggressive switching.
- The sigmoid switching becomes **unstable at high recoil energy for aggressive (low-$T^*$) switching**: for each $T^*$ the recoil series is terminated when $>25\%$ of PKA repetitions show an increasing thermal-spike volume after adaptive time stepping (e.g. the $T^* = 0.5$ eV series stops at 15 keV, while $T^* = 5.0$ eV reaches 200 keV). The instability traces to long residency on $\mathcal{U}_{ht}$ and the non-conservative force blend.
- Diatom binding curves and phase-formation energies vary smoothly with a global $\lambda$ between the two potentials.

## Contract — Here's exactly what counts as success

Your contribution is a **principled switching function** for the two-state Sommerfeld potential, with a stated selection criterion, that improves on the paper's sigmoid on the accuracy and stability axes while preserving the physical effect. The selection criterion is the headline idea; the empirical comparison is the witness that it works.

### Operationalized success thresholds

You commit up front, in `proposal/code/method_plan.*`, to the switching function's functional form and the **principle** that selects it (and its parameters) — replacing the paper's ad hoc $T^*$ and $\sigma = T^*/14$. A submission meets the core requirement when **all** of the following hold:

- **The switching function is principled and precise.** Its functional form and the criterion that fixes its parameters are stated precisely (for example, the $\lambda(T_e)$ that minimizes the blended potential's DFT error at the intermediate electronic temperature, or a free-energy / Sommerfeld-derived form), with the driver (local atomic temperature, or a derived electronic-temperature estimate) defined. A re-tuned sigmoid with hand-picked $T^*$ is not by itself a principled contribution.
- **Accuracy at the intermediate $T_e$.** Against the provided intermediate-$T_e$ DFT reference at $T_e = 0.1$ eV (`data/dft_reference/`; the held-out temperature between the two training endpoints, with the `NotConverged/` configurations excluded), the blended potential under your switching function reaches **strictly lower integrated energy and force error at the intermediate temperature** than the paper's sigmoid mapping. Report the calibration of $\lambda$ versus $T_e$ — anchored by the three provided temperatures ($\lambda \to 0$ at $T_e = 0.001$ eV, the DFT-optimal blend at $T_e = 0.1$ eV, $\lambda \to 1$ at $T_e = 0.5$ eV) — and show your switching function tracks it. The concrete target is the Supplemental Fig. 18 result: at $T_e = 0.1$ eV the energy-optimal blend fraction is $\lambda \approx 0.2$ while the force-optimal is $\lambda \approx 0.35$, so a single fixed blend cannot be simultaneously energy- and force-accurate.
- **MD stability.** Your switching function **extends the stable recoil-energy range** relative to the paper's sigmoid at matched physical aggressiveness (matched mean residency on $\mathcal{U}_{ht}$ / matched $\chi$), using the paper's own stability criterion (the recoil series terminates when $>25\%$ of PKA repetitions show an increasing thermal-spike volume after adaptive time stepping). The instability mechanism (long $\mathcal{U}_{ht}$ residency, non-conservative blend) must be characterized, not just observed.
- **Physical effect preserved.** Across the bulk of the recoil spectrum the blended dynamics still yield **reduced surviving Frenkel pairs and peak thermal-spike volume relative to ground-state-only** dynamics — the paper's signature — evaluated on the mean-with-spread and tolerating the same high-recoil-energy edge case the paper documents. A switching function that wins stability by collapsing toward $\lambda \approx 0$ (i.e. recovering ground-state-only dynamics) does not satisfy the contract.
- **Well-behaved.** The switching function is derivative-continuous, recovers the correct limits ($\lambda \to 0$ in cold equilibrium, $\lambda \to 1$ in the hot limit), and does not spontaneously toggle atoms under equilibrium thermal fluctuations.

Partial credit (does not by itself clear the bar): a principled switching function demonstrated to improve accuracy but with no stability or physical-effect evidence; or an improvement shown only at a single recoil energy, or only in energy error while ignoring force error (or vice versa), rather than across the declared recoil-energy range and both error channels.

### Held fixed (parity constraints)

These must remain identical to the original study — they are not part of the design space.

- **The layered framework.** $\mathcal{U}_{\text{tot}} = \lambda\,\mathcal{U}_{ht} + (1-\lambda)\,\mathcal{U}_{gs}$, with the per-atom $\lambda$ applied to the forces via the LAMMPS `pair_style hybrid/scaled` per-atom mechanism. You design $\lambda$, not the layering algebra.
- **The two trained potentials.** $\mathcal{U}_{gs}$ ($T_e = 0.001$ eV) and $\mathcal{U}_{ht}$ ($T_e = 0.5$ eV) are used as provided. You do not retrain them, change their ACE basis, or substitute different potentials — the contribution is the switching between them, not the potentials. The $T_e = 0.001$ eV and $T_e = 0.5$ eV DFT sets in `data/dft_reference/` are the data these potentials were fit to; they are provided as accuracy-axis reference and calibration anchors only, not as an invitation to refit or replace the potentials.
- **The driver.** The local atomic temperature $T_i$ (center-of-mass-velocity corrected) as defined in the paper remains the physical driver for the radiation-damage demonstration. You may derive an additional electronic-temperature estimate, but the comparison to the paper's sigmoid is made on the same $T_i$ definition.
- **The cascade protocol.** Single-crystal W, the recoil-energy sweep, 25 repetitions per energy, face thermostats at 1000 K, adaptive time stepping, 30 ps anneal, OVITO Wigner–Seitz Frenkel-pair counting, surface-mesh thermal-spike volume, the paper's stability-termination rule (a recoil series ends when $>25\%$ of PKA repetitions show an increasing thermal-spike volume after adaptive time stepping), and the Marinica EAM W comparison.
- **The DFT reference settings** (VASP, PBE, PAW W_sv_GW, 700 eV, k-mesh, extra bands for smearing) for any accuracy comparison.

If a parity item must change for a specific experiment, state the change and the isolating comparison; do not change it silently.

### What you may design

- **The functional form of $\lambda$** — a sigmoid is not required; you may use any precisely defined form motivated by your criterion.
- **The selection principle** — how the switching function and its parameters are fixed (DFT-error-minimizing calibration against the intermediate-$T_e$ reference, a free-energy / Sommerfeld-derived form, or another principle you motivate).
- **The driver mapping** — the local atomic temperature, or a derived electronic-temperature estimate from it. Switching functions that use the recoil energy (or other global variables) as dependent variable are permissible, but a continuous function rather than a piecewise dependency is required. Modifications to LAMMPS to add per-atom properties used in the switching function are also permissible.
- **The $\lambda$-versus-$T_e$ calibration** (the blend-fraction sweep at the intermediate reference and which configurations you evaluate) **and the recoil-energy / stability sweep**, declared up front.

### Anti-cheating

The contract is about a principled switching function and its honest comparison to the paper's sigmoid. The following moves are not allowed:

- **Retraining or substituting the potentials.** $\mathcal{U}_{gs}$ and $\mathcal{U}_{ht}$ are used as provided; improving the blend by changing the potentials is out of scope and would confound the comparison.
- **Stability by $\lambda$-collapse.** Achieving stability by driving $\lambda \to 0$ (recovering ground-state-only dynamics) defeats the purpose; the physical-effect axis must still hold.
- **Sandbagging the baseline.** The paper's sigmoid baseline must use $\sigma = T^*/14$ and the paper's $T^*$ values fairly; you may not weaken it to manufacture an advantage.
- **Overfitting to one point.** The accuracy calibration must span the declared $\lambda$ blend-fraction sweep at the held-out $T_e = 0.1$ eV reference and report **both** the energy and force error, not a single favorable blend fraction or one error channel; the stability comparison must span the declared recoil-energy range, not a single favorable recoil energy.
- **Criterion switching after the fact.** Commit the switching-function form and selection principle in `method_plan.*` before runs; relabeling the criterion after seeing results is not allowed.

## Evaluation — Here's how you'll be judged

You will be judged on (1) performance on the stated thresholds — whether the switching function is principled, more accurate against DFT across intermediate $T_e$, more stable across recoil energy, and preserves the physical effect — and (2) scientific merit — the soundness of the selection principle, the depth of the instability-mechanism analysis, engagement with the layered-potential / electron–phonon-coupling / two-temperature-model literature, and an honest failure analysis. Scientific merit values ambition and depth without prescribing what counts as ambitious.

### Two-baseline reporting (required)

Every comparison must report two references side by side:

1. **Paper-as-reported sigmoid** — the paper's Frenkel-pair, thermal-spike-volume, $\chi$, and stability-range results versus $T^*$, cited from specific figures/tables.
2. **Reproduced sigmoid** — your own runs of the paper's sigmoid switching function at the paper's $T^*$ values and protocol, the paired baseline against which your switching function is compared.

If your reproduced sigmoid diverges from the paper's, document the gap in the experimental setup and address whether your conclusion holds in the paper's regime.

### Evaluation protocol

- Declare in `proposal/code/method_plan.*`, before runs: the switching-function form, the selection principle, the $\lambda$ calibration (blend-fraction sweep at the intermediate-$T_e$ reference), the recoil-energy / stability sweep, and the baseline $T^*$ values.
- Report the accuracy calibration (blended-potential DFT error versus $T_e$) for your switching function and the sigmoid; the stable-recoil-range comparison at matched aggressiveness; and Frenkel-pair and thermal-spike-volume curves versus ground-state-only across the recoil spectrum (mean over the 25 repetitions).

## Rules — Here are the rules

### Available resources

All necessary code and the original paper are provided locally.

- `paper.md`, `paper.pdf` — the original layered-potential paper (Wood, Koknat, Thompson 2026), including the framework (Section IV), the switching function and cascade results (Section II), and the training/DFT protocol.
- `images/` — figures extracted from the paper.
- `code/FitSNAP/` — the MLIAP fitting toolkit (linear ACE/SNAP) used to build the potentials.
- `code/lammps-user-pace/` — LAMMPS with the PACE (ACE) pair style and the per-atom `hybrid/scaled` mechanism for the layered MD and cascades.
- `code/dakota/` — the DAKOTA optimization toolkit, used in the paper for parameter optimization; available for your switching-function parameter selection.
- `data/marinica_eam_w/` — the Marinica EAM W potential, the paper's classical comparison.
- `data/ace_potentials/` — the two author-provided trained ACE potentials, used as provided: `W_pot_T0.0.yace` (the ground-state $\mathcal{U}_{gs}$, $T_e = 0.001$ eV) and `W_pot_T0.5.yace` (the high-temperature $\mathcal{U}_{ht}$, $T_e = 0.5$ eV), each with a `.mod` `pair_style` include, plus `in.ACE_Sommerfeld_PKA` — an example LAMMPS input showing the `pair_style hybrid/scaled` two-state Sommerfeld syntax with the per-atom blend fractions (`v_psratio1` for $\mathcal{U}_{ht}$, `v_psratio2` for $\mathcal{U}_{gs}$).
- `data/dft_reference/` — **the author-provided DFT reference sets** (VASP energies and forces) at three electronic temperatures, shipped as tarballs of FitSNAP-readable per-configuration JSON (extract before use): `W-VASP_T0.001eV_JSON.tar`, `W-VASP_T0.1eV_JSON.tar`, `W-VASP_T0.5eV_JSON.tar`. The **$T_e = 0.1$ eV set is the held-out intermediate accuracy reference** — the Supplemental Fig. 18 set, deliberately smaller ($\sim$2400 configs) because the author paused those jobs early. The $T_e = 0.001$ eV and $T_e = 0.5$ eV sets are the DFT the two provided potentials were fit to: endpoint anchors for the $\lambda$-versus-$T_e$ calibration, not data for refitting. **Exclude every configuration under `NotConverged/` in each set** (unconverged DFT with unreliable energies/forces). No DFT engine is provided and no new DFT runs are permitted; the accuracy comparison uses these provided sets only. See `data/dft_reference/README.md` for the manifest, per-set counts, and JSON format.
- `proposal/` — your output: `proposal/code/`, `proposal/logs/`, `proposal/codebase_notes.md`, `proposal/attempts_log.md`, the paper at `proposal/report.tex` and rendered `proposal/report.pdf`. Use the NeurIPS LaTeX style file `neurips.sty` provided in the workspace.

### Compute budget

- **Hardware.** One CPU-bound node: 32 vCPU cores, ~60 GB RAM (no swap), ~470 GB disk; no GPU is required. Treat ~50 GB as a hard memory ceiling - size supercells, cascade batch sizes, and parallel workers to keep peak resident memory under it (a process that exhausts RAM is killed by the OS). The neutron-damage cascades (large W supercells, 25 repetitions per recoil energy) and the parameter optimization dominate the cost.
- **Scope priority.** Priority 1: commit the switching-function form and principle, calibrate $\lambda$ against the intermediate-$T_e$ reference, and run the stability + physical-effect comparison against the reproduced sigmoid on the cascade protocol. Priority 2: the mechanism analysis, failure analysis, and any generalization. Keep the recoil-energy sweep and repetition count large enough that the stability and Frenkel-pair comparisons are statistically meaningful; the contribution is the switching function, not large-scale runs.

### On wrapping up near the time limit

If you find yourself near the time limit with incomplete work, stop launching new cascades and use the remaining time to write up what you have. A switching function validated on a partial recoil sweep is incomplete, not fabricated — report it that way. Populate `proposal/attempts_log.md` with what you tried, what did not work, and why you abandoned approaches. Effort evidence is mandatory whenever partial work is claimed.

### Autonomy and external resources

**Internet search is permitted and encouraged.** Use it for literature lookup, library documentation, and citations — in particular the layered/two-state interatomic-potential, electron–phonon-coupling, two-temperature-model, and radiation-damage-MD literature. Read existing follow-up work to the paper for context and to differentiate your contribution. Cite anything you use.

**Workspace constraint.** You may not read local files outside the provided working directory.

**No questions.** Make your own decisions and justify them in your write-up.

**No replication of existing work.** Independent researchers will naturally converge on some shared choices — that is fine and expected. What is not allowed is substantive replication of an existing follow-up or of a published switching-function / two-temperature method presented as your own. If the evaluator finds your submission is substantively a reproduction of an existing method's contribution, your submission will be severely degraded. The contract evaluates your scientific work, not your literature search.

### No reward hacking

The following checks are applied to your submission. They are disclosed up front so the contract is clear.

1. **Number traceability.** Every quantitative value in the paper — accuracy errors, Frenkel-pair counts, thermal-spike volumes, $\chi$/$\eta$ values, stable-recoil-range thresholds — traces to a specific logged entry with the same value.
2. **Pre-stated method.** The switching-function form, the selection principle, the $\lambda$ calibration, and the recoil/stability sweep are recorded in `proposal/code/method_plan.*` with a timestamp predating the earliest run, and the method described in the paper matches it (refinements after runs allowed only if disclosed).
3. **Test-set integrity.** The $\lambda$ calibration (blend-fraction sweep) and the recoil-energy sweep are declared before runs; the accuracy comparison is made at the held-out intermediate $T_e = 0.1$ eV (the potentials were fit at 0.001 and 0.5 eV, so the intermediate is genuinely held out), excludes the `NotConverged/` configurations, and uses no post-hoc adjustment.
4. **Baseline honesty.** The sigmoid baseline numbers come from your own logged runs (at $\sigma = T^*/14$ and the paper's $T^*$ values), or are explicitly cited from the paper.
5. **No selective reporting.** All recoil energies, blend fractions in the $\lambda$ sweep, and repetitions run are reported (mean and spread), not just the favorable ones.
6. **No hardcoded values.** The code contains no hardcoded DFT energies/forces, defect counts, or stability thresholds used to bypass evaluation.
7. **Experiment completeness.** Every experiment described in the paper has corresponding log entries.
8. **Code–paper alignment.** The Methodology accurately describes what the LAMMPS / FitSNAP code does, and the switching function implemented in the code matches the one stated in the paper.
9. **Switching-function integrity.** The layering algebra ($\mathcal{U}_{\text{tot}} = \lambda\,\mathcal{U}_{ht} + (1-\lambda)\,\mathcal{U}_{gs}$, blend on forces) is unchanged and $\mathcal{U}_{gs}$/$\mathcal{U}_{ht}$ are used as provided (not retrained); and the stability improvement is not achieved by collapsing $\lambda \to 0$ (which the preserved-physical-effect evidence must rule out).
10. **MD execution integrity.** The cascades are actually run in LAMMPS (logged inputs, trajectories, and OVITO Wigner–Seitz outputs), not analytical estimates substituted as simulation results; Frenkel-pair counts and thermal-spike volumes come from the actual analysis pipeline.
11. **Codebase inspection evidence.** `codebase_notes.md` documents what was inspected and reused in FitSNAP, LAMMPS/PACE, and DAKOTA versus implemented.
12. **No replication of published methods.** The switching function and selection principle are distinguishable from substantive replications of published switching-function / two-temperature methods; convergent prior results are acknowledged with citation.
13. **Effort evidence.** If the paper claims partial/failed/unresolved work, `attempts_log.md` documents the approaches tried and why each was set aside.
14. **Held-fixed disclosure.** Any deviation from the held-fixed items (framework, potentials, driver, cascade protocol, DFT settings) is transparently disclosed.
15. **No external code.** All non-trivial methodology code is authored during the run, taken from FitSNAP/LAMMPS/DAKOTA, or from standard scientific libraries, with anything external disclosed.
16. **No criterion switching.** The committed selection principle is declared in `method_plan.*` before runs and matches the paper; you do not silently switch principles after seeing results.
17. **Multiple switching-function discipline.** If several candidate switching functions were tried, the reported one is pre-registered or its post-hoc selection is disclosed in the paper or `attempts_log.md`.

## Deliverables — Here's what to hand in

1. **The codebase** — under `proposal/code/`, including your switching-function implementation (the LAMMPS per-atom $\lambda$ definition), the calibration code, the cascade-run harness, and the analysis pipeline.
2. **Logs** — under `proposal/logs/`, with the accuracy calibration (blended-potential DFT error versus $T_e$) for your switching function and the sigmoid, the stable-recoil-range data, and per-recoil-energy Frenkel-pair / thermal-spike-volume / $\chi$ records for method, sigmoid baseline, ground-state-only, and EAM.
3. **The selection principle** — the derivation or calibration that fixes the switching function, inline in the paper and/or as a reproducible script under `proposal/code/derivation/`.
4. **`proposal/code/method_plan.*`** — the pre-run declaration of the switching-function form, principle, $\lambda$ calibration, recoil/stability sweep, and baseline $T^*$ values.
5. **The paper** — `proposal/report.tex` and a cleanly rendered `proposal/report.pdf` in NeurIPS style.
6. **`proposal/codebase_notes.md`** and **`proposal/attempts_log.md`**.

The main paper carries the substantive contribution — the switching function, its selection principle, the accuracy and stability comparisons, the preserved-physical-effect evidence, and the mechanism, failure, and literature analyses, all at journal standard. The appendix is a fallback for work attempted but not brought to journal quality, with corresponding `attempts_log.md` entries; the expected appendix is empty or near-empty, achieved by doing the work rather than by relegating contract content.

### Paper quality

Do not frame the paper as "a follow-up to Wood et al." Frame it as a self-contained scientific contribution with its own motivation, methods, results, and discussion, citing the original paper where appropriate. The title should describe the scientific contribution (the switching function and what it achieves) rather than the structure of the investigation. It must read as a journal contribution: a clear thesis stated up front, the switching function and its selection principle, the accuracy-versus-$T_e$ and stability-range comparisons with a main results table, a mechanism subsection, a failure subsection, an honest limitations discussion, and careful citation. Required figures and tables are specified by content, not title: the accuracy-versus-$T_e$ comparison (your switching function vs the sigmoid vs DFT), the stable-recoil-range comparison, and the Frenkel-pair / thermal-spike-volume curves versus ground-state-only.

**Independence from this task spec.** A reader who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. Do not reuse spec terminology, do not transliterate the spec's section structure, and do not use placeholder names like "the switching function" where a real name belongs. The content described under Deliverables specifies what the paper must contain, not the titles or order in which it must appear.

**Rendered output verification.** Before submitting, render your LaTeX to PDF and visually inspect the resulting PDF. Equations must not overflow page margins; tables must fit within the text width without column wrapping that breaks readability; figures must not be cut off. A submission with formatting issues in the rendered PDF will be flagged as not having been visually verified.

Good luck.
