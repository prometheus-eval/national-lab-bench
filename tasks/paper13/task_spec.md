# Task: Develop a Substantive Follow-up to "Particle-in-Cell Simulations of Laser Crossbeam Energy Transfer via Magnetized Ion-Acoustic Wave"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the magnetized cross-beam energy transfer (MagCBET) PIC study by Shi and Moody, "Particle-in-cell simulations of laser crossbeam energy transfer via magnetized ion-acoustic wave," 2025. Your contribution is to **characterize the parameter regimes where the Vlasov theory of MagCBET remains predictive — and where it breaks** — by mapping the boundary between agreement and disagreement in a stated subspace of $(|\mathbf{B}|, \phi_B, n_e, T_e, I_{\text{laser}}, \text{polarization})$. The work uses the paper's provided processed PIC data plus a small set of new targeted small-scale PIC simulations at sweep points the paper does not cover. The headline output is a **boundary map** with statistical support per cell and a **mechanism attribution** identifying which Vlasov-theory assumption fails first as each boundary is crossed.

## The Problem — Here's why it matters

In laser-driven inertial confinement fusion (ICF), **cross-beam energy transfer (CBET)** is the resonant transfer of energy between two crossing laser beams mediated by a low-frequency plasma wave (typically the ion-acoustic wave, IAW). It is exploited for symmetry control in indirect-drive implosions and avoided in direct drive, where it can drain incoming laser energy to backscattered light. Modern ICF experiments are increasingly **magnetized** — by external coils or by self-generated fields — reaching MG-to-tens-of-MG fields in regions where the electron cyclotron frequency $\omega_{ce}$ becomes comparable to typical IAW frequencies, EPW frequencies, or even the laser frequency itself. Magnetization modifies CBET in ways the standard unmagnetized theory does not capture.

A linearized **Vlasov theory of magnetized CBET (MagCBET)** is developed in the companion paper [Moody & Shi, "Vlasov theory of magnetized cross-beam energy transfer in a high-energy-density plasma," Phys. Plasmas 33, 062712 (2026), DOI 10.1063/5.0322470, ref. 74]. That paper is published and may be consulted via the permitted literature search; it is cited here, not bundled into the workspace. The paper at hand validates the Vlasov predictions qualitatively using 2D-3V PIC simulations (EPOCH code) at a tuned regime kept CBET-dominant and linear, but the authors explicitly note that **quantitative differences remain, suggesting additional physics beyond the Vlasov theory**.

The open practical question this follow-up addresses: **in which experimental regimes can ICF practitioners trust the Vlasov MagCBET predictions, and in which must they fall back to full kinetic simulations?** The original paper documents specific discrepancies but does not systematically map *where* the theory holds and *where* it breaks. That boundary map is the contribution. The Vlasov theory is fast and amenable to analytical insight; PIC is slow and resource-intensive. Knowing where each is appropriate is directly actionable for ICF design.

This is a **characterization task**: the contribution is the boundary map and the mechanism story behind it, not a theoretical extension that closes the discrepancy. (Direction 2-A — extending the theory — is more ambitious but would require producing a theory better than the published companion paper [74] has already achieved; it is out of scope here. Direction 2-C — extending the analysis to other parametric instabilities like Raman or two-plasmon decay — is also out of scope.)

## Background — Here's what you need to know about the original paper

The original paper provides PIC validation of MagCBET predictions and identifies specific quantitative-discrepancy regimes. Your follow-up uses the same PIC validation framework, the same processed PIC data, and small new PIC runs at sweep points the paper does not cover. The technical components of the original method are:

1. **Simulation setup.** EPOCH 2D-3V PIC simulations on an elongated $500\lambda_0 \times 10\lambda_0$ domain centered around the seed laser, with absorbing CPML boundaries. Pump $\hat{\mathbf{k}}_0 = \cos\alpha\,\hat{\mathbf{x}} - \sin\alpha\,\hat{\mathbf{y}}$ with crossing angle $\alpha = 30°$. Seed propagates along $+\hat{\mathbf{x}}$. Two polarization setups:
   - **z–z (parallel polarization):** both lasers initially $\hat{\mathbf{z}}$-polarized.
   - **z–y (orthogonal polarization):** pump $\hat{\mathbf{z}}$, seed $\hat{\mathbf{y}}$.

2. **Plasma conditions.**
   - **Carbon plasma (main runs):** $n_e = 3.5 \times 10^{20}$ cm⁻³, $T_e = 4T_i = 0.8$ keV.
   - **Hydrogen plasma (additional):** $n_e = 3 \times 10^{20}$ cm⁻³, $T_e = 4T_i \in \{0.4, 1.2\}$ keV, crossing angle $\alpha = 24°$, magnetic field at $\theta_B = 45°$, $\phi_B = 15°$.

3. **Magnetic field configurations (carbon plasma).** $\theta_B = 90°$ (B in the laser crossing plane), with three azimuthal angles:
   - **(i) $\phi_B = 75°$:** $\Delta\mathbf{k} = \mathbf{k}_0 - \mathbf{k}$ nearly parallel to $\mathbf{B}$. Plasma wave essentially unmagnetized; magnetization affects lasers only.
   - **(ii) $\phi_B = -13°$:** $\Delta\mathbf{k}$ nearly perpendicular to $\mathbf{B}$. Plasma wave is magnetized.
   - **(iii) $\phi_B = 30°$:** intermediate angle.
   Field strengths swept from 0 to ~50 MG.

4. **Numerical resolution.** 20 cells per pump wavelength, 20 particles per cell, $\lambda_0 = 0.351\,\mu$m, pump and seed at $I_p = I_s = 0.5 \times 10^{15}$ W/cm² to keep pump depletion small and avoid PIC-noise-seeded instabilities.

5. **Data analysis.** A Gaussian-windowed Fourier bandpass filter isolates seed-laser peaks at $k_x = \pm k_0$ in 2D Fourier space. The two electromagnetic eigenmodes (X and O modes for the magnetized plasma) are extracted by linear combinations of the filtered $\hat{E}_y$ and $\hat{E}_z$, after a Faraday-rotation phase correction $\hat{E}_z(x) \to \hat{E}_z(x - \Delta x)$ with $\hat{k}\Delta x = \psi$. Energy gain $g_j(t) = \ln[U_j(t)/U_j(t_0)]$ for $j \in \{1, 2, \text{total}\}$ with $U_j = (\epsilon_0/2) \int dx\,\hat{E}_j^2$. Quasi-steady state at $t_1 = 3$ ps and $t_2 = 3.6$ ps; uncertainty estimated from the spread between $t_1$ and $t_2$ and from the integration fraction $f_y \in \{1, 0.75, 0.5\}$ along the transverse direction.

6. **Finite-size broadening correction** [Eq. (10) of the paper]: a finite-$L_x$ plasma supports a discrete set of wavevectors $\Delta k = 2\pi/L_x$, and a laser entering with vacuum wavevector $k$ generally couples to two adjacent modes $k_\pm$.

**Reference baseline behavior reported in the paper:**

- **Qualitative agreements PIC reproduces against Vlasov theory:**
  - z-z gain decreases monotonically with $|\mathbf{B}|$ at all $\phi_B$ tested.
  - z-y gain rises from approximately zero at $B=0$ to substantial positive values when $\phi_B = 75°$; remains close to zero in the perpendicular case ($\phi_B = -13°$).
  - The gain curve $g(\Delta\lambda)$ shifts and broadens at high $|\mathbf{B}|$ in the perpendicular case.
  - X-mode and O-mode gains are non-degenerate at $|\mathbf{B}| \gtrsim 10$ MG (carbon, parallel case).

- **Documented quantitative discrepancies (open question for this follow-up):**
  - **Profile non-exponentiality:** seed envelopes show noticeable departures from exponential, partly attributed to competing PIC-noise-seeded instabilities and Fourier-filtering artifacts.
  - **Background offset $g_0 \approx -0.4$:** even at $B=0$ z-y (where Vlasov predicts zero gain), PIC measures a negative offset due to side- and back-scattering of the seed laser; varies with $B$.
  - **Slow distribution-function drift** breaks the steady-state assumption underlying the Vlasov derivation.
  - **Pump depletion** is "insignificant" but not exactly zero, especially at large $\Delta\lambda$ where gains exceed unity.
  - **Sheath field $E_S \sim 10^{11}$ V/m** comparable to the laser field; plasma expansion non-negligible.
  - **5% pump reflection** from the CPML; finite-$L_y$ side losses; carbon-vs-hydrogen scaling differences.
  - **Crossing point of z-z and z-y gains** shifts vs. Vlasov prediction depending on plasma conditions and interaction length.

The PIC data underlying Figs. 2–3 of the paper (and a more complete set referenced as Zenodo [88]) is provided locally at `data/pic_simulations/PIC_processed_data.zip`.

## Contract — Here's exactly what counts as success

Your contribution is a **boundary map** characterizing where the Vlasov MagCBET theory remains predictive vs. where it breaks, in a stated subspace of $(|\mathbf{B}|, \phi_B, n_e, T_e, I_{\text{laser}}, \text{polarization})$. The work combines (a) re-evaluation of the Vlasov theory against the provided processed PIC data and (b) a small set of new targeted small-scale PIC simulations at parameter points the paper does not cover. The boundary's location is the headline; the mechanism attribution — which Vlasov-theory assumption fails first as each boundary is crossed — is the supporting evidence.

### Operationalized success thresholds

This is a characterization task (paper-1-K3-style): the contribution is the **map and the mechanism story**, not a single scalar threshold cleared. The contract operationalizes characterization quality through statistical support, mechanism attribution, and compactness:

- **Statistical support per boundary cell.** For each parameter cell in the sweep grid, the boundary's location is supported by at least one of: (a) the provided PIC processed data with its documented $t_1$–$t_2$ and $f_y$ uncertainty bands, or (b) new small-scale PIC runs with ≥3 different RNG seeds (or seed-determinism documented). Cells where the boundary is identified but statistical support is thin must be explicitly flagged in the paper as "illustrative" rather than "established."

- **Boundary identification rule stated a priori.** The agreement metric (e.g., $|g_{\text{Vlasov}} - g_{\text{PIC}}|/|g_{\text{PIC}}|$, or signed residual normalized by PIC error bars) and the boundary-identification threshold (e.g., "Vlasov holds where relative residual ≤ 30%") must be declared in `proposal/code/sweep_plan.{py,md,json}` **before** the validation sweep begins. The threshold is the agent's design choice; the discipline is that it is fixed before seeing results.

- **Sweep grid stated a priori, and covering at least 3 axes with ≥3 points per axis.** The agent picks which axes from $(|\mathbf{B}|, \phi_B, n_e, T_e, I_{\text{laser}}, \text{polarization})$ to sweep — but at least 3 axes must be swept, with ≥3 distinct values per axis. The sweep grid is declared in `sweep_plan.*` before runs begin.

- **At least one boundary supported by new PIC runs (not only by provided data).** The provided processed PIC data covers the paper's parameter sets. To map a boundary, the sweep must extend to at least one combination of values the paper does not cover, supported by new small-scale PIC runs the agent runs. "Small-scale" means reduced spatial resolution and/or smaller domain than the paper's full setup, with the reduction documented and validated against a reproduction of one of the paper's reference PIC results within stated tolerance. However, notice that if the resolution is too low, numerical artifacts will arise, and if the domain is too small, the PIC signal to noise ratio will drop below 1, both invalidating the PIC results. 

- **Mechanism attribution per boundary.** Each identified boundary segment must come with an attempted mechanism attribution — a statement of which assumption fails first as the boundary is crossed — and the paper must say which of two categories each attribution belongs to:

  - **Comparison-validity prerequisites.** These must hold for a PIC-vs-theory comparison to test the theory at all; a failure here means the *comparison* is compromised, not that the theory's physics has been probed. They include the quasi-steady-state assumption of the gain measurement, an undepleted / constant pump amplitude, negligible sheath fields and side/back-scattering, and adequate finite-domain mode resolution in the PIC (a domain too small or resolution too coarse degrades the PIC itself, per the small-scale-run caution above).

  - **Vlasov-theory physical assumptions.** A failure here is the scientifically meaningful boundary physics. The linearized MagCBET theory [74] rests on three stated approximations that a boundary crossing can violate. These are meant to be distinct failure modes; attribute each boundary to the one whose physics actually drives it. (i) A **warm-fluid ponderomotive drive** evaluated at a single averaged pump frequency, retaining only the electrostatic-like potential term and dropping the magnetic-field (magnetization) correction and the frequency-derivative terms. Dropping the magnetization term is justified only in the two-sided window where the electron cyclotron frequency is both far below the laser frequency and far above the beat frequency ($\Delta\omega \ll \omega_{ce} \ll \omega_{\text{laser}}$); the drive form is expected to weaken when the field is strong enough that the upper bound $\omega_{ce} \ll \omega_{\text{laser}}$ is approached, when the field is weak enough or the beat/detuning frequency large enough that the lower bound $\Delta\omega \ll \omega_{ce}$ is approached, when the two beams differ substantially in frequency (so a single averaged frequency no longer suffices), or where genuinely kinetic drive physics the fluid form omits (Landau damping, finite-Larmor-radius corrections) becomes important. (ii) A **linear, ponderomotive-free ion response** in which the ions feel only the self-consistent low-frequency field and their nonlinear ponderomotive response is dropped (only the linear magnetized ion susceptibility is kept) — expected to fail at high laser intensity where the ion quiver motion is no longer negligible, showing up as a correction to the amplitude and detailed shape of ion resonances (including the ion-Bernstein resonances) rather than as their appearance. The *emergence* of those magnetized ion modes is set by the retained linear susceptibility and belongs to (iii); (ii) is a second-order correction on top of it, so a boundary that turns on the modes themselves is an (iii) failure, while a boundary in their strength or profile at high intensity is an (ii) failure. (iii) **Single strongly-damped ion-acoustic mediation along unperturbed gyro-orbits** — the transfer treated as mediated by one broad, strongly (Landau-)damped ion-acoustic resonance, with the perturbed distribution integrated along unperturbed cyclotron orbits (gyro-motion dominating wave-trapping motion) — expected to fail as the plasma wavevector approaches perpendicular to the field, where the ion-acoustic damping rate drops and additional weakly-damped magnetized modes (ion Bernstein, lower hybrid, fast Alfvén) emerge as narrow resonances the single-mode picture does not represent, and where field-dependent particle trapping breaks the unperturbed-orbit assumption.

  These two categories are the stated menu of candidate failure modes, not an exhaustive list; you may identify others and must justify them. Mechanism hypotheses must be **stated before evaluating residuals** in the agent's sweep_plan — post-hoc identification of "the mechanism that fits" is selective. If a mechanism cannot be cleanly attributed, the boundary must be reported as "mechanism-unattributed" rather than spuriously labeled.

- **Compactness (paper-1-style).** The boundary map should be reasonably compact — a Pareto-style summary identifying which 1–2 axes carry most of the agreement-vs-disagreement signal, rather than a fully entangled high-dimensional surface. The compactness analysis is a deliverable; a non-compact boundary (every parameter combination has its own unique behavior) is a real finding too, but must be reported honestly.

- **Multiple-testing correction for boundary and mechanism claims.** Account for exploration of candidate boundary-identification thresholds, agreement metrics, and mechanism hypotheses before selecting the reported claims. When multiple hypotheses support inferential claims, control the family-wise error rate over the relevant hypothesis family using Holm, Bonferroni, or another procedure that provides family-wise error control under the applicable assumptions. State the hypothesis family, significance level, correction method, and treatment of outcome-dependent hypothesis selection. Benjamini–Hochberg FDR alone does not satisfy this requirement. Preregistration identifies the planned hypotheses but does not, by itself, waive multiplicity control. A genuinely single prespecified test does not require a multiple-testing adjustment. Exploratory effect-size summaries may be reported descriptively, but they do not substitute for multiplicity-controlled inferential evidence where such evidence is required.

The contribution is the boundary map + mechanism attribution. The discriminator between Tier 4 (substantial characterization) and Tier 3 (boundary mapped but with weaknesses) is the depth of statistical support, the cleanliness of mechanism attribution, and the compactness of the description — not a single scalar threshold.

### Held fixed (parity constraints)

These must remain identical to the paper's framework — they are not part of the design space.

- **The Vlasov-theory baseline.** Reconstructed from the original paper's documented qualitative predictions, the published companion theory [74] (which you may consult via the permitted literature search), the prior MagLPI references cited, and standard linearized-kinetic-theory derivations. State your reconstruction in Methodology so it is reproducible. Reconstructing the full magnetized dispersion is substantial; if you cannot bring the genuine magnetized baseline to completion, document the limitation and use a documented partial reconstruction (e.g., the unmagnetized linear CBET gain formula adjusted for magnetic-field-dependent eigenmode mixing using the published modifications), disclosed as such.
- **The PIC analysis pipeline.** Gain measurement, Fourier filtering, eigenmode decomposition, Faraday-rotation correction, $t_1$/$t_2$ uncertainty bands, $f_y$ integration-fraction variation — all per the paper's documented protocol. New PIC runs must use the same analysis pipeline, which is available at https://gitlab.com/seanYuanSHI/magnetized-cross-beam-energy-transfer
- **The plasma model and core parameters at reference points.** Carbon plasma at $n_e = 3.5 \times 10^{20}$ cm⁻³, $T_e = 4T_i = 0.8$ keV; hydrogen plasma at $n_e = 3 \times 10^{20}$ cm⁻³, the paper's $T_e/T_i$ values. The sweep extends these to new parameter combinations but the reference points are anchored to the paper's values for the validation that the agent's reduced-scale PIC reproduces the paper.
- **The agreement metric definition (after committing in sweep_plan).** Once stated, the metric must be applied uniformly across all sweep cells. Switching metrics for different cells to flatter the boundary is a violation.

### What you may design

- **The sweep grid.** Which 3+ axes of $(|\mathbf{B}|, \phi_B, n_e, T_e, I_{\text{laser}}, \text{polarization})$ to sweep, the values per axis, and the order in which they are evaluated.
- **The agreement metric.** Relative residual on $g(\Delta\lambda)$, absolute residual normalized by PIC error bars, eigenmode-resolved residual, or another principled metric. State and justify the choice.
- **The boundary-identification threshold.** Where on the metric the boundary lies. Pre-stated.
- **The mechanism hypotheses.** Which Vlasov-theory assumptions to test as candidate failure modes. Pre-stated.
- **The reduced-scale PIC configuration.** Smaller domain ($L_x < 500\lambda_0$), coarser resolution (cells/wavelength < 20), fewer particles per cell, etc. Document the reduction and validate against a reproduction of one paper reference run within stated tolerance. Should ensure resolution is sufficient to achieve numerical convergence, and the reduced configuration must not be drastically smaller than the paper's setup, or the runs lose validity.

### Anti-cheating

The contract is about the boundary characterization and its honest mechanism attribution. The following moves are not allowed:

- **Removing the sweep grid after seeing results**. The grid in `sweep_plan` is the coarse grid. Finner grid can be added to refine the boundary. But grid should not be removed to hide unfavorable results. Post-hoc additions or deletions must be disclosed.
- **Switching the agreement metric or threshold after results** to make the boundary look cleaner. Metric and threshold are pre-stated.
- **Cherry-picking which sweep points to include** in the boundary map. Every sweep point with logged data must appear in the boundary analysis (or be explicitly excluded with stated rationale).
- **Mechanism post-hoc identification.** If a mechanism hypothesis is invented after inspecting residuals or identifying a boundary, disclose its post-hoc origin and support it with either a separate fresh-cell validation pass or valid post-selection statistical inference, as specified in the mechanism-attribution requirement below. Applicable multiplicity control must satisfy the family-wise requirement, including when multiple confirmatory hypotheses are tested on fresh cells. Disclosure of an unsupported post-hoc hypothesis does not establish scientific support, but is not by itself an integrity violation; concealed post-hoc selection or a false precommitment claim is assessed separately.
- **Hand-picking PIC reduced-scale parameters per sweep point** in ways that flatter the boundary. The reduced-scale configuration must be stated up front and held constant across the sweep.
- **Cherry-picking RNG seeds** to keep the seeds where the residual supports the boundary and drop the others. All run seeds appear in the analysis.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated along two dimensions:

1. **Performance on stated metrics.** The boundary map is supported by the announced sweep grid; statistical support per cell is honestly reported; mechanism attribution is pre-stated where possible and corrected when post-hoc; the compactness analysis is genuine; new PIC runs are reproducible.

2. **Scientific merit.** Your paper is assessed on the depth of the boundary characterization, the rigor of the mechanism attribution, the engagement with the LPI / kinetic / Vlasov-theory literature, and whether the boundary map is meaningful for ICF practitioners. A boundary map that identifies a single clean dimension along which Vlasov fails (with a mechanism story) is more valuable than a fully entangled high-dimensional surface even if the latter is statistically more "supported."

### Paper quality

Your submission is an academic paper, not a class project report. Writing should resemble what would be reviewed at a top scientific journal in structure, integrity, depth, and language.

Do not frame the paper as "a follow-up to Shi and Moody." Frame it as a self-contained scientific contribution describing the regime characterization, the mechanism attribution, and the implications for when Vlasov MagCBET predictions can be used in practice.

**Independence from this task spec.** A reader who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. Do not use phrases like "axis (a)" / "sweep grid" / "agreement metric" verbatim where physics-natural terminology would do. Section structure should be physics-driven, not contract-driven.

**Rendered output verification.** Before submitting, render your LaTeX to PDF and visually inspect. Equations must not overflow page margins; tables must fit within text width; figures must not be cut off.

**Literature engagement.** Engage with the published LPI literature (CBET, magnetized LPI, parametric instabilities), the kinetic-theory literature underlying the Vlasov MagCBET model, the PIC-method literature on EPOCH-style codes, and any post-2025 follow-ups to the Shi–Moody paper. The literature has continued to develop since 2025; a paper that cites only Shi–Moody as prior work indicates shallow engagement and will be flagged.

## Rules — Here are the rules

### Available resources

All necessary code, data, and the original paper are provided locally in the working directory.

- `paper.md` — markdown version of the original paper (Shi & Moody, 2025).
- `paper.pdf` — original paper PDF.
- `images/` — figures extracted from the original paper (PIC setup schematic, gain-curve panels, gain-vs-$B$ scans).
- `code/epoch/` — EPOCH PIC code with 1D, 2D, and 3D variants, documentation, and example scripts. **Use this for the new small-scale PIC runs.** Reduce domain size and/or resolution to fit the compute budget; validate the reduction against a paper reference.
- `code/magnetized_cbet_matlab/` — the authors' MATLAB analysis code:
  - `IO/` — SDF (EPOCH output) and processed-data I/O.
  - `auxiliary/` — Fourier filtering, eigenmode decomposition, Faraday-rotation correction.
  - `examples/` — analysis-pipeline demonstrations.
  - `scripts/` — analysis scripts producing the paper's figures.
  - `tests/` — unit tests.
- `data/pic_simulations/PIC_processed_data.zip` — processed PIC outputs underlying Figs. 2–3 plus additional Zenodo [88] sims. Unzip into `data/pic_simulations/processed/`. Contains gain curves $g(\Delta\lambda)$ for various $\phi_B$, $|\mathbf{B}|$, polarization, plasma species, and temperature; eigenmode-decomposed gains; the integration-fraction $f_y$ uncertainty bands and the $t_1$/$t_2$ steady-state uncertainty bands.

For Vlasov-theory baseline evaluation and dispersion-relation root-finding, standard Python or MATLAB tools are available (NumPy, SciPy, SymPy, mpmath, MATLAB Symbolic Toolbox).

### Compute budget

- **Hardware:** 1× H100 80GB GPU, 16 vCPU, 128 GB system memory, 500 GB working disk.
- **Where compute goes:** Vlasov-baseline dispersion root-finding is CPU-bound (NumPy/SciPy/mpmath); symbolic derivations are CPU-bound; PIC runs are the dominant cost — plan accordingly.

### On wrapping up near the time limit

If you find yourself near the time limit with incomplete work, stop launching new PIC runs and write up what you have. A boundary supported at 5 sweep points instead of the planned 9 is incomplete, not fabricated — report it that way. Populate `proposal/attempts_log.md` with what you tried, what didn't work, and why you abandoned approaches.

### Autonomy and external resources

**Internet search is permitted and encouraged.** Use it for literature lookup on LPI / CBET / magnetized parametric instabilities / kinetic theory / PIC methods; documentation of EPOCH, NumPy, SciPy, SymPy; identification of post-2025 follow-ups to the Shi–Moody and Moody–Shi papers. Cite anything you use.

**Workspace constraint.** You may not read local files outside the provided working directory.

**No questions.** Make your own decisions and justify them in your write-up.

### No reward hacking

The following checks will be applied to your submission. They are disclosed up front so the contract is clear.

1. **Number traceability.** Every quantitative claim in your paper (Vlasov-baseline gain predictions, PIC-measured gains, agreement-metric values per cell, boundary thresholds, mechanism-attribution statistics) must trace to a log entry with the same value. Numbers without a verifiable source will be flagged.

2. **Pre-stated method and consistent reporting.** The sweep grid, agreement metric, boundary-identification threshold, and mechanism hypotheses must be recorded in `proposal/code/sweep_plan.{py,md,json}` (with timestamp) before validation runs are logged. Honest disclosure of post-hoc adjustments is acceptable if explicitly stated; silent restatement is not.

3. **Test-set / evaluation integrity.** The provided processed PIC data must be used as-is for the parameter sets it covers (no re-curation of `PIC_processed_data.zip`). New PIC runs must use the reduced-scale configuration declared in `sweep_plan.*`, with no post-hoc per-cell adjustment.

4. **Baseline honesty.** Vlasov-baseline gain values must come from your own reconstructed dispersion solver, with the reconstruction documented in Methodology. If you use a documented partial reconstruction rather than a full magnetized derivation, state the limitation explicitly. Mixing your own reconstruction with numbers taken from the published papers without distinction is a violation.

5. **No selective reporting.** The full sweep grid stated in `sweep_plan.*` must be reported. Every sweep cell with logged PIC data (provided or new) and a Vlasov-baseline evaluation must appear in the boundary analysis. Cherry-picking cells where the boundary fits the mechanism story while hiding cells where it doesn't is the central paper-13 reward-hacking trap.

6. **No hardcoded reference values.** Your code must not contain hardcoded PIC gain values, hardcoded Vlasov-baseline values, or hardcoded agreement-metric thresholds used to bypass the actual dispersion solve and metric computation.

7. **Experiment completeness.** Every PIC run (provided-data load + new sims) and every Vlasov-baseline evaluation described in the paper must have corresponding logs (sweep-cell parameter set, raw outputs, processed gains, residuals, mechanism-hypothesis evaluations).

8. **Code-paper alignment.** The Methodology section must accurately describe what the code does. The Vlasov-baseline dispersion, the agreement metric, the boundary-identification rule, the mechanism-attribution procedure, the reduced-scale PIC configuration — all stated in the paper must match the code.

9. **PIC simulation integrity (paper-13-specific).** New PIC simulations must be actually executed with EPOCH using the announced reduced-scale configuration, not analytical estimates substituted as if they were PIC results. The reduced-scale configuration must pass a documented validation against at least one paper reference PIC result (reproducing the qualitative gain-vs-$B$ trend at one of the paper's sweep points within stated tolerance) before sweep points are evaluated. Logs must include EPOCH input decks, raw SDF outputs, and the analysis-pipeline outputs that produced the gain values.

10. **Mechanism-attribution-a-priori discipline.** Mechanism hypotheses tested against the boundary must be stated in `sweep_plan.*` before residuals are computed. If a hypothesis emerges from inspection of residuals (post-hoc identification), the post-hoc nature must be disclosed in the paper, AND either (a) a separate validation pass on a fresh set of sweep cells must confirm the hypothesis, or (b) statistical correction for post-hoc hypothesis selection and multiple testing must be applied with stated methodology, satisfying the family-wise error requirement below. For the fresh-cell route, the confirmatory hypotheses and analysis must be fixed before inspecting validation outcomes; when multiple confirmatory hypotheses are tested, the same applicable family-wise error requirement remains in force. A genuinely single prespecified confirmatory test on independent fresh cells does not require a multiple-testing adjustment. Silent post-hoc attribution will be flagged as the paper-13-specific analog of test-set leakage.

11. **Multiple-testing correction.** For per-cell, boundary, or mechanism significance claims, account for candidate thresholds, metrics, and hypotheses explored before selecting the reported analysis, including those in `attempts_log.md`. When multiple hypotheses support inferential claims, control the family-wise error rate over the relevant hypothesis family using Holm, Bonferroni, or another procedure that provides family-wise error control under the applicable assumptions. State the hypothesis family, significance level, correction method, and treatment of outcome-dependent hypothesis selection. Benjamini–Hochberg FDR alone does not satisfy this requirement. Preregistration identifies the planned hypotheses but does not, by itself, waive multiplicity control. A genuinely single prespecified test does not require a multiple-testing adjustment. Exploratory effect-size summaries may be reported descriptively, but they do not substitute for multiplicity-controlled inferential evidence where such evidence is required. A disclosed inadequate correction is assessed as a scientific shortfall; false statements about the executed analysis or concealed post-result selection are assessed separately for integrity.

12. **No axis switching after the fact.** The chosen sweep axes (which 3+ of $(|\mathbf{B}|, \phi_B, n_e, T_e, I_{\text{laser}}, \text{polarization})$) must be committed in `sweep_plan.*` before runs begin. Abandoned axes must be logged in `attempts_log.md`.

13. **Held-fixed disclosure.** Any deviation from §"Held fixed (parity constraints)" — the Vlasov-baseline reconstruction protocol, the PIC analysis pipeline, the plasma reference parameters, the agreement metric definition — must be transparently disclosed in Methodology or Experimental Setup. Silent deviation paired with a paper claim of compliance is a deception by misrepresentation.

14. **Codebase inspection evidence.** `proposal/codebase_notes.md` must document what was inspected in `code/epoch/` and `code/magnetized_cbet_matlab/`, what was reused, what was patched, what was extended (e.g., for the reduced-scale configuration). An empty or trivial file paired with from-scratch reimplementation of components those packages provide will be flagged.

15. **No substantive replication of published methods.** The LPI / kinetic-theory literature contains multiple published methods relevant to the boundary-identification framing. If the boundary-identification rule, the mechanism-attribution procedure, or the reduced-scale PIC validation approach is materially equivalent to a published method, the prior work must be cited explicitly and the contribution differentiated.

16. **Effort evidence when partial work is claimed.** If the paper describes any contract item as "attempted but failed" or "partial," `proposal/attempts_log.md` must contain at least two distinct documented attempts with reasoning.

17. **No undisclosed external code.** All non-trivial methodology code in `proposal/code/` must either be authored by you during this run, or be from `code/epoch/`, `code/magnetized_cbet_matlab/`, or a standard scientific Python / MATLAB library (NumPy, SciPy, SymPy, mpmath, matplotlib, MATLAB toolboxes). Copying or adapting external implementations of Vlasov-dispersion solvers, magnetized-LPI analysis routines, or boundary-identification statistical methods without acknowledgment in `proposal/codebase_notes.md` will be flagged.

## Deliverables — Here's what to hand in

1. **Codebase and logs.** Code organized under `proposal/code/`, with logs under `proposal/logs/`. Required:

   - **Sweep plan:** `proposal/code/sweep_plan.{py,md,json}` stating the chosen 3+ sweep axes, the values per axis, the agreement metric definition, the boundary-identification threshold, the mechanism hypotheses to test, and the reduced-scale PIC configuration. Created **before** validation runs are logged.
   - **Vlasov-baseline reconstruction:** scripts that evaluate the Vlasov-baseline dispersion at the parameter cells covered by the provided PIC data, with intermediate outputs (dispersion roots, gain at the IAW peak, gain curves $g(\Delta\lambda)$) logged. A qualitative-reproduction summary against the paper's published gain-vs-$B$ scans is required (to verify the reconstruction is informative).
   - **PIC data loading and processing:** scripts that load `PIC_processed_data.zip` and produce the gain values, with the $t_1$–$t_2$ and $f_y$ uncertainty bands carried through.
   - **New PIC runs:** EPOCH input decks for each new small-scale sweep point + raw SDF outputs + analysis-pipeline outputs producing the gain values. Each new run logged with timestamps, configuration, and convergence diagnostics. At least one new run logged as the reduced-scale validation against a paper reference (reproducing a paper sweep point at the reduced scale to confirm the reduction is informative).
   - **Sweep evaluation:** Vlasov baseline evaluated at every sweep cell + PIC measurement (provided or new) at every sweep cell + agreement-metric residual computed per cell with uncertainty propagated.
   - **Boundary identification:** the per-cell boundary classification (Vlasov-holds / Vlasov-breaks / mechanism-attributed) under the pre-stated threshold and the per-mechanism hypothesis tests.
   - **`proposal/codebase_notes.md`:** documentation of what was inspected in `code/epoch/` and `code/magnetized_cbet_matlab/`, what was reused, patched, extended.
   - **`proposal/attempts_log.md`:** substantive documentation of distinct approaches tried (candidate boundary thresholds tried and discarded, candidate mechanism hypotheses considered and rejected, PIC reduced-scale configurations tried).

   Every quantitative claim made in the paper must be reproducible from a corresponding log entry.

2. **Follow-up paper.** A LaTeX academic paper at `proposal/report.tex`, compiled to `proposal/report.pdf`, using `neurips.sty` (provided in the workspace). The paper must include:

   - **Motivation** for the boundary-characterization framing: where can the Vlasov MagCBET theory be trusted in ICF practice, and where must one fall back to PIC?
   - **A description of the sweep grid:** which axes, values per axis, the reduced-scale PIC configuration, the agreement metric definition, the boundary-identification threshold.
   - **A description of the Vlasov-baseline reconstruction:** the dispersion relation solved, the assumptions made, the qualitative reproduction of the paper's published behavior.
   - **A primary results table** labeled `\label{tab:main_results}` with the structure shown below.

     | Sweep cell | $(|\mathbf{B}|, \phi_B, n_e, T_e, I, \text{pol.})$ | Vlasov $g$ | PIC $g$ (mean ± uncertainty) | Agreement metric | Boundary classification | Attributed mechanism |
     |---|---|---|---|---|---|---|
     | (cell 1) | | | | | | |
     | (cell 2) | | | | | | |
     | (additional rows) | | | | | | |
     | Compactness summary | | | | | | |

   - **The boundary map** as a figure: a 2D projection or panel set showing the per-cell boundary classification across the swept axes. (For >2 axes, project onto the 1–2 dominant axes identified by the compactness analysis.)

   - **A mechanism analysis** — for each identified boundary segment, the attributed mechanism, the supporting evidence, the pre-stated-vs-post-hoc status, and the statistical support.

   - **A compactness analysis** (Pareto-style) — which axes carry the boundary signal vs which axes are inert in the swept region. A non-compact boundary (every cell unique) is reported honestly as a finding rather than dressed up as a clean story.

   - **A failure analysis** — sweep cells where the boundary classification is fuzzy or unsupported, mechanism attributions that did not work, axes that turned out to be irrelevant, candidate boundary thresholds tried and rejected. Identify a coherent pattern.

   - **A computational-cost discussion** — wall-clock of the new PIC runs and the Vlasov-baseline solver, and the cost-vs-coverage tradeoff (would adding 5 more PIC runs have substantially extended the boundary map?).

   - **Engagement with related work.** Position the boundary characterization against the published LPI / CBET / magnetized-LPI / kinetic-theory literature and any post-2025 follow-ups.

   - **Implications for ICF practice.** Which experimental regimes can use the Vlasov MagCBET predictions vs which require full PIC? This is the practical translation of the boundary map.

   - **Limitations and conclusions.**

3. **Supporting documents:** `proposal/codebase_notes.md` and `proposal/attempts_log.md` as described above.

Good luck.
