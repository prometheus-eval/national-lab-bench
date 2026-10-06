# Task: Develop a Substantive Follow-up to "Simulating Plasma Wave Propagation on a Superconducting Quantum Chip"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the spin-chain plasma-wave simulation by Sundar, Evert, Geyko, Patterson, Joseph, and Shi, "Simulating plasma wave propagation on a superconducting quantum chip," Phys. Rev. Applied 25, 024077 (2026). Your contribution is to **extend the spin-chain Hamiltonian with ZZ terms to capture a specific nonlinear plasma effect** — and demonstrate that the extended model captures it via a three-step program the author of the original paper sketched as the natural follow-up:

1. **Derive the mapping**: show analytically how the ZZ-augmented spin-chain Hamiltonian maps to the classical partial differential equation(s) governing the chosen nonlinear plasma phenomenon.
2. **Emulate the quantum model classically**: simulate the ZZ-augmented Hamiltonian's dynamics on a 9–12 qubit lattice using classical statevector simulation (the actual quantum chip is not part of this contract).
3. **Compare emulation to classical PDE solutions**: at sweep points in a regime where both the classical statevector emulation and the classical PDE reference are tractable, demonstrate quantitative agreement on the chosen observable, within a pre-stated tolerance.

The headline contribution is the *quantitative emulation-vs-PDE agreement* for your ZZ-augmented model on the nonlinear phenomenon it describes; the mapping derivation is the supporting theoretical machinery; the classical emulation is the computational artifact. The contract requires you to commit to one ZZ model (the $K_{ij}$ matrix) up front, identify which nonlinear plasma phenomenon (if any) that model describes — EIT, modulational instability, or another well-motivated nonlinear plasma effect — and demonstrate agreement with that phenomenon's classical PDE within a stated tolerance across an announced parameter sweep.

## The Problem — Here's why it matters

Quantum plasmas — strongly coupled or relativistic-quantum regimes that arise in warm dense matter, stellar interiors, and laser-driven high-energy-density experiments — are classically hard to simulate at the level of the full electronic-ionic Hamiltonian. Linear plasma-wave propagation, governed by $(\partial_t^2 - c^2 \partial_x^2 + \omega_p^2) A = 0$ with dispersion $\omega_k = \pm\sqrt{\omega_p^2 + c^2 k^2}$, is the foundational building block: it underlies laser–plasma interactions in inertial confinement fusion, ionospheric and astrophysical plasmas, and stopping-power calculations. The Sundar et al. spin-chain model maps single-excitation states to plasma waves, enabling shallow NISQ-compatible Trotterized evolution at $\mathcal{O}(N)$ gate depth on current superconducting devices, where $N$ is the number of spatial grid points.

But the original paper covers only the *linear* regime — the single-excitation Hilbert space, where the model is classically simulable in polynomial time. The interesting physics — laser–plasma scattering, electromagnetically-induced transparency (EIT), modulational instabilities — is *nonlinear*, and the quantum version of these problems lives outside the single-excitation subspace, and is classically intractable at scale. The paper explicitly identifies the path forward: "With more general spin Hamiltonians, for example, with additional ZZ interactions, our approach can be used to simulate nonlinear effects in plasmas, such as electromagnetically induced transparency, laser-plasma scattering, and modulational instabilities." This task takes that path.

The contribution this follow-up makes is *connecting the spin model to the plasma physics*. Adding ZZ terms is mathematically straightforward; demonstrating that the resulting dynamics quantitatively captures a specific nonlinear plasma phenomenon is not. The author of the original paper sketched the 3-step program above as the way to do it: derive how the spin-model dynamics maps to the classical PDE, emulate the quantum model on a small lattice, and quantitatively compare. The 3 steps together establish the spin-model–plasma-physics correspondence in the nonlinear regime, with the empirical agreement at sweep points carrying the burden of demonstration.

## Background — Here's what you need to know about the original paper

The original paper introduces and validates a spin-chain Hamiltonian whose single-excitation excitation spectrum maps onto linear plasma-wave dispersion, runs it on a 9-qubit sublattice of Rigetti's Ankaa-3, and demonstrates plasma-wave propagation in vacuum, at a sharp plasma boundary, and through an inhomogeneous plasma. The technical components are:

1. **Spin-chain Hamiltonian** [Eq. (3) of the paper]:
   $$H_0 = -\frac{J}{4} \sum_{j=1}^{N-1} (\sigma_j^x \sigma_{j+1}^y - \sigma_j^y \sigma_{j+1}^x) + \frac{1}{2} \sum_{j=1}^N \Delta_j (-1)^j \sigma_j^z.$$
   Single-excitation states $|0\cdots 1_j \cdots 0\rangle$ with amplitudes $b_j$ obey
   $$i\hbar \partial_t b_j = \frac{i J}{2}(b_{j-1} - b_{j+1}) + \Delta_j (-1)^j b_j,$$
   which (for uniform $\Delta_j = \Delta$) reduces to a central-difference discretization of the plasma wave equation. Long-wavelength dispersion: $\hbar\omega_k = \pm\sqrt{\Delta^2 + J^2 \sin^2(ka)} \to \pm\sqrt{\Delta^2 + (Jka)^2}$, matching the plasma-wave dispersion with $\Delta \leftrightarrow \omega_p$.

2. **Trotterized evolution** of $H_0$ on a 9-qubit sublattice using $\sqrt{\text{iSWAP}}$-like FSIM gates as the native two-qubit operation plus single-qubit $R_X(\pi/2)$ rotations. Trotter step $\delta = 0.8$.

3. **Error mitigation** via pseudotwirling (FSIM-specific extension of Pauli twirling, since FSIM is non-Clifford), Clifford data regression, and conservation rescaling of $\langle \sigma_{\text{tot}}^z \rangle$. Classical shadows for state estimation.

4. **Three demonstration experiments** (Secs. IV–V):
   - **Dispersion measurement** ($\Delta = 0$ and $\Delta = J/4$): Ramsey-type protocol prepares one qubit in $(|0\rangle + |1\rangle)/\sqrt{2}$, evolves, and measures $\langle\sigma_j^x + i\sigma_j^y\rangle$ to extract eigenfrequencies. All 9 spin-wave eigenfrequencies match exact and noiseless-simulation values within a few percent.
   - **Vacuum propagation** ($\Delta = 0$): a localized two-site spin-wave packet propagates ballistically; experimental data track noiseless simulation through ~half the lattice (~5 sites of 9).
   - **Sharp boundary** ($\Delta = J$ jump) and **inhomogeneous propagation** (Gaussian density profile): wave packet reflects from the overdense plasma; mitigated experimental traces match noiseless simulation.

5. **Quantum-advantage discussion (Sec. VI and Appendix C):** linear plasma-wave evolution maps to a noninteracting bosonic model, classically simulable in $\mathcal{O}(M_0 K N_s^2)$ to $\mathcal{O}(M_0 N_s^3)$. **The ZZ-extension breaks the noninteracting structure**: classical simulation generically takes exponential time $\mathcal{O}(2^{3N_s})$ to $\mathcal{O}(2^{4N_s})$ in the worst case. The ZZ-extension is therefore the natural route to nonlinear plasma physics with potential quantum advantage.

**The paper's explicit invitation (Sec. I and Sec. VI):** "With more general spin Hamiltonians, for example, with additional ZZ interactions, our approach can be used to simulate nonlinear effects in plasmas, such as electromagnetically induced transparency, laser-plasma scattering, and modulational instabilities. We leave such an experiment to future efforts."

**Reference baseline behavior reported in the paper:**

- **Dispersion ($\Delta = 0$ and $\Delta = J/4$):** all 9 spin-wave eigenfrequencies match exact and noiseless-simulation values within a few percent.
- **Vacuum propagation:** wave-packet center-of-mass tracks group velocity $v = J/(a\hbar)$ qualitatively through ~5 sites of the 9-site lattice.
- **Sharp boundary and inhomogeneous propagation:** wave-packet center-of-mass reflects from the overdense region; mitigated experimental trace matches noiseless simulation.
- **Hardware metrics (Table I):** $T_1 = 37.6\,\mu$s, $T_2 = 22.3\,\mu$s, FSIM infidelity $0.60\%$, FSIM duration 54 ns, $R_X(\pi/2)$ infidelity $0.09\%$, EPLG $1.48\%$ on the 9-qubit sublattice.

## Contract — Here's exactly what counts as success

Your contribution is a **ZZ-augmented spin-chain model** for a specific nonlinear plasma phenomenon, demonstrated via the three-step program above. The contract operationalizes verifiability through the empirical emulation-vs-PDE agreement metric: the agent commits to one modification to the spin Hamiltonian up front, derives the mapping, identifies what plasma phenomenon the model can describe, emulates the model classically, and shows quantitative agreement at announced sweep points within a pre-stated tolerance.

### Operationalized success thresholds — single-ZZ-model commitment

There are multiple ways to add ZZ terms in the spin-Hamiltonian. It is not clear what each way of adding ZZ term can describe. You should commit in `proposal/code/method_plan.{py,md,json}` to **one specific way of adding a ZZ term** before the rest of the steps begins. The original paper named candidate plasma phenomena that adding the ZZ term may describe. These phenomena, in order of decreasing classical-PDE accessibility, are:

- **Axis (1): Modulational instability of plasma waves.** The classical reference is the nonlinear Schrödinger equation (NLSE) in the appropriate plasma limit. The headline observable is the growth rate $\Gamma(k)$ of the most unstable side-band as a function of the perturbation wavenumber $k$, in a sweep over the ZZ-coupling strength $\epsilon$. NLSE prediction: $\Gamma(k) = k\sqrt{2|q|\,|\psi_0|^2 - k^2}/2$ in normalized units, with the maximum at $k = \sqrt{|q|}\,|\psi_0|$, where $q$ is the NLSE nonlinearity coefficient set by the ZZ coupling.
- **Axis (2): Electromagnetically-induced transparency (EIT).** The classical reference is the EIT three-level density-matrix solution. The headline observable is the transparency-window width and depth in the absorption spectrum, as a function of the probe-pump detuning, in a sweep over the ZZ coupling. EIT prediction comes from the standard analytic solution of the three-level Λ-system master equation.
- **Axis (3): Stimulated Raman scattering (SRS) growth rate.** The classical reference is the textbook SRS coupled-mode equations in the undepleted-pump limit. The headline observable is the Stokes-wave growth rate as a function of detuning, in a sweep over the ZZ coupling.
- **Axis (4): Another well-motivated nonlinear plasma phenomenon you justify.** Subject to the same operationalization shape: a stated PDE reference, a headline observable, a parameter sweep, a pre-stated agreement threshold.

For the chosen axis, the strict criterion is:

- **Agreement on the headline observable** at $\geq 4$ sweep points spanning a documented dynamic range of the ZZ coupling, within a pre-stated tolerance (typical target: $\leq 15\%$ relative error on the headline observable at each sweep point). The threshold is committed in `method_plan.*` before runs.
- **Linear-limit reduction verified** both algebraically and numerically: $\lim_{\epsilon \to 0} H = H_0$ by construction, and at $\epsilon = 0$ the simulation reproduces the paper's noiseless-simulation dispersion (≤5% relative error on all 9 spin-wave eigenfrequencies in both $\Delta = 0$ and $\Delta = J/4$ cases) AND the wave-packet propagation traces (qualitative match to Fig. 4 noiseless-simulation curves).
- **Contrast against the unextended $H_0$**: at the same finite $\epsilon$ used for the headline observable, run the unextended $H_0$ as a baseline. $H_0$ cannot capture the nonlinear effect (free-boson model); the contrast between $H = H_0 + H_{ZZ}$ and $H_0$ is the demonstration that the ZZ term is what produces the captured nonlinear effect.
- **Full-Hilbert-space evolution** (not single-excitation truncation): the simulation must operate on the full $2^N$-dimensional Hilbert space, because ZZ terms behave differently on single- and multiple-excitation subspaces, and the nonlinear effect emerges precisely from multiple excitations. Single-excitation truncation removes the very nonlinearity the ZZ extension is meant to introduce.

### Held fixed (parity constraints)

These must remain identical to the paper's framework — they are not part of the design space.

- **The original spin-chain Hamiltonian $H_0$**: exactly as in Eq. (3) of the paper, with the same $J$ and $\Delta_j(-1)^j \sigma_j^z$ staggered-mass structure. The ZZ extension is added as $H = H_0 + H_{ZZ}$; $H_0$ is not modified.
- **Trotter step**: $\delta = 0.8$ (the paper's value) or smaller with stated justification. Larger Trotter steps to mask discretization error in the nonlinear regime are not allowed.
- **Lattice size**: 9–12 qubits. The 9-qubit case allows direct comparison to the paper's results; up to 12 qubits is permitted to gain headroom for the nonlinear-effect demonstration. The contract is about the model's physics, not a hardware demonstration; the actual quantum chip is not part of the contract.
- **Linear-limit reduction**: $\lim_{\epsilon \to 0} H = H_0$ must hold by construction, verifiable algebraically. The reduction must be exact (not approximate).
- **Single-ZZ commitment**: once committed in `method_plan.*`, the agent does not switch to another way of introducing ZZ terms. In other words, the agent should fix the $K_{ij}$ matrix described below. Different phenomena have different PDE references, observables, and tolerances — switching defeats the verifiability of the agreement metric.

### What you may design

- **The ZZ coupling structure $K_{ij}(\epsilon)$**: nearest-neighbor, next-nearest-neighbor, longer-range, or a position-dependent profile. The choice must be physically motivated (e.g., capturing the plasma-density-dependent ponderomotive coupling for modulational instability, or the dipole-coupling pattern for EIT). Dense all-to-all $K_{ij}$ without physical motivation is not acceptable.
- **The PDE reference**: which classical PDE governs the chosen phenomenon at the operating regime. Cite from the plasma-physics literature.
- **The headline observable**: which quantity to extract from the emulation and compare to the PDE prediction (growth rate, transmission spectrum, gain coefficient, etc.).
- **The $\epsilon$ sweep grid**: the parameter values at which you evaluate the headline observable. At least 4 sweep points spanning a documented range; stated in `method_plan.*` before runs.
- **The agreement tolerance**: the threshold at which agreement is declared. Typical $\leq 15\%$ relative error; the agent may justify a different threshold. Pre-stated.

### Anti-cheating

The contract is about the ZZ extension and its honest demonstration. 

- The chosen ZZ model (namely, the $K_{ij}$ matrix) and tolerance should be committed in `method_plan.*` before runs. Abandonment must be logged in `proposal/attempts_log.md`.
- **Single-excitation truncation as "emulation."** The ZZ term acts nontrivially in multi-excitation subspaces; truncating to the single-excitation block removes the nonlinearity. Logs must demonstrate full-$2^N$-dimensional evolution.
- **Linear-limit verification at "small $\epsilon$" instead of $\epsilon = 0$.** The linear-limit reduction must be verified at exactly $\epsilon = 0$ — at that value, the simulation must match the unextended $H_0$ at numerical precision. Small but nonzero $\epsilon$ that happens to look approximately linear is not the verification asked for.
- All sweep points of $\epsilon$ values declared in `method_plan.*` must appear in the main results. Reporting only the favorable ones is selective.
- **Cannot adjust the PDE-reference prediction after running the emulation.** The PDE-reference computation must be defined from the cited literature and computed independently from the emulation, before residuals are calculated.
- **No hardcoded reference values.** No hardcoded growth rates, transmission curves, or simulation outputs that bypass actual statevector evolution.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated along two dimensions:

1. **Performance on stated metrics.** Linear-limit reproduction of the paper's dispersion and propagation traces within the stated tolerance; emulation-vs-PDE agreement on the headline observable at $\geq 4$ sweep points within the pre-stated tolerance; contrast against the unextended $H_0$ at $\epsilon=0$; mapping derivation reproducible from logs.

2. **Scientific merit.** Your paper is assessed on the novelty of the ZZ extension (or the rigor of its motivation if convergent with published work), the depth of the mapping derivation (is it traceable from $H = H_0 + H_{ZZ}$ to the chosen PDE?), the mechanism analysis (which terms in $H_{ZZ}$ correspond to which terms in the PDE?), the failure analysis (where the mapping breaks down), and the engagement with the plasma-physics literature.

### Paper quality

Your submission is an academic paper, not a class project report. Writing should resemble what would be reviewed at a top scientific journal in structure, integrity, depth, and language.

Do not frame the paper as "a follow-up to Sundar et al." Frame it as a self-contained scientific contribution describing the spin-model–plasma-physics correspondence in the nonlinear regime, with the ZZ-extension and the chosen phenomenon as the headline.

**Independence from this task spec.** A reader who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. Do not use phrases like "axis (1)" / "step 1" / "headline observable" verbatim where physics-natural terminology would do. Section structure should be physics-driven, not contract-driven.

**Rendered output verification.** Render LaTeX to PDF and visually inspect before submitting.

**Literature engagement.** Engage with the published plasma-physics literature for the chosen nonlinear effect (the classical NLSE / EIT / SRS / equivalent literature you cite for the PDE reference), the quantum-simulation-of-plasma-physics literature, the spin-model–to–field-theory mapping literature, and any post-2026 follow-ups to the Sundar et al. paper. A paper that cites only Sundar et al. as prior work indicates shallow engagement.

## Rules — Here are the rules

### Available resources

All necessary references and the original paper are provided locally in the working directory.

- `paper.md` — markdown version of the original paper (Sundar et al., 2026).
- `paper.pdf` — original paper PDF.
- `images/` — figures extracted from the original paper (lattice schematic, dispersion-relation panels, wave-packet propagation traces, Trotter-circuit diagram, error-mitigation comparison, classical emulation of large-lattice propagation).
- `code/` — empty. The paper does not provide source code; you will implement the classical statevector simulator yourself in your preferred language.
- `data/` — empty. The paper's experimental data is publicly available on Zenodo at doi.org/10.5281/zenodo.16115660 but is not part of this contract; the linear-limit reproduction uses the paper's noiseless-simulation traces (which you compute yourself by classical statevector simulation) as the reference.

For statevector simulation and Trotterized evolution, standard scientific Python tooling is available (NumPy, SciPy, Qiskit, PennyLane, OpenFermion). For symbolic derivation of the linear-limit reduction and the mapping to the classical PDE, use SymPy or `mpmath`.

### Compute budget

- **Hardware:** CPU machine with 32 vCPU cores, ~60 GB RAM, ~470 GB working disk. No GPU, and no swap. Treat ~50 GB as a hard memory ceiling (the node has ~60 GB RAM, no swap, and the OS + agent harness need headroom): size grids, dense matrices, ensembles, batch sizes, and parallel workers to keep peak resident memory under it; estimate/monitor peak memory before scaling up and prefer chunked / streaming / sparse computation. A process that exhausts RAM is killed by the OS.
- **Scope guidance:** The full Hilbert space of 12 qubits ($2^{12} = 4096$ dimensions) fits trivially in memory; dynamics costs $\mathcal{O}(\text{Trotter steps} \times d^2)$ in time per run.

### On wrapping up near the time limit

If you find yourself near the time limit with incomplete work, stop running new sweeps and write up what you have. A sweep validated at 3 points instead of the planned 5 is incomplete, not fabricated — report it that way. Populate `proposal/attempts_log.md` with what you tried, what didn't work, and why you abandoned approaches.

### Autonomy and external resources

**Internet search is permitted and encouraged.** Use it for literature lookup on the nonlinear plasma phenomenon you choose, on spin-model–to–PDE mappings, on quantum simulation of plasma physics, and on recent follow-ups to Sundar et al.; for documentation of NumPy, SciPy, Qiskit, PennyLane, OpenFermion, SymPy; for citation of any plasma-physics textbook results used. Cite anything you use.

**Workspace constraint.** You may not read local files outside the provided working directory.

**No questions.** Make your own decisions and justify them in your write-up.

### No reward hacking

The following checks will be applied to your submission. They are disclosed up front so the contract is clear.

1. **Number traceability.** Every quantitative claim in your paper (dispersion-frequency values, propagation center-of-mass traces, nonlinear-effect observables, $\epsilon$ values, agreement metrics) must trace to a corresponding log entry with the same value. Numbers without a verifiable source will be flagged.

2. **Pre-stated method and consistent reporting.** The chosen ZZ-coupling structure $K_{ij}(\epsilon)$, PDE reference, headline observable, sweep grid, and agreement tolerance must be recorded in `proposal/code/method_plan.{py,md,json}` with timestamp predating the validation runs. Honest disclosure of hyperparameter refinement after validation (e.g., the $\epsilon$ sweep grid being extended) is acceptable if explicitly stated in the paper; silent restatement is not.

3. **Test-set / evaluation integrity.** Linear-limit verification must be at exactly $\epsilon = 0$. The sweep at finite $\epsilon$ must be at the values declared in `method_plan.*`. PDE-reference predictions must be computed from the cited classical reference, not adjusted after seeing the emulation results.

4. **Baseline honesty.** The paper's noiseless-simulation results (dispersion-relation values, Fig. 4 propagation traces) must be reproduced from your own classical statevector simulator at $\epsilon = 0$ (the paper's noiseless simulation *is* a classical statevector computation, so you can match it numerically). The unextended-$H_0$ contrast runs must come from your own logged runs.

5. **No selective reporting.** All sweep points declared in `method_plan.*` must appear in the main results table. If multiple ZZ-coupling structures or $\epsilon$ ranges were tried, all must be reported (or principled aggregation must be used). Cherry-picking the $\epsilon$ where the PDE agreement is best while hiding others will be flagged.

6. **No hardcoded reference values.** Your code must not contain hardcoded growth rates, transmission curves, Stokes-amplitude trajectories, dispersion-frequency values, or other ground-truth literals. PDE-reference predictions must be computed at runtime from the cited classical reference equations.

7. **Experiment completeness.** Every experiment described in the paper must have corresponding logs (statevector trajectories, Trotter-step records, observable measurements, $\epsilon$-sweep summaries, PDE-reference computations).

8. **Code-paper alignment.** The Methodology section must accurately describe what the code does. The Hamiltonian construction, the Trotter discretization, the ZZ-extension structure $K_{ij}(\epsilon)$, the observable measurements, and the PDE-reference computation stated in the paper must match the implementation.

9. **Full-Hilbert-space evolution (paper-18-specific).** The ZZ-extended simulation must operate on the full $2^N$-dimensional Hilbert space. Logs must demonstrate (a) the statevector norm conservation across the full $2^N$ dimensions during evolution, (b) population transfer outside the single-excitation subspace at finite $\epsilon$, and (c) the absence of any single-excitation truncation in the code. Substituting single-excitation truncation for the full-Hilbert-space evolution is the central paper-18 reward-hacking trap — it removes the nonlinearity the ZZ term is meant to introduce.

10. **Linear-limit reduction integrity.** The linear-limit reduction $\lim_{\epsilon \to 0} H = H_0$ must be verifiable algebraically (the ZZ term must contain $\epsilon$ as an overall multiplicative factor or equivalent structure that vanishes at $\epsilon = 0$) AND numerically (at $\epsilon = 0$, the ZZ-extended simulation must reproduce the unextended $H_0$ simulation at numerical precision, not "approximately"). Logs must include (a) the algebraic verification, (b) the side-by-side numerical comparison at $\epsilon = 0$ showing machine-precision agreement.

11. **No ZZ model switching after the fact.** The chosen ZZ model must be committed in `method_plan.*` before validation runs are logged. Abandoned models must be logged in `proposal/attempts_log.md`.

12. **Held-fixed disclosure.** Any deviation from §"Held fixed (parity constraints)" — the original $H_0$ structure, the Trotter step, the lattice size range, the linear-limit-reduction-by-construction requirement, the single-model commitment — must be transparently disclosed in Methodology or Experimental Setup. Silent deviation paired with a paper claim of compliance is a deception by misrepresentation.

13. **Codebase inspection evidence.** The provided `code/` directory is empty (the paper does not ship code). `proposal/codebase_notes.md` must document what you implemented from scratch vs. what you adapted from standard scientific Python libraries (NumPy, SciPy, Qiskit, etc.). A trivial or absent file paired with substantial reimplementation that should have leveraged standard tooling will be flagged.

14. **No substantive replication of published methods.** Published spin-model-to-PDE mapping techniques exist in the quantum-simulation-of-field-theories literature. If your mapping derivation is materially equivalent to a published technique (e.g., the bosonization mapping for spin-chain to nonlinear field theories, the Holstein-Primakoff transformation followed by classical limit, etc.), cite explicitly and differentiate. Convergent ideas openly acknowledged are fine; silent overlap is not.

15. **Effort evidence when partial work is claimed.** If the paper describes any contract item as "attempted but failed" or "partial," `proposal/attempts_log.md` must contain at least two distinct documented attempts with reasoning.

16. **No undisclosed external code.** All non-trivial methodology code in `proposal/code/` must either be authored by you during this run or be from a standard scientific Python library (NumPy, SciPy, Qiskit, PennyLane, OpenFermion, SymPy, mpmath, matplotlib). Copying or adapting external implementations of nonlinear plasma solvers, spin-model dynamics, or PDE-reference computations without acknowledgment in `proposal/codebase_notes.md` will be flagged.

## Deliverables — Here's what to hand in

1. **Codebase and logs.** Code organized under `proposal/code/`, with logs under `proposal/logs/`. Required:

   - **Method plan**: `proposal/code/method_plan.{py,md,json}` stating the chosen ZZ-coupling structure $K_{ij}(\epsilon)$, the PDE reference (with citation), the plasma phenomenon it describes, the headline observable, the $\epsilon$ sweep grid, and the agreement tolerance. Created **before** validation runs are logged.
   - **$H_0$ reproduction**: implementation of the unextended Hamiltonian, with the dispersion measurement (Sec. IV protocol) and the three propagation experiments (Sec. V protocols) reproduced. Logs show all 9 spin-wave eigenfrequencies for $\Delta = 0$ and $\Delta = J/4$ matching the paper's noiseless-simulation values within tolerance; wave-packet center-of-mass traces for vacuum, sharp boundary, and inhomogeneous propagation qualitatively matching the paper's Fig. 4 curves.
   - **Mapping derivation**: symbolic-computation artifacts (SymPy notebooks or equivalent) showing the derivation from $H = H_0 + H_{ZZ}$ to the classical PDE reference, with intermediate expressions saved. The derivation should make explicit which term in $H_{ZZ}$ corresponds to which nonlinear term in the PDE.
   - **Linear-limit verification**: at $\epsilon = 0$, side-by-side comparison of the ZZ-extended simulation with the unextended $H_0$ simulation, showing machine-precision agreement on the linear-regime observables (dispersion + propagation traces).
   - **$\epsilon$ sweep**: at each declared sweep point, full statevector trajectory, the headline observable extracted, and the PDE-reference prediction computed independently.
   - **$H_0$ contrast at finite $\epsilon$**: at the same finite-$\epsilon$ values as the sweep, the unextended-$H_0$ baseline runs, demonstrating that the nonlinear effect emerges from the ZZ term and not from some other artifact of the simulation.
   - **`proposal/codebase_notes.md`**: documentation of what was implemented from scratch vs. adapted from standard libraries.
   - **`proposal/attempts_log.md`**: substantive documentation of distinct approaches tried, including alternative ZZ-coupling structures considered and rejected, phenomena explored but not committed to, etc.

   Every quantitative claim made in the paper must be reproducible from a corresponding log entry.

2. **Follow-up paper.** A LaTeX academic paper at `proposal/report.tex`, compiled to `proposal/report.pdf`, using `neurips.sty` (provided in the workspace). The paper must include:

   - **Motivation** for extending the spin-chain plasma-wave model to a specific nonlinear phenomenon, tied to the original paper's invitation in Sec. VI.
   - **A description of the chosen ZZ model and PDE reference**: the classical PDE governing it, the headline observable, the parameter range where both the classical PDE and the classical statevector emulation are tractable.
   - **The ZZ-coupling structure $K_{ij}(\epsilon)$**: the mathematical form, the locality / range, the physical motivation (e.g., why nearest-neighbor ZZ corresponds to the local ponderomotive coupling, or why a position-dependent ZZ corresponds to a particular dipole-coupling pattern). The full Hamiltonian $H = H_0 + H_{ZZ}$.
   - **The mapping derivation**: how $H$ maps to the classical PDE in the appropriate limit (continuum, low-excitation-density, classical correspondence, etc.). Explicit identification of which term in $H_{ZZ}$ corresponds to which nonlinear term in the PDE. Full derivation may live in an appendix if length-prohibitive.
   - **A primary results table** labeled `\label{tab:main_results}` with the structure below:

     | $\epsilon$ | Headline observable (simulation) | Headline observable (PDE) | Relative error | Linear-limit observable (dispersion or propagation) | Linear-limit tolerance check | Notes |
     |---|---|---|---|---|---|---|
     | 0 (linear baseline) | n/a | n/a | n/a | (dispersion / propagation values) | (paper-match) | $H_0$ recovered exactly |
     | $\epsilon_1$ | | | | | (paper-match within tolerance) | |
     | $\epsilon_2$ | | | | | | |
     | $\epsilon_3$ | | | | | | |
     | $\epsilon_4$ | | | | | | |
     | $\epsilon_{>4}$ (sweep points) | | | | | | |

   - **The sweep figure**: emulation observable vs. PDE prediction across the $\epsilon$ sweep, with the agreement metric per point and the threshold marked.
   - **The linear-limit figure**: at $\epsilon = 0$, overlay of the ZZ-extended simulation and the unextended $H_0$ simulation, showing machine-precision agreement on the linear observables.
   - **The $H_0$ contrast figure**: at a finite $\epsilon$ point in the sweep, side-by-side comparison of the ZZ-extended simulation and the unextended $H_0$ simulation, showing that the nonlinear effect emerges from the ZZ term.
   - **A mechanism analysis** — why the chosen ZZ structure produces the chosen nonlinear effect, tying the ZZ term's mathematical form to the nonlinear coupling in the PDE.
   - **A failure analysis** — $\epsilon$ values or sweep regions where the agreement breaks down (e.g., where the PDE-reference's small-amplitude assumption fails, where higher-order terms become important, where Trotter error becomes comparable to the nonlinearity).
   - **A computational-cost analysis** — wall-clock of the classical statevector emulation and the PDE-reference computation, with discussion of how the cost scales with lattice size (where would the quantum advantage emerge).
   - **Engagement with related work**: the chosen-phenomenon classical literature, the spin-model–to–field-theory mapping literature, the quantum-simulation-of-plasma-physics literature, and any post-Sundar follow-ups.
   - **Limitations and conclusions**, including honest discussion of which regimes the ZZ-extended model can be expected to capture and which it cannot.

3. **Supporting documents**: `proposal/codebase_notes.md` and `proposal/attempts_log.md`.

Good luck.
