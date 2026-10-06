# Task: Develop a Substantive Follow-up to "An Efficient Explicit Implementation of a Near-Optimal Quantum Algorithm for Simulating Linear Dissipative Differential Equations"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the LCHS implementation by Novikau and Joseph, "An efficient explicit implementation of a near-optimal quantum algorithm for simulating linear dissipative differential equations," 2025. Your contribution is to **separate the mathematical algorithm (e.g. in the choice of kernel in the LCHS method and the choice of numerical integration/quadrature strategy) from the implementation as a quantum algorithm (in terms of the selector, the Hamiltonian simulation in terms of QSP, and the implementation the quadrature in terms of QSVT, etc.) and improve the mathematical algorithm**. Specifically: keep Novikau–Joseph's implementation framework intact, substitute an alternative kernel that achieves better asymptotic complexity scaling (toward the optimal dependence on error $\mathcal{O}(\log(\epsilon_{\text{LCHS}}^{-1}))$ on at least one of the three resource axes — query complexity, gate complexity, quantum memory), validate the substitution on the digital emulator, and characterize which kernel family is best at which $(n_k, \epsilon_{\text{LCHS}})$ resolution.

Two specific prior works are the natural starting points for the kernel choice and for the more interesting test problem; the rest of this spec assumes you have read or will read them. **"Optimal quantum simulation of linear non-unitary dynamics"** (the optimal-kernel paper, the source of the $f_2$ and $f_{i,j}$ contour formulas / analytic-extension construction) provides the mathematical procedure that achieves the optimal $\mathcal{O}(\log(\epsilon_{\text{LCHS}}^{-1}))$ scaling but without a complete and minimal explicit selector circuit. **"Quantum algorithm for the advection-diffusion equation and the Koopman-von Neumann approach to nonlinear dynamical systems"** provides the more interesting test problem (the Koopman-von Neumann lifting of nonlinear dynamics to a linear PDE attacked by LCHS) but without a complete numerical implementation. Your follow-up combines all three: Novikau–Joseph's implementation framework, an optimal-class kernel from the $f_2$ and $f_{i,j}$ analytic-extension family, and (optionally, and rewarded) the KvN test problem.

## The Problem — Here's why it matters

Quantum computers natively implement unitary operators, but many problems of practical interest in plasma physics, fluid mechanics, finance, biology, and condensed-matter physics are **dissipative**: linear PDEs of the form $\partial_t \psi = -A\psi$ with non-Hermitian generator $A$, evolving under the nonunitary $e^{-At}$. Simulating such dynamics on a quantum computer therefore requires a way to encode nonunitary dynamics into a unitary circuit.

The **Linear Combination of Hamiltonian Simulations (LCHS)** algorithm represents the nonunitary propagator as a weighted Fourier integral over Hermitian Hamiltonian evolutions:

$$e^{-At} = \int_{-\infty}^\infty \hat f(k) \, e^{-i(A_H + k A_L) t}\, dk, \qquad A = A_L + i A_H,$$

with $A_L = (A + A^\dagger)/2 \succeq 0$ and $A_H = (A - A^\dagger)/(2i)$. The integration variable $k$ runs over an auxiliary Fourier register; choosing the right kernel $\hat f(k) = \xi(k)/(1-ik)$ controls the precision–cost trade-off. Three asymptotic scaling regimes are known, all measured against precision $\epsilon_{\text{LCHS}}$:

- **Sub-optimal** — $\mathcal{O}(\epsilon^{-1})$ scaling (the original Cauchy-kernel LCHS, Refs. [18, 19] of the paper)
- **Near-optimal** — $\mathcal{O}(\log^{1/\beta}(\epsilon^{-1}))$ scaling (An–Childs–Lin's exact LCHS with the $\beta$-kernel family, Ref. [20]; this is the regime the Novikau–Joseph paper occupies)
- **Optimal** — $\mathcal{O}(\log(\epsilon^{-1}))$ scaling (the approximate-LCHS kernels using analytic-extension techniques, Refs. [24, 25, 33])

Moving from near-optimal to optimal is a kernel-choice question: it is a mathematical change to which $\xi(k)$ weight function is used in the LCHS integral, and a change in how the weight function is computed via QSVT, but not a change to how the integral itself is compiled into a quantum circuit. The Novikau–Joseph implementation framework (FCC-equivalent quadrature via the sinusoidal coordinate transformation + single QSP circuit + QSVT weight construction) is independent of the kernel and works for any kernel in the An–Childs–Lin family or any of the optimal extensions. Note, however, that the QSVT angles do need to be redetermined and reimplemented for any new choice of kernel weight function. Also, if the weight function is real, then this can be used to simplify the implementation of the kernel in the integral formula. The open question is **which kernel, combined with this overall implementation structure, achieves the best end-to-end performance at the resolutions practical for digital emulation**.

**The three-paper triangle this work sits in.** The state of the art splits naturally across three prior works, each of which has one of the three things needed for an optimal end-to-end LCHS implementation but not the other two:

1. **Novikau and Joseph 2025** (the original paper, the subject of this follow-up) provides the **good implementation** — the FCC-equivalent quadrature + single-QSP-circuit selector + QSVT weight construction is the simplest known explicit and minimal LCHS circuit — but uses a kernel that is only near-optimal, and tests on the relatively uninteresting linear advection-diffusion equation.
2. **"Optimal quantum simulation of linear non-unitary dynamics"** provides the **good mathematical procedure** — the $f_2$ and $f_{i,j}$ contour formula and the analytic-extension construction give optimal $\mathcal{O}(\log(\epsilon^{-1}))$ precision scaling — but without a complete and minimal explicit selector circuit (the multiplexed block-encoding construction requires extra ancillary registers and is more complex than Novikau–Joseph's).
3. **"Quantum algorithm for the advection-diffusion equation and the Koopman-von Neumann approach to nonlinear dynamical systems"** provides the **more interesting test** — the Koopman-von Neumann lifting attacks nonlinear dynamics via the LCHS framework with added numerical dissipation — but without a complete numerical implementation that exercises the LCHS scheme end-to-end.

This task asks you to combine (1) and (2): the Novikau–Joseph implementation framework with an optimal-class kernel from the $f_2$ and $f_{i,j}$ analytic-extension family, and (optionally, as the more interesting and more rewarded test) (3): validate on the KvN embedding rather than only on the linear advection-diffusion equation. The author's specific framing is that the optimal implementation is taken from the original paper, and the question is *what is the optimal kernel for that implementation* — and which kernel family wins at which $(n_k, \epsilon_{\text{LCHS}})$ resolution.

## Background — Here's what you need to know about the original paper

The original paper provides the first end-to-end explicit, optimized quantum circuit implementing LCHS, validated on the advection-diffusion equation. Your follow-up builds on its **implementation strategy** and improves its **mathematical procedure**. The technical components of the original method that you will inherit are:

1. **Discretization via sinusoidal coordinate transformation.** The integration variable is mapped to the bounded interval $[-\pi/2, \pi/2]$ via $k = k_{\max} \sin\theta$, with $\theta_j = -\pi/2 + j\,\Delta\theta$ for $j = 0, \ldots, N_k - 1$ and $\Delta\theta = \pi/(N_k - 1)$. This yields $N_k = 2^{n_k}$ discretization nodes. Classically, the integration is **equivalent to Fejér-Clenshaw-Curtis (FCC) quadrature** at Chebyshev extrema; for analytic periodic functions, FCC equals Chebyshev-Gauss-Lobatto quadrature, with exponential convergence for analytic kernels (Lemma 2 of the paper).

2. **Single-QSP-circuit selector.** The discretized LCHS sum
   $$U_{\text{LCHS}} = \sum_{j=0}^{N_k-1} w_j\, V_j(t), \qquad V_j(t) = e^{-i C_j t}, \qquad C_j = A_H + \sin(\theta_j)\, k_{\max} A_L,$$
   is mapped to a single QSP circuit of $N_O$ qubitization operators $O_j$ (each costing $\mathcal{O}(1)$ ancillae), instead of a sequence of $N_k$ controlled time-evolution operators as in earlier LCHS implementations.

3. **Block-encoding** of $C_j$. The $\sin(\theta_j)$ dependence enters via the kernel encoding rather than via $\arcsin$ approximation, eliminating the need for trotterization of $kA_L$ and $A_H$.

4. **Weight construction via QSVT.** The complex weights $w_j = k_{\max} \cos(\theta_j) \Delta\theta\, \xi(k_j) / (1 - i k_j)$ are constructed by a QSVT circuit (the $U_{\sqrt{w}}$ operator in the paper's Fig. 2) followed by amplitude amplification ($O_{\sqrt{w}}^{\text{AA}}$).

5. **Kernel choice.** The paper uses the An–Childs–Lin family (Eq. 5 of the paper):
   $$\xi_{\beta}(k) = \left(2\pi\, e^{-2^\beta}\, \exp([1 + ik]^\beta)\right)^{-1}, \qquad \beta \in (0, 1),$$
   numerically observed to be near-optimal for $\beta \in [0.7, 0.8]$. This kernel exactly satisfies the LCHS contour integral (Eq. 3 of the paper) and gives precision scaling $\epsilon_{\text{LCHS}} = \mathcal{O}(e^{-\mathcal{O}(k_{\max}^\beta)})$, equivalent to $\mathcal{O}(\log^{1/\beta}(\epsilon^{-1}))$ query complexity in precision.

6. **Validation.** End-to-end LCHS-circuit emulations on the QuCF digital emulator on the 1D periodic advection-diffusion equation $\partial_t \psi = -v \partial_x \psi + D \partial_x^2 \psi$ with $N_x = 2^{n_x}$ grid points, parameter ranges $n_x \in \{4, 5, 6, 7, 8\}$, $n_k \in \{4, 5, 6, 7\}$, several time intervals $t$, and several values of $\beta \in [0.5, 1)$.

**The opening the paper acknowledges (Sec. I, after Eq. 5, and the post-submission note around Refs. [24, 25, 33]).** The β-family kernel is **near-optimal but not optimal**. After the paper was submitted, approximate-LCHS kernels using "analytic extension" techniques (Refs. [24, 25, 33]) were derived that achieve **optimal** $\mathcal{O}(\log(\epsilon^{-1}))$ precision scaling. Only Ref. [24] gives an explicit selector for those kernels, but it is a more complicated "multiplexed block-encoding" requiring extra ancillary registers — the original paper explicitly notes that their FCC + single-QSP-circuit selector is simpler. **The implementation gap.** The optimal-kernel papers have the best mathematical procedure but a more expensive implementation. The Novikau–Joseph paper has the best implementation but a near-optimal-only mathematical procedure. Combining the two — Novikau–Joseph's implementation framework with an optimal kernel — has not been studied numerically on a practical benchmark.

**A second underexplored test problem.** The paper's Sec. I notes that LCHS is the natural quantum algorithm for the **Koopman-von Neumann (KvN) embedding** of nonlinear dynamics, in which numerical dissipation (an upwind advection operator or a numerical-diffusion operator) is added to the generator to eliminate artifacts. The KvN test was a primary motivation for the LCHS work but is **not** numerically validated in the paper, which restricts attention to the linear advection-diffusion equation. The simpler upwind-advection problem and the KvN embedding are the natural next benchmarks where the kernel choice would matter most.

**Reference baseline behavior reported in the paper:**

- **Selector circuit cost.** $\mathcal{O}(1)$ ancillae (excluding block-encoding) plus $N_O$ qubitization operators where $N_O = \mathcal{O}(\|At\|)$ for a fixed precision target.
- **Numerical performance on advection-diffusion.** End-to-end relative precision $\epsilon_{\text{LCHS}} \sim 10^{-3}$ to $10^{-4}$ at $n_k \le 7$, success probability $\sim 0.2$.
- **β-family near-optimal behavior.** $\beta \in [0.7, 0.8]$ gives the best observed near-optimal trade-off; lower or higher β gives worse scaling.

## Contract — Here's exactly what counts as success

Your contribution is a **kernel-substitution algorithm**: keep the Novikau–Joseph STMC implementation framework *structure* (single-QSP-circuit selector, FCC-equivalent quadrature via the sinusoidal coordinate transformation, qubitization operators, and the block-encoding interface), substitute an alternative kernel, and demonstrate that the substitution achieves a strictly better asymptotic complexity scaling on at least one of the three standard LCHS resource axes — at matched end-to-end accuracy on a held-fixed benchmark problem, on the QuCF digital emulator. Substituting a kernel is not a paper-only change to $\xi(k)$: the QSVT weight-construction subroutine is kernel-dependent and must be reworked for the substituted kernel (new QSVT angles, plus a weight-selection option in the LCHS code), since the reference build effectively hard-codes the β-family weights. This rework is a required part of the contribution (see "Required: reworking the QSVT weight construction" below).

### Operationalized success thresholds — choose one of three axes

You must commit to exactly one of the three axes as your headline. "Strictly better" is operationalized per axis.

**Axis 1 — Query complexity in precision.** Substitute a kernel that pushes the $A$-oracle query complexity $Q_A(\epsilon)$ from the paper's near-optimal $\mathcal{O}(\log^{1/\beta}(\epsilon^{-1}))$ toward the optimal $\mathcal{O}(\log(\epsilon^{-1}))$ scaling. "Strictly better" means: (a) emulator-measured $Q_A$ as a function of target $\epsilon_{\text{LCHS}}$ exhibits a fitted scaling exponent statistically lower than the Novikau–Joseph β-family baseline at matched $\beta$, with both fits computed from $\geq 4$ precision targets spanning $\epsilon_{\text{LCHS}} \in [10^{-4}, 10^{-1.5}]$; (b) at the headline configuration $(n_x, n_k) = (6, 6)$ and a stated $\epsilon_{\text{LCHS}}$ target, the variant's $Q_A$ is below the baseline's $Q_A$ at matched final-time accuracy; (c) the asymptotic improvement is consistent with the chosen kernel's theoretical scaling (you must derive or cite the theoretical scaling and compare to the empirical fit).

**Axis 2 — Quantum memory (ancilla count).** Substitute a kernel that achieves matched precision with fewer ancillary qubits than the Novikau–Joseph baseline at the same $(n_x, n_k, t, \epsilon_{\text{LCHS}})$. "Strictly better" means: (a) the variant's total ancilla count (excluding the state register, but including the $k$-register of size $n_k = \log_2 N_k$ and all kernel-encoding, weight-construction, and amplitude-amplification ancillae) is strictly less than the baseline's at $\geq 3$ configurations spanning the $(n_x, n_k)$ grid; (b) the reduction holds at $\epsilon_{\text{LCHS}} \leq 10^{-3}$ on the headline configuration; (c) the reduction is not paid for by collapsed success probability — the success probability can only go below the baseline's $\sim 0.2$ if you can prove that it does not get worse with problem size. 

**Axis 3 — Gate complexity at matched precision.** Substitute a kernel that achieves matched precision with strictly fewer gates than the Novikau–Joseph baseline on at least one cost metric (total STMC gate count). "Strictly better" means: (a) the variant's gate count on the chosen metric is strictly below the baseline's on $\geq 3$ configurations; (b) the reduction holds at $\epsilon_{\text{LCHS}} \leq 10^{-3}$ on the headline configuration; (c) the empirical gate-count scaling exponent in $\|At\|$ or in $1/\epsilon_{\text{LCHS}}$ is no worse than the baseline's (you may not improve the prefactor at the cost of asymptotic scaling).

In all three cases:

- The chosen kernel must be a documented **kernel from the optimal-scaling family** (Refs. [24, 25, 33] of the paper or equivalent — analytic-extension kernels, kernels based on analytic approximations to the step function, or other kernels with known $\mathcal{O}(\log(\epsilon^{-1}))$ scaling for an LCHS-style decomposition), OR an alternative kernel from the An–Childs–Lin family at a different β than the paper's near-optimal range $[0.7, 0.8]$, OR a kernel you derive yourself. In all cases, the kernel must satisfy the An–Childs–Lin admissibility conditions (decay, normalization, continuity, lower-half-plane analyticity) for exact LCHS, or the documented approximate-LCHS conditions if you adopt an approximate kernel.
- The Novikau–Joseph **implementation framework** is held fixed (see below). Your contribution lives in the kernel choice and the kernel-resolution characterization, not in the STMC / selector / quadrature pipeline.
- The contribution includes a **kernel-family characterization**: a statement of which kernel family is best at which $(n_k, \epsilon_{\text{LCHS}})$ regime, validated by emulator runs across at least 3 $(n_k, \epsilon_{\text{LCHS}})$ combinations. The paper's headline metric is the asymptotic-scaling improvement on the chosen axis; the characterization is the supporting evidence.
- The contribution includes a **clear explanation of why the chosen kernel is optimal in practice** at the emulated resolutions — not only its asymptotic scaling, but why it wins over the other kernel families you characterize in this implementation framework.

### Required: reworking the QSVT weight construction

Substituting a kernel is not only a change to $\xi(k)$ on paper. The weight-construction subroutine that compiles the LCHS weights into the circuit is kernel-dependent, and in the reference QuCF-OPT-LCHS build the QSVT weights for the β-family kernel are effectively hard-coded. A genuine kernel substitution therefore requires you to:

1. **Find the QSVT angles for each weight function.** For your substituted kernel (and for any kernel families you compare in the characterization), compute the QSVT angles with a separate angle-finding program, and verify that the resulting polynomial reproduces the target weight function to a stated precision before using it.
2. **Add the option to select different weight-function constructions** in the near-optimal LCHS code, so the emulator can run with the substituted kernel's weights. Verify the new weights standalone (they match the target), and then verify that the full LCHS algorithm runs end to end and reproduces the expected dynamics with them.
3. **Run the scaling studies across weight-function families** through this reworked path, so the chosen-axis improvement and the kernel-family characterization are produced by the actual reworked circuit, not by analytical estimates.

QSVT/QSP angle-finding methods are described and cited in the original paper's Sec. 4.3 (computation of the LCHS weights); recent analytic and semi-analytic constructions you may draw on include Bernard and Wiebe, "Analytical Angle-Finding and Series Expansions for Quantum Signal Processing via Orthogonal Polynomial Theory" (arXiv:2605.05321), and the semi-analytic construction in arXiv:2206.02826. You may use any method whose precision you verify.

### Held fixed (parity constraints)

These must remain identical to the Novikau–Joseph paper's setup — they are not part of the design space.

- **The implementation framework structure.** Single-QSP-circuit selector + FCC-equivalent quadrature via the sinusoidal coordinate transformation $k = k_{\max}\sin\theta$ + qubitization operators with $\mathcal{O}(1)$ ancillae (excluding block-encoding) + the block-encoding interface for $A_L, A_H$ defined in the paper. You may not switch to a different quadrature, a different selector design (e.g., the Ref. [24] multiplexed block-encoding), or a sequence-of-controlled-evolutions implementation. The contribution is the kernel inside this framework, not a different framework. **The QSVT weight-construction subroutine is explicitly *not* held fixed:** it is kernel-dependent and must be reworked for your substituted kernel (new QSVT angles, a weight-selection option), as detailed in "Required: reworking the QSVT weight construction" above. What is frozen is the selector, the quadrature, the qubitization operators, and the block-encoding interface — not the weight construction.
- **Block-encoding overhead.** All gate counts and query counts must include the cost of the $A_H, A_L$ block-encodings (including any auxiliary state-preparation, controlled-SWAP, or arithmetic subroutines). You may not externalize the block-encoding to "an oracle" without constructing it.
- **Test problem.** Choose one of two: (a) the paper's 1D periodic advection-diffusion equation $\partial_t \psi = -v\partial_x\psi + D\partial_x^2\psi$ with the paper's $(v, D)$ values and Gaussian initial condition, OR (b) a Koopman-von Neumann embedding of a simple 1D nonlinear dynamical system with added numerical dissipation (upwind advection operator or numerical-diffusion term to eliminate artifacts), with the dissipation scheme stated explicitly. The choice must be committed in `proposal/code/kernel_plan.{py,md,json}` before validation simulations begin. For axis 1 specifically, the advection-diffusion test is recommended because its analytical structure makes the scaling-fit cleaner; the KvN test is permitted and rewarded as the more interesting test problem, but you must still match the baseline's precision on its native test (1D advection-diffusion at the headline configuration) so the comparison is anchored.
- **End-to-end accuracy.** The relative state-vector error $\|\psi^{\text{circuit}}(t) - \psi^{\text{exact}}(t)\| / \|\psi^{\text{exact}}(t)\|$ measured on the QuCF emulator must be no worse than the Novikau–Joseph circuit at the same $(n_x, n_k, t, \epsilon_{\text{LCHS}})$ target on the headline configuration. You may match or beat the baseline's accuracy; you may not regress it.
- **Quantum-circuit emulation.** All final accuracy, gate-count, query-count, and ancilla-count claims must come from running the actual circuit on the QuCF digital emulator, not from analytical estimates alone. Analytical complexity bounds motivate the kernel choice and explain the scaling; the QuCF emulator runs verify the bound on the practical configurations.
- **Success probability.** Must be at least as high as the Novikau–Joseph baseline ($\sim 0.2$ for the headline configuration), with one exception: on **Axis 2** (quantum memory) the success probability may fall below $\sim 0.2$ if you prove it does not get worse with problem size. In all cases, a reduced-cost circuit that recovers probability through additional amplitude-amplification rounds must include the AA cost in the headline gate-count metric; a circuit with collapsed success probability — below baseline with no scale-stability proof, or with the AA cost hidden — does not satisfy the contract.

### What you may design

- **The kernel** $\xi(k)$. The mathematical core of your contribution. Choices include kernels from the optimal-scaling family (analytic-extension kernels of Refs. [24, 25, 33] of the paper or equivalent), kernels from the An–Childs–Lin β-family at β values outside the paper's tested range, or kernels you derive. The kernel must come with a stated theoretical scaling on the chosen axis.
- **The kernel hyperparameters.** $k_{\max}$, any kernel-specific parameters (β, the analytic-extension contour, the step-function approximation parameters), the QSVT polynomial degree used to approximate the kernel-encoded weight function. State all hyperparameters before running validation.
- **The kernel-family characterization sweep.** Which $(n_k, \epsilon_{\text{LCHS}})$ configurations you evaluate to show which family is best where. State the sweep grid in `kernel_plan.*` before running.
- **The test problem within the two stated options.** Advection-diffusion (clean comparison to paper) or KvN with numerical dissipation (more interesting test).
- **The QSVT weight construction for your kernel.** The QSVT angles and the weight-function-selection option in the LCHS code, re-derived for your substituted kernel. This is required (see "Required: reworking the QSVT weight construction"); what is held fixed is the framework *structure*, not the kernel-dependent weight construction.

### Anti-cheating

The contract is about the kernel choice *and* its honest verification on the digital emulator. The following moves are not allowed:

- **Switching axes after the fact.** You must commit to axis 1, 2, or 3 before validation runs. Finding the bound is loose on axis 1, then re-labeling the work as axis 3, is not allowed. Abandonment of an axis must be logged in `proposal/attempts_log.md`.
- **Kernel restatement after validation.** The kernel's mathematical form (the function $\xi(k)$) must be stated in `kernel_plan.*` before validation runs. Refining hyperparameters (β, $k_{\max}$, polynomial degree) after seeing validation results is allowed only if disclosed in the paper.
- **Cherry-picking configurations.** The sweep grid for the kernel-family characterization and the precision-scaling fit must be stated up front. Reporting only the configurations where the variant looks better, while running but not reporting configurations where it does not, is selective reporting.
- **Implementation drift.** Changing the Novikau–Joseph implementation framework *structure* (different quadrature, different selector design, different block-encoding) to mask kernel deficiencies, and presenting the result as a "kernel improvement," is implementation drift and violates the held-fixed framework constraint. Reworking the QSVT weight-construction subroutine to support your substituted kernel (new angles, a weight-selection option) is *required*, not drift — drift is changing the selector, quadrature, qubitization, or block-encoding structure.
- **Externalizing block-encoding cost.** Counting only the $N_O$ qubitization operators while pretending the per-operator block-encoding of $A_H, A_L$ is free, in order to inflate the apparent improvement, is dishonest accounting.
- **Validation only by re-evaluating the LCHS sum classically.** Computing $U_{\text{LCHS}}$ as a numerical contour integral on the CPU and comparing to your kernel's classical contour integral is circular — both share the same kernel and the same discretization. The QuCF emulator-measured relative state-vector error is the independent witness.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated along two dimensions:

1. **Performance on stated metrics.** The kernel is mathematically valid (satisfies the relevant admissibility conditions), the chosen scaling axis is rigorously demonstrated (statistically lower scaling exponent, or strictly fewer ancillae / gates) at matched end-to-end accuracy across the announced configurations, and the kernel-family characterization holds on the announced $(n_k, \epsilon_{\text{LCHS}})$ sweep.

2. **Scientific merit.** Your paper is assessed on the novelty of the kernel choice (or the rigor of its theoretical justification if convergent with published work), the depth of the kernel-family characterization, the soundness of the mechanism analysis (why this kernel achieves the scaling it does in this implementation framework), and the engagement with the published LCHS / approximate-LCHS / analytic-extension literature.

### Paper quality

Your submission is an academic paper, not a class project report. The writing should resemble what would be reviewed at a top scientific journal in structure, integrity, depth, and language.

Do not frame the paper as "a follow-up to Novikau and Joseph." Frame it as a self-contained scientific contribution that contains its own motivation, methods, results, and discussion, with the original paper cited where appropriate. The paper's title should describe the scientific contribution (the kernel substitution and its scaling improvement, or the kernel-family characterization) rather than the structure of investigation.

**Independence from this task spec.** A reader of your paper who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. Do not use phrases like "axis 1/2/3," "the kernel-substitution direction," "Novikau–Joseph baseline" verbatim where physics-natural terminology would do. Section structure should be physics/algorithm-driven, not contract-driven.

**Rendered output verification.** Before submitting, render your LaTeX to PDF (e.g., `pdflatex report.tex`) and visually inspect the result. Equations must not overflow page margins; tables must fit within the text width; figures must not be cut off.

**Literature engagement.** Engage with the published LCHS and approximate-LCHS literature, including but not limited to: Refs. [18, 19, 20] for the original LCHS chain; Refs. [24, 25, 33] for the optimal-kernel approximate-LCHS work; the analytic-extension and step-function-approximation literature from which the optimal kernels are derived; the Koopman-von Neumann quantum-algorithm literature if your test is KvN. The LCHS literature has continued to develop since the original Novikau–Joseph paper appeared; a paper that cites only Novikau–Joseph as prior work indicates shallow engagement and will be flagged.

## Rules — Here are the rules

### Available resources

All necessary code, data, and the original paper are provided locally in the working directory.

- `paper.md` — markdown version of the original paper (Novikau & Joseph, 2025), including the LCHS framework, the FCC quadrature derivation, the explicit selector circuit, the Theorem 4 complexity proof, and the advection-diffusion validation.
- `paper.pdf` — original paper PDF.
- `images/` — figures extracted from the original paper (LCHS weights for different β, LCU circuit, selector QSP circuit, qubitization operators, advection-diffusion validation plots).
- `code/QuCF/` — QuCF, the Quantum Computer Framework digital emulator. Includes build directories for the emulator and its supporting components, core emulator source, example simulation drivers, unit tests, and dependencies.
- `code/QuCF-OPT-LCHS/` — QuCF specialized for the paper's optimized LCHS implementation. Same directory layout as `code/QuCF/` but with the LCHS-specific selector circuit and FCC-quadrature driver. **Use this as the reference implementation of the Novikau–Joseph baseline.**
- `code/QuCF.wiki/` — the QuCF wiki documentation: tutorial scripts, API references, simulation-input formats.
- `data/ade_opt_simulations/ADE-OPT/` — outputs of the paper's advection-diffusion LCHS simulations.

For QSVT angle generation, classical exact-propagation reference, kernel analytical work, and analysis of emulator outputs, standard Python tooling is available (NumPy, SciPy, SymPy, mpmath, matplotlib). For optional kernel-derivation work, additional symbolic computation tools (Mathematica via WolframScript, if available) may be useful. QSVT/QSP angle-finding methods are described and cited in the original paper's Sec. 4.3 (computation of the LCHS weights); recent analytic and semi-analytic constructions you may draw on include Bernard and Wiebe (arXiv:2605.05321) and arXiv:2206.02826.

### Compute budget

- **Hardware:** a single GPU (the QuCF emulator is GPU-accelerated), with a multicore CPU host and ample system memory and disk. A single GPU is at the limit of what these emulations can do, so each problem instance must be made to fit on one GPU.
- **Scope guidance:** Use the smallest $(n_x, n_k)$ that demonstrates the kernel-substitution improvement; do not waste budget on the largest emulations unless they exercise something the smaller ones do not. State-vector cost grows as $2^{n_x + n_k}$, so eliminate unnecessary ancillae and keep the $k$-register and weight-construction footprint minimal to keep each instance within a single GPU's memory.

### On wrapping up near the time limit

If you find yourself near the time limit with incomplete work, stop running new emulations and write up what you have. A kernel choice validated at 3 sweep points instead of the planned 5 is incomplete, not fabricated — report it that way. Populate `proposal/attempts_log.md` with what you tried, what didn't work, and why you abandoned approaches.

### Autonomy and external resources

**Internet search is permitted and encouraged.** Use it for literature lookup on the LCHS, approximate-LCHS, analytic-extension, and Koopman-von Neumann quantum-algorithm literature; documentation of QuCF, NumPy, SciPy, SymPy, mpmath; citations of post-2025 LCHS follow-up work. The optimal-kernel literature is the natural place to source an alternative kernel — read Refs. [24, 25, 33] of the paper and any follow-ups. Cite anything you use.

**Workspace constraint.** You may not read local files outside the provided working directory.

**No questions.** Make your own decisions and justify them in your write-up.

### No reward hacking

The following checks will be applied to your submission. They are disclosed up front so the contract is clear.

1. **Number traceability.** Every quantitative claim in your paper (kernel hyperparameters, scaling fits, gate counts, query counts, ancilla counts, relative errors, success probabilities) must trace to a log entry with the same value. Numbers without a verifiable source will be flagged.

2. **Pre-stated kernel and consistent reporting.** The kernel's mathematical form ($\xi(k)$ as a function), the chosen axis (1/2/3), and the sweep grid must be stated in `proposal/code/kernel_plan.{py,md,json}` (with timestamp) before validation simulations are logged. Honest disclosure of hyperparameter refinement (β, $k_{\max}$, polynomial degree) after validation is allowed if explicitly stated in the paper; silent restatement is not.

3. **Test-set / evaluation integrity.** Final relative error, gate counts, query counts, and ancilla counts must come from runs on the QuCF emulator at the announced configurations, not from analytical estimates or post-hoc interpolation. The sweep configurations must match what is declared in `kernel_plan.*`.

4. **Baseline honesty.** Novikau–Joseph baseline numbers must come from either (a) the original paper with specific table/figure citations, (b) the authors' `code/QuCF-OPT-LCHS/` implementation when used directly with logged outputs, or (c) your own logged baseline emulations. Mixing the three without distinction is selective framing and will be flagged.

5. **No selective reporting.** If multiple kernel-hyperparameter choices were tried (different β, different $k_{\max}$, different polynomial degrees, different analytic-extension contours), report all of them in the attempts log or use principled aggregation in the main results. Cherry-picking the configuration where the variant looks better while hiding configurations where it does not is selective reporting.

6. **No hardcoded reference values.** Your code must not contain hardcoded final states, hardcoded gate counts, hardcoded query counts, or other ground-truth literals that bypass actual emulator execution. The exact-evolution reference for the advection-diffusion (or KvN) test must be computed at runtime via classical propagation.

7. **Experiment completeness.** Every experiment described in the paper must have corresponding logs (circuit-construction scripts, QSP angles, emulator inputs, emulator outputs, gate-count tallies). Suspicious gaps will be flagged.

8. **Code-paper alignment.** The Methodology section must accurately describe what the code actually does — the kernel function form, kernel hyperparameters, QSP angle decomposition, block-encoding structure, ancilla accounting — and these must match what `proposal/code/` implements.

9. **Quantum-circuit-fidelity integrity.** All gate-count, query-count, ancilla-count, and accuracy claims must come from actually executing the proposed circuit on the QuCF emulator. The variational state must be evolved through the parametrized circuit at every time step, not directly substituted by exact classical evolution. The block-encoding cost of $A_L$ and $A_H$ must be counted within all reported metrics — claiming low total cost by externalizing the block-encoding to "an oracle" without constructing it will be flagged. Validation by re-evaluating the LCHS integral classically (without an emulator run) is circular and not accepted.

10. **No axis switching after the fact.** The chosen axis (1, 2, or 3) must be committed in `proposal/code/kernel_plan.*` before validation simulations are logged. Abandoned axes must be logged in `proposal/attempts_log.md`.

11. **Implementation-framework integrity.** The implementation framework *structure* — FCC-equivalent quadrature via the sinusoidal coordinate transformation, single-QSP-circuit selector with $\mathcal{O}(1)$ ancillae (excluding block-encoding), qubitization operators, and the block-encoding interface — is held fixed. The QSVT weight-construction subroutine is *not* part of this held-fixed structure: it is kernel-dependent and is expected to be reworked for the substituted kernel (new angles, a weight-selection option). Changing the selector, quadrature, qubitization, or block-encoding structure to mask kernel deficiencies, and presenting the result as a kernel improvement, will be flagged.

12. **Codebase inspection evidence.** `proposal/codebase_notes.md` must document what was inspected in `code/QuCF/`, `code/QuCF-OPT-LCHS/`, and the QuCF wiki, what was reused, what was reimplemented or extended for the kernel substitution. An empty or trivial file paired with a from-scratch reimplementation of components QuCF provides will be flagged.

13. **No substantive replication of published kernels.** The optimal-kernel literature (Refs. [24, 25, 33] of the paper and any follow-ups) has already proposed specific kernels. If your kernel is materially equivalent to one of these published kernels, the prior work must be cited explicitly and the contribution must be differentiated. A bound that substantively replicates a published kernel's contribution without acknowledgment is flagged. Convergent ideas openly acknowledged with citations are fine; convergent ideas claimed as novel are not.

14. **Effort evidence when partial work is claimed.** If the paper describes any contract item as "attempted but failed" or "partial," `proposal/attempts_log.md` must contain at least two distinct documented attempts with reasoning.

15. **Held-fixed disclosure.** Any deviation from the items in §"Held fixed (parity constraints)" — the implementation framework, the block-encoding overhead accounting, the test problem (advection-diffusion or KvN-with-disclosed-dissipation), the end-to-end accuracy parity, the success-probability floor — must be transparently disclosed in Methodology or Experimental Setup. Silent or misrepresented deviation will be flagged. Silent deviation paired with a paper claim of compliance is a deception by misrepresentation.

16. **No undisclosed external code.** All non-trivial methodology code in `proposal/code/` must either be authored by you during this run, or be from `code/QuCF/`, `code/QuCF-OPT-LCHS/`, the QuCF wiki examples, or a standard scientific Python library (NumPy, SciPy, SymPy, mpmath, matplotlib, Qiskit, Cirq, PennyLane, OpenFermion). Copying or adapting external implementations of QSP angle generation, kernel weight construction, or LCHS selector circuits without acknowledgment in `codebase_notes.md` will be flagged.

## Deliverables — Here's what to hand in

1. **Codebase and logs.** Code organized in `proposal/code/`, with logs in `proposal/logs/`. Required:

   - **Kernel plan:** `proposal/code/kernel_plan.{py,md,json}` stating the chosen axis (1/2/3), the kernel's mathematical form ($\xi(k)$ as a function, with all parameters explicit), the kernel-family being claimed (optimal-class, β-family at extreme β, or other), the chosen test problem (advection-diffusion or KvN-with-dissipation), and the validation sweep grid. Created **before** validation emulations are run.
   - **Kernel derivation artifacts:** `proposal/code/kernel_derivation/` containing the analytical work that motivates the kernel — derivation of the scaling (or citation to a paper that derives it), verification that the kernel satisfies the An–Childs–Lin admissibility conditions (or documented approximate-LCHS conditions), construction of the discretized weights $w_j$ for the chosen kernel, QSVT-angle generation for the kernel-encoded weight function with intermediate validation that the polynomial approximation matches the target within stated tolerance, the weight-function-selection option added to the LCHS code, and verification that the new weights work standalone (matching the target) and that the full LCHS algorithm runs end to end with them.
   - **Baseline reproduction:** Novikau–Joseph β-family runs at the headline configuration and at the precision-sweep configurations, with circuit-construction inputs, QSP angles, emulator outputs, gate-count tallies, success-probability traces logged.
   - **Variant runs:** your kernel-substituted variant at the same configurations as the baseline, with the same logging structure.
   - **Kernel-family characterization sweep:** runs at $\geq 3$ $(n_k, \epsilon_{\text{LCHS}})$ combinations for both the baseline and the variant, illuminating which kernel family is best at which resolution. For axis 1 specifically, the precision-scaling sweep at $\geq 4$ $\epsilon_{\text{LCHS}}$ targets at the headline $(n_x, n_k)$.
   - **Comparison artifacts:** side-by-side gate-count, query-count, ancilla-count, and accuracy tables for the headline + characterization configurations, with the improvement on the chosen axis explicitly computed; for axis 1, the fitted scaling exponents and the underlying data.
   - **Hyperparameter-selection runs:** any preliminary runs used to tune $\beta$, $k_{\max}$, $N_k$, the QSP polynomial degree, or kernel-specific parameters, with what was tried and how the final values were selected.
   - **`proposal/codebase_notes.md`:** documentation of what was inspected in `code/QuCF/`, `code/QuCF-OPT-LCHS/`, and the wiki, what was reused, and what was reimplemented or extended.
   - **`proposal/attempts_log.md`:** substantive documentation of distinct approaches tried, including kernels considered and rejected, abandoned axes, sweep configurations that didn't work.

   Every quantitative claim made in your paper must be reproducible from a corresponding log entry.

2. **Follow-up paper.** A LaTeX academic paper at `proposal/report.tex`, compiled to `proposal/report.pdf`, using `neurips.sty` (provided in the workspace). The paper must include:

   - **Motivation** for kernel substitution within the Novikau–Joseph implementation framework, and why moving from near-optimal to optimal scaling matters in the practical-emulation regime.
   - **A description of the proposed kernel:** the function $\xi(k)$, its parameters, its theoretical scaling, its admissibility (An–Childs–Lin conditions or approximate-LCHS analogues), and its construction via discretization on the FCC nodes.
   - **A description of the experimental setup:** the test problem (advection-diffusion or KvN with dissipation), the configurations $(n_x, n_k, t, \epsilon_{\text{LCHS}})$ evaluated, the QuCF emulator settings, the baseline reproduction summary.
   - **A primary results table** labeled `\label{tab:main_results}` with the structure below:

     | Method | Test | $(n_x, n_k, t, \epsilon_{\text{LCHS}})$ | Final rel. error | STMC | $Q_A$ | Ancillae | Success prob. |
     |--------|------|-----------------------------------------|-------------------|------|-------|----------|----------------|
     | Novikau–Joseph (baseline) | (your test) | (headline) | | | | | |
     | Your method | (your test) | (headline) | | | | | |
     | Novikau–Joseph (sweep point 1) | | | | | | | |
     | Your method (sweep point 1) | | | | | | | |
     | (additional rows) | | | | | | | |

     Add additional rows for the characterization sweep (≥3 $(n_k, \epsilon_{\text{LCHS}})$ combinations). The "Method" column may use the kernel-substituted variant's name rather than "Your method." Gate counts are reported as **STMC gate counts**: the QuCF / QuCF-OPT-LCHS build compiles circuits to the STMC gate set, so STMC is the gate-complexity metric for this task. Finer decompositions (CNOT, T, Toffoli counts) are out of scope here and are not required in the table.

   - **Axis-specific scaling/reduction analysis** appropriate to the chosen axis:
     - Axis 1: a precision-scaling plot of $Q_A$ versus $\log(\epsilon_{\text{LCHS}}^{-1})$ at the headline $(n_x, n_k)$, with the baseline and the variant overlaid, fitted scaling exponents, and the comparison to the asymptotic prediction.
     - Axis 2: an ancilla-count table or plot across the $(n_x, n_k)$ sweep, with the per-configuration reduction stated.
     - Axis 3: a gate-count vs. $\|At\|$ or vs. $\log(\epsilon^{-1})$ plot showing strictly fewer gates at matched precision, with both prefactor and exponent compared.

   - **The kernel-family characterization plot.** Which kernel family is best at which $(n_k, \epsilon_{\text{LCHS}})$ regime, supported by the characterization-sweep emulator runs.

   - **A mechanism discussion:** why the chosen kernel produces the scaling it does within the Novikau–Joseph implementation framework. Connect the kernel's analytical structure (decay rate, contour behavior, polynomial-degree dependence) to the empirically observed gate/query/ancilla scaling. State explicitly **why the chosen kernel is the optimal choice in practice** at the emulated resolutions — why it wins over the other kernel families you characterized in this framework, not only asymptotically.

   - **A failure analysis:** configurations where the variant does not improve, regimes where the kernel-family characterization changes (e.g., crossover points where a different family wins), kernel parameter choices that were tried and rejected.

   - **A computational-cost discussion:** classical-side cost of the kernel-substitution work (QSVT angle generation, kernel discretization on FCC nodes) vs. the baseline's classical-side cost, with explicit numbers.

   - **Engagement with related work.** Position the kernel against the published LCHS / approximate-LCHS / analytic-extension literature, including Refs. [18, 19, 20, 24, 25, 33] of the paper and any post-2025 follow-ups identified via literature search.

   - **Limitations and conclusions.**

3. **Supporting documents:** `proposal/codebase_notes.md` and `proposal/attempts_log.md` as described above.

Good luck.
