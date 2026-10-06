# Task: Develop a Substantive Follow-up to "Adaptive Variational Quantum Dynamics Simulations"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the AVQDS (Adaptive Variational Quantum Dynamics Simulations) method by Yao et al., "Adaptive variational quantum dynamics simulations," *PRX Quantum* 2, 030307 (2021). Your contribution is to **reformulate AVQDS for a fixed qubit topology, so that all generated two-qubit gates respect a specified nearest-neighbor connectivity, while preserving accuracy relative to the unconstrained baseline and quantifying the topology-induced overhead.**

## The Problem — Here's why it matters

Simulating quantum dynamics — computing $|\psi(t)\rangle = e^{-i\hat{H}t}|\psi(0)\rangle$ — is a fundamental task that variational quantum methods address by building a parametrized circuit ansatz that evolves under McLachlan's variational principle. AVQDS adapts the ansatz during the simulation, growing the circuit as needed to maintain accuracy, which yields dramatic reductions in CNOT count compared to first-order Trotterization at matched fidelity.

The unconstrained AVQDS algorithm, however, generates two-qubit operators between arbitrary qubit pairs. On real quantum hardware — whether IBM's heavy-hex superconducting platforms or square-lattice superconducting devices — most pairs of qubits are not physically connected, and two-qubit gates between non-neighbors must be implemented via SWAP networks that add depth and gate count. The mismatch between the algorithm's logical circuit and the device's physical connectivity is a central engineering obstacle to running these methods on near-term hardware.

The scientific question this follow-up addresses is whether AVQDS can be reformulated to respect a fixed qubit topology directly — choosing operators, ordering rotations, and managing ansatz growth so that every two-qubit gate lands on a connected pair — and whether such a topology-aware variant preserves the accuracy and circuit-cost advantages of the unconstrained method, or whether the connectivity constraint imposes a substantial penalty.

## Background — Here's what you need to know about the original paper

Read `paper.md` in full before beginning. The original paper introduces AVQDS with the following components:

- **Pseudo-Trotter ansatz.** The variational state is built from an initial state $|\psi_0\rangle$ via an ordered product of Pauli rotation gates $e^{-i\theta_\mu \hat{A}_\mu}$, where each $\hat{A}_\mu$ is a Pauli string from the operator pool. New gates are initialized with $\theta = 0$ so that appending them preserves the current state while expanding the variational manifold.

- **McLachlan's variational principle.** Parameters evolve according to $\sum_\nu M_{\mu\nu} \dot{\theta}_\nu = V_\mu$, with $M$ and $V$ defined in terms of the density matrix. **Tikhonov regularization** $M \rightarrow M + \xi I$ stabilizes the inversion (the AVQDS(T) follow-up uses $\xi = 10^{-6}$; see `paper.md` for exact values).

- **Adaptive time stepping.** The step $\delta t$ is adjusted so that $\max_\mu |\delta\theta_\mu| \leq \delta\theta_{\max}$ (follow-up paper uses $\delta\theta_{\max} = 0.005$). Forward Euler is used for the parameter update.

- **Adaptive ansatz growth.** At each step, the McLachlan distance $L^2$ is computed. If $L^2 \geq L^2_{\text{cut}}$ (the follow-up paper uses $L^2_{\text{cut}} = 10^{-3}$), candidate operators from the pool are trial-appended and the one producing the smallest $L^2$ is permanently added. The pool is iterated until $L^2 < L^2_{\text{cut}}$. The ansatz only grows; there is no gate-removal step.

- **Operator pool.** The standard pool is the set of non-identity Pauli strings appearing in the Hamiltonian (the "Hamiltonian pool"). For MFIM at $h_z = 0$ this is 16 operators ($Z_iZ_{i+1}$ on all bonds, $X_i$ on all sites); for $h_z = 0.5$ it is 24 operators (adding $Z_i$). An expanded pool with additional Pauli strings is explored for some MFIM simulations.

**Reference baseline behavior:** The paper reports fidelity $\geq 99.5\%$ for MFIM at $T = 3$, with $N_{\text{cx}} = 134$ for the integrable case ($h_z = 0$) and $N_{\text{cx}} = 210$ for the nonintegrable case ($h_z = 0.5$). AVQDS uses up to two orders of magnitude fewer CNOT gates than first-order Trotterization at comparable fidelity.

**CNOT counting convention in the original paper.** The paper assumes full all-to-all connectivity in its CNOT counts. Specifically, each two-qubit Pauli rotation $e^{-i\theta \hat{A}_{ij}}$ is compiled to **2 CNOTs** plus single-qubit gates (the standard Pauli rotation decomposition). The CNOT counts of 134 (MFIM $h_z=0$), 210 (MFIM $h_z=0.5$), and 100 (LSM $N=8$) are *logical* CNOT counts under this convention. The baseline against which your topology-aware variant's overhead is measured is this logical CNOT count from the unconstrained AVQDS run.

**Connectivity considerations.** The Hamiltonian-pool operators are all 2-local Pauli strings between specific qubit pairs (bond-aligned for $Z_iZ_{i+1}$, site-aligned for single-site terms). For LSM (open chain), the two-qubit operators in the Hamiltonian pool are nearest-neighbor in 1D. For MFIM (periodic ring), they form a ring of 8 bonds including the periodic $Z_{N-1}Z_0$. **On a fixed qubit topology, whether these operators remain nearest-neighbor depends on how the 8 spin sites are mapped onto the topology's qubits.** A 2D square lattice may admit a Hamiltonian path or cycle that puts every Hamiltonian-pool bond on a topology edge; sparser topologies typically do not. Beyond the Hamiltonian pool, AVQDS can also use an *expanded* operator pool that includes additional 2-site Pauli terms (the paper investigated this for some MFIM simulations and found it generally produces longer circuits without accuracy gain) — these expanded-pool operators are not constrained to Hamiltonian bonds and so are not naturally nearest-neighbor on any topology.

**Topology-aware framing for this task.** This follow-up specializes AVQDS for a fixed qubit topology — your variant must produce circuits where every two-qubit gate connects topology-adjacent qubits. Crucially, the Hamiltonians simulated remain the *original AVQDS benchmarks* (1D MFIM ring with PBC, 1D LSM chain with OBC) — **not** Hamiltonians whose interaction graph matches the topology's qubit-connectivity graph. The topology-awareness exercise becomes nontrivial precisely because the Hamiltonian's interaction graph (a ring or a chain) differs from the qubit-connectivity graph (a 2×4 lattice or an 8-qubit heavy-hex subgraph). On topologies that admit a Hamiltonian cycle covering all 8 vertices, a clever embedding can make every Hamiltonian-pool bond nearest-neighbor with zero SWAP overhead. On topologies that admit no Hamiltonian path, the Hamiltonian-pool operators cannot all be placed on topology edges regardless of mapping, and routing overhead is unavoidable. Both regimes appear in this task — see the Contract.

## Contract — Here's exactly what counts as success

You will design a **topology-aware variant of AVQDS** that satisfies the following:

1. **Inspect the provided AVQDS implementation and reuse it where applicable.** Before writing your own code, inspect `code/AVQDS/`. Identify which components are directly usable (the McLachlan EOM solver, the operator-pool trial mechanism, the ansatz-growth logic, the Tikhonov regularization, the adaptive time stepping). Reimplementing from scratch what the authors' code already provides is wasted effort and a source of comparison-quality risk. Record what you found and what you reused in `proposal/codebase_notes.md`.

2. **Sanity-check unconstrained AVQDS using the authors' code.** Run AVQDS on the four benchmark configurations (MFIM at $h_z = 0$ and $h_z = 0.5$, LSM with $h_z = -0.7$ and $h_z = 1.6$) with the parameter values specified in the Evaluation Protocol. Confirm that fidelity $\geq 99.5\%$ is achieved on MFIM and that the LSM benchmarks show qualitative consistency with the paper's results. Exact numerical match to the paper's figures is not expected — different operator-pool tie-breaking and floating-point order can produce different CNOT counts. If the unconstrained baseline does not achieve $\geq 99.5\%$ fidelity on MFIM after debugging, your topology-aware variant cannot be validly compared to it; resolve the configuration before proceeding.

3. **State your topology-aware variant in the Methodology before describing any empirical results.** Include the motivation, the mathematical form, the strategy for ensuring topology compliance (operator-pool restriction, dynamic SWAP insertion, qubit re-labeling, operator decomposition, or whatever combination supports your approach), any hyperparameters, and a clear positioning of the contribution against the literature on hardware-aware quantum compilation, qubit routing, and connectivity-restricted variational methods. Qualitative explanations ("we restrict the operator pool to nearest neighbors") do not satisfy this principle — a domain expert in quantum compilation should be able to reimplement your method from the Methodology alone. Where possible, implement the new variant as a modification of `code/AVQDS/`, so the only difference between baseline and new method is the topology-aware logic.

4. **Two qubit topologies are held fixed for this task: a 2×4 square lattice and an 8-qubit heavy-hex subgraph.** Your variant must produce topology-compliant circuits on *both*; both appear as required gates in the evaluation, and overhead is reported separately for each.

   **Topology A — 2×4 square lattice** (10 edges):
   ```
   Layout:  0 — 1 — 2 — 3
            |   |   |   |
            4 — 5 — 6 — 7

   Edges = {(0,1), (1,2), (2,3),       horizontal, top row
            (4,5), (5,6), (6,7),       horizontal, bottom row
            (0,4), (1,5), (2,6), (3,7)} vertical
   ```

   **Topology B — 8-qubit heavy-hex subgraph** (7 edges):
   ```
   Layout:  0 — 1 — 2
                |
                3
                |
            4 — 5 — 6 — 7

   Edges = {(0,1), (1,2), (1,3),
            (3,5),
            (4,5), (5,6), (6,7)}
   ```

   The two topologies probe meaningfully different regimes. Topology A admits a Hamiltonian cycle (0–1–2–3–7–6–5–4–0), so the MFIM PBC ring (8 ZZ bonds) and the LSM open chain (7 nearest-neighbor XX/YY bonds) can both be embedded such that every Hamiltonian-pool two-qubit operator is nearest-neighbor; on Topology A the routing problem is dominated by choices the variant makes about operator-pool selection, embedding, and (if expanded-pool operators are admitted) ansatz-growth ordering. Topology B has no Hamiltonian path on 8 vertices: four of its vertices (0, 2, 4, 7) have degree 1, and a Hamiltonian path on $n$ vertices can have at most two such endpoints. Pigeonhole then forces the MFIM ring (8 bonds against 7 topology edges) to leave at least one bond off-topology even with the best embedding; the LSM chain (7 bonds, matching 7 edges in count) still cannot be embedded because no Hamiltonian path is available to host it. Both benchmarks on Topology B therefore require SWAP routing regardless of mapping — this is the regime where the routing problem actually has bite, and it is where the variant's design decisions matter most.

   Every two-qubit gate in your variational circuit on each topology must connect a pair of qubits in that topology's edge set. SWAP gates are permitted as long as they themselves connect adjacent qubits in the topology, but they count toward the total gate-count and CNOT-count overheads (a SWAP gate decomposes to 3 CNOTs). Any two-qubit operation that does not respect the topology's edge set is a contract violation. Logs must demonstrate edge-set compliance separately for each topology.

5. **Run your topology-aware variant on all four benchmark configurations and both topologies.** MFIM ($h_z = 0$ and $h_z = 0.5$) and LSM ($h_z = -0.7$ and $h_z = 1.6$), each evaluated on Topology A (2×4 square) and Topology B (heavy-hex) — 8 (benchmark × topology) configurations in total. Report final-time fidelity, total CNOT count $N_{\text{cx}}$ (logical CNOTs under the all-to-all convention, plus any SWAP-introduced CNOTs), circuit depth, and number of variational parameters $N_\theta$ for each.

   The headline claim is that your variant achieves **accuracy preservation** (final-time fidelity within 1.0% absolute of the unconstrained AVQDS baseline on each (benchmark × topology) configuration) with **quantified overhead** in depth and CNOT count, separately reported per topology.

   "Accuracy preservation" is operationalized: if unconstrained AVQDS achieves fidelity $f_{\text{base}}$ on a given benchmark, your topology-aware variant must achieve at least $f_{\text{base}} - 0.01$ (i.e., 1% absolute drop or better) on both topologies. In noiseless simulation, SWAP gates are exact unitaries, so post-hoc routing strategies preserve fidelity to machine precision; the tolerance is meaningful only when your variant restricts the operator pool or otherwise changes ansatz-growth decisions in ways that affect the final state. Sacrificing accuracy below the 1% floor does not satisfy the contract on either topology.

   "Quantified overhead" is the differentiator. Report the depth multiplier ($\text{depth}_{\text{variant}} / \text{depth}_{\text{unconstrained}}$) and the CNOT-count multiplier on each benchmark, separately for Topology A and Topology B. Lower overhead is better. A **smart-embedding naive baseline** (unconstrained AVQDS run with an optimal initial qubit-to-topology mapping that maximizes nearest-neighbor coverage of Hamiltonian-pool operators, followed by post-hoc shortest-path SWAP insertion for any remaining non-local two-qubit gates) sets the reference for what good routing without ansatz adaptation looks like. The naive baseline must use a *smart* mapping, not an arbitrary one — on Topology A with the Hamiltonian pool, this means using the Hamiltonian-cycle embedding so that the trivial overhead is genuinely low; on Topology B, the best available embedding for each benchmark. Your variant should outperform this smart-embedding naive baseline by a meaningful margin to count as a substantive contribution, *especially* on Topology B where the unavoidable routing makes the methodological problem real.

6. **Provide a mechanism analysis.** Explain *why* your topology-aware variant achieves the overhead it does — through operator-pool design decisions, ansatz-growth strategy, dynamic routing, or whatever combination supports the claim. The analysis should isolate where the overhead reduction (or its absence) comes from relative to the naive SWAP-routing baseline.

7. **Provide a failure analysis.** Identify benchmarks, parameter regimes, or system features where your variant has higher overhead, lower accuracy, or other shortcomings. Methods that beat the naive routing baseline everywhere with no failure modes are uncommon; honestly characterizing where your method underperforms is part of the scientific contribution.

8. **Provide a computational cost analysis.** Report the wall-clock cost of your variant relative to unconstrained AVQDS, and discuss whether the topology-aware logic introduces non-trivial overhead in the classical optimization step (e.g., the trial-expansion loop iterating over operator-pool subsets restricted to the topology).

You should aim to fully implement your method, run all four benchmark configurations, and complete the analyses.

**Test functions and evaluation protocol.** See the Evaluation Protocol section below for the exact MFIM and LSM specifications. These benchmarks are fixed; you may not modify the Hamiltonians, system size, initial states, simulation times, or qubit ordering used for the comparison.

**Held fixed.** The 8-qubit MFIM and LSM systems with the parameters in the Evaluation Protocol; the initial states; the total simulation times; the two qubit topologies (2×4 square and 8-qubit heavy-hex with the edge sets specified above); the McLachlan variational principle and adaptive-ansatz-growth framework (the core algorithmic idea — your contribution is the topology-aware variant of this framework, not a replacement); statevector simulation with no noise model; the all-to-all CNOT-counting convention for unconstrained AVQDS (2 CNOTs per two-qubit Pauli rotation), against which your variant's overhead is measured.

**Where the contribution lives.** The contribution must be primarily in *how the ansatz is constructed and grown under the topology constraint*. You may restrict the operator pool, dynamically insert SWAP gates, re-label qubits adaptively, decompose non-local operators into nearest-neighbor sequences, or use whatever combination achieves topology compliance with minimal overhead. You may not modify the underlying McLachlan EOM solver or the variational-principle formulation (these are the AVQDS framework, held fixed). The underlying McLachlan EOM solver and variational-principle formulation must remain unchanged. Disclosure, justification, or an ablation does not exempt a modification from this requirement.

**Multi-topology demonstration (optional differentiator).** Two topologies are required by the contract. Demonstrating that your approach extends naturally to a *third* topology (e.g., linear chain, IBM-Eagle heavy-hex with a different 8-qubit subset, or all-to-all) at the same MFIM and LSM benchmarks is an additional differentiator. This is not required; it is rewarded.

**Effort evidence is mandatory.** For any contract item you describe as unresolved or partial — a benchmark configuration where your variant fails the accuracy floor, an overhead value worse than the naive baseline, an ablation that didn't converge, a topology-aware strategy you tried and abandoned — there must be a corresponding entry in `proposal/attempts_log.md` recording what was tried.

**Main paper vs. appendix structure.** The main paper contains work the agent claims represents a complete, deep contribution: the topology-aware method stated and motivated, the empirical results on all four benchmarks, the mechanism, failure, and compute-cost analyses, all at journal standard. The appendix is for work the agent attempted but could not bring to journal quality, with corresponding `attempts_log.md` entries. The expected outcome is an empty or near-empty appendix.

**Four principles your contribution must satisfy.**

1. **Originality.** Your topology-aware AVQDS variant must constitute original work. Substantively re-implementing a known published method for routing or hardware-aware compilation (e.g., the Qiskit transpiler's stochastic SWAP, SABRE, or published topology-restricted variational methods) presented as novel violates this principle. If your method is similar to a published one, position the contribution explicitly and demonstrate something not present in the prior work.

2. **Predictive sharpness.** Your method must be stated fully — operator-pool restriction, ansatz-growth strategy, routing logic, hyperparameters — in the Methodology *before* any empirical results are described. A domain expert should be able to reimplement your method from the Methodology alone.

3. **Diagnostic depth.** Your contribution must be supported by mechanism analysis (why does your variant achieve the overhead it does?), failure analysis (which benchmarks or regimes show higher overhead, and what does that reveal?), and compute-cost analysis (where does your method sit on the cost-quality trade-off compared to unconstrained AVQDS and naive routing?). These appear as subsections of the Analysis section.

4. **Statistical sharpness.** Quantitative claims must report all runs. If you ran multiple seeds or initializations, report all of them or use principled aggregation. Cherry-picking favorable runs violates this principle.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated using a leaf-level rubric and a tiered scoring function organized as gates (failing them precludes a high tier) and differentiators (among submissions that clear all gates, these determine tier ordering).

**Gates (all must pass):**

1. **Integrity.** Claims match evidence. Unconstrained AVQDS baseline runs are honestly logged. No hardcoded ground-truth values, no undisclosed external code, no falsified effort claims, no methodology misrepresented in the paper relative to what the code ran. Hyperparameters quoted in the paper match what the runs used.

2. **Topology compliance.** Every two-qubit gate in the topology-aware variant's circuits connects a pair of qubits in the relevant topology's edge set, separately verified on Topology A (2×4 square) and Topology B (heavy-hex). Logs must demonstrate this for both. Any two-qubit operation outside the edge set is a contract violation.

3. **Execution completeness.** Unconstrained AVQDS baseline verified on all four benchmark configurations. The smart-embedding naive SWAP-routing baseline run on all four configurations × both topologies. The topology-aware variant run on all four configurations × both topologies.

4. **Accuracy preservation.** Final-time fidelity of the topology-aware variant is within 1.0% absolute of the unconstrained AVQDS baseline on each of the four configurations × both topologies. A variant that drops below this floor on any (benchmark × topology) combination fails this gate.

5. **Honest accounting.** Work that is incomplete is admitted; the attempts log documents tried approaches; main/appendix placement reflects depth honestly.

**Differentiators (among gate-passing submissions):**

1. **Overhead magnitude.** The CNOT-count multiplier and depth multiplier relative to unconstrained AVQDS, averaged across the four benchmarks and reported separately for Topology A and Topology B. Lower is better. Topology A overhead near 1.0× is achievable with the right embedding; Topology B will always have non-trivial overhead because no Hamiltonian path exists. Rankings will weight Topology B overhead more heavily, since that is where the routing problem actually has bite.

2. **Margin over the smart-embedding naive baseline.** How much your variant beats the smart-embedding naive SWAP-routing baseline on each topology. A small margin (especially on Topology B) suggests the topology-aware logic is barely doing anything beyond what optimal embedding plus shortest-path routing achieves; a large margin indicates genuine algorithmic insight.

3. **Modular separation of topology-aware logic.** How cleanly does the variant isolate its topology-aware modifications from the AVQDS core (McLachlan EOM solver, operator-pool trial mechanism, ansatz-growth bookkeeping)? Strong submissions implement the variant as a contained set of changes (e.g., operator-pool restriction layer, routing-aware ansatz-growth wrapper, dynamic SWAP-insertion stage) layered over an unchanged AVQDS core, with a single switch that disables the topology-aware logic and recovers unconstrained AVQDS behavior. Tangled implementations — where the topology constraint is woven through every component — are weaker because they obscure what the contribution actually is and frustrate the mechanism and ablation analyses below.

4. **Mechanism analysis depth.** How clearly does the paper explain *why* your variant achieves the overhead it does on each topology? Strong submissions isolate the contributing component (operator-pool restriction, ansatz-growth ordering, dynamic routing strategy) via ablation, connect the behavior to quantum-compilation theory, and predict where the method's structure should and shouldn't help — including engagement with the structural difference between Topology A and Topology B.

5. **Failure analysis honesty.** How clearly does the paper characterize the benchmarks or topologies where your method underperforms? Strong submissions identify patterns (e.g., higher overhead on Topology B's branching qubit, or LSM's longer simulation time amplifying certain effects) rather than dismissing non-wins.

6. **Computational cost trade-off.** Where does the topology-aware logic sit on the cost-quality trade-off relative to unconstrained AVQDS? A method with comparable overhead at much higher classical compute cost is a weaker contribution than one with comparable cost.

7. **Third-topology generalization.** Submissions that demonstrate the approach extends naturally to a third topology (linear chain, an alternate heavy-hex subgraph, or another lattice) at the same benchmarks rank above submissions limited to the two required topologies.

8. **Originality and literature engagement.** Does the contribution constitute new work relative to the published quantum-compilation and hardware-aware variational literature?

9. **Writing quality.** Journal-quality structure, prose, and explicit limitations discussion.

A submission that overclaims accuracy preservation (e.g., reports fidelity within tolerance when logs show otherwise) or hides SWAP gates from overhead counts is worse than a submission that honestly admits its method underperforms the naive baseline on some benchmarks and explains why.

## Rules — Here are the rules

### Available resources

The working directory contains:

- `paper.md` — markdown version of the original paper (Yao et al., 2021).
- `paper.pdf` — original paper PDF.
- `images/` — figures extracted from the original paper.
- `code/AVQDS/` — the authors' AVQDS implementation. **Primary working dependency.** Contains the McLachlan EOM solver, the operator-pool trial mechanism, the adaptive-ansatz-growth logic, and the AVQDS time integrator. You will use this directly for the unconstrained baseline runs and (where practical) as the foundation for your topology-aware variant. Inspect this before writing baseline code.
- `data/` — working directory for auxiliary data you generate.
- `proposal/` — your output goes here, organized as `proposal/code/`, `proposal/logs/`, `proposal/codebase_notes.md`, `proposal/attempts_log.md`, with the final paper at `proposal/report.tex` (and rendered `proposal/report.pdf`).

For statevector simulation and quantum circuit construction, standard scientific Python libraries (NumPy, SciPy, Qiskit, PennyLane, QuTiP) are available via the environment's package manager.

### Evaluation Protocol

All simulations are run via **classical statevector simulation** (no noise model).

#### Qubit topologies (held fixed for this task)

Both topologies below are required. Your variant must produce topology-compliant circuits on both, and overhead must be reported separately for each.

**Topology A — 2×4 square lattice** (10 edges):

```
Row 0:  0 — 1 — 2 — 3
        |   |   |   |
Row 1:  4 — 5 — 6 — 7

Edges_A = {(0,1), (1,2), (2,3),       horizontal top
           (4,5), (5,6), (6,7),       horizontal bottom
           (0,4), (1,5), (2,6), (3,7)} vertical
```

**Topology B — 8-qubit heavy-hex subgraph** (7 edges):

```
        0 — 1 — 2
            |
            3
            |
        4 — 5 — 6 — 7

Edges_B = {(0,1), (1,2), (1,3),
           (3,5),
           (4,5), (5,6), (6,7)}
```

The MFIM chain ordering (spin sites $0$ through $N-1$) and the LSM chain ordering (sites $0$ through $N-1$) both correspond to logical spin indices, *not* to physical qubits in either topology. Mapping the spin chain onto each topology is part of what your topology-aware variant must handle.

#### System 1: Mixed-Field Ising Model (MFIM) — sudden quench

$$H_{\text{MFIM}} = -J \sum_{i} Z_i Z_{i+1} + h_x \sum_i X_i + h_z \sum_i Z_i$$

| Parameter | Value |
|-----------|-------|
| System size | **8 qubits** with **periodic boundary conditions (PBC)** |
| Coupling | $J = 1$ |
| Transverse field | $h_x = -2$ |
| Longitudinal field | $h_z = 0$ (integrable TFIM limit) **and** $h_z = 0.5$ (nonintegrable MFIM) |
| Initial state | $|0\rangle^{\otimes 8}$ |
| Dynamics | Sudden quench: full $H_{\text{MFIM}}$ turned on at $t = 0$ |
| Simulation time | $T = 3$ (in units of $1/J$) |

The Hamiltonian-pool operators are $\{Z_iZ_{i+1}\}$ for all bonds (8 with PBC, including the periodic bond $Z_7Z_0$), $\{X_i\}$ for all sites (8), and $\{Z_i\}$ for all sites (8, only when $h_z \neq 0$). Total: 16 operators for $h_z = 0$, 24 for $h_z = 0.5$.

#### System 2: Lieb-Schultz-Mattis (LSM) chain — finite-rate ramp quench

$$H_{\text{LSM}} = -\sum_i \left[(1+\gamma) \sigma^x_i \sigma^x_{i+1} + (1-\gamma) \sigma^y_i \sigma^y_{i+1}\right] + h_z \sum_i \sigma^z_i$$

Confirm exact prefactors from `paper.md` — some conventions differ by a factor of 2 in the XY coupling terms.

| Parameter | Value |
|-----------|-------|
| System size | **8 qubits** with **open boundary conditions (OBC)** |
| Anisotropy | $\gamma$ ramped linearly through the FMx-to-FMy phase transition (consult paper for exact start/end values and ramp protocol) |
| Longitudinal field | Two quench paths: $h_z = -0.7$ **and** $h_z = 1.6$ |
| Initial state | Ground state of the initial Hamiltonian in the FMx phase (compute via classical exact diagonalization) |
| Dynamics | Finite-rate linear ramp with inverse speed $T = 3$, followed by post-ramp dynamics for another period $T$ |
| Total simulation time | $2T = 6$ |

#### Metrics (all logged per time step)

- **State fidelity** $f(t) = |\langle \psi_{\text{exact}}(t) | \psi_{\text{var}}(t) \rangle|^2$
- **Infidelity** $1 - f$ at final time
- **CNOT gate count** $N_{\text{cx}}$ as a function of time
- **Circuit depth** as a function of time
- **Number of variational parameters** $N_\theta$ as a function of time
- **Loschmidt echo** $|\langle \psi(0) | \psi(t) \rangle|^2$ (MFIM)
- **Energy expectation** $\langle \hat{H} \rangle(t)$ vs. exact (LSM)
- **Spin correlation functions** $\langle \sigma^x_i \sigma^x_j \rangle$, $\langle \sigma^y_i \sigma^y_j \rangle$ (LSM)
- **Topology compliance audit**: per-step record of every two-qubit gate's qubit pair, with verification that each pair is in the topology edge set.

#### Reference computation

```python
import scipy.linalg
psi_exact = scipy.linalg.expm(-1j * H.toarray() * t) @ psi_0  # sudden quench
# For LSM ramp, use a converged time mesh.
```

#### Comparison methods

- **Unconstrained AVQDS** (the paper's method): the established baseline, run via `code/AVQDS/`. CNOT count uses the all-to-all convention (2 CNOTs per two-qubit Pauli rotation).
- **Smart-embedding naive SWAP-routing AVQDS**: unconstrained AVQDS with the best available initial qubit-to-topology mapping (e.g., Hamiltonian-cycle embedding on Topology A for MFIM PBC), followed by post-hoc shortest-path SWAP insertion for any remaining non-local two-qubit gates. This is the reference for what good routing without ansatz adaptation looks like.
- **Your topology-aware variant.**
- **Exact evolution** via `scipy.linalg.expm` as ground truth.

### Working directory

- **Do not read any local files or folders** outside of the working directory.
- Organize all code and outputs in `proposal/`.

### Compute budget

- Hardware: CPU machine with 32 vCPU cores, ~60 GB RAM, ~470 GB disk. No GPU, and no swap. Treat ~50 GB as a hard memory ceiling (the node has ~60 GB RAM, no swap, and the OS + agent harness need headroom): size grids, dense matrices, ensembles, batch sizes, and parallel workers to keep peak resident memory under it; estimate/monitor peak memory before scaling up and prefer chunked / streaming / sparse computation. A process that exhausts RAM is killed by the OS.
- Statevector simulation for 8 qubits is fast; the dominant cost is the trial-expansion loop over the operator pool during ansatz growth. The naive SWAP-routing baseline adds modest post-processing cost. Your topology-aware variant's cost depends on its specific design.

### Autonomy and external resources

**Internet search is permitted and encouraged.** Use it for literature lookup on hardware-aware quantum compilation, qubit routing, and connectivity-restricted variational methods; for documentation of the standard scientific Python libraries (NumPy, SciPy, Qiskit, PennyLane, QuTiP); and to read follow-up work to the original paper so you can identify what has already been published and differentiate your contribution. Cite anything you use.

Make your own decisions and justify them in the write-up. Do not ask questions.

**On replication.** Hardware-aware quantum compilation has a rich literature (Qiskit's transpiler, SABRE routing, t|ket⟩'s routing passes, published topology-restricted variational methods). Convergence with published ideas is expected — but substantive re-implementation of an existing method presented as novel will be severely degraded. Position your contribution explicitly against the prior work.

### No reward hacking

These rules are disclosed up front so the contract is clear:

1. **Number traceability.** Every quantitative claim in your paper (fidelity values, CNOT counts, depth values, overhead multipliers) must trace to a corresponding log entry with the same value.

2. **Pre-stated method and consistent reporting.** Your topology-aware variant must be stated in the Methodology before any empirical results. Hyperparameter values quoted in the paper must match what the runs used.

3. **Baseline honesty.** Any unconstrained AVQDS or naive SWAP-routing baseline numbers you cite must come from your own logged runs (or from `code/AVQDS/` used directly with stated configuration). The original paper's numbers are not substitutes — your local runs will produce different exact values.

4. **Topology compliance verification.** Logs must include per-step audits recording every two-qubit gate's qubit pair and a verification flag confirming the pair is in the relevant topology's edge set. Audits must be reported separately for Topology A (2×4 square) and Topology B (heavy-hex). Two-qubit gates outside an edge set are contract violations and will be flagged. SWAP gates are permitted, but they must themselves connect adjacent qubits in the relevant topology, and they count toward the total CNOT count (3 CNOTs per SWAP).

5. **Honest overhead reporting.** All gates in the topology-aware variant's circuit count toward depth and CNOT count, including SWAPs, including any auxiliary gates used for routing. Hiding SWAPs in "preprocessing" or excluding them from overhead counts is a contract violation. The CNOT-count multiplier and depth multiplier reported in the headline table are computed against the unconstrained AVQDS baseline run via `code/AVQDS/` on the same benchmark, using the paper's all-to-all CNOT-counting convention (2 CNOTs per two-qubit Pauli rotation) for the unconstrained run. Overhead is reported separately for each topology.

6. **Initial-state and seed pairing.** Within a single seed, unconstrained AVQDS, the naive SWAP-routing baseline, and your topology-aware variant must use the same initial state and the same time-stepping schedule. Logs must demonstrate this pairing.

7. **No hardcoded reference values.** Your code must not contain hardcoded exact-evolution outputs, fidelity targets, or other ground-truth numerical literals.

8. **Variational simulation integrity.** The variational state must be evolved through the parametrized circuit (parameter EOM). Ansatz-expansion decisions must be based on classical-side measurable quantities (such as the McLachlan distance), not on fidelity to the exact state. Logs must demonstrate this.

9. **Effort evidence.** Any failed or unresolved work must have corresponding `proposal/attempts_log.md` entries.

10. **Codebase inspection evidence.** `proposal/codebase_notes.md` must record what `code/AVQDS/` provides and what was reused. An absent or empty file, combined with from-scratch reimplementation of components AVQDS already provides, will be flagged.

11. **Experiment completeness.** Every experiment described in the paper must have corresponding logs.

12. **Code-paper alignment.** The Methodology must accurately describe what the code does. Significant divergences between described and implemented method, including the topology-aware logic, will be flagged.

13. **Generalization.** Your method must generalize across all four benchmark configurations × both topologies — methods that achieve gains on MFIM by sacrificing LSM performance, or that work well on Topology A by exploiting its Hamiltonian-cycle structure but fail on Topology B, will not satisfy the contract.

14. **Naive baseline protocol faithfulness.** The smart-embedding naive SWAP-routing baseline is defined precisely: unconstrained AVQDS run via `code/AVQDS/`, followed by post-hoc shortest-path SWAP insertion on the resulting circuit for any two-qubit gates that are non-local on the topology. Silently substituting a different routing algorithm — a longer-path heuristic, a deliberately weakened SWAP insertion, or a re-run of AVQDS with topology constraints (which defeats the comparison) — flatters your variant's margin and is a contract violation. The naive baseline's logs must show the unconstrained source circuit and the SWAP-inserted circuit side by side so the post-hoc transformation is auditable.

15. **No selective reporting.** If you tried multiple topology-aware variants (different operator-pool restrictions, different routing strategies, different ansatz-growth orderings), multiple initial embeddings (especially on Topology B where many smart embeddings exist), or multiple SWAP-insertion strategies during the work, the paper must either report all of them or apply a principled aggregation with the rule stated. Reporting only the favorable run while burying the others — for example, picking the worst-performing naive embedding to flatter the variant's margin — is the central paper-4 selective-reporting risk and will be flagged. The attempts log must document configurations set aside.

16. **Operator pool authorship.** The headline runs must use the canonical Hamiltonian operator pool (16 operators for MFIM at $h_z = 0$; 24 operators for MFIM at $h_z = 0.5$ with the $Z_i$ terms added; the natural pool for LSM). The original paper explored an *expanded* pool with additional Pauli strings and reported it generally produced longer circuits without accuracy gain. Silently switching to an expanded pool in the headline — to get better fidelity at the cost of comparability with the paper's reference numbers — is a contract violation. If your variant motivates an expanded pool, disclose it in Methodology and report an ablation against the canonical pool.

17. **Simulation time faithfulness.** All MFIM runs must reach $T = 3$ (in units of $1/J$); all LSM runs must reach total simulation time $2T = 6$ (the ramp of duration $T = 3$ plus post-ramp dynamics for another $T = 3$). Silently shortening simulation time — measuring fidelity, CNOT count, and depth at an earlier time — produces artificially low overhead numbers at the headline measurement point and will be flagged.

18. **CNOT-counting convention consistency.** The held-fixed convention for unconstrained AVQDS is 2 CNOTs per two-qubit Pauli rotation. The variant's overhead multiplier is computed against this reference. Using a different convention for the unconstrained baseline (e.g., 4 CNOTs per Pauli rotation under a hardware-aware basis-gate decomposition) inflates the reference and flatters the variant's overhead. Your variant's CNOT counts use the same per-Pauli-rotation decomposition for the variational two-qubit Pauli rotations; SWAP gates contribute 3 CNOTs each.

19. **Noiseless simulation faithfulness.** All runs (unconstrained AVQDS, smart-embedding naive baseline, topology-aware variant) must use noiseless statevector simulation. Silently adding a noise model (depolarizing channels, readout error, hardware noise) changes the comparison entirely and is a contract violation if undisclosed.

## Deliverables — Here's what to hand in

Organize all outputs under `proposal/`. Submit:

**Codebase, logs, codebase notes, and attempts log:**

- `proposal/code/` — your implementation, with clear entry points and scripts that produce the runs reported in the paper.
- `proposal/codebase_notes.md` — short document recording what you found in `code/AVQDS/`, which components are usable, which you reused, adapted, or bypassed.
- `proposal/logs/` — all run logs, including:
  - **Unconstrained AVQDS runs**: on all four benchmark configurations (CNOT count under the all-to-all convention).
  - **Smart-embedding naive SWAP-routing baseline runs**: all four configurations × both topologies (8 runs total).
  - **Topology-aware variant runs**: all four configurations × both topologies (8 runs total).
  - **Topology-compliance audit logs**: per-step record of every two-qubit gate's qubit pair with edge-set verification, separately for each topology.
  - **Initial-embedding logs**: the qubit-to-topology mapping(s) chosen for each (benchmark × topology), with justification (e.g., Hamiltonian cycle for Topology A on MFIM PBC).
  - **Ablation runs**: if your variant has multiple components, log ablations that isolate each.
  - **Hyperparameter-selection runs**: with what was tried and how final values were selected.
  - **Compute-cost measurements**: wall-clock time per simulation for each method.
- `proposal/attempts_log.md` — structured record of every distinct approach attempted, with what was tried, what was observed, and why it was kept or abandoned.

**Follow-up paper** at `proposal/report.tex`, compiled to a rendered `proposal/report.pdf`. Use `neurips.sty`. The paper must include: Introduction, Methodology, Experimental Setup, Experimental Results, Analysis (with mechanism, failure, and compute-cost subsections), Conclusion, References.

The Experimental Results section must include a primary table labeled `\label{tab:main_results_mfim}` reporting MFIM results, with the structure below. The "Your variant" row label is a placeholder — replace with a name that describes your contribution. "Topology A" and "Topology B" refer to the 2×4 square and 8-qubit heavy-hex respectively.

| Method | Topology | MFIM $h_z\!=\!0$ fidelity | MFIM $h_z\!=\!0$ $N_{\text{cx}}$ | MFIM $h_z\!=\!0$ depth | MFIM $h_z\!=\!0.5$ fidelity | MFIM $h_z\!=\!0.5$ $N_{\text{cx}}$ | MFIM $h_z\!=\!0.5$ depth |
|---|---|---|---|---|---|---|---|
| Unconstrained AVQDS | — (all-to-all) | | | | | | |
| Smart-embedding naive routing | A | | | | | | |
| Smart-embedding naive routing | B | | | | | | |
| Your variant | A | | | | | | |
| Your variant | B | | | | | | |

A second table labeled `\label{tab:main_results_lsm}` with the same column structure (replacing MFIM with LSM and its two $h_z$ values) must report LSM results. A third table labeled `\label{tab:overhead}` must report the depth-multiplier and CNOT-multiplier overhead of your variant vs. unconstrained AVQDS on each (benchmark × topology) combination, with the smart-embedding naive baseline's overhead listed alongside for comparison.

Beyond the tables, include: (1) fidelity vs. time curves for all four benchmark configurations × both topologies (three methods overlaid: unconstrained, smart-embedding naive, your variant); (2) CNOT-count vs. time curves for all four configurations × both topologies; (3) Loschmidt echo vs. time for the two MFIM cases; (4) energy expectation and spin correlation functions for both LSM cases; (5) ablation figures if your variant has multiple components; (6) a visualization of which two-qubit gates the variant uses on each topology, illustrating the topology-compliance pattern and the embedding choice.

The Analysis section must include subsections for:
- **Mechanism analysis** — why does your variant achieve the overhead it does on each topology, relative to unconstrained AVQDS and the smart-embedding naive baseline? Engage explicitly with the structural difference between Topology A (Hamiltonian-cycle admitting) and Topology B (no Hamiltonian path).
- **Failure analysis** — which (benchmark × topology) combinations show higher overhead, lower accuracy, or other shortcomings? What pattern explains this?
- **Computational cost trade-off** — wall-clock cost of your variant vs. unconstrained AVQDS, and discussion of any non-trivial classical-side overhead in the topology-aware logic.

Additional subsections (sensitivity to initial state, scaling discussion if extrapolated, third-topology generalization, synthesis across the configurations) are encouraged.

### Paper quality

Your submission is an academic paper, not a class project report. Writing should resemble what would be reviewed at a top scientific journal in structure, integrity, depth, and language.

Do not frame the paper as "a follow-up of Yao et al." Frame it as a self-contained scientific contribution that contains its own motivation, methods, results, and discussion, with the original paper cited where appropriate. Search the literature broadly to differentiate your contribution from prior work on hardware-aware quantum compilation, qubit routing, and connectivity-restricted variational methods. Include at least as many references as the original paper.

**Independence from this task spec.** A reader of your paper who has not seen this spec should not be able to reverse-engineer it from the paper's structure or terminology. The paper's title should describe your scientific contribution (a specific topology-aware variant, theoretical insight, or empirical finding), not the structure of your investigation. Section structure should be physics- or methodology-driven, not contract-driven.

**Rendered output verification.** Before submitting, render your LaTeX to PDF (e.g., `pdflatex report.tex`) and visually inspect the resulting PDF. Equations must not overflow page margins; tables must fit within the text width; figures must not be cut off.

Good luck.
