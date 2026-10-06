# Task: Develop a Substantive Follow-up to "Hierarchical Modeling of Molecular Energies Using a Deep Neural Network"

## Objective — Here's what you're doing

Develop a substantive scientific follow-up to the Hierarchically Interacting Particle Neural Network (HIP-NN) by Lubbers, Smith, and Barros, "Hierarchical modeling of molecular energies using a deep neural network," J. Chem. Phys. 148, 241715 (2018), in the direction of **characterizing where the hierarchical decomposition breaks down**, identifying the chemical motifs responsible for those breakdowns, and building a calibrated uncertainty model from the hierarchy-failure signal.

## The Problem — Here's why it matters

A machine-learning interatomic potential (MLIP) for molecular systems maps an atomic configuration $\mathcal{C} = \{(Z_i, \mathbf{r}_i)\}$ to a scalar energy $\hat{E}(\mathcal{C})$. Modern architectures (HIP-NN, SchNet, PhysNet, MPNN families) achieve sub-1 kcal/mol mean absolute error on benchmark organic-molecule datasets, often well below "chemical accuracy" thresholds. But the headline MAE hides a tail: even at $0.26$ kcal/mol MAE on QM9, HIP-NN's original paper reports that **2.3% of molecules have prediction errors exceeding 1 kcal/mol**. For molecular dynamics, transition-state characterization, or high-throughput chemical screening, this tail matters far more than the mean — a one-in-fifty failure rate is the difference between a tool a chemist trusts and a black box that needs human spot-checking.

The deeper question is: when a HIP-NN-style model fails, *can the model itself tell you it failed?* HIP-NN's hierarchical decomposition was designed in part to make this question tractable. Energies decompose as $\hat{E} = \sum_i \sum_n \hat{E}_i^n$ over per-atom hierarchical orders $n = 0, 1, \dots, N_{\text{interaction}}$, with order $n$ capturing roughly $(n{+}1)$-body interactions. When the model functions properly, $\hat{E}_i^n$ should decay rapidly with $n$, and the non-hierarchicality

$$R = \sum_{n=1}^{N_{\text{interaction}}} \sum_{i=1}^{N_{\text{atom}}} \frac{(\hat{E}_i^n)^2}{(\hat{E}_i^n)^2 + (\hat{E}_i^{n-1})^2}$$

should be small. The original paper's Figure 3 shows that when a model is regularized using R as a component in the loss function that larger $R$ correlates with larger prediction error — the model has a built-in uncertainty signal.

But the paper does not unpack that signal further. It does not identify *which chemical motifs* drive hierarchy failure, *which structural features* correspond to high-$R$ regions of the molecular manifold, *whether the $R$ signal is calibrated* (predicted-uncertainty bins matching actual-error magnitudes), or *whether the resulting failure modes are compact* (a few dominant motifs explain most failures) versus diluted (every molecule contributes a little to the failure tail with no clear structure). These are direct follow-ups to the paper's own uncertainty claim, and they matter for any downstream use that needs calibrated prediction-confidence intervals rather than just a low mean error.

## Background — Here's what you need to know about the original paper

The Lubbers, Smith, and Barros 2018 paper introduces HIP-NN and validates it on QM9 and the original MD17 trajectory benchmarks. Read `paper.md` in full before beginning. The technical components of the original method are:

1. **Molecular representation.** Atomic positions $\mathbf{r}_i$ are encoded via pairwise distances $r_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|$ with a radial cutoff $R_{\text{cut}} = 15$ Bohr (smooth, cosine-modulated). Atomic species $Z_i$ are one-hot encoded over the elements present in the dataset (H, C, N, O, F for QM9). The representation is invariant under rigid translations, rotations, and atom-index permutations by construction.

2. **Network architecture.** HIP-NN alternates *interaction layers* (mixing information between pairs of atoms within $R_{\text{cut}}$, modulated by 20 learned inverse-distance sensitivity functions $s_\nu^\ell(r_{ij})$) and *on-site layers* (per-atom feature transformations), with residual connections and softplus activations. The baseline architecture has $N_{\text{interaction}} = 2$ interaction layers, $N_{\text{on-site}} = 3$ on-site layers per interaction block, and $N_{\text{feature}} = 80$ atomic features per layer (~234k parameters total).

3. **Hierarchical energy regression.** The total energy is $\hat{E} = \sum_i \sum_n \hat{E}_i^n$ with $\hat{E}_i^n = \sum_a w_a^n z_{i,a}^{\ell_n} + b^n$ (linear regression on the features at selected layers $\ell_n$). Order $n=0$ is the dressed-atom approximation; higher $n$ adds increasingly many-body content.

4. **Hierarchical regularization.** A loss term $\mathcal{L}_R = \lambda_R \langle R \rangle_D$ (with $\lambda_R = 10^{-2}$) penalizes molecules whose hierarchical contributions $\hat{E}_i^n$ do *not* decay with $n$. This encourages the trained model to actually use the hierarchical structure, and creates the $R$-as-uncertainty-signal correspondence shown in Figure 3.

5. **Training protocol.** Adam optimizer ($\eta_{\text{init}} = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$), L2 regularization at $\lambda_{L2} = 10^{-6}$, minibatch size 30, validation-based early stopping with patience $t_{\text{patience}} = 50$ and learning-rate annealing factor $\alpha_{\text{decay}} = 0.5$; training terminates with a patience of 100 epochs. Initial training period $t_{\text{init}} = 100$ epochs, max $t_{\text{max}} = 2000$ epochs. The paper trained on $\sim$8 random splits per benchmark and reported sample-mean MAE with standard-error error bars.

**Reference baseline results from the paper (Table I, QM9 MAE in kcal/mol):**

| $N_{\text{train}} + N_{\text{validate}}$ | HIP-NN |
|---|---|
| 110,426 | $0.256 \pm 0.003$ |
| 100,000 | $0.261 \pm 0.002$ |
| 50,000 | $0.354 \pm 0.004$ |

**Figure 3 (the uncertainty story):** the quantile function $Q_{\text{err}}(p, R)$ — error level at the $p$-th percentile, conditioned on non-hierarchicality $R$ — increases monotonically with $R$ across the bulk of the QM9 dataset ($R \gtrsim 3 \times 10^{-2}$). This is the empirical correspondence the paper invokes as a built-in uncertainty estimate. The paper stops at showing the correspondence — it does not characterize the chemistry driving the high-$R$ tail, does not assess calibration quantitatively, and does not measure the compactness of the failure motifs.

## Contract — Here's exactly what counts as success

The substantive contribution is to characterize where HIP-NN's hierarchical decomposition breaks down on out-of-distribution chemistry, identify the structural motifs responsible for those breakdowns, build a calibrated uncertainty model based on hierarchy-failure signals, and assess whether the failure motifs are compact (a few motifs explain most failures) or diluted (failures spread across many uncorrelated structural features). You design the motif-identification methodology; you design the uncertainty model on top of (or extending) the paper's $R$; you choose how to validate calibration. The HIP-NN architecture and training protocol are held fixed so the comparison evaluates *what the hierarchy reveals about itself*, not *what a different model would reveal*.

Your submission clears **three independent bars**:

1. **Substantive contribution** — the analysis identifies a meaningful library of failure motifs in the out-of-distribution chemistry, the proposed uncertainty model is calibrated against actual prediction errors on held-out chemistry, and the motif library is shown to be compact (a Pareto analysis of motif → failure attribution). (See "Operationalized success thresholds" below.)
2. **Honest framing** — motif identification reports its statistical support (how many test-set molecules each motif covers, what fraction of failures it explains); calibration is evaluated quantitatively with reported false-positive and false-negative rates; cases where the uncertainty model fails are reported as themselves.
3. **Code–paper consistency** — the HIP-NN architecture and training procedure described in the paper match what the code (hippynn package) implements. Hyperparameters quoted in the paper match the logs.

These three bars are checked independently by the rubric.

### Operationalized success thresholds

| Axis | Floor (must clear) | "Substantial" (the hallmark) |
|---|---|---|
| **Reimplementation of HIP-NN on the in-distribution training subset** | Reimplemented HIP-NN matches or improves on the paper-as-reported MAE at the comparable training-set size (e.g., $\leq$ 0.354 kcal/mol when comparing to the paper's 50k-training reference, with reasonable extrapolation across training-size differences between the agent's $\leq 8$-heavy-atom subset and the paper's full-QM9 training sizes) | Reimplemented HIP-NN MAE is **at or below** paper-as-reported across every training-size comparison reported — failure motifs are anchored on a faithful baseline |
| **Failure motif identification** | At least 3 distinct failure motifs identified, each with quantitative statistical support (test-set coverage, error-distribution skew) | At least 5 motifs identified, spanning multiple chemistry categories (structural, conformational, electronic); each motif's connection to the hierarchical decomposition failure is mechanistically explained |
| **Uncertainty model calibration** | Calibration curve and expected calibration error (ECE) reported on held-out 9-heavy-atom molecules; precision and recall for flagging high-error predictions stated at one or more chosen thresholds | Calibration curve close to identity (within $\pm 10\%$ across the predicted-uncertainty range); ECE $\leq 0.05$; precision and recall reported across a sweep of thresholds with a chosen operating point justified |
| **Motif compactness** | A Pareto-style cumulative-attribution plot (motif rank vs. cumulative fraction of failures explained) is reported | Top-$K$ motifs (with $K \leq 5$) explain $\geq 80\%$ of the identified failures, demonstrating that the failure modes are interpretable rather than diluted across the entire test set |
| **Conformational transfer of the motif library** | The minimized-geometry limitation of the motif library is addressed at least by substantive analytical engagement — a reasoned account of which motifs would or would not transfer to non-minimized (MD-sampled) conformations, and why. (Running the quantitative transfer test empirically on ANI-1x or rMD17 is the substantial outcome, not the floor.) | Quantitative transfer test executed on ANI-1x (or, as a smaller-scope alternative, rMD17) with per-motif transfer-rate analysis and explicit characterization of which motifs do and don't transfer from minimized to MD-sampled geometries |

### Held fixed (parity constraints)

These are not part of the design space. They must remain identical to the paper's protocol:

- **The chemistry**: HCNOF only, as in QM9. No transition metals, no halogens beyond F. (Multi-element MLIP benchmarks built around other element sets are not transferable to this comparison.)
- **The dataset**: QM9, as fetched into `data/qm9/`, with the standard pruning of $\sim$3k geometric-consistency-failure molecules and the 11 additional energy-minimization failures the paper removes. Use the pruned $\sim$131k-molecule dataset.
- **The train/test split based on heavy-atom count**: train HIP-NN on the **$\leq 8$-heavy-atom subset** of QM9 (the in-distribution set, where "heavy atom" means C/N/O/F, not H); evaluate the trained model and identify failure motifs on the **9-heavy-atom subset** (the extrapolation set). The validation split for early stopping is drawn from within the $\leq 8$-heavy-atom subset.
- **The HIP-NN architecture and training protocol**: $N_{\text{interaction}} = 2$ interaction layers, $N_{\text{on-site}} = 3$ on-site layers, $N_{\text{feature}} = 80$ atomic features, 20 sensitivity functions, hierarchical regularization at $\lambda_R = 10^{-2}$, L2 regularization at $\lambda_{L2} = 10^{-6}$, Adam optimizer with the paper's $\eta_{\text{init}}$, $\beta_1$, $\beta_2$, minibatch size 30, validation-based early stopping at patience 50 and decay 0.5. The motivation for holding these fixed is that the comparison evaluates the hierarchy's failure structure, not the architecture's expressive power. (If your compute budget cannot reach the paper's $t_{\text{max}} = 2000$ epochs, reduce the schedule transparently and disclose; see the matched-budget note below.)
- **Energy-only training**: HIP-NN in the original paper trains on molecular energies only, not forces. Match this for the QM9 experiments. (Force-only training is a different study.)
- **No DFT generation**: you may not generate new DFT calculations, run quantum chemistry calculations yourself, or use PySCF / Psi4 / ORCA / Gaussian / VASP to compute new reference data. The contract holds the training data fixed at QM9 as provided. 
- **Validation anchor**: Comparison of MAE between agent's reimplementation and the paper's published number is anchored on QM9 (which has not changed since 2014). Transfer tests (on ANI-1x or rMD17) are reported as a complementary analysis, not as a substitute for the QM9 in-distribution comparison.

> **Note on matched-budget framing.** The contract evaluates your method's behaviour on the paper's published training regime ($\sim$1000–2000 epochs, $N_{\text{feature}} = 80$, $\sim$234k parameters). A reduced-regime comparison — e.g., training a smaller HIP-NN (reduced $N_{\text{feature}}$), fewer epochs, fewer seeds — does not satisfy the contract even if it produces well-defined paired comparisons within the reduced regime. If your compute budget cannot reach the published regime, run the largest model and longest schedule your budget supports, **disclose the gap to paper-as-reported in Experimental Setup**, and address whether your uncertainty-calibration and motif-compactness conclusions transfer to a full-convergence regime in your Failure Analysis. Hidden under-convergence is a contract violation; transparent under-convergence with substantive transfer analysis is acceptable.

> **Note on reimplementation accuracy as a contract-integrity requirement.** Failure motif identification depends on the reimplemented HIP-NN being a *faithful* reproduction of the paper. If your reimplementation is meaningfully weaker than paper-as-reported on every comparable training-set size, the "failure motifs" you identify may be artifacts of under-converged training (the reimplementation isn't learning some chemistry pattern) rather than properties of the hierarchical architecture (HIP-NN as designed has trouble with this chemistry). The headline contribution loses scientific validity to the extent the baseline is weak. The contract therefore enforces a strict floor: at every training-set size you report a HIP-NN run, the reimplementation's MAE must be **at or below** the paper-as-reported MAE at the comparable size (with reasonable extrapolation across training-size differences between your $\leq 8$-heavy-atom subset and the paper's full-QM9 training sizes). A reimplementation that's even slightly worse than paper-as-reported is a Tier 3 weakness; a reimplementation that's substantially worse with the gap disclosed is at the contract boundary; a reimplementation that's catastrophically worse (>2× paper MAE) or any gap silently is a Tier 2 / Tier 0 violation. This is the most important hygiene gate for the K3 direction's scientific validity.

### What you may design

- The **failure motif identification methodology**: clustering on hierarchical components $\{\hat{E}_i^n\}_{n,i}$, hand-coded structural features (e.g., functional-group counts, rotatable bonds, ring statistics — but identify your own), learned classifiers, attention-based attribution, decision trees on per-atom features, or whatever combination supports the claim. Standard cheminformatics representations (Morgan fingerprints, RDKit descriptors, etc.) are tools you may use; cite the libraries.
- The **uncertainty model** itself: $R$ from the paper is one signal, but you may extend it — include the per-order spread $\text{Var}_n(\hat{E}_i^n)$, the maximum hierarchical contribution $\max_n |\hat{E}_i^n|$, the total atomic energy magnitude, ensembles across multiple HIP-NN seeds, or whatever combination produces a better-calibrated uncertainty than $R$ alone. Document which signals you use and how they combine.
- The **calibration protocol**: how you bin predicted uncertainty, how you compute calibration error (ECE, MCE, reliability diagrams, etc.), how you choose operating thresholds for the precision/recall analysis.
- The **motif library structure and presentation**: whether to organize by chemistry class (e.g., conjugated systems, strained geometries, heteroatom motifs), by failure magnitude, or by some learned hierarchy.
- The choice between **ANI-1x** (`data/ani1x/`) and **rMD17** (`data/rmd17/`) for the conformational transfer test is part of the contribution. ANI-1x is the recommended substantial-outcome target (the canonical conformational-variance dataset used by that research group); rMD17 is acceptable as a smaller-scope alternative if ANI-1x integration is too heavy within the budget. Either way, addressing the conformational-transfer question is required (see "Conformational transfer of the motif library" below): running the empirical test on ANI-1x or rMD17 is the substantial outcome, and substantive analytical engagement with the limitation is the floor — the question is which, and how substantively.

## Evaluation — Here's how you'll be judged

Your submission will be evaluated along two axes:

1. **Performance on stated metrics** — reimplementation MAE relative to paper-as-reported, motif library coverage and statistical support, uncertainty calibration error, motif compactness ratio.
2. **Scientific merit** — depth of analysis, quality of the mechanistic story connecting motifs to hierarchy failure, engagement with the post-HIP-NN MLIP-uncertainty literature, soundness of the methodological choices. Scientific merit values ambition and depth without prescribing what counts as ambitious.

### Two-baseline reporting (required)

Every quantitative comparison must report two HIP-NN baselines side-by-side:

1. **Paper-as-reported HIP-NN** — values cited from Tables I/II of Lubbers, Smith, Barros 2018. Primary headline references: QM9 110k-training MAE $0.256 \pm 0.003$ kcal/mol (Table I); QM9 50k-training MAE $0.354 \pm 0.004$ kcal/mol (Table I). Cite every paper-as-reported value to its specific source location.
2. **Reimplemented HIP-NN** — your own run via the hippynn package on the $\leq 8$-heavy-atom QM9 subset at the held-fixed architecture and protocol. This is the paired-comparison baseline against which your motif identification and uncertainty calibration are evaluated.

Your motif analysis and uncertainty model build on the reimplemented HIP-NN's hierarchical outputs. The paper-as-reported baseline serves as an external anchor — if your reimplementation diverges substantially from paper-as-reported on the comparable training size, the gap must be documented and addressed in your failure analysis (under-convergence may itself contribute to motif identification artifacts).

### Training / evaluation protocol

Reproduce the paper's per-split methodology with adjustments for the heavy-atom-extrapolation framing:

- **Train set**: QM9 molecules with $\leq 8$ heavy atoms. Use $\sim$1k molecules as the validation split (for early stopping); the remainder is training data.
- **Test set**: QM9 molecules with $= 9$ heavy atoms. This is the out-of-distribution extrapolation set on which failure motifs are identified and the uncertainty model is calibrated.
- **Number of independent splits / seeds**: report at least **3 independent random shuffles** of the train/validation partition within the $\leq 8$-heavy-atom set, with mean ± standard error across the seeds for every reported number. (The paper uses 8; if compute permits, more is better. If compute requires fewer, document the reduction.)
- **Energy metric**: mean absolute error (MAE) on the test set, in kcal/mol, computed per-molecule, then averaged.

### Failure motif identification protocol

For each trained HIP-NN model on the $\leq 8$-heavy-atom training set:

- Evaluate on the 9-heavy-atom test set.
- Compute non-hierarchicality $R$ (Eq. 18 of the paper) and per-order contributions $\hat{E}_i^n$ for every test molecule.
- Identify a library of failure motifs: structural / conformational / chemical features that correlate with high $R$ *and* high prediction error.
- For each motif, report: definition (precise enough to be applied to a new molecule), test-set coverage (fraction of 9-heavy-atom molecules matching the motif), conditional error distribution (mean and percentiles of $|\hat{E} - E|$ for molecules matching the motif), and statistical support (compared to non-matching molecules, with a stated significance test).

### Uncertainty model calibration protocol

For your uncertainty model $\hat{u}(\mathcal{C})$ on test molecules:

- Compute $\hat{u}$ on every 9-heavy-atom test molecule.
- Bin molecules by predicted uncertainty (e.g., deciles or quartiles).
- For each bin, compute the actual error distribution (mean, std, quantiles of $|\hat{E} - E|$).
- Report a **calibration curve**: x-axis = predicted-uncertainty bin midpoint, y-axis = actual mean error in bin. A perfectly calibrated model has the curve on the identity line (up to a multiplicative scale).
- Report **expected calibration error (ECE)** or an equivalent scalar calibration metric.
- Choose one or more thresholds for flagging "high-uncertainty" predictions and report **precision** (of flagged predictions, what fraction are actually high-error?) and **recall** (of high-error predictions, what fraction are flagged?). Justify the threshold choice in the prose.

### Motif compactness protocol

Quantify whether the failure motifs are compact (interpretable) or diluted:

- For each identified motif, compute its **failure attribution**: the fraction of total test-set failures (e.g., predictions with $|\hat{E} - E| > 1$ kcal/mol) that match the motif.
- Sort motifs by attribution; report a **Pareto curve** (rank vs. cumulative attribution).
- State the **compactness ratio**: the smallest $K$ such that the top $K$ motifs cover $\geq 80\%$ of failures. A small $K$ (say, $\leq 5$) indicates compact, interpretable failure modes; a large $K$ indicates the failures are diluted across the motif library.

### Per-motif justification

For any motif identified as a failure mode, the paper must contain a methodological justification connecting the motif's structural / chemical properties to the hierarchical decomposition itself — for instance, why does this particular structural class produce non-decaying $\hat{E}_i^n$? Generic statements ("this motif is harder for the model") do not satisfy the principle. The form of the justification is up to you; what matters is that it ties a property of the motif to a property of the hierarchy.

### Conformational transfer of the motif library

The motif library you identify is conditioned on **minimized geometries** — every QM9 configuration was the energy-minimized structure under B3LYP/6-31G(2df,p). This is a load-bearing assumption: models trained on minimized geometries are typically not robust to other conformations of the same molecules, and a motif library identified on minimized chemistry may or may not predict failures on the conformations a molecule would actually see in molecular dynamics, transition-state characterization, or high-throughput screening. The contract therefore requires the submission to address the conformational transfer of the motif library — empirically on ANI-1x or rMD17 for the substantial outcome, or at minimum through substantive analytical engagement with the limitation for the floor.

**Protocol.** Take your motif library (identified on the 9-heavy-atom QM9 subset of minimized geometries) and apply it to a conformational-variance dataset of a subset chemistry (HCNO). Two pre-staged options:

- **ANI-1x** (`data/ani1x/`) — the canonical conformationally-varying training set from that research group, used for active-learning work. ~5M off-equilibrium conformations at the ωB97X/6-31G(d) DFT level. HCNO chemistry, a subset of QM9's HCNOF, but covers conformational space rather than just minimized geometries. The DFT functional differs (ωB97X/6-31G(d) vs. QM9's B3LYP/6-31G(2df,p)), so absolute energies are on different scales — for the transfer test, the relevant question is whether motif-flagged molecules show systematically larger *relative* errors, not whether absolute energies match. Recommended for the substantial transfer test.

- **rMD17** (`data/rmd17/`) — revised MD17, 100k MD-sampled conformations each of 10 small organic molecules at PBE/def2-SVP. Smaller and more focused than ANI-1x; useful for a smaller-scope transfer test if ANI-1x integration is too heavy within the budget. Note that the 4 HIP-NN-relevant rMD17 molecules (benzene, malonaldehyde, toluene, uracil) are all $\leq 8$ heavy atoms, so they're in your HIP-NN's training distribution by heavy-atom count — the test is whether the motif library still flags failures on their MD-sampled conformations.

**What to report.** For each motif in your library, evaluate its **transfer rate**: on the conformational-variance dataset, do molecules matching the motif show systematically larger HIP-NN prediction errors than molecules not matching the motif? Report per-motif transfer rates with statistical support, and characterize which motifs transfer cleanly (the failure mode is a property of the chemistry, not the geometry distribution) versus which do not transfer (the motif's failure signature was specific to minimized geometries and doesn't predict MD-conformation failures). Discuss the implication for the motif library's downstream usability in non-minimized-geometry contexts.

The transfer test executed substantively on ANI-1x is the substantial outcome; the same test on rMD17 (smaller scope) is the modest outcome; the test omitted with only an analytical discussion of the limitation is the floor; the limitation not addressed at all is a contract failure.

## Rules — Here are the rules

### Available resources

The following are provided in the working directory:

- `paper.md`, `paper.pdf` — Lubbers, Smith, Barros 2018, the source paper.
- `images/` — figures extracted from the source paper.
- `code/hippynn/` — the LANL hippynn package, the modern PyTorch implementation of HIP-NN maintained by the Lubbers group. **Primary working dependency.** Contains reference implementations of the HIP-NN architecture, hierarchical regularization, training loops, and the QM9 data loaders. The original 2018 paper used Theano + Lasagne; hippynn is the PyTorch successor that the authors recommend for current work. Inspect this before writing training code.
- `data/qm9/` — QM9 dataset. The standard $\sim$134k-molecule organic-chemistry benchmark, pre-staged from its figshare release. Apply the paper's standard pruning ($\sim$3k geometric-consistency failures + 11 minimization failures) to reach the $\sim$131k-molecule clean dataset before train/test partitioning.
- `data/rmd17/` — revised MD17 (Christensen & von Lilienfeld 2020), the duplicate-pruned and re-DFT'd successor to the original MD17 trajectory dataset. PBE/def2-SVP precision via ORCA. Available as a conformational-extrapolation supplement for the motif library's transfer test (smaller-scope alternative to ANI-1x); the original MD17 has a known 0.1 kcal/mol energy rounding issue that makes some published MAE comparisons unreliable. Any MAE comparisons with the paper's published numbers should be anchored on QM9, not rMD17. rMD17 results count as supporting evidence for the conformational transfer test, not as a substitute for the QM9 in-distribution comparison.
- `data/ani1x/` — ANI-1x dataset (Smith, Zubatyuk, Nebgen, Lubbers, Barros, Roitberg, Isayev, Tretiak — Sci. Data 7, 134, 2020). ~5M off-equilibrium conformations of organic molecules at the ωB97X/6-31G(d) DFT level, sampled via active learning from chemical space. HCNO chemistry, a subset of that in QM9, but covers conformational space rather than just minimized geometries. ANI-1x is the canonical conformationally-varying training set from that research group, and is the recommended dataset for the paper-1 follow-up's conformational transfer test. Extraction requires the `ani1x_datasets` package (github.com/aiqm/ANI1x_datasets) — the file is HDF5 with a specific schema; the package provides the loaders. DFT functional differs from QM9 (ωB97X/6-31G(d) vs. B3LYP/6-31G(2df,p)), so absolute energies are on different scales; for the transfer test, the relevant signal is whether motif-flagged molecules show systematically larger *relative* errors than non-flagged ones.
- `proposal/` — your output goes here, organized as `proposal/code/`, `proposal/logs/`, `proposal/codebase_notes.md`, `proposal/attempts_log.md`, with the final paper at `proposal/report.tex` (and rendered `proposal/report.pdf`).

### Compute budget

- **Hardware**: 1× NVIDIA H100 80GB, 16 vCPU, 128 GB RAM, 500 GB disk.

If compute pressure forces tradeoffs:
- **Priority 1**: train at least one reimplemented HIP-NN at the held-fixed architecture on the $\leq 8$-heavy-atom subset (the reimplementation baseline + the basis for motif analysis).
- **Priority 2**: motif identification + uncertainty calibration on that single trained model (the headline scientific content).
- **Priority 3**: additional seeds to demonstrate that the motif library and calibration are stable across random initializations.
- **Priority 4**: motif compactness analysis at finer granularity.

Document any compute-driven choices transparently in Methodology. Submissions that finish early without exhausting the budget typically have unrealized improvements.

### Autonomy and external resources

**Internet search is permitted and encouraged.** Use it for literature lookup, library documentation, citations of post-HIP-NN MLIP uncertainty literature (e.g., ensemble-based UQ, Bayesian neural-network MLIPs, dropout-based UQ, conformal-prediction MLIPs), and identification of post-2018 developments in molecular-property prediction. Cite anything you use.

**Workspace constraint.** You may not read local files outside the provided working directory.

**No DFT generation.** No quantum chemistry engine is provided in the workspace, and you may not run new DFT calculations or quantum chemistry of any kind. The training and evaluation data are restricted to the pre-staged QM9, rMD17, and ANI-1x datasets. Adding new molecules via DFT-of-your-choice would change the study from "characterizing the hierarchy's failure on existing chemistry" to "designing a custom probe set" — out of scope under this contract.

**Anti-replication.** Independent researchers working on MLIP uncertainty will naturally converge on some shared methodologies — that kind of convergence is fine and expected. What is not allowed is doing exactly what a published method already did: re-implementing the same uncertainty methodology, the same motif library, the same calibration protocol, and presenting the result as a new contribution. The contract evaluates your scientific work, not your literature search. If the evaluator finds that your submission is substantively a re-implementation of an existing published uncertainty-for-MLIP study, your submission will be severely degraded. Acknowledged convergence with prior work is fine; silent or misrepresented substantive overlap is not.

**No questions.** Do not query the user for clarification. Make your own design decisions and justify them in your write-up.

### No reward hacking

The following checks will be applied to your submission. They are disclosed up front so the contract is clear.

1. **Number traceability.** Every quantitative claim in the paper (MAE values, motif coverages, calibration errors, precision/recall numbers, attribution percentages) must trace to a log entry or to a specific citation in the source paper. Numbers without a verifiable source will be flagged.

2. **Two-baseline integrity.** Paper-as-reported HIP-NN values must trace to specific tables/figures in Lubbers, Smith, Barros 2018 (e.g., "Table I, row $N_{\text{train+val}} = 110{,}426$"). Reimplemented HIP-NN values must trace to your own hippynn training logs. Mixing the two — citing your reimplementation as if it were paper-as-reported, or vice versa — will be flagged.

3. **Pre-stated method.** The motif-identification methodology, the uncertainty model definition, and the calibration protocol must be stated in Methodology before any empirical results are described.

4. **Held-fixed disclosure.** Any deviation from the strictly-held-fixed items (chemistry, dataset pruning, train/test split by heavy-atom count, HIP-NN architecture, training protocol, energy-only training) must be transparently disclosed in Methodology. Silent or misrepresented deviations will be flagged.

5. **Train/test split integrity.** The $\leq 8$-heavy-atom training set must not include any 9-heavy-atom molecules; the 9-heavy-atom test set must not include any $\leq 8$-heavy-atom molecules; the validation split used for early stopping must come from within the training set ($\leq 8$ heavy atoms), not from the test set. Logs must demonstrate the partition.

6. **No selective reporting.** Motif library entries must include all motifs identified in the analysis, including ones that turn out to have low statistical support — not just the ones that look clean. Calibration curves must show the full range of predicted-uncertainty bins, not just the favorable middle.

7. **Calibration honesty.** ECE and precision/recall numbers must be computed on **held-out** chemistry. If you use any of the 9-heavy-atom test molecules to tune the uncertainty model's hyperparameters (e.g., choosing how to combine $R$ with other signals), that subset must not be used for calibration evaluation. The calibration protocol must include a clean held-out split.

8. **Motif statistical support.** Each reported motif must have stated test-set coverage and a statistical-significance test comparing its conditional error distribution against the non-matching molecules. Motifs identified from $\leq 5$ test-set molecules will be flagged as not having statistical support unless explicitly disclosed as illustrative cases.

9. **Code–paper alignment.** The HIP-NN architecture and training procedure described in Methodology must match what hippynn / your code actually runs. Hyperparameters quoted in the paper must match the logs.

10. **No hardcoded reference values.** Code must not contain hardcoded DFT energies, hardcoded molecular descriptors, hardcoded motif assignments derived from external sources, or any quantity that bypasses actual computation.

11. **No DFT generation.** Your submission must not contain quantum chemistry calculations you ran yourself. Training data is QM9 as provided; supplementary evidence and the conformational transfer test may come from rMD17 or ANI-1x (also as provided). The held-fixed training data clause is checked against your data-loading code paths.

12. **Codebase inspection.** Document in `codebase_notes.md` what you inspected in the hippynn package before writing your own training pipeline. Reimplementing from scratch what hippynn already provides (HIP-NN architecture, hierarchical regularization, QM9 data loaders) is wasted effort and a comparison-quality risk; if you did reimplement, document why.

13. **No replication of published uncertainty-for-MLIP methods.** Your uncertainty model and motif-identification methodology must not be a re-implementation of a published uncertainty-for-MLIP study (e.g., named ensemble-based UQ, dropout-based UQ, conformal prediction for molecular energies, learned uncertainty heads on existing MLIPs). Acknowledged convergence with prior work is fine; silent substantive overlap is not (see "Anti-replication" above).

14. **Effort evidence.** If you describe any work as partial, unresolved, or attempted-but-failed (e.g., a motif you tried to define but couldn't operationalize, an uncertainty signal you experimented with but didn't include), `proposal/attempts_log.md` must document the approaches tried and why each was set aside.

15. **Consistent calibration reporting.** Calibration metrics (ECE, precision, recall) must use consistent definitions and consistent error-threshold conventions across the paper. If you switch between "high-error" $> 1$ kcal/mol and "high-error" $> 2$ kcal/mol in different sections, this must be disclosed and the rationale stated.

16. **Reimplementation floor across training sizes.** At every training-set size you report a HIP-NN run, the reimplemented MAE must be at or below the paper-as-reported MAE at the comparable size (with reasonable extrapolation across training-size differences between your $\leq 8$-heavy-atom subset and the paper's full-QM9 training sizes). Reimplementations that miss the floor at some training sizes must be disclosed; reimplementations that miss the floor at every reported size without disclosure are flagged as silent under-convergence and weaken the motif analysis's anchoring. Active misrepresentation (citing paper-as-reported numbers as if they were the agent's reimplementation) is a more severe violation.

17. **Motif identification protocol integrity.** Motif definitions must be derived from structural / chemical features applied independently of test-set per-molecule error magnitudes. If you derived motif candidates by inspecting which test molecules have high prediction errors and matching their structural features (post-hoc identification), the post-hoc nature must be disclosed *and* you must either (a) validate motif predictions on a separate held-out subset with the discovery and validation splits clearly separated, or (b) apply post-selection inference statistics (conditional p-values, sample-splitting estimators) with stated methodology. Silent post-hoc identification — where the motif library was fit to test-set residuals and presented as predictive — is the K3-direction analog of test-set leakage at the methodology-design level and will be flagged as a serious integrity violation.

18. **Multiple-testing correction.** Motif-significance claims must account for testing and selecting among candidate motif definitions, including candidates recorded in `attempts_log.md`. When multiple hypotheses support inferential claims, control the family-wise error rate over the relevant hypothesis family using Holm, Bonferroni, or another procedure that provides family-wise error control under the applicable assumptions. State the hypothesis family, significance level, correction method, and treatment of outcome-dependent hypothesis selection. Benjamini–Hochberg FDR alone does not satisfy this requirement. Preregistration identifies the planned hypotheses but does not, by itself, waive multiplicity control. A genuinely single prespecified test does not require a multiple-testing adjustment. Exploratory effect-size summaries may be reported descriptively, but they do not substitute for multiplicity-controlled inferential evidence where such evidence is required. Reporting selected uncorrected p-values as though no selection or multiplicity occurred, while concealing the actual procedure, is an integrity concern; a disclosed inadequate correction is a scientific shortfall, not by itself deception.

19. **Conformational transfer sampling honesty.** For the conformational transfer test (on ANI-1x or rMD17), the evaluation subset must be selected representatively rather than hand-picked. Acceptable selections: the full available dataset, a representative random sample with stated RNG seed and sample size, or a stratified sample with stated stratification rule and rationale. A hand-picked subset where the motif library happens to transfer well — without disclosure of the selection — will be flagged.

## Deliverables — Here's what to hand in

1. **Codebase and logs.** Code organized under `proposal/code/`; logs under `proposal/logs/`. Logs must include:
   - Per-seed HIP-NN training run records: hyperparameters, RNG seeds, per-epoch training/validation MAE, intermediate-epoch checkpoints if used, final-epoch metrics, timestamps.
   - The trained model checkpoints (one per seed) — small enough that they can be saved alongside the logs.
   - Per-test-molecule outputs at evaluation: predicted energy, non-hierarchicality $R$, per-atom per-order hierarchical contributions $\hat{E}_i^n$, any additional uncertainty signals you compute. This per-molecule output is the raw material for both motif identification and calibration.
   - Motif-assignment logs: for each test molecule, which motifs it matches (under your operationalized motif definitions).
   - Calibration analysis logs: binned predicted-uncertainty vs. actual-error data, computed ECE and precision/recall numbers with the seed-resolved breakdown.
   - A `proposal/logs/paper_reference.md` (or equivalent) listing every paper-as-reported value cited in your paper with its specific table or figure source in Lubbers, Smith, Barros 2018.

2. **Follow-up paper.** A LaTeX paper at `proposal/report.tex` rendered to `proposal/report.pdf`, using the NeurIPS style file (`neurips.sty`, provided in the workspace). The paper should read as a self-contained journal-style contribution and must include:
   - **Motivation** for the failure-characterization study — why the long-tail-of-errors and uncertainty-calibration questions matter for HIP-NN usage downstream.
   - **A description of the experimental setup**: the train ($\leq 8$ heavy atoms) / test ($= 9$ heavy atoms) QM9 partition, the held-fixed HIP-NN architecture, the number of independent seeds, the two-baseline rows, the motif-identification methodology, the uncertainty model definition, the calibration protocol.
   - **A description of the motif-identification methodology**: precise enough that a reader can reproduce it. Whether you use clustering on hierarchical components, hand-coded structural descriptors, learned classifiers, or some combination — and which cheminformatics libraries (RDKit, etc.) you use.
   - **A description of the uncertainty model**: precise definition (including any extensions of $R$), with the rationale for each signal included.
   - **A primary results table** labeled `\label{tab:main_results}` with two-baseline rows. Suggested structure (you may adapt column names and add rows for variants of your method):

     | Method | Training subset | $N_{\text{train}}$ | MAE on test (kcal/mol) | ECE on test | Compactness $K$ (≥80% attribution) |
     |---|---|---|---|---|---|
     | HIP-NN (paper as reported) | full QM9, 110k | 110,426 | $0.256 \pm 0.003$ (Table I) | — | — |
     | HIP-NN (paper as reported) | full QM9, 50k | 50,000 | $0.354 \pm 0.004$ (Table I) | — | — |
     | HIP-NN (reimplemented) | $\leq 8$ heavy atoms | [agent's count] | [agent's MAE] | — | — |
     | This work | $\leq 8$ heavy atoms | [agent's count] | [same as reimpl.] | [agent's ECE] | [agent's $K$] |

     Numbers in `[brackets]` are placeholders; report actual values. The "This work" row uses the same trained HIP-NN as the reimplemented row — the difference is that "This work" adds the uncertainty-model and motif-analysis outputs.

   - **A motif library table** with at least 3 motifs, each with definition, coverage, conditional error distribution, and significance support.
   - **A calibration figure**: predicted-uncertainty bin vs. actual mean error, with the identity line overlaid; the ECE value annotated.
   - **A Pareto figure for motif compactness**: cumulative attribution vs. motif rank, with the compactness $K$ annotated.
   - **A mechanism discussion**: why does each identified motif drive hierarchical decomposition failure? Connect motif structure to the architecture's behaviour (which $n$ orders fail to decay, what aspects of the structural environment produce the failure).
   - **A failure analysis**: which 9-heavy-atom molecules are *not* captured by your motif library, and what does that suggest? Where does your uncertainty model fail (high uncertainty / low actual error, or vice versa) and what does the failure reveal about the model's assumptions?
   - **Computational-cost discussion**: per-seed wall-clock for HIP-NN training, per-test-molecule wall-clock for uncertainty + motif evaluation. State the cost-quality tradeoff if your method extends $R$ substantially.
   - **Engagement with related work**: position your uncertainty model and motif identification relative to the published MLIP-uncertainty literature (ensemble UQ, dropout UQ, Bayesian neural-network MLIPs, conformal prediction for molecules, etc., including post-2018 developments). Cite the families of approaches your method is closest to and explicitly differentiate your contribution.
   - **Limitations and conclusions**.

   Appendix material is for supplementary tables and figures that did not fit in the main body. The substantive contract — methodology, reimplemented MAE, motif library, calibration analysis, compactness analysis, mechanism, failure — must live in the main paper, not the appendix.

3. **Supporting documents.**
   - `proposal/codebase_notes.md` — a record of what you found and reused in the hippynn package: which components are usable, which you adapted, which you bypassed. The notes don't need to be exhaustive — a few paragraphs is enough — but they must demonstrate that you inspected what was available before writing code.
   - `proposal/attempts_log.md` — distinct approaches tried during the work, including motifs you considered but discarded, uncertainty signals you experimented with but didn't include, calibration protocols you tried but rejected. Required whenever the submission describes any work as partial or unresolved.

### Paper quality

Your submission is an academic paper, not a class project report. The writing should resemble what would be reviewed at a top scientific journal in structure, integrity, depth, and language: a clear thesis stated up front, claims supported by quantitative evidence, prose rather than bullet enumeration in main body sections, careful citation, and explicit honest discussion of limitations.

**Independence from this task spec.** Your paper should read as a self-contained journal contribution to a reader who has not seen this task spec. Do not transliterate spec terminology, do not adopt the spec's section structure verbatim, do not use placeholder names like "the proposed motif library" or "the proposed uncertainty model" when the contribution has a name. The section content described under "Deliverables" specifies what the paper must *contain*, not the titles or order in which the content must appear. Organize the paper however best serves the contribution.

**Rendered output verification.** Before submitting, render your LaTeX to PDF (e.g., `pdflatex report.tex`) and visually inspect the resulting PDF. Equations must not overflow page margins; tables must fit within the text width without column wrapping that breaks readability; figures must not be cut off. A submission with formatting issues in the rendered PDF will be flagged as not having been visually verified.

Good luck.
