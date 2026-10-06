# Science task overview

The identifiers P1–P15 below define the directory order in this repository. Each task starts from an original paper and a follow-up direction proposed by its domain scientist. Titles and direction summaries follow the supplied manuscript table; the complete task specification, not this one-line summary, defines the requirements.

## Group A: published in 2024 or earlier

| ID | Original paper | Scientist-proposed follow-up direction | Original archive directory |
| --- | --- | --- | --- |
| [P1](tasks/paper1/task_spec.md) | Hierarchical modeling of molecular energies using a deep neural network | Use HIP-NN's hierarchy-breakdown signal to identify failure motifs and calibrate uncertainty estimates that transfer across conformations. | `paper1` |
| [P2](tasks/paper2/task_spec.md) | Extending the accuracy of the SNAP interatomic potential form | Replace SNAP bispectrum components with a learned descriptor and test its accuracy, data efficiency, and cross-material transfer. | `paper2` |
| [P3](tasks/paper3/task_spec.md) | Convolutional Dictionary Learning: A Comparative Review and New Algorithms | Unify CDL dictionary-update algorithms within one multichannel derivation framework and use it to reveal a new method and equivalence. | `paper10` |
| [P4](tasks/paper4/task_spec.md) | A Kriging-based approach to autonomous experimentation with applications to X-ray scattering | Develop an acquisition strategy that reconstructs synthetic functions faster than SMART and gSMART. | `paper3` |
| [P5](tasks/paper5/task_spec.md) | Adaptive variational quantum dynamics simulations | Make AVQDS compatible with fixed nearest-neighbor qubit topology while measuring the resulting accuracy and resource overhead. | `paper4` |
| [P6](tasks/paper6/task_spec.md) | Simulations of relativistic-quantum plasmas using real-time lattice scalar QED | Stress-test variational lattice scalar QED in new valid and fundamentally invalid regimes, then diagnose and repair its failures. | `paper8` |
| [P7](tasks/paper7/task_spec.md) | The non-Riemannian nature of perceptual color space | Test whether perceptual diminishing returns are organized by cone channels rather than global scaling or generic anisotropy. | `paper9` |
| [P8](tasks/paper8/task_spec.md) | Parameterized Physics-informed Neural Networks for Parameterized PDEs | Extend P²INN into a problem-specification encoder that generalizes jointly across coefficients, initial conditions, and boundary conditions. | `paper19` |

## Group B: submitted in 2026

| ID | Original paper | Scientist-proposed follow-up direction | Original archive directory |
| --- | --- | --- | --- |
| [P9](tasks/paper9/task_spec.md) | An efficient explicit implementation of a near-optimal quantum algorithm for simulating linear dissipative differential equations | Improve the mathematical LCHS algorithm by substituting a kernel with better asymptotic resource scaling. | `paper14` |
| [P10](tasks/paper10/task_spec.md) | Efficient Berry Phase Calculation via Adaptive Variational Quantum Computing Approach | Determine how much realistic hardware noise the AVQDS Berry-phase method can tolerate before its topological output fails. | `paper17` |
| [P11](tasks/paper11/task_spec.md) | Projected Hessian Learning: Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials | Reduce the stochastic-probe requirement of Projected Hessian Learning without sacrificing estimator or downstream model quality. | `paper11` |
| [P12](tasks/paper12/task_spec.md) | gp2Scale: Compactly-Supported Non-Stationary Kernels and Distributed Computing for Exact Gaussian Processes on 10 Million Data Points | Apply exact large-scale Gaussian processes to a new domain using a domain-specific kernel or sparsity construction. | `paper16` |
| [P13](tasks/paper13/task_spec.md) | Particle-in-cell simulations of laser crossbeam energy transfer via magnetized ion-acoustic wave | Map the boundary where Vlasov theory ceases to predict MagCBET PIC behavior and identify the first invalid assumption. | `paper13` |
| [P14](tasks/paper14/task_spec.md) | Simulating plasma wave propagation on a superconducting quantum chip | Add ZZ interactions to the spin-chain model and test whether it quantitatively reproduces nonlinear plasma dynamics. | `paper18` |
| [P15](tasks/paper15/task_spec.md) | Beyond Classical Molecular Dynamics with Layered Interatomic Potentials | Replace the Sommerfeld potential's heuristic sigmoid with a physically principled switching function. | `paper15` |

## File layout and historical IDs

For each Pn, `tasks/papern/` holds `task_spec.md`, the original paper (`paper.pdf`, its OCR text `paper.md` and the extracted figures `images/`), and, after `python prepare_tasks.py`, the `code/` and `data/` given to the agent. `eval_artifacts/papern/` holds `direction_specific_rubric.json`, `integrity_check_rubric.json` and the self-contained `grading.py`.

The old-to-new ordering is **1, 2, 10, 3, 4, 8, 9, 19, 14, 17, 11, 16, 13, 18, 15**. Task text can still mention an old paper identifier; use the mapping above.
