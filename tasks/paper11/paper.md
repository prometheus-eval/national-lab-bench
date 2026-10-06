# Projected Hessian Learning: Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials

Austin Rodriguez^{1,2}, Justin S. Smith^{3}, Sakib Matin^{1}, Nicholas Lubbers^{4,*}, Kipton Barros^{1,5,*}, and Jose L. Mendoza-Cortes^{2,6,*}
^{1}Theoretical Division, Los Alamos National Laboratory, Los Alamos, New Mexico 87545, United States.
^{2}Department of Chemical Engineering & Materials Science, Michigan State University, East Lansing, Michigan 48824, United States.
^{3}NVIDIA Corp., 2788 San Tomas Expy, Santa Clara, California 95051, United States.
^{4}Computer and Artificial Intelligence Division, Los Alamos National Laboratory, Los Alamos, New Mexico 87545, United States.
^{5}Center for Nonlinear Studies, Los Alamos National Laboratory, Los Alamos, New Mexico 87545, United States.
^{6}Department of Physics and Astronomy, Michigan State University, East Lansing, MI, 48823, United States.
^{*}Corresponding authors: jmendoza@msu.edu, kbarros@lanl.gov, nlubbers@lanl.gov

###### Abstract

The Hessian matrix of the second derivatives contains substantially richer information about the local geometry of the potential energy surface than energies and forces alone. Although incorporating full Hessians into machine-learning interatomic potential (MLIP) training can significantly improve accuracy and robustness, the quadratic computational and memory cost of explicitly constructing and storing Hessian matrices has limited its practical use.

Here, we introduce *Projected Hessian Learning* (PHL), a scalable second-order training framework that incorporates curvature information using only Hessian–vector products (HVPs). By avoiding explicit Hessian construction and instead projecting curvature along stochastic probe directions, PHL reduces the cost of second-derivative supervision to near force-level complexity. The resulting stochastic loss, based on a Hutchinson trace estimator, is unbiased and exhibits favorable scaling with system size, enabling curvature-informed training without quadratic memory growth.

We evaluated various training approaches on a chemically diverse dataset of reactants, products, transition states, intrinsic reaction coordinates, and normal-mode sampled geometries generated at the theoretical level $\omega$B97XD / 6-31G(d). Four training schemes are considered: energy-force training (E-F), two HVP-based approaches (E-F-HVP with one-hot or Hutchinson vectors) and energy-force-Hessian training (E-F-H). When probe vectors are randomized for each minibatch, both HVP-based approaches achieve statistically indistinguishable energy, force, and Hessian accuracy relative to full Hessian training while providing more than 24$\times$ speedups per epoch for the small molecular systems studied here. In a more realistic fixed-vector regime with only one HVP per molecule, Hutchinson projections consistently outperform one-column probing, particularly for far-from-equilibrium geometries.

Overall, PHL effectively replaces explicit Hessian supervision with force-complexity curvature training, retaining most of the accuracy benefits of full second-order methods while enabling scalable MLIP development for larger and more complex molecular systems.

Keywords: Machine-Learning Interatomic Potentials (MLIPs), Hessian-Vector Products (HVPs), Projected Hessian Learning (PHL), Curvature supervision, Second-derivative training, Stochastic trace estimation, Hutchinson estimator, Reactive potential energy surfaces, Transition states and reaction pathways, Data-efficient training

# 1 Introduction

![images/image1.jpg](images/image1.jpg)
Figure 1: Conceptual comparison of curvature supervision strategies used to train MLIPs. Left: Full Hessian training explicitly uses all  $(3N) \times (3N)$  second-derivative elements of  $M$  systems. Middle: One-column HVP probing samples a single Hessian column via a canonical basis vector. Right: Projected Hessian Learning (PHL) (this work). Curvature is enforced through stochastic projections using random probing vectors to form Hessian-vector products, yielding random weighted combinations of Hessian columns (Hutchinson-style probing). Both HVP-based strategies avoid explicit Hessian construction; PHL aggregates information across multiple curvature directions in expectation.

The quality of training data often imposes a fundamental limit on the accuracy of a machine learning interatomic potential (MLIP).[1] In neural-network-based MLIPs, reference data are typically obtained from quantum chemistry calculations such as density functional theory (DFT) or

Coupled Cluster (CC), and the ML model is trained to reproduce these reference energies while enabling orders-of-magnitude faster inference.*[1]* A key insight established across multiple MLIP frameworks is that including gradients of the energy with respect to atomic coordinates (i.e. atomic forces) in the training loss substantially improves both accuracy and generalization. This paradigm has been realized in a wide range of architectures, including Behler–Parrinello neural networks,*[2]* ANI models,*[3, 4, 5, 6]* HIP-NN,*[7]* the DeepMD framework,*[8]* kernel/descriptor-based MLIPs such as Gaussian Approximation Potentials (GAP),*[9]* Spectral Neighbor Analysis Potentials (SNAP),*[10]* and Moment Tensor Potentials (MTP),*[11]* as well as message-passing approaches such as SchNet and NequIP.*[12, 13]* More recent equivariant message-passing architectures, e.g. PaiNN and MACE, *[14, 15]* further reinforce the central role of force supervision in improving data efficiency and robustness. In all of these works, joint training on energies and forces consistently yields lower prediction errors and improved robustness compared to training on energy alone,*[16, 17]* reflecting the strong regularizing effect and the enhanced efficiency of the data provided by the first-derivative information. While energy and force training has become standard, response properties that depend explicitly on second derivatives can remain inaccurate even when energies and forces are well reproduced. In crystalline materials, for example, harmonic phonons are determined by the (mass-weighted) force-constant matrix, i.e., the energy Hessian, and recent benchmarks show that several high-performing universal MLIPs still exhibit substantial errors in phonon observables despite strong energy and force accuracy.*[18]*

A well-implemented quantum chemistry code can produce not just the total energy but also the gradient of energy with respect to atom positions. Indeed, the principle of reverse-mode automatic differentiation,*[19, 20]* known as backpropagation in machine learning, ensures that all force components can be calculated at a cost comparable to that of the energy evaluation alone; in electronic structure theory this is closely related to analytic energy-derivative techniques used to obtain nuclear forces.*[21]* Consequently, it is now standard for DFT codes to provide forces in addition to energies. The situation is different, however, for the Hessian matrix of second derivatives,

$H_{(i,\alpha),(j,\beta)}=\frac{\partial^{2}E}{\partial r_{i}^{\alpha}\partial r_{j}^{\beta}},$ (1)

where $r_{i}^{\{x,y,z\}}$ denotes components of the position of the atom $i$. Elements of the Hessian can be understood as the sensitivity of the force on an atom $i$ with respect to perturbations to the position of an atom $j$. Calculating all such sensitivities is substantially slower than calculating forces alone, as analytic second derivatives generally require additional response equations beyond those used for gradients.*[21, 22]*

For a system of $N$ atoms, the total energy is a single scalar, whereas the full gradient is a $3N$-dimensional vector, dramatically expanding the informational content available for training. This naturally raises the question of whether second derivatives of the energy, i.e. curvature information, can further improve the accuracy and generalization of the MLIP. The second derivative matrix, or Hessian, contains $(3N)^{2}$ components and governs the local geometry of the potential energy surface (PES), including vibrational frequencies, transition-state curvature, and reaction pathways.*[21, 23, 24]* Motivated by these physical insights, recent work has begun to explore Hessian-informed approaches in ML potential development, including direct Hessian learning,*[25]* analytical ML Hessians for the characterization of transition-states,*[26]* Hessian distillation frameworks for specialized force fields,*[27]* and the use of Hessians as diagnostic tools to assess force locality in MLIPs.*[28]* Very recently, Koker et al. proposed phonon fine-tuning (PFT), which directly supervises second-order force constants by matching MLIP energy Hessians to DFT-derived force constants from finite-displacement phonon workflows.*[29]* In a complementary direction, Burger et al. introduced Hessian

Interatomic Potentials (HIP), which predict symmetry-preserving Hessians directly from equivariant message-passing features without relying on automatic differentiation or finite differences.*[30]*

Complementing these methodological developments, the availability of curvature-labeled datasets is rapidly expanding. For equilibrium molecular structures, Hessian QM9 provides numerical DFT Hessians for 41,645 QM9 molecules (including vacuum and implicit-solvent environments), enabling systematic evaluation of Hessian-aware training for vibrational properties.*[31]* For reactive chemistry, the recently released HORM database scales this idea dramatically, reporting 1.84 million quantum-chemistry Hessian matrices for diverse reactive geometries and transition-state workflows.*[32]* The dataset used in this work complements these resources by targeting chemically diverse reaction pathways with equilibrium reactants, products, and transition states (RTP), continuous intrinsic reaction coordinate trajectories (IRC), and far-from-equilibrium normal-mode perturbations (NMS), providing a controlled arena to assess how curvature supervision impacts both interpolation and extrapolation in reactive MLIPs.*[33]*

Despite this promise, incorporating full Hessian information presents substantial practical challenges. First, acquiring Hessian data from quantum chemistry calculations can be prohibitively expensive for many electronic structure methods, particularly beyond mean-field levels where analytic second derivatives are substantially more complex or unavailable.*[21, 22]* Second, training MLIPs with explicit Hessian supervision is significantly more expensive than energy–force training alone.*[25]* Although gradients can be computed efficiently using reverse-mode automatic differentiation, there is no general method to form all Hessian elements with comparable efficiency.*[19]* Moreover, storing the Hessian requires $(3N)^{2}$ floating-point values per structure, leading to quadratic memory scaling with system size, which rapidly becomes a bottleneck for both data generation and GPU-based neural network training. These combined computational and memory costs severely limit the direct use of full Hessians in large-scale MLIP workflows, motivating alternative strategies that retain essential curvature information without explicitly constructing the Hessian.

In this work, we introduce Projected Hessian Learning (PHL) as a stochastic strategy to efficiently incorporate curvature information into the training of machine learning interatomic potentials (Fig. 1). The central idea in PHL is to avoid explicit construction of the full Hessian matrix and instead supervise projected curvature information through Hessian-vector products (HVPs) of the learned potential energy surface, as schematized by the randomized probing strategy in Fig. 1. The complete PHL training loop is summarized in Appendix A. Within an automatic differentiation framework used to train MLIPs, HVPs can be efficiently computed using forward-over-reverse or reverse-over-reverse differentiation without explicitly forming the Hessian.*[19, 34]* HVP-based methods are also foundational in large-scale second-order optimization (e.g. Hessian-free and Newton conjugate gradient methods) precisely because they provide curvature access without forming Hessians.*[35, 36]* In terms of floating-point operations, a single HVP can often be evaluated at a cost comparable to a small constant number of gradient evaluations, regardless of the size of the system $N$.*[34, 36]* Finally, PHL’s use of low-dimensional probing subspaces is closely related to randomized “sketching” and low-rank approximation ideas for implicitly accessed matrices.*[37, 38]*

## 2 Mathematical Background

### 2.1 Stochastic estimation of the Hessian loss

We can use the Hutchinson trace estimator to work with Hessian-vector products instead of the expensive calculation of the full Hessian matrix.*[39]* This estimator says that given any matrix $A$, one can form an unbiased approximation to the trace using only the matrix-vector product;

$\text{tr}\,A\approx v^{T}Av.$ (2)

Here, $v$ is a random vector sampled from a distribution with components that are not correlated and have unit variance;

$\left\langle v_{i}v_{j}\right\rangle=\delta_{ij}.$ (3)

The unbiased nature of this approximation can be verified by applying the definition of the Kronecker symbol $\delta_{ij}$ and the linearity of the expectation values $\left\langle\cdot\right\rangle$,

$\text{tr}\,A=\sum_{i}A_{ii}=\sum_{ij}\delta_{ij}A_{ij}\approx\sum_{ij}\left\langle v_{i}v_{j}\right\rangle A_{ij}=\left\langle v^{T}Av\right\rangle.$ (4)

A simple and reasonable choice that satisfies Eq. (3), as suggested by Hutchinson, is to independently randomize each component $v_{i}=\{\pm 1\}$. Other choices are possible and will be considered below.

Equation 2 can be incorporated into MLIP training as follows: First, consider a training loss function that involves energy, force, and Hessian loss terms,

$\mathcal{L}=\lambda_{E}\mathcal{L}_{E}+\lambda_{F}\mathcal{L}_{F}+\lambda_{H}\mathcal{L}_{H}.$ (5)

The hyperparameters $\lambda_{\{E,F,H\}}$ describe the relative weighting of these loss terms. Frequently, the $l2$ loss on components would be used. For example, a molecule of $N$ atoms may appear in the loss as

$\mathcal{L}_{H}=\frac{1}{(3N)^{2}}\sum_{i,j=1}^{3N}\left(\tilde{H}_{ij}-H_{ij}\right)^{2}.$ (6)

The symbol $\tilde{H}_{ij}$ will denote the Hessian of the ML-predicted energy $\tilde{E}$, which has an implicit dependence on the model parameters $\theta$. The symbol $H_{ij}$ will denote true (reference) quantum chemistry Hessian data. It is our goal to train the model parameters to minimize the loss. In particular, stochastic gradient descent requires one to calculate $\nabla_{\theta}\mathcal{L}_{H}$. Clearly, this seems to require performing the complete sum on the components of the $(3N)^{2}$ matrix $\tilde{H}_{i,j}$.

Let us now review the construction of an efficient stochastic approximation. Equation (6) can be interpreted as a matrix trace,

$\mathcal{L}_{H}=\text{tr}A.$ (7)

using the following definitions:

$A$ $=B^{T}B$ (8)
$B$ $=\frac{\bar{H}-H}{3N}.$ (9)

The matrix $B$ denotes the squared error in the Hessian predicted by the ML model $\tilde{H}$. One could also write $A=B^{2}$ since Hessian matrices are symmetric (i.e. derivatives with respect to atom positions commute). From the matrix trace, one can readily obtain a stochastic approximation,

$\mathcal{L}_{H}\approx\hat{\mathcal{L}}_{H}=v^{T}Av=|Bv|^{2}.$ (10)

Back-substitution yields an explicit approximation formula,

$\mathcal{L}_{H}\approx\frac{1}{(3N)^{2}}|\tilde{H}v-Hv|^{2},$ (11)

where $v$ is a random vector of component $3N$ that satisfies Eq. (3). Importantly, the right-hand side involves only Hessian-vector products.*[19, 34]* In particular, individual elements of the ML-predicted Hessian $\tilde{H}$ never need to be calculated explicitly. Instead, one can employ, e.g., the torch.hvp function to efficiently evaluate $\tilde{H}v$ as a vector.

#### 2.1.1 Randomizing over mini-batches

Generally, a stochastic Hutchinson estimator such as Eq. 11 will be deployed as an average over a large number $N_{v}$ of random vectors $v$. Since each random vector is associated with an independent and unbiased estimate, the total stochastic error in estimating $\mathcal{L}_{H}$ decays like the inverse square root, $N_{v}^{-1/2}$.*[40, 41]*

In practice, training of MLIPs uses stochastic gradient descent (SGD) over a very large number of mini-batches. It is natural that each mini-batch could employ a new random vector $v$. Then, for purposes of SGD training, the overall quality of the Hutchinson approximator can benefit strongly from having many mini-batches.

#### 2.1.2 Random vector distributions

As discussed above, Hutchinson’s original proposal was to sample each vector component independently under a uniform distribution,

$v_{i}^{\text{Hutch}}=\{\pm 1\}.$ (12)

Recently, in the context of Hessian training, an alternative stochastic estimator was suggested: Work with a randomly selected column of $(\tilde{H}-H)$ at a time.*[27, 29, 32]* This method can be cast into our notation taking $v$ to have a single non-zero component (“hot”) at a random index $c=\{1,\ldots,3N\}$. This one-hot distribution can be expressed concisely as follows.

$v_{i}^{\text{1Hot}}=\sqrt{3N}\,\delta_{i,c}.$ (13)

To verify the correctness of the 1-hot stochastic estimator, we must check Eq. (3). Since $v_{i}^{\text{1Hot}}$ has a single nonzero element, it is guaranteed

$\left\langle v_{i}^{\text{1Hot}}v_{j}^{\text{1Hot}}\right\rangle=0\quad(i\neq j).$ (14)

Furthermore, the prefactor $\sqrt{3N}$ is precisely what is needed to satisfy

$\left\langle v_{i}^{\text{1Hot}}v_{i}^{\text{1Hot}}\right\rangle$ $=\frac{1}{3N}\sum_{c=1}^{3N}\left[v_{i}^{\text{1Hot}}\right]^{2}$
$=\frac{1}{3N}\sum_{c=1}^{3N}\left[\sqrt{3N}\,\delta_{i,c}\right]^{2}$
$=\sum_{c=1}^{3N}\delta_{i,c}$
$=1.$

These two equations demonstrate consistency with Eq. (3). Therefore, both the Hutchinson random vectors $v^{\text{Hutch}}$ and the one-hot random vectors $v^{\text{1Hot}}$ yield valid and unbiased stochastic estimators. A detailed analysis of the stochastic error of the estimators used in this work is provided in the Supplementary Information and suggests that, under the physically motivated locality assumption that Hessian errors decay with interatomic distance, Hutchinson probing yields a mean-squared error that scales as $O(N)$ (implying an RMSE $\sim\sqrt{N}$ and a relative error that decreases with system size), whereas one-hot (column) probing concentrates error on diagonal terms and exhibits less favorable scaling, with the gap between the two estimators expected to widen as the number of atoms $N$ increases. The stochastic estimator based on $v^{\text{Hutch}}$ will be defined as the PHL method from now on.

## 3 Methodology

We investigate how second-derivative information can be efficiently incorporated into MLIP training. Starting from a chemically diverse quantum-chemical dataset, we train neural-network potentials under multiple supervision strategies that progressively include higher-order derivatives of the potential energy surface. Specifically, we compare standard energy–force (E-F) training with approaches that incorporate curvature information either through full Hessians or through HVPs. By keeping the model architecture and the underlying data fixed while varying only the form of derivative supervision, we isolate the impact of curvature information on training dynamics, predictive accuracy, and computational cost.

### 3.1 Dataset Preparation

To evaluate the role of Hessian information in training MLIPs, we employed the same dataset introduced in our previous work,*[25, 33]* which spans both equilibrium and non-equilibrium regions of a potential energy surface. All reference calculations were performed with Gaussian16 at the $\omega$B97XD/6-31G(d) level of theory.*[42]* Analytical Hessians were obtained through frequency analyzes. The choice of $\omega$B97XD provides a reliable treatment of non-covalent interactions and barrier heights, while the 6-31G(d) basis set balances accuracy and computational efficiency and ensures compatibility with the ANI-1x models.*[3, 4, 43, 44, 6, 43]*

The dataset is divided into three complementary components:

1. Benchmark Test Set: A collection of 35,087 equilibrium geometries from 11,961 reactions, including optimized reactants, products, and transition states. These structures were excluded from training and serve as a reference for interpolation accuracy.
2. Intrinsic Reaction Coordinate (IRC) Dataset: A set of 34,248 geometries sampled from 600 IRC trajectories, providing continuous reaction pathways that probe the curvature of the potential energy surface near transition states.
3. Normal Mode Sampling (NMS) Dataset: A collection of 62,527 non-equilibrium geometries generated by perturbing intermediate IRC structures along vibrational normal modes for 574 reactions. This dataset is designed to rigorously test the robustness of the extrapolation under large structural distortions.

The molecular systems in these datasets are *small-molecule* geometries, with a typical atom count $N$ in the *tens* (e.g., median $N\approx 14$ across the datasets) rather than the hundreds or thousands characteristic of condensed-phase simulations. This point is important because our asymptotic

stochastic-error analysis is derived in the large-$N$ limit: as $N$ increases and Hessian errors remain localized, Hutchinson probing is predicted to become increasingly favorable relative to one-hot (one-column) probing. Thus, while the present benchmarks already show clear advantages for Hutchinson probing in data-limited settings, the theoretical scaling suggests that these advantages should grow for larger systems such as extended materials, large clusters, or supercells.

For our purposes, these datasets provide diverse coverage of the potential energy surface, with the benchmark test set and IRC dataset assessing interpolation performance in chemically relevant regions, and the NMS dataset challenging the models under far-from-equilibrium conditions. These datasets form the common foundation for all the training strategies described below, enabling direct comparison of how different forms of derivative supervision influence learning behavior and generalization.

### 3.2 Training Objective and Strategies

Using these datasets, our objective was to quantify the trade-off between predictive accuracy and computational efficiency as progressively higher-order derivative information is introduced into MLIP training. To this end, we evaluated four training schemes that progressively incorporate higher-order derivatives of the potential energy surface into the objective function. The schemes are designed to explore the tradeoffs between accuracy and computational cost; a detailed description of the optimization setup (dataset split, hyperparameters, batching, probe sampling, and other information) is provided in the Supplementary Information under Training Procedure. The training schemes are the following:

1. E–F: Training on total energies $E$ and atomic forces $F$. This baseline reflects standard practice in the development of MLIPs.
2. E–F–HVP (one-column): Training on energies, forces, and Hessian–vector products (HVP), where $Hv=\nabla^{2}E(\mathbf{R})\,v$ is evaluated with a single one-hot probe vector $v^{1\text{Hot}}$ aligned with a coordinate axis. This approach introduces curvature information at minimal additional cost and has been recently used for training MLIPs to Hessian information.*[29, 32]*
3. E–F–HVP (PHL): Training on energies, forces, and HVPs using Projected Hessian Learning, where $Hv=\nabla^{2}E(\mathbf{R})\,v$ is evaluated using Hutchinson-type Gaussian probing vectors $v^{\text{Hutch}}$ with mean zero and unit variance. This stochastic estimator provides an unbiased approximation to full Hessians.
4. E–F–H: Training on energies, forces, and full Hessians. This provides the most complete curvature information but is associated with a prohibitive increase in computational cost.

In summary, E-F-H represents the upper bound in accuracy but is computationally expensive, while E-F serves as the lowest-cost baseline. The two HVP-based methods offer intermediate strategies that capture much of the benefit of Hessian training while remaining computationally efficient.

In both HVP-based schemes, we considered two vector protocols: (i) randomized-vector training, in which new probe vectors are resampled at every minibatch, and (ii) fixed-vector training, in which a single probe vector per configuration is retained throughout training. The fixed-vector setting serves as a controlled surrogate for scenarios in which the ground-truth Hessian is not known in full, but itself sampled.

Finally, we examine both the running time of the machine-learning model training and the cost of generating the underlying quantum mechanical reference data to assess the efficiency and

benefits of the different training schemes. We compared the cost of computing energies, forces, full Hessians and HVPs using DFT. To allow fair comparison between training schemes, all models were optimized using a unified loss framework that differs only in how curvature information is incorporated.

### 3.3 Loss Function Formulation

The total training loss was defined as a weighted sum of errors in predicted energies, forces, and, when included, Hessian information. Two formulations were used depending on whether the model was trained with full Hessians or with Hessian–vector products:

$\mathcal{L}_{\text{full}}$ $=\mathcal{L}_{E}(E^{\text{pred}},E^{\text{ref}})+\lambda_{F}\,\mathcal{L}_{F}(F^{\text{pred}},F^{\text{ref}})+\lambda_{H}\,\mathcal{L}_{H}(H^{\text{pred}},H^{\text{ref}}),$ (15)
$\mathcal{L}_{\text{HVP}}$ $=\mathcal{L}_{E}(E^{\text{pred}},E^{\text{ref}})+\lambda_{F}\,\mathcal{L}_{F}(F^{\text{pred}},F^{\text{ref}})+\lambda_{H}\,\hat{\mathcal{L}}_{H}(H^{\text{pred}}v,H^{\text{ref}}v).$ (16)

Here, $E$ represents the total molecular energies, $F$ represents atomic forces, $H$ represents the Hessian matrices, and $v$ represents probe vectors. In the full-Hessian formulation, the model is trained against all second derivatives, whereas in the HVP formulation, only products of the Hessian with probe vectors are included. The weights ($\lambda_{F}=0.30$) and ($\lambda_{H}=0.09$) were tuned to balance the relative contributions of forces and Hessian information against energies. Typically, energies set the absolute scale, while forces and Hessian terms are given smaller weights to ensure they contribute comparably during optimization. All models were trained under identical optimization settings, with differences arising solely from the form of derivative supervision.

## 4 Results and Discussion

### 4.1 Validation Loss Convergence

The convergence behavior of the different training schemes was examined using validation RMSE curves for energies, forces, and Hessians under the fixed-vector approach (Figure 2). These curves provide insight into how curvature information influences optimization dynamics, training stability, and the relative behavior of one-column and Hutchinson-based Hessian-vector product estimators (PHL). The validation results for the randomized-vector experiments reached comparable final RMSE values for the characteristic system size of our training dataset (a median of 14 atoms per system) and are therefore omitted from this section for brevity; their performance in generalization is analyzed separately below.

For energy errors, all training schemes eventually converge to similar RMSE validation levels. However, clear differences appear during the early and intermediate stages of training. Methods incorporating curvature information exhibit reduced fluctuations in validation loss after approximately 700 epochs, resulting in smoother and more stable convergence trajectories. Full Hessian training (E–F–H) shows the most stable behavior early in training and converges more rapidly, while both HVP-based approaches achieve greater stability than the E–F baseline during the intermediate and late stages of training.

This smoother convergence suggests that the incorporation of second-derivative information regularizes the optimization process, reducing the sensitivity to stochastic noise and sharp variations in the loss landscape. Such stabilization effects have been widely observed in both machine learning and atomistic modeling,*[45, 46]* where smoother optimization trajectories are often associated with improved generalization and robustness, even when the final training errors are similar.

For force errors, E–F training achieves the lowest absolute validation RMSE, consistent with its objective function placing the greatest emphasis on force accuracy. Importantly, introducing

![images/image2.jpg](images/image2.jpg)
Figure 2: Validation loss curves for energy, force, and Hessian predictions under the fixed-vector approach. Each panel shows the RMSE of the validation set as a function of training epoch for energy and force training (E-F, red), energy, force, and Hessian-vector product training using the one-column method (E-F-HVP One-Column, orange), energy, force, and HVP training using the Hutchinson estimator (E-F-HVP PHL, purple), and full energy-force-Hessian training (E-F-H, blue).

second-derivative information, either via full Hessians or HVP estimators, does not degrade force prediction accuracy. Both E-F-HVP models closely track the convergence behavior of E-F-H, indicating that curvature information can be incorporated without compromising force fidelity. In particular, although E-F achieves slightly lower force RMSE in the validation set, this does not necessarily translate to improved generalization, particularly for configurations outside of the training distribution, as discussed in the following section.

For Hessian errors, the benefit of incorporating second-derivative information is most pronounced under the data regimes considered here (i.e., equilibrium and saddle-point structures). Models trained only on energies and forces exhibit large and poorly converged Hessian validation errors, indicating that curvature information is not reliably recovered from first-order data alone. In contrast, compared to the E-F training, incorporating curvature supervision through HVPs reduced Hessian RMSE by  $71 - 86\%$  for the one-column estimator and  $74 - 88\%$  for the Hutchinson estimator across the Test, IRC, and NMS datasets. The full Hessian training unsurprisingly achieves the lowest overall error with  $77 - 90\%$  RMSE reduction compared to the E-F training. It is important to note that the Hutchinson estimator achieved  $11 - 15\%$  lower Hessian RMSE than the one-column

estimator in the fixed-vector training scheme, even for our ”small” system sizes.

Crucially, HVP-based methods recover much of the curvature accuracy and training stability of full Hessian supervision at a dramatically reduced computational cost, making them effective in regimes where explicit Hessian evaluation is cost prohibitive. Overall, HVP-based training captures most of the optimization and generalization benefits of full Hessian inclusion, with the Hutchinson-based PHL estimator offering improved robustness and accuracy when only a single fixed HVP per molecular system is available.

### 4.2 Predictive Accuracy Across Datasets

To evaluate generalization performance, we compared the final RMSE values of all models on the Test Set, the IRC dataset, and the NMS dataset. Figure 3 summarizes the results under the randomized-vector and fixed-vector approaches with separate panels for the energy, force, and Hessian predictions.

#### 4.2.1 Randomized HVP Vectors per Minibatch

When the HVP probe vector is randomized in each minibatch step (Figure 3(a)), the performance of the one-column and PHL estimators is nearly indistinguishable across all datasets and all training targets for the system sizes in the training data.

- Energy RMSE: Under randomized-vector training, both HVP estimators substantially improve upon the E–F baseline and approach full Hessian accuracy. On the NMS dataset, the energy RMSE is reduced by approximately 29% compared to E–F for both one-column and PHL estimators, with only a difference of 1 to 2% between them. Similar trends are observed on the Test and IRC datasets, indicating that stochastic curvature sampling provides sufficient coverage for accurate energy prediction.
- Force RMSE: Although E–F remains marginally optimal for in-domain force prediction, HVP-based methods closely match full Hessian performance and substantially outperform E–F on extrapolative NMS geometries. Specifically, the force RMSE on NMS is reduced by approximately 48 to 49% relative to E–F for both HVP estimators, with a separation of less than 2% between the one-column and PHL approaches.
- Hessian RMSE: HVP training dramatically improves Hessian prediction accuracy compared to E–F, reducing the NMS Hessian RMSE by approximately 77% for both estimators. The one-column and PHL methods exhibit nearly identical performance in this randomized setting, differing only by $\sim$1%, confirming that minibatch-level randomization provides sufficiently rich curvature sampling to approximate full Hessian supervision.

Randomized HVP vectors therefore enable broad curvature coverage during training, yielding performance nearly indistinguishable from full Hessian training for energies, forces, and Hessians, while avoiding the substantial computational cost of explicit Hessian supervision. Note also that our theoretical analysis suggests that the Hutchinson-based PHL method would become preferred when training on larger system sizes $N$, for which the $N\times N$ Hessian matrix carries much more information.

![images/image3.jpg](images/image3.jpg)
(a) Randomized vectors
Figure 3: Validation RMSE comparison across the Test Set, IRC dataset, and NMS dataset for models trained with different levels of information using (a) randomized probe vectors (resampled each minibatch) and (b) fixed probe vectors (one probe per system). Results are shown for energy and force training (E-F); energy, force, and Hessian-vector products (HVP) using the one-column method (E-F-HVP One-Column); energy, force, and HVP using the Hutchinson estimator (E-F-HVP PHL); and energy, force, and full Hessian training (E-F-H). From top to bottom, panels report RMSE for energies, forces, and Hessians. Bars represent mean RMSE values, with error bars indicating variability across ensembles of five independently trained models. Incorporating Hessian information improves accuracy relative to E-F across all datasets; under randomized probing, one-hot and PHL achieve statistically indistinguishable performance for the small molecular systems studied here, whereas under fixed probes PHL provides systematically lower errors, most notably for extrapolative NMS geometries.

![images/image4.jpg](images/image4.jpg)
(b) Fixed vectors

4.2.2 Fixed HVP Vectors per System

The fixed-vector setting (Figure 3(b)) provides a more realistic scenario in which only one HVP per system is available from quantum chemistry. Under these conditions, the differences between the one-column and PHL estimators become more pronounced.

- Energy RMSE: All Hessian-informed methods improve energy RMSE relative to E-F training across all datasets, with gains that are largest on extrapolative geometries. On the Test set, energy RMSE decreases by 6.1% with the one-column estimator and 10.3% with PHL, compared to E-F (11.8% for full Hessian training). On the IRC dataset, the reductions are smaller but consistent: 3.8% for one-column and 5.6% for PHL. On the NMS dataset, the improvements are substantial: one-column reduces the energy RMSE by 23.8% and PHL by 28.5% relative to E-F, with PHL achieving a 6.2% reduction in RMSE compared to one-column (and essentially matching complete Hessian training, 28.2%).
- Force RMSE: The benefit of curvature supervision is most apparent for intermediate or NMS configurations. Although E-F training attains the lowest RMSE on the Test set, both HVP methods match force accuracy on the IRC geometries and substantially improve force accuracy on NMS geometries. Compared to E-F, the one-column and PHL estimators reduce the force RMSE of the NMS dataset by 42.3% and 45.6%, respectively (47.8% for full Hessian training). The PHL estimator consistently achieves a lower RMSE than the one-column estimator across datasets, with a 2.1% reduction in RMSE on the Test set, 2.7% on the IRC set, and 5.6% on the NMS set.
- Hessian RMSE: The strongest separation between estimators appears for Hessian prediction. Relative to E-F training, Hessian RMSE is reduced by 86.2% with one-column estimators and 88.3% with PHL estimators on the Test set; 85.3% and 87.0% on IRC; and 71.3% and 74.5% on NMS. The PHL estimator consistently outperforms the one-column estimator across all datasets, with Hessian RMSE lower by 15.1% on the Test set, 12.1% on IRC, and 11.2% on NMS.

When only a single HVP per molecular configuration is available, the PHL estimator yields errors that are systematically lower than those of the one-column approach across all predicted properties. On the extrapolative NMS dataset, PHL reduces the energy RMSE by 6.2%, the force RMSE by 5.6%, and the Hessian RMSE by 11.2% relative to the one-column estimator. These improvements indicate that random Hutchinson vectors provide a more uniform sampling of curvature directions under fixed-vector constraints. While the one-column estimator remains a computationally inexpensive alternative, its reliance on a single coordinate direction limits performance in data-sparse regimes and large system sizes.

### 4.3 Statistical Significance of RMSE Differences

To quantify whether the one-column and PHL estimators produce statistically distinguishable predictive accuracy, we conducted paired two-sample $t$-tests using the RMSE values from the five independently trained models for each method. The tests were performed separately for the randomized-vector and fixed-vector regimes, allowing us to isolate the effect of the vector-sampling strategy on estimator behavior. The results are summarized in Tables 1 and 2, where the mean RMSE difference is defined as (one-column RMSE – PHL RMSE). Thus, negative values indicate lower RMSE for the one-column estimator, while positive values indicate lower RMSE for the PHL estimator.

# 4.3.1 Randomized HVP Vectors per Minibatch

Table 1 reports the results of the paired  $t$ -test for experiments in which the HVP vector was randomly resampled in each minibatch. Under this configuration, no statistically significant differences were observed between the two estimators across any of the datasets or properties used (all  $p &gt; 0.05$ ).

Table 1: Two-sample  $t$ -test results comparing RMSE values from the one-column and PHL estimators under the randomized-vector training approach. The mean difference is defined as (one-column - PHL). Negative values indicate lower RMSE for the one-column method. Asterisks denote statistically significant differences ( $p &lt; 0.05$ ).

|  Property | Dataset | Mean Difference (kcal/mol) | p-value | Significance  |
| --- | --- | --- | --- | --- |
|  Energy | Test | -0.0039 | 0.950 | -  |
|  Energy | IRC | -0.0775 | 0.348 | -  |
|  Energy | NMS | -0.2320 | 0.342 | -  |
|  Force | Test | 0.0134 | 0.832 | -  |
|  Force | IRC | -0.0198 | 0.732 | -  |
|  Force | NMS | -0.1444 | 0.195 | -  |
|  Hessian | Test | -0.1051 | 0.405 | -  |
|  Hessian | IRC | -0.1468 | 0.166 | -  |
|  Hessian | NMS | -0.3608 | 0.189 | -  |

Across energies, forces, and Hessians, the mean RMSE differences are small and fluctuate around zero, consistent with the interpretation that randomized stochastic sampling effectively averages out directional bias for our characteristic system sizes. Randomized vectors ensure broad coverage of curvature directions throughout training, allowing both estimators to recover similarly accurate approximations to the Hessian.

These results indicate that when HVP vectors vary throughout training, the one-column and PHL estimators perform equivalently, and neither method shows a statistically significant advantage for these system sizes.

# 4.3.2 Fixed HVP Vectors per System

The corresponding results of the paired  $t$ -test for fixed-vector experiments are shown in Table 2. In the fixed-vector regime, a single HVP vector is assigned to each molecule and reused across all epochs. This setting simulates scenarios in which only one HVP is available per system from quantum chemical calculations. Under this more restrictive condition, statistically significant differences emerge between the two estimators for several datasets and training targets.

- Energy RMSE: For the Test and IRC datasets, the mean differences are not statistically significant ( $p &gt; 0.05$ ), indicating comparable energy predictions. However, for the NMS dataset, the PHL estimator yields significantly lower energy RMSE ( $p = 0.006$ ), demonstrating better extrapolative capability.
- Force RMSE: The PHL estimator performs significantly better on the IRC dataset ( $p = 0.013$ ) and the NMS dataset ( $p = 0.006$ ). This suggests that its broader curvature sampling improves first-derivative accuracy in intermediate and far-from-equilibrium geometries, where directional bias from a single one-column vector becomes limiting.

- Hessian RMSE: Across all datasets, the PHL estimator yields a significantly lower Hessian RMSE ( $p &lt; 0.01$  for all comparisons). These results confirm that the Hutchinson vectors provide substantially better curvature information when only one vector per molecule is available.

Table 2: Two-sample  $t$ -test results comparing RMSE values from the one-column and PHL estimators under the fixed-vector training approach. The mean difference is defined as (one-column - PHL). Negative values indicate lower RMSE for the one-column method. Asterisks denote statistically significant differences ( $p &lt; 0.05$ ).

|  Property | Dataset | Mean Difference (kcal/mol) | p-value | Significance  |
| --- | --- | --- | --- | --- |
|  Energy | Test | 0.2370 | 0.063 | -  |
|  Energy | IRC | 0.1303 | 0.399 | -  |
|  Energy | NMS | 0.8989 | 0.006 | *  |
|  Force | Test | 0.1268 | 0.097 | -  |
|  Force | IRC | 0.2076 | 0.013 | *  |
|  Force | NMS | 0.8086 | 0.006 | *  |
|  Hessian | Test | 2.9105 | 0.0000029 | *  |
|  Hessian | IRC | 2.4722 | 0.00022 | *  |
|  Hessian | NMS | 4.1780 | 0.005 | *  |

This analysis indicates that when curvature information is limited to a single HVP per system, the PHL estimator consistently outperforms the one-column method, especially for the NMS dataset and for Hessian predictions, due to its isotropic distribution and reduced directional bias.

# 4.4 Computational Efficiency

A key motivation for incorporating Hessian-vector products into MLIP training is the ability to reap the benefits of second-derivative information without incurring the cost of full Hessian evaluations. To quantify this advantage, we assessed computational efficiency at two levels: (1) MLIP training runtime per epoch in Figure 4 and (2) quantum-chemistry cost scaling for computing energies, forces, Hessians, and HVPs in Figure 5.

# 4.4.1 MLIP Training Runtime

Figure 4 reports the execution time per epoch for all training schemes. Full Hessian training (E-F-H) is by far the most computationally demanding, requiring over an order of magnitude more time per epoch than the other methods. This cost arises from the need to backpropagate through the complete Hessian matrix, which involves repeated second-derivative evaluations.

In contrast, both HVP-based approaches (E-F-HVP One-Column and E-F-HVP PHL) achieve epoch times of approximately 13 s, representing a 24-fold speedup relative to full Hessian training. Importantly, both estimators preserve most of the predictive gains afforded by full Hessians, providing a near-optimal tradeoff between speed and accuracy.

Energy-force training (E-F) remains the fastest configuration at roughly 4 s per epoch, but lacks any curvature information. The HVP methods therefore occupy a favorable middle ground: only three times slower than E-F training but dramatically more informative. This comparison confirms that incorporating curvature information via HVPs is highly practical in large-scale MLIP training pipelines, particularly when full Hessians would be computationally prohibitive.

![images/image5.jpg](images/image5.jpg)
Figure 4: Execution time per training epoch for different methods: energy and force training (E-F); energy, force, and Hessian-vector products (HVP) using the one-column method (E-F-HVP One-Column); energy, force, and HVP using the PHL method (E-F-HVP PHL); and energy, force, and full Hessian training (E-F-H). Bars show mean times with error bars indicating variability across epochs. Both one-column and PHL estimators achieve more than a 24-fold speedup compared to full Hessian training, while E-F training is the fastest but lacks Hessian information.

![images/image6.jpg](images/image6.jpg)

# 4.4.2 Quantum-Chemistry Cost Scaling

To contextualize the MLIP results with first-principles cost considerations, Figure 5 presents the CPU scaling for evaluating energies, forces, full Hessians, and approximate HVP costs using Gaussian16. Energy and force calculations show similar scaling behavior as system size increases, reflecting their reliance on first derivatives of the electronic energy. However, full Hessian evaluations grow much more steeply. This superlinear scaling rapidly dominates computational budgets for systems beyond a few dozen atoms, limiting the practical generation of high-quality second-derivative datasets.

By contrast, HVP evaluations scale much closer to forces. A single Hessian-vector product can be computed using two force-like operations via forward-over-backward or backward-over-backward automatic differentiation. Alternatively, when the probing direction  $v$  is fixed, the same quantity can be obtained from finite differences of forces by evaluating  $\mathbf{F}(\mathbf{R} \pm \epsilon v)$  and forming  $Hv \approx -\left[\mathbf{F}(\mathbf{R} + \epsilon v) - \mathbf{F}(\mathbf{R} - \epsilon v)\right] / (2\epsilon)$ , which again requires only two force evaluations. As a result, a single HVP is expected to cost on the order of two force evaluations, making it orders of magnitude cheaper than computing a full Hessian matrix. Because HVPs capture directional curvature information at such a low cost, they are an attractive surrogate for full Hessians both at the quantum-chemistry level (for data generation) and at the MLIP training level (for incorporating curvature into the loss function).

# 5 Conclusions

We introduced Projected Hessian Learning (PHL), a scalable curvature-supervision framework for MLIPs that replaces explicit Hessian construction with fast Hessian-vector product (HVP) probes. Using a chemically diverse dataset that spans reactants, products, transition states, intrinsic reaction coordinates (IRC), and normal-mode-sampled (NMS) geometries, we compared four training

![images/image7.jpg](images/image7.jpg)
(a) Semi-log plot

![images/image8.jpg](images/image8.jpg)
(b) Log-log plot
Figure 5: Scaling of CPU time with system size for density functional theory (DFT) calculations using Gaussian16 on a semi-log plot (a) and a log-log plot (b). Shown are the computational costs for evaluating total energies (green squares), forces (red triangles), and full Hessians (blue diamonds), and the estimated cost for evaluating Hessian-vector products (HVP, purple line). While force and energy costs scale comparably, the cost of full Hessian evaluations grows much more steeply with the number of atoms. In contrast, HVP evaluations provide a significantly cheaper alternative, with scaling closer to that of two force calculations, highlighting their efficiency for incorporating second-derivative information.

strategies of increasing physical fidelity: energy-force (E-F) training, HVP-based (E-F-HVP) training using one-column or PHL estimators, and full Hessian (E-F-H) training.

Incorporating curvature information stabilizes training dynamics and improves predictive accuracy. Across Test, IRC, and NMS datasets, the HVP-trained models substantially outperform the E-F baselines, approach full-Hessian training accuracy for the energies and forces, and reduce Hessian errors by approximately  $77\%$  on extrapolative NMS geometries. Under randomized-vector training, both HVP-based estimators achieve nearly identical performance for our system sizes, reducing the NMS energy RMSE by  $\sim 29\%$ , the force RMSE by  $\sim 48\%$  and the Hessian RMSE by  $\sim 77\%$  relative to E-F, indicating that stochastic curvature sampling effectively approximates full Hessian supervision. In contrast, under fixed-vector conditions representative of data-limited regimes, Hutchinson-based PHL consistently outperforms one-hot estimators, yielding additional reductions of  $6.2\%$  in energy RMSE,  $5.6\%$  in force RMSE, and  $11.2\%$  in Hessian RMSE on NMS geometries, as confirmed by paired  $t$ -tests.

Computationally, full Hessian training is dramatically more expensive, requiring on average 326.5 s per epoch, compared to 13.6 s and 13.3 s for the one-column probing and our Hutchinson-based PHL, respectively, corresponding to  $\sim 24\times$  speedups relative to E-F-H while incurring only a modest overhead compared to standard E-F training. This improvement arises because full Hessian training requires evaluating and backpropagating through all second-derivative components  $(3N)^{2}$ , while HVP-based methods access curvature only through Hessian-vector products, whose cost scales similarly to force evaluations. As a result, HVP-based approaches recover most of the benefits of Hessian supervision at less than  $8\%$  of the computational cost of full Hessian training. Consistent with this observation, at the quantum-chemistry level, HVP evaluations scale comparably to forces

and far more favorably than full Hessians, making them a practical surrogate for incorporating second-derivative information in MLIP training.

These results establish PHL as an efficient and scalable alternative to full Hessian supervision, delivering substantial accuracy gains over energy-force models at a greatly reduced cost. The PHL perspective also naturally connects to recent phonon fine-tuning (PFT) work, which uses stochastic curvature supervision to improve force-constant (phonon) accuracy in periodic materials; PHL provides a general projected-curvature framework that can be applied to materials settings where curvature governs vibrational and elastic response, and where large supercells make explicit Hessians impractical.

Future work will extend PHL to larger and more complex systems, develop adaptive probing strategies, and integrate projected-curvature supervision with active learning and uncertainty quantification. Our asymptotic analysis indicates that Hutchinson probing should become increasingly advantageous over one-hot probing as system size $N$ grows, suggesting particularly strong benefits in condensed-phase and materials settings beyond the tens-of-atoms regime explored here (typical $N=14$). In addition, when the probing vector $v$ is fixed, reference HVPs, i.e., $Hv$, can be computed in specialized quantum chemistry codes via finite differences of gradients along $v$ (forces in geometries displaced by $\pm\epsilon v$), enabling curvature-informed training on much larger systems and facilitating access to bulk materials properties that require large supercells, such as surfaces and defects.

## Data and code availability

The datasets used in this study are publicly available through the OpenREACT Figshare repository (OpenREACT-CHON-EFH; energies, forces, and Hessians for the RTP/IRC/NMS datasets used in this work): https://doi.org/10.6084/m9.figshare.29189858.

The code used to train and evaluate the ANI models in this paper, including notebooks and instructions for accessing the datasets, is available at: https://github.com/Austinrg14/PHL.

In addition, we implemented Hessian and Hessian–vector product (HVP) training functionality in the HIPPYNN framework to enable curvature-supervised training of HIP-NN models (including full-Hessian and projected-curvature/PHL training workflows).

HIPPYNN is available through the Los Alamos National Laboratory (LANL) GitHub repository: https://github.com/lanl/hippynn. An example script demonstrating our implementation is provided at hippynn/examples/hessian_training.py.

## Acknowledgements

This work is supported by the U.S. Department of Energy, Office of Basic Energy Sciences (FWP: LANLE8AN and FWP: LANLE3F2) and by the U.S. Department of Energy through Los Alamos National Laboratory. Los Alamos National Laboratory is operated by Triad National Security, LLC, for the National Nuclear Security Administration of the U.S. Department of Energy Contract No. 892333218NCA000001. This research used resources provided by the LANL CAI-1 Darwin cluster at LANL. This work was supported in part through computational resources and services provided by the Institute for Cyber-Enabled Research at Michigan State University (MSU). This work used the MSU Data Machine, which is supported through the NSF Campus CyberInfrastructure program through grant #2200792.

##

References

- [1] Oliver T Unke, Stefan Chmiela, Huziel E Sauceda, Michael Gastegger, Igor Poltavsky, Kristof T. Schütt, Alexandre Tkatchenko, and Klaus-Robert Müller. Machine learning force fields. Chemical Reviews, 121(16):10142–10186, 2021.
- [2] Jörg Behler and Michele Parrinello. Generalized neural-network representation of high-dimensional potential-energy surfaces. Physical Review Letters, 98:146401, Apr 2007. doi: 10.1103/PhysRevLett.98.146401.
- [3] J. S. Smith, O. Isayev, and A. E. Roitberg. ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost. Chemical Science, 8(4):3192–3203, 2017. doi: 10.1039/C6SC05720A.
- [4] J. S. Smith, O. Isayev, and A. E. Roitberg. ANI-1, A data set of 20 million calculated off-equilibrium conformations for organic molecules. Scientific Data, 4(1):1–8, 2017. doi: 10.1038/sdata.2017.193.
- [5] Justin S. Smith, Ben Nebgen, Nicholas Lubbers, Olexandr Isayev, and Adrian E. Roitberg. Less is more: Sampling chemical space with active learning. The Journal of Chemical Physics, 148(24):241733, 2018. doi: 10.1063/1.5023802.
- [6] Justin S. Smith, Roman Zubatyuk, Benjamin Nebgen, Nicholas Lubbers, Kipton Barros, Adrian E. Roitberg, Olexandr Isayev, and Sergei Tretiak. The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional theory properties for molecules. Scientific Data, 7(1): 1–10, 2020. doi: 10.1038/s41597-020-0473-z.
- [7] Nicholas Lubbers, Justin S Smith, and Kipton Barros. Hierarchical modeling of molecular energies using a deep neural network. The Journal of Chemical Physics, 148(24):241715, 2018. doi: 10.1063/1.5023802.
- [8] Linfeng Zhang, Jiequn Han, Han Wang, Roberto Car, and Weinan E. Deep potential molecular dynamics: a scalable model with the accuracy of quantum mechanics. Physical Review Letters, 120(14):143001, 2018.
- [9] Albert P Bartók, Mike C Payne, Risi Kondor, and Gábor Csányi. Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons. Physical Review Letters, 104(13):136403, 2010.
- [10] Aidan P Thompson, Laura P Swiler, Christian R Trott, Stephen M Foiles, and Garritt J Tucker. Spectral neighbor analysis method for automated generation of quantum-accurate interatomic potentials. Journal of Computational Physics, 285:316–330, 2015.
- [11] Alexander V Shapeev. Moment tensor potentials: A class of systematically improvable interatomic potentials. Multiscale Modeling & Simulation, 14(3):1153–1173, 2016.
- [12] Kristof T Schütt, Huziel E Sauceda, P-J Kindermans, Alexandre Tkatchenko, and K-R Müller. Schnet–a deep learning architecture for molecules and materials. The Journal of Chemical Physics, 148(24):241722, 2018. doi: 10.1063/1.5019779.
- [13] Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E Smidt, and Boris Kozinsky. E (3)-equivariant graph neural

networks for data-efficient and accurate interatomic potentials. Nature Communications, 13(1): 2453, 2022.
- [14] Kristof Schütt, Oliver Unke, and Michael Gastegger. Equivariant message passing for the prediction of tensorial properties and molecular spectra. In International Conference on Machine Learning, pages 9377–9388, 2021.
- [15] Ilyes Batatia, David P Kovacs, Gregor Simm, Christoph Ortner, and Gábor Csányi. Mace: Higher order equivariant message passing neural networks for fast and accurate force fields. Advances in Neural Information Processing Systems, 35:11423–11436, 2022.
- [16] Stefan Chmiela, Huziel E Sauceda, Igor Poltavsky, Klaus-Robert Müller, and Alexandre Tkatchenko. sgdml: Constructing accurate and data efficient molecular force fields using machine learning. Computer Physics Communications, 240:38–45, 2019.
- [17] Oliver T Unke and Markus Meuwly. Physnet: A neural network for predicting energies, forces, dipole moments, and partial charges. Journal of Chemical Theory and Computation, 15(6): 3678–3693, 2019.
- [18] Antoine Loew, Dewen Sun, Hai-Chen Wang, Silvana Botti, and Miguel AL Marques. Universal machine learning interatomic potentials are ready for phonons. npj Computational Materials, 11(1):178, 2025.
- [19] Andreas Griewank and Andrea Walther. Evaluating derivatives: principles and techniques of algorithmic differentiation. SIAM, 2008.
- [20] Atilim Gunes Baydin, Barak A Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. Automatic differentiation in machine learning: a survey. Journal of Machine Learning Research, 18(153):1–43, 2018.
- [21] Peter Pulay. Analytical derivatives, forces, force constants, molecular geometries, and related response properties in electronic structure theory. Wiley Interdisciplinary Reviews: Computational Molecular Science, 4(3):169–181, 2014.
- [22] J.A Pople, R Krishnan, HB Schlegel, and J S. Binkley. Derivative studies in hartree-fock and møller-plesset theories. International Journal of Quantum Chemistry, 16(S13):225–241, 1979.
- [23] Carlos Gonzalez and H Bernhard Schlegel. An improved algorithm for reaction path following. The Journal of Chemical Physics, 90(4):2154–2161, 1989.
- [24] Kenichi Fukui. The path of chemical reactions-the irc approach. Accounts of Chemical Research, 14(12):363–368, 1981.
- [25] Austin Rodriguez, Justin S Smith, and Jose L Mendoza-Cortes. Does hessian data improve the performance of machine learning potentials? Journal of Chemical Theory and Computation, 21 (14):6698–6710, 2025. doi: 10.1021/acs.jctc.5c00402.
- [26] Yuan, Eric C.-Y. and Kumar, Anup and Guan, Xingyi and Hermes, Eric D. and Rosen, Andrew S and Zádor, Judit and Head-Gordon, Teresa and Blau, Samuel M. Analytical ab initio hessian from a deep learning potential for transition state optimization. Nature Communications, 15 (1):8865, 2024. doi: 10.1038/s41467-024-52481-5.

[27] Ishan Amin, Sanjeev Raja, and Aditi Krishnapriyan. Towards fast, specialized machine learning force fields: Distilling foundation models via energy hessians. In The Thirteenth International Conference on Learning Representations, 2025. doi: 10.48550/arXiv.2501.09009. ICLR 2025. arXiv:2501.09009.
- [28] Marius Herbold and Jörg Behler. A hessian-based assessment of atomic forces for training machine learning interatomic potentials. The Journal of Chemical Physics, 156(11):114106, 2022. doi: 10.1063/5.0082952.
- [29] Teddy Koker, Abhijeet Gangan, Mit Kotak, Jaime Marian, and Tess Smidt. Pft: Phonon fine-tuning for machine learned interatomic potentials, 2026. arXiv:2601.07742.
- [30] Andreas Burger, Luca Thiede, Nikolaj Rønne, Varinia Bernales, Nandita Vijaykumar, Tejs Vegge, Arghya Bhowmik, and Alan Aspuru-Guzik. Shoot from the hip: Hessian interatomic potentials without derivatives, 2025. arXiv:2509.21624.
- [31] Nicholas J Williams, Lara Kabalan, Ljiljana Stojanovic, Viktor Zólyomi, and Edward O Pyzer-Knapp. Hessian qm9: A quantum chemistry database of molecular hessians in implicit solvents. Scientific Data, 12(1):9, 2025. doi: 10.1038/s41597-024-04361-2.
- [32] Taoyong Cui, Yonghong Han, Haojun Jia, Chenru Duan, and Qiyuan Zhao. A large scale molecular hessian database for optimizing reactive machine learning interatomic potentials. Scientific Data, 13(1):37, 2026. doi: 10.1038/s41597-025-06350-5.
- [33] Austin Rodriguez, Justin S. Smith, and Jose L. Mendoza-Cortes. OpenREACT-CHON-EFH — Open REaction Dataset of Atomic ConfiguraTions comprising C, H, O, N with Energies, Forces, and Hessians, 5 2025. Figshare dataset, doi:10.6084/m9.figshare.29189858.
- [34] Barak A Pearlmutter. Fast exact multiplication by the hessian. Neural computation, 6(1): 147–160, 1994.
- [35] James Martens et al. Deep learning via hessian-free optimization. In Proceedings of the 27th International Conference on Machine Learning, volume 27, pages 735–742, 2010.
- [36] Jorge Nocedal and Stephen J Wright. Numerical optimization. Springer, 2006.
- [37] N. Halko, P. G. Martinsson, and J. A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288, 2011. doi: 10.1137/090771806.
- [38] David P. Woodruff. Sketching as a tool for numerical linear algebra, 2014. arXiv:1411.4357.
- [39] M.F. Hutchinson. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics - Simulation and Computation, 19(2):433–450, 1990. doi: 10.1080/03610919008812866.
- [40] Haim Avron and Sivan Toledo. Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix. Journal of the ACM, 58(8):8:1–8:17, 2011. doi: 10.1145/1944345.1944349.
- [41] Arvind K Saibaba, Alen Alexanderian, and Ilse CF Ipsen. Randomized matrix-free trace and log-determinant estimators. Numerische Mathematik, 137(2):353–395, 2017.

[42] M. J. Frisch, G. W. Trucks, H. B. Schlegel, G. E. Scuseria, M. A. Robb, J. R. Cheeseman, G. Scalmani, V. Barone, G. A. Petersson, H. Nakatsuji, X. Li, M. Caricato, A. V. Marenich, J. Bloino, B. G. Janesko, R. Gomperts, B. Mennucci, H. P. Hratchian, J. V. Ortiz, A. F. Izmaylov, J. L. Sonnenberg, D. Williams-Young, F. Ding, F. Lipparini, F. Egidi, J. Goings, B. Peng, A. Petrone, T. Henderson, D. Ranasinghe, V. G. Zakrzewski, J. Gao, N. Rega, G. Zheng, W. Liang, M. Hada, M. Ehara, K. Toyota, R. Fukuda, J. Hasegawa, M. Ishida, T. Nakajima, Y. Honda, O. Kitao, H. Nakai, T. Vreven, K. Throssell, J. A. Montgomery, Jr., J. E. Peralta, F. Ogliaro, M. J. Bearpark, J. J. Heyd, E. N. Brothers, K. N. Kudin, V. N. Staroverov, T. A. Keith, R. Kobayashi, J. Normand, K. Raghavachari, A. P. Rendell, J. C. Burant, S. S. Iyengar, J. Tomasi, M. Cossi, J. M. Millam, M. Klene, C. Adamo, R. Cammi, J. W. Ochterski, R. L. Martin, K. Morokuma, O. Farkas, J. B. Foresman, and D. J. Fox. Gaussian˜16 Revision C.01, 2016. Gaussian Inc. Wallingford CT.
- [43] Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S Smith, and Adrian E Roitberg. Torchani: A free and open source pytorch-based deep learning implementation of the ani neural network potentials. Journal of Chemical Information and Modeling, 60(7):3408–3415, 2020. doi: 10.1021/acs.jcim.0c00451.
- [44] Christian Devereux, Justin S. Smith, Kate K. Huddleston, Kipton Barros, Roman Zubatyuk, Olexandr Isayev, and Adrian E. Roitberg. Extending the applicability of the ani deep learning molecular potential to sulfur and halogens. Journal of Chemical Theory and Computation, 16 (7):4192–4202, 2020. doi: 10.1021/acs.jctc.0c00121.
- [45] Moritz Hardt, Benjamin Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In Proceedings of the 33rd International Conference on Machine Learning, volume 48, pages 1225–1234, 2016.
- [46] Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural computation, 9(1):1–42, 1997.
- [47] W. Kohn. Density functional and density matrix method scaling linearly with the number of atoms. Physical Review Letters, 76:3168–3171, Apr 1996. doi: 10.1103/PhysRevLett.76.3168.

# A Projected Hessian Learning Training Algorithm

|  Algorithm 1: Projected Hessian Learning (PHL) for second-order derivative training of MLIPs using Hessian-Vector Products (HVPs)  |   |
| --- | --- |
|  Input: Dataset D = {(R, E, F, C)}, where R ∈ R3N, E is reference energy, F = -∇R E is reference force. Curvature data C may contain one or more HVP pairs (v, Hv).  |   |
|  Input: MLIP Eθ(R), loss weights λE, λF, λH, probes per structure K, step size η.  |   |
|  Output: Trained parameters θ.  |   |
|  Initialize parameters θ  |   |
|  while not converged do  |   |
|  Sample minibatch B ⊂ D  |   |
|  L←0  |   |
|  foreach (R, E, F, C) ∈ B do  |   |
|  E← Eθ(R)  |   |
|  F← -∇R E  |   |
|  LE← ||E - E||2  |   |
|  LF← 1/3N ||F - F||2  |   |
|  // PHL curvature loss: average over K Hessian--vector probes  |   |
|  LH←0  |   |
|  for k←1 to K do  |   |
|  if (vk, Hvk) ∈ C then  |   |
|  v← vk  |   |
|  y← Hvk  |   |
|  else  |   |
|  // Generate Gaussian probe and obtain reference y = Hv  |   |
|  Sample v ~ N(0, I)  |   |
|  Obtain reference y ← Hv using the available QC procedure  |   |
|  // Compute predicted HVP without forming H  |   |
|  ŷ← Hv via AD HVP (e.g.ŷ = ∇R(Ŷ)v)  |   |
|  LH← LH + 1/(3N)2 ||ŷ - y||2  |   |
|  LH← 1/K LH  |   |
|  L← L + λE L_E + λF L_F + λH L_H  |   |
|  θ← θ - η∇θ L  |   |

Supplementary Information

Projected Hessian Learning: Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials

### A Stochastic error analysis

Although $\langle v_{i}v_{j}\rangle=\delta_{ij}$ guarantees an unbiased approximation, the accuracy of the approximation can vary significantly according to the distribution of $v$. We follow Avron et al.,*[40]* and analyze the mean squared error of the stochastic estimator $\operatorname{tr}A\approx v^{T}Av$,

$\operatorname{MSE}[v]=\left\langle\left(\operatorname{tr}A-v^{T}Av\right)^{2}\right\rangle.$ (S1)

Since we are working with unbiased estimators,

$\operatorname{tr}A=\sum_{i}A_{ii}=\sum_{ij}\delta_{ij}A_{ij}=\sum_{ij}\left\langle v_{i}v_{j}\right\rangle A_{ij}=\left\langle v^{T}Av\right\rangle.$ (S2)

this becomes

$\operatorname{MSE}[v]$ $=(\operatorname{tr}A)^{2}-2\operatorname{tr}A\left\langle v^{T}Av\right\rangle+\left\langle\left(v^{T}Av\right)^{2}\right\rangle$
$=\left\langle\left(v^{T}Av\right)^{2}\right\rangle-(\operatorname{tr}A)^{2}.$ (S3)

It remains to evaluate $\left\langle\left(v^{T}Av\right)^{2}\right\rangle$:

$\left\langle\left(v^{T}Av\right)^{2}\right\rangle$ $=\left\langle\sum_{ij}v_{i}A_{ij}v_{j}\sum_{kl}v_{k}A_{kl}v_{l}\right\rangle$
$=\sum_{ijkl}A_{ij}A_{kl}\left\langle v_{i}v_{j}v_{k}v_{l}\right\rangle.$ (S4)

Observe that the random vector $v$ enters the error entirely through the four-point expectation $\left\langle v_{i}v_{j}v_{k}v_{l}\right\rangle$.

Let us first consider the Hutchinson estimator $v_{i}^{\text{Hutch}}=\{\pm 1\}$. There are only two possible values,

\[ \left\langle v_{i}^{\text{Hutch}}v_{j}^{\text{Hutch}}v_{k}^{\text{Hutch}}v_{l}^{\text{Hutch}}\right\rangle=\begin{cases}1&\text{indices in even groups}\\
0&\text{otherwise}\end{cases}. \]

The first branch includes four possible cases: (1) the indices are all equal, $i=j=k=l$, or the indices come in two equal pairs, (2) $i=j\neq k=l$, (3) $i=k\neq j=l$, or (4) $i=l\neq j=k$. Using the fact that $A$ is symmetric, cases (3) and (4) are equivalent and can be merged. This yields three terms;

$\left\langle\left(v^{\text{Hutch},T}Av^{\text{Hutch}}\right)^{2}\right\rangle=\sum_{i}A_{ii}^{2}+\sum_{i\neq k}A_{ii}A_{kk}+2\sum_{i\neq j}A_{ij}^{2}.$ (S5)

One can also expand this definition.

$(\operatorname{tr}A)^{2}=\left(\sum_{i}A_{ii}\right)^{2}=\sum_{i}A_{ii}^{2}+\sum_{i\neq k}A_{ii}A_{kk}.$ (S6)

Subtraction yields the mean squared error for the Hutchinson estimator,

$\operatorname{MSE}[v^{\operatorname{Hutch}}]=2\sum_{i\neq j}A_{ij}^{2}.$ (S7)

Now, let us turn to the 1-hot encoding distribution,

$v_{i}^{\operatorname{1Hot}}=\sqrt{3N}\,\delta_{i,c}$ (S8)

In this case,

\[ v_{i}^{\operatorname{1Hot}}v_{j}^{\operatorname{1Hot}}v_{k}^{\operatorname{1Hot}}v_{l}^{\operatorname{1Hot}}=\begin{cases}(3N)^{2}&\text{all indices are the hot index}\\
0&\text{otherwise}\end{cases}. \] (S9)

Recall that, in the context of Hessian training, $N$ is the number of atoms and $3N$ is our matrix dimension. There is a $1/3N$ chance that a given index is the randomized hot index, so

\[ \left\langle v_{i}^{\operatorname{1Hot}}v_{j}^{\operatorname{1Hot}}v_{k}^{\operatorname{1Hot}}v_{l}^{\operatorname{1Hot}}\right\rangle=\begin{cases}3N&\text{all indices equal}\\
0&\text{otherwise}\end{cases}. \] (S10)

It follows that

$\left\langle\left(v^{\operatorname{1Hot},T}Av^{\operatorname{1Hot}}\right)^{2}\right\rangle=3N\sum_{i}A_{ii}^{2}$ (S11)

Using Eqs. (S3) and (S6) we arrive at the mean-squared error of the 1-hot column estimator

$\operatorname{MSE}[v^{\operatorname{1Hot}}]=(3N-1)\sum_{i}A_{ii}^{2}-\sum_{i,k;i\neq k}A_{ii}A_{kk}.$ (S12)

In our setting, $A=B^{T}B$, where $B$ represents the error in the Hessian predicted by our Machine Learning (ML) model,

$A$ $=B^{T}B$ (S13)
$B$ $=\frac{\tilde{H}-H}{3N}.$ (S14)

Physical Hessians are expected to decay rapidly with interatomic distance, $|\mathbf{r}_{i}-\mathbf{r}_{j}|$, a behavior observed empirically and consistent with the locality assumptions underlying most MLIP architectures.*[28, 47]* Consequently, only $O(N)$ matrix elements of $A$ are expected to contribute appreciably in the limit of the system size $N$.

Under this locality assumption, the Hutchinson estimator yields a mean-squared error (Eq. (S7)) that scales as $O(N)$ for extensive quantities, implying an RMSE that grows as $\sqrt{N}$ and a relative error that decreases with increasing system size. By contrast, for one-hot random vectors (Eq. (S8)), all contributions to the mean-squared error (Eq. (S12)) arise from diagonal elements $A_{ii}$. Although these terms are likewise localized, the resulting MSE scales as $O(N^{2})$, leading to less favorable asymptotic behavior for large systems compared to Hutchinson under the same locality assumptions.

B Training Procedure

The models were trained using mini-batch gradient descent with adaptive learning-rate schedules. The maximum number of epochs was set to 5000, although this limit was never reached in practice. Each method was trained as an ensemble of five independent models with different random seeds, and the reported RMSE values correspond to the ensemble means with error bars representing standard deviations.

- Dataset split: For the reactant, transition state, and product dataset, 80% of the structures were used for training, 10% for validation and 10% were held as a benchmark test set.
- Optimizers: Model weights were updated using the AdamW optimizer, while biases were updated with stochastic gradient descent (SGD). Both optimizers were set to a learning rate of $1\times 10^{-4}$ , $\beta_{1}=0.9$, $\beta_{2}=0.999$, $eps=1\times 10^{-8}$ , and a weight decay of $1\times 10^{-2}$
- Batch composition: Each minibatch contained 400 molecular structures drawn from a dataset of reactants, transition states, and products, ensuring coverage of equilibrium configurations.
- Hutchinson estimator (PHL): Probe vectors were sampled as independent Gaussian random variables $v_{i}^{\text{Hutch}}$, which satisfy $\langle v_{i}\rangle=0$ and $\langle v_{i}v_{j}\rangle=\delta_{ij}$. This ensures that the estimator is unbiased, i.e., $\mathbb{E}[v^{T}Av]=\text{tr}(A)$.
- One-column estimator: Following recent work in Hessian training,*[32]* probe vectors were chosen to be one-hot basis vectors $v_{i}^{\text{1Hot}}$ scaled by $\sqrt{3N}$. Only one element in each probe vector is set to one, while the rest are set to zero. This approach corresponds to selecting a single column of the Hessian at random. In mathematical terms,

$v_{i}^{\text{1Hot}}=\sqrt{3N}\,\delta_{i,c}\qquad c\in\{1,\ldots,3N\}$
- Framework: The models were implemented in PyTorch with custom operators for force and Hessian backpropagation.
- Hardware: Training was performed on various models of NVIDIA GPUs in high performance computing clusters.
- Training runtime: Wall-clock time per epoch was recorded for each training method (E–F, E–F–HVP One-Column, E–F–HVP PHL, and E–F–H) on NVIDIA A6000 GPUs. The reported values correspond to averages over multiple epochs.
- DFT data generation: CPU scaling of reference calculations was evaluated using Gaussian16 for molecular systems containing up to 100 atoms.

## Appendix C Bland–Altman Analysis of PHL Probing Strategies

This section compares two methods that differ only in the distribution used to probe curvature through Hessian-vector products (HVPs): (i) one-column (one-hot) probing, which samples a single Hessian column per probe, and (ii) PHL probing (Hutchinson-type randomized projections), which aggregates information across many curvature directions in expectation. For each metric (energy, force, and Hessian RMSE), we use Bland-Altman plots to visualize the paired differences between methods, defined throughout as

$\Delta\text{RMSE}\equiv\text{RMSE}_{\text{one-column}}-\text{RMSE}_{\text{PHL}}$

Thus, a positive $\Delta$RMSE indicates that Hutchinson-based PHL is more accurate.

# C.1 Randomized HVP Probes per Minibatch

In the randomized-probe setting, we resample the probing vector(s) at each minibatch, so that both estimators access many independent curvature directions over the course of training. Figures S1-S3 show that the zero line lies within the  $95\%$  confidence interval (CI) of the mean difference across the Test, IRC, and NMS datasets for all three metrics. Therefore, when probes are randomized per minibatch, the one-hot and Hutchinson estimators yield statistically indistinguishable accuracy, consistent with our main-text conclusion that both stochastic estimators effectively approximate full Hessian supervision in this regime for our characteristic system sizes.

![images/image9.jpg](images/image9.jpg)
Figure S1: Bland-Altman analysis of energy RMSE differences for randomized probes per minibatch. Differences are defined as  $\Delta \mathrm{RMSE} = \mathrm{RMSE}_{\mathrm{one - column}} - \mathrm{RMSE}_{\mathrm{PHL}}$  across the Test, IRC, and NMS datasets. Each point represents a paired set of trained models. The green line denotes the mean difference and the shaded region its  $95\%$  CI; black lines indicate limits of agreement. The  $95\%$  CI includes zero for all datasets, indicating no statistically significant difference in energy accuracy between probing strategies when probes are randomized each minibatch.

![images/image10.jpg](images/image10.jpg)
Figure S2: Bland-Altman analysis of force RMSE differences for randomized probes per minibatch, with  $\Delta$ RMSE defined as in Fig. S1. The  $95\%$  CI includes zero for all datasets, indicating no statistically significant difference in force accuracy between one-column and PHL methods when probes are randomized each minibatch.

![images/image11.jpg](images/image11.jpg)
Figure S3: Bland-Altman analysis of Hessian RMSE differences for randomized probes per minibatch, with  $\Delta$ RMSE defined as in Fig. S1. The  $95\%$  CI includes zero for all datasets, indicating no statistically significant difference in Hessian accuracy between probing strategies in the randomized-probe regime.

# C.2 Fixed HVP Probe per System (Data-Limited Regime)

In the fixed-probe setting, we assign each molecular system a single probe vector (i.e., one reference HVP) that remains fixed throughout training. This mimics a data-limited regime where only restricted second-derivative information is available per structure. In this setting, Gaussian-probed PHL becomes consistently more accurate than one-column probing, with the strongest gains appearing on the extrapolative NMS dataset.

Figures S4-S6 show that the mean difference shifts positively (favoring PHL), and the zero line falls outside the  $95\%$  CI for increasingly many datasets as the target becomes more challenging.

Specifically, energy differences are significant on NMS, force differences are significant on IRC and NMS, and Hessian differences are significant on all three datasets. These trends agree with our main-text statistics: under fixed-vector conditions on NMS, Hutchinson-based PHL yields additional reductions of  $6.2\%$  (energy RMSE),  $5.6\%$  (force RMSE) and  $11.2\%$  (Hessian RMSE) relative to one-column probing (paired  $t$ -tests).

![images/image12.jpg](images/image12.jpg)
Figure S4: Bland-Altman analysis of energy RMSE differences for fixed probes per system. Differences are defined as  $\Delta \mathrm{RMSE} = \mathrm{RMSE}_{\mathrm{one - column}} - \mathrm{RMSE}_{\mathrm{PHL}}$  across the Test, IRC, and NMS datasets. The mean difference is not significant for Test and IRC (95% CI includes zero), but is significantly positive for NMS (95% CI excludes zero), indicating better energy extrapolation for Hutchinson-based PHL in the data-limited regime.

![images/image13.jpg](images/image13.jpg)
Figure S5: Bland-Altman analysis of force RMSE differences for fixed probes per system, with  $\Delta$ RMSE defined as in Fig. S4. The  $95\%$  CI excludes zero for IRC and NMS, demonstrating that the one-column estimator yields significantly higher force errors than the Hutchinson-based PHL estimator on intermediate (IRC) and extrapolative (NMS) geometries.

![images/image14.jpg](images/image14.jpg)
Figure S6: Bland-Altman analysis of Hessian RMSE differences for fixed probes per system, with  $\Delta$ RMSE defined as in Fig. S4. The  $95\%$  CI excludes zero for Test, IRC, and NMS, indicating that the one-column estimator produces significantly higher Hessian errors than the Hutchinson-based PHL estimator across all datasets in the fixed-probe regime, with the largest discrepancies observed on NMS.

Summary. When HVP probes are randomized at each minibatch, one-column and PHL are statistically indistinguishable for our characteristic system size (a median of  $N \sim 14$ ). When only a single probe per system is available (data-limited regime), PHL is consistently more accurate, particularly for extrapolative geometries, establishing it as the more reliable default probing strategy for practical second-order training with limited curvature data.