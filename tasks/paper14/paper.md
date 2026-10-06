# Simulating plasma wave propagation on a superconducting quantum chip

Bhuvanesh Sundar [ bhundar@rigetti.com R. Sundar@rigetti@Ibnl.gov ] Bram Evert [ joseph5@Ibnl.gov ] Vasily Geyko [ joseph5@Ibnl.gov ] Andrew Patterson [ yuan.shi@colorado.edu ] Ilon Joseph [ ilon@colorado.edu ] Yuan Shi [ yuan@colorado.edu Department of Physics, Center for Integrated Plasma Studies, University of Colorado Boulder, Colorado 80309, USA ]

###### Abstract

Quantum computers may one day enable the efficient simulation of strongly coupled plasmas that lie beyond the reach of classical computation in regimes where quantum effects are important and the scale separation is large. In this article, we take a first step toward efficient simulation of quantum plasmas by demonstrating linear plasma wave propagation on a superconducting quantum chip. Using high-fidelity and highly expressive device-native gates, combined with an error-mitigation technique, we simulate the scattering of laser pulses from inhomogeneous plasmas. Our approach is made feasible by the identification of a suitable local spin model whose excitations mimic plasma waves, and whose circuit implementation requires a lower gate count than other proposed approaches that would require a future fault-tolerant quantum computer. This work opens avenues to study more complicated phenomena that cannot be simulated efficiently on classical computers, such as nonlinear quantum dynamics when strongly coupled plasmas are driven out of equilibrium.

## I Introduction

The ability of future fault-tolerant quantum computers to naturally simulate fully entangled quantum dynamics may one day enable simulations of the fundamental forces of nature with unprecedented accuracy and resolution *[1]*. This will ultimately enable the simulation of high-energy-density plasmas where quantum effects become increasingly important at high energies and/or densities *[2, 3, 4, 5, 6]*. In fact, the strongly coupled quantum effects that occur in degenerate fermionic matter also occur in the warm dense matter regime in stars and planetary cores *[4]*, in stopping power and opacity calculations *[6, 7]*, and in extreme astrophysical plasmas such as those that occur near black holes and neutron stars *[8, 9]*. At a fundamental level, all of these problems are classically hard to simulate (bounder-error quantum polynomial-time complete *[10]*).

The rapid development of quantum computing capabilities has spurred the development of efficient quantum algorithms that target plasma simulation. Recent work has developed efficient algorithms for evolving general wave equations *[11]*, the propagation of waves in plasmas *[12, 13, 14, 15, 16, 17, 18, 19]*, and the underlying physical processes of advection and diffusion *[20, 21]*. Simulating intrinsically quantum plasmas can leverage the significant progress in quantum algorithms for scattering in quantum field theory *[22]*, condensed-matter systems *[23]*, scalar field theory *[24]*, and lattice gauge theories *[25]*. In fact, although answering certain questions about the classical dynamics of linear oscillators is intrinsically hard, these problems can be solved efficiently with quantum computers *[26]*.

While these algorithms generally provide an exponential compression, their implementation on near-term hardware is challenged by the high gate depth required to implement them, typically requiring a gate depth that grows as a polynomial in qubit number. Thus, several studies that have explored the use of noisy intermediate-scale quantum (NISQ) devices for simulating plasma-relevant point examples *[27, 28, 29, 7, 20, 21, 22, 23, 24, 25, 26, 28, 29]* have shown that fault-tolerant quantum computers will be required for high-precision calculations *[17, 30]*.

In this article, we take the first steps toward simulating the physics of wave propagation in quantum plasmas using a strategy that is natural for near-term quantum hardware and apply it to modeling the scattering of electromagnetic waves in plasmas. Typically, solving even a linear quantized scattering problem with a classical computer requires fully solving for the eigenvalues and eigenvectors of the dispersion operator, which requires $\mathcal{O}(N^{\omega})$ operations, where $N$ is the number of discretized spatial points and $2<\omega<3$. Yet, with a quantum computer we can use Trotterized evolution of a local spin-chain model to efficiently simulate linear plasma wave propagation in a plasma with an inhomogeneous density profile, with no requirement to solve for the eigenvalues or eigenvectors. As shown in Appendix C, this leads to a polynomial speedup for simulating the physics of quantum plasmas. Moreover, the model only requires shallow circuits and nearest-neighbor qubit couplings, making it suitable for current devices.

We present a one-dimensional version of our model, which, as shown in the Appendix A, can be easily generalized to two, three, or even higher dimensions with a cost that only scales linearly with dimension (the number of interacting neighbors) rather than as a polynomial in the number of qubits. Similar spin-lattice models have already been demonstrated on quantum devices to simulate quantum many-body dynamics *[23, 31]* and magnetization at infinite temperature *[32]*; however, the

models in the class studied here have distinct characteristics that make them relevant to plasma physics. While we simulate a solvable spin-chain model in this article, it is possible to add simple extensions to the model with, e.g., $ZZ$ terms in the spin Hamiltonian, which can be used to study classically hard-to-simulate phenomena, such as quantum plasma dynamics including nonlinear and many-body quantum effects. We leave such an experiment to future efforts, and instead focus here on a proof-of-principle demonstration of plasma wave scattering.

We implemented the plasma wave simulations on a nine-qubit sublattice of Rigetti’s Ankaa-3 superconducting chip using a strategy that targets universal quantum computation on near-term devices. We compiled our Trotterized evolution operator using the $\sqrt{\text{iSWAP}}$-like FSIM gates to achieve high-fidelity low gate depth circuits. Unlike previous experiments *Bergman et al. (2016)* in which targeted gates were calibrated for specific use cases, the native $\sqrt{\text{iSWAP}}$-like FSIM gates are highly expressive when combined with single-qubit rotations *Bergman et al. (2016)* and, in general, allow efficient implementation of arbitrary logical circuits.

While these experiments have relatively forgiving requirements, they still require sophisticated error-characterization and error-mitigation techniques in order to achieve meaningful results *Bergman et al. (2016); Berré et al. (2017); Berré et al. (2018)*. We developed a pseudotwirling technique to twirl the coherent noise in the FSIM gate, and mitigated this noise using Clifford data regression *Bergman et al. (2018)*, which is a heuristic and scalable error-mitigation technique. The overall improvement in fidelity yields the first qualitatively accurate quantum hardware simulation of plasma waves.

## II Model

We wish to simulate the propagation of electromagnetic (EM) waves in an unmagnetized plasma. These waves, which are coupled oscillations of the electromagnetic fields, including the response of the plasma current, are effectively described by the second-order wave equation

$(\partial_{t}^{2}-c^{2}\partial_{x}^{2}+\omega_{p}^{2})A=0$ (1)

and have the dispersion relation

$\omega_{k}=\pm\sqrt{\omega_{p}^{2}+c^{2}k^{2}}.$ (2)

Here, $\omega_{p}$ is the plasma frequency, defined by $\omega_{p}^{2}=e^{2}n/m\epsilon_{0}$, for particles of charge $e$, mass $m$, and density $n$, and $c$ is the speed of light (or the electron sound speed for electrostatic Langmuir waves). For a magnetized plasma, this dispersion relation is still valid for long wavelength EM modes (L-,R-,O- and X-waves *Bergman et al. (2016); Berré et al. (2017)*), but where $\omega_{p}$ has more complicated dependence on the plasma frequency and the cyclotron frequency, $\omega_{c}=eB/m$ for magnetic field $B$, for both electrons and ions. In the limit that $\omega_{p}\rightarrow 0$, the spectrum resembles that of low-frequency long-wavelength ion acoustic waves and magnetohydrodynamic waves in a plasma, where $c$ is replaced by the sound or the Alfvén speed.

We show here that these waves are mimicked by spin-wave excitations in a local spin model given by

$H=-\frac{J}{4}\sum_{j=1}^{N-1}(\sigma_{j}^{x}\sigma_{j+1}^{y}-\sigma_{j}^{y}\sigma_{j+1}^{x})+\frac{1}{2}\sum_{j=1}^{N}\Delta_{j}(-1)^{j}\sigma_{j}^{z}.$ (3)

where $\sigma_{i}^{\alpha}$ are Pauli operators on spin (qubit) $i$. We show in detail in Appendix A how to derive the spin model that is equivalent to the model of plasma waves. The lattice model of Eq. (3) can represent quasiparticles or spins in a condensed-matter system interacting with strength $J$, where the field $\vec{\Delta}_{i}$ alternates in direction, as illustrated in Fig. 1(a). The mapping to EM waves in plasmas become apparent when we write the equation of motion for singly excited states $\ket{0\cdots 1_{j}\cdots 0}$. Writing the amplitude of the local excitation as $b_{j}$, the equation of motion for these amplitudes is

$i\hbar\partial_{t}b_{j}=\frac{i}{2}J(b_{j-1}-b_{j+1})+\Delta_{j}(-1)^{j}b_{j}.$ (4)

Differentiating again and assuming uniform $\Delta_{j}$ gives

$\hbar^{2}\partial_{t}^{2}b_{j}=\frac{1}{4}J^{2}(b_{j+2}-2b_{j}+b_{j-2})-\Delta^{2}b_{j},$ (5)

which is a central-difference discretization of Eq. (1). The degrees of freedom at the odd and even sites of the lattice model can be interpreted as linear combinations of the electric field and current in a domain with half the total number of grid points. For uniform $\Delta_{j}=\Delta$, this discretized wave equation can be solved exactly: the eigenmodes have the dispersion relation *Bergman et al. (2016)*

$\hbar\omega_{k}=\pm\sqrt{\Delta^{2}+J^{2}\sin^{2}ka}.$ (6)

where $a$ is the lattice spacing. At long wavelength, $\hbar\omega_{k}\simeq\pm\sqrt{\Delta^{2}+(Jka)^{2}}$ is the same dispersion relation as Eq. (2), and is illustrated in Fig. 1(b).

Thus, we can efficiently simulate the linear dynamics of plasma waves on quantum hardware using Trotterized evolution of $H$. By varying $\Delta_{j}$, e.g. making it uniform, having sharp jumps, or having a spatial profile, we can achieve full-wave simulations of linear wave dynamics in uniform plasmas, ducted plasmas, or inhomogeneous plasmas. In this paper, we consider simulation of all three scenarios.

## III Experiment

We perform our experiments on a nine-qubit sublattice of Rigetti’s Ankaa-3 chip which is composed of transmon qubits arranged on a square grid. The qubits are connected by tunable couplers, which, when activated,

![images/image1.jpg](images/image1.jpg)

![images/image2.jpg](images/image2.jpg)
FIG. 1. We implement a one-dimensional spin model with nearest-neighbor interactions  $J_{ij}$  and a staggered field  $\vec{\Delta}_i$  [see Eq. (3)], which has the same energy spectrum as that of electromagnetic waves in a plasma, on a superconducting quantum computer. We use this model to simulate linear electromagnetic wave dynamics in a plasma.

|  Metric | Value  |
| --- | --- |
|  T1 | 37.6μs  |
|  T2 | 22.3μs  |
|  FSIM Infidelity | 0.60%  |
|  FSIM Duration | 54ns  |
|  RX(π/2) Infidelity | 0.09%  |
|  RX(π/2) Duration | 20ns  |

TABLE I. Summary metrics of the sublattice of the Ankaa-3 QPU. The  $T_{1}$  and  $T_{2}$  values are measured individually, and the gate fidelities are mean values reported in the circuit context.

realize an interaction between the qubits [42]. The interaction, which is predominantly a transverse interaction with an additional small longitudinal interaction in the two-qubit Hilbert space, realizes the so-called FSIM gate [32] up to one-qubit phases (see Appendix E). We calibrate the duration of the interaction to target a gate that is close to a  $\sqrt{\mathrm{iSWAP}}$  up to one-qubit phases. The  $\sqrt{\mathrm{iSWAP}}$  is a highly expressive gate that can be used to efficiently compile arbitrary two-qubit gates [33]. Since the  $\sqrt{\mathrm{iSWAP}}$  gate has a shorter duration than the iSWAP gate, it has reduced decoherence and thus a higher fidelity than iSWAP. On each edge of our quantum processor, we calibrate and learn the gate being realized using process tomography and a variant of cross-entropy benchmarking, as detailed in Appendix F1. We implement arbitrary single-qubit unitary operations using four phased microwave pulses known as the "PMW4" decomposition [43]. The error per layered gate (EPLG) [44] in the relevant nine-qubit sublattice on Ankaa-3 was found to be  $1.48\%$  (see Table I for other performance data). Using these gates, we express the target logical circuits using an approximate numerical compilation technique, as explained in Appendix F4. As we describe below, the log

![images/image3.jpg](images/image3.jpg)

![images/image4.jpg](images/image4.jpg)

![images/image5.jpg](images/image5.jpg)
FIG. 2. The dispersion relation of the spin Hamiltonian [Eq. (3)] is analogous to waves in plasmas. The spectrum is measured via a many-site Ramsey-type experiment, where the time evolution of the complex phase of local spin observables reveals the energies of the eigenmodes. We consider (a-b) a low density plasma  $(\Delta \rightarrow 0)$  and (c-d)  $\Delta = J / 4$ . Panels (a,c) show the error-mitigated observables versus time; panels (b,d) show the extracted excitation spectrum from noisy (yellow) and mitigated (blue) data, along with the exact spectrum (teal) and the expected spectrum from a noiseless simulation of the circuit (gray).

![images/image6.jpg](images/image6.jpg)

ical circuits we target realize Trotterized evolution with Eq. (3). The resulting experimental circuit is illustrated in Fig. 8.

We mitigate hardware noise in the experiment using a combination of twirling and a linear regression technique, similar to Ref. [38]. The twirling method that is commonly used in quantum circuits with Clifford two-qubit gates is Pauli twirling, and it transforms any Markovian noise that is closed within the qubit's computational space to stochastic Pauli noise, which is amenable to error mitigation; however, the FSIM gate that we used in our experiment is not a Clifford gate, and therefore it cannot be twirled using Pauli gates. We therefore developed a pseudotwirling technique for the FSIM gates that converts Markovian coherent noise within the computational space, other than errors in the calibration of the FSIM parameters, to stochastic Pauli noise (see Appendix G1 for details). We then apply the linear regression technique to mitigate the effects of the now mostly stochastic noise (see Appendix G2). Additionally, where applicable, we rescale the observables such that the expectation value of  $\sigma_{tot}^{z}$  is conserved (see Appendix G3).

At the end of all of the circuits, we obtain classical shadows of the state via measurements of all the qubits in random bases [45]. Classical shadows is a powerful technique for estimating several observables to high accuracy using relatively few shots, and, importantly here, it also effectively acts as readout-error symmetrization due to the randomization of the measurement basis. The random basis for measurement on individual shots is triggered by a pseudorandom number generator on the quan

tum chip’s control system, which enables $10^{6}$ shots in a few seconds.

![images/image7.jpg](images/image7.jpg)
Figure 3: Phase evolution of the eigenmodes of the system versus time for (a) $\Delta=0$, and (b) $\Delta=J/4$. Magenta, teal, and yellow points plot the complex phase of the eigenmodes obtained from error-mitigated data, the raw experimental data, and a noiseless simulation of the circuit. Black lines show linear fits to the error-mitigated data. Other colored lines are guides to the eye. Each panel corresponds to one eigenmode $q$. The eigenfrequency of each eigenmode is obtained by applying a linear fit to the error-mitigated data.

## IV Measuring the dispersion relation

We design our first experiment to measure the energies of the spin-wave excitations, using a many-body Ramsey-type scheme.

We initialize one qubit in the superposition $(|0\rangle+|1\rangle)/\sqrt{2}$ and all remaining qubits in the $|0\rangle$ state. This initial state is a superposition of the vacuum of spin waves, and of spin waves at all wave vectors $k$ with appropriate coefficients. Denoting $|\mathrm{vac}\rangle$ as the state with all qubits as $|0\rangle$, which is the vacuum of spin waves, $|i\rangle\equiv\sigma_{i}^{+}|\mathrm{vac}\rangle$ as the state with qubit $i$ in $|1\rangle$ and all others as $|0\rangle$, and $|k\rangle$ as the spin wave with wave vector $k$, the initial state is

$|\psi(t=0)\rangle=\frac{|\mathrm{vac}\rangle+|i\rangle}{\sqrt{2}}=\frac{|\mathrm{vac}\rangle+\sum_{k}c_{ik}\,|k\rangle}{\sqrt{2}}.$ (7)

Here, $c_{ik}$ are the real-space amplitudes of the spin wave with wave vector $k$ (see Appendix B for the exact solutions of the spin waves).

We then realize first-order Trotterized evolution with $H$ on the chip, using a Trotter time-step size $\delta=0.8$, wherein each wave vector component $k$ accumulates phase at the rate $\omega_{k}$. The vacuum state remains as a reference, having no phase evolution. After time evolution, the wave function is

$|\psi(t)\rangle=\frac{|\mathrm{vac}\rangle+\sum_{k}c_{ik}\,|k\rangle\,e^{-i\omega_{k}t}}{\sqrt{2}}.$ (8)

We measure $\langle\sigma_{j}^{x}\rangle+i\,\langle\sigma_{j}^{y}\rangle$, which is sensitive to the relative phases accumulated between the vacuum and the spin waves during Trotterized evolution. The expectation value of $\sigma_{j}^{x}+i\sigma_{j}^{y}$ is

$\langle\psi(t)|\sigma_{j}^{x}+i\sigma_{j}^{y}|\psi(t)\rangle=$ $\frac{1}{2}\sum_{k}c_{ik}e^{i\omega_{k}t}\,\langle k|j\rangle$
$=$ $\frac{1}{2}\sum_{k}c_{ik}c_{jk}^{*}e^{i\omega_{k}t}.$ (9)

We perform the experiment with $(\Delta&gt;0)$ and without $(\Delta=0)$ a plasma. We remark that $(\sigma_{j}^{x}+i\sigma_{j}^{y})$ is not a Hermitian observable, but we can measure its real part $\sigma_{j}^{x}$ and imaginary part $\sigma_{j}^{y}$ separately.

Figures 2(a) and (c) show the localized spin-wave packet spreading through the lattice in the cases $\Delta=0$ and $\Delta=J/4$, respectively. We then extract the frequency $\hbar\omega_{q}$ from the complex phase evolution of $\sum_{j}\langle\sigma_{j}^{x}+i\sigma_{j}^{y}\rangle\,c_{jq}$. We know the amplitudes $c_{jq}$ for the eigenmodes of this Hamiltonian (see Appendix B); therefore, we are able to extract the spectrum of $H$ in this way. Figure 3 shows that the complex phase of the eigenmodes evolves approximately linearly with time. The rate of phase evolution is the eigenfrequency of the modes.

The resulting spin-wave excitation spectra are shown in Figs. 2(b) and (d) for $\Delta=0$ and $\Delta=J/4$, respectively. The teal lines show the exact frequencies of the spin-wave excitations. The gray dots are obtained by extracting the frequency spectrum from a noiseless simulation of the circuit, using the same postprocessing procedure. There is a small difference between the exact values (teal curve) and the noiseless simulation values (gray dots) due to Trotter error in the evolution. Finally, the blue and yellow dots are the frequency spectra from experimental data with and without error mitigation, respectively. The frequency spectrum obtained from experimental data agrees excellently with that from the noiseless simulation and with the exact eigenfrequencies.

In particular, the experimental data show the presence of a mode at $\omega=0$ when $\Delta=0$ [Fig. 2(b)], whereas there

![images/image8.jpg](images/image8.jpg)
FIG. 4. An electromagnetic wave packet propagating (a) in vacuum  $(\Delta = 0)$ , (b) from vacuum to a sharp jump in plasma density (which mimics the edge of a confined overdense plasma), and (c) through an inhomogeneous overdense plasma with a Gaussian density profile. In each case, the profiles of plasma frequency are shown on the bottom, and the intensity of the propagating wave packet is shown on the top. In (a), the wave packet propagates nearly ballistically until it approaches the edge, where the boundary condition is reflective. In (b) and (c), the wave packet propagates until it approaches the sharp jump or inhomogeneous profile and mostly reflects back. The black lines show the center of mass (CoM) of the wave packet obtained from the mitigated experimental data, and the red lines show the wave packet's CoM from a noiseless simulation of the experiment.

is no mode at  $\omega = 0$  when  $\Delta &gt; 0$  [Fig. 2(d)]. In fact, the spectrum for  $\Delta &gt; 0$  has an energy gap. The energy gap is the plasma frequency  $\omega_{p}$ , which is determined by the plasma density; only EM waves with frequency greater than  $\omega_{p}$  propagate in the plasma.

In the absence of wave-wave interactions, this method represents a highly sensitive scalable test of the ability to accurately reconstruct each eigenmode. Extracting the spectrum of a general nonsolvable Hamiltonian may require sophisticated techniques like quantum phase estimation.

# V. PROPAGATION OF WAVES IN THE PLASMA

Modeling the propagation of EM waves in plasmas with complex geometries is a computationally expensive, but important, task in many plasma-relevant scenarios. Here, we take the first step toward this goal on a quantum computer, by simulating propagation of EM waves in plasma with increasingly more general density profiles. In particular, we consider propagation in three scenarios - vacuum  $(\Delta = 0)$ , a plasma with a sharp jump in density, and an inhomogeneous plasma with a smooth density profile. While the first two scenarios are well understood analytically, they serve as control experiments to benchmark and validate our simulation. The third scenario is a proof-of-principle demonstration of how a quantum computer could be used to simulate real EM wave propagation in more realistic plasmas.

In our experiments, the spin-wave packet,  $|\psi \rangle = \sum_{i}\alpha_{i}|i\rangle$ , is prepared with an entangling operation compiled into our native gates. This wave packet has exactly one spin-wave excitation, distributed over multiple sites in real space, and multiple  $k$  in wave-vector space. The mean  $k$  is dictated by the complex phases of  $\alpha_{i}$ , and the spread over  $k$  is dictated by the inverse of the spread

in real space. In a large-scale fault-tolerant experiment, a broad real-space wave packet may be prepared such that it is concentrated in  $k$ . Here, we initialize a two-site wave packet. The relative complex phase between  $\alpha_0$  and  $\alpha_1$  is 0; therefore, the initial condition is a superposition of left- and right-propagating waves. We launch the wave near the left boundary from the region with  $\Delta = 0$ . Because we use a reflective boundary condition, the net effect is that a wave packet is launched with a group velocity  $v = J / a\hbar$  that is positive.

Figure 4 shows the wave propagation in the three scenarios. In the first case, Fig. 4(a), the plasma density is zero; therefore, the wave is the vacuum EM wave. The solid curves plot the center of mass of the wave packet. In a noiseless simulation (red), the wave packet propagates freely at its group velocity. Deviation from a constant speed occurs at later times due to finite size of the lattice. The experimental data track this propagation qualitatively well until at least half the lattice.

In Figs. 4(b) and (c), the wave packet encounters a sharp jump in the plasma density and an inhomogeneous plasma density profile, respectively. The maximum plasma barrier,  $\Delta = J$ , is larger than the wave packet's vacuum frequency, and it therefore corresponds to an overdense plasma. In both scenarios, most of the wave packet is reflected because the plasma is overdense. The error-mitigated experimental data show good qualitative agreement with the noiseless numerical simulation, as shown by the qualitative agreement in the traces of their centers of mass.

State-of-the-art full-wave simulations use  $\mathcal{O}10^{6} - 10^{8}$  spatial grid points on classical computers. While current qubit error rates limit us to simulations of plasma wave propagation on nine qubits, which corresponds to a spatial discretization on nine grid points, smaller error rates that allow experiments on larger lattices are within reach in the near term. Fault-tolerant computers, which would be needed for large-scale quantum simulations, are

![images/image9.jpg](images/image9.jpg)

![images/image10.jpg](images/image10.jpg)
FIG. 5. Classical simulations of EM waves propagating in various mass profiles: (a) reflection from a sharp boundary and (b) propagation through a more complicated profile. The inset in (a) shows the intensity of the reflected wave,  $|r|^2$ , versus  $k$ . The solid line shows the analytically predicted reflection (see Appendix I) and the points show the numerically computed values.

also rapidly advancing. With this motivation, in Fig. 5, we present a classical emulation of the lattice model for a large system with 100 sites. The Hilbert-space size for simulating a single wave packet propagating in a plasma scales in proportion to the number of grid points  $n$  and is therefore feasible to simulate classically.

# VI. QUANTUM ADVANTAGE

Quantum computers are naturally able to simulate quantum dynamics and, thus, can offer significant quantum advantage. Here, we show a brief calculation of the expected quantum advantage for simulating propagating plasma waves, with more details given in Appendix C.

The model that approximates the quantum field theory of linear electromagnetic wave propagation in the low photon (or plasmon) density limit is the noninteracting bosonic model in Eq. (A16), which is an integrable model. For a lattice with  $N_{s}$  sites, the evolution of the noninteracting model can be classically computed in time  $C = \mathcal{O}(M_0KN_s^2) - \mathcal{O}(M_0N_s^3)$ , where  $M_0$  is a constant that depends on the initial conditions and/or the

number of single-particle observables and  $K$  represents the sparsity of the matrix or the number of Krylov iterations. In fact, in one dimension (1D), it only takes  $C = \mathcal{O}(M_0KN_s^2)$ . The dominant contribution to the classical cost is either exact diagonalization or exponentiation of the single-particle Hamiltonian. Therefore, classical simulation is feasible if the initial condition can be expressed in terms of a polynomial number of single-particle states or if a polynomial number of single-particle observables is desired.

Since the quantum computer does not need to explicitly diagonalize the Hamiltonian, it can still provide significant advantage in these cases. The quantum algorithm's cost only scales with the number of time steps,  $N_{t}$ , which is proportional to  $N_{x}$  due to the Courant-Friedrichs-Lewy criterion for stability and accuracy,  $Q \propto \mathcal{O}(d^{2}N_{x})$ , where  $d$  is the number of spatial dimensions. Therefore, the quantum speedup range is

$$
C / Q = \mathcal {O} \left(M _ {0} K N _ {x} ^ {2 d - 1} / d ^ {2}\right) \text {t o} \mathcal {O} \left(M _ {0} N _ {x} ^ {3 d - 1} / d ^ {2}\right). \tag {10}
$$

Nonetheless, a practical advantage may not be realizable on currently feasible system sizes; we discuss cases where quantum advantage may still be achievable in Appendix C.

A more general Hamiltonian with nonlinear interactions, e.g. with additional ZZ terms, does not map to a noninteracting model and, hence, cannot be classically simulated in poly  $(N_{s})$  time. Solving this system classically generically takes exponential time in  $N_{s}$ , scaling as  $\mathcal{O}(2^{3N_s})$  in the worst case. Thus, the case of nonlinear interactions is also promising for realizing quantum advantage.

# VII. SUMMARY

We simulate the scattering of plasma waves from an inhomogeneous medium using a quantum device. Our spin-lattice model represents quantum plasma waves using  $n$  qubits for  $n$  spatial grid points, which results in a circuit with a shallow gate depth of  $\mathcal{O}(n)$  that can be run on NISQ devices. While we have only used singly excited spin states, which form an  $n$ -dimensional subspace of the  $2^{n}$ -dimensional Hilbert space, to represent linear plasma waves, more complicated spin excitations can be simulated just as efficiently on quantum computers. With more general spin Hamiltonians, for example, with additional  $ZZ$  interactions, our approach can be used to simulate nonlinear effects in plasmas, such as electromagnetically induced transparency [46], laser-plasma scattering, and modulational instabilities [7].

# ACKNOWLEDGMENTS

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under

Award no. DE-SC0021661. This publication was prepared to include an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof. The work by Lawrence Livermore National Laboratory was performed under the auspices of the U.S. Department of Energy (DOE) under Contract No. DE-AC52-07NA27344. I.J. and V.G. were supported by the DOE Office of Fusion Energy Sciences projects SCW1736 and SCW1680. Y.S. is supported in part by U.S. Department of Energy under Grant No. DE-SC0020393. B.S. and B.E. thank Mark J. Hodson, Maxime Dupont, and Tyler Wilson for valuable discussions.

I.J. and B.S. developed the spin chain model to emulate waves in plasmas and derived the protocol for measuring the dispersion relation. Y.S. conceived of the scattering experiments performed in this work. B.E. ran the experiments on Rigetti’s quantum chips. A.P. and B.E. developed the error-mitigation techniques used in the experiment. B.S. provided theory support, analyzed the experimental results, and wrote the manuscript with contributions and editing from all authors.

B.S., A.P., and B.E. are, have been, or may in the future be participants in incentive stock plans at Rigetti Computing Inc.

## Data Availability

All the relevant data created from experiments in this work are publicly available at doi.org/10/5281/zenodo.16115660 *[47]*.

## Appendix A Mapping Plasma Waves to Spin Chains

Here, we show how to derive the spin model [Eq. (3)] to simulate a continuum plasma model.

### A.1 Continuum model

Plasma dynamics is determined by Maxwell’s equations,

$\partial_{t}\mathbf{B}$ $=-\nabla\times\mathbf{E}$ $\nabla\cdot\mathbf{B}$ $=0$ (10)
$\partial_{t}\mathbf{E}$ $=c^{2}\nabla\times\mathbf{B}-\frac{\mathbf{J}}{\epsilon_{0}}$ $\nabla\cdot\mathbf{E}$ $=\frac{\rho}{\epsilon_{0}}$ (11)

and the evolution of the charge carriers in response to the electromagnetic fields, where $\epsilon_{0}$ is the electric permittivity. For an unmagnetized plasma composed of light electrons and heavier ion species, the linearized electric current evolves via

$\partial_{t}\mathbf{J}=-\nabla v_{e}^{2}\rho+\epsilon_{0}\omega_{p}^{2}\mathbf{E}$ (12)

where $\omega_{p}$ is the plasma frequency and $v_{e}^{2}=\gamma T_{e}/m_{e}$ sets the electron sound speed. Combining these equations leads to the wave equation for a massive vector field

$\partial_{t}^{2}\mathbf{E}+c^{2}\nabla\times\nabla\times\mathbf{E}=-\frac{\partial_{t}\mathbf{J}}{\epsilon_{0}}=-\omega_{p}^{2}\mathbf{E}+\nabla v_{e}^{2}\nabla\cdot\mathbf{E}.$ (13)

Assuming that the coefficients are constant, then in radiation gauge, $\mathbf{E}=-\partial_{t}\mathbf{A}$, integrating in time gives

$\partial_{t}^{2}\mathbf{A}+c^{2}\nabla\times\nabla\times\mathbf{A}+\omega_{p}^{2}\mathbf{A}-\nabla v_{e}^{2}\nabla\cdot\mathbf{A}=0.$ (14)

These equations can be derived from the Hamiltonian

$H=\frac{\epsilon_{0}}{2}\int d^{3}x\left(\left|\mathbf{E}\right|^{2}+\left|c\nabla\times\mathbf{A}\right|^{2}+\left|v_{e}\nabla\cdot\mathbf{A}\right|^{2}+\left|\omega_{p}\mathbf{A}\right|^{2}\right).$ (15)

There are three modes: two electromagnetic waves with polarization perpendicular to the wave vector that can travel near the speed of light and one electrostatic Langmuir wave with longitudinal polarization along the wave vector that can travel near the electron sound speed.

In the main text, we assume that the spatial variation in the plasma is along the direction of wave propagation. This causes the polarization to decouple from the evolution and, hence, to remain fixed in time. Thus, we can treat the polarization of each mode individually. In this case, Eq. (14) can be written as three one-dimensional Klein-Gordon equations, one for each vector component,

$\partial_{t}^{2}A_{\mu}+\omega_{p}^{2}A_{\mu}-v^{2}\partial_{x}^{2}A_{\mu}=0,$ (16)

where we assume that the direction of wave propagation is $\hat{x}$. Here, $v$ is the velocity of the wave with polarization along $\mu$, in which $v=c$ is the speed of light for electromagnetic waves when $\mu$ is perpendicular to $x$, and $v=v_{e}$ is the electron sound speed for Langmuir waves when $\mu=x$.

Canonical quantization for bosonic fields leads to the equal-time canonical commutation relations (CCR) $\epsilon_{0}[E_{\mu}^{\dagger}(\mathbf{x}),A_{\nu}(\mathbf{y})]=i\hbar\delta_{\mu\nu}\delta^{3}(\mathbf{x}-\mathbf{y}).$ In general, the solutions decompose into linear eigenfunctions labeled by

the discrete indices, $\mu$ for the polarization of the different modes, and $\mathbf{k}$, representing spatial degrees of freedom. In a uniform plasma, $\mathbf{k}$ is the wave vector, but in a nonuniform plasma, this is simply an index over all eigenstates for each mode, which is discrete for a bounded spatial domain. We define creation and destruction operators that obey the CCR $\left[a_{\mu,\mathbf{k}},a^{\dagger}_{\nu,\mathbf{q}}\right]=\delta_{\mu\nu}\delta_{\mathbf{k},\mathbf{q}}$ via

$a^{\dagger}_{\mu,\mathbf{k}}=\frac{\alpha_{\mu,\mathbf{k}}}{\sqrt{2\left|\hbar\omega_{\mathbf{k}}/\epsilon_{0}\right|}}\hskip 17.07164pta_{\mu,\mathbf{k}}=\frac{\alpha^{*}_{\mu,\mathbf{k}}}{\sqrt{2\left|\hbar\omega_{\mathbf{k}}/\epsilon_{0}\right|}}$ (10)

where $\alpha_{\mu,\mathbf{k}}$ represents the projection of each orthonormal eigenfunction $\boldsymbol{\phi}^{*}_{i,\mathbf{k}}(\mathbf{x})$ onto the fields

$\alpha_{\mu,\mathbf{k}}=\int d^{3}\mathbf{x}\ \boldsymbol{\phi}^{*}_{\mu,\mathbf{k}}(\mathbf{x})\cdot\left[\mathbf{E}^{*}(\mathbf{x})-i\left|\omega_{\mathbf{k}}\right|\mathbf{A}(\mathbf{x})\right].$ (11)

This leads to the Hamiltonian

$H=\sum_{\mu,\mathbf{k}}\hbar\left|\omega_{\mu,\mathbf{k}}\right|a^{\dagger}_{\mu,\mathbf{k}}a_{\mu,\mathbf{k}}+H_{0}.$ (12)

where $H_{0}$ is the vacuum energy of the plasma model.

In what follows, we will write a bosonic lattice model that produces the dynamics of Eq. (10) and map that bosonic model to a spin model; however, this is made nontrivial by the fact that while $a_{\mu,\mathbf{k}}$ satisfies the standard bosonic commutation relation, the vector potential $\mathbf{A}$ does not. Nevertheless, one can show that $a_{\mu}(x)\equiv\sqrt{\epsilon_{0}/(2\hbar)}\left((f*E^{*}_{\mu})(x)+i(g*A_{\mu})(x)\right)$ also satisfies Eq. (10), where $*$ is the convolution operator, and $f$ and $g$ are Fourier transforms of $1/\sqrt{\left|\omega_{k}\right|}$ and $\sqrt{\left|\omega_{k}\right|}$, respectively. Therefore, we will derive our bosonic lattice model to simulate Eq. (10) for $a_{\mu}(x)$.

### A.2 Mapping to local spin model

Due to the limited resources of present-day hardware platforms, we would like to simulate the plasma wave equation using a local spin model that is naturally represented by the qubits of a quantum computer. Hence, we limit the occupation number at each point in space to two possibilities, $0$ or $1$. The qubit Hamiltonian is first order in momentum, so it cannot directly represent the plasma Hamiltonian, which is second order in momentum, and still maintain the correct bosonic quantization conditions. Moreover, generating the second-order dispersion relation for plasma waves requires a kinetic term that must be approximated as a first-order differential operator. For real bosons, this requires at least two qubits per lattice site, e.g. a complex bosonic field.

One approach is to collect the components of the field, represented by qubits or spins, in the form of a Dirac spinor, while retaining bosonic statistics. For each mode with velocity $v$, we can write the Dirac equation for the Dirac spinor $\Psi(t,\mathbf{x})$ as

$i\gamma^{0}\partial_{t}\Psi=-iv\gamma^{1}\partial_{x}\Psi+\omega_{p}\Psi,$ (13)

where $\gamma^{j}$ are Dirac matrices satisfying $(\gamma^{0})^{2}=-(\gamma^{1})^{2}=1$ and $\{\gamma^{0},\gamma^{1}\}=0$, and $\Psi(x)$ is a spinor. One is free to use any representation of $\gamma^{\mu}$ provided that it satisfies these anticommutation relations. Multiplying both sides of Eq. (13) by $i\gamma^{\mu}\partial_{\mu}$ gives the Klein-Gordon equation in Eq. (10) for $\Psi$. Since both $\Psi(x)$ and $a_{\mu}(x)$ obey Eq. (10), a natural conclusion is that $\Psi(x)\propto a_{\mu}(x)$.

The spatially discretized form of the Dirac equation is

$i\gamma^{0}\partial_{t}\Psi(x)=-i\frac{v}{2a}\gamma^{1}\left[\Psi(x+a)-\Psi(x-a)\right]+\omega_{p}\Psi(x).$ (14)

Without loss of generality, we set $\gamma^{0}=-\tau^{z}$ and $\gamma^{1}=i\tau^{y}$. Then, the spinor $\Psi(x)$ must be defined in terms of $a_{\mu}(x)$ such that $\partial_{t}^{2}a_{\mu}$ is given by Eq. (10). Choosing $\Psi(2x)=a_{\mu}(2x)(1\ 0)^{T}$ on even sites and $\Psi(2x+a)=a_{\mu}(2x+a)(0\ 1)^{T}$ on odd sites accomplishes this. The two components of the spinor equation [Eq. (14)] can be written as two separate equations,

$\left[i\partial_{t}+\omega_{p}\right]a_{\mu}(2x)=$ $-i\frac{v}{2a}\left[a_{\mu}(2x+a)-a_{\mu}(2x-a)\right]$ (15)
$\left[i\partial_{t}-\omega_{p}\right]a_{\mu}(2x+a)=$ $-i\frac{v}{2a}\left[a_{\mu}(2x+2a)-a_{\mu}(2x)\right]$ (16)

which combine to give

$\partial_{t}^{2}a_{\mu}+\omega_{p}^{2}a_{\mu}=\frac{v^{2}}{4a^{2}}\left[a_{\mu}(x+2a)+a_{\mu}(x-2a)-2a_{\mu}(x)\right].$ (17)

Now, to limit the occupation number to $0$ or $1$ (to limit computational costs), we modify the Hamiltonian in Eq. (11) to a model for hardcore bosons that has the same dispersion relation and then map the hardcore boson model to a spin model. The Hamiltonian that achieves this for hardcore bosons is

$H=i\frac{J}{2}\sum_{j=1}^{N-1}(b^{\dagger}_{j}b^{\phantom{\dagger}}_{j+1}-b^{\dagger}_{j+1}b^{\phantom{\dagger}}_{j})-\hbar\omega_{p}\sum_{j=1}^{N}(-1)^{j}b^{\dagger}_{j}b^{\phantom{\dagger}}_{j}.$ (18)

where $b_{j}\equiv a_{\mu}(ja)$ and $J\equiv-2\hbar v/a$. We then map the hardcore bosons to spins via $b^{\dagger}_{j}\equiv\sigma_{j}^{+}$ and $b_{j}\equiv\sigma_{j}^{-}$, which gives the spin Hamiltonian of Eq. (3). We note that the Heisenberg equations of motion for this model are not exactly those in Eq. (17), but are modified to the version appropriate for hardcore bosons, which eliminates higher occupation numbers.

We believe that there is a path forward for extending this model to higher dimensions. In this case, one can use any representation of the Dirac gamma matrices with a complex Dirac spinor field. In dimensions $0-4\mod 8$, the Majorana representation may be preferable because then the spinor field can be taken to be real. The sum over neighboring sites approximates the relevant Laplacian operators. For example, for the case we study here, $-\nabla\times\nabla=\nabla^{2}-\nabla\nabla\cdot$ is the perpendicular Laplacian for transverse EM waves and $\nabla\nabla\cdot$ is the parallel Laplacian for longitudinal Langmuir waves.

## Appendix B Exact solution for the spin Hamiltonian

In Sec. II, we considered the exactly solvable spin Hamiltonian

$H=-\frac{J}{4}\sum_{i=1}^{N-1}(\sigma_{i}^{x}\sigma_{i+1}^{y}-\sigma_{i}^{y}\sigma_{i+1}^{z})+\frac{1}{2}\sum_{i=1}^{N}\Delta_{i}(-1)^{i}\sigma_{i}^{z}.$ (10)

We will assume uniform $\Delta_{i}=\Delta$. The standard way to solve this Hamiltonian is by mapping it to a spinless hardcore bosonic Hamiltonian *[48]*,

$\sigma_{i}^{x}$ $=a_{i}^{\dagger}+a_{i}$
$\sigma_{i}^{y}$ $=i(a_{i}^{\dagger}-a_{i}^{\phantom{\dagger}})$
$\sigma_{i}^{z}$ $=a_{i}^{\phantom{\dagger}}a_{i}^{\dagger}-a_{i}^{\dagger}a_{i}^{\phantom{\dagger}},$ (11)

where $a_{i}^{\phantom{\dagger}}(a_{i}^{\dagger})$ annihilates (creates) a hardcore boson at site $i$. On each site, the ground qubit state $|0\rangle$ maps to the vacuum of bosons, and the excited qubit state $|1\rangle$ maps to the singly occupied state. Under the hardcore-boson transformation, the spin Hamiltonian [Eq. (10)] maps to

$H=i\frac{J}{2}\sum_{i=1}^{N-1}(a_{i}^{\dagger}a_{i+1}^{\phantom{\dagger}}-a_{i+1}^{\dagger}a_{i}^{\phantom{\dagger}})-\Delta\sum_{i=1}^{N}(-1)^{i}a_{i}^{\dagger}a_{i}^{\phantom{\dagger}}.$ (12)

After performing a Fourier transform on the lattice index, the hardcore boson Hamiltonian can be written as

$H=\sum_{k}-J\sin(ka)\tilde{a}_{k}^{\dagger}\tilde{a}_{k}^{\phantom{\dagger}}-\Delta\sum_{k}\tilde{a}_{k}^{\dagger}\tilde{a}_{k+\pi/a}.$ (13)

where $\tilde{a}_{k}^{\phantom{\dagger}}(\tilde{a}_{k}^{\dagger})$ annihilates (creates) a hardcore boson with wave vector $k$. Two single-particle energy bands emerge, with energies

$\hbar\omega_{k}=\pm\sqrt{J^{2}\sin^{2}ka+\Delta^{2}}.$ (14)

For periodic boundary conditions, the single-particle eigenstates are $|\psi_{k}\rangle=b_{k}^{\dagger}\,|\text{vac}\rangle$, with

$b_{k}=\frac{(J\sin ka-\hbar\omega_{k})\,\tilde{a}_{k}+\Delta\tilde{a}_{k+\pi/a}}{\sqrt{2\hbar\omega_{k}\,(\hbar\omega_{k}-J\sin ka)}}.$ (15)

For open boundary conditions, the single-particle eigenstates are created by $b_{k}^{\dagger}=\sum_{j}c_{jk}^{\phantom{\dagger}}a_{j}^{\dagger}\,|\text{vac}\rangle$ where

$c_{jk}$ $=\sqrt{\frac{2}{N+1}}\sqrt{\frac{\hbar\omega_{k}+\Delta}{\hbar\omega_{k}}}\cos jka,$ $\text{if }j\text{ is odd},$
$c_{jk}$ $=-i\,\,\text{sgn}(\omega_{k})\sqrt{\frac{2}{N+1}}\sqrt{\frac{\hbar\omega_{k}-\Delta}{\hbar\omega_{k}}}\sin jka,$ $\text{if }j\text{ is even}.$ (16)

Due to the hard-wall boundary condition, $ka=p\frac{\pi}{N+1}$ if $N$ is odd, and $ka=\left(p+\frac{1}{2}\right)\frac{\pi}{N+1}$ if $N$ is even, with $p$ an integer running from $0$ to $N/2$. The allowed values for $ka$ cover only half of the usual range because it takes two coupled lattice sites to generate the two branches of the dispersion relation. The physical interpretation is that the overall system size is really only $N/2$ because there are actually two coupled degrees of freedom per lattice site; i.e. the effective lattice spacing is actually $2a$. For example, for an EM plasma wave, the degrees of freedom at the two sites represent linear combinations of the charge density and electric current.

## Appendix C Quantum Advantage

There has been recent development of so-called “qubit lattice algorithms” for simulating the classical physics of electromagnetic wave propagation in plasmas. Certain algorithms target classical computers *[13, 14]*, while others target quantum computers *[49, 50]*. While these works have obtained real speedups on classical supercomputers, and claim to offer efficient representations for quantum computers, they target the classical linearized equations of motion, where it has been much more difficult to find significant quantum advantage. While the recent work of Ref. *[26]* has determined classes of computationally intractable questions about the dynamics of linear oscillators that can be solved efficiently with quantum algorithms, it is more natural to investigate the potential speedup for intrinsically quantum dynamics.

Quantum computers are naturally adapted to simulating quantum systems and, thus, can offer significant quantum advantage for quantum dynamics. We approximated the quantum field theory of linear electromagnetic wave propagation in the low photon (or plasmon) density limit by a hard-core boson model that maps to a noninteracting fermionic model, which is an integrable model. While this makes classical simulations easy in certain cases, simulating the general case is still hard, as we argue in detail below.

The best classical algorithm for simulating a noninteracting quantum model depends on the initial condition. Multiparticle states can be expressed using products of single-particle states, so simulating the evolution classically is tractable when the initial state can be written as a polynomial number of products of single-particle states. Let $M_{0}$ represent the number of single-particle states required to represent the initial condition. Then, one needs to simulate the evolution of each single-particle eigenstate in the multiparticle wave function efficiently. For small enough problem sizes, one can use exact diagonalization, which generally scales as $\mathcal{O}(M_{0}N_{s}^{3})$. If the overall number of single-particle states appearing in the initial condition is small or if the single-particle Hamiltonian matrix of size $N_{s}\times N_{s}$ is too large to form explicitly, then other methods, such as computing the matrix exponential for each state using only matrix-vector products would be cheaper *[51, 52, 53]*. Forming the matrix exponential for large problem sizes is often based on iterative Krylov-subspace methods *[52]* which have a

cost scaling as $\mathcal{O}(KN_{s}^{2})$ where $K\leq N_{s}$ is the size of the largest Krylov subspace. However, if the number of single-particle states appearing in the initial condition, $M_{0}$, is large, then this cost becomes $\mathcal{O}(M_{0}N_{s}^{3})$. Thus, we conclude that the cost of the classical algorithm ranges from

$C\sim\mathcal{O}(M_{0}KN_{s}^{2})\text{ to }\mathcal{O}(M_{0}N_{s}^{3})$ (10)

depending on the initial conditions. Computing the evolution of single-particle observables also has a similar cost, where $M_{0}$ now represents the number of independent observables.

In 1D, the Hamiltonian of interest is banded, and, in fact, is tridiagonal in our example. In this case, the worst scaling is not $\mathcal{O}(N_{s}^{3})$ but rather $\mathcal{O}(KN_{s}^{2})$ where, in this context, $K$ represents the bandwidth or some measure of sparsity for a more general sparse matrix.

The initial state in the experiment that we implemented in Sec. V had one particle localized on two lattice sites. We only reported the density observable, which is a single-particle observable. The experiment was implemented in one dimension. Therefore, the time it takes to classically simulate one instance of that experiment is $C=\mathcal{O}(N_{s}^{2})$. This is, for example, why we were able to numerically simulate a large system with $N_{s}=100$, as shown in Fig. 5.

Even for these simple cases, using a quantum computer to simulate the quantum problem, as explored in this work, provides a complexity advantage. The quantum algorithm only requires simulating the problem in time and does not require performing an eigendecomposition. The cost of Trotterizing the evolution operator scales as the number of interactions between sites, $n_{i}$, and, because the Hamiltonian only has local interactions, this scales as the spatial dimension $n_{i}\propto d$. Thus, evolving this quantum algorithm, which scales as $n_{i}$ per time step, for the same total time interval only has a cost of $\mathcal{O}(n_{i}N_{t})$. Due to the Courant-Friedrichs-Lewy (CFL) criterion for stability and accuracy, the number of time steps scales linearly in grid spacing $N_{t}\propto dN_{x}$. Therefore, the total time for the quantum algorithm scales as

$Q=\mathcal{O}(d^{2}N_{x}).$ (11)

Thus, the quantum speedup ranges from

$C/Q=\mathcal{O}(M_{0}KN_{x}^{2d-1}/d^{2})\text{--}\mathcal{O}(M_{0}N_{x}^{3d-1}/d^{2}).$ (12)

Nonetheless, in the most general case, the initial state may be any state that is easy to prepare with a quantum circuit but hard to express in terms of the noninteracting model, and the observable of interest may be any easily measurable observable in the quantum circuit but highly nonlocal or multisite in the particles. As a concrete example, a state that is easy to prepare with a quantum circuit but hard to express in terms of the noninteracting model, is the equal superposition of all computational basis states, which can be prepared by applying a Hadamard gate to all the qubits. In such cases, the only way to classically compute the observables of interest is via exact diagonalization of the full Hilbert space whose size is $2^{N_{s}}$, which takes exponential time. In these cases, executing the quantum circuit on a quantum computer has an exponential advantage over classically simulating the circuit. As such, exactly classically simulating a circuit with $>50$ qubits is typically out of reach of current classical computers. We note that the best classical methods to simulate quantum systems in 1D are time-dependent matrix-product-state (MPS) methods, but even these would struggle due to the need for a rapidly increasing bond dimension with time. Moreover, MPSs scale poorly in $>1D$, and higher-dimensional tensor-network methods are difficult to compute efficiently.

Finally, although the quantum algorithm still has a scaling advantage for the simple cases we considered in this paper, it does not necessarily translate to a practical advantage when real time scales are compared at current qubit scales. Indeed, for the model we studied, quantum advantage is not expected even for $N_{s}=\mathcal{O}(10^{5})$ qubits in one dimension. If we consider a more general Hamiltonian, e.g. with additional ZZ interactions, then it does not map to a noninteracting model, and this model cannot be classically simulated in $\text{poly}\left(N_{s}\right)$ time. Solving this system classically generally takes exponential time in $N_{s}$, where $C\sim\mathcal{O}(2^{4N_{s}})$.

## Appendix D Initial conditions

For classical plasma simulation, one would need to specify two initial conditions, $\mathbf{A}(\mathbf{x})$ and $\mathbf{E}(\mathbf{x})$, at the initial time, as well as boundary conditions for the given time interval. However, because the fields are canonically conjugate, for the quantized problem, the fields do not commute and must instead obey the Heisenberg uncertainty relations. Thus, one cannot specify both or measure both fields at the same time and must set the conditions for the wave function or density matrix in a manner consistent with the laws of quantum mechanics. This can be performed by setting boundary conditions for the spin Hamiltonian and setting an initial spin-wave function or density matrix. For example, one could specify a definite initial spin orientation or one could specify the initial conditions in terms of spin coherent states, which balance the uncertainty in the fields in an optimal manner. To compare this to a classical simulation, observables such as the energy of any given configuration can be determined by their expectation value.

## Appendix E Hardware characteristics

The Rigetti Ankaa-3 quantum processing unit (QPU) is composed of transmon qubits connected via floating tunable couplers (which are also transmons) *Rigetti and Ankaa (2005)* arranged in a square grid. The tunable couplers en

able control of the effective qubit-qubit coupling between neighboring qubits via the modulation of their frequency, which is controlled by threading external magnetic flux through the superconducting quantum interference device (SQUID) loops. The coupling is turned on by applying a baseband flux pulse to the coupler when actuating a two-qubit gate. The use of tunable couplers allows the interaction strength between the two qubits to be tuned, minimizing unwanted interactions with spectator qubits during the operation of gates while enhancing the interaction strength between the partners of the two-qubit gate.

Entangling gates between neighboring qubits are implemented using a bipolar baseband flux pulse applied to the higher-frequency qubit, bringing it into resonance with its neighbor *[54]*. This induces an XX+YY interaction, typically accompanied by a small ZZ component. In the frame rotating at the qubit frequencies, this interaction realizes the so-called FSIM gate *[32]* up to one-qubit phases. We define the FSIM gate as

$\text{FSIM}(\theta,\phi)=$
\[ \left(\begin{array}[]{cccc}1&0&0&0\\
0&\cos\frac{\theta}{2}&i\sin\frac{\theta}{2}&0\\
0&i\sin\frac{\theta}{2}&\cos\frac{\theta}{2}&0\\
0&0&0&e^{i\phi}\end{array}\right). \] (10)

We note that this definition has the opposite signs for $\theta$ and $\phi$ as compared to some other conventions, e.g., that given in Ref. *[55]*. The native gate realized on the device is a “PHASEDFSIM” gate,

$\text{PHASEDFSIM}(\theta,\phi,\zeta,\gamma,\chi)=$
\[ \left(\begin{array}[]{cccc}1&0&0&0\\
0&e^{-i(\gamma+\zeta)}\cos\frac{\theta}{2}&ie^{-i(\gamma-\chi)}\sin\frac{\theta}{2}&0\\
0&ie^{-i(\gamma+\chi)}\sin\frac{\theta}{2}&e^{-i(\gamma-\zeta)}\cos\frac{\theta}{2}&0\\
0&0&0&e^{i(\phi-2\gamma)}\end{array}\right), \] (11)

which is equal to $Rz\left(\frac{\zeta-\chi}{2}-\gamma\right)\otimes Rz\left(\frac{\chi-\zeta}{2}-\gamma\right)\cdot\text{FSIM}(\theta,\phi)\cdot Rz\left(\frac{\zeta+\chi}{2}\right)\otimes Rz\left(-\frac{\zeta+\chi}{2}\right)$. Here, $\theta$ and $\phi$ are iSWAP-like and ZZ-like entangling phases, respectively, and the corresponding iSWAP-like and ZZ-like interactions are also referred to as transverse and longitudinal interactions. For example, $\text{FSIM}(\theta=\pi,\phi=0)$ is the iSWAP gate, and $\text{FSIM}(\theta=0,\phi=\pi)$ is the CZ gate. Here, $(\chi,\zeta,\gamma)$ are single-qubit phases that arise from the qubits’ different idle frequencies and their frequency excursions during the two-qubit gate. The $\chi$ phase depends on the time at which the gate is played in the circuit, and it is given by $\chi=\Delta\nu t$, where $\Delta\nu$ is the difference in the qubits’ idle frequencies and $t$ is the time difference between the start of the circuit and the time at which this gate is executed. We do not assume this form for $\chi$ and instead learn $\chi$ along with the other angles using the gate-learning technique described in Sec. F.1.

We tuned the gates such that $\theta\simeq\pi/2$ and $\phi\simeq 0$ on each edge, and more accurately inferred $(\theta,\phi,\zeta,\chi,\gamma)$ for each edge using tomography and a variant of cross-entropy benchmarking (see Sec. F.1).

The characteristics of the Ankaa-3 device at the time of the experiment are shown in Table 1. The T1 and T2 values are reported on Rigetti Quantum Cloud Services while the gate fidelities were measured directly prior to the experiment. For the experimental configuration of circuits executed in this report, the EPLG was measured to be $1.48\pm 0.02\%$.

## Appendix F Executing the experiment

A schematic overview of the experimental pipeline is depicted in Fig. 6. Before executing the logical target experiments, we perform a few steps. First, we learn the native entangling gates between the qubits, as explained in Sec. F.1. Then, we compile the target logical circuit into logical gates between the qubits (Sec. F.2) and express the logical gates in terms of native gates (Sec. F.4). Once we have all the circuit instances ready for execution, we construct a list of Clifford circuits with the same brickwork gate structure as the target circuits. The Clifford circuits are useful for error mitigation, as explained in Sec. G.2. We transform all the logical circuit instances and Clifford circuit instances using twirling (Sec. G.1). Finally, we execute all our circuits and take shots, implement error mitigation (Appendix G), and calculate the error-mitigated observables.

### F.1 Gate-learning via tomography and benchmarking

An important step in executing our experiments is learning the gates between the qubits. We learn the native gate between the qubits in a two-step process: first, a rough estimation of the gate using process tomography, and then a finer estimation of the gate using a variant of randomized benchmarking.

Process tomography is a technique that is commonly used to learn a quantum process. The basic recipe is to prepare a complete set of basis states, apply the process, and measure a complete set of measurement bases *[56]*. To learn the gate from the process-tomography results, the superoperator is first reconstructed from the observable data. This includes the effects of noise, but the condition that the superoperator be physical, i.e. completely positive and trace preserving, is enforced. We numerically maximize the fidelity of this superoperator against a parameterized PHASEDFSIM candidate. This gives us a rough calibration of the native entangling gate. The process tomography was performed on an entire entangling gate cycle at once, allowing the unitaries to be learned in context.

To obtain a finer calibration of the entangling gate, we use a method inspired by cross-entropy benchmarking *[57]*. A circuit is constructed with many layers of the

![images/image11.jpg](images/image11.jpg)
Figure 6: A schematic overview of the experiment. A two-step gate-learning protocol learns the native two-qubit gates on the hardware. These native gates are used in the compilation of the target logical circuit. Meanwhile, a set of Clifford circuits with the same gate placements as the target circuits are also executed. The results of the target circuits are subject to error mitigation using Clifford data regression.

Two-qubit gate cycle interleaved with random one-qubit gates, as shown in Fig 7(a). Then, we calculate the fidelity of the output state with the predicted state in a noiseless numerical simulation for the circuit. Similar to above, we numerically maximize the fidelity of the experimentally produced state against the state produced by a parameterized PHASEDFSIM candidate. The $\chi$ phase of each gate in the circuit is parameterized with two parameters ($\chi_{0},\Delta\nu$) as $\chi=\chi_{0}+\Delta\nu t$, where $t$ is the time at which the gate appears in the circuit. The other four parameters are constant for all the PHASEDFSIM gates on the same edge in the circuit. Learning the gate parameters from a circuit with many layers of the two-qubit gate provides robustness to state preparation and measurement (SPAM) errors.

Because the benchmark circuit consists of separable states with independently entangled pairs of qubits, this technique is scalable. We note that unlike cross-entropy benchmarking, where the fidelity is estimated using the cross-entropy of the observed bitstrings with the expected bitstring distribution, we instead estimate the state fidelity using direct fidelity estimation, applying the approach of classical shadows *Klein et al. (2015)*. The main reason to use classical shadows is to mitigate nonMarkovian errors in the measurements of the qubits.

Since the benchmarking sequence consists of a long sequence of gates, the results will be affected by noise. Typically, the purity $\mathcal{P}$ and the fidelity $\mathcal{F}$ (with the ideal state computed from the right gate parameters) of the states decay from $1$ to $1/4$, as the sequence length approaches infinity. It is therefore useful to compute the shifted fidelity and the shifted purity,

$\tilde{\mathcal{P}}$ $=\frac{4\mathcal{P}-1}{3}$
$\tilde{\mathcal{F}}$ $=\frac{4\mathcal{F}-1}{3}$ (10)

at each sequence length. For a depolarizing noise model, the shifted purity and fidelity would decay as $f^{2n}$ and $f^{n}$, respectively. Here, $f$ is known as the unitarity of the noise channel. Any difference between the shifted fidelity and the square root of the shifted purity is indicative of coherent errors.

To perform direct fidelity estimation at each sequence length, we use the classical shadow to estimate all ($4^{2}-1$) Pauli observables for each two-qubit subsystem. The fidelity is given by

$\mathcal{F}=\frac{1}{2^{N}}\sum_{P}\left\langle P\right\rangle_{\text{experiment}}\left\langle P\right\rangle_{\text{ideal}}$ (11)

where $N=2$ qubits here. The uncertainty of each Pauli observable estimate is determined by its Pauli weight $w$ and the sample count $M$, and it is given by $\sim\sqrt{3^{w}/M}$. A typical state produced by the benchmarking circuit is Haar-random; therefore, $\left\langle P\right\rangle_{\text{ideal}}\sim 1/\sqrt{2^{N}}$. Putting these together, the uncertainty $\Delta F$ of the fidelity is given by

\[ (\Delta\mathcal{F})^{2}=\frac{1}{4^{N}}\sum_{w=0}^{N}\begin{pmatrix}N\\
w\end{pmatrix}3^{w}\frac{3^{w}}{M}\frac{1}{2^{N}} \] (12)

since there are \[ \begin{pmatrix}N\\
w\end{pmatrix}3^{w} \] Paulis with weight $w$. Using the binomial theorem, we simplify the above equation to $(\Delta\mathcal{F})^{2}=\frac{1}{M}\left(\frac{5}{4}\right)^{N}$. For $N=2$ and $M=3,000$ shots, we can thus estimate the fidelity of the final two-qubit state to an uncertainty of $\Delta\mathcal{F}=2.3\%$. For comparison, the cross-entropy uncertainty, $\Delta\bar{H}_{lin}=1/\sqrt{M}$. The cross-entropy metric does not depend on the system size, which is a useful property for scalable benchmarking, but in the regime of $N=2$, the direct fidelity method requires only $25\%$ more samples and has the advantage of being bounded between $0$ and $1$.

Our experiment also allows measurement of the purity of the state,

$\mathcal{P}=\frac{1}{2^{N}}\sum_{P}\left\langle P\right\rangle_{\text{experiment}}^{2},$ (13)

![images/image12.jpg](images/image12.jpg)

![images/image13.jpg](images/image13.jpg)
FIG. 7. Schematic of a shadow benchmarking experiment. (a) A circuit with a long sequence of PHASEDF-SIM gates interleaved with random one-qubit gates is implemented. (b) The decay of the shifted fidelity (green) of the experimentally prepared state with a noiseless numerical simulation, and the square root of the shifted purity (red) of the state in the experiment, versus sequence length. A classical routine learns the gate parameters such that the fidelity is maximized. The gap between the shifted fidelity and purity curves can indicate a unitary error, which is very small here. The benchmarking is performed in context.

which we will use later to track coherent errors in the gate. A similar calculation to above, and using  $\Delta (\langle P\rangle)^2 = 4|\langle P\rangle |^2 (\Delta P)^2$ , yields

$$
\left(\Delta \mathcal {P}\right) ^ {2} = \frac {1}{4 ^ {N}} \sum_ {w = 0} ^ {N} \binom {N} {w} 3 ^ {w} 4 \times \frac {3 ^ {w}}{M} \frac {1}{2 ^ {N}} = \frac {4}{M} \left(\frac {5}{4}\right) ^ {N}. \tag {F5}
$$

For  $N = 2$  and  $M = 3,000$  shots, we can thus estimate the purity of the final two-qubit state to an uncertainty of  $\Delta \mathcal{P} = 4\%$ .

In our benchmarking experiments, we typically average these estimates over 30 random sequences. An example decay with a fit is shown in Fig. 7(b). The use of effective readout-error mitigation [58] with reasonable assumptions about the noise allows us to assert that the estimate begins at 1 and decays to 0, and we use the simplified fit form of  $f(n) = f^n$ , allowing for fewer points and higher confidence in the fidelity estimate.

# 2. Trotterized evolution

We implement evolution with  $H$  using a first-order Trotter expansion. Each Trotter step can be conceptually split into three logical layers - one layer implements evolution with  $\sigma_{i}^{z}\sigma_{i + 1}^{y} - \sigma_{i}^{y}\sigma_{i + 1}^{x}$  for odd  $i$ , one layer implements the above for even  $i$ , and one layer implements

![images/image14.jpg](images/image14.jpg)

![images/image15.jpg](images/image15.jpg)
FIG. 8. Circuit for Trotterized evolution. (a) A single Trotter evolution step with the Hamiltonian  $H$  [Eq. (3)]. Here,  $U_{xy}$  and  $U_z$  are the evolutions with the spin-spin interaction and plasma-gap terms in the Hamiltonian, respectively. b) Approximate decomposition of one Trotter evolution step into native two-qubit gates and one-qubit gates implemented via the PMW4 scheme.

evolution with the  $\sigma_{i}^{z}$  terms in  $H$ . We denote the evolution due to the interaction terms as  $U_{xy}$  in Fig. 8(a), and evolution due to the  $\sigma^z$  terms as  $U_{z}$ . These logical layers are then compiled to native gates using an approximate numerical compilation explained in Sec. F4. In practice, the  $U_{z}$  gates are not implemented separately from the two-qubit layers; They are absorbed into the compilation of the two-qubit layers. Each compiled Trotter step has the structure shown in Fig. 8(b).

# 3. PMW4 decomposition

Rigetti Ankaa-class devices can realize  $\mathrm{RX}(\pi)$  and  $\mathrm{RX}(\pm \pi /2)$  gates, where  $\mathrm{RX}(\theta) = \exp (-i\frac{\theta}{2}\sigma^x)$ . Ankaa-class devices also realize a parametric  $\mathrm{RZ}(\theta) = \exp (-i\frac{\theta}{2}\sigma^z)$  gate using local updates of in-sequence phases that consume zero runtime and introduce negligible error [59]. Any one-qubit operation can be implemented using at most three  $\mathrm{RZ}(.)$  and two  $\mathrm{RX}(\pi /2)$  operations, where each  $\mathrm{RX}(\pi /2)$  is implemented via a pulse, and each  $\mathrm{RZ}(.)$  determines the in-sequence phase update [60]. The above implementation of the RZ gate is called a virtual RZ, and it is useful when two-qubit gates  $U$  in circuits are phase-carrier gates, i.e. they satisfy

$$
U \cdot \left(\mathrm {R Z} \left(\theta_ {1}\right) \otimes \mathrm {R Z} \left(\theta_ {2}\right)\right) = \left(\mathrm {R Z} \left(\theta_ {3}\right) \otimes \mathrm {R Z} \left(\theta_ {4}\right)\right) \cdot U \tag {F6}
$$

for some  $\theta_{i = 1\dots 4}$ . In this case, the RZ gate is just carried over the two-qubit phase-carrying gate, and the in-sequence phases are adjusted accordingly.

The PHASEDFSIM gate, however, is not phase carrying; therefore, the virtual RZ scheme is not feasible with PHASEDFSIM gates. Instead of the above scheme,

we thus implement one-qubit gates using four microwave pulses, known as the PMW4 scheme *[43]*,

$U_{1Q}=X_{\pi/2}(\theta)X_{\pi/2}(\phi)X_{\pi/2}(\phi)X_{\pi/2}(\omega)$ (100)

where $X_{\alpha}(\phi)\equiv\text{RZ}(-\phi)\text{RX}(\alpha)\text{RZ}(\phi)$. In this scheme, the net phase advanced by a one-qubit gate is $0$; therefore, no phase needs to be carried forward by the PHASEDFSIM gate.

### F.4 Approximate numerical compilation of gates

Once we have calibrated arbitrary one-qubit and native two-qubit gates, we are ready to implement arbitrary logical circuits. Implementing an arbitrary logical two-qubit gate $U_{\text{tgt}}$ requires us to express it in terms of the native gates. We do this using an approximate numerical compilation technique.

Our compilation scheme finds the best compilation that uses at most two PHASEDFSIM gates to express the target logical gate; i.e., we express

$U_{\text{approx}}=$ $(u_{1}\otimes u_{2})\cdot\text{PHASEDFSIM}\cdot(u_{3}\otimes u_{4})\cdot$
$\text{PHASEDFSIM}(u_{5}\otimes u_{6})$ (101)

where PHASEDFSIM refers to the learned native PHASEDFSIM$(\theta,\phi,\zeta,\chi,\gamma)$ on the edge. Here, $u_{i}$ are parameterized single-qubit gates, where we numerically find the parameters such that $|\text{tr}(U_{\text{approx}}^{\dagger}U_{\text{tgt}})|$ is maximized, i.e. $U_{\text{approx}}$ is as close to $U_{\text{tgt}}$ as possible. We point out that Eq. (100) only has two PHASEDFSIM gates, meaning that it is not possible to exactly express the full range of two-qubit gates. Nonetheless, it can be useful to approximately express gates *[61]*, provided that the error in doing so is smaller than the error incurred by an additional native entangling gate. Thus, the approximate-expression technique has broad applicability. For the target logical gates in our experiment, the maximum compilation error $1-|\text{tr}(U_{\text{approx}}^{\dagger}U_{\text{tgt}})|^{2}/16$ had a relatively small value of $0.12\%$.

The numerical compilation scheme involves a classical optimization of the $u_{i}$, which is expensive and can become a bottleneck. Therefore, to enable fast compilation, we developed a vector database of expression instances. An expression instance is composed of both the target unitary $U_{\text{tgt}}$, and the native entangling unitaries PHASEDFSIM$(\theta,\phi,\zeta,\chi,\gamma)$ which attempt to express it when combined with the arbitrary single-qubit rotations. The goal is to retrieve the value of the expression instance from the database; however, instances rarely match to floating-point precision and unitaries are equivalent up to a global phase, making traditional databases unsuitable for storing instances. Therefore, we turn to a vector database, where the keys are vectors and the retrieval is based on a distance metric *[62]*.

We formulate the instance matching as the minimization of a distance metric. The process fidelity of two unitary matrices $U$ and $U^{\prime}$, $F(U,U^{\prime})=\left(\frac{\text{Tr}[U^{\dagger}U^{\prime}]}{4}\right)^{2}=\frac{\text{Tr}[\mathcal{U}^{\dagger}\mathcal{U}^{\prime}]}{16}$, where $\mathcal{U}$ and $\mathcal{U}^{\prime}$ are the Pauli-Liouville matrices for $U$ and $U^{\prime}$, is maximized when $U=U^{\prime}$ up to a global phase. We note that the trace can be written as $\text{Tr}[\mathcal{U}^{\dagger}\mathcal{U}^{\prime}]=\vec{\mathcal{U}}\cdot\vec{\mathcal{U}}^{\prime}$, where $\vec{\mathcal{U}}$ and $\vec{\mathcal{U}}^{\prime}$ are vectors obtained by raveling the superoperators. Thus, we can define a *distance* between two unitaries as $d(U,U^{\prime})=|\vec{\mathcal{U}}\cdot\vec{\mathcal{U}}^{\prime}-1|$. To find the closest match of an expression instance ($U_{\text{tgt}},\text{PHASEDFSIM},\text{PHASEDFSIM}$) with instances ($u,\text{PHASEDFSIM},\text{PHASEDFSIM}$) in the database, we minimize the sum of the distances, $d(U_{\text{tgt}},u)+2d(\text{PHASEDFSIM},\text{PHASEDFSIM})$.

## Appendix G Error mitigation

Quantum hardware suffers from several kinds of noise that introduce errors into their output. Error mitigation has been shown to be a useful technique for evaluating observables of interest in noisy quantum computers *[63]*. Generally, one first tailors the hardware noise to a form that is amenable to mitigation. Then, an appropriate error-mitigation technique is implemented. Various error-mitigation techniques have been proposed in the literature *[35]*. Here we describe our noise-tailoring and error-mitigation methods.

### G.1 Twirling

Noise in gates is often described via quantum channels. For example, the action of a noisy gate on a state $\rho$ is written as $\mathcal{U}_{\text{noisy}}(\rho)=\Phi\mathcal{U}\rho$, where $\mathcal{U}$ is the noiseless gate $U$ in superoperator space, and $\Phi$ is the noise channel. We use calligraphic symbols for operators in superoperator space, normal font for operators in Hilbert space, and capital Greek letters for noise channels.

Typically, the noise channel $\Phi$ on the device is not known. It is generally desirable to minimize the coherent noise, since coherent errors can interfere constructively.

The most common method for transforming coherent noise into incoherent noise is Pauli twirling. Pauli twirling converts arbitrary noise to stochastic Pauli noise, which is more amenable to error mitigation than other noise. In the above case for measuring unitarity, for example, the experiment would have layers of the target two-qubit gate interleaved with random Pauli gates, which would convert the coherent noise to incoherent noise.

#### G.1.1 Background: Pauli twirling

Twirling maps a circuit to a logically equivalent circuit if the gates were noiseless. Pauli twirling does this by injecting random Pauli gates and their conjugates into the circuit. Any Clifford gate $U$ may be equivalently written

![images/image16.jpg](images/image16.jpg)
FIG. 9. Two continuous families of the FSIM twirling group, which are functions of the continuous parameter  $\gamma$ . The symbol  $X_{\pi}(\gamma)$  means  $R_z(-\gamma)R_z(\pi)R_z(\gamma)$  (defined earlier in Sec. F3 and repeated here for convenience). Three distinct Pauli twirls can be obtained as special cases of these continuous families. For example, the ZZ twirl is obtained by setting  $\gamma = 0$  in the first line, the XX twirl by setting  $\gamma = 0$  in the second line, and the YY twirl by setting  $\gamma = \pi /2$  in the first line. Due to the absence of the remaining 12 Pauli twirls, errors are only partially twirled by this group.

as  $PUP'$ , where  $P$  and  $P'$  are conjugated Pauli products. An ensemble of circuits is executed, where the circuits twirl  $U$  with different Paulis  $P$ , and the ensemble average for a target observable is computed. Averaging over all Pauli twirls, the noisy gate  $\mathcal{U}_{\mathrm{noisy}}$  gets transformed to

$$
\mathcal {U} _ {\text {n o i s y}} \rightarrow \frac {1}{1 6} \sum_ {\mathcal {P}} \mathcal {P} \mathcal {U} _ {\text {n o i s y}} \mathcal {P} ^ {\prime} \equiv \frac {1}{1 6} \sum_ {\mathcal {P}} \mathcal {P} \Phi \mathcal {U} \mathcal {P} ^ {\prime} \tag {G1}
$$

where the factor 16 in the denominator refers to the number of two-qubit Pauli products. Since  $\mathcal{P}\mathcal{U}\mathcal{P}' = \mathcal{U}$ , Eq. (G1) becomes  $\mathcal{U}_{\mathrm{noisy}} \to \Phi_{\mathrm{twirl}}\mathcal{U}$  with  $\Phi_{\mathrm{twirl}} = \frac{1}{16} \sum_{\mathcal{P}} \mathcal{P}\Phi \mathcal{P}^{\dagger}$ . It can be shown the twirled noise  $\Phi_{\mathrm{twirl}}$  is stochastic Pauli noise.

# b. Twirling the FSIM gate

The native entangling gate used in our experiment, the FSIM gate, is not a Clifford gate, and therefore it cannot be twirled by Pauli gates; however, we show below that it is possible to partially twirl the FSIM gate using only one-qubit gates. Since PHASEDFSIM  $(\theta ,\phi ,\zeta ,\chi ,\gamma)$  is equal to  $\mathrm{FSIM}(\theta ,\phi)$  sandwiched between additional  $Rz$  gates, it is then straightforward to implement the one-qubit gates that twirl PHASEDFSIM  $(\theta ,\phi ,\zeta ,\chi ,\gamma)$  which are the actual native gates on our device.

For the  $\mathrm{FSIM}(\theta, \phi)$  gate, a subset of the Pauli twirls can be efficiently inverted, specifically the II, XX, YY, and ZZ twirls. Furthermore, we found two continuous families of pairs of one-qubit gates that twirl the  $\mathrm{FSIM}(\theta, \phi)$  gate, as shown in Fig. 9. We note that setting  $\gamma = \pi$  in the first family of twirls (first line in Fig. 9) gives the twirl with ZZ, setting  $\gamma = 0$  in the second family of twirls (second line in Fig. 9) gives the twirls with XX, and  $\gamma = \pi / 2$  in the second line gives the twirls with YY. Furthermore, the entire family of twirls in the second line at arbitrary  $\gamma$  can be obtained by composing the first line and twirling with XX.

Unlike the Paulis, these twirls are unable to fully convert an arbitrary coherent error channel into a stochastic one, for some sensible choice of distribution over  $\gamma$ . Rather, certain coherent errors are twirled, while others are not. Notably, over-rotation-type errors of the FSIM gate, where the angles  $\theta$  or  $\phi$  are different from their calibrated value, are not twirled (see also Ref. [64]). We aim to learn the gate such that over-rotation errors, which could occur due to errors in calibration, are negligible. We assume the angles to be stable over the course of the experiment, however over a long enough time scale, drift can result in the calibration becoming stale. Nevertheless, the process learns only unitaries of the FSIM type, and it thus cannot capture other classes of coherent errors. The twirling group is well suited to addressing this type of coherent error, which may arise from second-order interactions and control errors.

To understand the effect of the twirling group, we turn to some examples and examine the unitary and nonunitary components of the twirled superoperator. In what follows, the overall infidelity is

$$
\mathcal {E} _ {F} = 1 - \operatorname {T r} (\Phi) / d ^ {2}
$$

where  $\Phi$  is the superoperator of the channel,  $d$  is the dimension, and the stochastic, or incoherent, infidelity is

$$
\mathcal {E} _ {S} = 1 - \sqrt {\operatorname {T r} (\Phi \Phi^ {\dagger})} / d
$$

The coherent infidelity is then the overall infidelity minus the stochastic infidelity,

$$
\mathcal {E} _ {U} \equiv \mathcal {E} _ {F} - \mathcal {E} _ {S} = \frac {\sqrt {\operatorname {T r} (\Phi \Phi^ {\dagger})}}{d} - \frac {\operatorname {T r} (\Phi)}{d ^ {2}}.
$$

Twirling preserves the trace of  $\Phi$ ; i.e., it does not change the overall infidelity, but converts some or all of the coherent infidelity to incoherent infidelity.

We first examine the effect of a RZZ-type error in Fig. 10, and find that it is unaffected by the twirling group. We then consider a RXX-type error in Fig. 11 and RZX-type error in Fig. 12, and we find that the coherent error is partially mapped and completely mapped to incoherent error, respectively.

Finally, we numerically generate 1000 Haar-random coherent errors on an  $\mathrm{FSIM}(\pi /2,\pi /12)$  gate, and apply the twirling group. We find that the twirling effectiveness is approximately  $80\%$ , meaning that the coherent error of the resulting superoperator is about  $20\%$  of the initial value. We also observe in Fig. 13 that the remaining coherent errors are unitaries of the simplified form  $e^{i(aXX + bYY + cZZ)}$ .

# 2. Clifford data regression

Clifford data regression is a heuristic error-mitigation technique that estimates the effects of gate errors on observables as a suppression coefficient and mitigates the

![images/image17.jpg](images/image17.jpg)
Figure 10: The FSIM twirling group is applied to a RZZ$(\pi/12)$ error. The two panels show $\Lambda-1$ for the initial and twirled error channels, where $\Lambda$ is the Pauli transfer matrix. Left : the initial error channel has an overall infidelity of $\sin^{2}(\pi/24)\approx 1.70\%$ entirely due to coherent error. Right: the twirled channel, which also has a coherent infidelity of $1.70\%$. Thus, the RZZ-type error is not tailored at all by the twirling group. This type of error must be addressed by FSIM learning.

As shown in Fig. 11, the FSIM twirling group is applied to a RZZ$(\pi/12)$ error. The two panels show $\Lambda-1$ for the initial and twirled error channels, where $\Lambda$ is the Pauli transfer matrix. Left : the initial error channel has an overall infidelity of $\sin^{2}(\pi/24)\approx 1.70\%$ entirely due to coherent error. Right: the twirled channel, which also has a coherent infidelity of $1.70\%$. Thus, the RZZ-type error is not tailored at all by the twirling group. This type of error must be addressed by FSIM learning.

observables by amplifying them. The method is shown schematically in Fig. 14

For each target circuit to be executed, we construct an ensemble of random Clifford circuits with the same native gate structure. This is so that the target circuit and the Clifford circuits are affected by noise similarly. We execute the target and Clifford circuits and measure observables $O_{\text{Clifford}}^{\text{noisy}}$ and $O_{\text{tgt}}^{\text{noisy}}$. Due to the Gottesmann-Knill theorem *Gottesmann and Knill (1965)*, we can also exactly compute the noiseless observable in the Clifford circuits efficiently, $O_{\text{Clifford}}^{\text{noiseless}}$. Fitting $O_{\text{Clifford}}^{\text{noisy}}$ versus $O_{\text{Clifford}}^{\text{noiseless}}$, we find the average amount $r_{\text{suppress}}$ by which noise suppresses $\langle O\rangle$. The mitigated observable is then taken as

$O_{\text{tgt}}^{\text{mitigated}}=O_{\text{tgt}}^{\text{noisy}}/r_{\text{suppress}}.$ (10)

Typically, the suppression factor decreases exponentially with the circuit depth; therefore, a large number of measurements are needed to accurately measure observables for deep circuits. For our error rates, the suppression factor is typically $\gtrsim 0.01$ (see Figs. 17 and 18); therefore, $\mathcal{O}(10^{4})$ shots are sufficient.

#### V.1.1 Constructing the Clifford circuits

The typical expectation value of a given local Pauli observable $O$ in a random Clifford circuit is $0$; however, to measure the suppression $r_{\text{suppress}}$ due to noise, we need a nonzero value for the noiseless expectation $\langle O\rangle$. Reference *Hoffmann et al. (2008)* describes a method to mutate Clifford circuits such that $\langle O\rangle\neq 0$. Here, we describe a simpler method that only needs to mutate the first layer of one-qubit gates. We assume that the first layer in our Clifford circuit consists of only one-qubit gates.

A common method for simulating Clifford circuits is to propagate the stabilizers of the initial state forward through the circuit. Here, we consider an equivalent method, which evaluates the expectation value of any

![images/image18.jpg](images/image18.jpg)
![images/image19.jpg](images/image19.jpg)
![images/image20.jpg](images/image20.jpg)
![images/image21.jpg](images/image21.jpg)
Figure 11: The FSIM twirling group is applied to a RXX$(\pi/12)$ error. a) The Pauli transfer matrix of the initial channel is shown. The initial error channel has a coherent infidelity of $\sin^{2}(\pi/24)\approx 1.70\%$. b) The twirled channel has an overall infidelity of approximately $1.70\%$, with roughly equal coherent and incoherent infidelities. Note that the coherent error has also been modified by the twirling group. c) The stochastic component of the twirled channel, with a stochastic infidelity of $0.84\%$. The stochastic errors include both Pauli errors on the diagonal and non-Pauli errors off the diagonal. d) The unitary component of the twirled channel, with an infidelity of $0.87\%$. While the initial coherent error was $e^{-i\pi 0.042XX}$, the twirled coherent error is $e^{-i\pi(0.023XX+0.018YY)}$.

Pauli observable by *flowing* the Pauli observable backward through the circuit *Gottesmann and Knill (1965)*. Essentially, this method simulates the evolution of the observable in the Heisenberg picture. An example is shown in Fig. 15. We denote the Paulis $X,Y$, and $Z$ with three distinct colors – magenta, teal, and yellow, respectively, and we ignore the sign of the Pauli although it is straightforward to track if necessary. The identity matrix is not assigned any color.

For concreteness, let us express the random Clifford circuit as the product of the first layer of gates and all the rest of the gates, $U_{\text{Clifford}}=U_{\text{rest}}U_{\text{init}}$. Let $P_{i}$ be the local Pauli observable that we wish to measure,

$\langle P_{i}\rangle=\langle\psi_{0}|U_{\text{init}}^{\dagger}U_{\text{rest}}^{\dagger}P_{i}U_{\text{rest}}U_{\text{init}}|\psi_{0}\rangle\,.$ (11)

Without loss of generality, we assume $|\psi_{0}\rangle=|00\cdots\rangle$.

Here, $P_{i}$ is a weight-$1$ Pauli string at the end of the circuit. As it is back-propagated, each Clifford gate back-propagates it to another Pauli string, as exemplified in Fig. 15. The expectation value $\langle P_{i}\rangle$ is nonzero *iff* $P_{i}$ back-propagates to a Pauli string containing only $Z$ or

![images/image22.jpg](images/image22.jpg)
FIG. 12. The FSIM twirling group is applied to a RZX  $(\pi /12)$  (CNOT-like) error. The two panels show  $\Lambda -1$  for the initial and twirled error channels, where  $\Lambda$  is the Pauli transfer matrix. Left: the initial error channel has a coherent infidelity of  $1.70\%$ . Right: the twirled channel has an incoherent infidelity of  $1.70\%$  and no coherent error. The RZX-type error is entirely tailored by the twirling group.

![images/image23.jpg](images/image23.jpg)

![images/image24.jpg](images/image24.jpg)
FIG. 13. The FSIM twirling group is applied to an ensemble of random coherent errors. Top: the average generators of the random coherent errors are plotted. Since the ensemble is Haar random, the aggregate magnitude of each generator is approximately equal, although individual instances vary randomly. Bottom: the tailored unitary generators are plotted in the same format. We observe that the remaining coherent errors have only three generators: XX, YY, and ZZ.

$I$ ; i.e.,  $U_{\mathrm{Clifford}}^{\dagger}P_{i}U_{\mathrm{Clifford}}$  is a Pauli string with only  $\pm Z$  or  $I$ . If any qubit  $j$  in the back-propagated Pauli string contains any Pauli observable other than  $\pm Z$  or  $I$ , then  $\langle P_i\rangle = 0$ ; however, in this case, it is straightforward to mutate the one-qubit gate on qubit  $j$  in  $U_{\mathrm{init}}$  such that that Pauli on that qubit becomes  $\pm Z$ . After adjusting

![images/image25.jpg](images/image25.jpg)
FIG. 14. Clifford data regression. We execute Clifford circuits with the same structure as the target Trotter circuit, but with all the logical gates being replaced by Clifford gates. While the native gates on the hardware do not belong to the Clifford space, we express each logical Clifford gate in terms of at most two native two-qubit gates and several one-qubit gates. To mitigate errors in the target circuit that generically damp the result, we enhance the observables for the target circuit by the same amount we would in the Clifford circuit to obtain the correct result [Eq. (G2)].

![images/image26.jpg](images/image26.jpg)
FIG. 15. Scheme for constructing Clifford circuits. Given a random Clifford circuit instance and a target Pauli observable to measure, the Pauli observable is backpropagated through the circuit. The back-propagation is visualized here as a backward flow of the Pauli string, where the color scheme we use is that magenta, teal, and yellow stand for  $X, Y$ , and  $Z$  respectively, and  $I$  is not assigned any color. Each Pauli is also written on top of the flow for convenience. If the back-propagated Pauli string has any  $X$  or  $Y$ , then that Pauli observable would be measured as 0. Here, we find  $\langle Z_3 \rangle = 0$ . To make  $\langle Z_3 \rangle \neq 0$ , the one-qubit gates in the first layer can be adjusted (see main text and Fig. 16) such that the back-propagated observable is a Pauli string with only  $Z$  or  $I$  on all the qubits.

the gates in  $U_{\mathrm{init}}$ , we are left with a Clifford circuit such that  $\langle P_i \rangle = \pm 1$ .

Furthermore, rather than executing a circuit with appropriate  $U_{\mathrm{init}}$  for each Pauli observable  $P_{i}$ , it would be convenient if we could find  $U_{\mathrm{init}}$  such that expectation values of several Pauli observables  $\{P_i\}$  are simultaneously nonzero. Thus, we are tasked with finding a layer of one-qubit gates  $U_{\mathrm{init}}$  such that the number of observables for which  $\langle P_i\rangle \neq 0$  is maximized. Let us de

![images/image27.jpg](images/image27.jpg)
FIG. 16. The backward flow for the  $Z$  observable on each qubit through the circuit in Fig. 15. Since  $(Z_{1}, Z_{2}, Z_{3}, Z_{4})$  backflow to Pauli strings with no conflicts, we can pick  $U_{\mathrm{init}}$  which yields  $\langle Z_1\rangle, \langle Z_2\rangle, \langle Z_3\rangle, \langle Z_4\rangle \neq 0$ , e.g.,  $U_{\mathrm{init}} = H_3$ .

![images/image28.jpg](images/image28.jpg)
FIG. 17. Suppression of observables in Clifford data regression. (a) The measured expectation values of  $\langle X\rangle_{i}$  and  $\langle Y\rangle_{i}$  versus their values in a noiseless experiment, for the qubit indexed as 0 in the circuit. Each color corresponds to a circuit with different depth (steps). (b) The suppression of observables in Clifford circuits, estimated from the slopes of measured versus exact expectation values in (a). This suppression factor is used to rescale the observables in the target logical circuits to mitigate their errors.

note the Pauli string just after the first layer as  $P_{i}^{\prime}$ , i.e.  $U_{\mathrm{rest}}^{\dagger}P_{i}U_{\mathrm{rest}} = P_{i}^{\prime}$ . We want to find the largest group of  $P_{i}^{\prime}$  that do not have conflicting Paulis on any qubit [67]. Once we find this group of  $P_{i}^{\prime}$ , we can determine the one-qubit gates in  $U_{\mathrm{init}}$ . To this end, we construct a graph whose nodes are  $P_{i}$ . Two nodes are connected by an edge if their  $P_{i}^{\prime}$  do not have conflicting Paulis on any qubit. Then, the largest group of  $P_{i}^{\prime}$  that do not have differing Paulis on any qubit is the largest clique on this graph. We use a clique-finding algorithm in PYTHON's NETWORKX package, and find the largest set of  $P_{i}^{\prime}$  for which this can be satisfied. We then determine  $U_{\mathrm{init}}$  appropriately.

We exemplify the above procedure in Fig. 16 using the same Clifford circuit instance as Fig. 15. The four Pauli observables  $(Z_{1}, Z_{2}, Z_{3}, Z_{4})$  get back-propagated to  $(Z_{1}, Z_{2}, X_{3}Z_{4}, Z_{4})$ , respectively, just after  $U_{\mathrm{init}}$ . These Pauli strings do not have any conflicts; i.e., they form a clique of size 4. Therefore we can, e.g., choose  $U_{\mathrm{init}} = H_{3}$ .

![images/image29.jpg](images/image29.jpg)
FIG. 18. Suppression of observables in Clifford data regression. The suppression of observables in Clifford circuits versus qubit index and circuit depth. These suppression factors are used to rescale the observables in the target logical circuits, to mitigate the error in the target logical circuits. Panel (b) in Fig. 17 shows the trace for qubit 0.

This will give  $\langle Z_1\rangle ,\langle Z_2\rangle ,\langle Z_3\rangle ,\langle Z_4\rangle \neq 0$

# 3. Conservation of total spin

The spin Hamiltonian  $H$  [Eq. (3)] conserves  $\sigma_{\mathrm{tot}}^z = \sum_i\sigma_i^z$ . Therefore, in the experiments with a propagating wave packet [Fig. 4], where the initial state is an eigenstate of  $\sigma_{\mathrm{tot}}^z$  with eigenvalue  $N - 2$ , the quantum state after any number of Trotter steps still remains an eigenstate of  $\sigma_{\mathrm{tot}}^z$  with eigenvalue  $N - 2$ .

In practice, due to hardware noise, we also measure bitstrings with  $\sigma_{\mathrm{tot}}^z \neq N - 2$ . Postselecting for bitstrings with  $\sigma_{\mathrm{tot}}^z = N - 2$  is not necessarily scalable. We instead enforce conservation of  $\sigma_{\mathrm{tot}}^z$  on an ensemble level by rescaling each expectation value  $\langle \sigma_i^z \rangle$  with  $(N - 2) / \langle \sigma_{\mathrm{tot}}^z \rangle$ .

# Appendix H: Additional experimental data

Figures 19(a) and 19(b) show the raw and error-mitigated experimental data, respectively, for a wave packet incident on a plasma with  $\Delta_{\mathrm{max}} / J = 1 / 2$ , and the results of the noiseless simulation are given in Fig. 19(c). The plasma barrier max  $\Delta = J / 2$  is larger than the incident wave packet's average energy  $\epsilon_{k} = 0$ ; therefore, it is an overdense plasma. In our experiment, the  $|0\rangle$  state of the qubit corresponds to wave-packet density 1 and the  $|1\rangle$  state corresponds to wave packet density 0; an equal incoherent mixture (or coherent superposition) corresponds to density  $1 / 2$ . One of the dominant error sources in the experiment is the qubits' natural tendency to decay to the  $|0\rangle$  state on a time scale  $T_{1}\sim 37.6\mu s$  (see Table I), which is roughly  $50\%$  longer than the duration of the deepest circuits. This effect, which manifests as an

amplitude-damping noise channel, gets partially twirled in our circuits. It is reasonable to expect that the twirling drives the qubits to approximately an incoherent mixture of $|0\rangle$ and $|1\rangle$, which has a uniform wave-packet density of $1/2$, as can be observed in Fig.19(a). Removing this bias is crucial to obtaining accurate results, and it is qualitatively accomplished by Clifford data regression, as evidenced by the qualitatively better agreement between Figs.19(b) and (c).

Figure 20 shows the raw experimental data for wave-packet propagation in vacuum ($\Delta/J=0$) or incident on a plasma ($\Delta/J\neq 0$), starting with different relative phases between the wave-packet amplitudes. In principle, the relative phase sets the wave vector $k$, which determines the packet’s group velocity. In practice, this fact is not so visible in our small system, which can only initiate a relatively narrow wave packet that is not monochromatic. Figure 21 shows that the results are significantly improved after implementing Clifford data regression. In Figs. 20 and 21, the plasma is underdense in panels (e) and (f), where the barrier height $\Delta=J/2$ is smaller than the wave packet’s average energy of $\epsilon_{k}=J/\sqrt{2}$ and $\epsilon_{k}=J$, respectively. The plasma’s density is critical in panel (i), with $\epsilon_{k}=\Delta=J$.

Figures 5 and 22 show that a larger system can reliably initiate nearly monochromatic wave packets and probe the $k$ dependence of the reflection of the wave packets at the plasma boundary.

## Appendix I Reflection at a sharp jump in the plasma density

Reflection and transmission of EM waves at a sharp boundary is the simplest case in which the plasma density is nonuniform.

Reflection and transmission coefficients are well defined in a thermodynamically large system where a plane wave is incident on a sharp boundary, with some of it reflected and some transmitted. Analogous to the eigenmodes with hard walls in Appendix B, the plane-wave eigenmodes in a uniform thermodynamically large system have amplitudes

$c_{jk}=\sqrt{\frac{\Delta+\hbar\omega_{k}}{8\hbar\omega_{k}}}\exp(ikja),\quad\text{ if }j\text{ is odd}$
$c_{jk}=\frac{\sin ka}{\sqrt{8\hbar\omega_{k}(\Delta+\hbar\omega_{k})}}\exp(ikja),\quad\text{ if }j\text{ is even.}$ (I1)

We suppose an incoming wave with amplitude $1$ hits the sharp density jump at $j=0$ from the left, i.e. from $j<0$. Here, we assume an EM wave is incident from the left of the boundary. The boundary is at $j=0$, and the plasma density is such that $\Delta=0$ when $j<0$. We denote the reflected wave as having amplitude $r$ and the transmitted wave as having amplitude $t$. Thus, the spatial amplitudes of the plane wave to the left ($j<0$) are

$c_{jk_{\text{in}}}=\frac{1}{\sqrt{8}}\left(\exp(ik_{\text{in}}ja)+r\exp(-ik_{\text{in}}ja)\right),\text{ if }j\text{ is odd},$
$c_{jk_{\text{in}}}=\frac{\sin k_{\text{in}}a}{\sqrt{8}\hbar\omega_{k_{\text{in}}}}\left(\exp(ik_{\text{in}}ja)-r\exp(-ik_{\text{in}}ja)\right),\text{ if }j\text{ is even}$ (I2)

where $\hbar\omega_{k_{\text{in}}}=|\sin k_{\text{in}}a|$, and that of the transmitted wave ($j\geq 0$) is

$c_{jk_{\text{out}}}=t\sqrt{\frac{\Delta+\hbar\omega_{k_{\text{out}}}}{8\hbar\omega_{k_{\text{out}}}}}\exp(ik_{\text{out}}ja),\quad\text{ if }j\text{ is odd}$
$c_{jk_{\text{out}}}=t\frac{\sin ka}{\sqrt{8}\hbar\omega_{k_{\text{out}}}}\exp(ik_{\text{out}}ja),\quad\text{ if }j\text{ is even}$ (I3)

where $k_{\text{out}}$ is related to $k_{\text{in}}$ by energy conservation,

$\hbar\omega_{k_{\text{out}}}\equiv\sqrt{J^{2}\sin^{2}k_{\text{out}}a+\Delta^{2}}=\hbar\omega_{k_{\text{in}}}=J\sin k_{\text{in}}a.$ (I4)

Hereafter we denote $\hbar\omega\equiv\hbar\omega_{k_{\text{out}}}=\hbar\omega_{k_{\text{in}}}$. We denote $c_{jk_{\text{in}}}$ and $c_{jk_{\text{out}}}$ as simply $c_{j}$.

At the boundary, we have the relations:

$\hbar\omega c_{0}=\frac{iJ}{2}(c_{1}-c_{-1})-\Delta c_{0}$
$\hbar\omega c_{-1}=\frac{iJ}{2}(c_{0}-c_{-2}).$ (I5)

Solving for $r$ using the expressions for $c_{j<0}$ and $c_{j\geq 0}$ in Eqs. (I2) and (I3), we obtain

$r=\frac{(e^{ik_{\text{in}}a}-2i\frac{\Delta}{J})\sqrt{\sin k_{\text{in}}a+\frac{\Delta}{J}}-e^{ik_{\text{out}}a}\sqrt{\sin k_{\text{in}}a-\frac{\Delta}{J}}}{(e^{-ik_{\text{in}}a}+2i\frac{\Delta}{J})\sqrt{\sin k_{\text{in}}a+\frac{\Delta}{J}}+e^{ik_{\text{in}}a}\sqrt{\sin k_{\text{in}}a-\frac{\Delta}{J}}}.$ (I6)

The reflection coefficient $|r|$ versus $k_{\text{in}}$ is plotted in Fig. 22. When $|\sin ka|<\frac{\Delta}{J}$, $r$ is a pure phase; therefore, the incident wave is entirely reflected, $|r|=1$. When $|\sin ka|>\frac{\Delta}{J}$, the incident wave is partially transmitted, i.e. $|r|<1$.

## References

- (1) C. W. Bauer, Z. Davoudi, N. Klco, and M. J. Savage, Quantum simulation of fundamental particles and forces, Nat. Rev. Phys. 5, 420 (2023).
- (2) R. P. Drake, *High-Energy-Density Physics: Foundation of Inertial Fusion and Experimental Astrophysics*, Graduate Texts in Physics (Springer, 2018).
- (3) P. E. Grabowski, A. Markmann, I. V. Morozov, I. A. Valuev, C. A. Fichtl, D. F. Richards, V. S. Batista, F. R.

![images/image30.jpg](images/image30.jpg)
FIG. 19. Additional data from the experiment with an electromagnetic wave incident on an inhomogeneous plasma with  $\Delta_{\mathrm{max}} = J / 2$  from a region with  $\Delta = 0$ . Wave-packet densities (a) measured in the experiment without error mitigation, (b) after error mitigation on the experimental data, and (c) in a noiseless simulation of the circuits.

![images/image31.jpg](images/image31.jpg)

![images/image32.jpg](images/image32.jpg)

![images/image33.jpg](images/image33.jpg)

![images/image34.jpg](images/image34.jpg)

![images/image35.jpg](images/image35.jpg)

![images/image36.jpg](images/image36.jpg)

![images/image37.jpg](images/image37.jpg)

![images/image38.jpg](images/image38.jpg)

![images/image39.jpg](images/image39.jpg)
FIG. 20. Additional raw data from experiments with an electromagnetic wave incident on a plasma with  $\Delta \neq 0$  from a region with  $\Delta = 0$ , without error mitigation. In the first column, the initial wave packet is prepared with a relative phase between the two sites as  $ka = 0$ ; this phase is  $ka = \pi /4$  in the second column and  $\pi /2$  in the third column. The wave packet is incident on a plasma with  $\Delta /J = 0$  in the first row,  $\Delta /J = 0.5$  in the second row, and  $\Delta /J = 1$  in the third row.

![images/image40.jpg](images/image40.jpg)

![images/image41.jpg](images/image41.jpg)

Graziani, and M. S. Murillo, Wave packet spreading and localization in electron-nuclear scattering, Phys. Rev. E 87, 063104 (2013).
[4] F. R. Graziani, J. D. Bauer, and M. S. Murillo, Kinetic theory molecular dynamics and hot dense matter: Theoretical foundations, Phys. Rev. E 90, 033104 (2014).
[5] Y. Shi, J. Xiao, H. Qin, and N. J. Fisch, Simulations of relativistic quantum plasmas using real-time lattice scalar QED, Phys. Rev. E 97, 053206 (2018).
[6] N. C. Rubin, D. W. Berry, A. Kononov, F. D. Malone, T. Khattar, A. White, J. Lee, H. Neven, R. Babbush, and A. D. Baczewski, Quantum computation of stopping power for inertial fusion target design, Proc. Natl. Acad.

Sci. 121, e2317772121 (2024).
[7] Y. Shi, B. Evert, A. F. Brown, V. Tripathi, E. A. Sete, V. Geyko, Y. Cho, J. L. DuBois, D. Lidar, I. Joseph, and et al., Simulating nonlinear optical processes on a superconducting quantum device, J. Plasma Phys. 90, 805900602 (2024).
[8] D. A. Uzdensky and S. Rightley, Plasma physics of extreme astrophysical environments, Rep. Prog. Phys. 77, 036902 (2014).
[9] R. Gueroult, Y. Shi, J.-M. Rax, and N. J. Fisch, Determining the rotation direction in pulsars, Nat. Commun. 10, 3232 (2019).
[10] S. P. Jordan, H. Krovi, K. S. Lee, and J. Preskill, BQP-

![images/image42.jpg](images/image42.jpg)

![images/image43.jpg](images/image43.jpg)

![images/image44.jpg](images/image44.jpg)

![images/image45.jpg](images/image45.jpg)

![images/image46.jpg](images/image46.jpg)

![images/image47.jpg](images/image47.jpg)

![images/image48.jpg](images/image48.jpg)
FIG. 21. Wave-packet densities in the experiments of Fig. 20 after implementing error mitigation on the data. The order of the panels is the same as in Fig. 20.

![images/image49.jpg](images/image49.jpg)

![images/image50.jpg](images/image50.jpg)

![images/image51.jpg](images/image51.jpg)
FIG. 22. Reflection coefficient vs  $k$  at  $\Delta = J / 2$ .

completeness of scattering in scalar quantum field theory, Quantum 2, 44 (2018).
[11] P. C. S. Costa, S. Jordan, and A. Ostrander, Quantum algorithm for simulating the wave equation, Phys. Rev. A 99, 012323 (2019).
[12] A. Engel, G. Smith, and S. E. Parker, Quantum algorithm for the Vlasov equation, Phys. Rev. A 100, 062315 (2019).
[13] G. Vahala, L. Vahala, M. Soe, and A. K. Ram, Unitary quantum lattice simulations for Maxwell equations in vacuum and in dielectric media, J. Plasma Phys. 86, 905860518 (2020).
[14] G. Vahala, L. Vahala, M. Soe, and A. K. Ram, One-and two-dimensional quantum lattice algorithms for Maxwell

equations in inhomogeneous scalar dielectric media i: theory, Radiat. Eff. Defect S. 176, 49 (2021).
[15] I. Y. Dodin and E. A. Startsev, On applications of quantum computing to plasma simulations, Phys. Plasmas 28, 092101 (2021).
[16] I. Novikau, E. A. Startsev, and I. Y. Dodin, Quantum signal processing for simulating cold plasma waves, Phys. Rev. A 105, 062444 (2022).
[17] I. Joseph, Y. Shi, M. D. Porter, A. R. Castelli, V. I. Geyko, F. R. Graziani, S. B. Libby, and J. L. DuBois, Quantum computing for fusion energy science applications, Phys. Plasmas 30, 010501 (2023).
[18] I. Novikau, I. Dodin, and E. Startsev, Simulation of linear non-hermitian boundary-value problems with quantum singular-value transformation, Phys. Rev. Appl. 19, 054012 (2023).
[19] I. Novikau, I. Y. Dodin, and E. A. Startsev, Encoding of linear kinetic plasma problems in quantum circuits via data compression, J. Plasma Phys. 90, 805900401 (2024).
[20] I. Novikau and I. Joseph, Quantum algorithm for the advection-diffusion equation and the Koopman-von Neumann approach to nonlinear dynamical systems, Comput. Phys. Commun. 309, 109498 (2025).
[21] I. Novikau and I. Joseph, Explicit near-optimal quantum algorithm for solving the advection-diffusion equation, arXiv preprint arXiv:2501.11146 (2025).
[22] S. P. Jordan, K. S. Lee, and J. Preskill, Quantum algorithms for quantum field theories, Science 336, 1130 (2012).
[23] W. Hofstetter and T. Qin, Quantum simulation of

strongly correlated condensed matter systems, J. Phys. B: At., Mol. Opt. Phys. 51, 082001 (2018).
- (24) N. A. Zemlevskiy, Scalable quantum simulations of scattering in scalar field theory on 120 qubits, arXiv preprint arXiv:2411.02486 (2024).
- (25) R. C. Farrell, M. Illa, and M. J. Savage, Steps toward quantum simulations of hadronization and energy loss in dense matter, Phys. Rev. C 111, 015202 (2025).
- (26) R. Babbush, D. W. Berry, R. Kothari, R. D. Somma, and N. Wiebe, Exponential quantum speedup in simulating coupled classical oscillators, Phys. Rev. X 13, 041041 (2023).
- (27) Y. Shi, A. R. Castelli, X. Wu, I. Joseph, V. Geyko, F. R. Graziani, S. B. Libby, J. B. Parker, Y. J. Rosen, L. A. Martinez, and J. L. DuBois, Simulating non-native cubic interactions on noisy quantum machines, Phys. Rev. A 103, 062608 (2021).
- (28) J. Zylberman, G. Di Molfetta, M. Brachet, N. F. Loureiro, and F. Debbasch, Quantum simulations of hydrodynamics via the Madelung transformation, Phys. Rev. A 106, 032408 (2022).
- (29) M. Porter and I. Joseph, Impact of dynamics, entanglement and markovian noise on the fidelity of few-qubit digital quantum simulation, J. Plasma Phys. 91, E39 (2025).
- (30) J. Andress, A. Engel, Y. Shi, and S. Parker, Quantum simulation of nonlinear dynamical systems using repeated measurement, J. Plasma Phys. 91, E49 (2025).
- (31) A. Smith, M. S. Kim, F. Pollmann, and J. Knolle, Simulating quantum many-body dynamics on a current digital quantum computer, npj Quantum Inf. 5, 106 (2019).
- (32) E. Rosenberg, T. I. Andersen, R. Samajdar, A. Petukhov, J. Hoke, D. Abanin, A. Bengtsson, I. K. Drozdov, C. Erickson, P. V. Klimov, et al., Dynamics of magnetization at infinite temperature in a Heisenberg spin chain, Science 384, 48 (2024).
- (33) C. Huang, T. Wang, F. Wu, D. Ding, Q. Ye, L. Kong, F. Zhang, X. Ni, Z. Song, Y. Shi, et al., Quantum instruction set design for performance, Phys. Rev. Lett. 130, 070601 (2023).
- (34) Z. Cai, R. Babbush, S. C. Benjamin, S. Endo, W. J. Huggins, Y. Li, J. R. McClean, and T. E. O’Brien, Quantum error mitigation, Rev. Mod. Phys. 95, 045005 (2023).
- (35) K. Temme, S. Bravyi, and J. M. Gambetta, Error mitigation for short-depth quantum circuits, Phys. Rev. Lett. 119, 180509 (2017).
- (36) E. Van Den Berg, Z. K. Minev, A. Kandala, and K. Temme, Probabilistic error cancellation with sparse pauli–lindblad models on noisy quantum processors, Nat. Phys. 19, 1116 (2023).
- (37) S. Filippov, M. Leahy, M. A. C. Rossi, and G. García-Pérez, Scalable tensor-network error mitigation for near-term quantum computing, arXiv preprint arXiv:2307.11740 (2023).
- (38) P. Czarnik, A. Arrasmith, P. J. Coles, and L. Cincio, Error mitigation with Clifford quantum-circuit data, Quantum 5, 592 (2021).
- (39) T. H. Stix, Waves in plasmas (Springer Science & Business Media, 1992).
- (40) D. G. Swanson, Plasma waves (CRC Press, 2020).
- (41) The standard way to solve this Hamiltonian is by mapping it to a free-fermionic Hamiltonian or a hardcore bosonic Hamiltonian; See Appendix B for details.
- (42) E. A. Sete, A. Q. Chen, R. Manenti, S. Kulshreshtha, and S. Poletto, Floating tunable coupler for scalable quantum computing architectures, Phys. Rev. Appl. 15, 064063 (2021).
- (43) J. Chen, D. Ding, C. Huang, and Q. Ye, Compiling arbitrary single-qubit gates via the phase shifts of microwave pulses, Phys. Rev. Res. 5, L022031 (2023).
- (44) D. C. McKay, I. Hincks, E. J. Pritchett, M. Carroll, L. C. G. Govia, and S. T. Merkel, Benchmarking quantum processor performance at scale (2023), arXiv:2311.05933 [quant-ph].
- (45) H.-Y. Huang, R. Kueng, and J. Preskill, Predicting many properties of a quantum system from very few measurements, Nat. Phys. 16, 1050 (2020).
- (46) S. E. Harris, Electromagnetically induced transparency in an ideal plasma, Phys. Rev. Lett. 77, 5357 (1996).
- (47) B. Sundar, Simulating plasma wave propagation on a superconducting quantum chip, 10.5281/zenodo.16115660 (2025).
- (48) Alternatively, one may map it to a free-fermionic Hamiltonian using a Jordan-Wigner transformation [68]. They are equivalent in one dimension.
- (49) E. Koukoutsis, K. Hizanidis, A. K. Ram, and G. Vahala, Dyson maps and unitary evolution for maxwell equations in tensor dielectric media, Phys. Rev. A 107, 042215 (2023).
- (50) E. Koukoutsis, K. Hizanidis, G. Vahala, M. Soe, L. Vahala, and A. K. Ram, Quantum computing perspective for electromagnetic wave propagation in cold magnetized plasmas, Phys. Plasmas 30, 122108 (2023), https://pubs.aip.org/aip/pop/article-pdf/doi/10.1063/5.0177589/18261245/122108_1_5.0177589.pdf.
- (51) C. Moler and C. Van Loan, Nineteen dubious ways to compute the exponential of a matrix, twenty-five years later, SIAM Review 45, 3 (2003), https://doi.org/10.1137/S00361445024180.
- (52) S. Gaudreault, G. Rainwater, and M. Tokman, Kiops: A fast adaptive krylov subspace solver for exponential integrators, Journal of Computational Physics 372, 236 (2018).
- (53) P. Bader, S. Blanes, F. Casas, and M. Seydaoğlu, An efficient algorithm to compute the exponential of skew-hermitian matrices for the time integration of the schrodinger equation, Mathematics and Computers in Simulation 194, 383 (2022).
- (54) E. A. Sete, N. Didier, A. Q. Chen, S. Kulshreshtha, R. Manenti, and S. Poletto, Parametric-resonance entangling gates with a tunable coupler, Phys. Rev. Appl. 16, 024050 (2021).
- (55) Convention for fsim.
- (56) M. Mohseni, A. T. Rezakhani, and D. A. Lidar, Quantum-process tomography: Resource analysis of different strategies, Phys. Rev. A 77, 032322 (2008).
- (57) F. Arute, K. Arya, R. Babbush, D. Bacon, J. C. Bardin, R. Barends, R. Biswas, S. Boixo, F. G. S. L. Brandao, D. A. Buell, et al., Quantum supremacy using a programmable superconducting processor, Nature 574, 505 (2019).
- (58) A. Arrasmith, A. Patterson, A. Boughton, and M. Paini, Development and demonstration of an efficient readout error mitigation technique for use in misq algorithms (2023), arXiv:2303.17741 [quant-ph].
- (59) D. C. McKay, C. J. Wood, S. Sheldon, J. M. Chow, and J. M. Gambetta, Efficient Z-Gates for Quantum Computing, Phys. Rev. A 96, 022330 (2017), arXiv:1612.00858 [quant-ph].

- (60) A. Barenco, C. H. Bennett, R. Cleve, D. P. DiVincenzo, N. Margolus, P. Shor, T. Sleator, J. A. Smolìn, and H. Weinfurter, Elementary gates for quantum computation, Phys. Rev. A 52, 3457 (1995).
- (61) P. Jurcevic, A. Javadi-Abhari, L. S. Bishop, I. Lauer, D. F. Bogorin, M. Brink, L. Capelluto, O. Günlük, T. Itoko, N. Kanazawa, A. Kandala, G. A. Keefe, K. Krsulich, W. Landers, E. P. Lewandowski, D. T. McClure, G. Nannicini, A. Narasgond, H. M. Nayfeh, E. Pritchett, M. B. Rothwell, S. Srinivasan, N. Sundaresan, C. Wang, K. X. Wei, C. J. Wood, J.-B. Yau, E. J. Zhang, O. E. Dial, J. M. Chow, and J. M. Gambetta, Demonstration of quantum volume 64 on a superconducting quantum computing system, Quantum Sci. Technol. 6, 025020 (2021), publisher: IOP Publishing.
- (62) chroma-core/chroma (2025), original-date: 2022-10-05T17:58:44Z.
- (63) Y. Kim, A. Eddins, S. Anand, K. X. Wei, E. Van Den Berg, S. Rosenblatt, H. Nayfeh, Y. Wu, M. Zaletel, K. Temme, et al., Evidence for the utility of quantum computing before fault tolerance, Nature 618, 500 (2023).
- (64) K. Tsubouchi, Y. Mitsuhashi, K. Sharma, and N. Yoshioka, Symmetric clifford twirling for cost-optimal quantum error mitigation in early ftqc regime, arXiv preprint arXiv:2405.07720 (2024).
- (65) D. Gottesman, The Heisenberg representation of quantum computers, arXiv preprint quant-ph/9807006 (1998).
- (66) This flow of a Pauli through a Clifford circuit is called by various names in the literature – as stabilizer flow in [69], Pauli web in [70], and as spacetime codes in [71].
- (67) Here, $X$ and $Y$ are conflicting Paulis, as are $X$ and $Z$, and $Y$ and $Z$; $X$ and $X$ are not conflicting Paulis, and so on. Note that $I$ does not conflict with any Pauli.
- (68) P. Jordan and E. P. Wigner, Über das paulische äquivalenzverbot, Vol. 47 (Springer (), 1928) pp. 631–651.
- (69) M. McEwen, D. Bacon, and C. Gidney, Relaxing hardware requirements for surface code circuits using time-dynamics, Quantum 7, 1172 (2023).
- (70) H. Bombin, D. Litinski, N. Nickerson, F. Pastawski, and S. Roberts, Unifying flavors of fault tolerance with the zx calculus, Quantum 8, 1379 (2024).
- (71) N. Delfosse and A. Paetznick, Spacetime codes of Clifford circuits, arXiv preprint arXiv:2304.05943 (2023).