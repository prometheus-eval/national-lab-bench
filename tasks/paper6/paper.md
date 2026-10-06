# Simulations of relativistic-quantum plasmas using real-time lattice scalar QED

Yuan Shi [ yshi@pppl.gov Department of Astrophysical Sciences, Princeton University, Princeton, NJ 08544 USA ] Jianyuan Xiao Hong Qin [ Department of Astrophysical Sciences, Princeton University, Princeton, NJ 08544 USA Hong Qin@pppl.gov ] Nathaniel J. Fisch [ Department of Astrophysical Sciences, Princeton University, Princeton, NJ 08543 USA School of Nuclear Science and Technology and Department of Modern Physics, University of Science and Technology of China, Hefei, Anhui 230026, China ]

###### Abstract

Real-time lattice quantum electrodynamics (QED) provides a unique tool for simulating plasmas in the strong-field regime, where collective plasma scales are not well-separated from relativistic-quantum scales. As a toy model, we study scalar QED, which describes self-consistent interactions between charged bosons and electromagnetic fields. To solve this model on a computer, we first discretize the scalar-QED action on a lattice, in a way that respects geometric structures of exterior calculus and U(1)-gauge symmetry. The lattice scalar QED can then be solved, in the classical-statistics regime, by advancing an ensemble of statistically equivalent initial conditions in time, using classical field equations obtained by extremizing the discrete action. To demonstrate the capability of our numerical scheme, we apply it to two example problems. The first example is the propagation of linear waves, where we recover analytic wave dispersion relations using numerical spectrum. The second example is an intense laser interacting with a 1D plasma slab, where we demonstrate natural transition from wakefield acceleration to pair production when the wave amplitude exceeds the Schwinger threshold. Our real-time lattice scheme is fully explicit and respects local conservation laws, making it reliable for long-time dynamics. The algorithm is readily parallelized using domain decomposition, and the ensemble may be computed using quantum parallelism in the future.

## I Introduction

Lattice QED, a scheme usually used to study vacuum quantum electrodynamics, can also be used to simulate plasmas. By adding dynamical background fields, we extend lattice QED into a valuable tool for plasma physics, especially when plasmas are dense or when fields are strong. Under these extreme conditions where collective QED effects are important, the commonly adopted plasma kinetic model, which arises as the geometrical optics approximation of the relativistic-quantum world [1], is no longer sufficient. An example where QED effects are important is the production of electron-positron pairs when intense lasers interact with plasma targets [2; 3; 4; 5]. To describe such phenomena in semiclassical framework, source terms must be inserted into quantum kinetic or fluid equations [6; 7; 8; 9; 10], which can then be solved by numeric integration [11; 12] or QED-PIC simulations [13; 14; 15]. However, prefabricated source terms take little account of the interplay between coexisting processes [16], nor can they describe quantum interference, through which the created pairs may be entangled. Therefore, while semiclassical approximations may be applicable for long-wavelength lasers, large errors are expected when fields, such as those of x-ray lasers, evolve on scales comparable to intrinsic QED scales. Moreover, in semiclassical treatments, there is no obvious way to subtract both energy and momentum from fields once pairs are produced. Although errors may be tolerable when fields dominate particles, disrespecting energy-momentum conservation will likely have nonphysical consequences, after a large number of pairs are generated.

To model plasmas where QED processes have no clear scale separation from classical processes, a faithful description can only be provided on the relativistic-quantum level. While lattice simulations may be unfamiliar tools for plasma physics, they have been used extensively in quantum chromodynamics (QCD) to describe the strong interaction [17] and quark-gluon plasmas [18; 19]. In conventional lattice-QCD simulations, quantum correlation functions are computed using numerical path integrals, from which observables are extracted as coefficients of scaling laws [20]. This scheme can be analytically continued to imaginary time to describe statistical systems in thermal equilibrium [21]. For out-of-equilibrium systems, real-time simulations can be carried out using the Schwinger-Keldysh time contours [22; 23]. The above formulations, based on numerical path integrals, are capable of capturing genuine quantum loop effects, but are numerically expensive. Fortunately, the computational cost can be dramatically reduced when the occupation numbers of quantum states are high and when the coupling is weak. This is precisely the case for plasma physics, where a large number of particles are present, and the coupling coefficient $e\approx 0.3$ is small. In this classical-statistic regime, tree-level effects dominate loop effects [24; 25; 26; 27; 28], and the quantum system can be adequately described by time-advancing the classical field equations with an ensemble of statistically equivalent initial conditions [29; 30; 31; 32]. Based on this approach, lattice spinor-QED simulations have been carried out to demonstrate production of fermion pairs from the vacuum by a prescribed external electric field in one spatial dimension [33; 34; 35]. However, the role of background plasmas during pair production has not been investigated. By incorporating a nonperturbative amount of background particle fields, we turn real-time lattice simulations into numerical tools useful for plasma physics.

In this paper, we demonstrate plasma effects during pair productions using the scalar-QED model, where the electromagnetic (EM) fields evolve in a self-consistent manner, instead of being imposed from boundary and initial conditions. Using the scalar-QED model, we avoid the fermion doubling problem when discretizing the Dirac field [36] and focus on unambiguous plasma contributions. The scalar-QED model governs interactions between EM fields and spin-0 charged bosons, such as charged pions or Cooper pairs. Although laboratory plasmas are typically made of spin-1/2 charge fermions, classical plasma physics takes no account of particle spin-statistics at all. Therefore, to demonstrate that lattice simulations are useful for plasma physics, it is sufficient to study scalar QED, which has been used to describe laser-plasma interaction in relativistic-quantum regime [37; 38]. In the classical-statistics regime, scalar QED is governed by Klein-Gordon-Maxwell (KGM) equations

$(D_{\zeta,\mu}D_{\zeta}^{\mu}+m_{\zeta}^{2})\phi_{\zeta}=0,$ (1)
$\partial_{\mu}F^{\mu\nu}=j^{\nu},$ (2)

where we have used the natural units $\hbar=c=\epsilon_{0}=1$. In the above equations, $\phi_{\zeta}$ is the complex scalar field, describing spin-0 bosons of species $\zeta$, whose charge is $q_{\zeta}$ and mass is $m_{\zeta}$. The real-valued 1-form $A_{\mu}$ is the gauge field, describing spin-1 bosons, and $F_{\mu\nu}=\partial_{\mu}A_{\nu}-\partial_{\nu}A_{\mu}$ is the field strength tensor. Charged bosons couple to the gauge field through the covariant derivative $D_{\zeta,\mu}=\partial_{\mu}-iq_{\zeta}A_{\mu}$, and the gauge field couples with charged fields through the gauge-invariant current density

$j^{\mu}=\sum_{\zeta}\frac{q_{\zeta}}{i}\big{[}\bar{\phi}_{\zeta}(D_{\zeta}^{\mu}\phi_{\zeta})-\text{c.c.}\big{]},$ (3)

where $\bar{\phi}_{\zeta}$ denotes complex conjugation of $\phi_{\zeta}$. By the famous Klein paradox [39], the charged scalar field $\phi_{\zeta}$ cannot be interpreted as the probability amplitude of a single particle. A more appropriate interpretation is that the classical field $\phi_{\zeta}$ is intrinsically a many-particle field, which can be represented as $\phi_{\zeta}(x)=\int\sqrt{V}\Phi_{\zeta}(x,x_{2},x_{3},\dots)$, where $\Phi_{\zeta}(x,x_{2},x_{3},\dots)$ is the symmetrized many-body wave function, and the integration is carried out on the many-body configurations space [38]. Regardless of the interpretation, we can solve the KGM equations as coupled partial differential equations, whose solutions model the tree-level behavior of charged bosons interacting with EM fields.

This paper is organized as follows. In Sec. II, we develop a variational algorithm for solving the KGM equations. In Sec. III, we apply this algorithm to two example problems in plasma physics. The first example is the propagation of linear waves, where we compare numerical spectra with analytical dispersion relations. The second example is wakefield acceleration and pair production, when intense lasers interact with a 1D plasma slab. Conclusion and discussion are given in Sec. IV. In Appendix A, we discuss local conservation laws underlying our algorithm. In Appendix B, we summarize an explicit numerical scheme using the Lorenz gauge condition.

## II Variational algorithm

In the continuum, the KGM equations can be derived from the action $S=\int d^{4}x\mathcal{L}$, where the Lorentz-invariant and U(1)-gauge-invariant scalar-QED Lagrangian

$\mathcal{L}=(\overline{D_{\mu}\phi})(D^{\mu}\phi)-m^{2}\bar{\phi}\phi-\frac{1}{4}F_{\mu\nu}F^{\mu\nu}.$ (4)

Here we have omitted the species subscript $\zeta$, and the summation of charged species is implied. By Noether’s theorem, the U(1)-gauge symmetry of the action

$\phi\rightarrow\phi e^{iq\alpha},~{}~{}A_{\mu}\rightarrow A_{\mu}+\partial_{\mu}\alpha,$ (5)

implies charge conservation $\partial_{\mu}j^{\mu}=0$, where the current density $j^{\mu}$ is given by Eq. (3). Similarly, by the Lorentz symmetry, energy and momentum are also conserved $\partial_{\mu}\mathcal{T}^{\mu\nu}=0$, where the gauge invariant stress-energy tensor

$\mathcal{T}^{\mu\nu}$ $=(\overline{D^{\mu}\phi})(D^{\nu}\phi)+(D^{\mu}\phi)(\overline{D^{\nu}\phi})$
$+F^{\mu\sigma}F_{\sigma}{}^{\nu}-g^{\mu\nu}\mathcal{L}.$ (6)

Here, $g^{\mu\nu}$ is the Minkowski metric with characteristics $(+,-,-,-)$. This scalar-QED theory, omitting the $\phi^{4}$ self-coupling, is the underlying model of our algorithm on the discrete spacetime lattice. In fact, a variational algorithm for solving the KGM equations has already been developed in the numerical analysis community [40], which shows superior charge conservation property when gauge symmetry is respected. In this paper, we rederive the variational algorithm in arbitrary gauge, using local energy conservation to justify the choice of Yee-type action [41] over Wilson-type action [17], and emphasize on the application of such algorithm to plasma physics.

### II.1 Discretization of fields and action

To solve the continuous system numerically, let us discretize the spacetime manifold using a rectangular lattice. Then the scalar field $\phi$, namely, a function on the spacetime manifold, naturally lives on the vertexes of the discrete manifold

$\phi^{n}_{i,j,k}:=\phi(t_{n},x_{i},y_{j},z_{k}),$ (7)

where $(t_{n},x_{i},y_{j},z_{k})$ is the coordinate of the vertex. In comparison, the gauge 1-form $A=A_{\mu}dx^{\mu}$ naturally lives along the edges of the discrete spacetime manifold. For example, the $t$- and $x$-components

$A^{n+\frac{1}{2}}_{i,j,k}:=+A^{0}(t_{n}+\frac{\Delta t}{2},x_{i},y_{j},z_{k}),$ (8)
$A^{n}_{i+\frac{1}{2},j,k}:=-A^{1}(t_{n},x_{i}+\frac{\Delta x}{2},y_{j},z_{k}),$ (9)

where $\Delta t=t_{n+1}-t_{n}$ and $\Delta x=x_{i+1}-x_{i}$. The minus sign comes from the Minkowski metric $g_{\mu\nu}$, which lower the index $A_{\mu}=g_{\mu\nu}A^{\nu}$. In the above discretization, a half-integer index indicates which edge does the field resides

![images/image1.jpg](images/image1.jpg)
Figure 1: Discretization of the $txy$-submanifold of spacetime. The discrete function $\phi_{v}$ lives on the vertexes (blue squares). For example, $\phi_{i,j,k}^{n}=\phi(t_{n},x_{i},y_{j},z_{k})$ lives on the vertex $(n,i,j,k)$. The discrete 1-form $A_{e}$ lives along edges (red circles). For example, the $t$-component $A_{i+1,j,k}^{n+1/2}=A^{0}(t_{n}+\Delta t/2,x_{i+1},y_{j},z_{k})$ lives along the edge connecting vertexes $(n,i+1,j,k)$ and $(n+1,i+1,j,k)$, and the $x$-component $A_{i+1/2,j,k}^{n}=-A^{1}(t_{n},x_{i}+\Delta x/2,y_{j},z_{k})$ lives along the edge connecting vertexes $(n,i,j,k)$ and $(n,i+1,j,k)$. The discrete 2-form $F_{f}$ lives on faces (green crosses). For example, electric field $E_{i+1/2,j,k}^{n+1/2}=E^{x}(t_{n}+\Delta t/2,x_{i}+\Delta x/2,y_{j},z_{k})$ lives on the time-like face spanned by vertexes $(n,i,j,k),(n+1,i,j,k),(n+1,i+1,j,k)$ and $(n,i+1,j,k)$; magnetic field $B_{i+1/2,j+1/2,k}^{n+1}=B^{z}(t_{n+1},x_{i}+\Delta x/2,y_{j}+\Delta y/2,z_{k})$ lives on the space-like face spanned by vertexes $(n+1,i,j,k),(n+1,i,j+1,k),(n+1,i+1,j+1,k)$ and $(n+1,i+1,j,k)$.

along. For example, $A_{i,j,k}^{n+1/2}$ resides along the edge connecting vertices $(t_{n},x_{i},y_{j},z_{k})$ and $(t_{n+1},x_{i},y_{j},z_{k})$, and is therefore the $A_{0}$ component of $A$. Notice that since $A$ is a 1-form living along edges, only one of its four indices can take half-integer values, while the other three indices must take integer values. Moreover, to each edge of the lattice, the discrete 1-form only assigns the component of $A$ that is parallel to this edge (Fig. 1), while other components of $A$ are not assigned.

Now that we have discretized the fields, the gauge-covariant derivatives can be computed using the Wilson’s lines [17]. Since the gauge-covariant derivatives are 1-forms, they also lives along edges when discretized. For example, the $t$- and $x$-components of the pull-back gauge covariant derivatives are

$(D_{0}^{&lt;}\phi)_{i,j,k}^{n+\frac{1}{2}}$ $=\frac{1}{\Delta t}\Big{(}\phi_{i,j,k}^{n+1}e^{-iq\Delta tA_{i,j,k}^{n+\frac{1}{2}}}-\phi_{i,j,k}^{n}\Big{)},$ (10)
$(D_{1}^{&lt;}\phi)_{i+\frac{1}{2},j,k}^{n}$ $=\frac{1}{\Delta x}\Big{(}\phi_{i+1,j,k}^{n}e^{-iq\Delta xA_{i+\frac{1}{2},j,k}^{n}}-\phi_{i,j,k}^{n}\Big{)}.$ (11)

These pull-back covariant derivatives transform under the discrete U(1)-gauge symmetry [Eqs. (12)-(13)] as $\phi_{i,j,k}^{n}$ [Eq. (14)]. Analogously, one can define push-forward covariant derivatives, which we shall not use in this paper. The gauge field $A$ serves as the 1-form defining the connection on the U(1)-bundle, which enables parallel transport $\phi$ on the spacetime manifold.

To compute the field strength tensor $F_{\mu\nu}$, notice that $F=dA$ is the curvature 2-form and hence lives on faces of the lattice upon discretization. For example, the time-like component $F_{01}=E^{1}$ is the electric field in the $x$-direction, which can be computed by

$E_{i+\frac{1}{2},j,k}^{n+\frac{1}{2}}=\frac{A_{i+\frac{1}{2},j,k}^{n+1}-A_{i+\frac{1}{2},j,k}^{n}}{\Delta t}-\frac{A_{i+1,j,k}^{n+\frac{1}{2}}-A_{i,j,k}^{n+\frac{1}{2}}}{\Delta x}.$ (12)

This component lives on the time-like face spanned by four vertices $(n,i,j,k)$, $(n,i+1,j,k)$, $(n+1,i+1,j,k)$, and $(n+1,i,j,k)$. Analogously, we can compute the space-like components of $F$. For example, $F_{12}=-B^{3}$ is the magnetic field in the $z$-direction

$-B_{i+\frac{1}{2},j+\frac{1}{2},k}^{n}$ $=\frac{1}{\Delta x}\bigg{(}A_{i+1,j+\frac{1}{2},k}^{n}-A_{i,j+\frac{1}{2},k}^{n}\bigg{)}$
$-\frac{1}{\Delta y}\bigg{(}A_{i+\frac{1}{2},j+1,k}^{n}-A_{i+\frac{1}{2},j,k}^{n}\bigg{)}.$ (13)

This $z$-component of the magnetic field lives on the space-like face spanned by four vertices $(n,i,j,k)$, $(n,i+1,j,k)$, $(n,i+1,j+1,k)$ and $(n,i,j+1,k)$. Notice that the sign of the discretized $F$ is determined by the orientation of the face. Since the above discretization respects geometric structures of exterior calculus, the Bianchi identities, namely, $\nabla\cdot{\bf B}=0$ and the Faraday’s law, are automatically and exactly satisfied (Appendix A).

Using the discrete gauge-covariant derivatives and the discrete field strength, the action can be discretized by

$S_{d}=\sum_{c}\Delta VL_{d}[\phi_{v},A_{e}],$ (14)

where $\phi_{v}$ and $A_{e}$ are the discrete fields. Here the subscript $v$ denotes vertexes, and $e$ denotes edges. In the discrete action, $\Delta V$ is the volume 4-form, and the summation runs over all cells of the lattice. In each unit cell, the discrete Lagrangian function

$L_{d}=(\overline{D_{\mu}\phi})_{e}(D^{\mu}\phi)_{e}-m^{2}\bar{\phi}_{v}\phi_{v}+\frac{1}{2}(E_{f}^{2}-B_{f}^{2}),$ (15)

where summations over unique $e$, $v$, and $f$ are implied. Here, the subscript $f$ denotes faces. Notice that in contrast to what is typically done in lattice gauge theories, here we have directly used the discrete field strength $F_{\mu\nu}F^{\mu\nu}$, instead of the Wilsonian plaquettes $L_{w}\propto{\rm Re}[1-\exp(ieF_{\mu\nu}\Delta_{\mu}\Delta_{\nu})]$. We made this choice so that a discrete version of the local energy conservation law for $EM$ fields is exactly satisfied when the coupling goes to zero (Appendix A) .

### II.2 Equations of motion for discrete fields

Having discretized the action, the classical equation of motion (EOM) for the discrete field $\phi_{v}$ can be obtained by extremizing $S_{d}$. Taking variation with $\bar{\phi}_{v}$ and set $\delta S_{d}/\delta\bar{\phi}_{v}=0$, a discrete version of equation of Eq. (1) is

$\frac{1}{\Delta t^{2}}\bigg{(}\phi_{s}^{n+1}e^{-iq\Delta tA_{s}^{n+\frac{1}{2}}}-2\phi_{s}^{n}+\phi_{s}^{n-1}e^{iq\Delta tA_{s}^{n-\frac{1}{2}}}\bigg{)}$ (16)
$=\frac{1}{\Delta_{t}^{2}}\bigg{(}\phi_{s+t}^{n}e^{-iq\Delta_{l}A_{s+\frac{1}{2}}^{n}}-2\phi_{s}^{n}+\phi_{s-l}^{n}e^{iq\Delta_{l}A_{s-\frac{1}{2}}^{n}}\bigg{)}-m^{2}\phi_{s}^{n},$

where the time index is explicit, the vertex-centered spatial index is abbreviated as $s=(i,j,k)$, and summations in $l=i,j,k$ directions are implied. By taking variation with $\phi_{v}$, we can obtain the EOM for $\bar{\phi}_{v}$, which is the complex conjugation of the above equation. The finite difference equation (16) is centered around vertexes, and couples $\phi_{v}$ with its eight nearest neighbors though $A_{e}$, as illustrated by Fig. 2(a) in the $tx$-submanifold.

To find the equation for the electric field, take variation of $S_{d}$ with respect to the time-like component $A_{s}^{n+1/2}$. Setting $\delta S_{d}/\delta A_{s}^{n+1/2}=0$, we obtain a discrete version of the Gauss’ law, centered along time-like edges

$\frac{1}{\Delta_{l}}\Big{(}E_{s+\frac{1}{2}}^{n+\frac{1}{2}}-E_{s-\frac{1}{2}}^{n+\frac{1}{2}}\Big{)}=J_{s}^{n+1/2}.$ (17)

The charge density 1-form $J_{s}^{n+1/2}$ is the hodge dual of the charge density 3-form $j_{0}=\star j^{0}$, discretized by

$J_{s}^{n+1/2}=\frac{iq}{\Delta t}\Big{(}\bar{\phi}_{s}^{n+1}e^{iq\Delta tA_{s}^{n+\frac{1}{2}}}\phi_{s}^{n}-\text{c.c.}\Big{)}.$ (18)

When there are multiple charged species, the right-hand side (RHS) should sum over charge densities of all species. In Fig. 2(b), we illustrate the coupling pattern of the above finite difference equation.

To find the equations involving components of the magnetic field, take variation of $S_{d}$ with respect to the space-like components $A_{s+l/2}^{n}$. For example, by setting $\delta S_{d}/\delta A_{i+1/2,j,k}^{n}=0$, we obtain an equation advancing the $x$-component of the electric field in time by

$\frac{E_{s+\frac{1}{2}}^{n+\frac{1}{2}}-E_{s+\frac{1}{2}}^{n-\frac{1}{2}}}{\Delta t}=\epsilon_{ijk}\frac{B_{r-\frac{k}{2}}^{n}-B_{r-\frac{k}{2}-j}^{n}}{\Delta_{j}}+J_{s+\frac{1}{2}}^{n}.$ (19)

Here, $r=(i+1/2,j+1/2,k+1/2)$ is the abbreviated index for the body center, $\epsilon_{ijk}$ is the Levi-Civita symbol, and summations over repeated indexes are implied. The current density 1-form $J_{s+i/2}^{n}$ is the hodge dual of the current density 3-form $j_{i}=\star j^{i}$. The hodge dual gives rise to a negative sign, so that the $x$-component of the current density $-j^{x}$ is discretized by

$J_{s+\frac{l}{2}}^{n}=\frac{iq}{\Delta_{l}}\Big{(}\bar{\phi}_{s+l}^{n}e^{iq\Delta_{l}A_{s+\frac{l}{2}}^{n}}\phi_{s}^{n}-\text{c.c.}\Big{)}.$ (20)

The finite difference equation (19) is the discrete version of the Maxwell-Ampère’s law $\partial_{t}E^{x}=\partial_{y}B^{z}-\partial_{z}B^{y}-j^{x}$ centered around space-like edges, whose coupling pattern is illustrated in Fig. 2(c). When computing $j_{x}$ on the RHS, summation over charged species is implied.

In order to advance the above finite difference equations in time, we need to fix a gauge to eliminate the extra degree of freedom. Since the discrete action $S_{d}$ is U(1)-gauge invariant (Appendix A), we can choose any gauge. For example, one convenient choice is the Lorenz gauge $\partial_{\mu}A^{\mu}=0$. When discretized, the Lorenz gauge condition allows time advance $A_{s}^{n-1/2}\rightarrow A_{s}^{n+1/2}$ in a very simple way [Eq. (B), Fig. 2(d)]. Another convenient choice is the temporal gauge $A^{0}=0$. When discretized, $A_{s}^{n+1/2}$ remains zero on all time-like edges.

![images/image2.jpg](images/image2.jpg)
Figure 2: Coupling pattern of $\phi_{v}$ (blue squares), $A_{e}$ (red circles) and $F_{f}$ (green crosses) in the $tx$-submanifold. (a) The discretized KG equation [Eq. (16)] couples $\phi_{v}$ with its nearest neighbors though $A_{e}$. (b) The discretized Gauss’ law [Eq. (17)] couples $E_{i-1/2}$ and $E_{i+1/2}$ through $\phi_{v}$ on the shared vertexes. (c) The Maxwell-Ampère’s law [Eq. (19)] couples $E^{n+1/2}$ to $E^{n-1/2}$ through $\phi_{v}$ on the shared vertexes, and through $B^{n}$ (not depicted here) who shares the common space-like edge. (d) The Lorenz gauge condition couples $A_{e}$ that shares the same vertex.

Having obtained discrete equations and fixed the gauge, an explicit time advance scheme can be constructed (Appendix B). We first initialize the simulation by giving values of $\phi_{s}^{n}$ at both $n=0$ and $n=1$ for every spatial lattice points $s$. This is necessary because the KG equation [Eq. (1)] is a second order partial differential equation and therefore needs two initial conditions. Similarly, we need to give initial values of $A_{e}$ at $n=0$ and $n=1/2$. Second, we use the discrete Gauss’ Law to calculate $A_{e}$ at $n=1$ by solving a system of linear equations [Eq. (B), Fig. 2b]. Notice that the continuous version of this equation can be written as $\partial_{t}\nabla\cdot\mathbf{A}=-\nabla^{2}A^{0}-\rho$, where the RHS is known. Because the unknowns on the left-hand side (LHS) involve only first order spatial derivative, the discrete Gauss’ law couples less number of points than the discrete Poisson’s equation, and is therefore easier to solve. Third, we use the Lorenz gauge condition to advance $(A^{n-1/2},A^{n})\rightarrow A^{n+1/2}$ [Eq. (B), Fig. 2d]. Fourth, we use the discrete KG equation to calculate $\phi^{n+1}$ in terms of $\phi^{n}$ and $\phi^{n-1}$ [Eq. (B), Fig. 2a]. This involves exponentiation of $A^{n}$ and $A^{n+1/2}$, whose values are already known at this point. Simultaneously, we can compute $A^{n+1}$ in terms of $A_{e}$ at previous time steps [Eq. (B), Fig. 2c], using the discrete Maxwell-Ampère’s law. Notice that the discrete Gauss’ Law is preserved during time advance, which is a consequence of the discrete local charge conservation law [Eq. (A)]. Finally, having computed both $\phi_{v}$ and $A_{e}$ at $t=n+1$, we can move forward in the time loop by updating $n\rightarrow n+1$, with proper boundary conditions supplied. In similar fashion, explicit time advance schemes can be constructed when other gauge conditions are used.

![images/image3.jpg](images/image3.jpg)
Figure 3: Time evolution scheme for discrete KGM equations using the Lorenz gauge. As initial conditions, the values of $\phi_{v}(n=0)$ and $\phi_{v}(n=1)$ are given (blue squares), so are $A_{e}(n=0)$ and $A_{e}(n=1/2)$ (red circles). Then the Gauss’ Law [Eq. (17)] is used to calculate $A_{e}(n=1)$. On entering the time loop, the first step is to calculate $A^{n+1/2}$ using the Lorenz gauge condition. The second step is to use the KG equation [Eq. (16)] to calculate $\phi^{n+1}$, and independently, use the Maxwell-Ampère’s law [Eq. (19)] to calculate $A^{n+1}$. The time loop is advanced by $n\to n+1$ and repeat.

## III Numerical examples

In this section, we demonstrate our numerical scheme using two examples. The first example is the propagation of linear waves, and the second example is laser-plasma interaction in one spatial dimensional (1D).

### III.1 Linear wave spectra

To test our code implementation, we compare numerical spectra with analytical linear wave dispersion relations [37; 38; 42; 43]. For small-amplitude waves, the dispersion relation constrains the wave frequency $\omega$ as function of the wave vector $\mathbf{k}$. In unmagnetized cold scalar-QED plasma, the tree-level dispersion relation of the transverse $EM$ wave is

$\omega^{2}=\omega_{p}^{2}+\mathbf{k}^{2},$ (21)

where $\omega_{p}^{2}=\sum_{\zeta}\omega_{p\zeta}^{2}$ is the total plasma frequency, and $\omega_{p\zeta}^{2}=q_{\zeta}^{2}n_{\zeta 0}/m_{\zeta}$ is the plasma frequency of individual charged species $\zeta$. The other eigenmode is the longitudinal electrostatic wave, whose dispersion relation is

$1+\chi_{p}=0,$ (22)

where the cold plasma susceptibility

$\chi_{p}=\sum_{\zeta}\frac{\omega_{p\zeta}^{2}(\mathbf{k}^{2}-\omega^{2}+4m_{\zeta}^{2})}{(\omega^{2}-\mathbf{k}^{2})^{2}-4m_{\zeta}^{2}\omega^{2}}.$ (23)

The dispersion relation of electrostatic wave contains three branches. The gapless branch is the acoustic wave, the low frequency branch is the Langmuir mode, and the high frequency branch is the pair mode. While acoustic mode and Langmuir mode exist in classical plasmas, the pair mode only exists in relativistic-quantum plasmas [44]. The pair mode can be excited when gamma photons ($\omega&gt;2m$) inelastically scatter in high density plasmas, creating longitudinal oscillations in which virtual pairs are created and annihilated to carry the wave quanta.

We compute the numerical spectra in a single species plasma, in which immobile ions serve as homogeneous neutralizing background. To initialize the simulation so that a broad spectrum of linear waves are excited, the initial values of $A_{e}$ are given using small amplitude white noise with mean $\mu(A_{e})=0$ and standard deviation $\sigma(A_{e})=10^{-4}m$. Assuming the charged field is initially free, then its initial conditions can be given using the free field expansion $\phi(x)=\int d^{3}\mathbf{p}[a_{\mathbf{p}}\exp(-ipx)+b_{\mathbf{p}}^{\dagger}\exp(ipx)](2\pi)^{-3}(2E_{\mathbf{p}})^{-1/2}$, where $px=E_{\mathbf{p}}t-\mathbf{p}\cdot\mathbf{x}$ is Minkowski inner product, and $E_{\mathbf{p}}=\sqrt{\mathbf{p}^{2}+m^{2}}$ is the relativistic energy corresponding to momentum $\mathbf{p}$. From the above expansion, the momentum space distribution functions for particles and antiparticles are $f_{a}(\mathbf{p})=a_{\mathbf{p}}^{\dagger}a_{\mathbf{p}}$ and the $f_{b}(\mathbf{p})=b_{\mathbf{p}}^{\dagger}b_{\mathbf{p}}$, respectively. Con

![images/image4.jpg](images/image4.jpg)
Figure 4: Power spectra (color) of the transverse electric field $E_{y}$ (a) and the longitudinal electric field $E_{x}$ (b) are well-traced by tree-level dispersion relations (black lines) up to the grid resolution. The power spectra are averaged over an ensemble of 100 simulations with statistically equivalent initial conditions. In these simulations, immobile ion background is homogeneous. The charge $q=0.3$, such that the fine structure constant $q^{2}/4\pi\approx 1/137$ is physical. The unperturbed background plasma density is extremely high, such that the plasma frequency $\omega_{p}=0.85m$ can be shown on the same scale as $m$. The resolution $m\Delta x=0.04$ and $m\Delta t=0.02$. The number of spatial grid point is $L=512$, and the total number of time steps, including the initial conditions, is $T=1024$. The dashed gray lines is the light cone.

sider the simple example where the plasma is initially homogeneous and constituted of cold particles, namely, $f_{a}(\mathbf{p})=n_{0}\delta^{(3)}(\mathbf{p})$ and $f_{b}(\mathbf{p})=0$, where $n_{0}$ is the background plasma density. Then, the free charged field $\phi(x)=\sqrt{n_{0}/2m}\exp(-imt+i\alpha)$, where $\alpha$ is some random phase. When discretized, this free field corresponds to the initial conditions $\phi_{v}^{0}=\sqrt{n_{0}/2m}\exp(i\alpha)$ and $\phi_{v}^{1}=\phi_{s}^{0}\exp(-im\Delta t)$. An ensemble of statistically equivalent initial conditions can then be constructed by randomly sample the phase $\alpha$ and the gauge field.

After advancing the initial conditions in time using periodic boundary conditions, numerical spectra can be read out from simulations by taking discrete Fourier transforms of electric field components. Since the unmagnetized plasma is isotropic, it is sufficient to read out the dispersion relation in the tx-submanifold. In this submanifold, the spectra of either $E_{y}$ or $E_{z}$ correspond to the dispersion relation of transverse EM modes, and the spectrum of $E_{x}$ corresponds to the dispersion relation of longitudinal electrostatic modes. The ensemble-averaged power spectrum of $E_{y}$ [Fig. 4(a)] is indistinguishable from that of $E_{z}$, and is well-traced by the analytical dispersion relation (black line) of the transverse EM wave [Eq. (21)], until $k\Delta x\sim 1$ where the spatial resolution is no longer sufficient. Similarly, the ensemble-averaged power spectrum of $E_{x}$ [Fig. 4(b)] is localized near three bands, corresponding to the cold acoustic mode, the Langmuir mode and the pair mode [Eq. (22)]. That the analytical dispersion relations are recovered by numerical spectra indicates that our solutions faithfully capture the propagation of linear waves up to the grid resolution.

### III.2 Laser-plasma interaction

Having verified our code implementation, let us study laser-plasma interaction as another example, which can no longer be described self-consistently under the classical framework once the laser wavelength becomes too short and the field strength becomes too large. Before discussing our simulations in this relativistic-quantum regime, it is helpful to recall what happens in the classical regime [45]. Classically, when the plasma slab is under-dense, namely when the laser frequency $\omega>\omega_{p}$, much of the laser will travel through the plasma slab, with some reflection and inverse Bremsstrahlung absorption. In an initially quiescent slab, the laser will propagate uneventfully, if its frequency stays away from the two-plasmon-decay resonance, and its intensity is not strong enough to grow instabilities within the pulse duration. Beyond nonlinear wave instabilities, when the laser field becomes relativistically strong, namely when $a=qE/m\omega\gtrsim 1$, the ponderomotive force of a short laser pulse can expels a significant fraction of plasma electrons and form wakefield [46]. The wakefield can then accelerate particles, generating energetic beams of particles and radiations trailing the laser pulse. When the beams are energetic enough, they may produce gamma photons through synchrotron radiation or Bremsstrahlung. The virtual gamma photons may then decay into electron-positron pairs through the trident process [47]. Alternatively, the on-shell gamma photons may interact with ion potentials and produce pairs through the Breit-Wheeler process [48]. Finally, when the laser field becomes even stronger, namely when $qE/m^{2}\gtrsim 1$, pairs may also be produced directly through the Schwinger process [49].

Many aspects of laser-plasma interaction can be studied using our new numerical tool. Here, to validate that our code can capture genuine relativistic-quantum effects, we select parameters in our 1D simulations to demonstrate transition from wakefield acceleration to Schwinger pair production as we increase the laser intensity. Notice that in 1D, the phase space is highly constrained. Using periodic boundary conditions in directions transverse to laser propagation, Schwinger pair production by laser fields is suppressed. This is because when transverse fields try to pull $e^{-}/e^{+}$ pairs apart, their wave functions are enforced to be the same by the periodic boundary condition, which prevents pairs from emerging out of vacuum fluctuations. Therefore, in 1D simulations, Schwinger pair production requires longitudinal field $E_{x}$. To generate $E_{x}$ beyond the Schwinger field $E_{c}=m^{2}/q$ through wakefield, the plasma density must be extremely high. Heuristically, to produce on-shell pairs, the critical electric field needs to separate the pair by Compton wavelength $1/m$ within the Compton time $T\sim\pi/m$, namely, $qE_{x}T^{2}/m\gtrsim 1/m$. In the wave-breaking regime, $E_{x}\simeq am\omega_{p}/q$, so the inequality requires that the plasma density be high enough such that the plasma frequency $\omega_{p}/m\gtrsim 1/a\pi^{2}$. In reality, at those densities, we should treat the electron Fermi degeneracy to capture the full physical effects. However, simulating instead a high-density bosonic plasma is just a toy model that tests our code, with the density picked so high that we can already see laser Schwinger pair production in 1D simulations.

With this basic understanding of how laser pair production happen in 1D, we choose parameters to suppress the trident and Breit-Wheeler processes, by treating ions as immobile homogeneous neutralizing background, so that there is no spiky ion potentials from which energetic “electrons” and gamma photons can scatter. The smooth ion background provides an electrostatic potential that initially confines the “electrons”. We initialize the charged boson wave function according to $\phi(x)=\sqrt{n_{0}(x)/2m}\exp(-imt)$, where $n_{0}(x)$ is the background ion density with a plateau of width $L\approx 100/m$ and Gaussian off-ramps with $\sigma=20/m$. For density of the bosonic plasma to be high enough to enable pair production, we take $n_{0}=m^{3}$ so that the plasma frequency $\omega_{p}=0.3m$ is enormous. The above wave function is a linear superposition of many eigenstates of the system. In our simulations, we let the wave function evolve to statistically stationary states through phase mixing, before we start to draw samples at random time intervals. The sampled wave functions are then used as initial conditions for $\phi_{v}$, which are combined with initial values $A_{e}$ of

![images/image5.jpg](images/image5.jpg)
FIG. 5. Charge density (a, b) and energy density (c, d) of the  $\phi$  field. When the gamma-ray laser  $(\omega_0 = 0.7m)$  is relativistic  $(a \approx 1)$ , but not strong enough to produce Schwinger pairs  $(E_x \approx 0.3E_c)$ , "electrons" are expelled by the laser ponderomotive force, accelerated by the wakefield, and splashed from the plasma boundaries (a, c). When the laser field exceeds the Schwinger threshold  $(a \approx 16, E_x \approx 5E_c)$ , copious pairs are produced when laser interacts with plasma waves (b, d). The spin-0 "electrons" are initially confined by a smooth immobile neutralizing background, with a density plateau  $n_0 = m^3$  and a Gaussian off-ramp  $\sigma = 20/m$ . The trajectories of the pulse center (black lines) and the pulse half widths (dashed lines) are well traced by geometric optics. Both the charge density (normalized by  $em^3$ ) and the energy density (normalized by  $m^4$ ) are averaged over an ensemble of size 200. The resolutions are such that  $m\Delta x = 0.04$  and  $m\Delta t = 0.005$ .

a Gaussian pulse to construct an ensemble. The linearly-polarized Gaussian pulse is initialized in the vacuum region with zero carrier phase  $A_{y} \propto \exp(-\xi^{2}/2\tau^{2})\cos\omega\xi$ , where  $\xi = x - t$  and  $\tau = 20/m$ . For the laser to be able to transmit the high-density plasma slab, we use a gamma-ray laser with frequency  $\omega_{0} = 0.7m$ , for which semiclassical treatments are far from valid. The laser envelope is slowly varying  $(\omega_{0}\tau = 14)$ , and has full width at half maximum about twice the plasma skin depth. When intense laser pulse propagates, it can excite plasma waves, from which the laser can be Raman scattered.

With the above setup, the laser pulse simply travels through the plasma with some refraction and reflections when the laser field is weak ( $a \ll 1$ ). More interesting phenomena happen when the laser field becomes strong. For example, when  $a \approx 1$  is relativistically strong but the resulting  $E_{x} \approx 0.3E_{c}$  is below the Schwinger field, our simulation recovers what happens in classical plasmas [50-52]. First, let us look at what happens to charged particles. After the laser enters the plasma, beams of "electrons" are formed in the forward direction by both ponderomotive snow-plow and laser wakefield acceleration. At the same time, some "electrons" are splashed in the backward direction from strongly-driven plasma boundaries (Fig. 5a, c). Next, for the laser pulse, its center (solid black lines) and half widths (dotted black lines) are well-traced by geometric optics in the  $xt$ -space (Fig. 6a), as well as in the  $kt$ -space (Fig. 6c, dashed white line), because the background plasma is smooth on the

laser wavelength scale. Beyond geometric optics, as the laser travels through the plasma slab, ponderomotive expulsion of "electrons" cause the laser pulse to adiabatically lose a small amount of energy in the form of frequency redshift  $\omega &lt; \omega_0$  (Figs. 6a,c and 7b). In addition, the laser excites plasma waves, from which the laser is Raman-scattered in both forward and backward directions. In the insert of Fig. 6c, the final spectrum (red) shows distinctive Raman scattering peaks at  $\omega + n\omega_p$  up to  $n = 8$ , and second harmonics peaks  $2\omega$  and  $2\omega + \omega_p$  in the forward direction. In the backward direction, peaks at  $\omega - \omega_p, \omega, \omega + \omega_p$  and  $2\omega$  can also be identified unambiguously.

When we increase the laser field beyond the Schwinger threshold  $(a_{c} = m / \omega)$ . For example, when  $a\approx 16$ $(E_{x}\approx 5E_{c})$ , a large amount of  $e^{-} / e^{+}$  pairs are produced (Figs. 5b, d). A very small fraction of pairs are produced and trapped in the laser wakefield, forming low-luminosity "electron" (negative charge density, blue) and "positron" (positive charge density, red) beams that leave the plasma slab from its right boundary. In contrast, a much larger fraction of pairs are produced when the backscattered  $EM$  wave, whose intensity is near the Schwinger threshold (Fig. 6b), interacts with forwardpropagating plasma waves. "Positrons" produced in this way form high-luminosity collimated beams, leav

![images/image6.jpg](images/image6.jpg)
FIG. 6. Total energy density of  $EM$  fields (a, b), and the power spectral density of its transverse components (c, d). The inserts show the initial (blue) and final (red) spectra of  $EM$  waves. When  $a \approx 1$  ( $E_x \approx 0.3E_c$ ) is below the Schwinger field, the laser excites plasma waves and is Raman scattered (a, c). The time evolution of the main pulse is well-traced by geometric optics (dashed lines). When the laser field  $a \approx 16$  ( $E_x \approx 5E_c$ ) is above the Schwinger field, a noticeable amount of energy is lost due to pair production (b), and the  $k$ -spectrum is substantially broadened (d). The field energy density is normalized by the Schwinger field  $E_c^2$ , and are averaged over an ensemble of size 200. The resolutions are such that  $m\Delta x = 0.04$  and  $m\Delta t = 0.005$ . The dotted gray lines mark where the geometric-optics trajectory of the pulse center crosses the plasma plateau boundaries.

ing the plasma slab from its left boundary. Apart from these beams, many “positrons” never manage to leave the plasma slab. These trapped “positrons” have large probabilities to annihilate with “electrons” in the highly constrained 1D phase space. Due to pair creation and particle acceleration, the laser initially looses a significant amount of energy, until pair creation and annihilation roughly balance (Figs. 6b,c and 7b). At that point, the $k$-spectrum of the laser is substantially broadened (Fig. 6d). Such a spectral broadening is expected from general wave action considerations [53; 54], which predict frequency up-shift due to pair creation, and frequency down-shift due to pair annihilation and plasma expulsion. In the insert of Fig. 6d, the final EM wave spectrum (red) shows distinctive annihilation bumps near integer multiples of “electron” rest mass. These annihilation peaks are very broad since “electrons” and “positrons” annihilate with large kinetic energy. Finally, notice that no pair is produced when the laser travels through the vacuum region, which is expected in 1D. It is remarkable that very rich physics can already be captured by simply solving the classical field equations with proper initial conditions.

To extract observables from simulations, the charge density (Figs. 5a, b) is computed using Eq. (18), which includes no contribution from background ions. Therefore, negative charge (blue) indicates “electron” density in excess of “positron” density, whereas positive charge (red) indicates the contrary. The energy density of the charged field (Figs. 5c, d) and the EM fields (Figs. 6a, b) are computed using Eqs. (12) and (13), respectively. To compute $k$-spectra of EM waves (Figs. 6c, d), notice that a monochromatic EM wave satisfies $k_{x}E_{y}=\omega B_{z}$. Upon discretization, this relation remains exactly satisfied if we take $k_{x}=\sin(k\Delta x)/\Delta x$ and $\omega=2\tan(\omega_{k}\Delta t/2)/\Delta t$, where $\omega_{k}>0$ is the positive solution of the local numerical dispersion relation $4\sin^{2}(\omega_{k}\Delta t/2)/\Delta t^{2}=4\sin^{2}(k\Delta x/2)/\Delta x^{2}$. In the discrete version of $k_{x}E_{y}=\omega B_{z}$, it is necessary that we take $E_{y}=E_{s+j/2}^{n+1/2}$, and center $B_{z}$ on time-like faces $B_{r-k/2-i/2}^{n+1/2}=(B_{r-k/2}^{n}+B_{r-k/2-i}^{n}+B_{r-k/2}^{n+1}+B_{r-k/2-i}^{n+1})/4$. A similar relation holds for the $E_{z}$ and $B_{y}$ components, which are subdominant in our simulations. Using these momentum-space Faraday’s law, the $k$-spectrum of right-propagating EM waves ($k>0$) and left-propagating EM waves ($k<0$) can be separated from the spatial Fourier transforms of electric and magnetic fields.

Results presented in Figs. 5-7 are averaged over an ensemble of 200 simulations with statistically equivalent initial conditions. The ensemble average starts to show convergence for tens of realizations. In these simulations, temporal gauge $A^{0}=0$ is used, and periodic boundary conditions are employed for both $\phi_{v}$ and $A_{e}$. We choose resolutions $mdx=0.04$ and $mdt=0.005$, high enough that the fastest dynamics is resolved and the simulation results converge. The 1D box is large enough such that the laser does not transit the spatial domain before we terminate the simulations.

## IV Discussion and Summary

In this paper, we develop an algorithm for solving the Klein-Gordon-Maxwell’s equations [Eqs. (1) and (2)], which can be used to model bosonic plasmas in the relativistic-quantum regime. This algorithm is derived by first discretizing the action [Eq. (14)] in a way that respects the U(1)-gauge symmetry. We then take variations with respect to the discrete fields to find their classical equations of motion. The resultant variational algorithm guarantees that the Bianchi identities, namely, $\nabla\cdot\mathbf{B}=0$ and the Faraday’s law, are automatically and exactly satisfied. The remaining equations of motions are the discrete Gauss’s law [Eq. (17)], which can be used to initialize the simulation; the discrete Klein-Gordon’s equation [Eq. (16)], which can be used to advance the charged field; and the discrete Maxwell-Ampère’s law [Eq. (19)], which can be used to advance the gauge field. After fixing a gauge, explicit scheme for advancing the discrete fields in time can be constructed (Appendix B, Fig. 3). Our variational scheme respects local conservation laws (Appendix A), and can be easily parallelized using domain decomposition. Moreover, the numerical scheme we have developed can be inherently mimicked by quantum systems with local couplings [55; 56], which can be efficiently realized using quantum parallelism [57; 58].

The numerical scheme we have developed has a number of advantages over conventional methods for simulating plasmas. As comparison, the two conventional methods that can fully simulate kinetic effects are the particle-in-cell (PIC) solvers and the Vlasov solvers. The PIC solvers represent particles in the continuum, while representing EM fields on grids. Particles feel EM fields through interpolations, and EM fields feel particles through depositions. Using proper smoothing functions, these two steps can preserve gauge symmetry and symplectic structures, thereby respect local conservation properties when used in geometrical algorithms [59; 60; 61; 62]. Nevertheless, interpolation and deposition introduce artificial collisions, which are absent in physical systems. In the other scheme, the Vlasov solvers, EM fields are represented on the three-dimensional space, while particles are represented in the six-dimensional phase space. Particles are directly forced by fields on spatial grids, while the fields feel particles though velocity space integrals, which requires resolving three extra dimensions with substantial computational cost. In contrast, our algorithm represents both particles and EM fields on the same grid. Therefore, there is no need for interpolations and depositions as in the case of PIC solvers, nor is there need for resolving extra velocity space dimensions as in the case of Vlasov solvers. Our algorithm folds the phase space dynamics of charged particles into the complex plane, and enables modeling of relativistic and quantum dynamics in regimes that cannot be described using semiclassical treatments. In the example of linear waves (Sec. III.A), we show that relativistic-quantum wave dispersion relations can be recovered. Moreover, using the example of a gamma-ray laser interacting with a dense plasma slab (Sec. III.B),

we show that our algorithm naturally allows pair production when the laser intensity exceeds the Schwinger threshold.

Of course, the advantages of our algorithm come at an expense. The expense comes from the necessity of resolving the relativistic-quantum scales, which are much smaller than scales that classical plasma physics typically deals with. The coarsest resolution needed in relativistic quantum plasma simulations is determined by the lowest energy scale of the problem, which is the rest mass of electrons $\sim 0.5$ MeV, corresponding to time scale of $\sim 10^{-21}$ s, and spatial scale of $\sim 10^{-13}$ m. This resolution requirement can be seen from the discrete KG equation [Eq. (B3)], in which we must have $m\Delta t\ll 1$ in order for $\delta\phi_{v}\ll\phi_{v}$. Moreover, since we are solving a system of hyperbolic partial differential equations, the Courant–Friedrichs–Lewy (CFL) condition $\Delta t&lt;\Delta x$ must be satisfied. Finally, it is worth noting that high resolution is required for large gauge fields. Since the gauge field appears through the Wilson’s lines in complex exponentials [Eqs. (10) and (11)], the discrete theory is invariant under the gauge transformation $A_{e}\rightarrow A_{e}+2\pi/(q\Delta)$. Therefore, the discrete gauge field $A_{e}$ lives on the torus $\mathbb{T}^{1,3}$, which has a very different topology than $\mathbb{R}^{1,3}$. Therefore, the step size must be small enough such that $qA\Delta&lt;2\pi$, in order to avoid exciting topological modes that are absent in the continuous theory.

In summary, we develop a variational algorithm for solving the Klein-Gordon-Maxwell equations, which described tree-level scalar QED in the classical-statistical regime and may be solved using quantum computers. We demonstrate that remarkably rich physics are contained in solutions to classical field equations, which can be used to model high-density plasmas interacting with short-wavelength electromagnetic fields. Our work uses scalar QED as a toy model to make the case that real-time lattice QED is a powerful tool for plasma physics in the strong-field regime, where relativistic-quantum scales overlap with collective plasma scales. The applications of lattice spinor-QED to laboratory relevant plasma conditions remain to be demonstrated in the future.

The authors are grateful to Qun Wang and Sebastian Meuren for helpful discussions. This rsearch is supported by NNSA Grant No. DE-NA0002948 and DOE Research Grant No. DEAC02-09CH11466. Jianyuan Xiao is supported by National Magnetic Confinement Fusion Energy Research Project (2015GB111003, 2014GB124005), National Natural Science Foundation of China (NSFC-11575185, 11575186, 11305171), JSPS-NRF-NSFC A3 Foresight Program (NSFC-11261140328), Chinese Scholar Council (201506340103), Key Research Program of Frontier Sciences CAS (QYZDB-SSW-SYS004), and the GeoAlgorithmic Plasma Simulator (GAPS) Project.

## Acknowledgments

The authors are grateful to Qun Wang and Sebastian Meuren for helpful discussions. This rsearch is supported by NNSA Grant No. DE-NA0002948 and DOE Research Grant No. DEAC02-09CH11466. Jianyuan Xiao is supported by National Magnetic Confinement Fusion Energy Research Project (2015GB111003, 2014GB124005), National Natural Science Foundation of China (NSFC-11575185, 11575186, 11305171), JSPS-NRF-NSFC A3 Foresight Program (NSFC-11261140328), Chinese Scholar Council (201506340103), Key Research Program of Frontier Sciences CAS (QYZDB-SSW-SYS004), and the GeoAlgorithmic Plasma Simulator (GAPS) Project.

## References

- [1] ![images/image7.jpg](images/image7.jpg)
Figure 7: Evolution of total charge (a) and total energy (b) in the numerical example discussed in Sec. III.B, where periodic boundary conditions are used. The total charge is constant in time up to the machine precision, both when $E<e_{c}$ $e="" (cyan)="" and="">E_{c}$ (blue). When $E&lt;E_{c}$ is below the Schwinger field, a small amount of energy is transfered from the electromagnetic field (magenta) to the charged field (cyan) due to wakefield acceleration and plasma wave excitation, while the total energy (gray) remains constant. In contrast, when $E&gt;E_{c}$, a large amount of laser energy (red) is consumed by pair production. The energy of the charged field (blue) significantly increases, while the total energy (black) remains constant. The total charge $Q^{n+1/2}=\sum_{s}J_{s}^{n+1/2}$ is normalized by the total ion charge, and the total energy $\mathcal{U}^{n+1/2}=\sum_{s}\mathcal{H}_{s}^{n+1/2}$ is normalized by $m^{3}/\Delta x$. The vertical dashed gray lines mark the time when the laser pulse center enters and leaves the plasma plateau boundaries.

## Appendix A Geometric identities and local conservation laws

When discretizing the gauge 1-form $A$ and calculating the curvature 2-form $F=dA$ in Sec. II.A, geometric structures of discrete exterior calculus are respected. Consequently, the identity $d^{2}=0$ holds for the discrete exterior derivative. In components, this Bianchi identity can be written as $0=dF=(\partial_{\sigma}F_{\mu\nu}+\partial_{\mu}F_{\nu\sigma}+\partial_{\nu}F_{\sigma\mu})dx^{\mu}\wedge dx^{\nu}\wedge dx^{\sigma}/3!$. One nontrivial identity, corresponding to all indexes being spatial, is $\nabla\cdot\mathbf{B}=0$. When discretized, this identity becomes

$\frac{1}{\Delta_{l}}\Big{(}B_{r+\frac{1}{2}}^{n}-B_{r-\frac{1}{2}}^{n}\Big{)}=0.$ (10)

The other nontrivial identity, corresponding to two spatial indexes and one temporal index, is the Faraday’s law $\partial_{t}\mathbf{B}=-\nabla\times\mathbf{E}$, whose discrete version is

$\frac{1}{\Delta t}\Big{(}B_{r-\frac{1}{2}}^{n+1}-B_{r-\frac{1}{2}}^{n}\Big{)}=\frac{\epsilon_{ijk}}{\Delta_{k}}\Big{(}E_{s+\frac{1}{2}+k}^{n+\frac{1}{2}}-E_{s+\frac{1}{2}}^{n+\frac{1}{2}}\Big{)}.$ (11)

The above finite difference equations are automatically satisfied in our algorithm by geometric constructions, in contrast to standard elecromagnetic algorithms, such as the Yee’s algorithm [41], in which the Faraday’s law needs to be solved as a dynamical equation.

In addition to the above geometric identities, we also have a number of local conservation laws. The first is</e_{c}$>

local charge conservation, which is a direct consequence of local U(1)-gauge symmetry. Under the continuous U(1)-gauge transformation

$\phi_{s}^{n}$ $\rightarrow\phi_{s}^{n}e^{iq\alpha_{s}^{n}},$ (10)
$A_{s}^{n+\frac{1}{2}}$ $\rightarrow A_{s}^{n+\frac{1}{2}}+\frac{1}{\Delta t}(\alpha_{s}^{n+1}-\alpha_{s}^{n}),$ (11)
$A_{s+\frac{l}{2}}^{n}$ $\rightarrow A_{s+\frac{l}{2}}^{n}+\frac{1}{\Delta_{l}}(\alpha_{s+l}^{n}-\alpha_{s}^{n}),$ (12)

where $\alpha_{s}^{n}$ is any real-valued function living on vertexes. These transformations leave the discrete face-centered field strength tensor $F_{f}$ invariant, while transforming the pull-back covariant derivative by

$(D_{\mu}^{<}\phi)_{s}^{n}\rightarrow e^{iq\alpha_{s}^{n}}(D_{\mu}^{<}\phi)_{s}^{n}.$ (13)

Since the action $S_{d}$ is invariant, we can use the classical field equations $\delta S_{d}/\delta\phi_{v}=0$ and write

$\frac{\delta S_{d}}{\delta\phi_{s}^{n}}\delta\phi_{s}^{n}+c.c.=0.$ (14)

Substituting in the infinitesimal transformation $\delta\phi_{s}^{n}=iq\alpha_{s}^{n}\phi_{s}^{n}$, the above identity is equivalent to the discrete charge conservation law

$\frac{1}{\Delta t}\Big{(}J_{s}^{n+\frac{1}{2}}-J_{s}^{n-\frac{1}{2}}\Big{)}=\frac{1}{\Delta_{l}}\Big{(}J_{s+\frac{l}{2}}^{n}-J_{s-\frac{l}{2}}^{n}\Big{)}.$

Here, the charge density $J_{s}^{n+\frac{1}{2}}$ is given by Eq. (18), and the current density $J_{s+l/2}^{n}$ is given by Eq. (20), and the sign is due to the Minkowski metric. It is straightforward to check that the above discrete charge conservation law is compatible with the discrete Gauss’ law [Eq. (17)] and the discrete Maxwell-Ampère’s law [Eq. (19)]. In the numeric example discussed in Sec. III.2, the total charge $Q^{n+1/2}=\sum_{i}J_{i}^{n+1/2}$ is constant up to the machine precision (Fig. 7a), both when the laser field is below ($Q^{<}$) and above ($Q^{>}$) the Schwinger field.

Moreover, the discrete action $S_{d}$ is invariant under translations on the discrete spacetime manifold. Although the symmetry group in this case is discrete and hence the Noether’s theorem does not immediately apply, we do have local energy conservation laws for the charged field and EM fields separately when their coupling vanishes. Using the classical field equations $\delta S_{d}/\delta\phi_{v}=0$ and $\delta S_{d}/\delta A_{s+l/2}^{n}=0$, as well as the Bianchi identity, we have the following identity

$0$ $=\frac{\delta S_{d}}{\delta\phi_{s}^{n}}(\mathcal{D}_{0}\phi)_{s}^{n}+\frac{\delta S_{d}}{\delta\phi_{s}^{n}}(\overline{\mathcal{D}_{0}\phi})_{s}^{n}$ (15)
$+\frac{\delta S_{d}}{\delta A_{s+l/2}^{n}}\frac{1}{2}\Big{(}E_{s+l/2}^{n+1/2}+E_{s+l/2}^{n-1/2}\Big{)}$
$+B_{r-l/2}^{n}\frac{1}{2}\Big{[}(d^{2}A)_{r-l/2}^{n+1/2}+(d^{2}A)_{r-l/2}^{n-1/2}\Big{]},$

where the vertex-centered time covariant derivative

$(\mathcal{D}_{0}\phi)_{s}^{n}=\frac{e^{-iq\Delta tA_{s}^{n+\frac{1}{2}}}\phi_{s}^{n+1}-e^{iq\Delta tA_{s}^{n-\frac{1}{2}}}\phi_{s}^{n-1}}{2\Delta t}.$ (16)

After rearranging terms, the above identity gives rise to the local energy conservation law

$\frac{\mathcal{H}_{s}^{n+1/2}-\mathcal{H}_{s}^{n-1/2}}{\Delta t}=\frac{\mathcal{P}_{s+l/2}^{n}-\mathcal{P}_{s-l/2}^{n}}{\Delta_{l}}+\mathcal{O}(q\Delta^{2}),$ (17)

where the sign is again due to the Minkowski metric. The energy density can be separated into three terms

$\mathcal{H}_{s}^{n+1/2}=\mathcal{H}_{s}^{n+1/2}[\phi]+\mathcal{H}_{s}^{n+1/2}[A]+h_{s}^{n+1/2},$ (18)

where the energy density of the charged field is

$\mathcal{H}_{s}^{n+\frac{1}{2}}[\phi]$ $=\frac{1}{2}\Big{[}(D_{0}^{<}\phi)_{s}^{n+\frac{1}{2}}(\overline{D_{0}^{<}\phi})_{s}^{n+\frac{1}{2}}+m^{2}\phi_{s}^{n}e^{iq\Delta tA_{s}^{n+\frac{1}{2}}}\tilde{\phi}_{s}^{n+1}$
$+(D_{l}^{<}\phi)_{s+\frac{l}{2}}^{n}e^{iq\Delta tA_{c}^{n+\frac{1}{2}}}(\overline{D_{l}^{<}\phi})_{s+\frac{l}{2}}^{n+1}\Big{]}+\text{c.c.},$ (19)

and the energy density of the EM fields is

$\mathcal{H}_{s}^{n+\frac{1}{2}}[A]=\frac{1}{2}\Big{[}\big{(}E_{s+\frac{l}{2}}^{n+\frac{1}{2}}\big{)}^{2}+B_{r-\frac{l}{2}}^{n+1}B_{r-\frac{l}{2}}^{n}\Big{]}.$ (20)

The energy density correction $h=\mathcal{O}(q\Delta^{2})$ can take many different forms, each has a corresponding error term at finite-resolution. As expected, the energy density is U(1)-gauge invariant, so is the momentum density, which can be split into two terms

$\mathcal{P}_{s+l/2}^{n}=\mathcal{P}_{s+l/2}^{n}[\phi]+\mathcal{P}_{s+l/2}^{n}[A].$ (21)

The momentum density of the charged field is

$\mathcal{P}_{s+\frac{l}{2}}^{n}[\phi]=(D_{l}^{<}\phi)_{s+\frac{l}{2}}^{n}e^{iq\Delta_{l}A_{s+\frac{l}{2}}^{n}(\overline{\mathcal{D}_{0}^{<}\phi})_{s+l}^{n}}+\text{c.c.},$ (22)

and the momentum density of the EM fields $\mathcal{P}_{i}=-\mathcal{P}^{i}=-(\mathbf{E}\times\mathbf{B})^{i}$ is

$\mathcal{P}_{s+\frac{1}{2}}^{n}[A]$ $=\epsilon_{ijk}B_{r-\frac{l}{2}}^{n}\frac{1}{2}\Big{(}E_{s+i+\frac{k}{2}}^{n+\frac{1}{2}}+E_{s+i+\frac{k}{2}}^{n-\frac{1}{2}}\Big{)}.$ (23)

Since the stress-energy tensor $\mathcal{T}^{\mu\nu}$ is not a 2-form, neither the energy density $\mathcal{H}$ nor the momentum density $\mathcal{P}$ is well-defined on the discrete spacetime manifold. Hence, it can be shown, by enumerating combinations of U(1)-gauge invariant basis terms, that the resulting error in the local energy conservation law [Eq. (17)] is always second order. Except when the coupling $q\rightarrow 0$, in which case the conservation law becomes exact even at finite spacetime resolutions. This remarkable feature would be lost if we had used the Wilsonian plaquettes in the discrete action instead. In examples discussed in Sec. III.2, the total energy $\mathcal{U}^{n+1/2}$ $=\sum_{i}\mathcal{H}_{i}^{n+1/2}$, whose error is of order $\mathcal{O}(qn\Delta t^{2})$, is redistributed among $\phi$ and $A$ (Fig. 7b). The total energy fluctuates up to 6 ppm and 0.2% when the laser wakefield is below ($\mathcal{U}^{<}$) and above ($\mathcal{U}^{>}$) the Schwinger field.

## Appendix B Numerical scheme

In this appendix, we list the four equations that are necessary for implementing the algorithm. We rewrite these

equations from Sec. II.B, such that the explicit nature of the algorithm becomes apparent. The first step in the simulation is initializing values of $\phi_{s}^{0},\phi_{s}^{1},A_{s+l/2}^{0}$, and $A_{s}^{1/2}$ for all spatial indexes. This step is crucial and determines what physical system will be evolved subsequently.

The second step is calculating $A_{s+l/2}^{1}$ using the discrete Gauss’ law [Eq. (17)], which can be rewritten as

$\frac{A_{s+l/2}^{1}-A_{s-l/2}^{1}}{\Delta_{l}}$ $=$ $\frac{A_{s+l/2}^{0}-A_{s-l/2}^{0}}{\Delta_{l}}+\Delta tJ_{s}^{1/2}$
$+$ $\frac{\Delta t}{\Delta l^{2}}\Big{(}A_{s+l}^{1/2}-2A_{s}^{1/2}+A_{s-l}^{1/2}\Big{)},$

where all terms on the RHS are known. Since the LHS couples only two adjacent $A_{s+l/2}^{1}$ in each direction, the discrete Gauss’ law is easier to solve than the Poisson’s equation, which couples three nearest neighbors in each direction.

The third step is advancing the time-component of the gauge field $(A_{s}^{n-1/2},A_{s+l/2}^{n})\rightarrow A_{s}^{n+1/2}$. This step depends on the choice of the gauge condition. For example, when Lorenz gauge is used

$A_{s}^{n+1/2}=A_{s}^{n-1/2}+C_{l}\Big{(}A_{s+l/2}^{n}-A_{s-l/2}^{n}\Big{)},$ (10)

where $C_{l}=\Delta t/\Delta_{l}$ is the dimensionless Courant number. In comparison, when temporal gauge is used instead, $A_{s}^{n+1/2}=0$ and the time advance is trivial. Using the temporal gauge, one only needs to store values of $A_{e}$ at integer time steps $t=n$. However, when a background electric field is present, $A_{s+l/2}^{n}$ will grow indefinitely in the temporal gauge. In this case, long-time dynamics may be more accurately computed using the Lorenz gauge.

In the fourth step, we can use the discrete KG equation [Eq. (16)] to time advance the charged field $(\phi_{s}^{n-1},\phi_{s}^{n};A_{s+l/2}^{n},A_{s}^{n\pm 1/2})\rightarrow\phi_{s}^{n+1}$. The explicit time advance is given by

$\phi_{s}^{n+1}$ $=$ $\Big{[}(2-2C_{l}^{2}-\Delta t^{2}m^{2})\phi_{s}^{n}-\phi_{s}^{n-1}e^{iq\Delta tA_{s}^{n-\frac{1}{2}}}$
$+$ $C_{l}^{2}\Big{(}\phi_{s+l}^{n}e^{-iq\Delta_{l}A_{s+\frac{1}{2}}^{n}}+\phi_{s-l}^{n}e^{iq\Delta_{l}A_{s-\frac{1}{2}}^{n}}\Big{)}\Big{]}e^{iq\Delta tA_{s}^{n+\frac{1}{2}}}.$

For free $\phi$ field, suppose the fluctuation is of the form $\exp(ip_{l}x^{l}-iEt)$, then the numerical dispersion relation of the massive particle is

$\frac{4}{\Delta t^{2}}\sin^{2}\frac{E\Delta t}{2}=\frac{4}{\Delta_{l}^{2}}\sin^{2}\frac{p_{l}\Delta_{l}}{2}+m^{2},$

which is consistent with the continuum energy-momentum relation $E^{2}=\mathbf{p}^{2}+m^{2}$ for relativistic particles. For the numerical solution to be stable, $E$ must be real, which holds only if for each $l=i,j,k$, the CFL condition $C_{l}<1$ is satisfied.

Finally, without relying on the fourth step, we can use the discrete Maxwell-Ampère’s law [Eq. (19)], simultaneously with the KG equation, to advance the spatial component of the gauge field $(A_{s+l/2}^{n-1},A_{s}^{n\pm 1/2},A_{s+l/2}^{n};\phi_{s}^{n})\rightarrow A_{s+l/2}^{n+1}$. The explicit time advance is given by

$A_{s+\frac{1}{2}}^{n+1}$ $=$ $A_{s+\frac{1}{2}}^{n}+C_{i}\Big{(}A_{s+i}^{n+\frac{1}{2}}-A_{s}^{n+\frac{1}{2}}\Big{)}+\Delta t^{2}J_{s+\frac{1}{2}}^{n}$ (11)
$+\Delta t\Big{[}E_{s+\frac{1}{2}}^{n-\frac{1}{2}}+\epsilon_{ijk}C_{j}\Big{(}B_{r-\frac{k}{2}}^{n}-B_{r-\frac{k}{2}-j}^{n}\Big{)}\Big{]},$

where the RHS is known. For free gauge field, it is straight forward to show that the numerical solution is stable if and only if the CFL condition $C_{l}<1$ is satisfied.

## References

- (1) D. Ruiz and I. Dodin, Phys. Lett. A 379, 2623 (2015).
- (2) E. P. Liang, S. C. Wilks, and M. Tabak, Phys. Rev. Lett. 81, 4887 (1998).
- (3) C. Gahn, G. Tsakiris, G. Pretzler, K. Witte, C. Delfin, C.-G. Wahlström, and D. Habs, Appl. Phys. Lett. 77, 2662 (2000).
- (4) E. Liang, T. Clarke, A. Henderson, W. Fu, W. Lo, D. Taylor, P. Chaguine, S. Zhou, Y. Hua, X. Cen, et al., Sci. Rep. 5 (2015).
- (5) G. Sarri, K. Poder, J. Cole, W. Schumaker, A. Di Piazza, B. Reville, T. Dzelzainis, D. Doria, L. Gizzi, G. Grittani, et al., Nat. Commun. 6 (2015).
- (6) V. I. Berezhiani, D. D. Tskhakaya, and P. K. Shukla, Phys. Rev. A 46, 6608 (1992).
- (7) Y. Kluger, E. Mottola, and J. M. Eisenberg, Phys. Rev. D 58, 125015 (1998).
- (8) S. Schmidt, D. Blaschke, G. Röpke, S. Smolyansky, A. Prozorkevich, and V. Toneev, Int. J. Mod. Phys. E 7, 709 (1998).
- (9) C. D. Roberts, S. M. Schmidt, and D. V. Vinnik, Phys. Rev. Lett. 89, 153901 (2002).
- (10) F. Hebenstreit, R. Alkofer, and H. Gies, Phys. Rev. D 82, 105026 (2010).
- (11) F. Hebenstreit, R. Alkofer, and H. Gies, Phys. Rev. D 78, 061701 (2008).
- (12) F. Hebenstreit, R. Alkofer, G. V. Dunne, and H. Gies, Phys. Rev. Lett. 102, 150404 (2009).
- (13) R. Duclous, J. G. Kirk, and A. R. Bell, Plasma Phys. Contr. F. 53, 015009 (2011).
- (14) E. N. Nerush, I. Y. Kostyukov, A. M. Fedotov, N. B. Narozhny, N. V. Elkina, and H. Ruhl, Phys. Rev. Lett. 106, 035001 (2011).
- (15) C. P. Ridgers, C. S. Brady, R. Duclous, J. G. Kirk, K. Bennett, T. D. Arber, A. P. L. Robinson, and A. R. Bell, Phys. Rev. Lett. 108, 165006 (2012).
- (16) R. Schützhold, H. Gies, and G. Dunne, Phys. Rev. Lett. 101, 130404 (2008).
- (17) K. G. Wilson, Phys. Rev. D 10, 2445 (1974).
- (18) S. A. Bass, M. Gyulassy, H. Stcker, and W. Greiner, J. Phys. G Nucl. Part. 25, R1 (1999).
- (19) H. Satz, Rep. Prog. Phys. 63, 1511 (2000).
- (20) M. Creutz, Phys. Rev. D 21, 2308 (1980).
- (21) K. Yagi, T. Hatsuda, and Y. Miake, Quark-gluon plasma: From big bang to little bang, Vol. 23 (Cambridge University Press, 2005).
- (22) J. Schwinger, J. Math. Phys. 2, 407 (1961).

- (23) L. V. Keldysh, Sov. Phys. JETP 20, 1018 (1965).
- (24) G. Aarts and J. Berges, Phys. Rev. Lett. 88, 041603 (2002).
- (25) A. Mueller and D. Son, Phys. Lett. B 582, 279 (2004).
- (26) S. Jeon, Phys. Rev. C 72, 014907 (2005).
- (27) J. Berges and T. Gasenzer, Phys. Rev. A 76, 033604 (2007).
- (28) J. Berges, K. Boguslavski, S. Schlichting, and R. Venugopalan, J. High Energy Phys. 2014, 54 (2014).
- (29) G. Aarts and J. Smit, Nucl. Phys. B 555, 355 (1999).
- (30) F. Gelis and N. Tanji, Phys. Rev. D 87, 125035 (2013).
- (31) A. Polkovnikov, Phys. Rev. A 68, 053604 (2003).
- (32) S. Borsanyi and M. Hindmarsh, Phys. Rev. D 79, 065010 (2009).
- (33) F. Hebenstreit, J. Berges, and D. Gelfand, Phys. Rev. D 87, 105006 (2013).
- (34) F. Hebenstreit, J. Berges, and D. Gelfand, Phys. Rev. Lett. 111, 201601 (2013).
- (35) V. Kasper, F. Hebenstreit, and J. Berges, Phys. Rev. D 90, 025016 (2014).
- (36) S. Chandrasekharan and U.-J. Wiese, Prog. Part. Nucl. Phys. 53, 373 (2004).
- (37) B. Eliasson and P. K. Shukla, Phys. Rev. E 83, 046407 (2011).
- (38) Y. Shi, N. J. Fisch, and H. Qin, Phys. Rev. A 94, 012124 (2016).
- (39) O. Klein, Z. Phys. A-Hadrons Nucl. 53, 157 (1929).
- (40) S. H. Christiansen and T. G. Halvorsen, IMA J. Numer. Anal. 31, 1 (2011).
- (41) K. Yee, IEEE T. Antenn. Propag. 14, 302 (1966).
- (42) D. Hines and N. Frabkel, Phys. Lett. A 69, 301 (1978).
- (43) V. Kowalenko, N. E. Frankel, and K. C. Hines, Phys. Rep. 126, 109 (1985).
- (44) M. G. Fuda and E. Furlani, Am. J. Phys. 50, 545 (1982).
- (45) W. L. Kruer, The physics of laser plasma interactions (Addison-Wesley Pub. Co. Inc., Reading, MA, 1988).
- (46) A. Pukhov and J. Meyer-ter Vehn, Appl. Phys. B-Lasers O. 74, 355 (2002).
- (47) J. D. Bjorken and M. C. Chen, Phys. Rev. 154, 1335 (1967).
- (48) G. Breit and J. A. Wheeler, Phys. Rev. 46, 1087 (1934).
- (49) J. Schwinger, Phys. Rev. 82, 664 (1951).
- (50) C. J. McKinstrie and E. A. Startsev, Phys. Rev. E 54, R1070 (1996).
- (51) N. M. Naumova, J. A. Nees, B. Hou, G. A. Mourou, and I. V. Sokolov, Opt. Lett. 29, 778 (2004).
- (52) V. I. Geyko, G. M. Fraiman, I. Y. Dodin, and N. J. Fisch, Phys. Rev. E 80, 036404 (2009).
- (53) S. C. Wilks, J. M. Dawson, and W. B. Mori, Phys. Rev. Lett. 61, 337 (1988).
- (54) I. Dodin and N. Fisch, Phys. Plasmas 17, 112113 (2010).
- (55) U.-J. Wiese, Ann. Phys. 525, 777 (2013).
- (56) E. A. Martinez, C. A. Muschik, P. Schindler, D. Nigg, A. Erhard, M. Heyl, P. Hauke, M. Dalmonte, T. Monz, P. Zoller, et al., Nature 534, 516 (2016).
- (57) R. P. Feynman, Found. Phys. 16, 507 (1986).
- (58) S. Lloyd, Science , 1073 (1996).
- (59) J. Squire, H. Qin, and W. M. Tang, Phys. Plasmas 19, 084501 (2012).
- (60) J. Xiao, J. Liu, H. Qin, and Z. Yu, Phys. Plasmas 20, 102517 (2013).
- (61) J. Xiao, H. Qin, J. Liu, Y. He, R. Zhang, and Y. Sun, Phys. Plasmas 22, 112504 (2015).
- (62) H. Qin, J. Liu, J. Xiao, R. Zhang, Y. He, Y. Wang, Y. Sun, J. W. Burby, L. Ellison, and Y. Zhou, Nucl. Fusion 56, 014001 (2015).