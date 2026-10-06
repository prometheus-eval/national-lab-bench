# Convolutional Dictionary Learning: A Comparative Review and New Algorithms

Cristina Garcia-Cardona and Brendt Wohlberg

###### Abstract

Convolutional sparse representations are a form of sparse representation with a dictionary that has a structure that is equivalent to convolution with a set of linear filters. While effective algorithms have recently been developed for the convolutional sparse coding problem, the corresponding dictionary learning problem is substantially more challenging. Furthermore, although a number of different approaches have been proposed, the absence of thorough comparisons between them makes it difficult to determine which of them represents the current state of the art. The present work both addresses this deficiency and proposes some new approaches that outperform existing ones in certain contexts. A thorough set of performance comparisons indicates a very wide range of performance differences among the existing and proposed methods, and clearly identifies those that are the most effective.

###### Index Terms:

Sparse Representation, Sparse Coding, Dictionary Learning, Convolutional Sparse Representation

## I Introduction

Sparse representations *[1]* have become one of the most widely used and successful models for inverse problems in signal processing, image processing, and computational imaging. The reconstruction of a signal $\mathbf{s}$ from a sparse representation $\mathbf{x}$ with respect to dictionary matrix $D$ is linear, i.e. $\mathbf{s}\approx D\mathbf{x}$, but computing the sparse representation given the signal, referred to as sparse coding, usually involves solving an optimization problem. When solving problems involving images of any significant size, these representations are typically independently applied to sets of overlapping image patches due to the intractability of learning an unstructured dictionary matrix $D$ mapping to a vector space with the dimensionality of the number of pixels in an entire image.

The convolutional form of sparse representations replaces the unstructured dictionary $D$ with a set of linear filters $\{\mathbf{d}_{m}\}$. In this case the reconstruction of $\mathbf{s}$ from representation $\{\mathbf{x}_{m}\}$ is $\mathbf{s}\approx\sum_{m}\mathbf{d}_{m}*\mathbf{x}_{m}$, where $\mathbf{s}$ can be an entire image instead of a small image patch. This form of representation was first introduced some time ago under the label translation-invariant sparse representations *[3]*, but has recently enjoyed a revival of interest as convolutional sparse representations, inspired by deconvolutional networks *[4]* (see *[5, Sec. II]*). This interest was spurred by the development of more efficient methods for the computationally-expensive convolutional sparse coding (CSC) problem *[6, 7, 8, 9]*, and has led to a number of applications in which the convolutional form provides state-of-the-art performance *[10, 11, 12, 13, 14, 15]*.

The current leading CSC algorithms *[8, 9, 16]* are all based on the Alternating Direction Method of Multipliers (ADMM) *[17]*, which decomposes the problem into two subproblems, one of which is solved by soft-thresholding, and the other having a very efficient non-iterative solution in the DFT domain *[8]*. The design of convolutional dictionary learning (CDL) algorithms is less straightforward. These algorithms adopt the usual approach for standard dictionary learning, alternating between a sparse coding step that updates the sparse representation of the training data given the current dictionary, and a dictionary update step that updates the current dictionary given the new sparse representation. It is the inherent computational cost of the latter update that makes the CDL problem more difficult than the CSC problem.

Most recent batch-mode CDL algorithms share the structure introduced in *[7]* (and described in more detail in *[22]*), the primary features of which are the use of Augmented Lagrangian methods and the solution of the most computationally expensive subproblems in the frequency domain. Earlier algorithms exist (see *[5, Sec. II.D]* for a thorough literature review), but since they are less effective, we do not consider them here, focusing on subsequent methods:

1. Proposed a number of improvements on the algorithm of *[7]*, including more efficient sparse representation and dictionary updates, and a different Augmented Lagrangian structure with better convergence properties (examined in more detail in *[23]*).
2. Proposed a number of dictionary update methods that lead to CDL algorithms with better performance than that of *[7]*.
3. Proposed a CDL algorithm that allows the inclusion of a spatial mask in the data fidelity term by exploiting the mask decoupling technique *[25]*.
4. Proposed an alternative masked CDL algorithm that has much lower memory requirements than that of *[9]*, and that converges faster in some contexts.

Unfortunately, due to the absence of any thorough performance comparisons between all of them (for example, *[24]* provides comparisons with *[7]* but not *[5]*), as well as due to the absence of a careful exploration of the optimum choice of algorithm parameters in most of these works, it is difficult to determine

which of these methods truly represents the state of the art in CDL.

Three other very recent methods do not receive the same thorough attention as those listed above. The algorithm of *[26]* addresses a variant of the CDL problem that is customized for neural signal processing and not relevant to most imaging applications, and *[27, 28]* appeared while we were finalizing this paper, so that it was not feasible to include them in our analysis or our main set of experimental comparisons. However, since the authors of *[27]* have made an implementation of their method publicly available, we do include this method in some additional performance comparisons in Sec. SV to SVII of the Supplementary Material.

The main contributions of the present paper are:

- Providing a thorough performance comparison among the different methods proposed in *[5, 24, 9, 16]*, allowing reliable identification of the most effective algorithms.
- Demonstrating that two of the algorithms proposed in *[24]*, with very different derivations, are in fact closely related and fall within the same class of algorithm.
- Proposing a new approach for the CDL problem without a spatial mask that outperforms all existing methods in a serial processing context.
- Proposing new approaches for the CDL problem with a spatial mask that respectively outperform existing methods in serial and parallel processing contexts.
- Carefully examining the sensitivity of the considered CDL algorithms to their parameters, and proposing simple heuristics for parameter selection that provide good performance.

## II Convolutional Dictionary Learning

CDL is usually posed in the form of the problem

$\operatorname*{arg\,min}_{\{\mathbf{d}_{m}\},\{\mathbf{x}_{m,k}\}}\frac{1}{2}\sum_{k}\Bigl{\|}\sum_{m}\mathbf{d}_{m}*\mathbf{x}_{m,k}-\mathbf{s}_{k}\Bigr{\|}_{2}^{2}+\lambda\sum_{m,k}\left\lVert\mathbf{x}_{m,k}\right\rVert_{1}$
$\text{such that }\left\lVert\mathbf{d}_{m}\right\rVert_{2}=1\ \forall m\ ,$ (1)

where the constraint on the norms of filters $\mathbf{d}_{m}$ is required to avoid the scaling ambiguity between filters and coefficients. The training images $\mathbf{s}_{k}$ are considered to be $N$ dimensional vectors, where $N$ is the number of pixels in each image, and we denote the number of filters and the number of training images by $M$ and $K$ respectively. This problem is non-convex in both variables $\{\mathbf{d}_{m}\}$ and $\{\mathbf{x}_{m,k}\}$, but is convex in $\{\mathbf{x}_{m,k}\}$ with $\{\mathbf{d}_{m}\}$ constant, and vice versa. As in standard (non-convolutional) dictionary learning, the usual approach to minimizing this functional is to alternate between updates of the sparse representation and the dictionary. The design of a CDL algorithm can therefore be decomposed into three components: the choice of sparse coding algorithm, the choice of dictionary update algorithm, and the choice of coupling mechanism, including how many iterations of each update should be performed before alternating, and which of their internal variables should be transferred across when alternating.

### II-A Sparse Coding

While a number of greedy matching pursuit type algorithms were developed for translation-invariant sparse representations *[5, Sec. II.C]*, recent algorithms have largely concentrated on a convolutional form of the standard Basis Pursuit DeNoising (BPDN) *[29]* problem

$\operatorname*{arg\,min}_{\mathbf{x}}\ (1/2)\left\lVert D\mathbf{x}-\mathbf{s}\right\rVert_{2}^{2}+\lambda\left\lVert\mathbf{x}\right\rVert_{1}\ .$ (2)

This form, which we will refer to as Convolutional BPDN (CBPDN), can be written as

$\operatorname*{arg\,min}_{\{\mathbf{x}_{m}\}}\frac{1}{2}\Bigl{\|}\sum_{m}\mathbf{d}_{m}*\mathbf{x}_{m}-\mathbf{s}\Bigr{\|}_{2}^{2}+\lambda\sum_{m}\left\lVert\mathbf{x}_{m}\right\rVert_{1}\ .$ (3)

If we define $D_{m}$ such that $D_{m}\mathbf{x}_{m}=\mathbf{d}_{m}*\mathbf{x}_{m}$, and

\[ D=\left(\begin{array}[]{cccc}D_{0}&D_{1}&\ldots\end{array}\right)\qquad\mathbf{x}=\left(\begin{array}[]{c}\mathbf{x}_{0}\\
\mathbf{x}_{1}\\
\vdots\end{array}\right)\ , \] (4)

we can rewrite the CBPDN problem in standard BPDN form Eq. (2). The Multiple Measurement Vector (MMV) version of CBPDN, for multiple images, can be written as

$\operatorname*{arg\,min}_{\{\mathbf{x}_{m,k}\}}\frac{1}{2}\sum_{k}\Bigl{\|}\sum_{m}\mathbf{d}_{m}*\mathbf{x}_{m,k}-\mathbf{s}_{k}\Bigr{\|}_{2}^{2}+\lambda\sum_{m,k}\left\lVert\mathbf{x}_{m,k}\right\rVert_{1}\ ,$ (5)

where $\mathbf{s}_{k}$ is the $k^{\text{th}}$ image, and $\mathbf{x}_{m,k}$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter and the $k^{\text{th}}$ image. By defining

\[ X=\left(\begin{array}[]{cccc}\mathbf{x}_{0,0}&\mathbf{x}_{0,1}&\ldots\\
\mathbf{x}_{1,0}&\mathbf{x}_{1,1}&\ldots\\
\vdots&\vdots&\ddots\end{array}\right)\quad S=\left(\begin{array}[]{cccc}\mathbf{s}_{0}&\mathbf{s}_{1}&\ldots\end{array}\right)\ , \] (6)

we can rewrite Eq. (5) in the standard BPDN MMV form,

$\operatorname*{arg\,min}_{X}\ (1/2)\left\lVert DX-S\right\rVert_{F}^{2}+\lambda\left\lVert X\right\rVert_{1}\ .$ (7)

Where possible, we will work with this form of the problem instead of Eq. (5) since it simplifies the notation, but the reader should keep in mind that $D$, $X$, and $S$ denote the specific block-structured matrices defined above.

The most effective solution for solving Eq. (5) is currently based on ADMM *[17]*, which solves problems of the form

$\operatorname*{arg\,min}_{\mathbf{x},\mathbf{y}}f(\mathbf{x})+g(\mathbf{y})\ \text{ such that }\ A\mathbf{x}+B\mathbf{y}=\mathbf{c}$ (8)

by iterating over the steps

$\mathbf{x}^{(i+1)}=\operatorname*{arg\,min}_{\mathbf{x}}f(\mathbf{x})+\frac{\rho}{2}\left\lVert A\mathbf{x}+B\mathbf{y}^{(i)}-\mathbf{c}+\mathbf{u}^{(i)}\right\rVert_{2}^{2}$ (9)
$\mathbf{y}^{(i+1)}=\operatorname*{arg\,min}_{\mathbf{y}}g(\mathbf{y})+\frac{\rho}{2}\left\lVert A\mathbf{x}^{(i+1)}+B\mathbf{y}-\mathbf{c}+\mathbf{u}^{(i)}\right\rVert_{2}^{2}$ (10)
$\mathbf{u}^{(i+1)}=\mathbf{u}^{(i)}+A\mathbf{x}^{(i+1)}+B\mathbf{y}^{(i+1)}-\mathbf{c}\ ,$ (11)

where *penalty parameter* $\rho$ is an algorithm parameter that plays an important role in determining the convergence rate

of the iterations, and $\mathbf{u}$ is the *dual variable* corresponding to the constraint $A\mathbf{x}+B\mathbf{y}=\mathbf{c}$. We can apply ADMM to problem Eq. (7) by *variable splitting*, introducing an auxiliary variable $Y$ that is constrained to be equal to the primary variable $X$, leading to the equivalent problem

$\operatorname*{arg\,min}_{X,Y}\left(1/2\right)\left\|DX-S\right\|_{F}^{2}+\lambda\left\|Y\right\|_{1}\quad\text{s.t.}\quad X=Y\;,$ (12)

for which we have the ADMM iterations

$X^{(i+1)}=\operatorname*{arg\,min}_{X}\frac{1}{2}\left\|DX-S\right\|_{F}^{2}+\frac{\rho}{2}\left\|X-Y^{(i)}+U^{(i)}\right\|_{F}^{2}$ (13)
$Y^{(i+1)}=\operatorname*{arg\,min}_{Y}\lambda\left\|Y\right\|_{1}+\frac{\rho}{2}\left\|X^{(i+1)}-Y+U^{(i)}\right\|_{F}^{2}$ (14)
$U^{(i+1)}=U^{(i)}+X^{(i+1)}-Y^{(i+1)}\;.$ (15)

Step Eq. (15) involves simple arithmetic, and step Eq. (14) has a closed-form solution

$Y^{(i+1)}=\mathcal{S}_{\lambda/\rho}\left(X^{(i+1)}+U^{(i)}\right)\;,$ (16)

where $\mathcal{S}_{\gamma}(\cdot)$ is the soft-thresholding function *[30, Sec. 6.5.2]*

$\mathcal{S}_{\gamma}(V)=\operatorname*{sign}(V)\odot\max(0,|V|-\gamma)\;,$ (17)

with $\operatorname*{sign}(\cdot)$ and $|\cdot|$ of a vector considered to be applied element-wise, and $\odot$ denoting element-wise multiplication. The most computationally expensive step is Eq. (13), which requires solving the linear system

$(D^{T}D+\rho I)X=D^{T}S+\rho(Y-U)\;.$ (18)

Since $D^{T}D$ is a very large matrix, it is impractical to solve this linear system using the approaches that are effective when $D$ is not a convolutional dictionary. It is possible, however, to exploit the FFT for efficient implementation of the convolution via the DFT convolution theorem. Transforming Eq. (18) into the DFT domain gives

$(\hat{D}^{H}\hat{D}+\rho I)\hat{X}=\hat{D}^{H}\hat{S}+\rho(\hat{Y}-\hat{U})\;,$ (19)

where $\hat{Z}$ denotes the DFT of variable $Z$. Due to the structure of $\hat{D}$, which consists of concatenated diagonal matrices $\hat{D}_{m}$, linear system Eq. (19) can be decomposed into a set of $NK$ independent linear systems *[7]*, each of which has a left hand side consisting of a diagonal matrix plus a rank-one component, which can be solved very efficiently by exploiting the Sherman-Morrison formula *[8]*.

### II-B Dictionary Update

In developing the dictionary update, it is convenient to switch the indexing of the coefficient map from $\mathbf{x}_{m,k}$ to $\mathbf{x}_{k,m}$, writing the problem as

$\operatorname*{arg\,min}_{\{\mathbf{d}_{m}\}}\frac{1}{2}\sum_{k}\left\|\sum_{m}\mathbf{x}_{k,m}*\mathbf{d}_{m}-\mathbf{s}_{k}\right\|_{2}^{2}\text{ s.t. }\left\|\mathbf{d}_{m}\right\|_{2}=1\;,$ (20)

which is a convolutional form of Method of Optimal Directions (MOD) *[31]* with a constraint on the filter normalization. As for CSC, we will develop the algorithms for solving this problem in the spatial domain, but will solve the critical sub-problems in the frequency domain. We want to solve for $\{\mathbf{d}_{m}\}$ with a relatively small support, but when computing convolutions in the frequency domain, we need to work with $\mathbf{d}_{m}$ that have been zero-padded to the common spatial dimensions of $\mathbf{x}_{k,m}$ and $\mathbf{s}_{k}$. The most straightforward way of dealing with this complication is to consider the $\mathbf{d}_{m}$ to be zero-padded and add a constraint that requires that they be zero outside of the desired support. If we denote the projection operator that zeros the regions of the filters outside of the desired support by $P$, we can write a constraint set that combines this support constraint with the normalization constraint as

$C_{\text{PN}}=\{\mathbf{x}\in\mathbb{R}^{N}:\left(I-P\right)\mathbf{x}=0,\left\|\mathbf{x}\right\|_{2}=1\}\;,$ (21)

and write the dictionary update as

$\operatorname*{arg\,min}_{\{\mathbf{d}_{m}\}}\frac{1}{2}\sum_{k}\left\|\sum_{m}\mathbf{x}_{k,m}*\mathbf{d}_{m}-\mathbf{s}_{k}\right\|_{2}^{2}\text{ s.t. }\mathbf{d}_{m}\in C_{\text{PN}}\;\forall m\;.$ (22)

Introducing the indicator function $\iota_{C_{\text{PN}}}$ of the constraint set $C_{\text{PN}}$, where the indicator function of a set $S$ is defined as

\[ \iota_{S}(X)=\left\{\begin{array}[]{ll}0&\text{ if }X\in S\\
\infty&\text{ if }X\notin S\end{array}\right.\;, \] (23)

allows Eq. (22) to be written in unconstrained form *[32]*

$\operatorname*{arg\,min}_{\{\mathbf{d}_{m}\}}\frac{1}{2}\sum_{k}\left\|\sum_{m}\mathbf{x}_{k,m}*\mathbf{d}_{m}-\mathbf{s}_{k}\right\|_{2}^{2}+\sum_{m}\iota_{C_{\text{PN}}}(\mathbf{d}_{m})\;.$ (24)

Defining $X_{k,m}$ such that $X_{k,m}\mathbf{d}_{m}=\mathbf{x}_{k,m}*\mathbf{d}_{m}$ and

\[ X_{k}=\left(\begin{array}[]{cccc}X_{k,0}&X_{k,1}&\ldots\end{array}\right)\quad\mathbf{d}=\left(\begin{array}[]{c}\mathbf{d}_{0}\\
\mathbf{d}_{1}\\
\vdots\end{array}\right)\;, \] (25)

this problem can be expressed as

$\operatorname*{arg\,min}_{\mathbf{d}}\left(1/2\right)\sum_{k}\left\|X_{k}\mathbf{d}-\mathbf{s}_{k}\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{d})\;,$ (26)

or, by defining

\[ X=\left(\begin{array}[]{cccc}X_{0,0}&X_{0,1}&\ldots\\
X_{1,0}&X_{1,1}&\ldots\\
\vdots&\vdots&\ddots\end{array}\right)\quad\mathbf{s}=\left(\begin{array}[]{c}\mathbf{s}_{0}\\
\mathbf{s}_{1}\\
\vdots\end{array}\right)\;, \] (27)

as

$\operatorname*{arg\,min}_{\mathbf{d}}\left(1/2\right)\left\|X\mathbf{d}-\mathbf{s}\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{d})\;.$ (28)

Algorithms for solving this problem will be discussed in Sec. III. A common feature of most of these methods is the need to solve a linear system that includes the data fidelity term $(1/2)\left\|X\mathbf{d}-\mathbf{s}\right\|_{2}^{2}$. As in the case of the $X$ step Eq. (13) for CSC, this problem can be solved in the frequency domain, but there is a critical difference: $\hat{X}^{H}\hat{X}$ is composed of independent components of rank $K$ instead of rank 1, so that the very efficient Sherman Morrison solution cannot be directly exploited. It is this property that makes the dictionary update inherently more computationally expensive than the sparse coding stage, complicating the design of algorithms, and leading to the present situation in which there is far less clarity as to the best choice of dictionary learning algorithm than there is for the choice of the sparse coding algorithm.

### II-C Update Coupling

Both the sparse coding and dictionary update stages are typically solved via iterative algorithms, and many of these algorithms have more than one working variable that can be used to represent the current solution. The major design choices in coupling the alternating optimization of these two stages are therefore:

1. how many iterations of each subproblem to perform before switching to the other subproblem, and
2. which working variable from each subproblem to pass across to the other subproblem.

Since these issues are addressed in detail in *[23]*, we only summarize the conclusions here:

- When both subproblems are solved by ADMM algorithms, most authors have coupled the subproblems via the primary variables (corresponding, for example, to $X$ in Eq. (12)) of each ADMM algorithm.
- This choice tends to be rather unstable, and requires either multiple iterations of each subproblem before alternating, or very large penalty parameters, which can lead to slow convergence.
- The alternative strategy of coupling the subproblems via the auxiliary variables (corresponding, for example, to $Y$ in Eq. (12)) of each ADMM algorithm tends to be more stable, not requiring multiple iterations before alternating, and converging faster.

## III Dictionary Update Algorithms

Since the choice of the best CSC algorithm is not in serious dispute, the focus of this work is on the choice of dictionary update algorithm.

### III-A ADMM with Equality Constraint

The simplest approach to solving Eq. (28) via an ADMM algorithm is to apply the variable splitting

$\operatorname*{arg\,min}_{\mathbf{d},\mathbf{g}}\left(1/2\right)\left\|X\mathbf{d}-\mathbf{s}\right\|_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g})\ \ \text{s.t.}\ \ \mathbf{d}=\mathbf{g}\ ,$ (29)

for which the corresponding ADMM iterations are

$\mathbf{d}^{(i+1)}$ $=\operatorname*{arg\,min}_{\mathbf{d}}\frac{1}{2}\left\|X\mathbf{d}-\mathbf{s}\right\|_{2}^{2}+\frac{\sigma}{2}\left\|\mathbf{d}-\mathbf{g}^{(i)}+\mathbf{h}^{(i)}\right\|_{2}^{2}$ (30)
$\mathbf{g}^{(i+1)}$ $=\operatorname*{arg\,min}_{\mathbf{g}}\iota_{C_{\text{PS}}}(\mathbf{g})+\frac{\sigma}{2}\left\|\mathbf{d}^{(i+1)}-\mathbf{g}+\mathbf{h}^{(i)}\right\|_{2}^{2}$ (31)
$\mathbf{h}^{(i+1)}$ $=\mathbf{h}^{(i)}+\mathbf{d}^{(i+1)}-\mathbf{g}^{(i+1)}\ .$ (32)

Step Eq. (31) is of the form

$\operatorname*{arg\,min}_{\mathbf{x}}\left(1/2\right)\left\|\mathbf{x}-\mathbf{y}\right\|_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{x})=\operatorname{prox}_{\iota_{C_{\text{PS}}}}(\mathbf{y})\ .$ (33)

It is clear from the geometry of the problem that

$\operatorname{prox}_{\iota_{C_{\text{PS}}}}(\mathbf{y})=\frac{PP^{T}\mathbf{y}}{\left\|PP^{T}\mathbf{y}\right\|_{2}}\ ,$ (34)

or, if the normalization $\left\|\mathbf{d}_{m}\right\|_{2}\leq 1$ is desired instead,

\[ \operatorname{prox}_{\iota_{C_{\text{PS}}}}(\mathbf{y})=\begin{cases}PP^{T}\mathbf{y}&\text{if}\ \left\|PP^{T}\mathbf{y}\right\|_{2}\leq 1\\
\frac{PP^{T}\mathbf{y}}{\left\|PP^{T}\mathbf{y}\right\|_{2}}&\text{if}\ \left\|PP^{T}\mathbf{y}\right\|_{2}>1\end{cases}\ \ \ \ . \] (35)

Step Eq. (30) involves solving the linear system

$(X^{T}X+\sigma I)\mathbf{d}=X^{T}\mathbf{s}+\sigma(\mathbf{g}-\mathbf{h})\ ,$ (36)

which can be expressed in the DFT domain as

$(\hat{X}^{H}\hat{X}+\sigma I)\hat{\mathbf{d}}=\hat{X}^{H}\hat{\mathbf{s}}+\sigma(\hat{\mathbf{g}}-\hat{\mathbf{h}})\ .$ (37)

This linear system can be decomposed into a set of $N$ independent linear systems, but in contrast to Eq. (19), each of these has a left hand side consisting of a diagonal matrix plus a rank $K$ component, which precludes direct use of the Sherman-Morrison formula *[5]*.

We consider three different approaches to solving these linear systems:

#### III-A1 Conjugate Gradient

An obvious approach to solving Eq. (37) without having to explicitly construct the matrix $\hat{X}^{H}\hat{X}+\sigma I$ is to apply an iterative method such as Conjugate Gradient (CG). The experiments reported in *[5]* indicated that solving this system to a relative residual tolerance of $10^{-3}$ or better is sufficient for the dictionary learning algorithm to converge reliably. The number of CG iterations required can be substantially reduced by using the solution from the previous outer iteration as an initial value.

#### III-A2 Iterated Sherman-Morrison

Since the independent linear systems into which Eq. (37) can be decomposed have a left hand side consisting of a diagonal matrix plus a rank $K$ component, one can iteratively apply the Sherman-Morrison formula to obtain a solution *[5]*. This approach is very effective for small to moderate $K$, but performs poorly for large $K$ since the computational cost is $\mathcal{O}(K^{2})$.

#### III-A3 Spatial Tiling

When $K=1$ in Eq. (37), the very efficient solution via the Sherman-Morrison formula is possible. As pointed out in *[24]*, a larger set of training images can be spatially tiled to form a single large image, so that the problem is solved with $K^{\prime}=1$.

### III-B Consensus Framework

In this section it is convenient to introduce different block-matrix and vector notation for the coefficient maps and dictionary, but we overload the usual symbols to emphasize their corresponding roles. We define $X_{k}$ as in Eq. (25), but define

\[ X=\left(\begin{array}[]{cccc}X_{0}&0&\ldots\\
0&X_{1}&\ldots\\
\vdots&\vdots&\ddots\end{array}\right)\ \mathbf{d}_{k}=\left(\begin{array}[]{c}\mathbf{d}_{0,k}\\
\mathbf{d}_{1,k}\\
\vdots\end{array}\right)\ \mathbf{d}=\left(\begin{array}[]{c}\mathbf{d}_{0}\\
\mathbf{d}_{1}\\
\vdots\end{array}\right) \] (38)

where $\mathbf{d}_{m,k}$ is distinct copy of dictionary filter $m$ corresponding to training image $k$.

As proposed in *[24]*, we can pose problem Eq. (28) in the form of an ADMM consensus problem *[17, Ch. 7]*

$\operatorname*{arg\,min}_{\mathbf{d}_{k}}\left(1/2\right)\sum_{k}\left\|X_{k}\mathbf{d}_{k}-\mathbf{s}_{k}\right\|_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g})$
$\text{s.t.}\ \ \mathbf{g}=\mathbf{d}_{k}\ \forall k\ ,$ (39)

which can be written in standard ADMM form as

$\operatorname*{arg\,min}_{\mathbf{d}}\ \frac{1}{2}\left\|X\mathbf{d}-\mathbf{s}\right\|_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g})\ \ \text{s.t.}\ \ \mathbf{d}-E\mathbf{g}=0\ ,$ (40)

where $E=\left(\begin{array}[]{cccc}I&I&\ldots\end{array}\right)^{T}$.

### III-

The corresponding ADMM iterations are

$\mathbf{d}^{(i+1)}=\operatorname*{arg\,min}_{\mathbf{d}}\frac{1}{2}\big{\|}X\mathbf{d}-\mathbf{s}\big{\|}_{2}^{2}+\frac{\sigma}{2}\left\|\mathbf{d}-E\mathbf{g}^{(i)}+\mathbf{h}^{(i)}\right\|_{2}^{2}$ (41)
$\mathbf{g}^{(i+1)}=\operatorname*{arg\,min}_{\mathbf{g}}\iota_{C_{\text{PN}}}(\mathbf{g})+\frac{\sigma}{2}\left\|\mathbf{d}^{(i+1)}-E\mathbf{g}+\mathbf{h}^{(i)}\right\|_{2}^{2}$ (42)
$\mathbf{h}^{(i+1)}=\mathbf{h}^{(i)}+\mathbf{d}^{(i+1)}-E\mathbf{g}^{(i+1)}\;.$ (43)

Since $X$ is block diagonal, Eq. (41) can be solved as the $K$ independent problems

$\mathbf{d}_{k}^{(i+1)}=\operatorname*{arg\,min}_{\mathbf{d}_{k}}\frac{1}{2}\big{\|}X_{k}\mathbf{d}_{k}-\mathbf{s}_{k}\big{\|}_{2}^{2}+\frac{\sigma}{2}\left\|\mathbf{d}_{k}-\mathbf{g}^{(i)}+\mathbf{h}_{k}^{(i)}\right\|_{2}^{2}\;,$ (44)

each of which can be solved via the same efficient DFT-domain Sherman-Morrison method used for Eq. (13). Sub-problem Eq. (42) can be expressed as *[17, Sec. 7.1.1]*

$\mathbf{g}^{(i+1)}=$ $\operatorname*{arg\,min}_{\mathbf{g}}\iota_{C_{\text{PN}}}(\mathbf{g})+$
$\frac{K\sigma}{2}\bigg{\|}\mathbf{g}-K^{-1}\bigg{(}\sum_{k=0}^{K-1}\mathbf{d}_{k}^{(i+1)}+\sum_{k=0}^{K-1}\mathbf{h}_{k}^{(i)}\bigg{)}\bigg{\|}_{2}^{2}\;.$ (45)

which has the closed-form solution

$\mathbf{g}^{(i+1)}=\operatorname{prox}_{\iota_{C_{\text{PN}}}}\bigg{(}K^{-1}\bigg{(}\sum_{k=0}^{K-1}\mathbf{d}_{k}^{(i+1)}+\sum_{k=0}^{K-1}\mathbf{h}_{k}^{(i)}\bigg{)}\bigg{)}\;.$ (46)

### III-C 3D / Frequency Domain Consensus

Like spatial tiling (see Sec. III-A3), the “3D” method proposed in *[24]* maps the dictionary update problem with $K>1$ to an equivalent problem for which $K^{\prime}=1$. The “3D” method achieves this by considering an array of $K$ 2D training images as a single 3D training volume. The corresponding dictionary filters are also inherently 3D, but the constraint is modified to require that they are zero other than in the first 3D slice (this can be viewed as an extension of the constraint that the spatially-padded filters are zero except on their desired support) so that the final results is a set of 2D filters, as desired.

While ADMM consensus and “3D” were proposed as two entirely distinct methods *[24]*, it turns out they are closely related: the “3D” method is ADMM consensus with the data fidelity term and constraint expressed in the DFT domain. Since the notation is a bit cumbersome, the point will be illustrated for the $K=2$ case, but the argument is easily generalized to arbitrary $K$.

When $K=2$, the dictionary update problem can be expressed as

\[ \operatorname*{arg\,min}_{\mathbf{d}}\frac{1}{2}\left\|\left(\begin{array}[]{c}X_{0}\\
X_{1}\end{array}\right)\mathbf{d}-\left(\begin{array}[]{c}\mathbf{s}_{0}\\
\mathbf{s}_{1}\end{array}\right)\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{d})\;\;, \] (47)

which can be rewritten as the equivalent problem

$\operatorname*{arg\,min}_{\mathbf{d}_{0},\mathbf{d}_{1}}\frac{1}{2}\left\|\left(\begin{array}[]{cc}X_{0}&X_{1}\\
X_{1}&X_{0}\end{array}\right)\left(\begin{array}[]{c}\mathbf{d}_{0}\\
\mathbf{d}_{1}\end{array}\right)-\left(\begin{array}[]{c}\mathbf{s}_{0}\\
\mathbf{s}_{1}\end{array}\right)\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{g})$
$\text{s.t.}\quad\mathbf{d}_{0}=\mathbf{g}\quad\mathbf{d}_{1}=\mathbf{0}\;,$ (48)

where the constraint can also be written as

\[ \left(\begin{array}[]{c}\mathbf{d}_{0}\\
\mathbf{d}_{1}\end{array}\right)=\left(\begin{array}[]{c}I\\
0\end{array}\right)\mathbf{g}\;\;. \] (49)

The general form of the matrix in Eq. (48) is a block-circulant matrix constructed from the blocks $X_{k}$. Since the multiplication of the dictionary block vector by the block-circulant matrix is equivalent to convolution in an additional dimension, this equivalent problem represents the “3D” method.

Now, define the un-normalized $2\times 2$ block DFT matrix operating in this extra dimension as

\[ F=\left(\begin{array}[]{cc}I&I\\
I&-I\end{array}\right)\;, \] (50)

and apply it to the objective function and constraint, giving

$\operatorname*{arg\,min}_{\mathbf{d}_{0},\mathbf{d}_{1}}\frac{1}{2}\left\|F\left(\begin{array}[]{cc}X_{0}&X_{1}\\
X_{1}&X_{0}\end{array}\right)F^{-1}F\left(\begin{array}[]{c}\mathbf{d}_{0}\\
\mathbf{d}_{1}\end{array}\right)-F\left(\begin{array}[]{c}\mathbf{s}_{0}\\
\mathbf{s}_{1}\end{array}\right)\right\|_{2}^{2}$
$+\iota_{C_{\text{PN}}}(\mathbf{g})\quad\text{ s.t. }\quad F\left(\begin{array}[]{c}\mathbf{d}_{0}\\
\mathbf{d}_{1}\end{array}\right)=F\left(\begin{array}[]{c}I\\
0\end{array}\right)\mathbf{g}\;.$ (51)

Since the DFT diagonalises a circulant matrix, this is

$\operatorname*{arg\,min}_{\mathbf{d}_{0},\mathbf{d}_{1}}\frac{1}{2}\left\|\left(\begin{array}[]{cc}X_{0}+X_{1}&0\\
0&X_{0}-X_{1}\end{array}\right)\left(\begin{array}[]{c}\mathbf{d}_{0}+\mathbf{d}_{1}\\
\mathbf{d}_{0}-\mathbf{d}_{1}\end{array}\right)-\left(\begin{array}[]{c}\mathbf{s}_{0}+\mathbf{s}_{1}\\
\mathbf{s}_{0}-\mathbf{s}_{1}\end{array}\right)\right\|_{2}^{2}$
$+\iota_{C_{\text{PN}}}(\mathbf{g})\quad\text{ s.t. }\quad\left(\begin{array}[]{c}\mathbf{d}_{0}+\mathbf{d}_{1}\\
\mathbf{d}_{0}-\mathbf{d}_{1}\end{array}\right)=\left(\begin{array}[]{c}\mathbf{g}\\
\mathbf{g}\end{array}\right)\;.$ (52)

In this form the problem is an ADMM consensus problem in variables

$X_{0}^{\prime}=X_{0}+X_{1}\quad$ $\mathbf{d}_{0}^{\prime}=\mathbf{d}_{0}+\mathbf{d}_{1}\quad$ $\mathbf{s}_{0}^{\prime}=\mathbf{s}_{0}+\mathbf{s}_{1}$
$X_{1}^{\prime}=X_{0}-X_{1}\quad$ $\mathbf{d}_{1}^{\prime}=\mathbf{d}_{0}-\mathbf{d}_{1}\quad$ $\mathbf{s}_{1}^{\prime}=\mathbf{s}_{0}-\mathbf{s}_{1}\;.$ (53)

### III-D FISTA

The Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) *[33]*, an accelerated proximal gradient method, has been used for CSC *[6, 5, 19]*, and in a recent online CDL algorithm *[18]*, but has not previously been considered for the dictionary update of a batch-mode dictionary learning algorithm.

The FISTA iterations for solving Eq. (28) are

$\mathbf{y}^{(i+1)}=\operatorname{prox}_{\iota_{C_{\text{PN}}}}\bigg{(}\mathbf{d}^{(i)}-\frac{1}{L}\nabla_{\mathbf{d}}\Big{(}\frac{1}{2}\left\|X\mathbf{d}-\mathbf{s}\right\|_{2}^{2}\Big{)}\bigg{)}$ (54)
$t^{(i+1)}=\frac{1}{2}\bigg{(}1+\sqrt{1+4\left(t^{(i)}\right)^{2}}\bigg{)}$ (55)
$\mathbf{d}^{(i+1)}=\mathbf{y}^{(i+1)}+\frac{t^{(i)}-1}{t^{(i+1)}}\Big{(}\mathbf{y}^{(i+1)}-\mathbf{d}^{(i)}\Big{)}\;,$ (56)

where $t^{0}=1$, and $L>0$ is a parameter controlling the gradient descent step size. Parameter $L$ can be computed adaptively by using a backtracking step size rule *[33]*, but in the experiments reported here we used a constant $L$ for simplicity. The gradient of the data fidelity term $(1/2)\left\|X\mathbf{d}-\mathbf{s}\right\|_{2}^{2}$ in Eq. (54) is computed in the DFT domain

$\nabla_{\hat{\mathbf{d}}}\Big{(}\frac{1}{2}\big{\|}\hat{X}\hat{\mathbf{d}}-\hat{\mathbf{s}}\big{\|}_{2}^{2}\Big{)}=\hat{X}^{H}\big{(}\hat{X}\hat{\mathbf{d}}-\hat{\mathbf{s}}\big{)}\;,$ (57)

as advocated in *[5]* for the FISTA solution of the CSC problem, and the $\mathbf{y}^{(i+1)}$ variable is taken as the result of the dictionary update.

## IV Masked Convolutional Dictionary Learning

When we wish to learn a dictionary from data with missing samples, or have reason to be concerned about the possibility of boundary artifacts resulting from the circular boundary conditions associated with the computation of the convolutions in the DFT domain, it is useful to introduce a variant of Eq. (1) that includes a spatial mask *[9]*, which can be represented by a diagonal matrix $W$

$\operatorname*{arg\,min}_{\{\mathbf{d}_{m}\},\{\mathbf{x}_{m,k}\}}\frac{1}{2}\sum_{k}\Big{\|}W\Big{(}\sum_{m}\mathbf{d}_{m}*\mathbf{x}_{m,k}-\mathbf{s}_{k}\Big{)}\Big{\|}_{2}^{2}+$
$\lambda\sum_{m,k}\left\|\mathbf{x}_{m,k}\right\|_{1}\ \text{s.t.}\ \left\|\mathbf{d}_{m}\right\|_{2}=1\ \forall m\ .$ (58)

As in Sec. II, we separately consider the minimization of this functional with respect to $\{\mathbf{x}_{m,k}\}$ (sparse coding) and $\{\mathbf{d}_{m}\}$ (dictionary update).

### IV-A Sparse Coding

A masked form of the MMV CBPDN problem Eq. (7) can be expressed as the problem

$\operatorname*{arg\,min}_{X}\ (1/2)\big{\|}W(DX-S)\big{\|}_{F}^{2}+\lambda\|S\|_{1}\ .$ (59)

There are two different methods for solving this problem. The one, proposed in *[9]*, exploits the mask decoupling technique *[25]*, involving applying an alternative variable splitting to give the ADMM problem

$\operatorname*{arg\,min}_{X}\ (1/2)\left\|WY_{1}\right\|_{F}^{2}+\lambda\|Y_{0}\|_{1}$
$\text{s.t.}\ Y_{0}=X\quad Y_{1}=DX-S\ ,$ (60)

where the constraint can also be written as

\[ \left(\begin{array}[]{c}Y_{0}\\
Y_{1}\end{array}\right)=\left(\begin{array}[]{c}I\\
D\end{array}\right)X-\left(\begin{array}[]{c}0\\
S\end{array}\right)\ . \] (61)

The corresponding ADMM iterations are

$X^{(i+1)}=\operatorname*{arg\,min}_{X}\frac{\rho}{2}\left\|DX-(Y_{1}^{(i)}+S-U_{1}^{(i)})\right\|_{F}^{2}+$
$\frac{\rho}{2}\left\|X-(Y_{0}^{(i)}-U_{0}^{(i)})\right\|_{F}^{2}$ (62)
$Y_{0}^{(i+1)}=\operatorname*{arg\,min}_{Y_{0}}\lambda\left\|Y_{0}\right\|_{1}+\frac{\rho}{2}\left\|Y_{0}-(X^{(i+1)}+U_{0}^{(i)})\right\|_{F}^{2}$ (63)
$Y_{1}^{(i+1)}=\operatorname*{arg\,min}_{Y_{1}}\frac{1}{2}\big{\|}WY_{1}\big{\|}_{F}^{2}+$
$\frac{\rho}{2}\left\|Y_{1}-(DX^{(i+1)}-S+U_{1}^{(i)})\right\|_{F}^{2}$ (64)
$U_{0}^{(i+1)}=U_{0}^{(i)}+X^{(i+1)}-Y_{0}^{(i+1)}$ (65)
$U_{1}^{(i+1)}=U_{1}^{(i)}+DX^{(i+1)}-Y_{1}^{(i+1)}-S\ .$ (66)

The functional minimized in Eq. (62) is of the same form as Eq. (13), and can be solved via the same frequency domain method, the solution to Eq. (63) is as in Eq. (16), and the solution to Eq. (64) is given by

$(W^{T}W+\rho I)Y_{1}^{(i+1)}=\rho(DX^{(i+1)}-S+U_{1}^{(i)})\ .$ (67)

The other method for solving Eq. (59) involves appending an impulse filter to the dictionary and solving the problem in a way that constrains the coefficient map corresponding to this filter to be zero where the mask is unity, and to be unconstrained where the mask is zero *[34, 16]*. Both approaches provide very similar performance *[16]*, the major difference being that the former is a bit more complicated to implement, while the latter is restricted to addressing problems where $W$ has only zero or one entries. We will use the mask decoupling approach for the experiments reported here since it does not require any restrictions on $W$.

### IV-B Dictionary Update

The dictionary update requires solving the problem

$\operatorname*{arg\,min}_{\mathbf{d}}\ (1/2)\big{\|}W(X\mathbf{d}-\mathbf{s})\big{\|}_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{d})\ .$ (68)

Algorithms for solving this problem are discussed in the following section.

## V Masked Dictionary Update Algorithms

### V-A Block-Constraint ADMM

Problem Eq. (68) can be solved via the splitting *[9]*

$\operatorname*{arg\,min}_{\mathbf{d}}\ (1/2)\left\|W\mathbf{g}_{1}\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{g}_{0})$
$\text{s.t.}\ \mathbf{g}_{0}=\mathbf{d}\quad\mathbf{g}_{1}=X\mathbf{d}-\mathbf{s}\ ,$ (69)

where the constraint can also be written as

\[ \left(\begin{array}[]{c}\mathbf{g}_{0}\\
\mathbf{g}_{1}\end{array}\right)=\left(\begin{array}[]{c}I\\
X\end{array}\right)\mathbf{d}-\left(\begin{array}[]{c}0\\
\mathbf{s}\end{array}\right)\ . \] (70)

This problem has the same structure as Eq. (60), the only difference being the replacement of the $\ell_{1}$ norm with the indicator function of the constraint set. The ADMM iterations are thus largely the same as Eq. (62) – (66), the differences being that the $\ell_{1}$ norm in Eq. (63) is replaced with the indicator function of the constraint set, and that the step corresponding to Eq. (62) is more computationally expensive to solve, just as Eq. (30) is more expensive than Eq. (13).

### V-B Extended Consensus Framework

In this section we re-use the variant notation introduced in Sec. III-B. The masked dictionary update Eq. (68) can be solved via a hybrid of the mask decoupling and ADMM consensus approaches, which can be formulated as

$\operatorname*{arg\,min}_{\mathbf{d}}\ (1/2)\left\|W\mathbf{g}_{1}\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{g}_{0})$
$\text{s.t.}\ E\mathbf{g}_{0}=\mathbf{d}\quad\mathbf{g}_{1}=X\mathbf{d}-\mathbf{s}\ ,$ (71)

where the constraint can also be written as

\[ \left(\begin{array}[]{c}I\\
X\end{array}\right)\mathbf{d}+\left(\begin{array}[]{cc}-E&0\\
0&-I\end{array}\right)\left(\begin{array}[]{c}\mathbf{g}_{0}\\
\mathbf{g}_{1}\end{array}\right)=\left(\begin{array}[]{c}0\\
\mathbf{s}\end{array}\right)\ , \] (72)

or, expanding the block components of $\mathbf{d}$, $\mathbf{g}_1$, and $\mathbf{s}$,

$$
\left( \begin{array}{cccc}
I &amp; 0 &amp; \dots \\
0 &amp; I &amp; \dots \\
\vdots &amp; \vdots &amp; \ddots \\
X_0 &amp; 0 &amp; \dots \\
0 &amp; X_1 &amp; \dots \\
\vdots &amp; \vdots &amp; \ddots
\end{array} \right)
\left( \begin{array}{c}
\mathbf{d}_0 \\
\mathbf{d}_1 \\
\vdots
\end{array} \right)
-
\left( \begin{array}{c}
\mathbf{g}_0 \\
\mathbf{g}_0 \\
\vdots \\
\mathbf{g}_{1,0} \\
\mathbf{g}_{1,1} \\
\vdots
\end{array} \right)
=
\left( \begin{array}{c}
0 \\
0 \\
\vdots \\
\mathbf{s}_0 \\
\mathbf{s}_1 \\
\vdots
\end{array} \right). \tag{73}
$$

The corresponding ADMM iterations are

$$
\begin{aligned}
\mathbf{d}^{(i+1)} &amp;= \arg\min_{\mathbf{d}} \frac{\rho}{2} \left\| X \mathbf{d} - \left(\mathbf{g}_1^{(i)} + \mathbf{s} - \mathbf{h}_1^{(i)}\right) \right\|_2^2 + \\
&amp;\quad \frac{\rho}{2} \left\| \mathbf{d} - \left(E \mathbf{g}_0^{(i)} - \mathbf{h}_0^{(i)}\right) \right\|_2^2 \tag{74} \\
\mathbf{g}_0^{(i+1)} &amp;= \arg\min_{\mathbf{g}_0} \iota C_{\mathcal{m}}(\mathbf{g}_0) + \frac{\rho}{2} \left\| E \mathbf{g}_0 - \left(\mathbf{d}^{(i+1)} + \mathbf{h}_0^{(i)}\right) \right\|_2^2 \tag{75} \\
\mathbf{g}_1^{(i+1)} &amp;= \arg\min_{\mathbf{g}_1} \frac{1}{2} \| W \mathbf{g}_1 \|_2^2 + \\
&amp;\quad \frac{\rho}{2} \left\| \mathbf{g}_1 - \left(X \mathbf{d}^{(i+1)} - \mathbf{s} + \mathbf{h}_1^{(i)}\right) \right\|_2^2 \tag{76} \\
\mathbf{h}_0^{(i+1)} &amp;= \mathbf{h}_0^{(i)} + \mathbf{d}^{(i+1)} - E \mathbf{g}_0^{(i+1)} \tag{77} \\
\mathbf{h}_1^{(i+1)} &amp;= \mathbf{h}_1^{(i)} + X \mathbf{d}^{(i+1)} - \mathbf{g}_1^{(i+1)} - \mathbf{s} \tag{78}
\end{aligned}
$$

Steps Eq. (74), (75), and (77) have the same form, and can be solved in the same way, as steps Eq. (41), (42), and (43) respectively of the ADMM algorithm in Sec. III-B, and steps Eq. (76) and (78) have the same form, and can be solved in the same way, as the corresponding steps in the ADMM algorithm of Sec. V-A.

## C. FISTA

Problem Eq. (68) can be solved via FISTA as described in Sec. III-D, but the calculation of the gradient term is complicated by the presence of the spatial mask. This difficulty can be handled by transforming back and forth between spatial and frequency domains so that the convolution operations are computed efficiently in the frequency domain, while the masking operation is computed in the spatial domain, i.e.

$$
F \left(\nabla_{\mathbf{d}} \left(\frac{1}{2} \| W (X \mathbf{d} - \mathbf{s}) \|_2^2\right)\right) = \hat{X}^H F \left(W^T W F^{-1} (\hat{X} \hat{\mathbf{d}} - \hat{\mathbf{s}})\right), \tag{79}
$$

where $F$ and $F^{-1}$ represent the DFT and inverse DFT transform operators, respectively.

## VI. MULTI-CHANNEL CDL

As discussed in [35], there are two distinct ways of defining a convolutional representation of multi-channel data: a single-channel dictionary together with a distinct set of coefficient maps for each channel, or a multi-channel dictionary together with a shared set of coefficient maps. Since the dictionary learning problem for the former case is a straightforward extension of the single-channel problems discussed above, here we focus on the latter case, which can be expressed as

$$
\begin{aligned}
\arg\min_{\{\mathbf{d}_{c,m}\}, \{\mathbf{x}_{m,k}\}} &amp; \frac{1}{2} \sum_{c,k} \left\| \sum_{m} \mathbf{d}_{c,m} * \mathbf{x}_{m,k} - \mathbf{s}_{c,k} \right\|_2^2 + \\
&amp; \lambda \sum_{m,k} \| \mathbf{x}_{m,k} \|_1 \text{ s.t. } \| \mathbf{d}_{c,m} \|_2 = 1 \ \forall c, m, \tag{80}
\end{aligned}
$$

where $\mathbf{d}_{c,m}$ is channel $c$ of the $m^{\text{th}}$ dictionary filter, and $\mathbf{s}_{c,k}$ is channel $c$ of the $k^{\text{th}}$ training signal. We will denote the number of channels by $C$. As before, we separately consider the sparse coding and dictionary updates for alternating minimization of this functional.

### A. Sparse Coding

Defining $D_{c,m}$ such that $D_{c,m} \mathbf{x}_{m,k} = \mathbf{d}_{c,m} * \mathbf{x}_{m,k}$, and

$$
D_c = \left( \begin{array}{ccc}
D_{c,0} &amp; D_{c,1} &amp; \dots
\end{array} \right)
\quad
\mathbf{x}_k =
\left( \begin{array}{c}
\mathbf{x}_{0,k} \\
\mathbf{x}_{1,k} \\
\vdots
\end{array} \right), \tag{81}
$$

we can write the sparse coding component of Eq. (80) as

$$
\arg\min_{\{\mathbf{x}_k\}} (1/2) \sum_{c,k} \| D_c \mathbf{x}_k - \mathbf{s}_{c,k} \|_2^2 + \lambda \sum_{m,k} \| \mathbf{x}_k \|_1, \tag{82}
$$

or by defining

$$
D =
\left( \begin{array}{cccc}
D_{0,0} &amp; D_{0,1} &amp; \dots \\
D_{1,0} &amp; D_{1,1} &amp; \dots \\
\vdots &amp; \vdots &amp; \ddots
\end{array} \right) \tag{83}
$$

and

$$
X =
\left( \begin{array}{cccc}
\mathbf{x}_{0,0} &amp; \mathbf{x}_{0,1} &amp; \dots \\
\mathbf{x}_{1,0} &amp; \mathbf{x}_{1,1} &amp; \dots \\
\vdots &amp; \vdots &amp; \ddots
\end{array} \right)
\quad
S =
\left( \begin{array}{cccc}
\mathbf{s}_{0,0} &amp; \mathbf{s}_{0,1} &amp; \dots \\
\mathbf{s}_{1,0} &amp; \mathbf{s}_{1,1} &amp; \dots \\
\vdots &amp; \vdots &amp; \ddots
\end{array} \right), \tag{84}
$$

as

$$
\arg\min_X (1/2) \| D X - S \|_2^2 + \lambda \| X \|_1. \tag{85}
$$

This has the same form as the single-channel MMV problem Eq. (7), and the iterations for an ADMM algorithm to solve it are the same as Eq. (9) - (11). The only significant difference is that $D$ in Sec. II-A is a matrix with a $1 \times M$ block structure, whereas here it has a $C \times M$ block structure. The corresponding frequency domain matrix $\hat{D}^H \hat{D}$ can be decomposed into a set of $N$ components of rank $C$, just as $\hat{X}^H \hat{X}$ with $X$ as in Eq. (27) can be decomposed into a set of $N$ components of rank $K$. Consequently, all of the dictionary update algorithms discussed in Sec. III can also be applied to the multi-channel CSC problem, with the $\mathbf{g}$ step corresponding to the projection onto the dictionary constraint set, e.g. Eq. (31), replaced with a $Y$ step corresponding to the proximal operator of the $\ell_1$ norm, e.g. Eq. (14). The Iterated Sherman-Morrison method is very effective for RGB

9Multi-channel CDL is presented in this section as an extension of the CDL framework of Sec. II and III. Application of the same extension to the masked CDL framework of Sec. IV is straightforward, and is supported in our software implementations [36].

images with only three channels, but for a significantly larger number of channels the best choices would be the ADMM consensus or FISTA methods.

For the FISTA solution, we compute the gradient of the data fidelity term $(1/2)\sum_{c,k}\big{\|}D_{c}\mathbf{x}_{k}-\mathbf{s}_{c,k}\big{\|}_{2}^{2}$ in Eq. (82) in the DFT domain

$\nabla_{\hat{\mathbf{x}}_{k}}\Big{(}\frac{1}{2}\sum_{c}\big{\|}\hat{D}_{c}\hat{\mathbf{x}}_{k}-\hat{\mathbf{s}}_{c,k}\big{\|}_{2}^{2}\Big{)}=\sum_{c}\hat{D}_{c}^{H}\big{(}\hat{D}_{c}\hat{\mathbf{x}}_{k}-\hat{\mathbf{s}}_{c,k}\big{)}\ .$ (86)

In contrast to the ADMM methods, the multi-channel problem is not significantly more challenging than the single channel case, since it simply involves an additional sum over the $C$ channels.

### VI-B Dictionary Update

In developing the dictionary update it is convenient to re-index the variables in Eq. (80), writing the problem as

$\operatorname*{arg\,min}_{\{\mathbf{d}_{m,c}\}}\ \frac{1}{2}\sum_{k,c}\Big{\|}\sum_{m}\mathbf{x}_{k,m}*\mathbf{d}_{m,c}-\mathbf{s}_{k,c}\Big{\|}_{2}^{2}$
$\text{s.t. }\left\|\mathbf{d}_{m,c}\right\|_{2}=1\ \forall m,c\ .$ (87)

Defining $X_{k,m}$, $X_{k}$, $X$ and $C_{\text{PN}}$ as in Sec. II-B, and

\[ \mathbf{d}_{c}=\left(\begin{array}[]{c}\mathbf{d}_{0,c}\\
\mathbf{d}_{1,c}\\
\vdots\end{array}\right)\quad D=\left(\begin{array}[]{ccc}\mathbf{d}_{0,0}&\mathbf{d}_{0,1}&\dots\\
\mathbf{d}_{1,0}&\mathbf{d}_{1,1}&\dots\\
\vdots&\vdots&\ddots\end{array}\right)\ , \] (88)

we can write Eq. (87) as

$\operatorname*{arg\,min}_{\{\mathbf{d}_{c}\}}\left(1/2\right)\sum_{k,c}\big{\|}X_{k}\mathbf{d}_{c}-\mathbf{s}_{k,c}\big{\|}_{2}^{2}+\sum_{c}\iota_{C_{\text{PN}}}(\mathbf{d}_{c})\ ,$ (89)

or in simpler form

$\operatorname*{arg\,min}_{D}\left(1/2\right)\|XD-S\|_{2}^{2}+\iota_{C_{\text{PN}}}(D)\ .$ (90)

It is clear that the structure of $X$ is the same as in the single-channel case and that the solutions for the different channel dictionaries $\mathbf{d}_{c}$ are independent, so that the dictionary update in the multi-channel case is no more computationally challenging than in the single channel case.

### VI-C Relationship between $K$ and $C$

The above discussion reveals an interesting dual relationship between the number of images, $K$, in coefficient map set $X$, and the number of channels, $C$, in dictionary $D$. When solving the CDL problem via proximal algorithms such as ADMM or FISTA, $C$ controls the rank of the most expensive subproblem of the convolutional sparse coding stage in the same way that $K$ controls the rank of the main subproblem of the convolutional dictionary update. In addition, algorithms that are appropriate for the large $K$ case of the dictionary update are also suitable for the large $C$ case of sparse coding, and vice versa.

## VII Results

In this section we compare the computational performance of the various approaches that have been discussed, carefully selecting optimal parameters for each algorithm to ensure a fair comparison.

### VII-A Dictionary Learning Algorithms

Before proceeding to the results of the computational experiments, we summarize the dictionary learning algorithms that will be compared. Instead of using the complete dictionary learning algorithm proposed in each prior work, we consider the primary contribution of these works to be in the dictionary update method, which is incorporated into the CDL algorithm structure that was demonstrated in *[23]* to be most effective: auxiliary variable coupling with a single iteration for each subproblem before alternating. Since the sparse coding stages are the same, the algorithm naming is based on the dictionary update algorithms.

The following CDL algorithms are considered for problem Eq. (1) without a spatial mask

The CDL algorithm is as proposed in *[5]*.
The CDL algorithm is as proposed in *[5]*.
The CDL algorithm uses the dictionary update proposed in *[24]*, but the more effective variable coupling and alternation strategy discussed in *[23]*.
The CDL algorithm uses the dictionary update technique proposed in *[24]*, but the substantially more effective variable coupling and alternation strategy discussed in *[23]*.
The algorithm is the same as Cns, but with a parallel implementation of both the sparse coding and dictionary update stages. All steps of the CSC stage are completely parallelizable in the training image index $k$, as are the $\mathbf{d}$ and $\mathbf{h}$ steps of the dictionary update, the only synchronization point being in the $\mathbf{g}$ step, Eq. (42), where all the independent dictionary estimates are averaged and projected (see Eq. (46)) to update the consensus variable that all the processes share.
The CDL algorithm uses the dictionary update proposed in *[24]*, but the more effective variable coupling and alternation strategy discussed in *[23]*.
Not previously considered for this problem.

The following dictionary learning algorithms are considered for problem Eq. (58) with a spatial mask

Not previously considered for this problem.
The CDL algorithm is as proposed in *[16]*.

##

Extended Consensus (M-Cns) The CDL algorithm is based on a new dictionary update constructed as a hybrid of the dictionary update methods proposed in [9] and [24], with the effective variable coupling and alternation strategy discussed in [23].

Extended Consensus in Parallel (M-Cns-P) The algorithm is the same as M-Cns, but with a parallel implementation of both the sparse coding and dictionary update. All steps of the CSC stage and the  $\mathbf{d}$ ,  $\mathbf{g}_1$ ,  $\mathbf{h}_0$ , and  $\mathbf{h}_1$  steps of the dictionary update are completely parallelizable in the training image index  $k$ , the only synchronization point being in the  $\mathbf{g}_0$  step, Eq. (75), where all the independent dictionary estimates are averaged and projected to update the consensus variable that all the processes share.

FISTA (M-FISTA) Not previously considered for this problem.

In addition to the algorithms listed above, we investigated Stochastic Averaging ADMM (SA-ADMM) [38], as proposed for CDL in [10]. Our implementation of a CDL algorithm based on this method was found to have promising computational cost per iteration, but its convergence was not competitive with some of the other methods considered here. However, since there are a number of algorithm details that are not provided in [10] (CDL is not the primary topic of that work), it is possible that our implementation omits some critical components. These results are therefore not included here in order to avoid making an unfair comparison.

We do not compare with the dictionary learning algorithm in [7] because the algorithms of both [9] and [24] were both reported to be substantially faster. We do not include the algorithms of either [9] and [24] in our main set of experiments because we do not have implementations that are practical to run over the large number of different training image sets and parameter choices that are used in these experiments, but we do include these algorithms in some additional performance comparisons in Sec. SVII of the Supplementary Material. Multi-channel CDL problems are not included in our main set of experiments due to space limitations, but some relevant experiments are provided in Sec. SVIII of the Supplementary Material.

# B. Computational Complexity

The per-iteration computational complexities of the methods are summarized in Table I. Instead of just specifying the dominant terms, we include all major contributing terms to provide a more detailed picture of the computational cost. All methods scale linearly with the number of filters,  $M$ , and with the number of images,  $K$ , except for the ISM variants, which scale as  $\mathcal{O}(K^2)$ . The inclusion of the dependency on  $K$  for the parallel algorithms provides a very conservative view of their behavior. In practice, there is either no scaling or very weak scaling with  $K$  when the number of available cores exceeds  $K$ , and weak scaling with  $K$  when it exceeds the number of available cores. Memory usage depends on the method and implementation, but all the methods have an  $\mathcal{O}(KMN)$  memory requirement for their main variables.

TABLEI COMPUTATIONAL COMPLEXITIES FOR A SINGLE ITERATION OF THE CDL ALGORITHMS, BROKEN DOWN INTO COMPLEXITIES FOR THE SPARSE CODING (CSC AND M-CSC) AND DICTIONARY UPDATE STEPS, WHICH ARE THEMSELVES DECOMPOSED INTO COMPLEXITIES FOR THE FREQUENCY-DOMAIN SOLUTIONS (FFT), THE SOLUTION OF THE FREQUENCY-DOMAIN LINEAR SYSTEMS (LINEAR), THE PROJECTION CORRESPONDING TO THE PROXIMAL MAP OF THE INDICATOR FUNCTION  $\iota_{C_{\mathrm{PS}}}$  (PROX), AND ADDITIONAL OPERATIONS DUE TO A SPATIAL MASK (MASK). THE NUMBER OF Pixels IN THE TRAINING IMAGES, THE NUMBER OF DICTIONARY FILTERS, AND THE NUMBER OF TRAINING IMAGES ARE DENOTED BY  $N$ ,  $M$ , AND  $K$  RESPECTIVELY, AND  $\mathcal{O}_{CG}$  DENOTES THE COMPLEXITY OF SOLVING A LINEAR SYSTEM BY THE CONJUGATE GRADIENT METHOD.

|  Algorithm | Complexity  |   |   |   |
| --- | --- | --- | --- | --- |
|   |  FFT | Linear | Prox | Mask  |
|  CSC | O(KMN log N) | O(KMN) | O(KMN) |   |
|  CG | O(KMN log N) | OCG | O(MN) |   |
|  ISM | O(KMN log N) | O(K2MN) | O(MN) |   |
|  Tiled. 3D | O(KMN (log N + log K)) | O(KMN) | O(MN) |   |
|  Cns. Cns-P. FISTA | O(KMN log N) | O(KMN) | O(MN) |   |
|  M-CSC | O(KMN log N) | O(KMN) | O(KMN) | O(KMN)  |
|  M-CG | O(KMN log N) | OCG + O(KMN) | O(MN) | O(KN)  |
|  M-ISM | O(KMN log N) | O(K2MN) + O(KMN) | O(MN) | O(KN)  |
|  M-Cns M-Cns-P M-FISTA | O(KMN log N) | O(KMN) | O(MN) | O(KN)  |

# C. Experiments

We used training sets of 5, 10, 20, and 40 images. These sets were nested in the sense that all images in a set were also present in all of the larger sets. The parent set of 40 images consisted of greyscale images of size  $256 \times 256$  pixels, derived from the MIRFLICKR-1M dataset[14] [39] by cropping, rescaling, and conversion to greyscale. An additional set of 20 images, of the same size and from the same source, was used as a test set to allow comparison of generalization performance, taking into account possible differences in overfitting effects between the different methods.

The 8 bit greyscale images were divided by 255 so that pixel values were within the interval [0,1], and were high-pass filtered (a common approach for convolutional sparse representations [40], [41], [5][42, Sec. 3]) by subtracting a lowpass component computed by Tikhonov regularization with a gradient term [37, pg. 3], with regularization parameter  $\lambda = 5.0$ .

The results reported here were computed using the Python implementation of the SPORCO library [36], [37] on a Linux workstation equipped with two Xeon E5-2690V4 CPUs.

# D. Optimal Penalty Parameters

To ensure a fair comparison between the methods, the optimal penalty parameters for each method and training

14The image data directly included in the MIRFLICKR-1M dataset is of very low resolution since the dataset is primarily targeted at image classification tasks. We therefore identified and downloaded the original images that were used to construct the MIRFLICKR-1M dataset.

TABLE II DICTIONARY LEARNING: OPTIMAL PARAMETERS FOUND BY GRID SEARCH.

|   |   | Parameter |   |  | Parameter  |   |
| --- | --- | --- | --- | --- | --- | --- |
|  Method | K | ρ | σ | Method | ρ | σ  |
|  CG | 5 | 3.59 | 4.08 | M-CG | 3.59 | 5.99  |
|   |  10 | 3.59 | 12.91 |   | 3.59 | 7.74  |
|   |  20 | 2.15 | 24.48 |   | 2.15 | 7.74  |
|   |  40 | 2.56 | 62.85 |   | 2.49 | 11.96  |
|  ISM | 5 | 3.59 | 4.08 | M-ISM | 3.59 | 5.99  |
|   |  10 | 3.59 | 12.91 |   | 3.59 | 7.74  |
|   |  20 | 2.15 | 24.48 |   | 2.15 | 7.74  |
|   |  40 | 2.56 | 62.85 |   | 2.49 | 11.96  |
|  Tiled | 5 | 3.59 | 7.74 |   |   |   |
|   |  10 | 3.59 | 12.91  |   |   |   |
|   |  20 | 3.59 | 40.84  |   |   |   |
|   |  40 | 3.59 | 72.29  |   |   |   |
|  Cns | 5 | 3.59 | 1.29 | M-Cns | 3.59 | 1.13  |
|   |  10 | 3.59 | 1.29 |   | 3.59 | 0.68  |
|   |  20 | 3.59 | 2.15 |   | 3.59 | 1.13  |
|   |  40 | 3.59 | 1.08 |   | 3.59 | 1.01  |
|  3D | 5 | 3.59 | 7.74 |   |   |   |
|   |  10 | 3.59 | 12.91  |   |   |   |
|   |  20 | 3.59 | 40.84  |   |   |   |
|   |  40 | 3.59 | 72.29  |   |   |   |
|   |   |   | L |   |   |   |
|  FISTA | 5 | 3.59 | 48.14  |   |   |   |
|   |  10 | 3.59 | 92.95  |   |   |   |
|   |  20 | 3.59 | 207.71  |   |   |   |
|   |  40 | 3.59 | 400.00 |   |   |   |

image set were selected via a grid search, of CDL functional values obtained after 100 iterations, over  $(\rho, \sigma)$  values for the ADMM dictionary updates, and over  $(\rho, L)$  values for the FISTA dictionary updates. The grid resolutions were

$\rho$  10 logarithmically spaced points in  $[10^{-1}, 10^{4}]$
$\sigma$  15 logarithmically spaced points in  $[10^{-2}, 10^{5}]$
$L$  15 logarithmically spaced points in  $[10^{1}, 10^{5}]$

The best set of  $(\rho, \sigma)$  or  $(\rho, L)$  for each method i.e. the ones yielding the lowest value of the CDL functional at 100 iterations, was selected as a center for a finer grid search, of CDL functional values obtained after 200 iterations, with 10 logarithmically spaced points in  $[0.1\rho_{\mathrm{center}}, 10\rho_{\mathrm{center}}]$  and 10 logarithmically spaced points in  $[0.1\sigma_{\mathrm{center}}, 10\sigma_{\mathrm{center}}]$  or 10 logarithmically spaced points in  $[0.1L_{\mathrm{center}}, 10L_{\mathrm{center}}]$ . The optimal parameters for each method were taken as those yielding the lowest value of the CDL functional at 200 iterations in this finer grid. This procedure was repeated for sets of 5, 10, 20 and 40 images. As an indication of the sensitivities of the different methods to their parameters, results for the coarse grid search for the 20 image set can be found in Sec. SII in the Supplementary Material. The optimal parameters determined via these grid searches are summarized in Table II.

# E. Performance Comparisons

We compare the performance of the methods in learning a dictionary of 64 filters of size  $8 \times 8$  for sets of 5, 10, 20 and 40 images, setting the sparsity parameter  $\lambda = 0.1$ , and using the parameters determined by the grid searches for each method.

![images/image1.jpg](images/image1.jpg)
Fig. 1. Dictionary Learning ( $K = 5$ ): A comparison on a set of  $K = 5$  images of the decay of the value of the CBPDN functional Eq. (5) with respect to run time and iterations. ISM, Tiled, Cns and 3D overlap in the time plot, and Cns and Cns-P overlap in the iterations plot.

![images/image2.jpg](images/image2.jpg)
Fig. 2. Dictionary Learning ( $K = 20$ ): A comparison on a set of  $K = 20$  images of the decay of the value of the CBPDN functional Eq. (5) with respect to run time and iterations. Cns and 3D overlap in the time plot, and Cns, Cns-P and 3D overlap in the iterations plot.

![images/image3.jpg](images/image3.jpg)
Fig. 3. Dictionary Learning ( $K = 40$ ): A comparison on a set of  $K = 40$  images of the decay of the value of the CPBDN functional Eq. (5) with respect to run time and iterations. Cns and 3D overlap in the time plot, and Cns, Cns-P and 3D overlap in the iterations plot.

To avoid complicating the comparisons, we used fixed penalty parameters  $\rho$  and  $\sigma$ , without any adaptation methods [5, Sec. III.D][43], and did not apply relaxation methods [17, Sec. 3.4.3][5, Sec. III.D] in any of the ADMM algorithms. Similarly, we used a fixed  $L$  for FISTA, without applying any backtracking step-size adaptation rule. Performance in terms of the convergence rate of the CDL functional, with respect to

![images/image4.jpg](images/image4.jpg)
(a) Without Spatial Mask

![images/image5.jpg](images/image5.jpg)
(b) With Spatial Mask

both iterations and computation time, is compared in Figs. 1 - 3. The time scaling with  $K$  of all the methods is summarized in Fig. 4(a).

For the  $K = 5$  case, all the methods have quite similar performance in terms of functional value convergence with respect to iterations. For the larger training set sizes, CG and ISM have somewhat better performance with respect to iterations, but ISM has very poor performance with respect to time. CG has substantially better time scaling, depending on the relative residual tolerance. We ran our experiments for CG with a fixed tolerance of  $10^{-3}$ , resulting in computation times that are comparable with those of the other methods. A smaller tolerance leads to better convergence with respect to iterations, but substantially worse time performance.

The "3D" method behaves similarly to ADMM consensus, as expected from the relationship established in Sec. III-C, but has a larger memory footprint. The spatial tiling method (Tiled), on the other hand, tends to have slower convergence with respect to both iterations and time than the other methods. We do not further explore the performance of these methods since they do not provide substantial advantages over the others.

Both parallel (Cns-P) and regular consensus (Cns) have the same evolution of the CBPDN functional, Eq. (5), with respect to iterations, but the former requires much less computation time, and is the fastest method overall. Moreover, parallel consensus exhibits almost ideal parallelizability, with some overhead for  $K = 5$ , but scaling linearly for  $K \in [10,40]$ , and with very competitive computation times. FISTA is also very competitive, achieving good results in less time than any of the other serial methods, and even outperforming the time performance of Cns-P for the  $K = 40$  case shown in Fig. 3. We believe that this variation of relative performance with  $K$  is due to the unstable dependence of the CDL functional on  $L$  that is illustrated, for example, in Fig. 10(b) in the Supplementary Material. This functional decreases slowly as  $L$  is decreased, but then increases very rapidly after the minimum is reached, due to the constraint on  $L$  discussed in Sec. VII-G2.

All experiments with algorithms that include a spatial mask set the mask to the identity  $(W = I)$  to allow comparison with the performance of the algorithms without a spatial mask. Plots comparing the evolution of the masked CBPDN

![images/image6.jpg](images/image6.jpg)
Fig. 4. Comparison of time per iteration for the dictionary learning methods for sets of 5, 10, 20 and 40 images.
Fig. 5. Dictionary Learning with Spatial Mask ( $K = 5$ ): A comparison on a set of  $K = 5$  images of the decay of the value of the masked CBPDN functional Eq. (59) with respect to run time and iterations for masked versions of the algorithms. M-Cns and M-Cns-P overlap in the iterations plot.

![images/image7.jpg](images/image7.jpg)
Fig. 6. Dictionary Learning with Spatial Mask ( $K = 20$ ): A comparison on a set of  $K = 20$  images of the decay of the value of the masked CBPDN functional Eq. (59) with respect to run time and iterations for masked versions of the algorithms. M-Cns and M-Cns-P overlap in the iterations plot.

![images/image8.jpg](images/image8.jpg)
Fig. 7. Dictionary Learning with Spatial Mask ( $K = 40$ ): A comparison on a set of  $K = 40$  images of the decay of the value of the masked CBPDN functional Eq. (59) with respect to run time and iterations for masked versions of the algorithms. M-Cns and M-Cns-P overlap in the iterations plot.

functional, Eq. (59), over 1000 iterations and problem sizes of  $K \in \{5,20,40\}$  are displayed in Figs. 5 - 7, respectively. The time scaling of all the masked methods is summarized in Fig. 4(b).

While the convergence performance with iterations of the masked version of the FISTA algorithm, M-FISTA, is mixed (providing the worst performance for  $K = 5$  and  $K = 20$ ,

but the best performance for  $K = 40$ , it consistently provides good performance in terms of convergence with respect to computation time, despite the additional FFTs discussed in Sec. V-C. The parallel hybrid mask decoupling/consensus method, M-Cns-P, is the other competitive approach for this problem, providing the best time performance for  $K = 5$  and  $K = 20$ , while lagging slightly behind M-FISTA for  $K = 40$ .

In contrast with the corresponding mask-free variants, M-CG and M-ISM have worse performance in terms of both time and iterations. This suggests that M-CG requires a value for the relative residual tolerance smaller than  $10^{-3}$  to produce good results, but this would be at the expense of much longer computation times. With the exception of CG, for which the cost of computing the masked version increases for  $K \geq 20$ , the computation time for the masked versions is only slightly worse than the mask-free variants (Fig. 4). In general, using the masked versions leads to a marginal decrease in convergence rate with respect to iterations, and a small increase in computation time.

# F. Evaluation on the Test Set

![images/image9.jpg](images/image9.jpg)
Fig. 8. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 20$  images. Tiled, Cns and 3D overlap in the time plot, and Cns and Cns-P overlap in the iterations plot.

![images/image10.jpg](images/image10.jpg)
Fig. 9. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 40$  images. Tiled, Cns and 3D have a large overlap in the time plot, and Cns and Cns-P overlap in the iterations plot.

To provide a comparison that takes into account any possible differences in overfitting and generalization properties of the

![images/image11.jpg](images/image11.jpg)
Fig. 10. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 20$  images for masked versions of the algorithms. M-Cns and M-Cns-P overlap in the iterations plot.

![images/image12.jpg](images/image12.jpg)
Fig. 11. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 40$  images for masked versions of the algorithms. M-Cns and M-Cns-P overlap in the iterations plot.

dictionaries learned by the different methods, we ran experiments over a 20 image test set that is not used during learning. For all the methods discussed, we saved the dictionaries at 50 iteration intervals (including the final one obtained at 1000 iterations) while training. These dictionaries were used to sparse code the images in the test set with  $\lambda = 0.1$ , allowing evaluation of the evolution of the test set CBPDN functional as the dictionaries change during training. Results for the dictionaries learned while training with  $K = 20$  and  $K = 40$  images are shown in Figs. 8 and 9 respectively, and corresponding results for the algorithms with a spatial mask are shown in Figs. 10 and 11 respectively. Note that the time axis in these plots refers to the run time of the dictionary learning code used to generate the relevant dictionary, and not to the run time of the sparse coding on the test set.

As expected, independent of the method, the dictionaries obtained for training with 40 images exhibit better performance than the ones trained with 20 images. Overall, performance on training is a good predictor of performance in testing, which suggests that the functional value on a sufficiently large training set is a reliable indicator of dictionary quality.

# G. Penalty Parameter Selection

The grid searches performed for determining optimal parameters ensure a fair comparison between the methods, but

they are not convenient as a general approach to parameter selection. In this section we show that it is possible to construct heuristics that allow reliable parameter selection for the best performing CDL methods considered here.

1) Parameter Scaling Properties: Estimates of parameter scaling properties with respect to  $K$  are derived in Sec. SIII in the Supplementary Material. For the CDL problem without a spatial mask, these scaling properties are derived for the sparse coding problem, and for the dictionary updates based on ADMM with an equality constraint, ADMM consensus, and FISTA. These estimates indicate that the scaling of the penalty parameter  $\rho$  for the convolutional sparse coding is  $\mathcal{O}(1)$ , the scaling of the penalty parameter  $\sigma$  for the dictionary update is  $\mathcal{O}(K)$  for the ADMM with equality constraint and  $\mathcal{O}(1)$  for ADMM consensus, and the scaling of the step size  $L$  for FISTA is  $\mathcal{O}(K)$ . Derivations for the Tiled and 3D methods do not lead to a simple scaling relationship, and are not included.

For the CDL problem with a spatial mask, these scaling properties are derived for the sparse coding problem, and for the dictionary updates based on ADMM with a block-constraint, and extended ADMM consensus. The scaling of the penalty parameter  $\rho$  for the masked version of convolutional sparse coding is  $\mathcal{O}(1)$ , the scaling of the penalty parameter  $\sigma$  for the dictionary update in the extended consensus framework is  $\mathcal{O}(1)$ , while there is no simple rule of the  $\sigma$  scaling in the block-constraint ADMM of Sec. V-A.

2) Parameter Selection Guidelines: The derivations discussed above indicate that the optimal algorithm parameters should be expected to be either constant or linear in  $K$ . For the parameters of the most effective CDL algorithms, i.e. CG, Cns, FISTA, and M-Cns, we performed additional computational experiments to estimate the constants in these scaling relationships. Cns-P and M-Cns-P have the same parameter dependence as their serial counterparts, and are therefore not evaluated separately. Similarly, M-FISTA is not included in these experiments because it has the same functional evolution as FISTA for the identity mask  $W = I$ .

TABLE III GRID SEARCH RANGES

|  Parameter | Method | Range  |
| --- | --- | --- |
|  ρ | CG | [100.1, 101.1]  |
|   |  Cns | [100.25, 101.2]  |
|   |  M-Cns | [100.33, 10]  |
|   |  FISTA | [100.14, 10]  |
|  σ | CG | [1, 102.5]  |
|   |  Cns, M-Cns | [10-1, 10]  |
|  L | FISTA | [10, 102.9]  |

For each training set size  $K \in \{5, 10, 20\}$ , we constructed an ensemble of 20 training sets of that size by random selection from the 40 image training set. For each CDL algorithm and each  $K$ , the dependence of the convergence behavior on the algorithm parameters was evaluated by computing 500 iterations of the CDL algorithm for all 20 members of the ensemble of size  $K$ , and over grids of  $(\rho, \sigma)$  values for the ADMM dictionary updates, and  $(\rho, L)$  values for

the FISTA dictionary updates. The parameter grids consisted of 10 logarithmically spaced points in the ranges specified in Table III. These parameter ranges were set such that the corresponding functional values remained within  $0.1\%$  to  $1\%$  of their optimal values.

![images/image13.jpg](images/image13.jpg)
(a) CG  $\rho$

![images/image14.jpg](images/image14.jpg)
(b) CG  $\sigma$

![images/image15.jpg](images/image15.jpg)
(c) Cns  $\rho$

![images/image16.jpg](images/image16.jpg)
(d) Cns  $\sigma$

![images/image17.jpg](images/image17.jpg)
(e) M-Cns  $\rho$

![images/image18.jpg](images/image18.jpg)
(f) M-Cns  $\sigma$

![images/image19.jpg](images/image19.jpg)
(g) FISTA  $\rho$

![images/image20.jpg](images/image20.jpg)
(h) FISTA  $L$
Fig. 12. Contour plots of the ensemble median of the normalized CDL functional values for different algorithm parameters. The black lines correspond to level curves at the indicated values of the plotted surfaces, and the dashed red lines represent parameter selection guidelines that combine the analytic derivations with the empirical behavior of the plotted surfaces.

We normalized the results for each training set by dividing by the minimum of the functional for that set, and computed statistics over these normalized values for all sets of the same size,  $K$ . These statistics, which are reported as box plots in Sec. SIV of the Supplementary Material, were also aggregated into contour plots of the median (across the ensemble of training images sets of the same size) of the normalized CDL functional values, displayed in Fig. 12. (Results for ISM are the same as for CG and are not shown.) In each of these

TABLE IV PENALTY PARAMETER SELECTION GUIDELINES

|  Parameter | Method | Rule  |
| --- | --- | --- |
|  ρ | CG, ISM, FISTA | ρ = 2.2  |
|   |  Cns | ρ = 3.0  |
|   |  M-Cns | ρ = 2.7  |
|  σ | CG, ISM | σ = 0.5K + 7.0  |
|   |  Cns | σ = 2.2  |
|   |  M-Cns | σ = 3.0  |
|  L | FISTA | L = 14.0K  |

contour plots, the horizontal axis corresponds to the number of training images,  $K$ , and the vertical axis corresponds to the parameter of interest. The scaling behavior of the optimal parameter with  $K$  can clearly be seen in the direction of the valley in the contour plots. Parameter selection guidelines obtained by manual fitting of the constant or linear scaling behavior to these contour plots are plotted in red, and are also summarized in Table IV.

In Fig. 12(f), the guideline for  $\sigma$  for M-Cns does not appear to follow the path of the 1.001 level curves. We did not select the guideline to follow this path because (i) the theoretical estimate of the scaling properties of this parameter with  $K$  in Sec. SIII-G of the Supplementary Material is that it is constant, and (ii) the path suggested by the 1.001 level curves leads to a logarithmically decreasing curve that would reach negative parameter values for sufficiently large  $K$ . We do not have a reliable explanation for the unexpected behavior of the 1.001 level curves, but suspect that it may be related to the loss of diversity of training image sets for  $K = 20$ , since each of these sets of 20 images was chosen from a fixed set of 40 images. It is also worth noting that the upper level curves for larger functional values, e.g. 1.002, do not follow the same unexpected decreasing path.

To guarantee convergence of FISTA, the inverse of the gradient step size,  $L$ , has to be greater than or equal to the Lipschitz constant of the gradient of the functional [33]. In Fig. 12(h), the level curves below the guideline correspond to this potentially unstable regime where the functional value surface has a large gradient. The gradient of the surface is much smaller above the guideline, indicating that convergence is not very sensitive to the parameter value in this region. We chose the guideline precisely to be more biased towards the stable regime.

The parameter selection guidelines presented in this section should only be expected to be reliable for training data with similar characteristics to those used in our experiments, i.e. natural images pre-processed as described in Sec. VII-C, and for the same or similar sparsity parameter, i.e.  $\lambda = 0.1$ . Nevertheless, since the scaling properties derived in Sec. SIII of the Supplementary Material remain valid, it is reasonable to expect that similar heuristics, albeit with different constants, would hold for different training data or sparsity parameter settings.

# VIII. CONCLUSIONS

Our results indicate that two distinct approaches to the dictionary update problem provide the leading CDL algorithms. In a serial processing context, the FISTA dictionary update proposed here outperforms all other methods, including consensus, for CDL with and without a spatial mask. This may seem surprising when considering that ADMM outperforms FISTA on the CSC problem, but is easily understood when taking into account the critical difference between the linear systems that need to be solved when tackling the CSC and convolutional dictionary update problems via proximal methods such as ADMM and FISTA. In the case of CSC, the major linear system to be solved has a frequency domain structure that allows very efficient solution via the Sherman-Morrison formula, providing an advantage to ADMM. In contrast, except for the  $K = 1$  case, there is no such highly efficient solution for the convolutional dictionary update, giving an advantage to methods such as FISTA that employ gradient descent steps rather than solving the linear system.

In a parallel processing context, the consensus dictionary update proposed in [24] used together with the alternative CDL algorithm structure proposed in [23] leads to the CDL algorithm with the best time performance for the mask-free CDL problem, and the hybrid mask decoupling/consensus dictionary update proposed here provides the best time performance for the masked CDL problem. It is interesting to note that, despite the clear suitability of the ADMM consensus framework for the convolutional dictionary update problem, a parallel implementation is essential to outperforming other methods; in a serial processing context it is significantly outperformed by the FISTA dictionary update, and even the CG method is competitive with it.

We have also demonstrated that the optimal algorithm parameters for the leading methods considered here tend to be quite stable across different training sets of similar type, and have provided reliable heuristics for selecting parameters that provide good performance. It should be noted, however, that FISTA appears to be more sensitive to the  $L$  parameter than the ADMM methods are to the penalty parameter.

The additional experiments reported in the Supplementary Material indicate that the FISTA and parallel consensus methods are scalable to relatively large training sets, e.g. 100 images of  $512 \times 512$  pixels. The computation time exhibits linear scaling in the number of training images,  $K$ , and the number of dictionary filters,  $M$ , and close to linear scaling in the number of pixels in each image,  $N$ . The limited experiments involving color dictionary learning indicate that the additional computational cost compared with greyscale dictionary learning is moderate. Comparisons with the publicly available implementations of complete CDL methods by other authors indicate that:

- The method of Heide et al. [9] does not scale well to training images sets of even moderate size, exhibiting very slow convergence with respect to computation time.
- While the consensus CDL method proposed here gives very good performance, the consensus method of Sorel

and Šroubek *[24]* converges much more slowly, and does not learn dictionaries with properly normalized filters.
- The method of Papyan et al. *[27]* converges rapidly with respect to the number of iterations, and appears to scale well with training set size, but is slower than the FISTA and parallel consensus methods with respect to time, and the resulting dictionaries do not offer competitive performance to the leading methods proposed here in terms of performance on testing image sets.

In the interest of reproducible research, software implementations of the algorithms considered here have been made publicly available as part of the SPORCO library *[36, 37]*.

## References

- [1] J. Mairal, F. Bach, and J. Ponce, “Sparse modeling for image and vision processing,” *Foundations and Trends in Computer Graphics and Vision*, vol. 8, no. 2-3, pp. 85–283, 2014. doi:10.1561/0600000058
- [2] M. A. T. Figueiredo, “Synthesis versus analysis in patch-based image priors,” in *Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)*, Mar. 2017, pp. 1338–1342. doi:10.1109/ICASSP.2017.7952374
- [3] M. S. Lewicki and T. J. Sejnowski, “Coding time-varying signals using sparse, shift-invariant representations,” in *Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 11, 1999, pp. 730–736.
- [4] M. D. Zeiler, D. Krishnan, G. W. Taylor, and R. Fergus, “Deconvolutional networks,” in *Proc. IEEE Conf. Comp. Vis. Pat. Recog. (CVPR)*, Jun. 2010, pp. 2528–2535. doi:10.1109/cvpr.2010.5539957
- [5] B. Wohlberg, “Efficient algorithms for convolutional sparse representations,” *IEEE Trans. Image Process.*, vol. 25, no. 1, pp. 301–315, Jan. 2016. doi:10.1109/TIP.2015.2495260
- [6] R. Chalasani, J. C. Principe, and N. Ramakrishnan, “A fast proximal method for convolutional sparse coding,” in *Proc. Int. Joint Conf. Neural Net. (IJCNN)*, Aug. 2013. doi:10.1109/IJCNN.2013.6706854
- [7] H. Bristow, A. Eriksson, and S. Lucey, “Fast convolutional sparse coding,” in *Proc. IEEE Conf. Comp. Vis. Pat. Recog. (CVPR)*, Jun. 2013, pp. 391–398. doi:10.1109/CVPR.2013.57
- [8] B. Wohlberg, “Efficient convolutional sparse coding,” in *Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)*, May 2014, pp. 7173–7177. doi:10.1109/ICASSP.2014.6854992
- [9] F. Heide, W. Heidrich, and G. Wetzstein, “Fast and flexible convolutional sparse coding,” in *Proc. IEEE Conf. Comp. Vis. Pat. Recog. (CVPR)*, 2015, pp. 5135–5143. doi:10.1109/CVPR.2015.7299149
- [10] S. Gu, W. Zuo, Q. Xie, D. Meng, X. Feng, and L. Zhang, “Convolutional sparse coding for image super-resolution,” in *Proc. IEEE Intl. Conf. Comput. Vis. (ICCV)*, Dec. 2015. doi:10.1109/ICCV.2015.212
- [11] Y. Liu, X. Chen, R. K. Ward, and Z. J. Wang, “Image fusion with convolutional sparse representation,” *IEEE Signal Process. Lett.*, 2016. doi:10.1109/lsp.2016.2618776
- [12] H. Zhang and V. Patel, “Convolutional sparse coding-based image decomposition,” in *British Mach. Vis. Conf. (BMVC)*, York, UK, Sep. 2016, pp. 125.1–125.11. doi:10.5244/C.30.125
- [13] T. M. Quan and W.-K. Jeong, “Compressed sensing reconstruction of dynamic contrast enhanced MRI using GPU-accelerated convolutional sparse coding,” in *IEEE Intl. Symp. Biomed. Imag. (ISBI)*, Apr. 2016, pp. 518–521. doi:10.1109/ISBI.2016.7493321
- [14] A. Serrano, F. Heide, D. Gutierrez, G. Wetzstein, and B. Masia, “Convolutional sparse coding for high dynamic range imaging,” *Computer Graphics Forum*, vol. 35, no. 2, pp. 153–163, May 2016. doi:10.1111/cgf.12819
- [15] H. Zhang and V. M. Patel, “Convolutional sparse and low-rank coding-based rain streak removal,” in *Proc. IEEE Winter Conference on Applications of Computer Vision (WACV)*, March 2017. doi:10.1109/WACV.2017.145
- [16] B. Wohlberg, “Boundary handling for convolutional sparse representations,” in *Proc. IEEE Conf. Image Process. (ICIP)*, Phoenix, AZ, USA, Sep. 2016, pp. 1833–1837. doi:10.1109/ICIP.2016.7532675
- [17] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, “Distributed optimization and statistical learning via the alternating direction method of multipliers,” *Foundations and Trends in Machine Learning*, vol. 3, no. 1, pp. 1–122, 2010. doi:10.1561/2200000016
- [18] J. Liu, C. Garcia-Cardona, B. Wohlberg, and W. Yin, “Online convolutional dictionary learning,” in *Proc. IEEE Conf. Image Process. (ICIP)*, Beijing, China, Sep. 2017, pp. 1707–1711. doi:10.1109/ICIP.2017.8296573. 1706.09563
- [19] K. Degraux, U. S. Kamilov, P. T. Boufounos, and D. Liu, “Online convolutional dictionary learning for multimodal imaging,” in *Proc. IEEE Conf. Image Process. (ICIP)*, Beijing, China, Sep. 2017, pp. 1617–1621. doi:10.1109/ICIP.2017.8296555. 1706.04256
- [20] Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni, “Scalable online convolutional sparse coding,” *IEEE Transactions on Image Processing*, vol. 27, no. 10, pp. 4850–4859, Oct. 2018. doi:10.1109/TIP.2018.2842152. arXiv:1706.06972
- [21] J. Liu, C. Garcia-Cardona, B. Wohlberg, and W. Yin, “First and second order methods for online convolutional dictionary learning,” *SIAM J. Imaging Sci.*, vol. 11, no. 2, pp. 1589–1628, 2018. doi:10.1137/17M1145689. arXiv:1709.00106
- [22] B. Kong and C. C. Fowlkes, “Fast convolutional sparse coding (FCSC),” University of California, Irvine, Tech. Rep., May 2014.
- [23] C. Garcia-Cardona and B. Wohlberg, “Subproblem coupling in convolutional dictionary learning,” in *Proc. IEEE Conf. Image Process. (ICIP)*, Beijing, China, Sep. 2017, pp. 1697–1701. doi:10.1109/ICIP.2017.8296571
- [24] M. Šorel and F. Šroubek, “Fast convolutional sparse coding using matrix inversion lemma,” *Digital Signal Processing*, 2016. doi:10.1016/j.dsp.2016.04.012
- [25] M. S. C. Almeida and M. A. T. Figueiredo, “Deconvolving images with unknown boundaries using the alternating direction method of multipliers,” *IEEE Trans. Image Process.*, vol. 22, no. 8, pp. 3074–3086, Aug. 2013. doi:10.1109/tip.2013.2258354
- [26] M. Jas, T. Dupré la Tour, U. Şimşekli, and A. Gramfort, “Learning the morphology of brain signals using alpha-stable convolutional sparse coding,” in *Advances in Neural Information Processing Systems* (St. I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., 2017, pp. 1099–1108, arXiv:1705.08006.
- [27] V. Papyan, Y. Romano, J. Sulam, and M. Elad, “Convolutional dictionary learning via local processing,” in *Proc. IEEE Int. Conf. Comp. Vis. (ICCV)*, Venice, Italy, Oct. 2017, pp. 5306–5314. doi:10.1109/ICCV.2017.566. arXiv:1705.03239
- [28] I. Y. Chun and J. A. Fessler, “Convolutional dictionary learning: Acceleration and convergence,” *IEEE Trans. Image Process.*, vol. 27, no. 4, pp. 1697–1712, Apr. 2018. doi:10.1109/TIP.2017.2761545
- [29] S. S. Chen, D. L. Donoho, and M. A. Saunders, “Atomic decomposition by basis pursuit,” *SIAM J. Sci. Comput.*, vol. 20, no. 1, pp. 33–61, 1998. doi:10.1137/S1064827596304010
- [30] N. Parikh and S. Boyd, “Proximal algorithms,” *Foundations and Trends in Optimization*, vol. 1, no. 3, pp. 127–239, 2014. doi:10.1561/2400000003
- [31] K. Engan, S. O. Aase, and J. H. Husøy, “Method of optimal directions for frame design,” in *Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)*, vol. 5, 1999, pp. 2443–2446. doi:10.1109/icassp.1999.760624
- [32] M. V. Afonso, J. M. Bioucas-Dias, and M. A. T. Figueiredo, “An Augmented Lagrangian approach to the constrained optimization formulation of imaging inverse problems,” *IEEE Trans. Image Process.*, vol. 20, no. 3, pp. 681–695, Mar. 2011. doi:10.1109/tip.2010.2076294
- [33] A. Beck and M. Teboulle, “A fast iterative shrinkage-thresholding algorithm for linear inverse problems,” *SIAM Journal on Imaging Sciences*, vol. 2, no. 1, pp. 183–202, 2009. doi:10.1137/080716542
- [34] B. Wohlberg, “Endogenous convolutional sparse representations for translation invariant image subspace models,” in *Proc. IEEE Conf. Image Process. (ICIP)*, Paris, France, Oct. 2014, pp. 2859–2863. doi:10.1109/ICIP.2014.7025578
- [35] ——, “Convolutional sparse representation of color images,” in *Proc. IEEE Southwest Symp. Image Anal. Interp. (SSIAI)*, Santa Fe, NM, USA, Mar. 2016, pp. 57–60. doi:10.1109/SSIAI.2016.7459174
- [36] ——, “SParse Optimization Research COde (SPORCO),” Software library available from http://purl.org/brendt/software/sporco, 2016.
- [37] ——, “SPORCO: A Python package for standard and convolutional sparse representations,” in *Proceedings of the 15th Python in Science Conference*, Austin, TX, USA, Jul. 2017, pp. 1–8. doi:10.25080/shinma-7f4c6e7-001

[38] L. W. Zhong and J. T. Kwok, “Fast stochastic alternating direction method of multipliers,” in *Proc. Intl. Conf. Mach. Learn (ICML)*, Beijing, China, 2014, pp. 46–54.
- [39] M. J. Huiskes, B. Thomee, and M. S. Lew, “New trends and ideas in visual concept detection: The MIR Flickr retrieval evaluation initiative,” in *Proc. International Conference on Multimedia Information Retrieval (MIR ’10)*, 2010, pp. 527–536. doi:10.1145/1743384.1743475
- [40] K. Kavukcuoglu, P. Sermanet, Y. Boureau, K. Gregor, M. Mathieu, and Y. LeCun, “Learning convolutional feature hierarchies for visual recognition,” in *Adv. Neural Inf. Process. Syst. (NIPS)*, 2010, pp. 1090–1098.
- [41] M. D. Zeiler, G. W. Taylor, and R. Fergus, “Adaptive deconvolutional networks for mid and high level feature learning,” in *Proc. IEEE Int. Conf. Comp. Vis. (ICCV)*, Barcelona, Spain, Nov. 2011, pp. 2018–2025. doi:10.1109/iccv.2011.6126474
- [42] B. Wohlberg, “Convolutional sparse representations as an image model for impulse noise restoration,” in *Proc. IEEE Image, Video Multidim. Signal Process. Workshop (IVMSP)*, Bordeaux, France, Jul. 2016. doi:10.1109/IVMSPW.2016.7528229
- [43] ——, “ADMM penalty parameter selection by residual balancing,” arXiv, Tech. Rep. 1704.06209, Apr. 2017.

# Convolutional Dictionary Learning: A Comparative Review and New Algorithms (Supplementary Material)

# SI. INTRODUCTION

This document provides additional detail and results that were omitted from the main document due to space restrictions. All citations refer to the References section of the main document.

# SII. PENALTY PARAMETER GRID SEARCH

The penalty parameter grid searches discussed in Sec. VII-D in the main document generate 2D surfaces representing the CDL functional value after a fixed number of iterations, plotted against the parameters for the sparse coding and dictionary update components of the dictionary learning algorithm. The surfaces corresponding to the coarse grids for the set of 20 training images are shown here in Figs. S1 - S3.

![images/image21.jpg](images/image21.jpg)
(a) CG

![images/image22.jpg](images/image22.jpg)
(b) ISM

![images/image23.jpg](images/image23.jpg)
Fig. S1. Grid search surfaces for conjugate gradient (CG) and Iterated Sherman-Morrison (ISM) algorithms with  $K = 20$ . Each surface represents the value of the CBPDN functional (Eq. (5) in the main document) after 100 iterations, for different parameters  $\rho$  and  $\sigma$ .
(a) Tiled

![images/image24.jpg](images/image24.jpg)
(b) Cns

![images/image25.jpg](images/image25.jpg)
(c) 3D
Fig. S2. Grid search surfaces for spatial tiling (Tiled), consensus (Cns), frequency domain consensus (3D) and FISTA algorithms with  $K = 20$ . Each surface represents the value of the CBPDN functional (Eq. (5) in the main document) after 100 iterations, for different parameters  $\rho$ , and  $\sigma$  or  $L$ .

![images/image26.jpg](images/image26.jpg)
(d) FISTA

# SIII. ANALYTIC DERIVATION OF PENALTY PARAMETER SCALING

In order to estimate the scaling properties of the algorithm parameters with respect to the training set size,  $K$ , we consider the case in which the training set size is changed by replication of the same data. By removing the complexities associated with the characteristics of individual images, this simplified scenario allows analytic evaluation of the conditions under which an equivalent problem is obtained when the set size,  $K$ , is changed. In practice, changing  $K$  involves introducing different training images, and we cannot expect that these scaling properties will hold exactly, but they represent the best possible estimate that depends only on  $K$  and not on the properties of the training images themselves.

The following properties of the Frobenius norm,  $\ell_2$  norm, and  $\ell_1$  norm play an important role in these derivations:

![images/image27.jpg](images/image27.jpg)
(a) M-CG

![images/image28.jpg](images/image28.jpg)
(b) M-ISM

![images/image29.jpg](images/image29.jpg)
(c) M-Cns

![images/image30.jpg](images/image30.jpg)
(d) M-FISTA
Fig. S3. Grid search surfaces for masked conjugate gradient (M-CG), masked iterated Sherman-Morrison (M-ISM), masked consensus (M-Cns) and masked FISTA (M-FISTA) algorithms with  $K = 20$ . Each surface represents the value of the masked CBPDN functional (Eq. (59) in the main document) after 100 iterations, for different parameters  $\rho$ , and  $\sigma$  or  $L$ .

In order to estimate the scaling properties of the algorithm parameters with respect to the training set size,  $K$ , we consider the case in which the training set size is changed by replication of the same data. By removing the complexities associated with the characteristics of individual images, this simplified scenario allows analytic evaluation of the conditions under which an equivalent problem is obtained when the set size,  $K$ , is changed. In practice, changing  $K$  involves introducing different training images, and we cannot expect that these scaling properties will hold exactly, but they represent the best possible estimate that depends only on  $K$  and not on the properties of the training images themselves.

The following properties of the Frobenius norm,  $\ell_2$  norm, and  $\ell_1$  norm play an important role in these derivations:

$$
\left\| \left( \begin{array}{l l} \mathbf {x} &amp; \mathbf {y} \end{array} \right) \right\| _ {F} ^ {2} = \| \mathbf {x} \| _ {2} ^ {2} + \| \mathbf {y} \| _ {2} ^ {2} \tag {S1}
$$

$$
\left\| \left( \begin{array}{l} X \\ Y \end{array} \right) \right\| _ {F} ^ {2} = \| X \| _ {F} ^ {2} + \| Y \| _ {F} ^ {2} \tag {S2}
$$

$$
\left\| \left( \begin{array}{l l} \mathbf {x} &amp; \mathbf {y} \end{array} \right) \right\| _ {1} = \| \mathbf {x} \| _ {1} + \| \mathbf {y} \| _ {1} \tag {S3}
$$

$$
\left\| \left( \begin{array}{l} X \\ Y \end{array} \right) \right\| _ {1} = \| X \| _ {1} + \| Y \| _ {1}. \tag {S4}
$$

We will also make use of the invariance of the indicator function under scalar multiplication

$$
\alpha \iota_ {C} (\mathbf {x}) = \iota_ {C} (\mathbf {x}) \quad \forall \alpha &gt; 0, \tag {S5}
$$

which is due to the  $\{0,\infty\}$  range of this function.

### II-A ADMM Sparse Coding

The augmented Lagrangian for the ADMM solution to CSC problem Eq. (12) in the main document is

$L_{\rho}(X,Y,U)=$
$\frac{1}{2}\left\|DX-S\right\|_{F}^{2}+\lambda\left\|Y\right\|_{1}+\frac{\rho}{2}\left\|X-Y+U\right\|_{F}^{2}\ ,$ (S6)

where we omit the final term, $-\frac{\rho}{2}\left\|U\right\|_{F}^{2}$, which does not effect the minimizer of this functional. For $K=1$ we have $S=\mathbf{s}$, $X=\mathbf{x}$, $Y=\mathbf{y}$, and $U=\mathbf{u}$. If we construct the $K=2$ case by replicating the training data, we have $S^{\prime}=\left(\begin{array}[]{cccc}\mathbf{s}&\mathbf{s}\end{array}\right)$, $X^{\prime}=\left(\begin{array}[]{cccc}\mathbf{x}&\mathbf{x}\end{array}\right)$, $Y^{\prime}=\left(\begin{array}[]{cccc}\mathbf{y}&\mathbf{y}\end{array}\right)$, and $U^{\prime}=\left(\begin{array}[]{cccc}\mathbf{u}&\mathbf{u}\end{array}\right)$, and the augmented Lagrangian is

$L_{\rho}(X^{\prime},Y^{\prime},U^{\prime})=$
$\frac{1}{2}\left\|DX^{\prime}-S^{\prime}\right\|_{F}^{2}+\lambda\left\|Y^{\prime}\right\|_{1}+\frac{\rho}{2}\left\|X^{\prime}-Y^{\prime}+U^{\prime}\right\|_{F}^{2}$
$=2\frac{1}{2}\left\|D\mathbf{x}-\mathbf{s}\right\|_{2}^{2}+2\lambda\left\|\mathbf{y}\right\|_{1}+2\frac{\rho}{2}\left\|\mathbf{x}-\mathbf{y}+\mathbf{u}\right\|_{2}^{2}$
$=2L_{\rho}(X,Y,U)\ .$ (S7)

For this problem, the augmented Lagrangian for the $K=2$ case is just twice the augmented Lagrangian for the $K=1$ case, with the same penalty parameter $\rho$. Therefore we expect that the optimal penalty parameter should remain constant when changing the number of training images $K$.

### II-B Equality Constrained ADMM Dictionary Update

The augmented Lagrangian for the ADMM solution to the dictionary update problem Eq. (29) in the main document is

$L_{\sigma}(\mathbf{d},\mathbf{g},\mathbf{h})=$ $\frac{1}{2}\big{\|}X\mathbf{d}-\mathbf{s}\big{\|}_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g})+$
$\frac{\sigma}{2}\left\|\mathbf{d}-\mathbf{g}+\mathbf{h}\right\|_{2}^{2}\ ,$ (S8)

where we omit the final term, $-\frac{\sigma}{2}\left\|\mathbf{h}\right\|_{2}^{2}$, which does not effect the minimizer of this functional. We assume that the variables in the above equation represent the $K=1$ case, and construct the $K=2$ case by replicating the training data, i.e.

\[ X^{\prime}=\left(\begin{array}[]{c}X\\
X\end{array}\right)\ ,\ \ \ \mathbf{s}^{\prime}=\left(\begin{array}[]{c}\mathbf{s}\\
\mathbf{s}\end{array}\right)\ ,\ \ \ \mathbf{d}^{\prime}=\mathbf{d}\ , \]

$\mathbf{g}^{\prime}=\mathbf{g}$, and $\mathbf{h}^{\prime}=\mathbf{h}$. The corresponding augmented Lagrangian is

$L_{\sigma}(\mathbf{d}^{\prime},\mathbf{g}^{\prime},\mathbf{h}^{\prime})=$
$\frac{1}{2}\big{\|}X^{\prime}\mathbf{d}^{\prime}-\mathbf{s}^{\prime}\big{\|}_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g}^{\prime})+\frac{\sigma}{2}\left\|\mathbf{d}^{\prime}-\mathbf{g}^{\prime}+\mathbf{h}^{\prime}\right\|_{2}^{2}$
$=2\frac{1}{2}\left\|X\ \mathbf{d}-\mathbf{s}\right\|_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g})+\frac{\sigma}{2}\left\|\mathbf{d}-\mathbf{g}+\mathbf{h}\right\|_{2}^{2}$
$=2L_{2\sigma}(\mathbf{d},\mathbf{g},\mathbf{h})\ .$ (S9)

For this problem, the augmented Lagrangian for the $K=2$ case is twice the augmented Lagrangian for the $K=1$ case when the penalty parameter is also twice the penalty parameter used for the $K=1$ case. Therefore we expect that the optimal penalty parameter should scale linearly when changing the number of training images $K$.

### II-C Consensus ADMM Dictionary Update

The augmented Lagrangian for the ADMM Consensus form of the dictionary update problem Eq. (39) in the main document is

$L_{\sigma}(\mathbf{d},\mathbf{g},\mathbf{h})=$ $\frac{1}{2}\big{\|}X\mathbf{d}-\mathbf{s}\big{\|}_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g})+$
$\frac{\sigma}{2}\left\|\mathbf{d}-E\mathbf{g}+\mathbf{h}\right\|_{2}^{2}\ ,$ (S10)

where we omit the final term, $-\frac{\sigma}{2}\left\|\mathbf{h}\right\|_{2}^{2}$, which does not effect the minimizer of this functional, and

\[ E=\left(\begin{array}[]{c}I\\
I\\
\vdots\end{array}\right)\ . \] (S11)

We assume that the variables in the above equation represent the $K=1$ case, with $E=I$, and construct the $K=2$ case by replicating the training data, i.e.

\[ X^{\prime}=\left(\begin{array}[]{cc}X&0\\
0&X\end{array}\right)\ ,\ \mathbf{s}^{\prime}=\left(\begin{array}[]{c}\mathbf{s}\\
\mathbf{s}\end{array}\right)\ ,\ \mathbf{d}^{\prime}=\left(\begin{array}[]{c}\mathbf{d}\\
\mathbf{d}\end{array}\right)\ ,\ \mathbf{h}^{\prime}=\left(\begin{array}[]{c}\mathbf{h}\\
\mathbf{h}\end{array}\right)\ , \]

$\mathbf{g}^{\prime}=\mathbf{g}$, and $E^{\prime}=\left(\begin{array}[]{cccc}I&I\end{array}\right)^{T}$. The corresponding augmented Lagrangian is

$L_{\sigma}(\mathbf{d}^{\prime},\mathbf{g}^{\prime},\mathbf{h}^{\prime})=$
$\frac{1}{2}\big{\|}X^{\prime}\mathbf{d}^{\prime}-\mathbf{s}^{\prime}\big{\|}_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g}^{\prime})+\frac{\sigma}{2}\left\|\mathbf{d}^{\prime}-E^{\prime}\mathbf{g}^{\prime}+\mathbf{h}^{\prime}\right\|_{2}^{2}$
$=2\frac{1}{2}\left\|X\ \mathbf{d}-\mathbf{s}\right\|_{2}^{2}+\iota_{C_{\text{PS}}}(\mathbf{g})+2\frac{\sigma}{2}\left\|\mathbf{d}-E\mathbf{g}+\mathbf{h}\right\|_{2}^{2}$
$=2L_{\sigma}(\mathbf{d},\mathbf{g},\mathbf{h})\ .$ (S12)

For this problem, the augmented Lagrangian for the $K=2$ case is just twice the augmented Lagrangian for the $K=1$ case, with the same penalty parameter $\sigma$. Therefore we expect that the optimal penalty parameter should remain constant when changing the number of training images $K$.

### II-D FISTA Dictionary Update

The FISTA solution to the dictionary update problem requires computing the gradient of the data fidelity term in the DFT domain (Eq. (57) in the main document)

$\nabla_{\hat{\mathbf{d}}}\Big{(}\frac{1}{2}\big{\|}\hat{X}\hat{\mathbf{d}}-\hat{\mathbf{s}}\big{\|}_{2}^{2}\Big{)}=\hat{X}^{H}\big{(}\hat{X}\hat{\mathbf{d}}-\hat{\mathbf{s}}\big{)}\ .$ (S13)

We assume that the variables in the above equation represent the $K=1$ case, and construct the $K=2$ case by replicating the training data, i.e.

\[ \hat{X}^{\prime}=\left(\begin{array}[]{c}\hat{X}\\
\hat{X}\end{array}\right)\ ,\ \ \hat{\mathbf{s}}^{\prime}=\left(\begin{array}[]{c}\hat{\mathbf{s}}\\
\hat{\mathbf{s}}\end{array}\right)\ ,\ \ \hat{\mathbf{d}}^{\prime}=\hat{\mathbf{d}}\ , \]

and the gradient in the DFT domain is

$\nabla_{\hat{\mathbf{d}}^{\prime}}\Big{(}\frac{1}{2}\big{\|}\hat{X}^{\prime}\hat{\mathbf{d}}^{\prime}-\hat{\mathbf{s}}^{\prime}\big{\|}_{2}^{2}\Big{)}$ $=\hat{X}^{\prime H}\big{(}\hat{X}^{\prime}\hat{\mathbf{d}}^{\prime}-\hat{\mathbf{s}}^{\prime}\big{)}$
$=2\hat{X}^{H}\big{(}\hat{X}\hat{\mathbf{d}}-\hat{\mathbf{s}}\big{)}\ .$ (S14)

For this problem, the gradient in the DFT domain for the $K=2$ case is just twice the gradient in the DFT domain for the $K=1$ case. To obtain the same solution we need the

gradient step to be the same, which requires that the gradient step parameter be reduced by a factor of two to compensate for the doubling of the gradient. Therefore we expect that the optimal parameter $L$, which is the inverse of the gradient step size, should scale linearly when changing the number of training images $K$.

### II-E Mask Decoupling ADMM Sparse Coding

The augmented Lagrangian for the ADMM solution to the masked form of the MMV CBPDN problem Eq. (60) in the main document is

$L_{\rho}(X,Y_{0},Y_{1},U_{0},U_{1})=\frac{1}{2}\left\|WY_{1}\right\|_{F}^{2}+\lambda\left\|Y_{0}\right\|_{1}+$
\[ \frac{\rho}{2}\left\|\begin{pmatrix}Y_{0}\\
Y_{1}\end{pmatrix}-\left[\begin{pmatrix}I\\
D\end{pmatrix}X-\begin{pmatrix}0\\
S\end{pmatrix}\right]+\begin{pmatrix}U_{0}\\
U_{1}\end{pmatrix}\right\|_{F}^{2}\ , \] (S15)

where we omit the final term

$-\frac{\rho}{2}\left\|\begin{pmatrix}U_{0}\\
U_{1}\end{pmatrix}\right\|_{F}^{2}\ ,$

which does not effect the minimizer of this functional. We assume that the variables in the above equation represent the $K=1$ case, and construct the $K=2$ case by replicating the training data, i.e. $S^{\prime}=\left(\begin{array}[]{cc}\mathbf{s}&\mathbf{s}\end{array}\right)$, $X^{\prime}=\left(\begin{array}[]{cc}\mathbf{x}&\mathbf{x}\end{array}\right)$, $Y_{0}^{\prime}=\left(\begin{array}[]{cc}Y_{0}&Y_{0}\end{array}\right)$, $Y_{1}^{\prime}=\left(\begin{array}[]{cc}Y_{1}&Y_{1}\end{array}\right)$, $U_{0}^{\prime}=\left(\begin{array}[]{cc}U_{0}&U_{0}\end{array}\right)$, $U_{1}^{\prime}=\left(\begin{array}[]{cc}U_{1}&U_{1}\end{array}\right)$, and $0^{\prime}=\left(\begin{array}[]{cc}\mathbf{0}&\mathbf{0}\end{array}\right)$. The corresponding augmented Lagrangian is

$L_{\rho}(X^{\prime},Y_{0}^{\prime},Y_{1}^{\prime},U_{0}^{\prime},U_{1}^{\prime})=\frac{1}{2}\left\|WY_{1}^{\prime}\right\|_{F}^{2}+\lambda\left\|Y_{0}^{\prime}\right\|_{1}+$
\[ \frac{\rho}{2}\left\|\begin{pmatrix}Y_{0}^{\prime}\\
Y_{1}^{\prime}\end{pmatrix}-\left[\begin{pmatrix}I\\
D\end{pmatrix}X^{\prime}-\begin{pmatrix}0^{\prime}\\
S^{\prime}\end{pmatrix}\right]+\begin{pmatrix}U_{0}^{\prime}\\
U_{1}^{\prime}\end{pmatrix}\right\|_{F}^{2}\\
=2\frac{1}{2}\left\|WY_{1}\right\|_{2}^{2}+2\lambda\left\|Y_{0}\right\|_{1}+ \]
\[ 2\frac{\rho}{2}\left\|\begin{pmatrix}Y_{0}\\
Y_{1}\end{pmatrix}-\left[\begin{pmatrix}I\\
D\end{pmatrix}X-\begin{pmatrix}\mathbf{0}\\
\mathbf{s}\end{pmatrix}\right]+\begin{pmatrix}U_{0}\\
U_{1}\end{pmatrix}\right\|_{2}^{2}\\
=2L_{\rho}(X,Y_{0},Y_{1},U_{0},U_{1})\ . \]

For this problem, the augmented Lagrangian for the $K=2$ case is just twice the augmented Lagrangian for the $K=1$ case, with the same penalty parameter $\rho$. Therefore we expect that the optimal penalty parameter should remain constant when changing the number of training images $K$.

### II-F Mask Decoupling ADMM Dictionary Update

The augmented Lagrangian for the Block-Constraint ADMM solution of the masked dictionary update problem Eq. (69) in the main document is

$L_{\sigma}(\mathbf{d},\mathbf{g}_{0},\mathbf{g}_{1},\mathbf{h}_{0},\mathbf{h}_{1})=\frac{1}{2}\left\|W\mathbf{g}_{1}\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{g}_{0})+$
\[ \frac{\sigma}{2}\left\|\begin{pmatrix}\mathbf{g}_{0}\\
\mathbf{g}_{1}\end{pmatrix}-\left[\begin{pmatrix}I\\
X\end{pmatrix}\mathbf{d}-\begin{pmatrix}\mathbf{0}\\
\mathbf{s}\end{pmatrix}\right]+\begin{pmatrix}\mathbf{h}_{0}\\
\mathbf{h}_{1}\end{pmatrix}\right\|_{2}^{2}\ , \] (S16)

where we omit the final term

$-\frac{\sigma}{2}\left\|\begin{pmatrix}\mathbf{h}_{0}\\
\mathbf{h}_{1}\end{pmatrix}\right\|_{2}^{2}\ ,$

which does not effect the minimizer of this functional. We assume that the variables in the above equation represent the $K=1$ case, and construct the $K=2$ case by replicating the training data, i.e.

\[ X^{\prime}=\begin{pmatrix}X\\
X\end{pmatrix}\ ,\ \mathbf{s}^{\prime}=\begin{pmatrix}\mathbf{s}\\
\mathbf{s}\end{pmatrix}\ ,\ \mathbf{g}_{1}^{\prime}=\begin{pmatrix}\mathbf{g}_{1}\\
\mathbf{g}_{1}\end{pmatrix}\ ,\ \mathbf{h}_{1}^{\prime}=\begin{pmatrix}\mathbf{h}_{1}\\
\mathbf{h}_{1}\end{pmatrix}\ , \]

$\mathbf{d}^{\prime}=\mathbf{d}$, $\mathbf{g}_{0}^{\prime}=\mathbf{g}_{0}$, and $\mathbf{h}_{0}^{\prime}=\mathbf{h}_{0}$. The corresponding augmented Lagrangian is

$L_{\sigma}(\mathbf{d}^{\prime},\mathbf{g}_{0}^{\prime},\mathbf{g}_{1}^{\prime},\mathbf{h}_{0}^{\prime},\mathbf{h}_{1}^{\prime})=\frac{1}{2}\left\|W\mathbf{g}_{1}^{\prime}\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{g}_{0}^{\prime})+$
\[ \frac{\sigma}{2}\left\|\begin{pmatrix}\mathbf{g}_{0}^{\prime}\\
\mathbf{g}_{1}^{\prime}\end{pmatrix}-\left[\begin{pmatrix}I\\
X^{\prime}\end{pmatrix}\mathbf{d}^{\prime}-\begin{pmatrix}\mathbf{0}\\
\mathbf{s}^{\prime}\end{pmatrix}\right]+\begin{pmatrix}\mathbf{h}_{0}^{\prime}\\
\mathbf{h}_{1}^{\prime}\end{pmatrix}\right\|_{2}^{2}\\
=2\frac{1}{2}\left\|W\mathbf{g}_{1}\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{g}_{0})+\frac{\sigma}{2}\left\|\mathbf{g}_{0}-\mathbf{d}+\mathbf{h}_{0}\right\|_{2}^{2}\ +\\
2\frac{\sigma}{2}\left\|\mathbf{g}_{1}-(X\mathbf{d}-\mathbf{s})+\mathbf{h}_{1}\right\|_{2}^{2}\ . \] (S17)

For this problem, the augmented Lagrangian for the $K=2$ case has terms that are twice the augmented Lagrangian for the $K=1$ case, as well as a term that is the same as for the $K=1$ case. Therefore, there is no simple rule to scale the optimal penalty parameter $\sigma$ when changing the number of training images $K$.

It is, however, worth noting that a scaling relationship could be obtained by replacing the constraint $\mathbf{g}_{0}^{\prime}=\mathbf{d}^{\prime}$ with the equivalent constraint $2\mathbf{g}_{0}=2\mathbf{d}$ (or, more generally, $K\mathbf{g}_{0}=K\mathbf{d}$) and appropriate rescaling of the scaled dual variable $\mathbf{h}_{0}$, so that the problematic term above, $(\sigma/2)\left\|\mathbf{g}_{0}^{\prime}-\mathbf{d}^{\prime}+\mathbf{h}_{0}^{\prime}\right\|_{2}^{2}$, exhibits the same scaling as the other terms.

### II-G Hybrid Consensus Masked Dictionary Update

The augmented Lagrangian for the ADMM consensus solution of the masked dictionary update problem Eq. (71) in the main document is

$L_{\sigma}(\mathbf{d},\mathbf{g}_{0},\mathbf{g}_{1},\mathbf{h}_{0},\mathbf{h}_{1})=\frac{1}{2}\left\|W\mathbf{g}_{1}\right\|_{2}^{2}+\iota_{C_{\text{PN}}}(\mathbf{g}_{0})+$
\[ \frac{\sigma}{2}\left\|\begin{pmatrix}I\\
X\end{pmatrix}\mathbf{d}-\begin{pmatrix}E&0\\
0&I\end{pmatrix}\begin{pmatrix}\mathbf{g}_{0}\\
\mathbf{g}_{1}\end{pmatrix}-\begin{pmatrix}\mathbf{0}\\
\mathbf{s}\end{pmatrix}+\begin{pmatrix}\mathbf{h}_{0}\\
\mathbf{h}_{1}\end{pmatrix}\right\|_{2}^{2}\ , \] (S18)

where we omit the final term

$-\frac{\sigma}{2}\left\|\begin{pmatrix}\mathbf{h}_{0}\\
\mathbf{h}_{1}\end{pmatrix}\right\|_{2}^{2}\ ,$

which does not effect the minimizer of this functional. We assume that the variables in the above equation represent the $K=1$ case, with $E=I$, and construct the $K=2$ case by replicating the training data, i.e.

\[ X^{\prime}=\begin{pmatrix}X&0\\
0&X\end{pmatrix}\ ,\ \ \mathbf{s}^{\prime}=\begin{pmatrix}\mathbf{s}\\
\mathbf{s}\end{pmatrix}\ ,\ \ \mathbf{d}^{\prime}=\begin{pmatrix}\mathbf{d}\\
\mathbf{d}\end{pmatrix}\ , \]
\[ \mathbf{g}_{1}^{\prime}=\begin{pmatrix}\mathbf{g}_{1}\\
\mathbf{g}_{1}\end{pmatrix}\ ,\ \ \mathbf{h}_{0}^{\prime}=\begin{pmatrix}\mathbf{h}_{0}\\
\mathbf{h}_{0}\end{pmatrix}\ ,\ \ \mathbf{h}_{1}^{\prime}=\begin{pmatrix}\mathbf{h}_{1}\\
\mathbf{h}_{1}\end{pmatrix}\ , \]

$\mathbf{g}_0^{\prime} = \mathbf{g}_0$  , and  $E^{\prime} = \left( \begin{array}{ll}I &amp; I \end{array} \right)^{T}$  . The corresponding augmented Lagrangian is

$$
\begin{array}{l} L _ {\sigma} \left(\mathbf {d} ^ {\prime}, \mathbf {g} _ {0} ^ {\prime}, \mathbf {g} _ {1} ^ {\prime}, \mathbf {h} _ {0} ^ {\prime}, \mathbf {h} _ {1} ^ {\prime}\right) = \frac {1}{2} \left\| W \mathbf {g} _ {1} ^ {\prime} \right\| _ {2} ^ {2} + \iota_ {C _ {P n}} \left(\mathbf {g} _ {0} ^ {\prime}\right) + \\ \frac {\sigma}{2} \left\| \left( \begin{array}{c} I \\ X ^ {\prime} \end{array} \right) \mathbf {d} ^ {\prime} - \left( \begin{array}{c c} E ^ {\prime} &amp; 0 \\ 0 &amp; I \end{array} \right) \left( \begin{array}{c} \mathbf {g} _ {0} ^ {\prime} \\ \mathbf {g} _ {1} ^ {\prime} \end{array} \right) - \left( \begin{array}{c} 0 \\ \mathbf {s} ^ {\prime} \end{array} \right) + \left( \begin{array}{c} \mathbf {h} _ {0} ^ {\prime} \\ \mathbf {h} _ {1} ^ {\prime} \end{array} \right) \right\| _ {2} ^ {2}, \\ = 2 \frac {1}{2} \| W \mathbf {g} _ {1} \| _ {2} ^ {2} + \iota_ {C _ {P n}} (\mathbf {g} _ {0}) + \\ \end{array}
$$

$$
\begin{array}{l} 2 \frac {\sigma}{2} \left\| \left( \begin{array}{c} I \\ X \end{array} \right) \mathbf {d} - \left( \begin{array}{c c} E &amp; 0 \\ 0 &amp; I \end{array} \right) \left( \begin{array}{c} \mathbf {g} _ {0} \\ \mathbf {g} _ {1} \end{array} \right) - \left( \begin{array}{c} \mathbf {0} \\ \mathbf {s} \end{array} \right) + \left( \begin{array}{c} \mathbf {h} _ {0} \\ \mathbf {h} _ {1} \end{array} \right) \right\| _ {2} ^ {2} \\ = 2 L _ {\sigma} (\mathbf {d}, \mathbf {g} _ {0}, \mathbf {g} _ {1}, \mathbf {h} _ {0}, \mathbf {h} _ {1}). \\ \end{array}
$$

For this problem, the augmented Lagrangian for the  $K = 2$  case is just twice the augmented Lagrangian for the  $K = 1$  case, with the same penalty parameter  $\sigma$ . Therefore we expect that the optimal penalty parameter should remain constant when changing the number of training images  $K$ .

# SIV. EXPERIMENTAL SENSITIVITY ANALYSIS

Experiments to determine the median stability of the optimal parameters across an ensemble of training sets of the same size are discussed in Sec. VII-G2 in the main document. The corresponding results are plotted here in Figs. S4 - S15. The box plots represent median, quartiles, and the full range of variation of the normalized functional values obtained at each parameter value for the 20 different image subsets at each of the sizes  $K \in \{5, 10, 20\}$ . The red lines connect the medians of the distributions at each parameter value.

It can be seen in Figs. 10(b), 11(b), and 12(b) that FISTA has very skewed sensitivity plots for  $L$ , the inverse of the gradient step size. This is related to the requirement, mentioned in the main document, that  $L$  has to be greater than or equal to the Lipschitz constant of the gradient of the functional to guarantee convergence of the algorithm. Although this constant is not always computable [33], in these experiments we are able to estimate the threshold that indicates the change in behavior expected when  $L$  becomes greater than the Lipschitz constant. The variation of the normalized functional values is comparable to those for other methods and other parameters for values of  $L$  greater than this threshold. However, for values of  $L$  smaller than the threshold, the instability causes a much larger variance in the normalized functional values. We decided to clip the large vertical ranges resulting from the very large variances to the left of these plots in order to more clearly display the scaling in the useful range of  $L$ . As a result, some of the interquartile range boxes to the left are incomplete, or just the lower part of the full range of variation is visible.

# SV. LARGE TRAINING SET EXPERIMENTS

In order to evaluate the performance of the methods for larger training sets and images of different sizes, we performed additional experiments, including comparisons with the original implementations of competing algorithms. We used training sets of 25, 100 and 400 images of sizes 1024 × 1024 pixels, 512 × 512 pixels and 256 × 256 pixels,

![images/image31.jpg](images/image31.jpg)
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$

![images/image32.jpg](images/image32.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

![images/image33.jpg](images/image33.jpg)
Fig. S4. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the conjugate gradient (CG) grid search for 20 random selected sets of  $K = 5$  images.
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$

![images/image34.jpg](images/image34.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

![images/image35.jpg](images/image35.jpg)
Fig. S5. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the conjugate gradient (CG) grid search for 20 random selected sets of  $K = 10$  images.
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$
Fig. S6. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the conjugate gradient (CG) grid search for 20 random selected sets of  $K = 20$  images.

![images/image36.jpg](images/image36.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

respectively. These combinations of number,  $K$ , and size,  $N$ , of images were chosen to maintain a constant number of pixels in the training set, which provides a useful way of simultaneously exploring performance variations with respect to both  $N$  and  $K$ . All of these images were derived from images in the MIRFLICKR-1M dataset and pre-processed (scaling and highpass filtering) in the same way, as described in Sec. VII-C in the main document.

All the results using the methods discussed and analyzed in the main document were computed using the Python implementation of the SPORCO library [36], [37] on a Linux workstation equipped with two Xeon E5-2690V4 CPUs. We

![images/image37.jpg](images/image37.jpg)
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$

![images/image38.jpg](images/image38.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

![images/image39.jpg](images/image39.jpg)
Fig. S7. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the consensus (Cns / Cns-P) grid search for 20 random selected sets of  $K = 5$  images.
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$
Fig. S8. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the consensus (Cns / Cns-P) grid search for 20 random selected sets of  $K = 10$  images.

![images/image40.jpg](images/image40.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

![images/image41.jpg](images/image41.jpg)
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$
Fig. S9. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the consensus (Cns / Cns-P) grid search for 20 random selected sets of  $K = 20$  images.

![images/image42.jpg](images/image42.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

also include comparisons with the method proposed by Papyan et al. [27], using their publicly available Matlab and C implementation $^{16}$ .

We tried to include the publicly available Matlab implementations of the methods proposed by Šorel and Šroubek[17] [24] and by Heide et al.[18] [9] in these comparisons, but were unable

![images/image43.jpg](images/image43.jpg)
(a)  $\mathrm{CBPDN}(\rho)$  for best  $L$

![images/image44.jpg](images/image44.jpg)
(b)  $\mathrm{CBPDN}(L)$  for best  $\rho$

![images/image45.jpg](images/image45.jpg)
Fig. S10. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the FISTA grid search for 20 random selected sets of  $K = 5$  images.
(a)  $\mathrm{CBPDN}(\rho)$  for best  $L$

![images/image46.jpg](images/image46.jpg)
(b)  $\mathrm{CBPDN}(L)$  for best  $\rho$

![images/image47.jpg](images/image47.jpg)
Fig. S11. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the FISTA grid search for 20 random selected sets of  $K = 10$  images.
(a)  $\mathrm{CBPDN}(\rho)$  for best  $L$
Fig. S12. Distribution of normalized CBPDN functional (Eq. (5) in the main document) after 500 iterations, in the FISTA grid search for 20 random selected sets of  $K = 20$  images.

![images/image48.jpg](images/image48.jpg)
(b)  $\mathrm{CBPDN}(L)$  for best  $\rho$

to obtain acceptable results $^{19}$ . We therefore omit these methods from the comparisons here, including them only in a separate set of experiments on a smaller data set, reported in Sec. SVII below.

In all of these experiments we learned a dictionary of 100 filters of size  $11 \times 11$ , setting the sparsity parameter  $\lambda = 0.1$ . We set the parameters for our methods according to the scaling rules discussed in Sec. VII-G2 in the main document, using fixed penalty parameters  $\rho$  and  $\sigma$  without any adaptation methods. In contrast to the experiments reported in the main

19The methods were very slow, with partial results after running for 4 days still being noisy and far from convergence.

![images/image49.jpg](images/image49.jpg)
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$

![images/image50.jpg](images/image50.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

![images/image51.jpg](images/image51.jpg)
Fig. S13. Distribution of normalized masked CBPDN functional (Eq. (59) in the main document) after 500 iterations, in the masked consensus (M-Cns / M-Cns-P) grid search for 20 random selected sets of  $K = 5$  images.
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$

![images/image52.jpg](images/image52.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

![images/image53.jpg](images/image53.jpg)
Fig. S14. Distribution of normalized masked CBPDN functional (Eq. (59) in the main document) after 500 iterations, in the masked consensus (M-Cns / M-Cns-P) grid search for 20 random selected sets of  $K = 10$  images.
(a)  $\mathrm{CBPDN}(\rho)$  for best  $\sigma$
Fig. S15. Distribution of normalized masked CBPDN functional (Eq. (59) in the main document) after 500 iterations, in the masked consensus (M-Cns / M-Cns-P) grid search for 20 random selected sets of  $K = 20$  images.

![images/image54.jpg](images/image54.jpg)
(b)  $\mathrm{CBPDN}(\sigma)$  for best  $\rho$

document, relaxation methods [17, Sec. 3.4.3][5, Sec. III.D] were used, with  $\alpha = 1.8$ .

We used the default parameters from the demonstration scripts distributed with each of the publicly available Matlab implementations by the authors of [27], [9], and [24]. Our efforts to adjust the default parameters for the implementations of the methods of [9], and [24] to obtain better results were unsuccessful, at least in part due to the slow convergence of the methods and the absence of any parameter selection discussion or guidelines provided by the authors.

During training, the dictionaries were saved at 25 iteration intervals to allow evaluation on an independent test set, which

consisted of the same additional set of 20 images, of size  $256 \times 256$  pixels, that was used for this purpose for the experiments reported in the main document. This evaluation was performed by sparse coding of the images in the test set, for  $\lambda = 0.1$ , computing the evolution of the CBPDN functional over the series of dictionaries. This not only allows comparison of generalization performance, taking into account possible differences in overfitting effects between the different methods, but also allows for a fair comparison between the methods, avoiding the difficulty of comparing the training functional values that are computed differently by different implementations[20].

# A. CDL without Spatial Mask

![images/image55.jpg](images/image55.jpg)
Fig. S16. Dictionary Learning ( $K = 25$ ): A comparison on a set of  $K = 25$  images,  $1024 \times 1024$  pixels, of the decay of the value of the CPBDN functional Eq. (5) with respect to run time and iterations.

![images/image56.jpg](images/image56.jpg)
Fig. S17. Dictionary Learning ( $K = 100$ ): A comparison on a set of  $K = 100$  images,  $512 \times 512$  pixels, of the decay of the value of the CBPDN functional Eq. (5) with respect to run time and iterations.

Results for the training objective function are shown in Fig. S16 for  $K = 25$  with  $1024 \times 1024$  images, in Fig. S17 for  $K = 100$  with  $512 \times 512$  images, and in Fig. S18 for  $K = 400$  with  $256 \times 256$  images. It is clear that Cns-P consistently achieves the best performance, converging smoothly to a slightly smaller functional value than the other two methods in all the cases except for Fig. S16. It also

20All of our implementations calculate the functional values in the same way, but the implementations by other authors adopt slightly different approaches.

![images/image57.jpg](images/image57.jpg)
Fig. S18. Dictionary Learning ( $K = 400$ ): A comparison on a set of  $K = 400$  images,  $256 \times 256$  pixels, of the decay of the value of the CBPDN functional Eq. (5) with respect to run time and iterations.

exhibits the fastest convergence of the methods compared. In contrast, FISTA results are less stable, presenting some wild oscillations at the beginning and some small oscillations at the end, but nevertheless achieving similar final functional values to Cns-P. The method of Papyan et al. [27] has very rapid convergence in terms of iterations, but its time performance is the worst of the three methods.

The FISTA instability can be automatically corrected by using the backtracking step-size adaptation rule (see Sec. III-D in main document). However, due to the uni-directional correction of the backtracking rule that always increases  $L$  (i.e. it always decreases the gradient step size), the evolution of the functional is smooth, but also tends to converge to a larger functional value. A reasonable approach for methods that do not converge monotonically, such as FISTA, is to consider the solution at each time step as the best solution obtained until that step, as opposed to the solution specifically for that step, which has the effect of smoothing the functional value evolution. In all our experiments, we used a fixed  $L$  value, set in accordance with the parameter rules described in the main document, and report actual convergence without any post processing since this more accurately illustrates the real FISTA behavior and the tradeoff between convergence smoothness and final functional value determined by parameter  $L$ .

![images/image58.jpg](images/image58.jpg)
Fig. S19. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 25$  images,  $1024 \times 1024$  pixels, as in Fig. S16.

![images/image59.jpg](images/image59.jpg)
Fig. S20. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 100$  images,  $512 \times 512$  pixels, as in Fig. S17.

![images/image60.jpg](images/image60.jpg)
Fig. S21. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 400$  images,  $256 \times 256$  pixels, as in Fig. S18.

Testing results obtained for the additional 20 images of size  $256 \times 256$  are displayed in Fig. S19, for  $K = 25, 1024 \times 1024$  images, in Fig. S20 for  $K = 100, 512 \times 512$  images and in Fig. S21 for  $K = 400, 256 \times 256$  images. Note again that, as in the comparisons in the main document, the time axis in these plots refers to the run time of the dictionary learning code used to generate the relevant dictionary, and not to the run time of the sparse coding on the test set.

All the testing plots show that the methods perform as expected from the training comparison, with Cns-P achieving better performance also in the test set, followed by FISTA. Results for the method of Papyan et al. are always worse, and do not match the functional values achieved either by Cns-P or FISTA. For all methods, testing results are better for the dictionary filters obtained when training with  $K = 400,256 \times 256$  images (Fig. S21), followed by the dictionary filters obtained when training with  $K = 100,512 \times 512$  images (Fig. S20), with the worst results obtained for the dictionary filters obtained when training with  $K = 25,1024 \times 1024$  images (Fig. S19). In particular, the Cns-P functional increases near the end of the evolution in Fig. S19. We believe that this is due to overfitting effects for the  $K = 100$  and  $K = 25$  cases, resulting from the mismatch between training and validation image sizes. Additional experiments (results not shown) confirmed that the functional decreases monotonically

when the size of the images in the testing set corresponds to the size of the images in training set. Nevertheless, we decided to use the same testing set for all of these experiments so that the corresponding functionals would be comparable across the different training sets.

It can be seen from Fig. 22(a) that the time per iteration for both Cns-P and FISTA decreases very slowly with increasing  $K$  and decreasing  $N$ , i.e. it is roughly linear in  $NK$ , the number of pixels in the training image set. Since the results in Fig. 4 show that these algorithms scale linearly with  $K$ , this implies that the algorithms have approximately linear scaling with  $N$  as well. The slight deviation from linearity can be attributed to the  $N\log N$  complexity of the FFTs used in these algorithms (see the computational complexity analysis in Table I in the main document). The method of Papyan et al. seems to be more sensitive to the scaling in  $K$ , with time per iteration increasing as  $K$  increases (which is not evident from the complexity analysis, see Table I below), and requires more time per iteration than Cns-P or FISTA.

![images/image61.jpg](images/image61.jpg)
(a) Without Spatial Mask

![images/image62.jpg](images/image62.jpg)
(b) With Spatial Mask

# B. CDL with Spatial Mask

![images/image63.jpg](images/image63.jpg)
Fig. S23. Dictionary Learning with Spatial Mask ( $K = 25$ ): A comparison on a set of  $K = 25$  images,  $1024 \times 1024$  pixels, of the decay of the value of the masked CBPDN functional Eq. (59) with respect to run time and iterations for masked versions of the algorithms.

Comparisons for CDL with a spatial mask were performed with a random mask with values in  $\{0,1\}$ , with  $25\%$  zero

![images/image64.jpg](images/image64.jpg)
Fig. S24. Dictionary Learning with Spatial Mask ( $K = 100$ ): A comparison on a set of  $K = 100$  images,  $512 \times 512$  pixels, of the decay of the value of the masked CBPDN functional Eq. (59) with respect to run time and iterations for masked versions of the algorithms.

![images/image65.jpg](images/image65.jpg)
Fig. S25. Dictionary Learning with Spatial Mask ( $K = 400$ ): A comparison on a set of  $K = 400$  images,  $256 \times 256$  pixels, of the decay of the value of the masked CBPDN functional Eq. (59) with respect to run time and iterations for masked versions of the algorithms.

entries with a uniform random distribution. Three different random masks were generated, one for the set of images of  $1024 \times 1024$  pixels, one for the set of  $512 \times 512$  pixels, and one for the set of  $256 \times 256$  pixels. All the methods used the same randomly generated masks. The corresponding results are shown in Fig. S23 for  $K = 25$ ,  $1024 \times 1024$  images, in Fig. S24 for  $K = 100$ ,  $512 \times 512$  images and in Fig. S25 for  $K = 400$ ,  $256 \times 256$  images. These resemble the results obtained for the unmasked variants, with M-Cns-P yielding the fastest convergence and smallest final masked CBPDN functional values, followed by M-FISTA. M-FISTA is still initially unstable in some cases, but its convergence becomes much smoother than the unmasked variant by the end of the learning. Since both M-Cns-P and M-FISTA converge to a similar functional value in learning, it is difficult to see the differences in computation time in the plots, but M-Cns-P is almost  $2/3$  faster than M-FISTA. The functional values for the masked method of Papyan et al. [27] are inaccurate since the mask is not taken into account in the calculation.

A fair comparison can, however, be made by evaluating the CBPDN functional, Eq. (5), when sparse coding the test set with the dictionary filters learned in training. The results are shown in Fig. S26, for  $K = 25$ ,  $1024 \times 1024$  images, in Fig. S27 for  $K = 100$ ,  $512 \times 512$  images and in Fig. S28 for

![images/image66.jpg](images/image66.jpg)
Fig. S26. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 25$  images,  $1024 \times 1024$  pixels, for masked versions of the algorithms, as in Fig. S23.

![images/image67.jpg](images/image67.jpg)
Fig. S27. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 100$  images,  $512 \times 512$  pixels, for masked versions of the algorithms, as in Fig. S24.

![images/image68.jpg](images/image68.jpg)
Fig. S28. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 400$  images,  $256 \times 256$  pixels, for masked versions of the algorithms, as in Fig. S25.

$K = 400,256\times 256$  images. Again, note that testing results for the case of  $K = 400,256\times 256$  are better for all the methods, and that for our methods there are some overfitting effects for the  $K = 100$  and  $K = 25$  cases, although these are less significant than those for the unmasked ones. Also, it is clear that testing results for M-Cns-P and M-FISTA are much better than for the masked method of Papyan et al. [27].

It can be seen from Fig. 22(b) that M-Cns-P and M-FISTA exhibit similar behavior to the corresponding unmasked variants in that the time per iteration is almost constant when

the product of  $N$  and  $K$  remains unchanged. The difference in the time per iteration between unmasked and masked variants is larger for M-FISTA than for M-Cns-P. Conversely, the time per iteration between unmasked and masked variants decreases for the method of Papyan et al., for smaller  $K$  and larger  $N$ , while it increases slightly for larger  $K$  and smaller  $N$ . This behavior is not expected from the complexity analysis.

Finally, it is worth noting that, while we do not quantify the optimality of the parameters selected via the guidelines discussed in Sec. VII-G of the main document, they do appear to provide good performance even for the substantially larger problems, considered here, than those used to develop these guidelines. In contrast, we found parameter selection to be problematic for the methods proposed by other authors discussed in Sec. SVII.

# SVI. SCALING WITH DICTIONARY SIZE

![images/image69.jpg](images/image69.jpg)
Fig. S29. Comparison of time per iteration for sets of  $M \in \{50, 100, 200, 500\}$ , with  $11 \times 11$  dictionary filters and  $K = 40$  images of size  $256 \times 256$  pixels.

In this section we compare the scaling with respect to the number of filters,  $M$ , of our two leading methods (Cns-P and FISTA) and the method of Papyan et al. [27]. Dictionaries with  $M \in \{50, 100, 200, 500\}$  filters of size  $11 \times 11$  were learned, over 500 iterations, from the training set of  $K = 40$ ,  $256 \times 256$  greyscale images described in the main document. The time per iteration for the three methods is compared in Fig. S29, which shows that all three methods exhibit linear scaling (modulo the outlier at  $M = 100$  for the method of Papyan et al.) with the number of filters.

These experiments do not address the issue of filter size. While the performance of the DFT-domain methods proposed here is roughly independent of the filter size, spatial domain methods such as that of Papyan et al. become more expensive as the filter size increases. In addition, multi-scale dictionaries are easily supported by the DFT-domain methods, but are much more difficult to support for spatial domain methods.

# SVII. ADDITIONAL ALGORITHM COMPARISONS

We used the same training set as the previous section ( $K = 40$ ,  $256 \times 256$  greyscale images) to compare the performance between our two leading methods (Cns-P and FISTA) and the competing methods proposed by Heide et al. [9] and by Papyan et al. [27], and the consensus method proposed by

Šorel and Šroubek [24]. Our methods are implemented in Python, those of Heide et al. [9], and of Šorel and Šroubek [24] are implemented in Matlab, and that of Papyan et al. [27] is implemented in Matlab and C.

We compared the performance of the methods in learning a dictionary of 100 filters of size  $11 \times 11$ , setting the sparsity parameter  $\lambda = 0.1$ . We set the parameters for our methods according to the scaling rules discussed in Sec. VII-G2 in the main document, using fixed penalty parameters  $\rho$  and  $\sigma$  without any adaptation methods. Relaxation methods [17, Sec. 3.4.3][5, Sec. III.D] were used, with  $\alpha = 1.8$ . The parameters for the competing methods were set from the default parameters included in their respective demonstration scripts. As before, the additional set of 20 images of size 256  $\times$  256 pixels was used as a test set to evaluate the dictionaries learned. Again, we report the evolution of the CBPDN functional Eq. (5) for the test set to provide a meaningful comparison, independent of the training functional evaluation implemented by each method, which use slightly different expressions, sometimes calculated with un-normalized dictionaries.

![images/image70.jpg](images/image70.jpg)
Fig. S30. Dictionary Learning ( $K = 40$ ): A comparison on a set of  $K = 40$  images,  $256 \times 256$  pixels, of the decay of the functional value in training with respect to run time and iterations for Cns-P, FISTA, the method of Papyan et al., and the consensus method of Šorel and Šroubek.

![images/image71.jpg](images/image71.jpg)
Fig. S31. Dictionary Learning ( $K = 40$ ): A comparison on a set of  $K = 40$  images,  $256 \times 256$  pixels, of the decay of the functional value in training with respect to run time and iterations for Cns-P, FISTA, the method of Papyan et al., and the method of Heide et al..
Fig. S32. Dictionaries obtained for training with  $K = 40$  images,  $256 \times 256$  pixels. These are the direct outputs: Cns-P, FISTA and the implementation of the method of Papyan et al. produce dictionaries normalized to 1; the implementation of the consensus method of Šorel and Šroubek produces dictionaries with most norms greater than 1; and the implementation of the method of Heide et al. produces dictionaries with most norms smaller than 1.

Comparisons for training are shown in Figs. S30 and S31. Performance is comparable for Cns-P, FISTA and the method

![images/image72.jpg](images/image72.jpg)
(a) Cns-P

![images/image73.jpg](images/image73.jpg)
(b) FISTA

![images/image74.jpg](images/image74.jpg)
(c) Papyan et al. [27]

![images/image75.jpg](images/image75.jpg)
(d) Heide et al. [9]

![images/image76.jpg](images/image76.jpg)
(e) Sorel and Sroubek [24]

of Papyan et al., with FISTA initially exhibiting oscillatory behavior. Since the methods of Šorel and Šroubek, and of Heide et al. perform multiple inner iterations $^{21}$  of the sparse coding and dictionary learning subproblems for each outer iteration, the iteration counts for these methods are reported as the product of inner and outer iterations. The method of Heide et al. starts with a very large functional value and is slow to converge $^{22}$ . The consensus method of Šorel and Šroubek

21Set to 10 and 5 inner iterations in the demonstration scripts provided by Heide et al., and Sorel and Sroubek respectively.
22We were unable to coerce this code to run for a full 500 iterations (50 outer iterations with 10 inner iterations) by any adjustment of stopping conditions and tolerances.

appears to achieve significantly lower functional values than the other methods, but these results are not comparable since their dictionary filters are not properly normalized. The final dictionaries computed are displayed in Fig. S32.

![images/image77.jpg](images/image77.jpg)
Fig. S33. Evolution of the CBPDN functional Eq. (5) for the test set using the partial dictionaries obtained when training for  $K = 40$  images,  $256 \times 256$  pixels.

Sparse coding results on the test set are shown in Fig. S33. Note that Cns-P and FISTA produce the smallest CBPDN functional values, followed by the method of Papyan et al., while results for the methods of Sorel and Sroubek as well as Heide et al. are much worse. Since the functional value evolution for the method of Heide et al. is highly oscillatory, at each iteration we plot the best functional value obtained up until that point instead of the functional value for that iteration. In terms of time evolution, it is clear that Cns-P is the fastest to converge, followed by FISTA and the method of Papyan et al.. The methods of Sorel and Sroubek and of Heide et al. are slow even for this relatively small dataset.

TABLEI COMPUTATIONAL COMPLEXITIES PER ITERATION OF CDL ALGORITHMS. THE NUMBER OF Pixels IN THE TRAINING IMAGES, THE NUMBER OF DICTIONARY FILTERS, AND THE NUMBER OF TRAINING IMAGES ARE DENOTED BY  $N,M$  ,AND  $K$  RESPECTIVELY. ADDITIONALLY,  $n$  REPRESENTS THE LOCAL FILTER SUPPORT,  $\alpha$  THE MAXIMUM NUMBER OF NON-ZEROS IN A NEEDLE [27] AND  $P$  THE NUMBER OF INTERNAL ADMM ITERATIONS.

|  Algorithm | Complexity  |
| --- | --- |
|  Cns-P, FISTA | O(KMN log N + KMN) + O(KMN log N + KMN + MN)  |
|  Papyan et al. [27] | O(KMNn + KN(α3 + Mα2) + nM2) + O(KNnα + KNMα + nM2)  |
|  Sorel and Sroubek [24] | O(PKMN log N + PKMN) + O(PKMN log N + PKMN)  |
|  ADMM consensus |   |
|  Heide et al. [9] | O(MK2N + (P-1)MKN) + O(PKMN log N) + O(PKMN)  |
|  M > K |   |
|  Heide et al. [9] | O(M3N + (P-1)M2N) + O(PKMN log N) + O(PKMN)  |
|  M ≤ K |   |

The per-iteration computational complexities of the methods, including both sparse coding and convolutional dictionary learning subproblems, are summarized in Table I. The complexity expressions for the methods of Papyan et al. [27] and Heide et al. [9] are reproduced from those provided in those works, and the expression provided by Sorel and Sroubek [24] is modified to make explicit the dependence on the number

of images  $K$  (for the sparse coding subproblem) and the internal ADMM iterations  $P$ . Our methods have mostly linear scaling in the problem size variables, with the exception of the image size,  $N$ , for which the scaling is  $N\log N$ , which is shared by all of the methods that compute convolutions in the frequency domain. The corresponding scaling of the spatial domain method of Papyan et al. is  $Nn$ , where  $n$  is the number of samples in each filter kernel, i.e. the additional  $\log N$  scaling with image size of the frequency domain methods is replaced with a linear scaling with filter size. This suggests that frequency domain methods are to be preferred for images of moderate size and moderate to large filter kernels, while spatial domain methods have an advantage for very large images and small filter kernels.

# SVIII. MULTI-CHANNEL EXPERIMENTS

In this section we report on an experiment intended to demonstrate the multi-channel CDL capability discussed in Sec. VI of the main document. We only provide results for the two leading approaches proposed here (Cns-P and FISTA), and do not compare with the algorithms of Heide et al. [9], Sorel and Sroubek [24], or Papyan et al. [27] since none of the corresponding publicly available implementations support multi-channel CDL. All of the color images used for these experiments were derived from images in the MIRFLICKR-1M dataset and pre-processed (cropping, scaling and highpass filtering per channel) in the same way (except for conversion to greyscale) as described in Sec. VII-C in the main document. The parameters of the Cns-P method were set using the parameter selection rules for the single channel problem, without any additional tuning. These rules were also used to set the parameters of the FISTA method, but the rule for  $L$  was multiplied by 3 for a more stable convergence.

![images/image78.jpg](images/image78.jpg)
Fig. S34. Dictionary Learning ( $K = 40$ ): A comparison on a set of  $K = 40$  color images,  $256 \times 256$  pixels, of the decay of the value of the multi-channel CPBDN functional Eq. (82) with respect to run time and iterations.

A dictionary of  $M = 64$  filters of size  $8 \times 8$  and  $C = 3$  channels was learned from a set of  $K = 40$  color images of size  $256 \times 256$ , using a sparsity parameter setting  $\lambda = 0.1$ . The results for this experiment are reported in Fig. S34. Comparing with single-channel dictionary learning results for a dictionary of the same size, and a training image set of the same number of images of the same size, reported in Fig. 3 in the main document, it can be seen that Cns-P requires about  $2/3$  of

![images/image79.jpg](images/image79.jpg)
Fig. S35. Evolution of the multi-channel CBPDN functional Eq. (82) for the test set using the partial dictionaries obtained when training for  $K = 40$  color images,  $256 \times 256$  pixels, as in Fig. S34.

time to compute the greyscale result compared to the color result, while FISTA requires about  $3/4$  of time to compute the greyscale result compared to the color result. This additional cost for learning a color dictionary from color images is quite moderate considering that three times more training data is used.

Similarly to the other experiments, we saved the dictionaries at regular intervals during training and used an additional set of 10 color images, of size  $256 \times 256$  pixels and from the same source, for testing. We compared the methods by sparse coding the color images in the test set, with  $\lambda = 0.1$ , and computing the evolution of the CBPDN functional over the series of multi-channel dictionaries. Fig. S35 shows that Cns-P performs slightly better than FISTA in testing too, although, Cns-P convergence is less smooth in the final stages compared to the single-channel cases, perhaps due to suboptimal parameter selection. Further evaluation of the multi-channel performance, including parameter selection guidelines, is left for future work.