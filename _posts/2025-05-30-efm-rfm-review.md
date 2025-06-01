---
layout: post
title: "AI810 Blog Post (20258026)"
date: 2025-05-30
math: true
permalink: /blog/review-project/
---

<!-- MathJax for LaTeX rendering -->
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<!-- Extra styling for spacing -->
<style>
  h2, h3 {
    margin-top: 2em;
    margin-bottom: 0.5em;
  }
  .math-display {
    margin: 1em 0;
    text-align: center;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
  }
  th, td {
    border: 1px solid #999;
    padding: 0.5em;
    text-align: center;
  }
</style>

# Geometry-Aware Generative Paths: From Group Orbits to Premetrics

**Review of**
**“[Equivariant Flow Matching](https://arxiv.org/abs/2306.15030)” (NeurIPS 2023)** , and 
**“[Flow Matching on General Geometries](https://arxiv.org/abs/2302.03660)” (ICLR 2024)**

> _One fixes crooked paths by aligning external symmetries; the other bends the space itself so every path becomes straight again._

---

## Table of Contents

1. [Preliminaries](#prel)
   - [1.1 Flow Matching](#fm)
   - [1.2 Optimal Transport (OT)](#ot)
2. [Paper I: Equivariant Flow Matching (EFM)](#efm)
   - [2.1 Limitation of Naive OT Flow Matching](#efm-naive_ot)
   - [2.2 EFM: Orbit-aware Cost and Equivariant Model](#efm-methods)
   - [2.3 Experimental Results](#efm-results)
3. [Paper II: Riemannian Flow Matching (RFM)](#rfm)
   - [3.1 Definitions on Riemannian Manifolds](#rfm-def)
   - [3.2 Vector Fields via Distance Functions](#rfm-vf)
   - [3.3 Approximating Geodesics with Spectral Distances](#rfm-spectral)
   - [3.4 Experimental Results](#rfm-results)
4. [Summary](#summary)
5. [Discussions](#discussion)

---

<a name="prel"></a>

## 1 · Preliminaries

<a name="fm"></a>

### 1.1 Flow Matching

Flow Matching (FM) is a powerful framework for constructing continuous normalizing flows (CNFs) by learning a vector field that transports a simple, known base distribution $$q(x_0)$$ (such as a Gaussian) into a complex target distribution $$\mu(x_1)$$. The key idea is to define a continuous path $$x_t$$, governed by an ordinary differential equation (ODE):

$$
\frac{dx_t}{dt} = v_\theta(t, x_t), \quad t \in [0,1],
$$

where $$v_\theta$$ is a parameterized vector field (typically a neural network). By solving this ODE from $$t=0$$ to $$t=1$$, we transform samples from the base $$q(x_0)$$ to approximate samples from the target distribution $$\mu(x_1)$$.

The training of Flow Matching simplifies to a regression problem. Given pairs $$(x_0, x_1)$$, we define a straightforward, "ideal" vector field $$u_t(x_t \mid x_1)$$, typically chosen as the direct path from $$x_0$$ to $$x_1$$. In Euclidean space, this vector field is often defined as:

$$
u_t(x_t \mid x_1) = \frac{x_1 - x_t}{1 - t},
$$

which represents the constant velocity required to arrive at $$x_1$$ by time $$t = 1$$.

Training then proceeds by minimizing the mean squared error (MSE) loss between the learned vector field $$v_\theta$$ and the ideal vector field $$u_t$$ via conditional flow matching (CFM):

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim U[0,1], x_0 \sim q, x_1 \sim \mu}\left[\| v_\theta(t,x_t) - u_t(x_t \mid x_1) \|^2\right].
$$

The simplicity and computational efficiency of Flow Matching make it a popular choice, especially when integration at inference time can be performed with very few numerical steps.

<a name="ot"></a>

### 1.2 Optimal Transport (OT)

Optimal Transport provides a principled way of measuring and computing the minimal cost of transporting mass from one probability distribution to another. Specifically, given two distributions $$q(x_0)$$ and $$\mu(x_1)$$, OT finds a coupling $$\pi(x_0, x_1)$$ that minimizes the expected transport cost:

$$
\pi^\star = \underset{\pi \in \Pi(q, \mu)}{\mathrm{argmin}}\,\mathbb{E}_{(x_0,x_1) \sim \pi}[c(x_0, x_1)],
$$

where $$\Pi(q, \mu)$$ is the set of all couplings whose marginals are $$q$$ and $$\mu$$, and $$c(x_0, x_1)$$ is the transport cost function, often chosen as the squared Euclidean distance $$\|x_0 - x_1\|^2$$.

In practice, OT frequently appears in Flow Matching as the canonical choice for defining the ideal vector field. The vector field constructed from OT paths provides a natural notion of "straightness" or minimal-energy path in the Euclidean setting. In this case, the same time-scaled vector field is used:

$$
u_t(x_t \mid x_1) = \frac{x_1 - x_t}{1 - t}.
$$

However, in the presence of symmetries or non-Euclidean geometries, naive OT coupling can produce suboptimal or "curved" paths. Addressing these limitations motivates approaches such as **Equivariant Flow Matching (EFM)**—which introduces symmetry-aware OT—and **Riemannian Flow Matching (RFM)**—which generalizes OT paths to curved, intrinsic geometries.

---

<a name="efm"></a>

## 2. Paper I: Equivariant Flow Matching (EFM)

This paper focuses on improving OT-based flow matching in **Euclidean space with symmetric structures**, such as **translations**, **rotations**, and **permutations**. These symmetries commonly arise in molecular and physical systems, where multiple configurations belong to the same equivalence class (orbit).

<a name="efm-naive_ot"></a>

### 2.1 Limitation of Naive OT Flow Matching

Using a naive OT flow matching loss on symmetric datasets often results in **highly curved paths**. This occurs because the OT solver may match source and target points that belong to **different symmetry orbits**. As a result, the learned vector field needs to "undo" this mismatch, leading to:

- Bent transport trajectories
- Loss of straightness and interpretability
- Inefficient inference (more ODE steps required)

![OT flow matching](/assets/images/efm/OT.png)

<a name="efm-methods"></a>

### 2.2 EFM: Orbit-aware Cost and Equivariant Model

To address this, the paper modifies the OT cost function so that point pairs are matched **within the same symmetry orbit**. The new cost is:

$$
\tilde{c}(x_0, x_1) = \min_{g \in G} \|x_0 - \rho(g)x_1\|^2,
$$

where $$G$$ is the symmetry group and $$\rho(g)$$ is the group action.

To solve this practically:

- **Translation** symmetry is handled by using **mean-free coordinates** (just remove the mean value)
- **Permutation** symmetry is handled by the **Hungarian algorithm** to optimally match particle indices
- **Rotation** symmetry is handled using the **Kabsch algorithm**, which finds the optimal orthogonal alignment between two point clouds

#### Quick primer — why Hungarian + Kabsch capture the two key symmetries

| Algorithm               | 3-line summary                                                                                                                                                                                                                                                                              | Captures which symmetry?                             | Why it works in EFM                                                                                                                                                                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hungarian algorithm** | Optimal assignment solver for a cost matrix in $$O(n^3)$$ time. <br> Finds a one-to-one mapping that minimises total cost. <br> Guarantees global optimum for linear-sum problems.                                                                                                          | **Permutation** (which atom ↔ which atom)            | Permuting indices of identical particles just re-orders rows/cols of the cost matrix. Hungarian picks the assignment with minimal pairwise L2 cost $$\|x_0^i - x_1^{\sigma(i)}\|$$. Thus it detects the best **permutation $$\sigma$$** without trying all $$N!$$ possibilities. |
| **Kabsch algorithm**    | Given two point clouds $$\{p_i\},\{q_i\}$$ with known correspondence, computes the orthogonal matrix $$R$$ (and optional translation) that minimises $$\sum_i\|Rp_i - q_i\|^2$$. <br> Closed-form via SVD of covariance matrix. <br> Produces det$$(R)=+1$$ unless reflections are allowed. | **Rotation / reflection** of the whole configuration | After permutation alignment, the two clouds differ only by a global rigid motion. Kabsch returns the best-fit $$R\in O(D)$$. Applying $$R$$ removes the rotational mismatch, so OT is computed **within the same rotation orbit**.                                               |

> In short: **Hungarian** aligns _which_ points match, **Kabsch** aligns _how_ they are oriented. Together they place $$x_0$$ and $$x_1$$ into a common orbit before OT pairing.

The model uses an **equivariant GNN** architecture designed to respect these group symmetries. The result is a vector field that remains equivariant and produces consistent flow directions under symmetric transformations.

![Equivariant OT flow matching](/assets/images/efm/eq_OT.png)

<a name="efm-results"></a>

### 2.3 Experimental Results

EFM shows superior performance in tasks involving highly symmetric distributions:

![Experiment results](/assets/images/efm/results.png)

#### LJ-55 Lennard-Jones cluster (55 atoms)

| What was measured                               | OT-FM (naive)                 | **Equivariant OT-FM**          | Take-away                                                  |
| ----------------------------------------------- | ----------------------------- | ------------------------------ | ---------------------------------------------------------- |
| **⟨Batch transport cost⟩** during training      | **≈10 ×** higher              | Low                            | Orbit‐aware pairing avoids “expensive” cross-orbit matches |
| **Median integration-path length** at inference | ≈ 10.2 $$\sigma$$             | **≈ 8.9 $$\sigma$$**           | Flows are almost straight, reducing ODE steps              |
| **RK4 fixed-step error** with 20 steps          | Large → needs adaptive solver | **Minimal**                    | Straightness ⇒ fixed small step OK                         |
| **Sampling speed-up** vs naive OT-FM            | —                             | **≈ 3 ×** faster (20 step RK4) | Symmetry handling pays off                                 |
| **NLL / ESS / Energy histograms**               | Worse                         | **Best on all metrics**        | Numerical accuracy translates to statistical gains         |

#### Alanine dipeptide (15-dim Cartesian backbone representation)

| Metric                            | OT-FM                                          | **Equivariant OT-FM**                                               | Comment                                         |
| --------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------- |
| **Effective sample size (ESS)**   | 0.46 ×                                         | **1 $$\times$$** (baseline)                                         | EFM’s straight paths improve mixing             |
| **Negative log-likelihood (NLL)** | $$\uparrow$$ (worse)                           | **Best**                                                            | Consistent with ESS                             |
| **Integration path length**       | Longer, broader                                | **Shorter, concentrated**                                           | Indicates straighter trajectories               |
| **Ramachandran free-energy**      | Overestimates minima                           | **Matches umbrella-sampling**                                       | Physical faithfulness                           |
| **Chirality handling**            | Generates both mirror states, longer inference | Equivariant flow respects SO(3) symmetry; reflection can be toggled | Practical benefit when one chirality is desired |

Accounting for permutation + rotation + translation symmetry **reduces mean transport cost by an order of magnitude**, straightens trajectories, and turns a slow, curved OT-FM into a fast sampler that beats or matches likelihood-based baselines on all evaluation metrics.

---

<a name="rfm"></a>

## 3. Paper II: Riemannian Flow Matching (RFM)

This paper generalizes flow matching to **non-Euclidean Riemannian manifolds**, allowing vector fields and probability paths to evolve over curved geometric spaces.

<a name="rfm-def"></a>

### 3.1 Definitions on Riemannian Manifolds

RFM formulates flow matching over a manifold $$\mathcal{M}$$ equipped with Riemannian metric $$g$$. The loss compares tangent vector fields $$v_t(x), u_t(x) \in T_x \mathcal{M}$$ using the local metric:

$$
\mathcal{L}_\text{RFM}(\theta) = \mathbb{E}_{t, p_t(x)} \|v_t(x) - u_t(x)\|_g^2.
$$

Probability paths $$p_t \in \mathcal{P}$$ interpolate between boundary conditions $$p_0 = p$$, $$p_1 = q$$. Conditional paths $$p_t(x \mid x_1)$$ and conditional vector fields $$u_t(x \mid x_1)$$ are defined, then marginalized as:

$$
p_t(x) = \int_\mathcal{M} p_t(x \mid x_1) q(x_1) \, d\text{vol}_{x_1}, \quad
u_t(x) = \int_\mathcal{M} u_t(x \mid x_1) \frac{p_t(x \mid x_1) q(x_1)}{p_t(x)} \, d\text{vol}_{x_1}.
$$

![Riemannian FM](/assets/images/rfm/cond_ut.png)

And, as we know from the background of flow matching, the Riemannian FM objective is equivalent to the
following Riemannian CFM objective:

$$
\mathcal{L}_\text{RCFM}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x \mid x_1)} \|v_t(x) - u_t(x \mid x_1)\|_g^2.
$$

<a name="rfm-vf"></a>

### 3.2 Vector Fields via Distance Functions

To construct Riemannian vector fields $$ u_t(x \mid x_1) $$, RFM introduces the concept of a distance-like function $$ d(x, x_1) $$ that satisfies the following properties:

1. **Non-negativity**: $$ d(x, y) \geq 0 $$ for all $$ x, y \in \mathcal{M} $$
2. **Positive definiteness**: $$ d(x, y) = 0 \iff x = y $$
3. **Non-degeneracy**: $$ \nabla d(x, y) \neq 0 \iff x \neq y $$

The central idea is to impose a **shrinking-speed condition** on the flow $$ x_t $$ toward the target point $$ x_1 $$, expressed as:

$$
d(x_t, x_1) = \kappa(t) d(x_0, x_1), \quad \kappa(0) = 1, \; \kappa(1) = 0,
$$

which enforces that the distance to the target shrinks according to schedule $$ \kappa(t) $$. Differentiating both sides with respect to time gives:

$$
\frac{d}{dt} d(x_t, x_1) = \dot{\kappa}(t) d(x_0, x_1)
\quad \Rightarrow \quad
\langle \nabla_x d, u_t \rangle_g = \dot{\kappa}(t) d(x_t, x_1),
$$

where the inner product is with respect to the Riemannian metric $$ g $$, and $$ \nabla_x d $$ denotes the gradient of the distance function at $$ x_t $$.

Among all vector fields satisfying this constraint, we want the one with minimal energy (i.e., minimal norm):

$$
\min_{u \in T_x \mathcal{M}} \frac{1}{2}\|u\|_g^2
\quad \text{s.t.} \quad
\langle \nabla_x d, u \rangle_g = \dot{\kappa}(t) d(x_t, x_1).
$$

Solving this Lagrangian optimization problem gives the unique minimum-norm solution that is **parallel to the gradient**:

$$
u_t(x \mid x_1) = \frac{d \log \kappa(t)}{dt} d(x,x_1) \frac{\nabla d(x,x_1)}{\|\nabla d(x,x_1)\|_g^2}.
$$

This formulation aligns with the **Cauchy–Schwarz** optimality condition, confirming both minimality and uniqueness.

One of the good choices for the distance function is using **geodesic**.

<a name="rfm-spectral"></a>

### 3.3 Approximating Geodesics with Spectral Distances

When $$\exp$$ and $$\log$$ maps are closed-form (e.g., $$\mathbb{S}^2$$, $$\mathbb{T}^d$$, Lie groups), one can directly compute $$x_t$$ without ODE simulation.

In general manifolds, geodesics are costly to compute, so RFM proposes **spectral distances** (e.g., diffusion and biharmonic distances) as approximations with some monotonically decreasing weighting function $$w : \mathbb{R} \rightarrow \mathbb{R}_{+}$$:

$$
d_w(x, y)^2 = \sum_{i=1}^{\infty} w(\lambda_i)(\varphi_i(x) - \varphi_i(y))^2,
$$

where $$\varphi_i : \mathcal{M} \rightarrow \mathbb{R}$$ are the eigenfunctions of the Laplace-Beltrami operator $$\nabla_g$$ over $$\mathcal{M}$$ with corresponding eigenvaluse $$\lambda_i$$, i.e. $$\nabla_g \varphi_i = \lambda_i \varphi_i$$.

In practice:

- Use top-$$k$$ eigenfunctions for fast approximation (one-time cost)
- Diffusion distance requires hyperparameter $$\tau$$, while biharmonic has no tunable hyperparameter
- Spectral distances yield smooth, robust flows and enable one-step ODE integration during inference

<a name="rfm-results"></a>

### 3.4 Experimental Results

RFM is tested on various manifolds (spheres, tori, meshes with boundaries). Key findings:

- **S² scenarios** (volcano/flood/fire) using geodesic distance:
  - Accurate sampling with no ODE steps
  - NLL improvements of 0.9–2 nats vs diffusion baselines
- **T² Ramachandran** (protein angles):
  - Matches SOTA on multimodal densities
- **7D torus RNA**:
  - Handles high‐dimensional angles, large gain in performance
- **Bunny mesh**:
  - Biharmonic spectral distance yields smoother flows than geodesic
- **Maze with walls**:
  - Biharmonic distance respects Neumann boundary conditions

![Results for eigenfunction](/assets/images/rfm/eig_results.png)

---

<a name="summary"></a>

## 4 · Summary

| **Aspect**                     | **Naive Flow Matching<br>(prior works)**                         | **Equivariant Flow Matching<br>(EFM)**                                                                        | **Riemannian Flow Matching<br>(RFM)**                                                                                                |
| ------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Ambient space**              | Flat **Euclidean $$\mathbb{R}^n$$**                              | Euclidean $$\mathcal{R}^n$$ with **group symmetries** (translations $$\cup$$ rotations $$\cup$$ permutations) | **General Riemannian manifold** $$\mathcal{M}$$ (curved, possibly with boundary)                                                     |
| **Ideal vector field $$u_t$$** | Linear / OT displacement <br> $$u_t = \dfrac{x_1 - x_t}{1 - t}$$ | _Same formula_, but **x₀–x₁ pairs are first aligned inside each symmetry orbit**                              | Analytic minimal-norm field <br> $$u_t = \dot{\log\kappa}(t)\,d\,\dfrac{\nabla d}{\|\nabla d\|_g^{2}}$$ (geodesic or spectral $$d$$) |
| **OT cost for pairing**        | Plain $$c(x_0,x_1)=\|x_0 - x_1\|^2$$                             | Orbit-aware cost function                                       | _No batch OT_ → independent draws; coupling not required                                                                             |
| **Symmetry handling**          | ❌ None → curved paths                                           | ✓ Translation (mean-free), permutation (Hungarian), rotation (Kabsch)                                         | Handled intrinsically by geometry; external symmetry not the focus                                                                   |
| **Distance / pre-metric**      | Euclidean $$L_2$$ only                                           | Euclidean $$L_2$$ — **after alignment**                                                                       | Flexible: geodesic, diffusion, biharmonic, …                                                                                         |
| **Model architecture**         | Standard MLP / GNN                                               | **SE(3) × S(N) equivariant GNN**                                                                              | Any manifold-aware NN; metric only in loss                                                                                           |
| **Extra training cost**        | None                                                             | Hungarian + Kabsch **per mini-batch**                                                                         | One-time Laplacian eigen-solve (if spectral)                                                                                         |
| **Inference ODE steps**        | Few-step RK4, may blow up if paths bent                          | 4–8 fixed steps (straighter)                                                                                  | 0 steps (closed-form manifolds) or ≈1–3 steps (spectral)                                                                             |
| **Typical datasets**           | Toy 2-D Gaussians, simple point clouds                           | LJ-13/LJ-55 clusters, alanine dipeptide                                                                       | $$S^2$$ weather, $$T^2$$ Ramachandran, 7-D torus RNA, bunny mesh, maze                                                               |
| **Primary win**                | Simpler than likelihood CNFs                                     | **Straightens OT paths under symmetry → faster inference**                                                    | **Extends FM to curved spaces → intrinsic modeling, analytic sampling**                                                              |
| **Main limitation**            | Breaks under symmetry; Euclidean only                            | Still Euclidean; extra batch OT cost                                                                          | Needs manifold tools (exp/log or Laplacian eigen-solve)                                                                              |

---

<a name="discussion"></a>

## 5. Discussions

### 5.1 When should I use which flow?

| Situation                                                      | Best-suited method      | Rationale                                                                   |
| -------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------------------- |
| **Euclidean data, little or no symmetry**                      | _Vanilla_ Flow Matching | simplest; no Hungarian/Kabsch cost                                          |
| **Euclidean + large symmetry group** (molecules, point-clouds) | **EFM**                 | orbit-aware cost drastically shortens paths                                 |
| **Intrinsic manifold data** (angles, meshes, SPD, hyperbolic)  | **RFM**                 | Euclidean straight line is meaningless; geodesic/spectral flows are natural |

### 5.2 Synergies and hybrid avenues

- **EFM ▸ RFM**  
  – Extend orbit-aware cost to **manifold-valued but symmetric** data (e.g. proteins on $$S^2$$ with interchangeable atoms).  
  – Combine Hungarian/Kabsch with RFM’s geodesic distance.

- **Spectral distances in Euclidean space**  
  – Even in $$\mathbb{R}^n$$, spectral distances (diffusion / biharmonic) can yield _smoother_ vector fields than $$L_2$$ for rough point-clouds.

### 5.3 Practical tips

1. **Batch size vs symmetry size**  
   Large symmetry orbits ⇒ bigger batches or orbit-aware pairing mandatory.
2. **Hungarian+Kabsch cost**  
   Fine for $$N\!\lesssim\!100$$; for larger $N$ consider approximate solvers (Sinkhorn, auction, ICP).
3. **Spectral pre-computation**  
   One eigen-solve dominates wall-time; cache eigenfunctions and reuse across epochs.

### 5.4 Limitations & open questions

- **EFM**:  
  – Still Euclidean; cannot handle curved ambient spaces.<br>
  – Hungarian O($$N^3$$) may choke on 1000+ interchangeable atoms.

- **RFM**:  
  – Requires either closed-form $${\exp,\log}$$ or Laplacian eigen-solve.<br>
  – Choice of weighting function $$w(\lambda)$$ (diffusion vs biharmonic) lacks principled tuning guidelines.

- **Both**:  
  – Deterministic; how to mix with stochastic bridges for better mode coverage?<br>
  – Memory footprint of CNFs still higher than score-based samplers.

### 5.5 Future research directions

1. **Geometry-aware stochastic flow matching** (Brownian shortcuts on manifolds).
2. **Learned pre-metrics**: jointly optimise $$w(\lambda)$$ or a neural kernel instead of picking diffusion/biharmonic.
3. **Hardware-friendly symmetry solvers**: GPU Hungarian, differentiable Kabsch layers.
4. **Probabilistic couplings**: Blend OT plan with noise → smoother training signal.
5. **Applications**: protein folding on $$SE(3) \times S^2$$ manifolds, brain-connectome meshes, climate fields on the sphere.

> **Bottom line**: EFM fixes crooked paths by respecting _external_ symmetries, while RFM bends the space so that every path becomes straight _intrinsically_. A unified “symmetry-on-manifold” flow would marry the best of both worlds.

---

$$
$$
