---
layout: post
title: "[AI810] Blog post for reviewing papers"
date: 2025-05-30
math: true
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

**Review of “Equivariant Flow Matching” (NeurIPS 2023) & “Flow Matching on General Geometries” (ICLR 2024)**

> _One fixes crooked paths by aligning external symmetries; the other bends the space itself so every path becomes straight again._

---

## Table of Contents

1. [Preliminaries](#prel)
   - [1.1 Flow Matching](#fm)
   - [1.2 Optimal Transport (OT)](#ot)
2. [Part I – Equivariant Flow Matching (EFM)](#efm)
   - [2.1 Why standard OT-FM fails on symmetric data](#efm-motivation)
   - [2.2 Orbit-aligned cost & equivariant GNN](#efm-method)
   - [2.3 Empirical wins — Lennard-Jones 55 & alanine dipeptide](#efm-results)
3. [Part II – Riemannian Flow Matching (RFM)](#rfm)
   - [3.1 When Euclidean coordinates are curved](#rfm-motivation)
   - [3.2 Premetric trick: geodesic vs spectral distances](#rfm-method)
   - [3.3 Experiments — sphere, torus, bunny mesh, maze](#rfm-results)
4. [Common Threads & Diverging Strengths](#compare)
5. [Open Horizons](#outlook)

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

---

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

## 2. Paper 1: Equivariant Flow Matching (EFM)

This paper focuses on improving OT-based flow matching in **Euclidean space with symmetric structures**, such as **translations**, **rotations**, and **permutations**. These symmetries commonly arise in molecular and physical systems, where multiple configurations belong to the same equivalence class (orbit).

### 2.1 Limitation of Naive OT Flow Matching

Using a naive OT flow matching loss on symmetric datasets often results in **highly curved paths**. This occurs because the OT solver may match source and target points that belong to **different symmetry orbits**. As a result, the learned vector field needs to "undo" this mismatch, leading to:

- Bent transport trajectories
- Loss of straightness and interpretability
- Inefficient inference (more ODE steps required)

![OT flow matching](/assets/images/efm/OT.png)

### 2.2 EFM: Orbit-aware Cost and Equivariant Model

To address this, the paper modifies the OT cost function so that point pairs are matched **within the same symmetry orbit**. The new cost is:

$$
\tilde{c}(x_0, x_1) = \min_{g \in G} \|x_0 - \rho(g)x_1\|^2,
$$

where $G$ is the symmetry group and $\rho(g)$ is the group action.

To solve this practically:

- **Translation** symmetry is handled by using **mean-free coordinates**
- **Permutation** symmetry is handled by the **Hungarian algorithm** to optimally match particle indices
- **Rotation** symmetry is handled using the **Kabsch algorithm**, which finds the optimal orthogonal alignment between two point clouds

The model uses an **equivariant GNN** architecture designed to respect these group symmetries. The result is a vector field that remains equivariant and produces consistent flow directions under symmetric transformations.

![Equivariant OT flow matching](/assets/images/efm/eq_OT.png)

### 2.3 Experimental Results

EFM shows superior performance in tasks involving highly symmetric distributions:

- **Lennard-Jones clusters** (LJ-13, LJ-55): produces straighter paths, faster inference
- **Alanine dipeptide**: matches free energy profiles with fewer ODE steps

![Experiment results](/assets/images/efm/results.png)

These results demonstrate that symmetry-aware OT cost functions and equivariant architectures allow FM to fully exploit geometric structures in the data.

---

<a name="rfm"></a>

## 3 · Paper II – Riemannian Flow Matching (RFM)

<a name="rfm-motivation"></a>

### 3.1 When Euclidean coordinates are curved

Spherical data, torus angles, mesh surfaces—flattening them breaks intrinsic distances.  
We need “straight” **within** the manifold.

<a name="rfm-method"></a>

### 3.2 Premetric trick 🔧

**Theorem 3.1**  
Given any positive “distance-like” function $d(x,y)$, we define a conditional vector field $u_t(x \mid x_1)$ that satisfies a shrinking‐speed constraint on geodesic distance. Namely, if $x_t$ evolves so that

$$
d(x_t,x_1) \;=\; \kappa(t)\,d(x_0,x_1),
\quad \kappa(0)=1,\;\kappa(1)=0,
$$

then differentiating gives

$$
\frac{d}{dt}\,d(x_t,x_1) \;=\; \dot\kappa(t)\,d(x_0,x_1)
\;\;\Longrightarrow\;\;
\bigl\langle \nabla_x d(x_t,x_1),\,u_t(x \mid x_1)\bigr\rangle_g
\;=\; \dot\kappa(t)\,d(x_t,x_1).
$$

Among all tangent vectors $u \in T_x\mathcal{M}$ satisfying that inner‐product constraint, we choose the one with **minimal Riemannian norm** (i.e., minimal kinetic energy). Formally:

$$
\min_{\,u \in T_x\mathcal{M}\,}\;\frac12\,\|u\|_g^2
\quad\text{s.t.}\quad
\bigl\langle \nabla_x d(x,x_1),\,u \bigr\rangle_g
\;=\; \dot\kappa(t)\,d(x,x_1).
$$

Solving this Lagrangian‐dual condition via Cauchy–Schwarz yields the unique minimum‐norm field \!that is parallel to the gradient of $d$:

$$
u_t(x \mid x_1)
\;=\;
\frac{d\,\log\kappa(t)}{dt}
\;\;d(x,x_1)\;\;
\frac{\nabla_x d(x,x_1)}{\|\nabla_x d(x,x_1)\|_g^2}.
$$

This exactly enforces that $d(x_t,x_1)$ shrinks at rate $\dot\kappa(t)$ while remaining the **lowest‐energy** vector field on the manifold.

<a name="rfm-results"></a>

### 3.3 Approximating Geodesics with Spectral Distances

When $\exp$/$\log$ maps are available in closed form (e.g.\ on $\mathbb{S}^2$, tori, or certain Lie groups), one can compute

$$
x_t \;=\; \exp_{x_1}\bigl((1-t)\,\log_{x_1} x_0\bigr)
$$

directly without any ODE integration.

In general manifolds, exact geodesics are expensive, so RFM instead uses \textbf{spectral distances} (e.g.\ diffusion or biharmonic distances). These can be written in terms of the Laplace–Beltrami eigenfunctions $\{\varphi_i\}$ and eigenvalues $\{\lambda_i\}$:

$$
d_w^2(x,y)
\;=\;
\sum_{i=1}^{\infty} w(\lambda_i)\,
\bigl(\varphi_i(x) - \varphi_i(y)\bigr)^2.
$$

In practice, we truncate to the top $k$ eigenpairs for a one‐time cost. Typical choices:

- **Diffusion distance**: choose $w(\lambda_i) = e^{-2\lambda_i \tau}$ for some $\tau>0$
- **Biharmonic distance**: choose $w(\lambda_i) = \lambda_i^{-2}$, no tunable hyperparameter

These spectral approximations yield smooth vector fields that are still divergence‐free and allow “one‐step” ODE integration at inference:

<div class="math-display">
$$
\mathcal{L}_\text{RFM}(\theta)
\;=\;
\mathbb{E}_{t,p_t(x)}\Bigl\|
v_\theta(t,x)
\;-\;u_t(x)
\Bigr\|_g^2,
\;\;
u_t(x)
\;\text{constructed via }d_w(\cdot,\cdot).
$$
</div>

### 3.4 Experimental Highlights

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

<p align="center">
  <img src="/assets/images/rfm/eig_results.png" alt="Eigenfunction results" width="700">
</p>

---

<a name="compare"></a>

## 4 · Common Threads & Diverging Strengths

<table>
  <thead>
    <tr>
      <th>Theme</th><th>EFM</th><th>RFM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Goal</strong></td>
      <td>straighten OT paths in ℝⁿ</td>
      <td>straighten paths <em>inside</em> manifold</td>
    </tr>
    <tr>
      <td><strong>Geometry handle</strong></td>
      <td>orbit-aligned cost</td>
      <td>premetric (geodesic / spectral)</td>
    </tr>
    <tr>
      <td><strong>When to use</strong></td>
      <td>Euclidean data with large symmetry group</td>
      <td>intrinsic manifolds (sphere, SPD, mesh)</td>
    </tr>
    <tr>
      <td><strong>Training cost</strong></td>
      <td>Hungarian + Kabsch per batch</td>
      <td>eigen-solve once (spectral)</td>
    </tr>
    <tr>
      <td><strong>Inference</strong></td>
      <td>fixed RK4 (~8 steps)</td>
      <td>0 steps (simple) / few steps (general)</td>
    </tr>
  </tbody>
</table>

---

<a name="outlook"></a>

## 5 · Open Horizons

- Mixing both ideas: equivariant flows **on** manifolds (e.g., permutable atoms on a sphere)
- Adaptive or learned premetrics
- Memory-light eigen decomposition for mega-scale meshes
- Stochastic couplings & Brownian bridges within RFM

---

### Take-home sound-bite

> **EFM** aligns coordinates so external symmetries can’t fool the flow.
> **RFM** reshapes the notion of “straight” so the flow can live happily on any geometry.
> Together they turn few-step CNFs from a Euclidean toy into a geometry-aware toolbox.

$$
$$
