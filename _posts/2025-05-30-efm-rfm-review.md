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

1. [Preliminary: What is Flow Matching](#fm-refresher)
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

<a name="fm-refresher"></a>

## 1 · Preliminary: What is Flow Matching

Continuous Normalizing Flows (CNFs) learn a **vector field** $$v_\theta(t,x)$$ so that integrating the ODE

<div class="math-display">
$$
\dot x_t \;=\; v_\theta(t,x_t), \qquad t\in[0,1]
$$
</div>

pushes a simple base density to a complex target.

**Flow Matching (FM)** turns this into pure regression:

1. Pick an “ideal” vector field $$u_t(x\mid x_1)$$ that moves $$x$$ along a straight line to $$x_1$$.
2. Minimize an MSE loss

<div class="math-display">
$$
\mathcal{L} \;=\; \mathbb{E}\Bigl\|\,v_\theta(t,x_t)\;-\;u_t(x_t\mid x_1)\Bigr\|^2.
$$
</div>

If $$u_t$$ is simple (e.g. OT displacement), we can integrate the learned field with **just a few fixed RK4 steps** at inference time.

---

<a name="efm"></a>

## 2 · Part I – Equivariant Flow Matching (EFM)

<a name="efm-motivation"></a>

### 2.1 Why OT-FM meets symmetry hell 😵

Take a Lennard-Jones cluster of 55 atoms.  
_One_ configuration has $$55!\times 8\pi^2$$ symmetry copies (permutations × rotations).  
A mini-batch of 512 samples covers only $$\sim10^5$$ pairs—**far too few** to land on the “right” copy.

**Outcome with vanilla OT-FM:**

<table>
  <thead>
    <tr>
      <th></th>
      <th>consequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OT assignment matches <em>different</em> symmetry orbits</td>
      <td>vector field must “bend” to undo mismatch</td>
    </tr>
    <tr>
      <td>crooked paths</td>
      <td>need smaller ODE step → slow inference</td>
    </tr>
  </tbody>
</table>

<a name="efm-method"></a>

### 2.2 Orbit-aligned cost + equivariant field

**Key formula**

<div class="math-display">
$$
\tilde c(x_0,x_1)
= \min_{g\in G}\,\bigl\|x_0 - \rho(g)x_1\bigr\|_2^2,
$$
</div>

where $$G = O(D)\times S(N)$$ (rotations + permutations).

**Implementation steps:**

1. **Permutation alignment** — Hungarian algorithm
2. **Rotation alignment** — Kabsch algorithm
3. Hungarian again on $$\tilde c$$ → OT plan on the _orbit_ itself.

The learned vector field is an **SE(3) × S(N) equivariant GNN**, ensuring the push-forward density stays symmetry-invariant.

<a name="efm-results"></a>

### 2.3 What do we gain?

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Path length ↓</th>
      <th>Inference speed ↑</th>
      <th>Other wins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LJ-55</td>
      <td>×2 shorter</td>
      <td>3× faster</td>
      <td>same likelihood</td>
    </tr>
    <tr>
      <td>Alanine dipeptide</td>
      <td>—</td>
      <td>—</td>
      <td>ΔF(φ) matches umbrella sampling</td>
    </tr>
  </tbody>
</table>

Straight paths are back!  
EFM fixes symmetry with zero extra ODE cost.

---

<a name="rfm"></a>

## 3 · Part II – Riemannian Flow Matching (RFM)

<a name="rfm-motivation"></a>

### 3.1 When Euclidean coordinates are curved

Spherical data, torus angles, mesh surfaces—flattening them breaks intrinsic distances.  
We need “straight” **within** the manifold.

<a name="rfm-method"></a>

### 3.2 Premetric trick 🔧

**Theorem 3.1**  
Given any positive “distance-like” function $$d(x,y)$$:

<div class="math-display">
$$
u_t(x\mid x_1)
= \dot{\kappa}(t)\,\frac{d(x,x_1)}{\bigl\|\nabla d\bigr\|_g^2}\,\nabla d(x,x_1)
$$
</div>

is the _minimal-norm_ vector field that shrinks $$d(x_t,x_1)$$ according to schedule $$\kappa(t)$$.

> **Simple manifold** → choose **geodesic distance** → closed form
>
> <div class="math-display">
> $$x_t = \exp_{x_1}\!\bigl((1-t)\,\log_{x_1}x_0\bigr)$$
> </div>
>
> → **0 ODE steps**

> **General manifold** → choose **spectral distance**
>
> <div class="math-display">
> $$d_w^2(x,y)
>   = \sum_{i=1}^k w(\lambda_i)\bigl(\varphi_i(x)-\varphi_i(y)\bigr)^2$$
> </div>
>
> (one-time eigen solve) → **still divergence-free**

<a name="rfm-results"></a>

### 3.3 Highlights

<table>
  <thead>
    <tr>
      <th>Manifold / Data</th>
      <th>Metric used</th>
      <th>NLL / Quality</th>
      <th>Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S² volcano/flood/fire</td>
      <td>geodesic</td>
      <td>↓ 0.9–2 nats vs diffusions</td>
      <td>analytic sampling</td>
    </tr>
    <tr>
      <td>T² Ramachandran</td>
      <td>geodesic</td>
      <td>matches SOTA</td>
      <td>multimodal</td>
    </tr>
    <tr>
      <td>7-D torus RNA</td>
      <td>geodesic</td>
      <td>large gain</td>
      <td>high-dim scalability</td>
    </tr>
    <tr>
      <td>Bunny mesh</td>
      <td>biharmonic</td>
      <td>smoother than geodesic</td>
      <td>one eigen solve</td>
    </tr>
    <tr>
      <td>Maze (with walls)</td>
      <td>biharmonic</td>
      <td>flows respect boundary</td>
      <td>Neumann BC</td>
    </tr>
  </tbody>
</table>

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
