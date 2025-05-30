# Geometry-Aware Generative Paths: From Group Orbits to Premetrics  
**Review of “Equivariant Flow Matching” (NeurIPS 2023) & “Flow Matching on General Geometries” (ICLR 2024)**  

> *One fixes crooked paths by aligning external symmetries; the other bends the space itself so every path becomes straight again.*

---

## Table of Contents
1. [A 60-Second Refresher on Flow Matching](#fm-refresher)  
2. [Part I – Equivariant Flow Matching (EFM)](#efm)  
   2.1 [Why standard OT-FM fails on symmetric data](#efm-motivation)  
   2.2 [Orbit-aligned cost & equivariant GNN](#efm-method)  
   2.3 [Empirical wins — Lennard-Jones 55 & alanine dipeptide](#efm-results)  
3. [Part II – Riemannian Flow Matching (RFM)](#rfm)  
   3.1 [When Euclidean coordinates are the real problem](#rfm-motivation)  
   3.2 [Premetric trick: geodesic vs spectral distances](#rfm-method)  
   3.3 [Experiments — sphere, torus, bunny mesh, maze](#rfm-results)  
4. [Common Threads & Diverging Strengths](#compare)  
5. [Open Horizons](#outlook)

---

<a name="fm-refresher"></a>
## 1 · A 60-Second Refresher on Flow Matching

Continuous Normalizing Flows (CNFs) learn a **vector field** \\(v_\\theta(t,x)\\) so that integrating the ODE  
\\[
\\dot x_t = v_\\theta(t,x_t), \\qquad t\\in[0,1]
\\]
pushes a simple base density to a complex target.  

**Flow Matching (FM)** turns this into pure regression:  
1. Pick an “ideal” vector field \\(u_t(x\\mid x_1)\\) that would move \\(x\\) to \\(x_1\\) in straight line.  
2. Minimize an MSE loss  
\\[
\\mathcal L = \\mathbb E\\bigl\\|v_\\theta(t,x_t)-u_t(x_t\\mid x_1)\\bigr\\|^2.
\\]  

If \\(u_t\\) is *simple* (e.g. OT displacement), we can integrate the learned field with **just a few fixed RK4 steps** at inference time.

---

<a name="efm"></a>
## 2 · Part I – Equivariant Flow Matching (EFM)

<a name="efm-motivation"></a>
### 2.1 When OT-FM meets symmetry hell 😵

Take a Lennard-Jones cluster of 55 atoms.  
*One* configuration has \\(55!\\times 8\\pi^2\\) symmetry copies (permutations × rotations).  
A mini-batch of size 512 samples only \\(\\sim10^5\\) pairs—**far too few** to hit the “right” copy.  

Outcome with vanilla OT-FM:

| | consequence |
|---|---|
| OT assignment matches *different* symmetry orbits | vector field must “bend” to undo mismatch |
| crooked paths | need smaller ODE step → slow inference |

<a name="efm-method"></a>
### 2.2 Orbit-aligned cost + equivariant field

**Key formula**

\\[
\\tilde c(x_0,x_1)=\\min_{g\\in G}\\|x_0-\\rho(g)x_1\\|_2^2,
\\]

where \\(G= O(D)\\times S(N)\\) (rotations + permutations).  

**Implementation**

1. **Permutation alignment** — Hungarian algorithm  
2. **Rotation alignment** — Kabsch algorithm  
3. Hungarian again on \\(\\tilde c\\) → OT plan on the *orbit* itself.

The vector field is implemented as an **SE(3) × S(N) equivariant GNN**, guaranteeing that the push-forward density stays symmetry-invariant.

<a name="efm-results"></a>
### 2.3 What do we gain?

| Dataset | Path length ↓ | Inference speed ↑ | Other wins |
|---------|---------------|-------------------|------------|
| LJ-55 | ×2 shorter | 3× faster | same likelihood |
| Alanine dipeptide | — | — | ΔF(φ) matches umbrella sampling |

Straight paths are back!  
EFM fixes symmetry with zero extra ODE cost.

---

<a name="rfm"></a>
## 3 · Part II – Riemannian Flow Matching (RFM)

<a name="rfm-motivation"></a>
### 3.1 What if the coordinates themselves are curved?

Spherical data, torus angles, mesh surfaces—flattening them breaks intrinsic distances.  
We need “straight” **within** the manifold.

<a name="rfm-method"></a>
### 3.2 Premetric trick 🔧

**Theorem 3.1**

Given any positive “distance-like” function \\(d(x,y)\\):
\\[
u_t(x\\mid x_1)=
\\dot\\kappa(t)\\;\\frac{d(x,x_1)}{\\|\\nabla d\\|_g^2}\\;\\nabla d(x,x_1)
\\]
is the *minimal-norm* vector field that shrinks \\(d(x_t,x_1)\\) according to schedule \\(\\kappa(t)\\).

<div style="margin-left:1.5em">

*Simple manifold*   → choose **geodesic distance** → closed form  
\\(x_t = \\exp_{x_1}\\!\\bigl((1-t)\\,\\log_{x_1}x_0\\bigr)\\) → **0 ODE steps**

*General manifold* → choose **spectral distance**  
\\[
d_w^2(x,y)=\\sum_{i=1}^k w(\\lambda_i)(\\varphi_i(x)-\\varphi_i(y))^2
\\]
(one-time eigen solve) → 1 forward ODE, **still divergence-free**

</div>

<a name="rfm-results"></a>
### 3.3 Highlights

| Manifold / Data | Metric used | NLL / Quality | Note |
|-----------------|-------------|---------------|------|
| S² volcano/flood/fire | geodesic | ↓ 0.9–2 nats vs diffusions | analytic sampling |
| T² Ramachandran | geodesic | matches SOTA | multimodal |
| 7-D torus RNA | geodesic | large gain | high-dim scalability |
| Bunny mesh | biharmonic | smoother than geodesic | one eigen solve |
| Maze (with walls) | biharmonic | flows respect boundary | Neumann BC |

---

<a name="compare"></a>
## 4 · Common Threads & Diverging Strengths

| Theme | EFM | RFM |
|-------|-----|-----|
| **Goal** | straighten OT paths in ℝⁿ | straighten paths *inside* manifold |
| **Geometry handle** | orbit-aligned cost | premetric (geodesic / spectral) |
| **When to use** | Euclidean data with large symmetry group | intrinsic manifolds (sphere, SPD, mesh) |
| **Training cost** | Hungarian + Kabsch per batch | eigen-solve once (spectral) |
| **Inference** | fixed RK4 (~8 steps) | 0 steps (simple) / few steps (general) |

---

<a name="outlook"></a>
## 5 · Open Horizons

* Mixing both ideas: equivariant flows **on** manifolds (e.g., permutable atoms on a sphere)  
* Adaptive or learned premetrics  
* Memory-light eigen decomposition for mega-scale meshes  
* Stochastic couplings & Brownian bridges within RFM

---

### Take-home sound-bite

> **EFM** aligns coordinates so external symmetries can’t fool the flow.  
> **RFM** reshapes the notion of “straight” so the flow can live happily on any geometry.  
> Together they turn few-step CNFs from a Euclidean toy into a geometry-aware toolbox.
