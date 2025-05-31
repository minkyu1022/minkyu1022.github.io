
## 1. Preliminaries

### 1.1 Flow Matching

Flow Matching (FM) is a powerful framework for constructing continuous normalizing flows (CNFs) by learning a vector field that transports a simple, known base distribution \( q(x_0) \) into a complex target distribution \( \mu(x_1) \). This is done by defining a continuous path \( x_t \) governed by an ODE:

$$
\frac{dx_t}{dt} = v_\theta(t, x_t), \quad t \in [0, 1],
$$

where \( v_\theta \) is a learnable vector field. A common training objective regresses \( v_\theta \) to the conditional vector field

$$
u_t(x_t \mid x_1) = \frac{x_1 - x_t}{1 - t},
$$

resulting in the loss:

$$
\mathbb{E}_{t, x_0, x_1} \left[\|v_\theta(t, x_t) - u_t(x_t \mid x_1)\|^2\right].
$$

### 1.2 Optimal Transport (OT)

OT is a classic problem that finds a coupling \( \pi(x_0, x_1) \) between \( q(x_0) \) and \( \mu(x_1) \) that minimizes the transport cost:

$$
\pi^* = \arg\min_{\pi \in \Pi(q, \mu)} \mathbb{E}_{(x_0, x_1) \sim \pi}[c(x_0, x_1)].
$$

In flow matching, OT gives the "ideal" sample pairings for training.

---

## 2. Paper 1: Equivariant Flow Matching (EFM)

This paper addresses OT flow matching under group symmetries (translation, rotation, permutation) in Euclidean space.

### 2.1 Naive OT Flow Matching is Inefficient

Naively matching pairs without symmetry consideration leads to highly curved OT paths—resulting in poor regression and inference inefficiency.

### 2.2 Symmetry-aware OT Objective

To resolve this, the cost is modified to be orbit-aware:

$$
\tilde{c}(x_0, x_1) = \min_{g \in G} \|x_0 - \rho(g) x_1\|^2,
$$

for group \( G \), and action \( \rho(g) \). Each symmetry is handled via:

- **Translation**: mean-centering
- **Permutation**: Hungarian algorithm
- **Rotation**: Kabsch algorithm

An equivariant GNN is trained with OT pairs respecting symmetry.

### 2.3 Experimental Results

EFM outperforms naive OT on symmetric molecular datasets (e.g., LJ-13, LJ-55, alanine dipeptide), reducing path curvature and improving NLL.

---

## 3. Paper 2: Riemannian Flow Matching (RFM)

RFM generalizes FM to non-Euclidean manifolds, where straight lines are replaced by geodesics.

### 3.1 Riemannian Setup

RFM assumes data lies on a Riemannian manifold \( \mathcal{M} \), with geodesic distance \( d(x, y) \). A geodesic-based schedule defines:

$$
d(x_t, x_1) = \kappa(t) d(x_0, x_1),
$$

with constraint:

$$
\langle \nabla_x d, u_t \rangle_g = \dot{\kappa}(t) d(x_t, x_1),
$$

whose minimal-norm solution is:

$$
u_t(x \mid x_1) = \frac{d \log \kappa(t)}{dt} \cdot d(x,x_1) \cdot \frac{\nabla d(x,x_1)}{\|\nabla d(x,x_1)\|_g^2}.
$$

### 3.2 Training on Riemannian Manifold

The RFM training loss uses:

$$
\mathbb{E}_{x, t} \left[ \|v_t(x) - u_t(x \mid x_1)\|_g^2 \right].
$$

This requires evaluating Riemannian gradients \( \nabla d \), which depend on the manifold geometry.

### 3.3 Spectral Approximation of Distance

When geodesics are hard to compute, RFM proposes using **spectral distances**:

$$
d_w(x, y)^2 = \sum_{i=1}^\infty w(\lambda_i)(\varphi_i(x) - \varphi_i(y))^2
$$

where \( (\lambda_i, \varphi_i) \) are Laplacian eigenpairs. This yields smooth approximations like:

- **Diffusion distance**: \( w(\lambda) = e^{-\lambda t} \)
- **Biharmonic distance**: \( w(\lambda) = \lambda^{-2} \)

One eigen-solve suffices for all samples, enabling scalable learning.

### 3.4 Experimental Results

RFM matches or exceeds SOTA on tasks involving:

| **Manifold / Data**     | **Metric used** | **NLL / Quality**         | **Note**                  |
|-------------------------|------------------|----------------------------|----------------------------|
| S² volcano/flood/fire   | geodesic         | ↓ 0.9–2 nats vs diffusions | analytic sampling         |
| T² Ramachandran         | geodesic         | matches SOTA              | multimodal                |
| 7-D torus RNA           | geodesic         | large gain                | high-dim scalability      |
| Bunny mesh              | biharmonic       | smoother than geodesic    | one eigen solve           |
| Maze (with walls)       | biharmonic       | flows respect boundary    | Neumann BC                |
