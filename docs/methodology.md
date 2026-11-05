# FoSA methodology

This page summarizes the method in the paper, section by section, with the paper's equation
numbers. It describes what the alignment computes; [running.md](running.md) describes how to run it.

## Contents

1. [Problem](#1-problem)
2. [Global affine fit](#2-global-affine-fit)
3. [Scale and shift as Fourier fields](#3-scale-and-shift-as-fourier-fields-sec-iv-a)
4. [Density-aware damping](#4-density-aware-damping-sec-iv-b)
5. [Residual Temporal Gradient Matching (R-TGM)](#5-residual-temporal-gradient-matching-r-tgm-sec-iv-c)
6. [Closed-form solve](#6-closed-form-solve-sec-iv-d)
7. [Selecting the spatial order](#7-selecting-the-spatial-order-sec-iv-e)
8. [Per-clip procedure](#8-per-clip-procedure)
9. [Evaluation metrics](#9-evaluation-metrics)
10. [Settings stated in the paper](#10-settings-stated-in-the-paper)

## 1. Problem

A frozen monocular backbone $\Phi$ maps frame $I_t$ to a dense prediction $x_t = \Phi(I_t)$ whose
metric scale is unreliable: scale-and-shift-invariant backbones leave it unconstrained, and metric
backbones drift once the sensor rig, scene scale, or environment departs from training. A range
sensor returns sparse metric depth $S_t$ on a mask $M_t \in \{0,1\}^{H\times W}$. Given a stream
$\{I_t, S_t\}_{t=1}^{T}$, the task is to fit a correction to $x_t$ on the observed pixels, extend it
to the rest of the image, and output dense metric depth $D_t$ that is accurate in every frame and
free of the flicker that frame-wise alignment introduces.

## 2. Global affine fit

The standard alignment fits one scale $s$ and one shift $b$ by least squares,

$$
\min_{s,b}\ \sum_{i,j} M_t(i,j)\,\bigl(s\,x_t(i,j) + b - \xi_t(i,j)\bigr)^2 , \tag{1}
$$

where $\xi_t$ are the measurements in the same representation as $x_t$. It sees only the pair
$(x_t, \xi_t)$ at each anchor and never its position, so every pixel with the same predicted value
receives the same correction. The residual of this fit therefore carries systematic spatial
structure, which FoSA is designed to capture.

## 3. Scale and shift as Fourier fields (Sec. IV-A)

FoSA lets both terms vary with the normalized pixel coordinate $\mathbf p \in [0,1]^2$:

$$
\hat\xi_t(\mathbf p) = a_t(\mathbf p)\,x_t(\mathbf p) + b_t(\mathbf p). \tag{2}
$$

Each field is expanded over a fixed set of damped Fourier carriers,

$$
a_t(\mathbf p) = a_t^0 + \sum_{\omega \in F_U} \lambda_\omega(\mathbf p)\,\bigl(\alpha_\omega \cos(\omega^\top \mathbf p) + \beta_\omega \sin(\omega^\top \mathbf p)\bigr), \tag{3}
$$

and identically for $b_t(\mathbf p)$ with its own coefficients. The frequencies come from a
half-disk of integer pairs,

$$
F_U = \bigl\{(n_u, n_v) \in \mathbb Z^2 : 0 < n_u^2 + n_v^2 \le U^2,\ n_u > 0 \text{ or } (n_u = 0, n_v > 0)\bigr\},\qquad \omega = 2\pi(n_u, n_v). \tag{4}
$$

- The half-plane condition keeps one frequency of each conjugate pair, and the zero frequency is
  excluded because $a_t^0$ and $b_t^0$ already carry it.
- The two fields have $Q = 2(1 + 2|F_U|)$ unknowns in total ($Q = 226$ for $U = 6$), collected in
  $\theta_t$.
- Since $x_t$ enters Eq. (2) as a known coefficient, $\hat\xi_t(\mathbf p) = A_t(\mathbf p)^\top \theta_t$
  stays linear in $\theta_t$. The design vector $A_t(\mathbf p)$ stacks
  $1, \lambda_\omega(\mathbf p)\cos(\omega^\top\mathbf p), \lambda_\omega(\mathbf p)\sin(\omega^\top\mathbf p)$,
  once multiplied by $x_t(\mathbf p)$ (scale field) and once as is (shift field).
- Setting every non-constant coefficient to zero recovers Eq. (1) exactly, and a stronger ridge on
  those coefficients moves the solution continuously back toward the affine fit.

**Normalization.** Before fitting, $x_t$ is scaled by the clip-level median ratio
$s_\text{init} = \operatorname{med}(\xi)/\operatorname{med}(x)$, which places $a_t^0$ near one,
and both predictor and target are divided by $\operatorname{rms}(\xi)$, which brings every column of
the design matrix to $O(1)$ so that the ridge has an effect.

## 4. Density-aware damping (Sec. IV-B)

A correction that oscillates between two neighboring anchors leaves no trace in the data. FoSA
therefore damps each carrier according to the local anchor spacing $\Delta(\mathbf p)$:

$$
\lambda_\omega(\mathbf p) = \exp\!\left(-\tfrac12\,\lVert\omega\rVert^2\,\Delta(\mathbf p)^2\right). \tag{5}
$$

A carrier survives where its wavelength is long compared with the local spacing and is suppressed
where it is short, so the effective degrees of freedom follow the Nyquist limit of the
measurements. The roll-off is smooth because the anchors sample the image irregularly.

$\Delta(\mathbf p)$ is estimated on a coarse grid:

1. Count the anchors in each 32-pixel cell.
2. Smooth the counts with a Gaussian of standard deviation 1.5 cells, giving $\tilde n(\mathbf p)$.
3. Set $\Delta(\mathbf p) = \bigl(|c| / \tilde n(\mathbf p)\bigr)^{1/2}$, where $|c|$ is the cell
   area in normalized coordinates.
4. Where no anchor lies nearby, $\Delta = 1$. Clamp $\Delta$ to $[2/\max(H,W),\ 1]$.

The density is smoothed rather than the spacing, so that empty regions (for example above the
LiDAR field of view) do not inflate $\Delta$ in the measured cells next to them. $\Delta$ is a
single isotropic scalar per location.

## 5. Residual Temporal Gradient Matching (R-TGM) (Sec. IV-C)

R-TGM constrains what the correction does to the output rather than how its coefficients move.
With the correction residual $r_t(\mathbf p) = \hat\xi_t(\mathbf p) - x_t(\mathbf p)$,

$$
r_t(\mathbf p) - r_{t-1}(\mathbf p) = \bigl(\hat\xi_t(\mathbf p) - \hat\xi_{t-1}(\mathbf p)\bigr) - \bigl(x_t(\mathbf p) - x_{t-1}(\mathbf p)\bigr), \tag{6}
$$

the change of the aligned prediction minus the change of the base prediction. Penalizing it asks
the alignment to follow the backbone's own temporal behavior and add no motion of its own.

**Grid penalty.** The term is evaluated on a fixed grid of $K$ points,

$$
R_\text{temp}(\theta_t) = \sum_{k=1}^{K} c_k\,\bigl[\Pi_t\,(A_K\theta_t - \tau)\bigr]_k^2, \tag{7}
$$

where row $k$ of $A_K \in \mathbb R^{K\times Q}$ is $A_t(\mathbf p_k)^\top$ and $\tau$ stacks the
targets inherited from the previous frame. A grid keeps the term from outweighing the sparse data
term and from depending on image resolution.

**Backbone-adaptive target.** Two targets are formed and blended:

$$
\tau_k^\text{res} = x_t(\mathbf p_k) + r_{t-1}(\mathbf p_k), \tag{8}
$$

$$
\tau_k^\text{field} = a_{t-1}(\mathbf p_k)\,x_t(\mathbf p_k) + b_{t-1}(\mathbf p_k), \tag{9}
$$

$$
\tau_k = \gamma\,\tau_k^\text{res} + (1-\gamma)\,\tau_k^\text{field},\qquad \gamma = \frac{1}{1 + \eta f},\qquad f = \frac{\operatorname{Var}_t[x_t]}{\operatorname{Var}_{\mathbf p}[x_t]}. \tag{10}
$$

The residual target also carries the previous frame's fluctuation, while the field target does
not. A steady backbone (small $f$) uses the residual target; a backbone that fluctuates relative to
the scene's depth range (large $f$) moves toward the field target. $\gamma$ is computed once per
clip from the base prediction alone, with no ground truth, optical flow, or camera pose. The paper
presents Eq. (10) as a heuristic: camera and scene motion also inflate $f$.

**Projection.** When a spatial field is selected ($U^\star > 0$), $\Pi_t$ removes the part of the
mismatch that a global affine change of $x_t$ explains, so the temporal term does not compete with
the anchors that re-determine the global scale and shift in every frame:

$$
\Pi_t = I - E_t\,(E_t^\top C E_t)^{-1} E_t^\top C,\qquad C = \operatorname{diag}(c_1,\dots,c_K),
$$

where row $k$ of $E_t$ is $[\,x_t(\mathbf p_k),\ 1\,]$. When the global fit is selected
($U^\star = 0$), $\Pi_t = I$.

**Point weights.** $c_k = w_\text{temp}\,\exp\!\bigl(-|x_t(\mathbf p_k) - x_{t-1}(\mathbf p_k)|/\sigma\bigr)$
loosens the constraint where the prediction changes sharply (for example at occlusion boundaries).
$\sigma$ is the median of $|x_t(\mathbf p_k) - x_{t-1}(\mathbf p_k)|$ over the grid. Since each
anchor contributes $O(1)$ to the data term, the temporal term matters most when anchors are scarce.

**System terms.** Using $\Pi_t^\top C\,\Pi_t = C\,\Pi_t$, Eq. (7) adds one matrix and one vector to
the normal equations:

$$
G = A_K^\top C\,\Pi_t A_K,\qquad g = A_K^\top C\,\Pi_t\,\tau. \tag{11}
$$

**Sequence boundaries.** On the first frame of a sequence, $G = 0$ and $g = 0$. At a clip boundary
the predecessor is the last frame of the previous clip; its residual is stored in metric units and
converted to the normalization of the current clip.

## 6. Closed-form solve (Sec. IV-D)

With the robust weights fixed, every term is quadratic and the frame is aligned by one linear solve:

$$
\bigl(A^\top W A + R + G\bigr)\,\theta_t = A^\top W \xi + R\,\bar\theta + g, \tag{12}
$$

- $A$ stacks the design vectors over the observed pixels and $W$ holds the current weights.
- $R$ is a diagonal ridge, strong on the non-constant coefficients and weak on the constant ones.
  Its prior $\bar\theta$ is one for $a_t^0$ and zero elsewhere.
- **Robust IRLS.** The Huber objective is convex in $\theta_t$. It is minimized by iteratively
  reweighted least squares: recompute $W$ from the residuals, re-solve Eq. (12), repeat five times.
- **Uniqueness.** A small ridge on the constant coefficients, scaled with the anchor count and
  centered on $a_t^0 = 1$, keeps the system nonsingular on the first frame or when anchors barely
  differ in predicted value.
- **Too few anchors.** A frame with fewer than two anchors skips the solve and keeps
  $\theta_t = \bar\theta$, so its output is the backbone prediction scaled by $s_\text{init}$.

**Monotonic projection.** After the reweighting ends, $a_t^0$ is clamped from below and the
non-constant scale coefficients are rescaled so that

$$
\sup_{\mathbf p}\ \sum_{\omega\in F_U} \lambda_\omega(\mathbf p)\sqrt{\alpha_\omega^2 + \beta_\omega^2}\ \le\ \rho\,a_t^0,\qquad 0 < \rho < 1. \tag{13}
$$

This gives $a_t(\mathbf p) \ge (1-\rho)\,a_t^0 > 0$ everywhere, hence
$\partial\hat\xi_t/\partial x_t > 0$: at every pixel, the aligned value increases monotonically with
the prediction, so nearer surfaces are never turned into farther ones. It does not preserve the
ordering across pixels. The projection is a safeguard applied after the solve, not a constraint
inside the objective.

No gradient passes through the backbone at any point.

## 7. Selecting the spatial order (Sec. IV-E)

The order $U$ is chosen once per clip on held-out measurements:

- Ordering frequencies by $n_u^2 + n_v^2$ makes the coefficients of a smaller $U$ a leading block of
  a larger one, so every candidate's normal equations are a leading principal submatrix of one
  accumulated system.
- To keep that system shared, robust weights are computed once from a low-order probe fit with
  $U = 2$, and the temporal term is excluded from scoring.
- 15% of the measurements are held out. The lowest-scoring candidate is selected, provided it does
  not exceed the score of the affine candidate; otherwise FoSA falls back to the global fit.
- If fewer than 32 anchors are held out, the score is too noisy to rank candidates, so FoSA uses
  the global fit directly and still runs the R-TGM recursion on it.
- After selection, $U^\star$ is fixed and the full temporal recursion is run once in the final refit.

## 8. Per-clip procedure

```text
input : backbone predictions x_1..x_T, sparse measurements ξ_1..ξ_T (masks M_t)
1. normalize: s_init = med(ξ)/med(x); divide predictor and target by rms(ξ)
2. γ from Eq. (10) using x_1..x_T
3. U* by held-out selection (Sec. 7), without the temporal term
4. for t = 1..T:
     Δ(p) from anchor density → λ_ω(p) (Eq. 5) → design matrix A_t
     G, g from R-TGM (Eqs. 7–11); zero on the first frame
     repeat 5×: W ← Huber weights; solve Eq. (12)
     monotonic projection (Eq. 13)
     D_t ← a_t(p) x_t(p) + b_t(p), mapped back to metric depth
     store r_t (in metric units) for the next frame
```

## 9. Evaluation metrics

With aligned depth $D_t$, ground truth $D_t^{gt}$, and valid mask $V_t$ (which includes the anchor
pixels), RMSE and MAE pool the valid pixels of all frames:

$$
\text{RMSE} = \sqrt{\frac{\sum_t\sum_{i,j} V_t(i,j)\,\bigl(D_t(i,j) - D_t^{gt}(i,j)\bigr)^2}{\sum_t\sum_{i,j} V_t(i,j)}}, \tag{14}
$$

$$
\text{MAE} = \frac{\sum_t\sum_{i,j} V_t(i,j)\,\bigl|D_t(i,j) - D_t^{gt}(i,j)\bigr|}{\sum_t\sum_{i,j} V_t(i,j)}. \tag{15}
$$

TEPE warps $D_t$ to frame $t+1$ with GMFlow optical flow, $\tilde D_t = \mathcal W(D_t, F_{t\to t+1})$,
and averages over all clips:

$$
\text{TEPE} = \frac{1}{(T-1)HW}\sum_{t=1}^{T-1}\sum_{i,j}\bigl|\tilde D_t(i,j) - D_{t+1}(i,j)\bigr|. \tag{16}
$$

TEPE needs no ground truth. All metrics are in meters. The ground truth of KITTI-DC and DDAD is
clipped at 80 m.

## 10. Settings stated in the paper

| Setting | Value |
| :--- | :--- |
| Density grid cell | 32 pixels |
| Density smoothing | Gaussian, σ = 1.5 cells |
| Spacing clamp | $\Delta \in [2/\max(H,W),\ 1]$; $\Delta = 1$ where no anchor lies nearby |
| IRLS iterations | 5 (Huber loss) |
| Held-out fraction | 15% of the measurements |
| Minimum held-out anchors | 32 (fewer → global fit + R-TGM) |
| Order-selection probe | $U = 2$ |
| R-TGM tolerance σ | Median of $\lvert x_t(\mathbf p_k) - x_{t-1}(\mathbf p_k)\rvert$ over the grid |
| Minimum anchors per frame | 2 (fewer → $\theta_t = \bar\theta$) |
| Fitting representation | Depth for UniDepth V2; inverse depth for VDA, DVD, DAv3 |

<!--
TODO(server): Add a second table, "Defaults in this repository", listing the values the code uses
for: maximum order U_max (and candidate set), η (Eq. 10), w_temp, number of grid points K (grid
size), ridge strengths (non-constant / constant), Huber δ, ρ (Eq. 13). Take them from the config
used for the paper's experiments, not from memory.
-->
