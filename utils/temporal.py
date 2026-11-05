"""Residual Temporal Gradient Matching, R-TGM (Sec. IV-C).

Each frame is tied to the previous one (across clip boundaries too) on a K x K grid:
the aligned value at the grid is pulled toward a blend of two targets, with the blend
gamma set by how much the backbone flickers (Eq. 10). ``rtgm_terms`` returns the
quadratic term G, g that the per-frame solve adds to its normal equations (Eq. 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .basis import _Basis, build_phi


def _flicker_ratio(base_5d: torch.Tensor) -> float:
    """temporal_var / spatial_var of the backbone output, GT-free.

    High ratio: depth at a fixed pixel jumps around frame to frame relative
    to how much depth varies within one frame -> flickering, not motion.
    Low ratio: whatever change there is has spatial structure -> more likely
    real motion/geometry the model should track, not noise to damp out.
    """
    if base_5d.shape[1] < 2:
        return 0.0
    temporal_var = base_5d.var(dim=1, unbiased=False).mean()
    spatial_var = base_5d.var(dim=(3, 4), unbiased=False).mean()
    return float(temporal_var / spatial_var.clamp_min(1e-8))


def compute_flicker_lambda(base_5d: torch.Tensor, alpha: float = 5.0) -> float:
    """base_5d: [B,T,1,H,W] model output (not GT).  One scalar for the clip.

    lambda -> 0 (field target) as flickering dominates; lambda -> 1 (residual
    target) as the temporal change looks spatially coherent.  alpha controls
    how sharply lambda reacts to the ratio.
    """
    if base_5d.shape[1] < 2:
        return 1.0
    ratio = _flicker_ratio(base_5d)
    return float(1.0 / (1.0 + float(alpha) * ratio))


def _build_rtgm_grid(K: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """K x K regular grid over [0,1]^2, flattened.  Built once per clip."""
    gu = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32)
    gv = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(gu, gv, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def _sample_base_at_grid(base_frame: torch.Tensor, grid_u: torch.Tensor,
                         grid_v: torch.Tensor) -> torch.Tensor:
    """base_frame: [H,W] (raw backbone depth).  grid_u,grid_v: [K] in [0,1]."""
    H, W = base_frame.shape[-2:]
    row = (grid_u * float(H - 1)).long().clamp(0, H - 1)
    col = (grid_v * float(W - 1)).long().clamp(0, W - 1)
    return base_frame[row, col]


@dataclass
class GridRecord:
    """One frame's alignment sampled at the R-TGM grid, in units that survive a change
    of clip normalisation: ``a`` is dimensionless, ``b_m`` is the shift times y_scale,
    ``s`` is that clip's s_init and ``x`` the backbone value (fit domain, before s_init).
    The aligned value at the grid is then a * s * x + b_m."""
    a: torch.Tensor
    b_m: torch.Tensor
    s: float
    x: torch.Tensor


def field_record(theta: torch.Tensor, basis: "_Basis", grid_u: torch.Tensor,
                 grid_v: torch.Tensor, delta_grid: torch.Tensor, s: float, y: float,
                 x_grid: torch.Tensor, phi: Optional[torch.Tensor] = None) -> GridRecord:
    """a(p_k), b(p_k) of one frame. delta_grid = Delta at the grid points; phi, if
    given, is build_phi(basis, grid_u, grid_v, delta_grid)."""
    if phi is None:
        phi = build_phi(basis, grid_u, grid_v, delta_grid)      # [K, P] fp32
    th32 = theta.to(torch.float32)
    P = basis.P
    return GridRecord(a=phi @ th32[:P], b_m=(phi @ th32[P:]) * float(y), s=float(s), x=x_grid)


def rtgm_terms(basis: "_Basis", prev: Optional[GridRecord], x_grid: torch.Tensor,
               s: float, y: float, grid_u: torch.Tensor, grid_v: torch.Tensor,
               delta_grid: torch.Tensor, lam: float, weight: float,
               free_affine: bool = True, phi: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """G [Q,Q], g [Q] of Eq. (11) for the current frame, both fp64.

    tau blends the two targets of Eqs. (8)-(9), written in metres and divided by the
    current y_scale:
        residual  tau_res   = s x_t + (xi_hat_{t-1} - s_{t-1} x_{t-1})
        field     tau_field = a_{t-1} s_{t-1} x_t + b_{t-1}
    Inside one clip (same s, y) these are exactly x_t + r_{t-1} and
    a_{t-1} x_t + b_{t-1} in the normalised space the fit works in; across a clip
    boundary they convert the previous clip's record to this clip's units.

    prev is None (no predecessor): returns zeros, so that frame is fit with no
    temporal constraint.
    """
    device = grid_u.device
    Q = basis.Q
    if prev is None:
        return (torch.zeros((Q, Q), dtype=torch.float64, device=device),
                torch.zeros((Q,), dtype=torch.float64, device=device))
    s, y = float(s), float(y)
    xn = x_grid * (s / y)                                       # current base, fit units
    xi_prev = prev.a * (prev.s * prev.x) + prev.b_m
    tau_res = (s * x_grid + xi_prev - prev.s * prev.x) / y
    tau_field = (prev.a * (prev.s * x_grid) + prev.b_m) / y
    tau = float(lam) * tau_res + (1.0 - float(lam)) * tau_field

    # Base depth changing a lot at a grid point usually means real motion there;
    # loosen the constraint instead of fighting it. The ratio to the median makes
    # this independent of the units of x.
    diff = (xn - prev.x * (s / y)).abs()
    sigma = diff.median().clamp_min(1e-6)
    c = float(weight) * torch.exp(-diff / sigma)

    K = grid_u.numel()
    if phi is None:
        phi = build_phi(basis, grid_u, grid_v, delta_grid)      # [K, P] fp32
    P = basis.P
    A_grid = torch.empty((K, Q), device=device, dtype=torch.float32)
    A_grid[:, :P] = phi * xn.unsqueeze(1)
    A_grid[:, P:] = phi

    if free_affine:
        # Penalise only the part of (A theta - tau) that one global scale and shift
        # cannot explain: project both onto the c-weighted complement of span{x, 1}.
        # The per-frame global affine then stays with the anchors (they pin it with
        # thousands of points), and the temporal term acts on the spatially varying
        # correction, which is where the fit adds frame-to-frame noise of its own.
        Ad = A_grid.double()
        td = tau.double()
        cd = c.double()
        D = Ad[:, [0, P]]                                       # columns of a0 and b0
        Dc = D * cd.unsqueeze(1)
        M = D.t() @ Dc + 1e-9 * torch.eye(2, dtype=torch.float64, device=device)
        coef = torch.linalg.solve(M, torch.cat([Dc.t() @ Ad, Dc.t() @ td.unsqueeze(1)], 1))
        coef_A, coef_t = coef[:, :Q], coef[:, Q]                 # [2, Q], [2]
        Ap = Ad - D @ coef_A
        tp = td - D @ coef_t
        Apc = Ap * cd.unsqueeze(1)
        return Apc.t() @ Ap, Apc.t() @ tp

    Ac = A_grid * c.unsqueeze(1)
    G = (Ac.t() @ A_grid).double()
    g = (Ac.t() @ tau).double()
    return G, g
