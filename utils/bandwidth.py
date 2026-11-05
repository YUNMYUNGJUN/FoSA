"""Density-aware damping (Sec. IV-B).

Each frequency omega is attenuated by lambda_omega(p) = exp(-||omega||^2 Delta(p)^2 / 2),
where Delta(p) is the local anchor spacing. Delta lives on a coarse grid
(``compute_delta_coarse``) and is sampled with the same bilinear arithmetic at the seeds
(fit) and over row chunks (predict), so both see the same lambda.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .align import Seeds

TWO_PI_SQ = 2.0 * math.pi * math.pi          # 2*pi^2, the exponent of E(p)
_EPS = 1e-12
_DELTA_MAX = 1.0


# ===========================================================================
# lambda = E^{n^2} via integer powers
# ===========================================================================
def _unique_n2(n2: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
    uniq, inv = torch.unique(n2, sorted=True, return_inverse=True)
    return [int(v) for v in uniq.tolist()], inv


def lambda_exponent(delta: torch.Tensor) -> torch.Tensor:
    """g(p) = -2*pi^2*delta(p)^2, so that lambda_m = exp(g * n_m^2)."""
    return (-TWO_PI_SQ) * delta * delta


def lambda_powers(g: torch.Tensor, exponents: List[int]) -> List[torch.Tensor]:
    """lambda for each distinct n^2, one page per *distinct* value.

    Deliberate choice.  The textbook form is E = exp(-2*pi^2*d^2)
    followed by integer powers E^{n^2} built by repeated multiplication.  In
    fp32 that chain doubles its relative error at every squaring, so n^2 = 64
    lands ~40 ulp off — measured 4.9e-6 absolute, which fails the 1e-6 bound
    that tests/test_fosa_core.py checks for this exact quantity.  Evaluating exp(g*n^2) instead is
    accurate to 1.2e-7 and measured only 14% slower per chunk (0.243 ms vs
    0.213 ms at [64,1936], U=6), i.e. ~1.7 ms over a whole 1216x1936 T=3 clip.

    The constraint that actually mattered — never call exp M times per pixel —
    still holds: this calls it once per *distinct* n^2 (K = 14 at U=6, 20 at
    U=8, against M = 56 and 98).
    """
    return [torch.exp(g * float(k)) for k in exponents]


def lambda_matrix(delta: torch.Tensor, n2: torch.Tensor,
                  uniq: Optional[List[int]] = None,
                  inv: Optional[torch.Tensor] = None) -> torch.Tensor:
    """lambda[..., m] = exp(-0.5*||w_m||^2*delta^2), evaluated per distinct n^2."""
    if n2.numel() == 0:
        return delta.new_zeros((*delta.shape, 0))
    if uniq is None or inv is None:
        uniq, inv = _unique_n2(n2)
    cols = lambda_powers(lambda_exponent(delta), uniq)
    lam_u = torch.stack(cols, dim=-1)                       # [..., K]
    return lam_u.index_select(-1, inv.to(lam_u.device))     # [..., M]


def lambda_max(delta_min: float, n2: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """lambda at the *smallest* delta, i.e. the per-frequency upper bound.

    lambda is monotone decreasing in delta, so the max over the image is the
    value at delta_min.  No [H*W, M] tensor is ever built to find it.
    """
    if n2.numel() == 0:
        return torch.zeros((0,), dtype=dtype, device=n2.device)
    omega_norm2 = (4.0 * math.pi * math.pi) * n2.to(dtype)
    return torch.exp(-0.5 * (float(delta_min) ** 2) * omega_norm2)


# ===========================================================================
# bandwidth: coarse delta grid
# ===========================================================================
def _gaussian_blur_coarse(x: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Separable gaussian blur on the coarse [N,1,Hc,Wc] grid."""
    if sigma <= 0 or x.shape[-1] < 2 or x.shape[-2] < 2:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / float(sigma)) ** 2)
    k = k / k.sum().clamp_min(_EPS)
    ky = k.view(1, 1, -1, 1)
    kx = k.view(1, 1, 1, -1)
    x = torch.nn.functional.pad(x, (0, 0, radius, radius), mode="replicate")
    x = torch.nn.functional.conv2d(x, ky)
    x = torch.nn.functional.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = torch.nn.functional.conv2d(x, kx)
    return x


def compute_delta_coarse(seeds: Seeds, cell: int, scale: float) -> Tuple[torch.Tensor, int]:
    """Return (delta_coarse [B,T,Hc,Wc] fp32, cell_size).

    Per-cell anchor counts are blurred into a local density, which is turned into a
    spacing Delta = sqrt(cell area / density). A cell without anchors next to dense
    cells gets the spacing of its neighbourhood; only cells far from every anchor
    approach _DELTA_MAX.
    """
    B, T, H, W = seeds.B, seeds.T, seeds.H, seeds.W
    n_frames = B * T
    delta_min = 2.0 / float(max(H, W))

    S = max(1, int(cell))
    Hc = (H + S - 1) // S
    Wc = (W + S - 1) // S
    # per-cell seed counts, straight from the seed coordinates: this is
    # numerically identical to avg_pool2d over the validity mask but never
    # touches a full-resolution tensor.
    frame_id = torch.zeros_like(seeds.row)
    for k in range(n_frames):
        s0, s1 = seeds.starts[k], seeds.starts[k + 1]
        if s1 > s0:
            frame_id[s0:s1] = k
    cell_id = (frame_id * (Hc * Wc)
               + torch.div(seeds.row, S, rounding_mode="floor") * Wc
               + torch.div(seeds.col, S, rounding_mode="floor"))
    n_cell = torch.bincount(cell_id, minlength=n_frames * Hc * Wc).to(torch.float32)
    n_cell = n_cell.reshape(n_frames, 1, Hc, Wc)

    area_cell = float(S) * float(S) / (float(H) * float(W))
    n_s = _gaussian_blur_coarse(n_cell, sigma=1.5)
    d = torch.sqrt(area_cell / n_s.clamp_min(_EPS))
    d = torch.where(n_s > 1e-3, d, torch.full_like(d, _DELTA_MAX))
    d = (float(scale) * d).clamp(min=delta_min, max=_DELTA_MAX)
    return d.reshape(B, T, Hc, Wc), S


def _bilinear_1d(coord: torch.Tensor, n_pix: int, n_coarse: int, cell: int
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bilinear weights of a normalised coordinate against the coarse axis.

    Pixel i sits at centre (i + 0.5); coarse cell k at centre (k*S + S/2).
    Shared by the fit path (scattered seeds) and the predict path (row
    chunks) so that both see exactly the same Delta.
    """
    pos = (coord * float(max(n_pix - 1, 1)) + 0.5) / float(cell) - 0.5
    pos = pos.clamp(0.0, float(max(n_coarse - 1, 0)))
    i0 = pos.floor().to(torch.long).clamp(0, max(n_coarse - 1, 0))
    i1 = (i0 + 1).clamp(max=max(n_coarse - 1, 0))
    w1 = pos - i0.to(pos.dtype)
    return i0, i1, w1


def delta_at_seeds(delta_frame: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                   H: int, W: int, cell: int) -> torch.Tensor:
    """Delta at scattered seed positions.  delta_frame: [Hc, Wc]."""
    Hc, Wc = delta_frame.shape
    i0, i1, wy = _bilinear_1d(u, H, Hc, cell)
    j0, j1, wx = _bilinear_1d(v, W, Wc, cell)
    d00 = delta_frame[i0, j0]
    d01 = delta_frame[i0, j1]
    d10 = delta_frame[i1, j0]
    d11 = delta_frame[i1, j1]
    top = d00 + (d01 - d00) * wx
    bot = d10 + (d11 - d10) * wx
    return top + (bot - top) * wy


def delta_at_rows(delta_frame: torch.Tensor, u_rows: torch.Tensor, v_all: torch.Tensor,
                  H: int, W: int, cell: int) -> torch.Tensor:
    """Delta over one row chunk, shape [len(u_rows), W]. Same formula as above."""
    Hc, Wc = delta_frame.shape
    i0, i1, wy = _bilinear_1d(u_rows, H, Hc, cell)
    j0, j1, wx = _bilinear_1d(v_all, W, Wc, cell)
    row0 = delta_frame.index_select(0, i0)                  # [C, Wc]
    row1 = delta_frame.index_select(0, i1)
    d00 = row0.index_select(1, j0)
    d01 = row0.index_select(1, j1)
    d10 = row1.index_select(1, j0)
    d11 = row1.index_select(1, j1)
    top = d00 + (d01 - d00) * wx
    bot = d10 + (d11 - d10) * wx
    return top + (bot - top) * wy.unsqueeze(1)
