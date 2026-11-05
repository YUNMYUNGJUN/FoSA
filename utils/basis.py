"""Fourier basis (Sec. IV-A): the frequency set F_U and the damped design rows.

    theta = [a0, alpha_1, beta_1, ..., alpha_M, beta_M | b0, alpha'_1, beta'_1, ...]

The first block is the scale field a(p), the second the shift field b(p); each has
P = 1 + 2M coefficients, Q = 2P in total (Q = 226 for U = 6).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from .bandwidth import _unique_n2, lambda_matrix

TWO_PI = 2.0 * math.pi


# ===========================================================================
# frequency set
# ===========================================================================
def build_frequencies(order: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Half-plane integer lattice, sorted by n^2 ascending.

    Keeps n with n_u^2 + n_v^2 <= order^2 and (n_u > 0) or (n_u == 0 and n_v > 0),
    because cos(-w.p) = cos(w.p) makes +/-w the same basis pair.

    Sorting by n^2 is what makes every lower order U' a *prefix* of the
    coefficient vector, which the auto-order search relies on.
    """
    order = int(order)
    items: List[Tuple[int, int, int]] = []
    for n_u in range(0, order + 1):
        for n_v in range(-order, order + 1):
            if n_u == 0 and n_v <= 0:
                continue
            d = n_u * n_u + n_v * n_v
            if d <= order * order:
                items.append((d, n_u, n_v))
    items.sort(key=lambda it: (it[0], it[1], it[2]))
    if not items:
        return (torch.zeros((0, 2), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device))
    n_int = torch.tensor([[it[1], it[2]] for it in items], dtype=torch.long, device=device)
    n2 = torch.tensor([it[0] for it in items], dtype=torch.long, device=device)
    return n_int, n2


def order_counts(n2: torch.Tensor, order_max: int) -> List[int]:
    """M(U') for U' = 0..order_max, from the n^2-sorted frequency list."""
    n2_cpu = n2.detach().cpu()
    return [int((n2_cpu <= u * u).sum().item()) for u in range(int(order_max) + 1)]


def subset_indices(P_max: int, M_sel: int, device=None) -> torch.Tensor:
    """Coefficient indices of order U' inside the full-order layout.

    theta = [a0, a_1, b_1, ..., a_M, b_M | b0, a'_1, b'_1, ..., a'_M, b'_M]
    so order U' keeps the leading (1+2M') of each of the two blocks.
    """
    P_sel = 1 + 2 * int(M_sel)
    idx = list(range(0, P_sel)) + list(range(P_max, P_max + P_sel))
    return torch.tensor(idx, dtype=torch.long, device=device)


# ===========================================================================
# design matrix (built per chunk, never whole — memory rule 3)
# ===========================================================================
class _Basis:
    """Frequency table + cached n^2 de-duplication for the current order."""

    def __init__(self, n_int: torch.Tensor, n2: torch.Tensor):
        self.n_int = n_int
        self.n2 = n2
        self.M = int(n_int.shape[0])
        self.P = 1 + 2 * self.M
        self.Q = 2 * self.P
        if self.M:
            self.uniq, self.inv = _unique_n2(n2)
            self.wmat = (TWO_PI * n_int.to(torch.float32)).t().contiguous()   # [2, M]
        else:
            self.uniq, self.inv, self.wmat = [], None, None

    def truncate(self, M_sel: int) -> "_Basis":
        return _Basis(self.n_int[:M_sel], self.n2[:M_sel])


def build_phi(basis: _Basis, u: torch.Tensor, v: torch.Tensor, delta: torch.Tensor
              ) -> torch.Tensor:
    """Damped spatial basis Phi [c, P] = [1, lam*cos, lam*sin, ...] for one chunk."""
    c = u.numel()
    phi = u.new_empty((c, basis.P))
    phi[:, 0] = 1.0
    if basis.M:
        coords = torch.stack([u, v], dim=1)                     # [c, 2]
        phase = coords @ basis.wmat.to(u.device)                # [c, M]
        cs = torch.cos(phase)
        sn = torch.sin(phase)
        lam = lambda_matrix(delta, basis.n2, basis.uniq, basis.inv)
        cs.mul_(lam)
        sn.mul_(lam)
        phi[:, 1::2] = cs
        phi[:, 2::2] = sn
    return phi


def build_design(basis: _Basis, u, v, x_n, delta) -> torch.Tensor:
    """A [c, Q] = [x_n * Phi | Phi]."""
    phi = build_phi(basis, u, v, delta)
    A = u.new_empty((u.numel(), basis.Q))
    torch.mul(phi, x_n.unsqueeze(1), out=A[:, : basis.P])
    A[:, basis.P:] = phi
    return A
