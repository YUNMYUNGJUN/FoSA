"""Per-frame solve (Sec. IV-D): normal equations with ridge and R-TGM terms, Huber IRLS,
and the monotonic projection that keeps the scale field positive.

fp64 is used only for the [Q,Q] normal equations and their solution; the design matrix
is built per anchor chunk, folded into A^T W A and dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from .bandwidth import _EPS, lambda_max
from .basis import _Basis, build_design, build_phi
from .temporal import field_record, rtgm_terms

if TYPE_CHECKING:
    from .align import FosaConfig


# ===========================================================================
# regularisation
# ===========================================================================
def ridge_vector(basis: _Basis, ridge: float, n_seeds: int, device) -> torch.Tensor:
    """R[q] = ridge * N_seeds, with a tiny numerical ridge on the two DC terms."""
    r = torch.full((basis.Q,), float(ridge) * float(n_seeds),
                   dtype=torch.float64, device=device)
    dc_eps = 1e-8 * max(float(ridge), 1.0) * float(max(n_seeds, 1))
    r[0] = dc_eps
    r[basis.P] = dc_eps
    return r



# ===========================================================================
# linear algebra helpers
# ===========================================================================
# A [Q,Q] fp64 factorisation this small is pure launch overhead on the GPU.
# Measured at Q=226: cuSOLVER 0.839 ms, host LAPACK 0.234 ms *including* both
# transfers.  Above this size the O(Q^3) host cost takes over again.
_HOST_SOLVE_MAX_Q = 384


def _solve(mat: torch.Tensor, rhs: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Solve one [Q,Q] system.

    Deliberately does *not* validate the result with torch.isfinite here: that
    would cost a device sync per solve (measured 0.34 ms of the 1.19 ms total
    at Q=226).  Finiteness is checked once per clip on the assembled theta.
    """
    dev = mat.device
    host = dev.type == "cuda" and mat.shape[-1] <= _HOST_SOLVE_MAX_Q
    m, r = (mat.cpu(), rhs.cpu()) if host else (mat, rhs)
    try:
        sol, ls = torch.linalg.solve(m, r), False
    except Exception:
        sol, ls = torch.linalg.lstsq(m, r.unsqueeze(1)).solution[:, 0], True
    return (sol.to(dev) if host else sol), ls


def _solve_batched(mats: torch.Tensor, rhs: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Solve n symmetric positive-definite [Q,Q] systems at once.

    The auto-order search needs one solve per candidate per frame; done one at
    a time on the GPU that is ~0.84 ms each and dominates the whole fit.
    Batching plus the host route brings seven of them to ~0.5 ms total.
    Falls back to the general solver if any factorisation fails.
    """
    dev = mats.device
    host = dev.type == "cuda" and mats.shape[-1] <= _HOST_SOLVE_MAX_Q
    m, r = (mats.cpu(), rhs.cpu()) if host else (mats, rhs)
    L, info = torch.linalg.cholesky_ex(m)
    if not bool((info != 0).any()):
        sol, ls = torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1), False
    else:
        try:
            sol, ls = torch.linalg.solve(m, r), True
        except Exception:
            sol, ls = torch.linalg.lstsq(m, r.unsqueeze(-1)).solution.squeeze(-1), True
    return (sol.to(dev) if host else sol), ls


def _huber_weights(r: torch.Tensor, delta_arg: float) -> torch.Tensor:
    """w = 1 for |r| <= d, else d/|r|.  d = 1.345 * 1.4826 * MAD(r) when auto."""
    if r.numel() == 0:
        return r
    if delta_arg is not None and float(delta_arg) >= 0.0:
        d = r.new_tensor(float(delta_arg))
    else:
        med = r.median()
        mad = (r - med).abs().median()
        d = (1.345 * 1.4826) * mad.clamp_min(_EPS)
    a = r.abs()
    return torch.where(a <= d, torch.ones_like(r), d / a.clamp_min(_EPS))


# ===========================================================================
# per-frame seed records
# ===========================================================================
@dataclass
class _Frame:
    b: int
    t: int
    u: torch.Tensor
    v: torch.Tensor
    xn: torch.Tensor
    yn: torch.Tensor
    d: torch.Tensor
    hu: torch.Tensor
    hv: torch.Tensor
    hxn: torch.Tensor
    hyn: torch.Tensor
    hd: torch.Tensor

    @property
    def n_fit(self) -> int:
        return int(self.u.numel())

    @property
    def n_ho(self) -> int:
        return int(self.hu.numel())


def _accumulate_arrays(basis: _Basis, u, v, xn, yn, d, w, chunk: int
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """A^T W A and A^T W y, accumulated chunk by chunk in fp64.

    The design matrix never exists whole: each chunk is built, folded into the
    accumulators and dropped (memory rule 3).
    """
    device = u.device
    AtWA = torch.zeros((basis.Q, basis.Q), device=device, dtype=torch.float64)
    AtWy = torch.zeros((basis.Q,), device=device, dtype=torch.float64)
    n = int(u.numel())
    step = max(1, int(chunk))
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        A = build_design(basis, u[i0:i1], v[i0:i1], xn[i0:i1], d[i0:i1])
        Aw = A * w[i0:i1].unsqueeze(1)
        AtWA += (Aw.t() @ A).double()
        AtWy += (Aw.t() @ yn[i0:i1]).double()
        del A, Aw
    return AtWA, AtWy


def _residual_arrays(basis: _Basis, u, v, xn, yn, d, theta, chunk: int) -> torch.Tensor:
    n = int(u.numel())
    out = u.new_empty((n,))
    th = theta.to(torch.float32)
    step = max(1, int(chunk))
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        A = build_design(basis, u[i0:i1], v[i0:i1], xn[i0:i1], d[i0:i1])
        out[i0:i1] = A @ th - yn[i0:i1]
        del A
    return out


def _accumulate(basis: _Basis, fr: _Frame, w: torch.Tensor, chunk: int):
    return _accumulate_arrays(basis, fr.u, fr.v, fr.xn, fr.yn, fr.d, w, chunk)


def _residuals(basis: _Basis, fr: _Frame, theta: torch.Tensor, chunk: int):
    return _residual_arrays(basis, fr.u, fr.v, fr.xn, fr.yn, fr.d, theta, chunk)


def _grid_phi(ctx, basis: _Basis, b: int, t: int) -> torch.Tensor:
    """build_phi at the R-TGM grid of frame (b, t), computed once per order and reused
    by every IRLS round and the final solve (the frequencies are a prefix, so M
    identifies the basis)."""
    key = (basis.M, b, t)
    phi = ctx.phi_cache.get(key)
    if phi is None:
        phi = build_phi(basis, ctx.grid_u, ctx.grid_v, ctx.delta[b, t])
        ctx.phi_cache[key] = phi
    return phi


def _solve_chain(stats: List[List[Tuple[torch.Tensor, torch.Tensor]]],
                 basis: _Basis, ridge: float, n_seeds: List[List[int]],
                 ctx, device, sub: Optional[torch.Tensor] = None,
                 free_affine: bool = True
                 ) -> Tuple[List[List[torch.Tensor]], bool]:
    """Solve every frame, chaining each frame's R-TGM record into the next.

    Frame 0 uses ctx.carry, the previous clip's last frame, when one is given.

    ``sub`` optionally selects a lower-order sub-system out of the
    accumulated full-order normal equations (auto-order).  In that case
    ``basis`` must already be the *truncated* basis, so that R and the
    R-TGM grid regularizer are built at the sub-system size.
    """
    B = len(stats)
    T = len(stats[0])
    Rv = ridge_vector(basis, ridge, 1, device)          # per-seed unit, scaled below
    used_lstsq = False
    thetas: List[List[torch.Tensor]] = []
    for b in range(B):
        row: List[torch.Tensor] = []
        prev_rec = ctx.carry[b] if ctx.carry is not None else None
        for t in range(T):
            A, y = stats[b][t]
            if sub is not None:
                A = A.index_select(0, sub).index_select(1, sub)
                y = y.index_select(0, sub)
            ns = float(max(n_seeds[b][t], 1))
            mat = A + torch.diag(Rv * ns)
            # The DC ridge pulls a0 toward 1 (the clip normalisation already puts it
            # there) instead of toward 0. With anchors this is a 1e-8 relative nudge;
            # on a frame with no anchors and no temporal term it yields a0=1, b0=0,
            # i.e. the clip-level scale s_init, instead of a zero depth map.
            rhs = y.clone()
            rhs[0] += Rv[0] * ns
            phi = _grid_phi(ctx, basis, b, t)
            if prev_rec is not None:
                G, g = rtgm_terms(basis, prev_rec, ctx.x[b, t], ctx.s[b], ctx.y[b],
                                  ctx.grid_u, ctx.grid_v, ctx.delta[b, t], ctx.lam, ctx.weight,
                                  free_affine=free_affine, phi=phi)
                mat = mat + G
                rhs = rhs + g
            th, ls = _solve(mat, rhs)
            used_lstsq = used_lstsq or ls
            row.append(th)
            prev_rec = field_record(th, basis, ctx.grid_u, ctx.grid_v, ctx.delta[b, t],
                                    ctx.s[b], ctx.y[b], ctx.x[b, t], phi=phi)
        thetas.append(row)
    return thetas, used_lstsq


_IRLS_COARSE_M = 6   # leading frequencies (by n^2) used for the pre-final outlier-weight passes


def _irls(frames: List[List[_Frame]], basis: _Basis, ridge: float, cfg: FosaConfig, device,
          with_holdout: bool = False, ctx=None):
    """IRLS with per-frame normal equations and R-TGM chaining.

    Coarse-to-fine: only the *last* IRLS round needs to solve at the caller's
    requested order.  Earlier rounds exist purely to settle the Huber weights
    (i.e. find outliers), and a handful of leading low-n^2 frequencies gives
    residuals close enough to do that — the DC + low order terms explain most
    of the signal, so weight decisions barely change whether the probe fit
    used M=6 or M=56 columns. Doing those rounds on a small basis instead of
    the full one cuts the dominant cost (fit is essentially "solve accumulate
    iters times"). When the
    requested basis is already <= _IRLS_COARSE_M columns (order <~2) this is
    a no-op, since the "coarse" basis equals the full one.

    Solving always uses the fit part of each frame.  When ``with_holdout`` is
    set, the holdout seeds are folded into a second set of normal equations
    using the converged Huber weights, so that

        AtWA_fit + AtWA_holdout

    is the all-seed system at the same order.  Because the frequencies are
    sorted by n^2, slicing that sum gives the all-seed system at *any* lower
    order too — which is what lets the auto-order search finish without a
    second pass over the data.
    """
    B, T = len(frames), len(frames[0])
    n_seeds = [[frames[b][t].n_fit for t in range(T)] for b in range(B)]
    weights = [[torch.ones_like(frames[b][t].yn) for t in range(T)] for b in range(B)]
    iters = max(1, int(cfg.irls_iters))
    chunk = int(cfg.fit_chunk)

    coarse_M = min(basis.M, _IRLS_COARSE_M)
    coarse_basis = basis if coarse_M == basis.M else basis.truncate(coarse_M)

    stats: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[None] * T for _ in range(B)]
    thetas = None
    used_lstsq = False
    for it in range(iters):
        is_last = (it == iters - 1)
        b_it = basis if is_last else coarse_basis
        cur_stats: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[None] * T for _ in range(B)]
        for b in range(B):
            for t in range(T):
                cur_stats[b][t] = _accumulate(b_it, frames[b][t], weights[b][t], chunk)
        cur_thetas, ls = _solve_chain(cur_stats, b_it, ridge, n_seeds, ctx, device)
        used_lstsq = used_lstsq or ls
        if is_last:
            stats = cur_stats
            thetas = cur_thetas
        else:
            for b in range(B):
                for t in range(T):
                    if frames[b][t].n_fit == 0:
                        continue
                    r = _residuals(b_it, frames[b][t], cur_thetas[b][t], chunk)
                    weights[b][t] = _huber_weights(r, float(cfg.huber_delta))
    if not with_holdout:
        return thetas, stats, n_seeds, used_lstsq, None, None

    stats_ho = [[None] * T for _ in range(B)]
    n_all = [[0] * T for _ in range(B)]
    for b in range(B):
        for t in range(T):
            fr = frames[b][t]
            n_all[b][t] = fr.n_fit + fr.n_ho
            if fr.n_ho == 0:
                Z = torch.zeros((basis.Q, basis.Q), device=device, dtype=torch.float64)
                stats_ho[b][t] = (Z, torch.zeros((basis.Q,), device=device,
                                                 dtype=torch.float64))
                continue
            r = _residual_arrays(basis, fr.hu, fr.hv, fr.hxn, fr.hyn, fr.hd,
                                 thetas[b][t], chunk)
            w_ho = _huber_weights(r, float(cfg.huber_delta))
            stats_ho[b][t] = _accumulate_arrays(basis, fr.hu, fr.hv, fr.hxn, fr.hyn,
                                                fr.hd, w_ho, chunk)
    return thetas, stats, n_seeds, used_lstsq, stats_ho, n_all


def project_monotonic(theta: torch.Tensor, n2: torch.Tensor, delta_min: float,
                      rho: float) -> Tuple[torch.Tensor, int]:
    """Scale the scale-block amplitudes so a(p) >= a0*(1-rho) > 0 everywhere.

    theta: [B, T, Q].  Only the scale block is touched.  delta_min is a
    single scalar taken from the coarse grid — no full map is built.
    """
    B, T, Q = theta.shape
    P = Q // 2
    M = (P - 1) // 2
    out = theta.clone()
    if M == 0:
        return out, 0
    lam_max = lambda_max(delta_min, n2[:M], dtype=out.dtype).to(out.device)
    hits = 0
    alpha = out[:, :, 1:P:2]
    beta = out[:, :, 2:P:2]
    C = (lam_max * torch.sqrt(alpha * alpha + beta * beta)).sum(dim=-1)      # [B,T]
    eps_a = 1e-3
    a0 = out[:, :, 0]
    a0_safe = torch.where(torch.isfinite(a0) & (a0 > eps_a),
                          a0, torch.full_like(a0, eps_a))
    out[:, :, 0] = a0_safe
    bound = float(rho) * a0_safe
    need = torch.isfinite(C) & (C > bound) & (C > 0)
    if bool(need.any()):
        factor = torch.where(need, bound / C.clamp_min(_EPS), torch.ones_like(C))
        out[:, :, 1:P] *= factor.unsqueeze(-1)
        hits = int(need.sum().item())
    return out, hits
