"""Spatial order selection on held-out anchors (Sec. IV-E).

The frequencies are sorted by n^2, so the normal equations of every order U <= U_max
are a leading block of one accumulated system. One IRLS run at U_max gives that system
for the fit anchors; each candidate order is then solved from it and scored by its MAE
on the held-out anchors, next to the plain least-squares affine fit as a safety
candidate. The chosen order is never worse than the global affine fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch

from .basis import _Basis, build_design, subset_indices
from .solver import _Frame, _accumulate, _irls, _solve, _solve_batched, ridge_vector

if TYPE_CHECKING:
    from .align import FosaConfig


_PLAIN_LS_RIDGE_SENTINEL = -1.0   # marks the plain least-squares candidate (see select_order)
# [a0, b0] = [1, 0] in the clip-normalised space, i.e. depth = s_init * x: the fallback
# for a frame whose affine fit is undefined (fewer than two anchors) or non-finite.
_AFFINE_IDENTITY = torch.tensor([1.0, 0.0], dtype=torch.float64)


_HASH_MASK32 = 0xFFFFFFFF


def _holdout_split(row: torch.Tensor, col: torch.Tensor, frame_k: int, W: int,
                   frac: float) -> torch.Tensor:
    """Deterministic hash split.  Same clip -> same split, every run.

    Uses a MurmurHash3-style avalanche finalizer, not an LCG.  An LCG (even
    composed several times) stays an affine map h = a*x + c; for a fixed row,
    adjacent columns produce adjacent x, so h(col) is itself an arithmetic
    progression and the "< threshold" selection comes out as a handful of
    contiguous bands rather than a scatter (confirmed: 274/1936 selected
    columns in one row formed 7 runs, longest ~39 wide). Concatenating
    frame_k additively before the LCG made it worse — two frames differ only
    by a constant shift, so their selected sets nearly coincide (confirmed:
    conditional overlap 0.688 against an independent-selection expectation of
    0.15). XOR + multiply + shift breaks the affine structure; each output
    bit depends on many input bits, so neither artifact reproduces.
    """
    if frac <= 0.0:
        return torch.zeros_like(row, dtype=torch.bool)
    r = row.to(torch.int64)
    c = col.to(torch.int64)
    h = ((r * 73856093) ^ (c * 19349663) ^ (int(frame_k) * 83492791)) & _HASH_MASK32
    h = ((h ^ (h >> 15)) * 2246822519) & _HASH_MASK32
    h = ((h ^ (h >> 13)) * 3266489917) & _HASH_MASK32
    h = (h ^ (h >> 16)) & 0x7FFFFFFF
    return h.to(torch.float64) < (float(frac) * 2147483648.0)


def select_order(frames: List[List[_Frame]], basis_max: _Basis, order_max: int,
                 counts_by_order: List[int], cfg: FosaConfig, ctx, device):
    """Choose the clip's Fourier order on the held-out anchors.

    Returns (order_sel, ridge_sel, plain_ls_selected, stats_all, ns_all, used_lstsq):
    stats_all / ns_all are the all-anchor normal equations at the maximum order (None
    when the search did not run), from which fit() slices the selected order.
    """
    B, T = len(frames), len(frames[0])
    # ---- order selection on the held-out anchors --------------------------
    order_sel = order_max
    ridge_sel = float(cfg.ridge)
    plain_ls_selected = False
    n_ho_total = sum(f.n_ho for row in frames for f in row)
    used_lstsq = False

    stats_all = None            # all-seed normal equations at the maximum order
    ns_all = None

    if n_ho_total >= 32 and basis_max.M > 0:
        # One IRLS run at the maximum order.  It yields the fit-part normal
        # equations (used to score candidates against the untouched holdout)
        # and, from the same converged weights, the holdout block — so their
        # sum is the all-seed system and no second pass over the seeds is
        # needed for the final fit.
        _, stats_fit, ns_fit, ls0, stats_ho_block, ns_all = _irls(
            frames, basis_max, float(cfg.ridge), cfg, device, with_holdout=True, ctx=ctx)
        used_lstsq = used_lstsq or ls0
        stats_all = [[(stats_fit[b][t][0] + stats_ho_block[b][t][0],
                       stats_fit[b][t][1] + stats_ho_block[b][t][1])
                      for t in range(T)] for b in range(B)]

        # Every candidate order is solved in the *full* Q_max layout: the excluded
        # coefficients get a unit diagonal and a zero right-hand side, so they
        # come out exactly zero and the kept block reproduces the sub-system.
        # Equal shapes make the candidates batchable, which is what keeps the
        # search cheap. The candidates are scored without the R-TGM term (it
        # depends on each candidate's own previous frame); it is applied in the
        # final fit below.
        Qm = basis_max.Q
        cand_meta: List[Tuple[int, float]] = []
        keep_rows: List[torch.Tensor] = []
        ridge_rows: List[torch.Tensor] = []
        for U in range(order_max + 1):
            sub = subset_indices(basis_max.P, counts_by_order[U], device)
            k = torch.zeros((Qm,), device=device, dtype=torch.float64)
            k.index_fill_(0, sub, 1.0)
            keep_rows.append(k)
            ridge_rows.append(ridge_vector(basis_max, ridge_sel, 1, device) * k)
            cand_meta.append((U, ridge_sel))
        keep = torch.stack(keep_rows)                       # [n_cand, Q]
        Rc = torch.stack(ridge_rows)                        # [n_cand, Q]
        KK = keep.unsqueeze(2) * keep.unsqueeze(1)          # [n_cand, Q, Q]

        cand_theta = [[None] * T for _ in range(B)]
        for b in range(B):
            for t in range(T):
                A, y = stats_fit[b][t]
                ns = float(max(ns_fit[b][t], 1))
                mat = A.unsqueeze(0) * KK
                dvec = Rc * ns + (1.0 - keep)               # identity off the subset
                rhs = y.unsqueeze(0) * keep
                rhs[:, 0] += Rc[:, 0] * ns                  # DC prior a0 -> 1, as in _solve_chain
                mat = mat + torch.diag_embed(dvec)
                sol, ls = _solve_batched(mat, rhs)
                used_lstsq = used_lstsq or ls
                cand_theta[b][t] = sol.to(torch.float32).t().contiguous()   # [Q, n_cand]
                del mat, rhs, dvec
        del KK

        # Plain-LS safety candidate: the closed-form per-frame affine fit (ridge 0,
        # weights all 1, no temporal coupling). Without it, holdout only compares
        # "FoSA at U'=0", still Huber-reweighted, against the higher orders, and
        # Huber's ~95% efficiency relative to least squares can lose a little to the
        # plain affine fit on a clip with no spatial structure. One [2,2] solve per frame.
        basis0 = basis_max.truncate(0)
        chunk = int(cfg.fit_chunk)
        for b in range(B):
            for t in range(T):
                fr = frames[b][t]
                w1 = torch.ones_like(fr.yn)
                AtWA0, AtWy0 = _accumulate(basis0, fr, w1, chunk)
                if fr.n_fit < 2:            # affine fit undefined: keep the clip scale
                    th0, ls = _AFFINE_IDENTITY.to(device), False
                else:
                    th0, ls = _solve(AtWA0, AtWy0)
                used_lstsq = used_lstsq or ls
                col = torch.zeros((Qm,), device=device, dtype=torch.float32)
                col[0] = th0[0].to(torch.float32)
                col[basis_max.P] = th0[1].to(torch.float32)
                cand_theta[b][t] = torch.cat([cand_theta[b][t], col.unsqueeze(1)], dim=1)
        cand_meta.append((0, _PLAIN_LS_RIDGE_SENTINEL))
        n_cand = len(cand_meta)

        # one design pass over the holdout seeds evaluates every candidate
        mae_acc = torch.zeros((n_cand,), device=device, dtype=torch.float64)
        cnt_acc = 0
        for b in range(B):
            for t in range(T):
                fr = frames[b][t]
                if fr.n_ho == 0:
                    continue
                theta_mat = cand_theta[b][t]
                cnt_acc += fr.n_ho
                for i0 in range(0, fr.n_ho, chunk):
                    i1 = min(fr.n_ho, i0 + chunk)
                    A = build_design(basis_max, fr.hu[i0:i1], fr.hv[i0:i1],
                                     fr.hxn[i0:i1], fr.hd[i0:i1])
                    res = (A @ theta_mat - fr.hyn[i0:i1].unsqueeze(1)).abs()
                    mae_acc += res.sum(dim=0).double()
                    del A, res
        mae = (mae_acc / float(max(cnt_acc, 1))).cpu().tolist()

        best_i = 0
        for i in range(n_cand):
            if mae[i] < mae[best_i] * (1.0 - 1e-4):
                best_i = i
        # never accept a candidate that loses to U'=0 (the global affine fit)
        mae_zero = min(mae[i] for i, (U, _) in enumerate(cand_meta) if U == 0)
        if mae[best_i] > mae_zero:
            best_i = min((i for i, (U, _) in enumerate(cand_meta) if U == 0),
                         key=lambda i: mae[i])
        order_sel, ridge_sel = cand_meta[best_i]
        plain_ls_selected = (ridge_sel == _PLAIN_LS_RIDGE_SENTINEL)
        if plain_ls_selected:
            ridge_sel = 0.0
        del cand_theta, stats_fit, stats_ho_block
    elif basis_max.M > 0:
        # Too few held-out anchors to choose an order: use the order that cannot be
        # worse than the global affine fit, U' = 0, as the plain affine fit itself.
        print(f"[FOSA][warn] only {n_ho_total} holdout seeds; auto-order cannot run, "
              f"falling back to U'=0 (global affine fit)")
        order_sel = 0
        plain_ls_selected = True

    return order_sel, ridge_sel, plain_ls_selected, stats_all, ns_all, used_lstsq
