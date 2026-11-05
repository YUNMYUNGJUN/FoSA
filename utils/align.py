"""FoSA (Fourier-Oriented Sparse Alignment): the clip-level pipeline.

Aligns the depth of a frozen backbone to sparse metric anchors, one clip at a time:

    seeds = collect_seeds(sparse_5d, base_5d)
    state = fit(base_5d, seeds, cfg, carry=prev_state.carry)   # FosaState
    depth = predict(base_5d, state, cfg)                       # [B, T, 1, H, W]

The aligned depth is a(p) * x(p) + b(p), with a and b expanded on a 2D Fourier basis
whose high frequencies are damped by the local anchor spacing Delta(p). Per clip:
the Fourier order is chosen on held-out anchors (never worse than a global affine fit),
the fit is Huber-IRLS, the scale field is kept positive, and R-TGM ties each frame to
the previous one (across clip boundaries too, through ``carry``).

The steps live in the modules this one imports, one per section of the paper:
basis.py (IV-A), bandwidth.py (IV-B), temporal.py (IV-C), solver.py (IV-D) and
order.py (IV-E).

Memory rules:

  1. No tensor outside the backbone exceeds O(H*W) fp32.  No [H,W,M],
     [H*W,M] or [H,W,2] allocation exists anywhere in these modules.
  2. fp64 is used only for the [Q,Q] normal equations and their solution.
  3. The design matrix A is never materialised whole; it is built per seed
     chunk, accumulated into A^T W A, and dropped.
  4. Delta (local seed spacing) lives only on the coarse [B,T,Hc,Wc] grid.
  5. Coordinate grids are never materialised; only 1D tables u[H], v[W].
  6. The a(p) / b(p) fields are never materialised; they are consumed
     inside the row chunk that produced them.
  7. FosaState holds no full-resolution tensor.

The single source of truth for the bandwidth Delta is
``FosaState.delta_coarse``.  Both the fit path (scattered seed positions)
and the predict path (row chunks) sample it through ``_bilinear_1d`` with
identical arithmetic, so lambda is guaranteed consistent between them.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .bandwidth import compute_delta_coarse, delta_at_rows, delta_at_seeds, lambda_exponent
from .basis import TWO_PI, _Basis, build_frequencies, order_counts, subset_indices
from .order import _holdout_split, select_order
from .solver import _Frame, _accumulate_arrays, _grid_phi, _irls, _solve_chain, project_monotonic
from .temporal import (_build_rtgm_grid, _sample_base_at_grid, compute_flicker_lambda,
                       field_record)


# ===========================================================================
# configuration and state
# ===========================================================================
def _opt(default, help):
    return field(default=default, metadata={"help": help})


@dataclass
class FosaConfig:
    """FoSA hyper-parameters; the defaults are the paper's setting."""
    domain: str = _opt("depth", "fit in 'depth' or 'inv' (inverse depth)")
    max_depth: float = _opt(200.0, "output clamp [0, max_depth]; 1/max_depth floors inverse depth")
    spatial_order: int = _opt(6, "maximum Fourier order; the order used is chosen per clip")
    holdout_frac: float = _opt(0.15, "fraction of anchors held out to choose the order")
    ridge: float = _opt(1e-2, "ridge on the Fourier coefficients, per anchor")
    bandwidth_scale: float = _opt(1.0, "Delta = bandwidth_scale * local anchor spacing")
    bandwidth_cell: int = _opt(32, "cell size (pixels) of the coarse Delta grid")
    irls_iters: int = _opt(5, "Huber IRLS iterations")
    huber_delta: float = _opt(-1.0, "Huber threshold; < 0: 1.345 * 1.4826 * MAD of the residuals")
    mono_rho: float = _opt(0.8, "keep the scale field >= (1 - mono_rho) * a0")
    min_valid: int = _opt(16, "warn when a frame has fewer anchors than this")
    rtgm_grid: int = _opt(8, "R-TGM constraints on a rtgm_grid x rtgm_grid grid")
    rtgm_weight: float = _opt(1.0, "weight of each R-TGM grid constraint")
    rtgm_alpha: float = _opt(5.0, "sensitivity of the R-TGM blend lambda to backbone flicker")
    fit_chunk: int = _opt(16384, "anchors per design-matrix chunk")
    predict_chunk: int = _opt(256, "image rows per predict chunk")


@dataclass
class Seeds:
    """Valid sparse-depth samples of one clip, extracted exactly once.

    All arrays are 1D and ordered by (b, t, row, col).  ``starts`` holds
    B*T+1 offsets so frame k occupies ``[starts[k], starts[k+1])``.
    """

    B: int
    T: int
    H: int
    W: int
    row: torch.Tensor          # [N] int64, frame-local row index
    col: torch.Tensor          # [N] int64, frame-local column index
    u: torch.Tensor            # [N] fp32, row coordinate normalised to [0,1]
    v: torch.Tensor            # [N] fp32, col coordinate normalised to [0,1]
    x: torch.Tensor            # [N] fp32, backbone depth at the seed
    y: torch.Tensor            # [N] fp32, sparse (measured) depth at the seed
    starts: List[int]
    counts: List[int]

    @property
    def n_total(self) -> int:
        return int(self.u.numel())


@dataclass
class FosaState:
    theta: torch.Tensor            # [B, T, Q] fp32, Q = 2*(1+2*M_sel)
    omega_n: torch.Tensor          # [M_sel, 2] int64 (integer n, 2*pi excluded)
    n2: torch.Tensor               # [M_sel] int64
    delta_coarse: torch.Tensor     # [B, T, Hc, Wc] fp32
    s_init: torch.Tensor           # [B] fp32
    y_scale: torch.Tensor          # [B] fp32
    domain: str
    cell: int
    order_sel: int                 # Fourier order chosen for the clip
    n_seeds: int
    rtgm_lambda: float             # 0 = field target, 1 = residual target
    carry: Optional[list] = None   # per-b GridRecord of the last frame, for the next clip


# ===========================================================================
# seed extraction
# ===========================================================================
def collect_seeds(sparse_5d: torch.Tensor, base_5d: torch.Tensor) -> Seeds:
    """Extract every valid seed of the clip once, as 1D tensors.

    After this call nothing in the pipeline indexes a full [B,T,1,H,W]
    tensor with a boolean mask again.
    """
    if sparse_5d.dim() != 5 or sparse_5d.shape[2] != 1:
        raise ValueError(f"sparse_5d must be [B,T,1,H,W], got {tuple(sparse_5d.shape)}")
    if base_5d.shape != sparse_5d.shape:
        raise ValueError(f"shape mismatch: base {tuple(base_5d.shape)} vs sparse {tuple(sparse_5d.shape)}")

    B, T, _, H, W = sparse_5d.shape
    device = sparse_5d.device
    sp = sparse_5d.reshape(-1)
    bs = base_5d.reshape(-1)
    mask = (sp > 0) & torch.isfinite(sp) & torch.isfinite(bs)
    idx = mask.nonzero(as_tuple=False).squeeze(1)                  # sorted ascending

    frame_size = H * W
    fr = torch.div(idx, frame_size, rounding_mode="floor")
    rem = idx - fr * frame_size
    row = torch.div(rem, W, rounding_mode="floor")
    col = rem - row * W

    bounds = torch.searchsorted(fr, torch.arange(B * T + 1, device=device, dtype=fr.dtype))
    bounds = bounds.clamp_max(idx.numel())
    starts = bounds.cpu().tolist()                                  # the single sync
    starts[-1] = int(idx.numel())
    counts = [starts[k + 1] - starts[k] for k in range(B * T)]

    du = float(max(H - 1, 1))
    dv = float(max(W - 1, 1))
    return Seeds(
        B=B, T=T, H=H, W=W,
        row=row, col=col,
        u=(row.to(torch.float32) / du),
        v=(col.to(torch.float32) / dv),
        x=bs.index_select(0, idx).float(),
        y=sp.index_select(0, idx).float(),
        starts=starts, counts=counts,
    )


# ===========================================================================
# fit
# ===========================================================================
def _domain_xy(x: torch.Tensor, y: torch.Tensor, domain: str, cap: float
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    if domain == "depth":
        return x, y
    inv_cap = 1.0 / float(cap)
    xi = (1.0 / x.clamp_min(1e-6)).clamp_min(inv_cap)
    yi = (1.0 / y.clamp_min(1e-6)).clamp_min(inv_cap)
    return xi, yi


def fit(base_5d: torch.Tensor, seeds: Seeds, cfg: FosaConfig,
        carry: Optional[list] = None) -> FosaState:
    """Fit the FoSA field for one clip.  No autograd, no full-res allocation.

    carry: ``state.carry`` of the previous clip of the same sequence. It gives the
    first frame of this clip its R-TGM predecessor, so the temporal term also covers
    clip boundaries. None for the first clip of a sequence.
    """
    if base_5d.dim() != 5 or base_5d.shape[2] != 1:
        raise ValueError(f"base_5d must be [B,T,1,H,W], got {tuple(base_5d.shape)}")
    device = base_5d.device
    B, T, _, H, W = base_5d.shape
    domain = cfg.domain

    # ---- domain + double normalisation ----------------------------------
    x_dom, y_dom = _domain_xy(seeds.x, seeds.y, domain, cfg.max_depth)
    s_init = torch.ones((B,), device=device, dtype=torch.float32)
    y_scale = torch.ones((B,), device=device, dtype=torch.float32)
    one = torch.ones((), device=device, dtype=torch.float32)
    for b in range(B):
        lo, hi = seeds.starts[b * T], seeds.starts[(b + 1) * T]
        if hi > lo:
            # (1) scale normalisation, so that a0 lands near 1
            ratio = y_dom[lo:hi].median() / x_dom[lo:hi].median().clamp_min(1e-12)
            s_init[b] = torch.where(torch.isfinite(ratio) & (ratio > 0), ratio, one)
            # (2) magnitude normalisation, so every design column is O(1)
            ys = torch.sqrt((y_dom[lo:hi] ** 2).mean())
            y_scale[b] = torch.where(torch.isfinite(ys) & (ys > 0), ys, one)

    # ---- bandwidth --------------------------------------------------------
    delta_coarse, cell = compute_delta_coarse(seeds, cfg.bandwidth_cell, cfg.bandwidth_scale)
    delta_min = float(delta_coarse.min()) if delta_coarse.numel() else 0.0

    # ---- R-TGM grid -------------------------------------------------------
    rtgm_k = max(2, int(cfg.rtgm_grid))
    grid_u, grid_v = _build_rtgm_grid(rtgm_k, device)
    Kg = grid_u.numel()
    base_grid_raw = torch.empty((B, T, Kg), device=device, dtype=torch.float32)
    for b in range(B):
        for t in range(T):
            base_grid_raw[b, t] = _sample_base_at_grid(base_5d[b, t, 0], grid_u, grid_v)
    # Same domain transform as the seeds. The clip normalisation (s_init, y_scale) is
    # applied inside rtgm_terms, so that a record carried over from the previous clip,
    # which had its own s_init and y_scale, lands in this clip's units.
    if domain == "inv":
        inv_cap = 1.0 / float(cfg.max_depth)
        base_grid_dom = (1.0 / base_grid_raw.clamp_min(1e-6)).clamp_min(inv_cap)
    else:
        base_grid_dom = base_grid_raw
    # The grid is evaluated with the bandwidth envelope lambda(Delta(p_k)), so that
    # A_t(p_k)^T theta_t is the aligned value the fit outputs at p_k (Eq. 7).
    delta_grid = torch.zeros((B, T, Kg), device=device, dtype=torch.float32)
    for b in range(B):
        for t in range(T):
            delta_grid[b, t] = delta_at_seeds(delta_coarse[b, t], grid_u, grid_v, H, W, cell)
    rtgm_lambda = compute_flicker_lambda(base_5d, float(cfg.rtgm_alpha))
    ctx = types.SimpleNamespace(
        grid_u=grid_u, grid_v=grid_v, x=base_grid_dom, delta=delta_grid,
        s=[float(v) for v in s_init.tolist()], y=[float(v) for v in y_scale.tolist()],
        lam=rtgm_lambda, weight=float(cfg.rtgm_weight), carry=carry, phi_cache={})

    # ---- frequency basis --------------------------------------------------
    order_max = max(0, int(cfg.spatial_order))
    n_int, n2 = build_frequencies(order_max, device=device)
    basis_max = _Basis(n_int, n2)
    counts_by_order = order_counts(n2, order_max)

    # ---- per-frame seed records + holdout split ---------------------------
    ho_frac = float(cfg.holdout_frac)
    frames: List[List[_Frame]] = []
    n_small = 0
    for b in range(B):
        rowf: List[_Frame] = []
        for t in range(T):
            k = b * T + t
            lo, hi = seeds.starts[k], seeds.starts[k + 1]
            u = seeds.u[lo:hi]
            v = seeds.v[lo:hi]
            xn = s_init[b] * x_dom[lo:hi] / y_scale[b]
            yn = y_dom[lo:hi] / y_scale[b]
            d = delta_at_seeds(delta_coarse[b, t], u, v, H, W, cell)
            ho = _holdout_split(seeds.row[lo:hi], seeds.col[lo:hi], k, W, ho_frac)
            keep = ~ho
            rowf.append(_Frame(
                b=b, t=t,
                u=u[keep], v=v[keep], xn=xn[keep], yn=yn[keep], d=d[keep],
                hu=u[ho], hv=v[ho], hxn=xn[ho], hyn=yn[ho], hd=d[ho],
            ))
            if (hi - lo) < int(cfg.min_valid):
                n_small += 1
        frames.append(rowf)
    if n_small:
        print(f"[FOSA][warn] {n_small} frame(s) have fewer than "
              f"{int(cfg.min_valid)} valid seeds")

    # ---- order selection on the held-out anchors (Sec. IV-E) ---------------
    order_sel, ridge_sel, plain_ls_selected, stats_all, ns_all, used_lstsq = select_order(
        frames, basis_max, order_max, counts_by_order, cfg, ctx, device)

    # ---- final fit on every seed at the selected order --------------------
    M_sel = counts_by_order[order_sel] if order_sel <= order_max else basis_max.M
    basis = basis_max.truncate(M_sel)

    if plain_ls_selected:
        # Refit as the plain affine fit on all anchors (ridge 0, weights all 1), then
        # run the R-TGM recursion on that system. Here the correction *is* the global
        # affine, so R-TGM constrains it in full (free_affine=False).
        basis = basis_max.truncate(0)
        stats = [[None] * T for _ in range(B)]
        ns = [[frames[b][t].n_fit + frames[b][t].n_ho for t in range(T)] for b in range(B)]
        chunk = int(cfg.fit_chunk)
        for b in range(B):
            for t in range(T):
                fr = frames[b][t]
                u_all = torch.cat([fr.u, fr.hu])
                v_all = torch.cat([fr.v, fr.hv])
                xn_all = torch.cat([fr.xn, fr.hxn])
                yn_all = torch.cat([fr.yn, fr.hyn])
                d_all = torch.cat([fr.d, fr.hd])
                w_all = torch.ones_like(yn_all)
                stats[b][t] = _accumulate_arrays(basis, u_all, v_all, xn_all, yn_all,
                                                 d_all, w_all, chunk)
        thetas, ls = _solve_chain(stats, basis, 0.0, ns, ctx, device, free_affine=False)
    elif stats_all is not None:
        # The final fit at the selected order is not a separate refit: stats_all is already
        # the all-seed system at the maximum order, and because the frequencies
        # are n^2-sorted, its leading sub-block *is* the all-seed system at the
        # selected order.  So the "refit" is an exact slice-and-solve, and the
        # only thing carried over from the maximum order is the IRLS weighting.
        sub = subset_indices(basis_max.P, M_sel, device)
        thetas, ls = _solve_chain(stats_all, basis, ridge_sel, ns_all, ctx, device, sub=sub)
    else:
        # spatial_order 0: one global affine per frame, IRLS on every anchor
        frames_full: List[List[_Frame]] = []
        for b in range(B):
            rowf = []
            for t in range(T):
                fr = frames[b][t]
                rowf.append(_Frame(
                    b=b, t=t,
                    u=torch.cat([fr.u, fr.hu]), v=torch.cat([fr.v, fr.hv]),
                    xn=torch.cat([fr.xn, fr.hxn]), yn=torch.cat([fr.yn, fr.hyn]),
                    d=torch.cat([fr.d, fr.hd]),
                    hu=fr.hu[:0], hv=fr.hv[:0], hxn=fr.hxn[:0],
                    hyn=fr.hyn[:0], hd=fr.hd[:0],
                ))
            frames_full.append(rowf)
        thetas, _, _, ls, _, _ = _irls(frames_full, basis, ridge_sel, cfg, device, ctx=ctx)
        del frames_full
    used_lstsq = used_lstsq or ls

    if basis.Q > 0:
        n_min = min(f.n_fit + f.n_ho for row in frames for f in row)
        if n_min < basis.Q:
            print(f"[FOSA][warn] a frame has N_seeds={n_min} < Q={basis.Q}; "
                  f"the system is underdetermined and leans on the ridge")

    theta = torch.zeros((B, T, basis.Q), device=device, dtype=torch.float32)
    for b in range(B):
        for t in range(T):
            theta[b, t] = thetas[b][t].to(torch.float32)

    # ---- monotonicity projection ------------------------------------------
    theta, _ = project_monotonic(theta, basis.n2, delta_min, float(cfg.mono_rho))

    n_nonfinite = int((~torch.isfinite(theta)).sum().item())
    if n_nonfinite:
        # reset those frames to the clip-level scale, depth = s_init * x
        bad = ~torch.isfinite(theta).all(dim=-1)                 # [B, T]
        print(f"[FOSA][warn] {n_nonfinite} non-finite coefficient(s) in "
              f"{int(bad.sum())} frame(s); resetting them to depth = s_init * x")
        theta[bad] = 0.0
        theta[..., 0] = torch.where(bad, torch.ones_like(theta[..., 0]), theta[..., 0])

    carry_out = [field_record(theta[b, T - 1], basis, ctx.grid_u, ctx.grid_v,
                              ctx.delta[b, T - 1], ctx.s[b], ctx.y[b], ctx.x[b, T - 1],
                              phi=_grid_phi(ctx, basis, b, T - 1))
                 for b in range(B)]

    if used_lstsq:
        print("[FOSA][warn] torch.linalg.solve failed at least once; used the lstsq fallback")

    return FosaState(
        theta=theta,
        omega_n=basis.n_int,
        n2=basis.n2,
        delta_coarse=delta_coarse,
        s_init=s_init,
        y_scale=y_scale,
        domain=domain,
        cell=cell,
        order_sel=int(order_sel),
        n_seeds=int(seeds.n_total),
        rtgm_lambda=rtgm_lambda,
        carry=carry_out,
    )


# ===========================================================================
# predict
# ===========================================================================
class _PredictTables:
    """1D coordinate and trig tables, and the n^2 grouping of the frequencies."""

    def __init__(self, state: FosaState, H: int, W: int, device):
        self.M = int(state.omega_n.shape[0])
        n_cpu = state.omega_n.detach().cpu()
        n2_cpu = state.n2.detach().cpu()
        self.freq = [(int(n_cpu[m, 0]), int(n_cpu[m, 1]), int(n2_cpu[m]))
                     for m in range(self.M)]
        self.u = torch.arange(H, device=device, dtype=torch.float32) / float(max(H - 1, 1))
        self.v = torch.arange(W, device=device, dtype=torch.float32) / float(max(W - 1, 1))
        self.U = int(n_cpu[:, 0].max().item()) if self.M else 0
        self.Vmax = int(n_cpu[:, 1].abs().max().item()) if self.M else 0
        nu_tab = torch.arange(0, self.U + 1, device=device, dtype=torch.float32).unsqueeze(1)
        nv_tab = torch.arange(-self.Vmax, self.Vmax + 1, device=device,
                              dtype=torch.float32).unsqueeze(1)
        cu = torch.cos(TWO_PI * nu_tab * self.u)               # [U+1, H]
        su = torch.sin(TWO_PI * nu_tab * self.u)
        cv = torch.cos(TWO_PI * nv_tab * self.v)               # [NV, W], NV = 2*Vmax+1
        sv = torch.sin(TWO_PI * nv_tab * self.v)
        # [cu ; su] stacked, so a row-chunk slice of both is one index_select
        self.cusu = torch.cat([cu, su], dim=0)                 # [2*(U+1), H]
        self.cvsv = torch.cat([cv, sv], dim=0)                 # [2*NV, W]
        self.NV = 2 * self.Vmax + 1

        # Frequencies grouped by n^2 (every member of a group shares lambda). Group g
        # owns rows [off, off + 2k) of the stacked column operators: k rows pair with
        # cos(2 pi nu y), k rows with sin(2 pi nu y), one per distinct nu in the group.
        groups: Dict[int, List[int]] = {}
        for m, (_, _, d2) in enumerate(self.freq):
            groups.setdefault(d2, []).append(m)
        self.groups = []                                       # (d2, off, k, rows, members)
        self.n_rows = 0
        for d2, members in sorted(groups.items()):
            nus = sorted({self.freq[m][0] for m in members})
            pos = {nu: i for i, nu in enumerate(nus)}
            k = len(nus)
            rows = torch.tensor(nus + [nu + self.U + 1 for nu in nus], device=device,
                                dtype=torch.long)
            mem = [(m, pos[self.freq[m][0]], self.freq[m][1] + self.Vmax) for m in members]
            self.groups.append((d2, self.n_rows, k, rows, mem))
            self.n_rows += 2 * k


def _group_projections(tab: _PredictTables, th: torch.Tensor, P: int, device):
    """Fold the coefficients into per-group column operators, once per frame.

    For a group G of frequencies sharing n^2 (hence sharing lambda),

        sum_{m in G} alpha_m cos(w_m.p) + beta_m sin(w_m.p)
          = sum_nu cu_nu(y) * [sum_nv (alpha cv_nv + beta sv_nv)](x)
          + sum_nu su_nu(y) * [sum_nv (beta cv_nv - alpha sv_nv)](x)

    The bracketed terms depend on the column only. Their coefficients are laid out on
    the host and every group's operators come out of one [rows, 2*NV] x [2*NV, W] GEMM
    per field, so the row chunks reduce to one GEMM per field per group.
    """
    R, NV = tab.n_rows, tab.NV
    coef = torch.zeros((2, R, 2 * NV), dtype=torch.float32)    # [field a | field b]
    c = coef.numpy()
    t = th.numpy()
    for _, off, k, _, mem in tab.groups:
        for m, i, j in mem:
            al, be = t[1 + 2 * m], t[2 + 2 * m]
            ap, bp = t[P + 1 + 2 * m], t[P + 2 + 2 * m]
            c[0, off + i, j], c[0, off + i, NV + j] = al, be
            c[0, off + k + i, j], c[0, off + k + i, NV + j] = be, -al
            c[1, off + i, j], c[1, off + i, NV + j] = ap, bp
            c[1, off + k + i, j], c[1, off + k + i, NV + j] = bp, -ap
    PaPb = coef.to(device) @ tab.cvsv                         # [2, R, W]
    return [(d2, rows, PaPb[0, off:off + 2 * k], PaPb[1, off:off + 2 * k])
            for d2, off, k, rows, _ in tab.groups]


@torch.no_grad()
def predict(base_5d: torch.Tensor, state: FosaState, cfg: FosaConfig) -> torch.Tensor:
    """Evaluate the field over every pixel, one row chunk at a time.

    Live buffers per chunk are Delta, the lambda pages and the two
    accumulators — all [C, W].  Nothing scales with H*W*M.  Frequencies of
    equal n^2 share lambda and are folded into one GEMM per group.
    """
    device = base_5d.device
    B, T, _, H, W = base_5d.shape
    out = torch.empty_like(base_5d)
    M = int(state.omega_n.shape[0])
    P = 1 + 2 * M
    cap = float(cfg.max_depth)
    inv_cap = 1.0 / cap
    C = max(1, int(cfg.predict_chunk))

    # theta is pulled to the host once (one sync) so the inner loop can use
    # Python scalars with in-place add_/addcmul_ instead of tensor temporaries.
    theta_cpu = state.theta.detach().cpu()
    s_init_cpu = state.s_init.detach().cpu()
    y_scale_cpu = state.y_scale.detach().cpu()
    tab = _PredictTables(state, H, W, device)

    for b in range(B):
        s_init = float(s_init_cpu[b])
        y_scale = float(y_scale_cpu[b])
        for t in range(T):
            th = theta_cpu[b, t]
            a0 = float(th[0])
            b0 = float(th[P])
            dframe = state.delta_coarse[b, t]
            ops = _group_projections(tab, th, P, device) if M else None

            for y0 in range(0, H, C):
                y1 = min(H, y0 + C)
                acc_a = torch.full((y1 - y0, W), a0, device=device, dtype=torch.float32)
                acc_b = torch.full((y1 - y0, W), b0, device=device, dtype=torch.float32)
                if M:
                    dc = delta_at_rows(dframe, tab.u[y0:y1], tab.v, H, W, state.cell)
                    gexp = lambda_exponent(dc)                     # [C, W]
                    del dc
                    for d2, rows, Pa, Pb in ops:
                        lam = torch.exp(gexp * float(d2))
                        blk = tab.cusu.index_select(0, rows)[:, y0:y1].t().contiguous()
                        acc_a.addcmul_(lam, blk @ Pa)
                        acc_b.addcmul_(lam, blk @ Pb)
                        del lam, blk
                    del gexp

                xb = base_5d[b, t, 0, y0:y1]
                if state.domain == "inv":
                    xd = (1.0 / xb.clamp_min(1e-6)).clamp_min(inv_cap)
                else:
                    xd = xb
                res = acc_b.mul_(y_scale)
                res.addcmul_(acc_a, xd, value=s_init)
                if state.domain == "inv":
                    out[b, t, 0, y0:y1] = 1.0 / res.clamp_min(inv_cap)
                else:
                    out[b, t, 0, y0:y1] = res.clamp_(0.0, cap)
                del acc_a, acc_b, res
            del ops
    return out
