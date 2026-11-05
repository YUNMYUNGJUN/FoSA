"""
test_py — unit tests for FoSA (utils/align.py and the modules it uses) on synthetic clips.

    python -m pytest tests -v
    python tests/test_py           # same tests, no pytest needed

Runs on CUDA when available, otherwise on CPU (the allocation test then uses
the process RSS delta instead of torch.cuda.max_memory_allocated).
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.align import FosaConfig, collect_seeds, fit, predict             # noqa: E402
from utils.bandwidth import (compute_delta_coarse, delta_at_rows,           # noqa: E402
                             delta_at_seeds, lambda_matrix, lambda_max)
from utils.basis import (_Basis, build_frequencies, build_phi,             # noqa: E402
                         order_counts, subset_indices)
from utils.solver import project_monotonic                                  # noqa: E402
from utils.temporal import _build_rtgm_grid, compute_flicker_lambda, rtgm_terms  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_cfg(**over):
    kw = dict(spatial_order=2, irls_iters=3, predict_chunk=64)
    kw.update(over)
    return FosaConfig(**kw)


def make_clip(H=96, W=128, T=1, n_seeds=3000, seed=0, a_fn=None, b_fn=None,
              base_kind='ramp', noise=0.0, device=DEVICE):
    """Synthetic clip: returns (base_5d, sparse_5d, gt_5d).

    gt = a(p) * base + b(p); sparse is gt sampled at random pixels.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    uu = (torch.arange(H, dtype=torch.float32) / max(H - 1, 1)).view(H, 1).expand(H, W)
    vv = (torch.arange(W, dtype=torch.float32) / max(W - 1, 1)).view(1, W).expand(H, W)

    base = torch.empty(1, T, 1, H, W)
    gt = torch.empty(1, T, 1, H, W)
    for t in range(T):
        if base_kind == 'ramp':
            b_img = 3.0 + 12.0 * uu + 4.0 * vv + 0.5 * torch.sin(6.0 * vv + t)
        else:
            b_img = 5.0 + 10.0 * torch.rand((H, W), generator=g)
        a_img = torch.ones_like(b_img) if a_fn is None else a_fn(uu, vv)
        s_img = torch.zeros_like(b_img) if b_fn is None else b_fn(uu, vv)
        base[0, t, 0] = b_img
        gt[0, t, 0] = a_img * b_img + s_img
    if noise > 0:
        gt = gt + noise * torch.randn(gt.shape, generator=g)
    gt = gt.clamp_min(0.1)

    sparse = torch.zeros_like(gt)
    for t in range(T):
        idx = torch.randperm(H * W, generator=g)[:n_seeds]
        flat = sparse[0, t, 0].reshape(-1)
        flat[idx] = gt[0, t, 0].reshape(-1)[idx]
    return base.to(device), sparse.to(device), gt.to(device)


def _rel(a, b):
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)


def _ls_predict(base_5d, seeds):
    """Per-frame closed-form affine, i.e. the LS baseline of main_py."""
    out = torch.empty_like(base_5d)
    for b in range(seeds.B):
        for t in range(seeds.T):
            k = b * seeds.T + t
            lo, hi = seeds.starts[k], seeds.starts[k + 1]
            x = seeds.x[lo:hi].double()
            y = seeds.y[lo:hi].double()
            n = x.numel()
            Sx, Sy = x.sum(), y.sum()
            Sxx, Sxy = (x * x).sum(), (x * y).sum()
            det = n * Sxx - Sx * Sx
            s = (n * Sxy - Sx * Sy) / det
            sh = (Sxx * Sy - Sx * Sxy) / det
            out[b, t] = base_5d[b, t] * s.float() + sh.float()
    return out


# ---------------------------------------------------------------------------
# 1
# ---------------------------------------------------------------------------
def test_frequency_order_is_sorted_by_n2():
    expected = {0: 0, 1: 2, 2: 6, 3: 14, 4: 24, 6: 56, 8: 98}
    for U, M in expected.items():
        n_int, n2 = build_frequencies(U, device=DEVICE)
        assert n_int.shape[0] == M, f"U={U}: M={n_int.shape[0]}, expected {M}"
        assert n2.shape[0] == M
        # non-decreasing n^2
        if M > 1:
            assert bool((n2[1:] >= n2[:-1]).all()), f"U={U}: n2 not sorted"
        # half plane: no +/-w duplicate, and n^2 matches the stored pair
        pairs = {(int(a), int(b)) for a, b in n_int.cpu()}
        assert len(pairs) == M, f"U={U}: duplicate frequencies"
        for (a, b) in pairs:
            assert (a, b) != (0, 0)
            assert (-a, -b) not in pairs, f"U={U}: ({a},{b}) and its negation both present"
            assert a > 0 or (a == 0 and b > 0), f"U={U}: ({a},{b}) is outside the half plane"
            assert a * a + b * b <= U * U
        for m, (a, b) in enumerate(n_int.cpu().tolist()):
            assert int(n2[m]) == a * a + b * b


# ---------------------------------------------------------------------------
# 2
# ---------------------------------------------------------------------------
def test_order_subset_is_prefix():
    U_max = 8
    n_int, n2 = build_frequencies(U_max, device=DEVICE)
    counts = order_counts(n2, U_max)
    P_max = 1 + 2 * n_int.shape[0]
    prev = None
    for U in range(U_max + 1):
        M = counts[U]
        # the frequency list of order U is literally the leading M entries
        n_u, _ = build_frequencies(U, device=DEVICE)
        assert n_u.shape[0] == M
        assert bool((n_u == n_int[:M]).all()), f"U={U}: not a prefix of the U={U_max} list"
        idx = set(subset_indices(P_max, M, DEVICE).cpu().tolist())
        if prev is not None:
            assert prev.issubset(idx), f"U={U-1} indices are not a subset of U={U}"
        prev = idx
    # U=2 and U=3 select nested index sets
    i2 = set(subset_indices(P_max, counts[2], DEVICE).cpu().tolist())
    i3 = set(subset_indices(P_max, counts[3], DEVICE).cpu().tolist())
    assert i2.issubset(i3)


# ---------------------------------------------------------------------------
# 3
# ---------------------------------------------------------------------------
def test_lambda_via_integer_power_matches_exp():
    n_int, n2 = build_frequencies(8, device=DEVICE)
    delta = torch.linspace(0.0, 0.25, 501, device=DEVICE, dtype=torch.float32)
    lam = lambda_matrix(delta, n2)
    omega_norm2 = (4.0 * math.pi ** 2) * n2.to(torch.float32)
    ref = torch.exp(-0.5 * delta.unsqueeze(1) ** 2 * omega_norm2.unsqueeze(0))
    err = (lam - ref).abs().max().item()
    assert err < 1e-6, f"max |E^n2 - exp(-0.5 d^2 |w|^2)| = {err:.3e}"

    # and the delta_min bound used by the monotonicity projection
    lmax = lambda_max(0.02, n2, dtype=torch.float32)
    ref_max = torch.exp(-0.5 * (0.02 ** 2) * omega_norm2)
    assert (lmax - ref_max).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 4
# ---------------------------------------------------------------------------
def test_fit_predict_use_same_delta():
    H, W = 240, 320
    base, sparse, _ = make_clip(H=H, W=W, T=2, n_seeds=4000, seed=3)
    seeds = collect_seeds(sparse, base)
    state = fit(base, seeds, make_cfg(spatial_order=3, bandwidth_cell=16))

    for t in range(seeds.T):
        lo, hi = seeds.starts[t], seeds.starts[t + 1]
        u = seeds.u[lo:hi]
        v = seeds.v[lo:hi]
        d_fit = delta_at_seeds(state.delta_coarse[0, t], u, v, H, W, state.cell)

        # what predict sees at the very same pixels: full rows, then gathered
        rows = seeds.row[lo:hi]
        cols = seeds.col[lo:hi]
        u_all = torch.arange(H, device=base.device, dtype=torch.float32) / (H - 1)
        v_all = torch.arange(W, device=base.device, dtype=torch.float32) / (W - 1)
        d_pred = torch.empty_like(d_fit)
        C = 64
        for y0 in range(0, H, C):
            y1 = min(H, y0 + C)
            block = delta_at_rows(state.delta_coarse[0, t], u_all[y0:y1], v_all,
                                       H, W, state.cell)
            sel = (rows >= y0) & (rows < y1)
            if bool(sel.any()):
                d_pred[sel] = block[rows[sel] - y0, cols[sel]]

        assert torch.allclose(d_fit, d_pred, atol=1e-6, rtol=1e-5), \
            f"t={t}: max delta mismatch {(d_fit - d_pred).abs().max().item():.3e}"

        _, n2 = build_frequencies(3, device=DEVICE)
        lam_fit = lambda_matrix(d_fit, n2)
        lam_pred = lambda_matrix(d_pred, n2)
        assert (lam_fit - lam_pred).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 5
# ---------------------------------------------------------------------------
def test_perfect_base_roundtrip():
    for domain in ('depth', 'inv'):
        base, sparse, gt = make_clip(H=120, W=160, T=2, n_seeds=4000, seed=7)
        gt = base.clone()                       # base == gt
        sparse = torch.where(sparse > 0, base, torch.zeros_like(base))
        cfg = make_cfg(domain=domain)
        seeds = collect_seeds(sparse, base)
        pred = predict(base, fit(base, seeds, cfg), cfg)
        assert _rel(pred, gt) < 1e-5, \
            f"domain={domain}: relative error {_rel(pred, gt):.3e}"


# ---------------------------------------------------------------------------
# 6
# ---------------------------------------------------------------------------
def test_monotonicity_guarantee():
    H, W = 120, 160
    U = 3
    n_int, n2 = build_frequencies(U, device=DEVICE)
    M = n_int.shape[0]
    P = 1 + 2 * M
    rho = 0.8
    delta_min = 0.01

    torch.manual_seed(0)
    theta = torch.zeros(1, 2, 2 * P, device=DEVICE)
    theta[:, :, 0] = 1.3                                     # a0
    theta[:, :, 1:P] = 50.0 * torch.randn(1, 2, P - 1, device=DEVICE)   # absurd amplitudes
    theta[:, :, P] = -0.5

    out, hits = project_monotonic(theta, n2, delta_min, rho)
    assert hits > 0, "extreme coefficients should have triggered the projection"

    # evaluate a(p) densely and check the guaranteed lower bound
    u = torch.arange(H, device=DEVICE, dtype=torch.float32) / (H - 1)
    v = torch.arange(W, device=DEVICE, dtype=torch.float32) / (W - 1)
    uu = u.view(H, 1).expand(H, W).reshape(-1)
    vv = v.view(1, W).expand(H, W).reshape(-1)
    delta = torch.full_like(uu, delta_min)                   # the worst case: least damping
    basis = _Basis(n_int, n2)
    for bt in range(2):
        phi = build_phi(basis, uu, vv, delta)
        a = phi @ out[0, bt, :P]
        a0 = float(out[0, bt, 0])
        assert float(a.min()) > 0.0, f"a(p) went non-positive: min={float(a.min()):.4f}"
        assert float(a.min()) >= a0 * (1.0 - rho) - 1e-4, \
            f"a_min={float(a.min()):.4f} < a0*(1-rho)={a0 * (1 - rho):.4f}"


# ---------------------------------------------------------------------------
# 7
# ---------------------------------------------------------------------------
def test_spatially_varying_recovery():
    a_fn = lambda uu, vv: 1.3 + 0.25 * torch.cos(TWO_PI * vv)      # x -> column
    b_fn = lambda uu, vv: -0.5 + 0.3 * torch.sin(TWO_PI * uu)      # y -> row
    base, sparse, gt = make_clip(H=180, W=240, T=1, n_seeds=6000, seed=11,
                                 a_fn=a_fn, b_fn=b_fn)
    seeds = collect_seeds(sparse, base)
    mask = gt > 0
    cfg = make_cfg(spatial_order=3, ridge=1e-3)
    state = fit(base, seeds, cfg)
    pred = predict(base, state, cfg)
    mae = float((pred[mask] - gt[mask]).abs().mean())
    mae_ls = float((_ls_predict(base, seeds)[mask] - gt[mask]).abs().mean())
    print(f"    spatially-varying MAE: FoSA {mae:.4f} (order {state.order_sel}) vs affine {mae_ls:.4f}")
    assert state.order_sel >= 1
    assert mae * 3.0 < mae_ls, f"FoSA MAE {mae:.4f} is not 3x better than affine {mae_ls:.4f}"


# ---------------------------------------------------------------------------
# 8
# ---------------------------------------------------------------------------
def test_auto_order_picks_zero_on_noise():
    # base and gt differ only by a global affine plus unstructured noise:
    # there is nothing for a spatial field to learn.
    base, sparse, gt = make_clip(H=180, W=240, T=2, n_seeds=5000, seed=23,
                                 base_kind='random', noise=0.6)
    state = fit(base, collect_seeds(sparse, base), make_cfg(spatial_order=6))
    print(f"    noise-only clip -> order {state.order_sel}")
    assert state.order_sel == 0, f"auto-order picked U'={state.order_sel} on pure noise"


# ---------------------------------------------------------------------------
# 9
# ---------------------------------------------------------------------------
def _peak_fit_predict(H, W, T, n_seeds, cfg):
    """(fit peak bytes, predict peak bytes) for one synthetic clip."""
    base, sparse, _ = make_clip(H=H, W=W, T=T, n_seeds=n_seeds, seed=31)
    seeds = collect_seeds(sparse, base)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        b0 = torch.cuda.memory_allocated()
        state = fit(base, seeds, cfg)
        torch.cuda.synchronize()
        fit_peak = torch.cuda.max_memory_allocated() - b0
        torch.cuda.reset_peak_memory_stats()
        b1 = torch.cuda.memory_allocated()
        pred = predict(base, state, cfg)
        torch.cuda.synchronize()
        pred_peak = torch.cuda.max_memory_allocated() - b1
    else:
        import resource
        rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        r0 = rss()
        state = fit(base, seeds, cfg)
        r1 = rss()
        pred = predict(base, state, cfg)
        r2 = rss()
        fit_peak, pred_peak = max(r1 - r0, 0), max(r2 - r1, 0)
    assert torch.isfinite(pred).all()
    del base, sparse, seeds, state, pred
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return fit_peak, pred_peak


def test_no_large_allocation():
    """No allocation may scale with H*W*M; the whole clip must fit a 100 MB budget.

    The [fit_chunk, Q] fp32 design chunk (11 MB) already exceeds one
    1216x1936 image (9.4 MB), so "no tensor above O(H*W)" is not the property
    tested.  What is protected is that nothing scales with resolution x order:
    fit memory is invariant to H,W and stays inside the chunk bound, predict
    adds only its output plus [C,W] pages, and the pair stays under 100 MB.
    """
    cfg = make_cfg(spatial_order=6)
    H, W, T = 1216, 1936, 3
    hw4 = H * W * 4

    fit_small, pred_small = _peak_fit_predict(H // 2, W // 2, T, 19000, cfg)
    fit_big, pred_big = _peak_fit_predict(H, W, T, 19000, cfg)

    print(f"    fit  peak: {fit_small/1e6:.1f} MB @ {H//2}x{W//2} -> "
          f"{fit_big/1e6:.1f} MB @ {H}x{W}")
    print(f"    pred peak: {pred_small/1e6:.1f} MB @ {H//2}x{W//2} -> "
          f"{pred_big/1e6:.1f} MB @ {H}x{W}")

    # 1. fit is resolution independent: it never touches a full-res tensor
    assert fit_big < fit_small * 1.10 + 1e6, \
        f"fit memory grew with resolution: {fit_small/1e6:.1f} -> {fit_big/1e6:.1f} MB"

    # 2. fit stays inside the design-chunk bound (A plus the Phi/cos/sin/lambda
    #    scratch it is built from, ~4 chunk-matrices worth)
    n_int, _ = build_frequencies(6, device=DEVICE)
    Q = 2 * (1 + 2 * n_int.shape[0])
    chunk_bytes = cfg.fit_chunk * Q * 4
    assert fit_big < 4 * chunk_bytes, \
        f"fit peak {fit_big/1e6:.1f} MB exceeds 4 design chunks ({4*chunk_bytes/1e6:.1f} MB)"

    # 3. predict adds its [B,T,1,H,W] output plus chunk pages only
    out_bytes = T * hw4
    assert pred_big < out_bytes + 0.5 * hw4, \
        f"predict peak {pred_big/1e6:.1f} MB exceeds output + half an image"

    # 4. the 100 MB budget
    total = max(fit_big, pred_big)
    assert total < 100e6, f"FOSA peak {total/1e6:.1f} MB exceeds the 100 MB budget"


# ---------------------------------------------------------------------------
# the grouped-GEMM predict must equal evaluating every frequency per pixel
# ---------------------------------------------------------------------------
def test_predict_matches_dense_evaluation():
    a_fn = lambda uu, vv: 1.2 + 0.2 * torch.cos(TWO_PI * vv) + 0.1 * torch.sin(4 * TWO_PI * uu)
    b_fn = lambda uu, vv: -0.4 + 0.3 * torch.sin(TWO_PI * uu)
    H, W = 90, 120
    base, sparse, _ = make_clip(H=H, W=W, T=2, n_seeds=3000, seed=17,
                                a_fn=a_fn, b_fn=b_fn, noise=0.02)
    seeds = collect_seeds(sparse, base)
    u = (torch.arange(H, device=DEVICE, dtype=torch.float32) / (H - 1)).view(H, 1).expand(H, W).reshape(-1)
    v = (torch.arange(W, device=DEVICE, dtype=torch.float32) / (W - 1)).view(1, W).expand(H, W).reshape(-1)
    for U in (1, 3, 6):
        state = fit(base, seeds, make_cfg(spatial_order=U))
        basis = _Basis(state.omega_n, state.n2)
        for chunk in (16, 64, 251):          # 251 exercises a ragged last chunk
            pred = predict(base, state, make_cfg(spatial_order=U, predict_chunk=chunk))
            for t in range(2):
                d = delta_at_rows(state.delta_coarse[0, t], u[::W], v[:W], H, W,
                                       state.cell).reshape(-1)
                phi = build_phi(basis, u, v, d)
                th = state.theta[0, t]
                a, b = phi @ th[:basis.P], phi @ th[basis.P:]
                ref = (a * float(state.s_init[0]) * base[0, t, 0].reshape(-1)
                       + b * float(state.y_scale[0])).clamp(0.0, 200.0).view(H, W)
                rel = _rel(pred[0, t, 0], ref)
                assert rel < 1e-5, f"U={U} chunk={chunk} t={t}: grouped vs dense {rel:.3e}"


# ---------------------------------------------------------------------------
# global affine safety net
# ---------------------------------------------------------------------------
def test_plain_affine_candidate_wins_without_structure():
    """With no spatial structure and a very tight Huber threshold, the plain affine
    candidate (no IRLS) must win the order selection, and frame 0 (no R-TGM
    predecessor) must then be the closed-form affine fit."""
    base, sparse, _ = make_clip(H=160, W=200, T=2, n_seeds=4000, seed=71,
                                base_kind='random', noise=0.4)
    seeds = collect_seeds(sparse, base)
    cfg = make_cfg(spatial_order=3, huber_delta=0.02)
    state = fit(base, seeds, cfg)
    rel = _rel(predict(base, state, cfg)[:, :1], _ls_predict(base, seeds)[:, :1])
    print(f"    plain affine candidate: order_sel={state.order_sel} rel_vs_affine={rel:.3e}")
    assert state.order_sel == 0
    assert rel < 1e-4, f"output differs from the per-frame affine fit by {rel:.3e}"


def test_frame_without_anchors_keeps_clip_scale():
    """A frame with no anchors (and no temporal pull) falls back to depth = s_init * x."""
    base, sparse, _ = _rtgm_clip(T=3)
    sparse[:, 1] = 0.0                                   # frame 1 has no anchors
    seeds = collect_seeds(sparse, base)
    cfg = make_cfg(spatial_order=3, rtgm_weight=0.0)
    st = fit(base, seeds, cfg)
    pred = predict(base, st, cfg)
    rel = _rel(pred[:, 1], base[:, 1] * float(st.s_init[0]))
    print(f"    anchor-free frame vs s_init*x rel diff = {rel:.2e}")
    assert torch.isfinite(pred).all()
    assert rel < 1e-3, f"anchor-free frame drifted from s_init * x by {rel:.2e}"


# ---------------------------------------------------------------------------
# R-TGM
# ---------------------------------------------------------------------------
def _rtgm_clip(H=112, W=144, T=4, n_seeds=900, seed=21):
    a_fn = lambda uu, vv: 1.2 + 0.2 * torch.cos(TWO_PI * vv)
    b_fn = lambda uu, vv: -0.3 + 0.2 * torch.sin(TWO_PI * uu)
    return make_clip(H=H, W=W, T=T, n_seeds=n_seeds, seed=seed, a_fn=a_fn, b_fn=b_fn, noise=0.05)


def test_rtgm_terms_zero_without_predecessor():
    n_int, n2 = build_frequencies(3, device=DEVICE)
    basis = _Basis(n_int, n2)
    grid_u, grid_v = _build_rtgm_grid(8, DEVICE)
    G, g = rtgm_terms(basis, None, None, 1.0, 1.0, grid_u, grid_v, None, lam=1.0, weight=1.0)
    assert G.shape == (basis.Q, basis.Q) and g.shape == (basis.Q,)
    assert float(G.abs().max()) == 0.0 and float(g.abs().max()) == 0.0


def test_rtgm_leaves_frame0_and_moves_later_frames():
    """Without a carried record frame 0 has no predecessor, so R-TGM must not touch it;
    a strong weight must visibly move the later frames."""
    base, sparse, _ = _rtgm_clip()
    seeds = collect_seeds(sparse, base)
    cfg_off, cfg_on = make_cfg(spatial_order=3, rtgm_weight=0.0), make_cfg(spatial_order=3, rtgm_weight=5.0)
    st_off, st_on = fit(base, seeds, cfg_off), fit(base, seeds, cfg_on)
    assert st_off.order_sel == st_on.order_sel
    p_off, p_on = predict(base, st_off, cfg_off), predict(base, st_on, cfg_on)
    rel0, rel_rest = _rel(p_on[:, :1], p_off[:, :1]), _rel(p_on[:, 1:], p_off[:, 1:])
    print(f"    frame 0 rel diff {rel0:.2e}, frames 1..T-1 rel diff {rel_rest:.2e}")
    assert rel0 < 1e-4, f"frame 0 should be unconstrained, differs by {rel0:.3e}"
    assert rel_rest > 1e-3, "later frames should be clearly affected by a strong R-TGM term"


def test_flicker_lambda_reacts_to_flicker_vs_motion():
    """Auto-lambda check: heavy frame-to-frame noise -> lambda < 0.3;
    a perfectly static (repeated) clip -> lambda > 0.8."""
    H, W, T = 64, 80, 5
    g = torch.Generator(device='cpu').manual_seed(3)
    smooth = (3.0 + 2.0 * torch.rand(1, 1, 1, H, W, generator=g)).to(DEVICE)

    base_flicker = smooth.expand(1, T, 1, H, W).clone()
    base_flicker = base_flicker + 1.0 * torch.randn(1, T, 1, H, W, generator=g, device='cpu').to(DEVICE)
    lam_flicker = compute_flicker_lambda(base_flicker, alpha=5.0)
    print(f"    flicker clip: lambda = {lam_flicker:.4f}")
    assert lam_flicker < 0.3, f"expected lambda < 0.3 for a flickering clip, got {lam_flicker:.4f}"

    base_stable = smooth.expand(1, T, 1, H, W).clone()
    lam_stable = compute_flicker_lambda(base_stable, alpha=5.0)
    print(f"    stable clip:  lambda = {lam_stable:.4f}")
    assert lam_stable > 0.8, f"expected lambda > 0.8 for a static clip, got {lam_stable:.4f}"


def test_rtgm_auto_order_runs_end_to_end():
    base, sparse, _ = _rtgm_clip(n_seeds=1200)
    cfg = make_cfg(spatial_order=4)
    state = fit(base, collect_seeds(sparse, base), cfg)
    pred = predict(base, state, cfg)
    assert torch.isfinite(pred).all()
    assert 0.0 <= state.rtgm_lambda <= 1.0
    assert state.carry is not None and state.carry[0].a.numel() == 64
    print(f"    order_sel={state.order_sel} lambda={state.rtgm_lambda:.3f}")


def test_rtgm_carry_constrains_first_frame():
    """The record carried from the previous clip gives frame 0 a predecessor."""
    base, sparse, _ = _rtgm_clip(T=3)
    seeds = collect_seeds(sparse, base)
    cfg = make_cfg(spatial_order=3, rtgm_weight=5.0)
    prev_clip = fit(base, seeds, cfg)               # stand-in for the previous clip
    p_none = predict(base, fit(base, seeds, cfg, carry=None), cfg)
    p_none2 = predict(base, fit(base, seeds, cfg, carry=None), cfg)
    p_on = predict(base, fit(base, seeds, cfg, carry=prev_clip.carry), cfg)
    assert torch.equal(p_none, p_none2), "fit must be deterministic"
    d0 = _rel(p_on[:, :1], p_none[:, :1])
    print(f"    frame 0 with carry vs without: rel diff = {d0:.2e}")
    assert d0 > 1e-6, "the carried record should constrain frame 0"


def test_rtgm_runs_in_the_affine_fallback():
    """Too few held-out anchors -> per-frame affine fallback, with the R-TGM recursion
    run on it: frame 0 is the plain affine fit, later frames move toward their targets."""
    base, sparse, _ = _rtgm_clip(T=3, n_seeds=40)
    seeds = collect_seeds(sparse, base)
    cfg = make_cfg(spatial_order=3, rtgm_weight=1.0)
    st = fit(base, seeds, cfg)
    p = predict(base, st, cfg)
    ls = _ls_predict(base, seeds)
    assert st.order_sel == 0
    assert _rel(p[:, :1], ls[:, :1]) < 1e-4, "frame 0 has no predecessor"
    d = _rel(p[:, 1:], ls[:, 1:])
    print(f"    frames 1..T-1 with R-TGM in the fallback: rel diff vs affine = {d:.2e}")
    assert d > 1e-5


# ---------------------------------------------------------------------------
# properties stated in the paper
# ---------------------------------------------------------------------------
def test_order_zero_is_global_least_squares():
    """U = 0 leaves one global scale and shift per frame: the closed-form fit of Eq. (1).
    R-TGM acts only on what a global affine cannot explain, so it has nothing to act on."""
    base, sparse, _ = make_clip(H=96, W=128, T=2, n_seeds=3000, seed=5,
                                base_kind='random', noise=0.3)
    seeds = collect_seeds(sparse, base)
    cfg = make_cfg(spatial_order=0, irls_iters=1)        # one round: unweighted least squares
    state = fit(base, seeds, cfg)
    rel = _rel(predict(base, state, cfg), _ls_predict(base, seeds))
    print(f"    U=0 vs closed-form least squares: rel diff = {rel:.2e}")
    assert state.theta.shape[-1] == 2
    assert rel < 1e-5, f"U=0 differs from the global least-squares fit by {rel:.3e}"


def test_frequency_set_size_at_order_six():
    n_int, n2 = build_frequencies(6, device=DEVICE)
    basis = _Basis(n_int, n2)
    assert basis.M == 56 and basis.Q == 226, f"|F_6| = {basis.M}, Q = {basis.Q}"


def test_delta_clamped_and_lambda_in_unit_interval():
    """Delta in [2 / max(H, W), 1]; lambda in (0, 1]."""
    H, W = 120, 200
    for n_seeds in (H * W, 4000, 5):                     # every pixel, typical, almost none
        base, sparse, _ = make_clip(H=H, W=W, T=2, n_seeds=n_seeds, seed=9)
        seeds = collect_seeds(sparse, base)
        delta, _ = compute_delta_coarse(seeds, cell=32, scale=1.0)
        lo, hi = float(delta.min()), float(delta.max())
        assert lo >= 2.0 / max(H, W) - 1e-7 and hi <= 1.0, \
            f"n_seeds={n_seeds}: Delta in [{lo:.4f}, {hi:.4f}]"
        _, n2 = build_frequencies(6, device=DEVICE)
        d = delta.reshape(-1)
        lam = lambda_matrix(d, n2)
        assert float(lam.min()) >= 0.0 and float(lam.max()) <= 1.0
        # lambda = exp(-2 pi^2 n^2 Delta^2) > 0; it underflows to 0 in fp32 only where the
        # carrier is suppressed anyway (e.g. Delta = 1, n^2 = 36: e^-710)
        expo = 2.0 * math.pi ** 2 * d.unsqueeze(1) ** 2 * n2.to(d.dtype).unsqueeze(0)
        assert bool((lam[expo < 80.0] > 0.0).all())
    # every pixel measured: Delta reaches its lower clamp
    base, sparse, _ = make_clip(H=H, W=W, T=1, n_seeds=H * W, seed=9)
    delta, _ = compute_delta_coarse(collect_seeds(sparse, base), cell=32, scale=1.0)
    assert abs(float(delta.min()) - 2.0 / max(H, W)) < 1e-7
    # far from every anchor: Delta = 1
    sparse = torch.zeros_like(base)
    sparse[0, 0, 0, :8, :8] = base[0, 0, 0, :8, :8]
    delta, _ = compute_delta_coarse(collect_seeds(sparse, base), cell=16, scale=1.0)
    assert float(delta[0, 0, -1, -1]) == 1.0


def test_scale_field_positive_after_fit():
    """Monotonic projection at the end of fit(): a_t(p) > 0 at every pixel, even when
    the anchors ask for a scale that changes sign."""
    a_fn = lambda uu, vv: 0.1 + 1.5 * torch.cos(3 * TWO_PI * vv)       # negative in places
    base, sparse, _ = make_clip(H=96, W=128, T=2, n_seeds=4000, seed=13, a_fn=a_fn)
    seeds = collect_seeds(sparse, base)
    state = fit(base, seeds, make_cfg(spatial_order=4, ridge=1e-4))
    H, W = 96, 128
    u = (torch.arange(H, device=DEVICE, dtype=torch.float32) / (H - 1)).view(H, 1).expand(H, W).reshape(-1)
    v = (torch.arange(W, device=DEVICE, dtype=torch.float32) / (W - 1)).view(1, W).expand(H, W).reshape(-1)
    basis = _Basis(state.omega_n, state.n2)
    for t in range(2):
        d = delta_at_rows(state.delta_coarse[0, t], u[::W], v[:W], H, W, state.cell).reshape(-1)
        a = build_phi(basis, u, v, d) @ state.theta[0, t, :basis.P]
        print(f"    t={t}: order {state.order_sel}, min a(p) = {float(a.min()):.4f}")
        assert float(a.min()) > 0.0


def test_frame_without_anchors_returns_prior():
    """A frame with no anchors and no temporal term returns theta_bar exactly:
    a0 = 1, every other coefficient 0, i.e. depth = s_init * x."""
    base, sparse, _ = _rtgm_clip(T=2)
    sparse[:, 1] = 0.0
    cfg = make_cfg(spatial_order=3, rtgm_weight=0.0)
    st = fit(base, collect_seeds(sparse, base), cfg)
    th = st.theta[0, 1]
    prior = torch.zeros_like(th)
    prior[0] = 1.0
    assert float((th - prior).abs().max()) < 1e-6, f"theta = {th.tolist()[:4]}..."


def test_few_heldout_anchors_fall_back_to_global_fit():
    """Fewer than 32 held-out anchors: no order search, the global least-squares fit."""
    base, sparse, _ = make_clip(H=96, W=128, T=1, n_seeds=100, seed=19,
                                a_fn=lambda uu, vv: 1.2 + 0.3 * torch.cos(TWO_PI * vv))
    seeds = collect_seeds(sparse, base)
    cfg = make_cfg(spatial_order=6)
    state = fit(base, seeds, cfg)
    assert state.order_sel == 0
    assert _rel(predict(base, state, cfg), _ls_predict(base, seeds)) < 1e-5


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:            # noqa: BLE001 - report, keep going
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed on {DEVICE}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
