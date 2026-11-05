"""
main.py — FoSA: align a frozen depth backbone to sparse metric depth, clip by clip.

    # sequences stored as .pt files (format: utils/clips.py, load_sequence)
    python main.py --config config/unidepth.yaml --input input/scene.pt --save-vis
    python main.py --config config/unidepth.yaml --input input/ --output outputs/

    # benchmark evaluation: RMSE / MAE / TEPE (dataset layouts: docs/REPRODUCE.md)
    python main.py --config config/unidepth.yaml --dataset kitti_video --dataset-path /data/KITTI-DC

Frames are aligned in non-overlapping clips of --num-frames (3, as in the paper), and
R-TGM carries over from each clip to the next one of the same sequence. Everything this
script calls lives in utils/: the method (utils/align.py), the backbones
(utils/base_models/), the dataset loaders (utils/dataloader/), the metrics and the
visualisation.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings
from importlib import import_module

import numpy as np
import torch
import tqdm

from utils import align as fosa
from utils import visualize
from utils.base_models import forward_backbone, load_backbone
from utils.clips import (WithCache, center_crop, check_cache, crop_box, load_sequence,
                         sequence_files, thin_anchors)
from utils.config import load_config
from utils.dataloader.seeded import SeededClips
from utils.metrics import mae_eval, rmse_eval, tepe_eval

warnings.filterwarnings("ignore", category=UserWarning)

DATASETS = ('nyu_video', 'void', 'kitti_video', 'dydtof', 'ddad_vid')


def parse_args():
    p = argparse.ArgumentParser(description='FoSA: Fourier scale-and-shift alignment of a '
                                            'frozen depth backbone to sparse metric depth',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', '-c', required=True, help='config/<backbone>.yaml')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', '-i', help='a sequence .pt file, or a folder searched for *.pt')
    src.add_argument('--dataset', choices=DATASETS, help='evaluate on a benchmark instead')
    p.add_argument('--output', '-o', default='outputs',
                   help='--input: <stem>_depth.pt per sequence; --dataset: <time>_<name>/ per run')
    p.add_argument('--num-frames', type=int, default=3, help='frames per clip')
    p.add_argument('--save-vis', action='store_true', help='save a result panel per frame')
    p.add_argument('--vis-every', type=int, default=1, help='with --save-vis: every N-th clip')
    p.add_argument('--vis-width', type=int, default=640, help='panel width of the visualisation')

    g = p.add_argument_group('benchmark (--dataset)')
    g.add_argument('--dataset-path', help='dataset root, laid out as in docs/REPRODUCE.md')
    g.add_argument('--name', default='', help='run name (default <backbone>_<dataset>)')
    g.add_argument('--num-sparse-samples', type=int, default=500,
                   help='NYUv2 / DyDToF: anchors drawn from the dense depth per frame')
    g.add_argument('--num-lidar-lines', type=int, default=64, help='KITTI-DC: LiDAR lines kept')
    g.add_argument('--sparse-keep-ratio', type=float, default=1.0,
                   help='keep each anchor with this probability (seeded per clip)')
    g.add_argument('--random-seed', type=int, default=2025, help='seed of the random anchors')
    g.add_argument('--num-workers', type=int, default=4, help='data / cache loading processes')
    g.add_argument('--base-cache', default='',
                   help='folder of cached backbone outputs (one file per clip); missing clips '
                        'are computed and written. Use one folder per backbone / dataset / crop.')
    g.add_argument('--profile', action='store_true',
                   help='report backbone / FoSA time and FoSA GPU memory')

    g = p.add_argument_group('FoSA overrides (default: the value in the config)')
    for f in dataclasses.fields(fosa.FosaConfig):
        g.add_argument(f"--fosa-{f.name.replace('_', '-')}", dest=f'fosa_{f.name}',
                       type=type(f.default), default=None, help=f.metadata['help'])
    args = p.parse_args()
    if args.dataset and not args.dataset_path:
        p.error('--dataset needs --dataset-path')
    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True          # fixed-size inference only


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def backbone_opts(cfg):
    return {k: cfg[k] for k in ('vda_root', 'vda_ckpt', 'dvd_root', 'dvd_ckpt', 'dvd_max_side')}


def align_clip(base, sparse, fcfg, carry):
    """base / sparse [1,T,1,H,W] -> (aligned depth [1,T,1,H,W], FosaState)."""
    seeds = fosa.collect_seeds(sparse, base)
    state = fosa.fit(base, seeds, fcfg, carry=carry)
    return fosa.predict(base, state, fcfg), state


def check_finite(pred, fcfg, what):
    if not bool(torch.isfinite(pred).all()):
        print(f"[FOSA][warn] {what}: non-finite depth (non-finite backbone output?)")
        pred = torch.nan_to_num(pred, nan=0.0, posinf=fcfg.max_depth, neginf=0.0)
    return pred


def clip_metrics(pred, gt):
    mask = gt > 0
    return torch.stack([rmse_eval(pred, gt, mask), mae_eval(pred, gt, mask),
                        tepe_eval(pred, gt, t_valid=1e-3)]).cpu().numpy()


@torch.no_grad()
def run_inputs(args, cfg, fcfg):
    """Align every sequence file; write <output>/<stem>_depth.pt (and panels)."""
    files = sequence_files(args.input)
    model = load_backbone(cfg['backbone'], **backbone_opts(cfg))
    os.makedirs(args.output, exist_ok=True)
    ps, T = cfg['input_patch_size'], args.num_frames
    all_vals = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        rgb, sparse, gt = load_sequence(path)
        box = crop_box(*rgb.shape[-2:], ps) if ps > 0 else (0, 0, *rgb.shape[-2:])
        if ps > 0:
            rgb, sparse = center_crop(rgb, ps), center_crop(sparse, ps)
            gt = None if gt is None else center_crop(gt, ps)
        N, H, W = rgb.shape[0], rgb.shape[-2], rgb.shape[-1]
        depth = torch.empty((N, H, W), dtype=torch.float32)
        vals, carry = [], None
        for c, t0 in enumerate(range(0, N, T)):
            t1 = min(N, t0 + T)
            r, s = rgb[t0:t1].cuda(), sparse[t0:t1].cuda()
            base = forward_backbone(model, r, s, cfg['amp'], T=t1 - t0)
            base = base.view(1, t1 - t0, 1, H, W)
            pred, state = align_clip(base, s.unsqueeze(0), fcfg, carry)
            carry = state.carry
            pred = check_finite(pred, fcfg, f'{stem} frames {t0}-{t1 - 1}')
            depth[t0:t1] = pred[0, :, 0].cpu()
            g = None if gt is None else gt[t0:t1].cuda().unsqueeze(0)
            if g is not None:
                vals.append(clip_metrics(pred, g))
            if args.save_vis and c % args.vis_every == 0:
                visualize.save_clip(os.path.join(args.output, f'{stem}_vis'), c,
                                    *(v.float().cpu().numpy() for v in (r, s, base[0], pred[0])),
                                    None if g is None else g[0].cpu().numpy(),
                                    max_width=args.vis_width)
        out = os.path.join(args.output, f'{stem}_depth.pt')
        torch.save({'depth': depth, 'crop': torch.tensor(box)}, out)
        msg = f"{stem}: {N} frames -> {out}"
        if vals:
            avg = np.mean(vals, axis=0)
            msg += f"  RMSE {avg[0]:.4f}  MAE {avg[1]:.4f}  TEPE {avg[2]:.4f}"
            all_vals += vals
        print(msg)
    if all_vals and len(files) > 1:
        avg = np.mean(all_vals, axis=0)
        print(f"all ({len(all_vals)} clips): RMSE {avg[0]:.4f}  MAE {avg[1]:.4f}  TEPE {avg[2]:.4f}")


@torch.no_grad()
def run_dataset(args, cfg, fcfg):
    """Benchmark evaluation: per-clip RMSE / MAE / TEPE and their means over clips."""
    bb = cfg['backbone']
    if args.base_cache:
        check_cache(args.base_cache, {
            'backbone': bb, 'dataset': args.dataset, 'dataset_path': args.dataset_path,
            'num_frames': args.num_frames, 'input_patch_size': cfg['input_patch_size'],
            'vda_ckpt': os.path.basename(cfg['vda_ckpt']) if bb == 'VDA' else ''})
    dataset = SeededClips(import_module('utils.dataloader.' + args.dataset).valset(args),
                          args.random_seed)
    loader = torch.utils.data.DataLoader(WithCache(dataset, args.base_cache), batch_size=1,
                                         shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True)
    name = args.name or f'{bb}_{args.dataset}'
    save_dir = os.path.join(args.output, f"{time.strftime('%y%m%d_%H%M%S')}_{name}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(sys.argv[1:], f)
    print(f"{args.dataset}: {len(dataset)} clips of {args.num_frames} frames | backbone "
          f"{bb}, fit domain {fcfg.domain}, cache {args.base_cache or 'off'}")

    outfile = open(os.path.join(save_dir, 'eval_results.txt'), 'w')
    outfile.write("clip, RMSE, MAE, TEPE, order\n")
    model = None                            # loaded on the first cache miss
    totals = np.zeros(3)
    orders = {}
    prev = None                             # (scene, clip index, carry) of the previous clip
    t_backbone = t_fosa = 0.0
    peak_mb = sum_mb = 0.0
    ps = cfg['input_patch_size']

    for idx, sample in enumerate(tqdm.tqdm(loader)):
        rgb = sample['rgb'].cuda(non_blocking=True)        # [1,T,3,H,W]
        gt = sample['gt'].cuda(non_blocking=True)          # [1,T,1,H,W]
        sparse = sample['dep'].cuda(non_blocking=True)     # [1,T,1,H,W]
        sparse = thin_anchors(sparse, args.sparse_keep_ratio, args.random_seed, idx)
        if ps > 0:
            rgb, gt, sparse = (center_crop(v, ps) for v in (rgb, gt, sparse))
        T = rgb.shape[1]

        if args.profile:
            t0 = sync_time()
        if 'base' in sample:
            base = sample['base'][0].cuda(non_blocking=True)
            assert base.shape == sparse[0].shape, f"stale cache entry for clip {idx}"
        else:
            if model is None:
                model = load_backbone(bb, **backbone_opts(cfg))
            base = forward_backbone(model, rgb[0], sparse[0], cfg['amp'], T=T)
            if args.base_cache:
                path = os.path.join(args.base_cache, f'{idx:05d}.pt')
                torch.save(base.cpu(), path + '.tmp')
                os.replace(path + '.tmp', path)
        base = base.view(1, T, 1, *base.shape[-2:])

        if args.profile:
            t1 = sync_time()
            t_backbone += t1 - t0
            mem0 = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        # the last clip of the same sequence hands R-TGM its last frame
        scene = sample['scene_id'][0]
        carry = prev[2] if prev is not None and prev[0] == scene and prev[1] == idx - 1 else None
        pred, state = align_clip(base, sparse, fcfg, carry)
        prev = (scene, idx, state.carry)
        if args.profile:
            t_fosa += sync_time() - t1
            mb = (torch.cuda.max_memory_allocated() - mem0) / 2 ** 20
            peak_mb, sum_mb = max(peak_mb, mb), sum_mb + mb
        pred = check_finite(pred, fcfg, f'clip {idx}')

        vals = clip_metrics(pred, gt)
        totals += vals
        orders[state.order_sel] = orders.get(state.order_sel, 0) + 1
        outfile.write(f"{idx}, {vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}, {state.order_sel}\n")

        if args.save_vis and idx % args.vis_every == 0:
            visualize.save_clip(os.path.join(save_dir, 'vis'), idx,
                                *(v[0].float().cpu().numpy() for v in (rgb, sparse, base, pred, gt)),
                                max_width=args.vis_width)

    n = max(len(dataset), 1)
    avg = totals / n
    outfile.write(f"AVG, {avg[0]:.4f}, {avg[1]:.4f}, {avg[2]:.4f}\n")
    outfile.close()
    print(f"\n{args.dataset} / {bb} / FoSA ({n} clips): "
          f"RMSE {avg[0]:.4f}  MAE {avg[1]:.4f}  TEPE {avg[2]:.4f}")
    print(f"  selected Fourier order: {dict(sorted(orders.items()))}")
    if args.profile:
        print(f"  backbone (or cache read): {1000 * t_backbone / n:.1f} ms/clip, "
              f"FoSA: {1000 * t_fosa / n:.1f} ms/clip")
        print(f"  FoSA extra GPU memory: peak {peak_mb:.0f} MiB/clip, "
              f"mean {sum_mb / n / args.num_frames:.2f} MiB/frame")
    print(f"  per-clip results: {os.path.join(save_dir, 'eval_results.txt')}")


def main():
    args = parse_args()
    cfg, fcfg = load_config(args.config, {f.name: getattr(args, f'fosa_{f.name}')
                                          for f in dataclasses.fields(fosa.FosaConfig)})
    seed_everything(args.random_seed)
    if args.dataset:
        run_dataset(args, cfg, fcfg)
    else:
        run_inputs(args, cfg, fcfg)


if __name__ == '__main__':
    main()
