"""DVD (Wan2.1-T2V-1.3B with the DVD LoRA merged) as a frozen backbone.

Uses the official clone in third_party/DVD (``dvd_root`` in the config), installed with
pip install -e . and holding the Wan2.1-T2V-1.3B weights under <dvd_root>/models and the
DVD weights under <dvd_root>/ckpt, as scripts/setup_env.sh sets it up.

DVD returns per-clip normalised disparity in [0, 1] (affine-invariant inverse depth).
This wrapper returns 1 / disparity, so that the inverse-depth fit (domain: inv in
config/dvd.yaml, as for every backbone except UniDepth V2) fits the disparity itself against
1 / sparse depth: the alignment then happens in inverse depth, where DVD's output is affine.

Loading follows DVD/test_script/test_single_video.py (load_model) and the call follows
DVD/test_script/test_from_trained_all_vid.py (one deterministic step, denoise_step 0.5).
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_MEAN = (0.485, 0.456, 0.406)       # the dataloaders apply ImageNet normalisation
_STD = (0.229, 0.224, 0.225)


class DVDBackbone(nn.Module):
    def __init__(self, dvd_root, ckpt='model.safetensors', max_side=1280):
        super().__init__()
        if not dvd_root:
            raise ValueError("DVD backbone: set dvd_root in the config to a clone of "
                             "github.com/EnVision-Research/DVD (run: bash scripts/setup_env.sh dvd)")
        dvd_root = os.path.abspath(os.path.expanduser(dvd_root))
        os.environ.setdefault('HF_HUB_OFFLINE', '1')     # use the local Wan weights only
        if dvd_root not in sys.path:
            sys.path.insert(0, dvd_root)
        from accelerate import Accelerator
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule

        cwd = os.getcwd()
        os.chdir(dvd_root)                               # from_pretrained resolves ./models
        try:
            cfg = OmegaConf.load(os.path.join(dvd_root, 'ckpt', 'model_config.yaml'))
            m = WanTrainingModule(
                accelerator=Accelerator(),
                model_id_with_origin_paths=cfg.model_id_with_origin_paths,
                trainable_models=None, use_gradient_checkpointing=False,
                lora_rank=cfg.lora_rank, lora_base_model=cfg.lora_base_model, args=cfg)
            sd = load_file(os.path.join(dvd_root, 'ckpt', ckpt), device='cpu')
            m.pipe.dit.load_state_dict(
                {k.replace('pipe.dit.', ''): v for k, v in sd.items() if 'pipe.dit.' in k},
                strict=True)
            m.merge_lora_layer()
        finally:
            os.chdir(cwd)
        self.m = m
        self.cfg = cfg
        # DDAD (1216x1936) would need >24 GB in the VAE decode; run it at <= max_side
        self.max_side = int(max_side)

    def cuda(self, device=None):
        # DiffusionTrainingModule.to also moves pipe.device; nn.Module.cuda() would not.
        self.m = self.m.to('cuda')
        return self

    @torch.no_grad()
    def forward(self, sample):
        rgb = sample['rgb']                                           # [B,T,3,H,W] normalised
        B, T, _, H, W = rgb.shape
        mean = rgb.new_tensor(_MEAN).view(1, 1, 3, 1, 1)
        std = rgb.new_tensor(_STD).view(1, 1, 3, 1, 1)
        x = (rgb.float() * std + mean).clamp(0.0, 1.0)

        # H, W must be multiples of 16 and T of the form 4n+1 (DVD asserts both).
        sc = min(1.0, self.max_side / max(H, W)) if self.max_side > 0 else 1.0
        Hp, Wp = max(16, round(H * sc / 16) * 16), max(16, round(W * sc / 16) * 16)
        if (Hp, Wp) != (H, W):
            x = F.interpolate(x.flatten(0, 1), (Hp, Wp), mode='bilinear',
                              align_corners=False).view(B, T, 3, Hp, Wp)
        Tp = T + (1 - T % 4) % 4                                       # 3 -> 5
        if Tp > T:
            x = torch.cat([x, x[:, -1:].expand(B, Tp - T, 3, Hp, Wp)], 1)  # repeat last frame

        out = self.m.pipe(
            prompt=[''] * B, negative_prompt=[''] * B, mode=self.cfg.mode,
            height=Hp, width=Wp, num_frames=Tp, batch_size=B, input_video=x,
            denoise_step=self.cfg.denoise_step, cfg_scale=1, seed=0, tiled=False)
        disp = torch.from_numpy(out['depth']).to(rgb.device).float().mean(-1)  # [B,Tp,Hp,Wp]
        disp = disp[:, :T]
        if (Hp, Wp) != (H, W):
            disp = F.interpolate(disp.unsqueeze(2).flatten(0, 1), (H, W), mode='bilinear',
                                 align_corners=False).view(B, T, H, W)
        depth = 1.0 / disp.clamp_min(1e-3)
        return {'depth': depth.unsqueeze(2)}                           # [B,T,1,H,W]
