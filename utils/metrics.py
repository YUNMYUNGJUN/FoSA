"""Depth metrics on [B,T,1,H,W] tensors: RMSE, MAE (over valid GT) and TEPE (temporal)."""

import torch


def rmse_eval(pred, gt, mask):
    '''
    pred: [B, T, 1, H, W]
    gt: [B, T, 1, H, W]
    mask: [B, T, 1, H, W]
    '''
    with torch.no_grad():
        return torch.sqrt(((pred[mask] - gt[mask]) ** 2).mean())


def mae_eval(pred, gt, mask):
    '''
    pred: [B, T, 1, H, W]
    gt: [B, T, 1, H, W]
    mask: [B, T, 1, H, W]
    '''
    with torch.no_grad():
        return torch.abs(pred[mask] - gt[mask]).mean()


def tepe_eval(pred, gt, t_valid=1e-3, reduce_batch='mean'):
    '''
    pred: [B, T, 1, H, W]
    gt: [B, T, 1, H, W]
    '''
    with torch.no_grad():
        B, T, C, H, W = gt.shape
        if T < 2:
            return torch.tensor(0.0, device=gt.device)

        m1 = gt[:, :-1] > t_valid
        m2 = gt[:, 1:]  > t_valid
        m  = m1 & m2                          # [B,T-1,C,H,W]

        dg = gt[:, 1:] - gt[:, :-1]
        dp = pred[:, 1:] - pred[:, :-1]
        ae = (dg - dp).abs()                  # [B,T-1,C,H,W]

        pix_sum = (ae * m).sum(dim=(2,3,4))   # [B,T-1]
        pix_cnt = m.sum(dim=(2,3,4))          # [B,T-1]
        per_pair = pix_sum / pix_cnt.clamp_min(1)  # [B,T-1], empty pairs give 0

        pair_w = (pix_cnt > 0).float()        # average over non-empty pairs only
        denom = pair_w.sum(dim=1).clamp_min(1)
        tepe_per_b = (per_pair * pair_w).sum(dim=1) / denom  # [B]

        if reduce_batch == 'none':
            return tepe_per_b
        elif reduce_batch == 'sum':
            return tepe_per_b.sum()
        else:
            return tepe_per_b.mean()
