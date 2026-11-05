"""UniDepth V2 (ViT-L) as a frozen backbone; installed from third_party/UniDepth.

The paper's numbers come from UniDepth at commit bebc4b2, where this architecture is
called UniDepthV2. Later releases renamed it UniDepthV2old and changed the attention
kernels, which moves the bf16 output slightly; either name is accepted here.
"""

import torch
import torch.nn.functional as F

_AMP_DTYPE = {'fp16': torch.float16, 'bf16': torch.bfloat16}
INPUT_SIZE = (476, 630)


def load_unidepth():
    try:
        from unidepth.models import UniDepthV2old as UniDepthNet
    except ImportError:
        from unidepth.models import UniDepthV2 as UniDepthNet
    return UniDepthNet.from_pretrained("lpiccinelli/unidepth-v2old-vitl14")


def forward_unidepth(model, rgb_bt, sparse_bt, amp):
    """[B*T,3,H,W] normalised RGB -> metric depth at the network resolution."""
    rgb_uni = F.interpolate(rgb_bt, INPUT_SIZE, mode='bilinear', align_corners=False)
    with torch.autocast('cuda', dtype=_AMP_DTYPE.get(amp, torch.bfloat16),
                        enabled=(amp != 'off')):
        return model({'image': rgb_uni, 'depth': sparse_bt}, {})['depth']
