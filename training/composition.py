import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C

# Final implementation source: mte_anchor_auxiliary_2026_09_10/worker.py:12

class FrozenPair(nn.Module):
    """Deployment composition, not a replacement scientific encoder."""
    def __init__(self,anchor,aux=None,duplicate=False):
        super().__init__();self.anchor=anchor;self.aux=aux;self.duplicate=duplicate
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,x,hidden=None):
        h1,h2=(None,None) if hidden is None else hidden
        a,h1=self.anchor(x,h1)
        if self.duplicate:return torch.cat([a,a],-1),(h1,None)
        if self.aux is None:return a,(h1,None)
        b,h2=self.aux(x,h2);return torch.cat([a,b],-1),(h1,h2)

# Final implementation source: mte_anchor_auxiliary_2026_09_10/worker.py:24

def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['frozen_before_grounding'] and ck['native_action_labels_read']==0
    net=C.RecurrentPolicy(len(ck['obs_mean']),ck['z_dim']);net.load_state_dict(ck['state_dict'])
    return net.eval(),ck
