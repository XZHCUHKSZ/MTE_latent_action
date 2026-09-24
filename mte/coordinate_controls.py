from mte.method_names import resolve_control
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
from closed_loop_lam_v1.unified_models import MobiusSimple,MobiusGraphEdgeCARA,BidirectionalMobiusTreeEdgeCARA
from .structured_controls import make_model

# Final implementation source: mte_attribution_resolution_2026_09_12/coordinate_ablation.py:26

def factory(model,variant,dim,masks,adj):
    model = resolve_control(model)
    cm=torch.tensor(masks,device='cuda')
    if model=='simple':net=MobiusSimple(dim,cm,16,128).cuda()
    elif model=='graph':net=MobiusGraphEdgeCARA(dim,cm,16,128).cuda()
    elif model=='tree':net=BidirectionalMobiusTreeEdgeCARA(C.EdgeCARA(dim,16,16,128),dim,cm,16,128,use_downward=variant!='upward_only').cuda()
    else:net=make_model(dim,masks,adj,128,'mif_edge')
    with torch.no_grad():
        if variant=='raw_coordinates':
            for name in ['mobius','zeta']:getattr(net,name).copy_(torch.eye(len(masks),device='cuda'))
        elif variant.startswith('self_'):
            matrix=getattr(net,variant[5:]+'_adjacency');matrix.copy_(torch.eye(len(matrix),device='cuda'))
    return net

# Final implementation source: mte_attribution_resolution_2026_09_12/coordinate_ablation.py:39

def forward(net,y,model):
    model = resolve_control(model)
    if model=='mif':
        z,r,_=net(y,torch.ones(*y.shape[:-1],1,dtype=torch.bool,device=y.device))
    else:
        r=net(y);z=r[0];r=r[10] if model=='tree' else r[-1]
    return z,r
