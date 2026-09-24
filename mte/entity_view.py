import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/entities.py:4

def entities(x):
    x=np.asarray(x,np.float32);rows=[]
    for j in range(4):
        # Official k=0 order is alphabetical: ankle first, hip second.
        columns=[6+2*j,20+2*j]+list(range(33+18*j,51+18*j))+[5+2*j,19+2*j]+list(range(27,33))
        columns += [5+2*k for k in range(4) if k!=j]+list(range(5))+list(range(13,19))
        v=x[...,columns].copy();v[...,2:20]=v[...,2:20].clip(-1,1);v[...,22:28]=v[...,22:28].clip(-1,1)
        identity=np.zeros((*v.shape[:-1],4),np.float32);identity[...,j]=1
        rows.append(np.concatenate([v,identity],-1))
    return np.stack(rows,-2)
