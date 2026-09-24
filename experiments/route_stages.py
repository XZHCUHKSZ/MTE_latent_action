"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

from pathlib import Path

import numpy as np

from mte.frontends import train

from utils.artifacts import data

from training.history_mpe import train_history_policy as mpe_history

from training.representation_mamujoco import train_history_policy as ma_history

def train_global(s,p,out,progress):
    x,m,n=data(s);out=Path(out);front=out/'frontend';front.mkdir(exist_ok=False)
    fp=dict(p['frontend_config'],seed=s['seed']+100000)
    if fp['laom_joint_dim']!=16:raise ValueError('This control uses Global-Joint16')
    fr=train(x,m,n,s['env'],'laom_joint_k3',fp,front,progress)
    with np.load(front/'codes.npz',allow_pickle=False) as f:z=f['z']
    assert z.shape==(*m.shape,16)
    if s['env']=='mpe':h=mpe_history(x,z[:,1:23],n,fp['seed'],p['policy_updates'],out/'policy',progress)
    else:h=ma_history(x,z,m,n,fp['seed'],p['policy_updates'],out/'policy',progress,method='global_joint16_scope_control')
    return dict(frontend=fr,history=h,native_action_labels_read=0,simulator_queries=0)
