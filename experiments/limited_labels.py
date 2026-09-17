"""X1. Does observation-derived supervision support target-agent control?

Final representation/history stages only. Data exports, global freeze and label
grounding are separate; no simulator or native actions enter these functions.
"""
from pathlib import Path
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from utils.artifacts import data, endpoints
from training.representation_mpe import endpoint_view, observation_view, mif_bounded
from training.history_mpe import train_history_policy as mpe_history
from training.representation_mamujoco import train_backend, train_history_policy as ma_history
from mte.entity_view import entities
from mte.structured_controls import train as structured_train

OBSERVATION_METHODS={'continuous_lam','lapo_state_adapter','laom_state_adapter'}
FRONTEND_METHODS={'entity_target','entity_joint','lapo_joint','laom_joint_k3'}

def train_history(s, method, out, progress, p):
    """Route paper methods to their final implementation, with explicit inputs.

    s: env, seed, n, train, dev, endpoints, endpoint_version; frontends additionally
    need codes. p is configs/numeric.json. Caller installs utils.access.guard in
    a fresh process before invoking any training stage.
    """
    x,m,n=data(s);seed=s['seed'];env=s['env'];out=Path(out)
    if not out.is_dir():raise ValueError('Create a fresh stage output directory first')
    updates=p['backend_updates'][env]
    if method in FRONTEND_METHODS:
        with np.load(s['codes'],allow_pickle=False) as f:z=f['z']
        if method=='entity_target':z=z[:,:,0]
        elif method=='entity_joint':z=z.reshape(*z.shape[:2],-1)
        if env=='mpe':z=z[:,1:23]
        diag={'source':'frozen observation frontend','latent_dim':z.shape[-1]}
    else:
        obs=method in OBSERVATION_METHODS
        a=b=cm=None
        if not obs:
            with np.load(endpoints(s),allow_pickle=False) as f:a,b,cm=f['actual'],f['reference'],f['masks']
        if env=='mamujoco':
            if method=='edge_cara_mif':
                # Final repaired MIF, NOT the superseded direct backend MIF route.
                z,diag,ck=structured_train(entities(a),entities(b),m,n,cm,np.ones((4,4),np.float32),env,seed,updates,'mif_edge',progress)
                torch.save(ck,out/'representation.pt')
            else:
                h=0 if method=='edge_cara_h1' else 2
                native='edge_cara' if method in {'edge_cara_h1','edge_cara_h3'} else method
                z,diag=train_backend(x,m,None if obs else a[:,:,h],None if obs else b[:,:,h],cm,native,seed,n,updates)
        elif env=='mpe':
            view=observation_view(x) if obs else endpoint_view(x,a,b,cm)
            native='edge_cara' if method in {'edge_cara_h1','edge_cara_h3'} else method
            if method=='edge_cara_h3':
                aa=np.concatenate([a[:,:,2],np.broadcast_to(x[:,1:23,None,8:],(*a.shape[:2],8,8))],-1)
                bb=np.concatenate([b[:,:,2],np.broadcast_to(x[:,1:23,None,8:],(*b.shape[:2],8,8))],-1)
                view.update(cf_root=aa[:,:,0],cf_ego_null=bb[:,:,0],cf_other_null=aa[:,:,-1],cf_both_null=bb[:,:,-1],cf_context_root=aa,cf_context_ego_null=bb)
            if method=='edge_cara_mif':
                z,diag,net=mif_bounded(view,n,seed,updates,progress)
                torch.save(net.cpu().state_dict(),out/'representation.pt')
            else:
                r=C.train_representation(view,native,np.arange(n),seed,z_dim=16,hidden=128,updates=updates,batch_size=256,device='cuda',training_profile='matched')
                z=r.z;diag=dict(r.diagnostics,parameters=r.parameter_count)
                if obs:z=z.reshape(len(x),23,-1)[:,1:]
            z=z.reshape(len(x),22,-1)
        else:raise ValueError('Unknown numeric environment')
    if env=='mpe':row=mpe_history(x,z,n,seed,p['policy_updates'],out/'policy',progress)
    else:
        native='edge_cara' if method in {'edge_cara_h1','edge_cara_h3'} else method
        row=ma_history(x,z,m,n,seed,p['policy_updates'],out/'policy',progress,method=native)
    return dict(history=row,representation=diag,native_action_labels_read=0,simulator_queries=0)

def label_split(budget):
    """B includes fit AND validation episodes; same prefix for every method."""
    val=np.arange(0,budget,4);fit=np.setdiff1d(np.arange(budget),val)
    return fit,val
