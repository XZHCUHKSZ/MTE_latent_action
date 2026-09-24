import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
import json
from copy import deepcopy
from closed_loop_lam_v1.common import ActionDecoder,RecurrentPolicy,InverseDynamics,seed_all

# Final implementation source: mte_observation_only_2026_09_09/grounding.py:76

def train_decoder(z, actions, fit, val, seed, updates=500):
    seed_all(seed)
    net=ActionDecoder(z.shape[-1],2).cuda()
    x=torch.as_tensor(z,device='cuda');y=torch.as_tensor(actions,device='cuda')
    opt=torch.optim.Adam(net.parameters(),lr=1e-3);rng=np.random.default_rng(seed+211)
    xx=x[fit].reshape(-1,z.shape[-1]);yy=y[fit].reshape(-1,2)
    best=float('inf');best_state=None;selected=0
    for step in range(1,updates+1):
        ix=rng.integers(0,len(xx),min(256,len(xx)))
        loss=(net(xx[ix])-yy[ix]).square().mean()
        if not torch.isfinite(loss): raise ValueError('Decoder nonfinite')
        opt.zero_grad();loss.backward();opt.step()
        if step==1 or step%25==0 or step==updates:
            with torch.no_grad(): candidate=(net(x[val])-y[val]).square().mean().item()
            if candidate<best:
                best=candidate;selected=step
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.cpu().eval(),dict(validation_action_mse=best,selected_update=selected,updates=updates)

# Final implementation source: mte_observation_only_2026_09_09/grounding.py:116

def train_supervised_history(positions, targets, fit, validation_positions, validation_actions, seed, updates=1200):
    """Original GRU class and BC MSE, with common x0 warmup and counted validation."""
    seed_all(seed)
    obs=positions[:,:23]
    mean=obs.mean((0,1));std=obs.std((0,1)).clip(1e-6)
    x=torch.as_tensor((obs-mean)/std,device='cuda');y=torch.as_tensor(targets,device='cuda')
    vx=torch.as_tensor((validation_positions[:,:23]-mean)/std,device='cuda')
    vy=torch.as_tensor(validation_actions,device='cuda')
    net=RecurrentPolicy(16,2).cuda();opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed+313);best=float('inf');selected=0;best_state=None
    for step in range(1,updates+1):
        ix=rng.choice(fit,min(16,len(fit)),replace=False)
        pred,_=net(x[ix]);loss=(pred[:,1:]-y[ix]).square().mean()
        if not torch.isfinite(loss): raise ValueError('BC nonfinite')
        opt.zero_grad();loss.backward();opt.step()
        if step==1 or step%25==0 or step==updates:
            with torch.no_grad():
                pred,_=net(vx);candidate=(pred[:,1:]-vy).square().mean().item()
            if candidate<best:
                best=candidate;selected=step;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.cpu().eval(),mean,std,dict(validation_action_mse=best,selected_update=selected,updates=updates,
        source_class='closed_loop_lam_v1.common.RecurrentPolicy',objective='BC MSE on t=1..22; x0 history warmup')

# Final implementation source: mte_observation_only_2026_09_09/grounding.py:141

def train_idm(positions, local_ids, actions, fit, val, seed, updates=500, policy_updates=1200):
    seed_all(seed)
    mean=positions[:,:23].mean((0,1));std=positions[:,:23].std((0,1)).clip(1e-6)
    x=torch.as_tensor((positions[:,1:23]-mean)/std,device='cuda')
    nxt=torch.as_tensor((positions[:,2:24]-mean)/std,device='cuda')
    xx=x[local_ids[fit]].reshape(-1,16);nn=nxt[local_ids[fit]].reshape(-1,16)
    yy=torch.as_tensor(actions[fit].reshape(-1,2),device='cuda')
    model=InverseDynamics(16,2).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed+919)
    for _ in range(updates):
        ix=rng.integers(0,len(xx),min(512,len(xx)))
        loss=(model(xx[ix],nn[ix])-yy[ix]).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): pseudo=model(x,nxt).cpu().numpy()
    # Labels only from fit; policy early selection uses val labels within B.
    net,om,os,diag=train_supervised_history(positions,pseudo,np.arange(len(positions)),
        positions[local_ids[val]],actions[val],seed+1,policy_updates)
    diag.update(idm_updates=updates,idm_source_class='closed_loop_lam_v1.common.InverseDynamics',
        validation='budgeted true-action validation; no free development labels',idm_final_loss=loss.item())
    return net,om,os,diag

# Final implementation source: mte_observation_only_2026_09_09/grounding.py:163

class GroundedPolicy:
    def __init__(self,net,mean,std,decoder=None):
        self.net=net.cpu().eval();self.mean=np.asarray(mean);self.std=np.asarray(std);self.decoder=decoder

    def act(self,position,hidden):
        with torch.no_grad():
            x=torch.as_tensor((position-self.mean)/self.std,dtype=torch.float32).reshape(1,1,16)
            pred,hidden=self.net(x,hidden)
            action=pred[:,-1] if self.decoder is None else self.decoder(pred[:,-1])
        return action.numpy().reshape(2).clip(-1,1).astype(np.float32),hidden
