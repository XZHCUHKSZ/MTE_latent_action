import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C

# Final implementation source: reference_lam_mte_bridge_2026_09_10/frontend.py:7

def local_observations(x,env):
    if env=='mpe':return x[...,:8].reshape(*x.shape[:-1],4,2)
    return np.stack([x[...,[5+2*j,6+2*j,19+2*j,20+2*j]] for j in range(4)],-2)

# Final implementation source: reference_lam_mte_bridge_2026_09_10/frontend.py:11

def triplets(valid,k=1):
    # Complete valid prefix and episode-local timestamps, never cross padding.
    usable=np.zeros_like(valid);usable[:,:valid.shape[1]-k+1]=valid[:,k-1:]
    return np.argwhere(usable)

# Final implementation source: reference_lam_mte_bridge_2026_09_10/frontend.py:16

def train(x,valid,n,env,variant,p,out,progress):
    C.seed_all(p['seed']);rng=np.random.default_rng(p['seed'])
    local=variant=='laom_entity_k3';lapo=variant=='lapo_joint'
    y=local_observations(x,env) if local else x
    # Normalize only valid training current/next observations, pooled entities.
    ep,tt=triplets(valid[:n]).T
    norm=np.concatenate([y[ep,tt],y[ep,tt+1]],0).reshape(-1,y.shape[-1])
    mean,std=norm.mean(0),norm.std(0).clip(1e-6)
    v=torch.as_tensor((y-mean)/std,device='cuda',dtype=torch.float32)
    net=C.LAPOStateAdapter(y.shape[-1],p['frontend_hidden']) if lapo else C.LAOMStateAdapter(y.shape[-1],p['laom_entity_dim'] if local else p['laom_joint_dim'],p['frontend_hidden'])
    net=net.cuda();opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    kmax=1 if lapo else p['laom_max_offset'];idx=triplets(valid[:n],kmax)
    assert len(idx)>0
    for step in range(p['frontend_updates']):
        e,t=idx[rng.integers(len(idx),size=p['frontend_batch'])].T
        if lapo:
            pred,la,lq,vq,perplexity,_=net(torch.stack([v[e,np.maximum(t-1,0)],v[e,t],v[e,t+1]],1))
            loss=(pred-v[e,t+1]).square().mean()+vq
        else:
            k=rng.integers(1,kmax+1,size=len(e));cur=v[e,t];future=v[e,t+k];target=v[e,t+1]
            if local:cur,future,target=(z.reshape(-1,z.shape[-1]) for z in (cur,future,target))
            pred,la,_=net(cur,future);loss=(pred-net.target(target)).square().mean()
        assert torch.isfinite(loss), 'Nonfinite frontend'
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        if not lapo:net.update_target(p['ema_tau'])
        if step%100==0:progress('frontend',update=step+1,updates=p['frontend_updates'],loss=float(loss))
    net.eval();dim=128 if lapo else (p['laom_entity_dim'] if local else p['laom_joint_dim'])
    z=np.zeros((*valid.shape,4,dim) if local else (*valid.shape,dim),np.float32)
    losses=[]
    with torch.no_grad():
        ids=triplets(valid)
        for chunk in np.array_split(ids,max(1,(len(ids)+511)//512)):
            e,t=chunk.T
            if lapo:
                pred,code,lq,vq,_,_=net(torch.stack([v[e,np.maximum(t-1,0)],v[e,t],v[e,t+1]],1))
                err=(pred-v[e,t+1]).square().mean(-1)
            else:
                cur,nxt=v[e,t],v[e,t+1]
                if local:cur,nxt=(a.reshape(-1,a.shape[-1]) for a in (cur,nxt))
                pred,code,_=net(cur,nxt);err=(pred-net.target(nxt)).square().mean(-1)
                if local:code=code.reshape(len(e),4,dim);err=err.reshape(len(e),4).mean(-1)
            z[e,t]=code.cpu().numpy()
            losses.extend(err[e>=n].cpu().tolist())
    zv=z[:n][valid[:n]].reshape(-1,dim)
    assert np.isfinite(z).all() and float(zv.std(0).mean())>1e-7,'Collapsed/nonfinite latent'
    np.savez_compressed(out/'codes.npz',z=z)
    torch.save(dict(state_dict=net.cpu().state_dict(),mean=mean,std=std,variant=variant,
        source_class=type(net).__module__+'.'+type(net).__name__,frozen=True),out/'frontend.pt')
    return dict(source_class=type(net).__module__+'.'+type(net).__name__,latent_shape=list(z.shape),
        parameters=sum(t.numel() for t in net.parameters()),train_loss=float(loss),dev_loss=float(np.mean(losses)),
        dev_loss_scope='within objective only; LAOM learned feature errors not cross-model physical errors',
        mean_coordinate_std=float(zv.std(0).mean()),multi_step_target='next observation feature even when inverse future is t+k')

# Final implementation source: reference_lam_mte_bridge_2026_09_10/frontend.py:69

class ObservationReadout(nn.Module):
    """New interface adapter only; no redefinition of the original methods."""
    def __init__(self,d,zdim,output):
        super().__init__();self.layers=nn.Sequential(nn.Linear(2*d+zdim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,output))
    def forward(self,h,z):return self.layers(torch.cat([h,z],-1))
