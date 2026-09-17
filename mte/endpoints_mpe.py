import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
from .frontends import triplets,ObservationReadout

# Final implementation source: reference_lam_mte_bridge_2026_09_10/frontend.py:11

def triplets(valid,k=1):
    # Complete valid prefix and episode-local timestamps, never cross padding.
    usable=np.zeros_like(valid);usable[:,:valid.shape[1]-k+1]=valid[:,k-1:]
    return np.argwhere(usable)

# Final implementation source: reference_lam_mte_bridge_2026_09_10/frontend.py:69

class ObservationReadout(nn.Module):
    """New interface adapter only; no redefinition of the original methods."""
    def __init__(self,d,zdim,output):
        super().__init__();self.layers=nn.Sequential(nn.Linear(2*d+zdim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,output))
    def forward(self,h,z):return self.layers(torch.cat([h,z],-1))

# Final implementation source: reference_lam_mte_bridge_2026_09_10/frontend.py:75

def _build(x,valid,n,z,env,p,out,progress):
    C.seed_all(p['seed']);rng=np.random.default_rng(p['seed']+11);e,t=triplets(valid[:n]).T
    history=np.concatenate([x[:,np.maximum(np.arange(valid.shape[1])-1,0)],x[:,:-1]],-1)
    hmean,hstd=history[e,t].mean(0),history[e,t].std(0).clip(1e-6)
    z=z.reshape(*valid.shape,-1);zm,zs=z[e,t].mean(0),z[e,t].std(0).clip(1e-6)
    d=x.shape[-1];o=8 if env=='mpe' else d
    delta=x[:,1:,:o]-x[:,:-1,:o];ym,ys=delta[e,t].mean(0),delta[e,t].std(0).clip(1e-6)
    ht=torch.as_tensor((history-hmean)/hstd,device='cuda');zt=torch.as_tensor((z-zm)/zs,device='cuda')
    yt=torch.as_tensor((delta-ym)/ys,device='cuda');net=ObservationReadout(d,z.shape[-1],o).cuda();opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    for step in range(p['readout_updates']):
        ix=rng.integers(len(e),size=256);ee,tt=e[ix],t[ix]
        loss=(net(ht[ee,tt],zt[ee,tt])-yt[ee,tt]).square().mean()
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0:progress('observation_readout',update=step+1,updates=p['readout_updates'],loss=float(loss))
    net.eval()
    with torch.no_grad():
        de,dt=triplets(valid[n:]).T;de+=n
        pred=net(ht[de,dt],zt[de,dt]);shuf=net(ht[de,dt],zt[de,dt][torch.randperm(len(de),device='cuda')])
        diagnostics=dict(dev_standardized_increment_mse=float((pred-yt[de,dt]).square().mean()),
            shuffled_latent_mse=float((shuf-yt[de,dt]).square().mean()),persistence_mse=float((torch.as_tensor(delta[de,dt]/ys,device='cuda')).square().mean()))
    torch.save(dict(state_dict=net.cpu().state_dict(),history_mean=hmean,history_std=hstd,zmean=zm,zstd=zs,delta_mean=ym,delta_std=ys,frozen=True),out/'readout.pt');net.cuda()
    # A fixed training-only donor bank, picked by past/current observations alone.
    bank=rng.choice(len(e),size=min(2048,len(e)),replace=False);be,bt=e[bank],t[bank]
    bh=ht[be,bt];bz=z[be,bt];donors=np.zeros((*valid.shape,z.shape[-1]),np.float32)
    with torch.no_grad():
        ids=triplets(valid)
        for chunk in np.array_split(ids,max(1,(len(ids)+255)//256)):
            qe,qt=chunk.T;dist=torch.cdist(ht[qe,qt],bh)
            forbidden=torch.as_tensor((qe[:,None]<n)&(qe[:,None]==be[None]),device='cuda')
            dist.masked_fill_(forbidden,float('inf'));assert torch.isfinite(dist.min(1).values).all()
            donors[qe,qt]=bz[dist.argmin(1).cpu().numpy()]
    masks=np.zeros((8,4),bool)
    for i in range(8):masks[i,1:]=[(i>>j)&1 for j in range(3)]
    start=1 if env=='mpe' else 0;stop=23 if env=='mpe' else valid.shape[1]
    a=np.zeros((len(x),stop-start,3,8,o),np.float32);b=np.zeros_like(a)
    hm,hs,zzm,zzs,dm,ds=[torch.as_tensor(v,device='cuda') for v in (hmean,hstd,zm,zs,ym,ys)]
    with torch.no_grad():
        for t0 in range(start,stop):
            if (t0-start)%10==0:progress('paired_observation_rollout',time_index=t0,total=stop)
            base=z[:,t0].reshape(len(x),4,-1);ref=donors[:,t0].reshape(len(x),4,-1)
            for ci,mask in enumerate(masks):
                za=np.where(mask[None,:,None],ref,base);zb=za.copy();zb[:,0]=ref[:,0]
                for endpoint,code in ((a,za),(b,zb)):
                    prev=torch.as_tensor(x[:,max(t0-1,0)],device='cuda');cur=torch.as_tensor(x[:,t0],device='cuda')
                    for h in range(3):
                        # Identical observed latent continuation for both endpoints, no native action ledger.
                        cc=code.reshape(len(x),-1) if h==0 else z[:,min(t0+h,valid.shape[1]-1)]
                        inc=net((torch.cat([prev,cur],-1)-hm)/hs,(torch.as_tensor(cc,device='cuda')-zzm)/zzs)*ds+dm
                        nxt=cur.clone();nxt[:,:o]+=inc
                        # Terminal/padding: hold state, never insert fabricated continuation.
                        active=torch.as_tensor(valid[:,min(t0+h,valid.shape[1]-1)] & (t0+h<valid.shape[1]),device='cuda')
                        nxt=torch.where(active[:,None],nxt,cur)
                        endpoint[:,t0-start,h,ci]=nxt[:,:o].cpu().numpy();prev,cur=cur,nxt
    assert np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite learned branches'
    np.savez_compressed(out/'endpoints.npz',actual=a,reference=b,masks=masks)
    diagnostics.update(reference_bank=2048,reference_query='previous and current observation; excludes same training episode',
        rollout='same future observed latent continuation; terminal hold',evidence='learned observation endpoints, not identified native-action interventions',
        endpoint_shape=list(a.shape),mean_absolute_edge=float(np.abs(a-b).mean()))
    return diagnostics

def build(x, valid, n, z, p, out, progress):
    '''Final MPE endpoint construction. MaMuJoCo must use endpoints_mamujoco.'''
    if x.shape[-1] != 16: raise ValueError('MPE requires 16 observation coordinates')
    return _build(x, valid, n, z, 'mpe', p, out, progress)
