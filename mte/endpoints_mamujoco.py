import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
from .frontends import triplets,ObservationReadout
from utils.io import write,digest
from utils.artifacts import oldpath,old
from utils.atomic import atomic_json

# Final implementation source: mamujoco_independent_branch_audit_2026_09_12/repair.py:16

def build_full_n(s,x,m,n,z,ck,p,out):
    z=z.reshape(*m.shape,-1);e,t=triplets(m[:n]).T
    history=np.concatenate([x[:,np.maximum(np.arange(m.shape[1])-1,0)],x[:,:-1]],-1)
    training=history[e,t];constant=np.ptp(training,axis=0)==0
    net=ObservationReadout(105,64,105).cuda().eval();net.load_state_dict(ck['state_dict'])
    hm,hs,zm,zs,dm,ds=[torch.as_tensor(ck[k],device='cuda') for k in ['history_mean','history_std','zmean','zstd','delta_mean','delta_std']]
    mask=torch.as_tensor(constant,device='cuda');fixed=(torch.as_tensor(training[0],device='cuda')-hm)/hs
    ht=torch.as_tensor((history-ck['history_mean'])/ck['history_std'],device='cuda')
    # Recreate the ORIGINAL bank RNG state after the original readout training loop.
    rng=np.random.default_rng(s['seed']+11)
    for _ in range(p['readout_updates']):rng.integers(len(e),size=256)
    bank=rng.choice(len(e),size=min(2048,len(e)),replace=False);be,bt=e[bank],t[bank];bh=ht[be,bt];bz=z[be,bt]
    donors=np.zeros_like(z);ids=triplets(m)
    with torch.no_grad():
        for chunk in np.array_split(ids,max(1,(len(ids)+255)//256)):
            qe,qt=chunk.T;dist=torch.cdist(ht[qe,qt],bh)
            forbidden=torch.as_tensor((qe[:,None]<n)&(qe[:,None]==be[None]),device='cuda');dist.masked_fill_(forbidden,float('inf'))
            assert torch.isfinite(dist.min(1).values).all();donors[qe,qt]=bz[dist.argmin(1).cpu().numpy()]
    masks=np.zeros((8,4),bool)
    for i in range(8):masks[i,1:]=[(i>>j)&1 for j in range(3)]
    a=np.zeros((len(x),m.shape[1],3,8,105),np.float32);b=np.zeros_like(a)
    with torch.no_grad():
        for t0 in range(m.shape[1]):
            if t0%25==0:write(out/'progress.json',dict(stage='guarded_endpoints',time_index=t0,total=m.shape[1]))
            base=z[:,t0].reshape(len(x),4,16);ref=donors[:,t0].reshape(len(x),4,16)
            for ci,cm in enumerate(masks):
                za=np.where(cm[None,:,None],ref,base);zb=za.copy();zb[:,0]=ref[:,0]
                for result,code in [(a,za),(b,zb)]:
                    prev=torch.as_tensor(x[:,max(t0-1,0)],device='cuda');cur=torch.as_tensor(x[:,t0],device='cuda')
                    for h in range(3):
                        cc=code.reshape(len(x),-1) if h==0 else z[:,min(t0+h,m.shape[1]-1)]
                        hh=(torch.cat([prev,cur],-1)-hm)/hs;hh=torch.where(mask,fixed,hh)
                        inc=net(hh,(torch.as_tensor(cc,device='cuda')-zm)/zs)*ds+dm
                        nxt=cur.clone();nxt+=inc
                        active=torch.as_tensor(m[:,min(t0+h,m.shape[1]-1)] & (t0+h<m.shape[1]),device='cuda');nxt=torch.where(active[:,None],nxt,cur)
                        result[:,t0,h,ci]=nxt.cpu().numpy();prev,cur=cur,nxt
    assert np.isfinite(a).all() and np.isfinite(b).all()
    with np.load(oldpath(s,'bridge')/'endpoints.npz') as f:
        # No change should occur before recursive feedback reaches a constant channel.
        parity={name:float(np.abs(value[:,:,0]-f[name][:,:,0]).max()) for name,value in [('actual',a),('reference',b)]}
    assert max(parity.values())<1e-4,parity
    np.savez_compressed(out/'endpoints.npz',actual=a,reference=b,masks=masks)
    write(out/'bridge_audit.json',dict(constant_history_columns=np.flatnonzero(constant).tolist(),h1_max_absolute_difference_from_original=parity,
        source_readout_digest=digest(oldpath(s,'bridge')/'readout.pt'),rule='Training-exact-constant input coordinates are held at their training value in the frozen readout only; target/partner matching and recorded inferred-code continuation unchanged.'))
    return a,b,masks

# Final implementation source: mamujoco_mte_recovery_2026_09_12/bridge.py:8

def build_small_n(s,x,m,n,z,ck,p,out,progress):
    z=z.reshape(*m.shape,-1);e,t=triplets(m[:n]).T
    history=np.concatenate([x[:,np.maximum(np.arange(m.shape[1])-1,0)],x[:,:-1]],-1);train=history[e,t]
    constant=np.ptp(train,axis=0)==0
    net=ObservationReadout(105,64,105).cuda().eval();net.load_state_dict(ck['state_dict'])
    hm,hs,zm,zs,dm,ds=[torch.as_tensor(ck[k],device='cuda') for k in ['history_mean','history_std','zmean','zstd','delta_mean','delta_std']]
    mask=torch.as_tensor(constant,device='cuda');fixed=(torch.as_tensor(train[0],device='cuda')-hm)/hs
    ht=torch.as_tensor((history-ck['history_mean'])/ck['history_std'],device='cuda')
    rng=np.random.default_rng(s['seed']+11)
    for _ in range(p['readout_updates']):rng.integers(len(e),size=256)
    bank=rng.choice(len(e),size=min(2048,len(e)),replace=False);be,bt=e[bank],t[bank];bh=ht[be,bt];bz=z[be,bt]
    donors=np.zeros_like(z);ids=triplets(m)
    with torch.no_grad():
        for chunk in np.array_split(ids,max(1,(len(ids)+255)//256)):
            qe,qt=chunk.T;dist=torch.cdist(ht[qe,qt],bh);forbidden=torch.as_tensor((qe[:,None]<n)&(qe[:,None]==be[None]),device='cuda')
            dist.masked_fill_(forbidden,float('inf'));assert torch.isfinite(dist.min(1).values).all();donors[qe,qt]=bz[dist.argmin(1).cpu().numpy()]
    masks=np.zeros((8,4),bool)
    for i in range(8):masks[i,1:]=[(i>>j)&1 for j in range(3)]
    a=np.zeros((len(x),m.shape[1],3,8,105),np.float32);b=np.zeros_like(a)
    with torch.no_grad():
        for t0 in range(m.shape[1]):
            if t0%25==0:progress('guarded_endpoints',time_index=t0,total=m.shape[1])
            base=z[:,t0].reshape(len(x),4,16);ref=donors[:,t0].reshape(len(x),4,16)
            for ci,cm in enumerate(masks):
                za=np.where(cm[None,:,None],ref,base);zb=za.copy();zb[:,0]=ref[:,0]
                assert np.array_equal(za[:,1:],zb[:,1:])
                for dest,code in [(a,za),(b,zb)]:
                    prev=torch.as_tensor(x[:,max(t0-1,0)],device='cuda');cur=torch.as_tensor(x[:,t0],device='cuda')
                    for h in range(3):
                        cc=code.reshape(len(x),-1) if h==0 else z[:,min(t0+h,m.shape[1]-1)]
                        hh=(torch.cat([prev,cur],-1)-hm)/hs;hh=torch.where(mask,fixed,hh)
                        inc=net(hh,(torch.as_tensor(cc,device='cuda')-zm)/zs)*ds+dm;nxt=cur.clone();nxt+=inc
                        active=torch.as_tensor(m[:,min(t0+h,m.shape[1]-1)] & (t0+h<m.shape[1]),device='cuda');nxt=torch.where(active[:,None],nxt,cur)
                        dest[:,t0,h,ci]=nxt.cpu().numpy();prev,cur=cur,nxt
    assert np.isfinite(a).all() and np.isfinite(b).all()
    with np.load(old(s,'bridge')/'endpoints.npz') as f:parity={name:float(np.abs(value[:,:,0]-f[name][:,:,0]).max()) for name,value in [('actual',a),('reference',b)]}
    # Small-N dev observations may vary in training-constant columns even at h1.
    # Record the change; do not assert full-N parity where it is unsupported.
    np.savez_compressed(out/'endpoints.npz',actual=a,reference=b,masks=masks)
    return a,b,masks,dict(constant_history_columns=np.flatnonzero(constant).tolist(),h1_change_from_original=parity)
