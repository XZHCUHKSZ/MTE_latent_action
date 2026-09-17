import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
import time
from closed_loop_lam_v1.unified_models import MIFCARAIncidenceFlow

# Final implementation source: observation_mte_structural_v3_2026_09_12/representation.py:9

def complement_indices(masks):
    m=np.asarray(masks,bool)
    target=np.flatnonzero(~m.any(0));assert len(target)==1
    partners=np.flatnonzero(m.any(0))
    out=[]
    for row in m:
        wanted=row.copy();wanted[partners]=~wanted[partners]
        hit=np.flatnonzero((m==wanted).all(1));assert len(hit)==1;out.append(int(hit[0]))
    out=np.asarray(out)
    assert np.array_equal(out[out],np.arange(len(m))) and np.all(out!=np.arange(len(m)))
    return out

# Final implementation source: observation_mte_structural_v3_2026_09_12/representation.py:21

def make_model(dim,masks,adjacency,hidden,arm,device='cuda'):
    net=MIFCARAIncidenceFlow(dim,torch.as_tensor(masks,device=device),
        torch.as_tensor(adjacency,device=device),3,16,hidden).to(device)
    if arm.startswith('pooled_'):
        # Control-only buffers; original MIF arms retain their original buffers.
        with torch.no_grad():
            net.mobius.copy_(torch.eye(len(masks),device=device))
            net.zeta.copy_(torch.eye(len(masks),device=device))
            for name in ['coalition_adjacency','entity_adjacency','time_adjacency']:
                matrix=getattr(net,name);matrix.fill_(1/matrix.shape[0])
    return net

# Final implementation source: observation_mte_structural_v3_2026_09_12/representation.py:33

def canonical_prediction(pred,arm,input_mean,input_std,canonical_mean,canonical_std):
    if not arm.endswith('_pair'):return pred
    d=pred.shape[-1]//2
    # Evaluate the affine map without adding then subtracting large endpoint means.
    q=(pred[...,:d]*input_std[:d]+(input_mean[:d]-canonical_mean[:d]))/canonical_std[:d]
    edge=(pred[...,:d]*input_std[:d]-pred[...,d:]*input_std[d:]
          +(input_mean[:d]-input_mean[d:]-canonical_mean[d:]))/canonical_std[d:]
    return torch.cat([q,edge],-1)

# Final implementation source: observation_mte_structural_v3_2026_09_12/representation.py:42

def train(a,b,valid,n,masks,adjacency,env,seed,updates,arm,progress):
    assert a.shape==b.shape and a.ndim==6
    C.seed_all(seed)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    shape=a.shape[:2];dim=a.shape[-1]*2
    tr=np.flatnonzero(valid[:n].reshape(-1))
    # Canonical normalization is identical across every arm, fitted before perturbation.
    canonical=np.concatenate([a,a-b],-1).reshape(-1,*a.shape[2:-1],dim)
    cm,cs=C.fit_mean_std(canonical[tr].reshape(-1,dim))
    if arm=='partner_complement':
        b=np.take(b,complement_indices(masks),axis=3)
        canonical=np.concatenate([a,a-b],-1).reshape(canonical.shape)
    target=torch.as_tensor((canonical-cm)/cs,device='cuda')
    del canonical
    if arm.endswith('_pair'):
        pair=np.concatenate([a,b],-1).reshape(-1,*a.shape[2:-1],dim)
        im,istd=C.fit_mean_std(pair[tr].reshape(-1,dim))
        inputs=torch.as_tensor((pair-im)/istd,device='cuda');del pair
    elif arm=='edge_only_input':
        inputs=target.clone();inputs[...,:dim//2]=0.;im,istd=cm,cs
    else:inputs=target;im,istd=cm,cs
    hidden=128 if env=='mpe' else 108
    batch_size=256 if env=='mpe' else 512
    settings=C.resolve_method_training_settings('edge_cara_mif','matched',updates,1200,batch_size,len(tr),default_lr=1e-3)
    net=make_model(dim,masks,adjacency,hidden,arm)
    opt=torch.optim.Adam(net.parameters(),lr=settings.representation_lr)
    scheduler=C._profile_scheduler(opt,settings.use_linear_warmup_decay,settings.representation_updates)
    rng=np.random.default_rng(seed+17)
    means=[torch.as_tensor(x,device='cuda') for x in [im,istd,cm,cs]]
    torch.cuda.synchronize();training_start=time.monotonic()
    trace=[]
    for step in range(settings.representation_updates):
        ix=next(C._batches(tr,batch_size,rng));batch=inputs[ix];truth=target[ix]
        visible=torch.rand(*batch.shape[:-1],1,device='cuda')>.30;visible[:,0,0]=True
        z,reco,_=net(batch,visible)
        reco=canonical_prediction(reco,arm,*means)
        hidden_mask=(~visible).expand_as(truth);shown=visible.expand_as(truth)
        loss=(F.mse_loss(reco[hidden_mask],truth[hidden_mask]) if hidden_mask.any() else truth.new_tensor(0.))
        loss=loss+.25*F.mse_loss(reco[shown],truth[shown])+C._variance_floor(z)+.02*C._offdiag_cov(z)
        if not torch.isfinite(loss):raise ValueError('Nonfinite representation loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
        if scheduler is not None:scheduler.step()
        if step%100==0 or step+1==updates:
            trace.append(dict(update=step+1,loss=float(loss.detach())))
            progress('representation',arm=arm,update=step+1,updates=updates)
    torch.cuda.synchronize();train_seconds=time.monotonic()-training_start
    net.eval();codes=[]
    with torch.no_grad():
        for x in inputs.split(64):
            codes.append(net(x,torch.ones(*x.shape[:-1],1,dtype=torch.bool,device='cuda'))[0].cpu().numpy())
    z=np.concatenate(codes).reshape(*shape,16);z=np.where(valid[...,None],z,0.).astype(np.float32)
    stats=dict(arm=arm,parameters=sum(p.numel() for p in net.parameters()),hidden=hidden,z_dim=16,
        batch_size=batch_size,updates=settings.representation_updates,train_seconds=train_seconds,
        total_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        input_dim=dim,valid_train_transitions=len(tr),trace=trace,
        parameter_count_caveat='edge_only_input has inactive endpoint input weights; not equal effective input capacity',
        source_class='closed_loop_lam_v1.unified_models.MIFCARAIncidenceFlow',
        buffer_control_applied=arm.startswith('pooled_'),canonical_loss_coordinates='training-standardized original endpoint+edge')
    checkpoint=dict(state_dict=net.cpu().state_dict(),arm=arm,input_mean=im,input_std=istd,
                    canonical_mean=cm,canonical_std=cs,masks=masks,adjacency=adjacency,hidden=hidden,
                    native_action_labels_read=0,simulator_queries=0)
    return z,stats,checkpoint
