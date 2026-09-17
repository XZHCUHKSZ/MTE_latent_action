import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
from closed_loop_lam_v1.unified_models import MIFCARAIncidenceFlow

# Final implementation source: mte_observation_only_2026_09_09/full_pretrain.py:38

def endpoint_view(positions, actual, reference, masks):
    """Actual learned endpoints, with unchanged landmark positions, not carriers."""
    e,t,h,c,d=actual.shape
    assert actual.shape==reference.shape and (t,h,c,d)==(22,3,8,8)
    values=[]
    entities=[]
    for agent_endpoints in (actual,reference):
        full=np.zeros((e,t,h,c,16),np.float32)
        full[...,:8]=agent_endpoints
        full[...,8:]=positions[:,1:23,None,None,8:]
        table=np.zeros((e,t,h,c,8,4),np.float32)
        table[...,:2]=full.reshape(e,t,h,c,8,2)
        table[...,:4,2]=1.;table[...,4:,3]=1.
        values.append(full);entities.append(table)
    a,b=values
    adjacency=np.eye(8,dtype=np.float32)
    adjacency[:4,:]=1.;adjacency[:, :4]=1.
    return {'obs':positions[:,1:24], 'actions':np.zeros((e,t,2),np.float32),
            'cf_root':a[:,:,0,0], 'cf_ego_null':b[:,:,0,0],
            'cf_other_null':a[:,:,0,-1], 'cf_both_null':b[:,:,0,-1],
            'cf_context_root':a[:,:,0], 'cf_context_ego_null':b[:,:,0],
            'cf_context_masks':masks, 'effect_horizons':np.arange(1,4),
            'cf_horizon_entity_root':entities[0], 'cf_horizon_entity_ego_null':entities[1],
            'mif_entity_adjacency':adjacency}

# Final implementation source: mte_observation_only_2026_09_09/full_pretrain.py:64

def observation_view(positions):
    # Include x0 so LAPO's target at t=1 has real x0 history, not padding x1.
    obs=positions[:,:24];nxt=obs[:,1:]
    return {'obs':obs,'actions':np.zeros(nxt.shape[:2]+(2,),np.float32),
            'cf_root':nxt,'cf_ego_null':np.zeros_like(nxt),
            'cf_other_null':nxt,'cf_both_null':np.zeros_like(nxt)}

# Final implementation source: mte_observation_only_2026_09_09/full_pretrain.py:72

def mif_bounded(view, n, seed, updates, progress):
    """Original MIF class/objective; memory-bounded orchestration for encoding.

    Frozen common._encode_chunks uses 8192 examples, too large for the rich
    tensor. Its file/function is not patched. Training batch/objective remain
    256 with the same optimizer/schedule; only inference chunks are 64.
    """
    C.seed_all(seed)
    a,b=view['cf_horizon_entity_root'],view['cf_horizon_entity_ego_null']
    values=np.concatenate([a,a-b],-1)
    flat=values.reshape(-1,*values.shape[2:]);tr=C.valid_flat_indices(view,np.arange(n))
    m,s=C.fit_mean_std(flat[tr].reshape(-1,values.shape[-1]))
    vt=torch.as_tensor((flat-m)/s,device='cuda')
    settings=C.resolve_method_training_settings(method='edge_cara_mif',profile='matched',
        requested_representation_updates=updates,requested_policy_updates=1200,batch_size=256,
        train_transition_count=len(tr),default_lr=1e-3)
    net=MIFCARAIncidenceFlow(value_dim=8,context_masks=torch.as_tensor(view['cf_context_masks'],device='cuda'),
        entity_adjacency=torch.as_tensor(view['mif_entity_adjacency'],device='cuda'),time_steps=3,z_dim=16,hidden=128).cuda()
    opt=torch.optim.Adam(net.parameters(),lr=settings.representation_lr)
    scheduler=C._profile_scheduler(opt,settings.use_linear_warmup_decay,settings.representation_updates)
    rng=np.random.default_rng(seed+17)
    for step in range(settings.representation_updates):
        ix=next(C._batches(tr,256,rng));batch=vt[ix]
        visible=torch.rand(*batch.shape[:-1],1,device='cuda')>.30;visible[:,0,0]=True
        z,reconstructed,_=net(batch,visible)
        hidden=(~visible).expand_as(batch);shown=visible.expand_as(batch)
        loss=(F.mse_loss(reconstructed[hidden],batch[hidden]) if bool(hidden.any()) else batch.new_tensor(0.))
        loss=loss+.25*F.mse_loss(reconstructed[shown],batch[shown])+C._variance_floor(z)+.02*C._offdiag_cov(z)
        if not torch.isfinite(loss): raise ValueError('MIF nonfinite loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
        if scheduler is not None: scheduler.step()
        if step%100==0: progress('representation',update=step+1,updates=updates)
    net.eval();codes=[]
    with torch.no_grad():
        for batch in vt.split(64):
            codes.append(net(batch,torch.ones(*batch.shape[:-1],1,dtype=torch.bool,device='cuda'))[0].cpu().numpy())
    return np.concatenate(codes),{'source_class':'closed_loop_lam_v1.unified_models.MIFCARAIncidenceFlow',
            'source_objective':'common.train_representation MIF branch; identical loss and optimizer',
            'representation_parameters':sum(p.numel() for p in net.parameters()),
            'inference_chunk':64,'training_batch':256,'horizons':[1,2,3],
            'input':'predicted endpoint and difference, 8 entities with x/y/type; not same information as h1 difference-only methods'},net
