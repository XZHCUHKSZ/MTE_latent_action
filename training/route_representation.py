from mte.method_names import resolve_control
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
import gc,time
from closed_loop_lam_v1.unified_models import MobiusSimple,MobiusGraphEdgeCARA,MIFCARAIncidenceFlow,lattice_operators
from mte.structured_controls import complement_indices
from utils.artifacts import data,rich,endpoints
from evaluation.probes import reader
ARMS={"graph_matched":("graph","mobius"),"simple_matched":("simple","mobius"),"mif_matched":("mif","mobius")}

# Final implementation source: mte_family_transfer_2026_09_16/representation.py:13

def factory(model,variant,dim,masks,adj,env):
    model = resolve_control(model)
    cm=torch.as_tensor(masks,device='cuda')
    if model=='graph':net=MobiusGraphEdgeCARA(dim,cm,16,128).cuda()
    elif model=='simple':net=MobiusSimple(dim,cm,16,128).cuda()
    else:net=MIFCARAIncidenceFlow(dim,cm,torch.as_tensor(adj,device='cuda'),3,16,128 if env=='mpe' else 108).cuda()
    if variant=='raw_coordinates':
        with torch.no_grad():
            for name in ('mobius','zeta'):getattr(net,name).copy_(torch.eye(len(cm),device='cuda'))
    return net

# Final implementation source: mte_family_transfer_2026_09_16/representation.py:23

def forward(net,y,model):
    model = resolve_control(model)
    if model=='mif':
        z,r,_=net(y,torch.ones(*y.shape[:-1],1,dtype=torch.bool,device=y.device));return z,r
    r=net(y);return r[0],r[-1]

# Final implementation source: mte_family_transfer_2026_09_16/representation.py:28

def decode(net,z,model):
    """Expose the existing decoder for measurement only; checked against forward."""
    model = resolve_control(model)
    if model=='mif':
        coords=(net.time_embedding[:,None,None,:]+net.coalition_embedding[None,:,None,:]+net.entity_embedding[None,None,:,:])
        coords=coords[None].expand(len(z),-1,-1,-1,-1)
        coefficients=net.value_decoder(torch.cat([z[:,None,None,None,:].expand_as(coords),coords],-1))
        return torch.einsum('cd,btdek->btcek',net.zeta,coefficients)
    masks=(net.masks if model=='graph' else net.context_masks)[None].expand(len(z),-1,-1)
    orders=(net.order/net.order.max().clamp_min(1.))[None].expand(len(z),-1,-1)
    coefficients=net.coefficient_decoder(torch.cat([z[:,None].expand(-1,masks.shape[1],-1),masks,orders],-1))
    return torch.einsum('cd,bdk->bck',net.zeta,coefficients)

# Final implementation source: mte_family_transfer_2026_09_16/representation.py:40

def tables(s,arm):
    arm = resolve_control(arm)
    model,variant=ARMS[arm]
    if model=='mif':
        x,m,n,a,b,masks,adj,valid=rich(s)
        y=np.concatenate([a,a-b],-1);edge_dim=a.shape[-1]
    else:
        x,m,n=data(s)
        with np.load(endpoints(s),allow_pickle=False) as f:a,b,masks=f['actual'][:,:,2],f['reference'][:,:,2],f['masks']
        if variant=='partner_complement':b=np.take(b,complement_indices(masks),axis=2)
        y=a-b;adj=None;valid=m[:,1:23] if s['env']=='mpe' else m;edge_dim=y.shape[-1]
    assert y.shape[:2]==valid.shape and np.isfinite(y).all()
    del a,b;gc.collect()
    return x,m,n,y,valid,masks,adj,edge_dim

# Final implementation source: mte_family_transfer_2026_09_16/representation.py:54

def coefficients(y,model,M,edge_dim):
    model = resolve_control(model)
    if model=='mif':return np.einsum('cd,bhdek->bhcek',M,y[...,edge_dim:],optimize=True)
    return np.einsum('cd,bdk->bck',M,y,optimize=True)

# Final implementation source: mte_family_transfer_2026_09_16/representation.py:58

def measure(net,model,y,valid,n,z,history,masks,edge_dim,seed,checkpoint):
    model = resolve_control(model)
    tr=np.argwhere(valid[:n]);dv=np.argwhere(valid[n:]);dv[:,0]+=n
    rng=np.random.default_rng(811901+seed);tr=tr[rng.choice(len(tr),min(2048,len(tr)),replace=False)]
    rng=np.random.default_rng(811902+seed);dv=dv[rng.choice(len(dv),min(512,len(dv)),replace=False)]
    t=y[tuple(tr.T)];v=y[tuple(dv.T)];M=lattice_operators(torch.tensor(masks))['mobius'].numpy()
    ct=coefficients(t,model,M,edge_dim);cv=coefficients(v,model,M,edge_dim)
    rows=[]
    def score(stage,kind,pred,truth,mean):
        mse=float(np.square(pred.astype(float)-truth).mean());den=float(np.square(truth.astype(float)-mean).mean())
        rows.append(dict(stage=stage,kind=kind,mse=mse,mean_baseline_mse=den,skill=1-mse/den if den>1e-12 else None))
    for stage,source in [('representation',z),('history',history)]:
        zt,zv=source[tuple(tr.T)],source[tuple(dv.T)]
        physical=zv if stage=='representation' else zv*checkpoint['z_std']+checkpoint['z_mean']
        predictions=[]
        with torch.no_grad():
            for lo in range(0,len(physical),32):predictions.append(decode(net,torch.tensor(physical[lo:lo+32],device='cuda'),model).cpu().numpy())
        reco=np.concatenate(predictions)
        score(stage,'native_all_values',reco,v,t.mean(0))
        cr=coefficients(reco,model,M,edge_dim)
        linear=reader(zt.astype(float),ct.reshape(len(ct),-1).astype(float),zv.astype(float)).reshape(cv.shape)
        for kind,pred in [('native',cr),('linear',linear)]:
            for k,mask in enumerate(masks):
                axis=2 if model=='mif' else 1
                score(stage,kind+'_subset_'+('empty' if not mask.any() else '_'.join(map(str,np.flatnonzero(mask)))),np.take(pred,k,axis),np.take(cv,k,axis),np.take(ct.mean(0),k,axis-1))
    return dict(rows=rows,fit_count=len(tr),dev_count=len(dv),ridge=.01,
                target='Own frozen learned prediction table; not physical ground truth. Matched/raw share the exact target; complement does not.',
                history_native='Existing decoder applied after reversing history-target standardization; inability to decode is not information-theoretic absence.')

# Final implementation source: mte_family_transfer_2026_09_16/representation.py:86

def train(s,arm,p,out,progress):
    arm = resolve_control(arm)
    model,variant=ARMS[arm];x,m,n,y,valid,masks,adj,edge_dim=tables(s,arm)
    assert n==s['n']
    mu,sd=C.fit_mean_std(y[:n][valid[:n]].reshape(-1,y.shape[-1]));y=(y-mu)/sd
    train_flat=np.flatnonzero(valid[:n].reshape(-1));flat=y.reshape(-1,*y.shape[2:])
    C.seed_all(s['seed']);net=factory(model,variant,y.shape[-1],masks,adj,s['env'])
    C.seed_all(s['seed']);ref=factory(model,'mobius',y.shape[-1],masks,adj,s['env'])
    assert all(torch.equal(a,b) for a,b in zip(net.parameters(),ref.parameters()))
    parameters=sum(a.numel() for a in net.parameters());assert parameters==sum(a.numel() for a in ref.parameters())
    del ref;torch.cuda.empty_cache()
    with torch.no_grad():
        zz,rr=forward(net,torch.tensor(flat[train_flat[:4]],device='cuda'),model)
        parity=float((rr-decode(net,zz,model)).abs().max());assert parity<1e-6
    optimizer=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(s['seed']+17)
    start=time.time();torch.cuda.reset_peak_memory_stats();updates=p['backend_updates'][s['env']]
    for step in range(1,updates+1):
        ix=rng.choice(train_flat,p['batch'],replace=False);q=torch.tensor(flat[ix],device='cuda');zz,r=forward(net,q,model)
        loss=(r-q).square().mean()+C._variance_floor(zz)+.02*C._offdiag_cov(zz)
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
        if step==1 or step%100==0:progress('representation',update=step,updates=updates,loss=float(loss.detach()))
    seconds=time.time()-start;net.eval();active=sum(q.numel() for q in net.parameters() if q.grad is not None)
    z=np.zeros((*y.shape[:2],16),np.float32);zf=z.reshape(-1,16);all_valid=np.flatnonzero(valid.reshape(-1))
    with torch.no_grad():
        for lo in range(0,len(all_valid),64):
            ix=all_valid[lo:lo+64];zf[ix]=forward(net,torch.tensor(flat[ix],device='cuda'),model)[0].cpu().numpy()
    torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},mean=mu,std=sd,masks=masks,adjacency=adj,model=model,variant=variant,seed=s['seed']),out/'representation.pt')
    progress('history_start',updates=p['policy_updates'])
    if s['env']=='mpe':
        from training.history_mpe import train_history_policy
        h=train_history_policy(x,z,n,s['seed'],p['policy_updates'],out/'policy',progress)
    else:
        from training.representation_mamujoco import train_history_policy
        method={'graph':'edge_cara_mobius_graph','simple':'edge_cara_mobius_simple','mif':'edge_cara_mif'}[model]
        h=train_history_policy(x,z,m,n,s['seed'],p['policy_updates'],out/'policy',progress,method=method)
    with np.load(out/'policy/latents.npz',allow_pickle=False) as f:hz=f['deployment_z']
    ck=torch.load(out/'policy/policy.pt',map_location='cpu',weights_only=False)
    progress('transfer_diagnostics')
    offset=1 if s['env']=='mpe' else 0;times=np.arange(valid.shape[1])+offset
    support=valid & (times+3<=m.shape[1])[None] & m[:,np.minimum(times+2,m.shape[1]-1)]
    diagnostic=measure(net,model,y,support,n,z,hz,masks,edge_dim,s['seed'],ck)
    return dict(history=h,transfer=diagnostic,parameters=parameters,active_gradient_parameters=active,
                representation_seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                decoder_parity=parity,paired_initialization_equal=True,valid_training_transitions=len(train_flat),
                normalization_digest=__import__('hashlib').sha256(mu.tobytes()+sd.tobytes()).hexdigest(),
                frozen_before_grounding=True,native_action_labels_read=0,simulator_queries=0)
