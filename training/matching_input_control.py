"""Controlled graph input experiment; uses the existing Graph class unchanged.

All arms retain both endpoint tables through invertible transformations. This
isolates explicit matching as an input inductive bias, not all MTE components.
"""
import hashlib
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from training.route_representation import factory, forward
from mte.structured_controls import complement_indices
from training.history_mpe import train_history_policy
from utils.atomic import atomic_json


def fingerprint(values):
    h=hashlib.sha256()
    for value in values:
        a=np.asarray(value);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()


def views(a,b,n,masks,arm):
    assert a.shape==b.shape and a.ndim==4
    # One pooled endpoint normalization shared by every arm and both endpoints.
    mu,sd=C.fit_mean_std(np.concatenate([a[:n].reshape(-1,8),b[:n].reshape(-1,8)]))
    aa=(a-mu)/sd;bb=(b-mu)/sd
    target=np.concatenate([aa,bb],-1)
    perm=complement_indices(masks)
    if arm=='matched': inputs=np.concatenate([aa,aa-bb],-1)
    elif arm=='context_mismatch': inputs=np.concatenate([aa,aa-np.take(bb,perm,axis=2)],-1)
    elif arm=='endpoints': inputs=target.copy()
    else: raise ValueError(arm)
    recovered=inputs[...,8:]
    if arm=='matched':recovered=inputs[...,:8]-recovered
    elif arm=='context_mismatch':recovered=np.take(inputs[...,:8]-recovered,perm,axis=2)
    np.testing.assert_allclose(recovered,bb,rtol=2e-5,atol=2e-6)
    return inputs,target,mu,sd


def train(x,a,b,masks,n,arm,p,out,progress):
    inp,target,mu,sd=views(a[:,:,2],b[:,:,2],n,masks,arm)
    inputs=inp.reshape(-1,8,16);truth=target.reshape(-1,8,16)
    C.seed_all(p['seed']);net=factory('graph','mobius',16,masks,None,'mpe')
    init=fingerprint([v.detach().cpu().numpy() for v in net.parameters()])
    params=sum(v.numel() for v in net.parameters())
    opt=torch.optim.Adam(net.parameters(),lr=1e-3);rng=np.random.default_rng(p['seed']+17)
    batches=hashlib.sha256()
    for step in range(1,p['representation_updates']+1):
        ids=rng.integers(n*22,size=p['representation_batch']);batches.update(ids.tobytes())
        v=torch.tensor(inputs[ids],device='cuda');y=torch.tensor(truth[ids],device='cuda')
        z,reco=forward(net,v,'graph')
        loss=(reco-y).square().mean()+C._variance_floor(z)+.02*C._offdiag_cov(z)
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        if step==1 or step%100==0 or step==p['representation_updates']:
            progress('representation',arm=arm,update=step,updates=p['representation_updates'],loss=float(loss.detach()))
    net.eval();encoded=[]
    with torch.no_grad():
        for lo in range(0,len(inputs),64):
            encoded.append(forward(net,torch.tensor(inputs[lo:lo+64],device='cuda'),'graph')[0].cpu().numpy())
    z=np.concatenate(encoded).reshape(len(x),22,16)
    assert np.isfinite(z).all()
    torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},mean=mu,std=sd,masks=masks,
        arm=arm,source_class=type(net).__name__,input_dim=16,objective='shared normalized endpoint-pair reconstruction'),out/'representation.pt')
    del net;torch.cuda.empty_cache()
    history=train_history_policy(x,z,n,p['seed'],p['history_updates'],out/'policy',progress)
    result=dict(arm=arm,parameters=params,init_hash=init,normalization_hash=fingerprint([mu,sd]),
        target_hash=fingerprint([target]),batch_hash=batches.hexdigest(),history=history,
        native_action_labels_read=0,simulator_queries=0,input_dimension=16,latent_dimension=16,
        caveat='Controlled Graph16-input adapter, not the original edge-only Graph8-input performance row. All three inputs invertibly retain both endpoint tables; this is not a no-information control.')
    atomic_json(out/'training.json',result)
    return result
