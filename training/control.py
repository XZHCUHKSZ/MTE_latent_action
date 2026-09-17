"""Final numeric grounding, extracted as an explicit API from scale/recovery workers.

Call after all observation-trained sources are frozen and a separate budgeted
label export exists. This module does not perform simulator evaluation.
"""
import numpy as np
import torch
from .composition import restore,FrozenPair

def fit(env,x,valid,actions,ids,seed,method,primary_checkpoint=None,aux_checkpoint=None,duplicate=False):
    """Return a frozen-history grounded policy or same-budget BC/IDM policy.

    x contains concatenated train/dev observations. ids address the B labelled
    train episodes. For Aux, primary_checkpoint is Base and aux_checkpoint is
    the named MTE policy. Non-Aux passes only its own primary checkpoint.
    """
    if env=='mpe':from . import grounding_mpe as G
    elif env=='mamujoco':from . import grounding_mamujoco as G
    else:raise ValueError(env)
    ids=np.asarray(ids);b=len(ids);val=np.arange(0,b,4);fit_ids=np.setdiff1d(np.arange(b),val)
    if not np.array_equal(ids,np.arange(b)):raise ValueError('Formal labels use the first B local train episodes')
    if actions.shape[0]!=b or actions.shape[-1]!=2:raise ValueError('Wrong target-action export')
    updates=1000 if env=='mpe' else 800
    decoder=None
    if method in {'bc_scarce','bc_full'}:
        if env=='mpe':
            y=np.zeros((len(x),22,2),np.float32);y[ids[fit_ids]]=actions[fit_ids]
            net,mean,std,diag=G.train_supervised_history(x,y,ids[fit_ids],x[ids[val]],actions[val],seed,3200)
        else:net,mean,std,diag=G.train_bc(x,valid,actions,ids,fit_ids,val,seed,3200)
    elif method=='idm_relabel':
        if env=='mpe':net,mean,std,diag=G.train_idm(x,ids,actions,fit_ids,val,seed,updates,1200)
        else:net,mean,std,diag=G.train_idm(x,valid,actions,ids,fit_ids,val,seed,updates,1200)
    else:
        if primary_checkpoint is None:raise ValueError('Pass the frozen history checkpoint')
        net,ck=restore(primary_checkpoint)
        if aux_checkpoint is not None:
            aux,ac=restore(aux_checkpoint)
            assert np.array_equal(ck['obs_mean'],ac['obs_mean']) and np.array_equal(ck['obs_std'],ac['obs_std'])
            net=FrozenPair(net,aux)
        elif duplicate:net=FrozenPair(net,duplicate=True)
        mean,std=ck['obs_mean'],ck['obs_std'];length=23 if env=='mpe' else 200
        with torch.no_grad():z=net(torch.as_tensor((x[ids,:length]-mean)/std))[0].numpy()
        if env=='mpe':
            z=z[:,1:];decoder,diag=G.train_decoder(z,actions,fit_ids,val,seed,updates)
        else:
            z=np.where(valid[ids,...,None],z,0.)
            decoder,diag=G.train_decoder(z,actions,valid[ids],fit_ids,val,seed,updates)
        diag.update(latent_dimension=z.shape[-1],decoder_parameters=sum(p.numel() for p in decoder.parameters()))
    return G.GroundedPolicy(net,mean,std,decoder),diag
