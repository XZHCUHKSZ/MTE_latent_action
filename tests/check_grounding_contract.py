"""CPU test of label budget, deployment feature slicing and final update routing.

Actual frozen GRUs are used; train_decoder is replaced with a recording stub so
this check cannot train new models or touch an environment.
"""
import sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from closed_loop_lam_v1.common import RecurrentPolicy,ActionDecoder
from training.control import fit
from training.composition import FrozenPair

def run():
    rng=np.random.default_rng(9717);cases=[]
    for env,dim,steps,updates in [('mpe',16,23,1000),('mamujoco',105,200,800)]:
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);torch.manual_seed(17)
            base=RecurrentPolicy(dim,16).eval();aux=RecurrentPolicy(dim,16).eval()
            for name,net in [('base',base),('aux',aux)]:
                torch.save(dict(state_dict=net.state_dict(),obs_mean=np.zeros(dim,np.float32),obs_std=np.ones(dim,np.float32),z_dim=16,frozen_before_grounding=True,native_action_labels_read=0),p/(name+'.pt'))
            x=rng.normal(size=(10,steps+1,dim)).astype(np.float32)
            mask=np.ones((10,steps),bool);mask[0,-3:]=False
            actions=rng.normal(size=(8,22 if env=='mpe' else 200,2)).astype(np.float32)
            records=[]
            def decoder(*args):records.append(args);return ActionDecoder(32,2),{}
            module='training.grounding_'+env
            with patch(module+'.train_decoder',decoder):
                policy,diag=fit(env,x,mask,actions,np.arange(8),4,'anchor_graph',p/'base.pt',p/'aux.pt')
            args=records[0]
            with torch.no_grad():expected=FrozenPair(base,aux)(torch.from_numpy(x[:8,:steps]))[0].numpy()
            if env=='mpe':expected=expected[:,1:]
            else:expected=np.where(mask[:8,...,None],expected,0.)
            assert np.array_equal(args[0],expected)
            offset=0 if env=='mpe' else 1
            assert np.array_equal(args[2+offset],[1,2,3,5,6,7])
            assert np.array_equal(args[3+offset],[0,4])
            assert args[-1]==updates and diag['latent_dimension']==32
            cases.append(dict(environment=env,decoder_updates=updates,feature_parity=True,fit=6,validation=2))
    return cases

if __name__=='__main__':print(run())
