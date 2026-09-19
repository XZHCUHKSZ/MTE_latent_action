"""Actual interfaces and scientific bookkeeping, not expected win assertions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from training.temporal_completion_mpe import edge_view
from training.route_representation import factory
from evaluation.temporal_donor_effects import donor_bank
from experiments.mpe_evidence_completion import plan,completion_path
from closed_loop_lam_v1 import common as C


def main():
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/mpe_evidence_completion.json').read_text())
    x=np.broadcast_to(np.arange(26)[None,:,None],(2,26,16)).astype(np.float32).copy()
    a=np.stack([np.broadcast_to(x[:,1+h:23+h,None,:8],(2,22,8,8)) for h in (1,2,3)],2)
    b=a-.25;m=np.zeros((8,4),bool)
    for k in range(8):m[k,1:]=[(k>>j)&1 for j in range(3)]
    for h in (2,3):
        v=edge_view(x,a,b,m,h)
        assert np.allclose(v['cf_context_root'][...,:8],a[:,:,h-1])
        assert not v['actions'].any()
        assert np.allclose((v['cf_context_root']-v['cf_context_ego_null'])[...,8:],0)
    C.seed_all(45);a=factory('simple','mobius',8,m,None,'mpe')
    C.seed_all(45);b=factory('simple','raw_coordinates',8,m,None,'mpe')
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert torch.equal(b.mobius,torch.eye(8,device='cuda'))
    pf=plan(c,700);ps=plan(c,32)
    assert len(pf['seeds'])*len(pf['budgets'])*len(pf['control_configs'])==240
    assert len(ps['seeds'])*len(ps['budgets'])*len(ps['control_configs'])*len(c['scales'])==90
    assert ps['budgets']==[32] and pf['seeds']==[45,46,47,48,49]
    bank=donor_bank(32,dict(seed=45,readout_updates=3,batch=16))
    assert len(set(bank))==704 and bank.max()<704
    assert completion_path(Path('x'),'ground',8)!=completion_path(Path('x'),'ground',16)
    print('PASS: physical-horizon edge view, zero action placeholders, Simple raw parameter parity, 330 new rows, donor-bank bounds and disjoint writes')


if __name__=='__main__':main()
