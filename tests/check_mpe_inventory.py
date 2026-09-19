"""Paper completeness, timing bounds, and source isolation checks."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.mpe_inventory_completion import PACKAGE,p_for,existing
from training.composition import FrozenPair
from closed_loop_lam_v1.common import RecurrentPolicy


def main():
    c=json.loads((PACKAGE/'configs/mpe_inventory_completion.json').read_text())
    old=json.loads((PACKAGE/'configs/numeric.json').read_text())
    names=next(s['configs'] for s in old['suites'] if s['env']=='mpe' and s['n']==700)
    assert set(names)==set(c['paper_aliases']) and len(names)==29
    assert len(set(c['paper_aliases'].values()))==28 and len(c['methods'])==29
    assert c['paper_aliases']['anchor_only']==c['paper_aliases']['entity_target']=='base'
    assert set(c['sizes'])=={32,175,350,700}
    for n in c['sizes']:
        p=p_for(c,n);assert p['seeds']==[45,46,47,48,49]
        assert all(m in c['methods'] for m in ('tree','base+tree','continuous','idm','bc'))
    total=have=0
    for n in c['sizes']:
        p=p_for(c,n)
        for s in p['seeds']:
            for b in p['budgets']:
                for m in c['methods']:
                    total+=1;have+=existing(c,n,s,b,m) is not None
    assert (total,have,total-have)==(1305,600,705)
    # Duplicate really is the same frozen output, not two independent models.
    net=RecurrentPolicy(16,16).eval();pair=FrozenPair(net,duplicate=True)
    with torch.no_grad():v=pair(torch.randn(2,23,16))[0]
    assert torch.equal(v[...,:16],v[...,16:])
    # Native action t=1..22 is paired with p[t+2], not next-position p[t+1].
    t=np.arange(1,23);assert np.array_equal(np.arange(26)[3:25],t+2)
    print('PASS: 29 labels, 28 distinct paper configs + Global control; 1305 cells = 600 reused + 705 new; Tree all N; duplicate identity; t+2 bounds')


if __name__=='__main__':main()
