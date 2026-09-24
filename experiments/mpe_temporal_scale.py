"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import time

import traceback

import torch

from experiments.mpe_temporal_stages import PACKAGE, digest, exports, pretrain, ground

from utils.atomic import atomic_json

def paths(out,stage,seed,budget=None):
    root=out/f'seed{seed}'
    name=stage if budget is None else f'{stage}_b{budget}'
    complete=root/({'pretrain':'pretrain/complete.json','physical':'physical_effects.json'}.get(stage,
                   f'grounding/budget_{budget}/complete.json'))
    return root,name,complete

def worker(args,p):
    p=dict(p,seed=args.seed)
    root,name,complete=paths(args.out,args.stage,args.seed,args.budget)
    torch.set_num_threads(p['threads_per_process'])
    def progress(stage,**kw):
        row=dict(stage=stage,seed=args.seed,budget=args.budget,time=time.time());row.update(kw)
        atomic_json(root/f'status_{name}.json',row)
    try:
        if args.stage=='pretrain':pretrain(root,p,progress)
        elif args.stage=='ground':
            assert args.budget in p['budgets']
            ground(root,dict(p,budgets=[args.budget],_ground_budget=args.budget),progress)
        else:
            from evaluation.temporal_mpe_effects import evaluate
            evaluate(root,p,progress)
        assert complete.exists()
        progress('complete')
    except BaseException:
        atomic_json(root/f'failure_{name}.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
