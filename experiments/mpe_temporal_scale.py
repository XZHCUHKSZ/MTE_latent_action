"""Full-scale scheduler: isolated seed/budget outputs, four workers, fixed plan.

No scientific model is defined here. Reuses the tested temporal-repair stages.
New training RNG seeds can be mapped explicitly onto archived observation splits.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from experiments.mpe_temporal_repair import PACKAGE, digest, exports, pretrain, ground
from evaluation.temporal_scale_report import summarize
from utils.atomic import atomic_json


def source_hashes():
    return {str(f.relative_to(PACKAGE)):digest(f)
        for sub in ['mte','training','experiments','evaluation','environments','utils','closed_loop_lam_v1']
        for f in (PACKAGE/sub).glob('*.py')}


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


def manage(args,p):
    args.out.mkdir(parents=True,exist_ok=True)
    lock=args.out/'manager.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    started=time.time()
    try:
        protocol=args.out/'protocol.json'
        if protocol.exists():assert json.loads(protocol.read_text())==p,'Changed protocol on resume'
        else:atomic_json(protocol,p)
        hashes=source_hashes(); manifest=args.out/'implementation_sources.json'
        if manifest.exists():assert json.loads(manifest.read_text())==hashes,'Changed implementation on resume'
        else:atomic_json(manifest,hashes)
        exports(args.workspace,args.out,p)
        # Limit scheduling changes to concurrency; never alter a scientific parameter.
        def run(stage,seed,budget=None):
            root,name,complete=paths(args.out,stage,seed,budget)
            if complete.exists():return
            if stage=='pretrain' and (root/'pretrain').exists():
                raise RuntimeError('Incomplete pretraining retained. Inspect failure before resuming; no silent restart/overwrite.')
            cmd=[sys.executable,'-m','experiments.mpe_temporal_scale','--workspace',str(args.workspace),
                 '--out',str(args.out),'--config',str(args.config),'--stage',stage,'--seed',str(seed)]
            if budget is not None:cmd += ['--budget',str(budget)]
            with (root/f'{name}.log').open('a',encoding='utf-8') as log:
                subprocess.run(cmd,cwd=PACKAGE,stdout=log,stderr=subprocess.STDOUT,check=True,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for stage in ('pretrain','ground','physical'):
            jobs=[(stage,seed,b) for b in p['budgets'] for seed in p['seeds']] if stage=='ground' else [(stage,seed,None) for seed in p['seeds']]
            concurrency=p['parallel_pretrain'] if stage=='pretrain' else p['parallel_ground']
            phase_start=time.time(); finished=0
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending={pool.submit(run,*job):job for job in jobs}
                while pending:
                    done,_=concurrent.futures.wait(pending,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in done:
                        try:f.result()
                        except BaseException:
                            for queued in pending:queued.cancel()
                            raise
                        pending.pop(f);finished+=1
                    rows,total=summarize(args.out,p)
                    atomic_json(args.out/'status.json',dict(status='running',stage=stage,
                        phase_jobs_completed=finished,phase_jobs_total=len(jobs),parallel_limit=concurrency,
                        completed_control_rows=rows,expected_control_rows=total,
                        elapsed_seconds=time.time()-started,phase_elapsed_seconds=time.time()-phase_start,time=time.time()))
            if stage=='pretrain':
                atomic_json(args.out/'global_freeze.json',dict(all_pretraining_complete=True,
                    training_seeds=p['seeds'],time=time.time()))
                exports(args.workspace,args.out,p,labels=True)
        rows,total=summarize(args.out,p);assert rows==total
        assert source_hashes()==hashes,'Implementation changed during execution'
        for seed in p['seeds']:
            frozen=args.out/f'seed{seed}/pretrain'
            weights=json.loads((frozen/'complete.json').read_text())['checkpoints']
            assert all(digest(frozen/f)==h for f,h in weights.items())
        atomic_json(args.out/'status.json',dict(status='complete',stage='complete',completed_control_rows=rows,
            expected_control_rows=total,source_and_frozen_weight_checks=True,elapsed_seconds=time.time()-started,time=time.time()))
    except BaseException:
        atomic_json(args.out/'status.json',dict(status='failed',traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:lock.unlink(missing_ok=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=PACKAGE/'configs/mpe_temporal_full_scale.json')
    ap.add_argument('--stage',choices=['pretrain','ground','physical']);ap.add_argument('--seed',type=int);ap.add_argument('--budget',type=int)
    args=ap.parse_args();args.workspace=args.workspace.resolve();args.out=args.out.resolve();args.config=args.config.resolve()
    p=json.loads(args.config.read_text())
    if args.stage:worker(args,p)
    else:manage(args,p)


if __name__=='__main__':main()
