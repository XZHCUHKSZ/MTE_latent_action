"""Stage-isolated fixed-budget matching controls. No changes to old runs."""
import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from experiments.mpe_temporal_repair import PACKAGE,digest,ground
from utils.atomic import atomic_json


def worker(args,p):
    root=args.out/f'seed{args.seed}';p=dict(p,seed=args.seed)
    torch.set_num_threads(p['threads_per_process'])
    def progress(stage,**kw):
        atomic_json(root/f'progress_{args.stage}.json',dict(stage=stage,time=time.time(),**kw))
    if args.stage=='pretrain':
        from utils.access import guard
        src=PACKAGE/p['source_run']/f'seed{args.seed}'
        allowed=[root/'data/train.npz',root/'data/dev.npz',src/'pretrain/offset2/bridge/endpoints.npz']
        audit=guard(root/'pretrain',allowed,pretraining=True)
        def deny_sim(event,aa):
            if event=='import' and aa[0].split('.')[0] in {'mpe2','pettingzoo','mujoco','gymnasium'}:raise RuntimeError('No simulators in pretraining')
        sys.addaudithook(deny_sim)
        from training.matching_input_control import train
        from training.composition import restore
        with np.load(allowed[0]) as f:x=f['positions']
        with np.load(allowed[1]) as f:x=np.concatenate([x,f['positions']])
        n=p['train_episodes']
        with np.load(allowed[2]) as f:a,b,masks=f['actual'],f['reference'],f['masks']
        if p.get('smoke'):
            ids=np.r_[np.arange(n),np.arange(700,700+p['dev_episodes'])]
            a,b=a[ids],b[ids]
        assert len(x)==n+p['dev_episodes'] and len(a)==len(x)
        infos=[]
        for arm in p['arms']:
            dest=root/'pretrain'/arm;dest.mkdir()
            infos.append(train(x,a,b,masks,n,arm,p,dest,progress))
            net,ck=restore(dest/'policy/policy.pt')
            with torch.no_grad():
                v=torch.tensor((x[:2,:23]-ck['obs_mean'])/ck['obs_std']);q=v.clone();q[:,13:]+=17
                assert torch.equal(net(v)[0][:,:13],net(q)[0][:,:13])
        for key in ['parameters','init_hash','normalization_hash','target_hash','batch_hash']:
            assert len({i[key] for i in infos})==1,key
        assert not audit['violations']
        atomic_json(root/'pretrain/access_audit.json',audit)
        atomic_json(root/'pretrain/complete.json',dict(checkpoints={str(f.relative_to(root/'pretrain')):digest(f) for f in (root/'pretrain').rglob('*.pt')},
            paired_gates_passed=True,native_action_labels_read=0,simulator_queries=0))
    else:
        assert (args.out/'global_freeze.json').exists()
        ground(root,dict(p,control_configs=['base+'+a for a in p['arms']],primary_contrasts=[]),progress)
    progress('complete')


def aggregate(out,p):
    from scipy.stats import t
    import itertools
    rows=[]
    for seed in p['seeds']:
        rs=json.loads((out/f'seed{seed}/grounding/results.json').read_text())
        assert len(rs)==len(p['arms'])
        rows+=rs
    identities={(r['seed'],r['budget'],r['method']) for r in rows}
    assert len(identities)==len(rows)==len(p['seeds'])*len(p['arms'])
    contrasts=[]
    for plus,minus in p['primary_contrasts']:
        d=np.array([next(r['mean_return'] for r in rows if r['seed']==s and r['method']==plus)-next(r['mean_return'] for r in rows if r['seed']==s and r['method']==minus) for s in p['seeds']])
        radius=float(t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None
        null=np.abs(np.array(list(itertools.product([-1,1],repeat=len(d))))@d/len(d))
        contrasts.append(dict(plus=plus,minus=minus,mean_delta=float(d.mean()),seed_deltas=d.tolist(),
            ci95=None if radius is None else [float(d.mean()-radius),float(d.mean()+radius)],
            exact_p=float((null>=abs(d.mean())-1e-15).mean())))
    order=sorted(range(len(contrasts)),key=lambda i:contrasts[i]['exact_p']);bound=0
    for j,i in enumerate(order):bound=max(bound,min(1.,(len(order)-j)*contrasts[i]['exact_p']));contrasts[i]['holm_p']=bound
    source=json.loads((out/'source_manifest.json').read_text())
    assert all(digest(f)==v for f,v in source.items())
    freeze=json.loads((out/'global_freeze.json').read_text())
    assert all(digest(out/f)==v for f,v in freeze.items())
    means=[dict(method='base+'+a,mean_return=float(np.mean([r['mean_return'] for r in rows if r['method']=='base+'+a]))) for a in p['arms']]
    atomic_json(out/'summary.json',dict(rows=rows,means=means,contrasts=contrasts,source_and_weight_checks=True,scope=p['scope']))
    lines=['# 同资源 matching 输入控制','',''+p['scope'],'','|方法|平均控制回报 ↑|','|---|---:|']
    for r in means:lines.append(f"|{r['method']}|{r['mean_return']:.6f}|")
    lines+=['','|比较|回报差|t95|精确p|Holm p|','|---|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['plus']} − {r['minus']}|{r['mean_delta']:+.6f}|{r['ci95']}|{r['exact_p']}|{r['holm_p']}|")
    lines+=['','五种子的双侧精确检验最小p为0.0625。保留全部比较，不将均值正转写为稳定显著优势。',
        '这是Graph类的等资源机制控制，输入为两个8维端点向量；不是主实验原始8维edge-only Graph的替换值。端点预测目标对各臂完全相同，没有给matched额外标签。',
        '全部输入保持相同信息，差分是可逆变换。实验回答显式matching的归纳偏置，而非声称只有matching拥有更多信息。',
        '未更新论文图表，未上传GitHub。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def manage(args,p):
    args.out.mkdir(parents=True,exist_ok=False)
    atomic_json(args.out/'protocol.json',p)
    hashes={str(f):digest(f) for folder in ['experiments','training','evaluation','environments','mte','utils','closed_loop_lam_v1'] for f in (PACKAGE/folder).glob('*.py')}
    for seed in p['seeds']:
        src=PACKAGE/p['source_run']/f'seed{seed}';root=args.out/f'seed{seed}'
        (root/'data').mkdir(parents=True);(root/'pretrain/base/policy').mkdir(parents=True)
        for split in ['train','dev']:
            f=src/f'data/{split}.npz';hashes[str(f)]=digest(f)
            if p.get('smoke'):
                with np.load(f) as ff:v=ff['positions'][:p['train_episodes'] if split=='train' else p['dev_episodes']]
                np.savez_compressed(root/f'data/{split}.npz',positions=v)
            else:shutil.copy2(f,root/f'data/{split}.npz')
        f=src/'pretrain/base/policy/policy.pt';hashes[str(f)]=digest(f);shutil.copy2(f,root/'pretrain/base/policy/policy.pt')
        f=src/'pretrain/offset2/bridge/endpoints.npz';hashes[str(f)]=digest(f)
        # Verify copied frozen source weights against their original registry.
        freeze={Path(k).as_posix():v for k,v in json.loads((src/'pretrain/complete.json').read_text())['checkpoints'].items()}
        assert digest(src/'pretrain/base/policy/policy.pt')==freeze['base/policy/policy.pt']
    atomic_json(args.out/'source_manifest.json',hashes)
    def run(stage,seed):
        root=args.out/f'seed{seed}'
        cmd=[sys.executable,'-X','utf8','-m','experiments.mpe_matching_input_control','--out',str(args.out),'--stage',stage,'--seed',str(seed)]
        with (root/f'{stage}.stdout.log').open('x') as so,(root/f'{stage}.stderr.log').open('x') as se:
            proc=subprocess.Popen(cmd,cwd=PACKAGE,stdout=so,stderr=se)
            atomic_json(root/f'{stage}_process.json',dict(pid=proc.pid,command=cmd,started_at=time.time()))
            ret=proc.wait()
            if ret:raise RuntimeError(f'{stage} seed{seed} failed with code {ret}; new dispatch stopped')
    def stage(name,parallel):
        queue=iter(p['seeds']);done=0
        with futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            active={pool.submit(run,name,s):s for s in [next(queue,None) for _ in range(parallel)] if s is not None}
            while active:
                finished,_=futures.wait(active,return_when=futures.FIRST_COMPLETED)
                for f in finished:f.result();active.pop(f);done+=1
                atomic_json(args.out/'status.json',dict(status='running',stage=name,completed_seeds=done,total_seeds=len(p['seeds']),pid=os.getpid(),time=time.time()))
                for _ in finished:
                    s=next(queue,None)
                    if s is not None:active[pool.submit(run,name,s)]=s
    try:
        atomic_json(args.out/'status.json',dict(status='running',stage='pretrain',pid=os.getpid(),time=time.time()))
        stage('pretrain',p['parallel_pretrain'])
        freeze={str(f.relative_to(args.out)):digest(f) for f in args.out.rglob('*.pt')}
        atomic_json(args.out/'global_freeze.json',freeze)
        # First action-label access occurs only after every history has frozen.
        for seed in p['seeds']:
            src=PACKAGE/p['source_run']/f'seed{seed}'
            f=src/'data/labels_b32.npz';hashes[str(f)]=digest(f)
            shutil.copy2(f,args.out/f'seed{seed}/data/labels_b32.npz')
        atomic_json(args.out/'source_manifest.json',hashes)
        stage('ground',p['parallel_ground'])
        aggregate(args.out,p)
        atomic_json(args.out/'status.json',dict(status='complete',source_and_weight_checks=True,control_units=len(p['arms'])*len(p['seeds'])))
    except BaseException:
        atomic_json(args.out/'failure.json',dict(traceback=traceback.format_exc()))
        atomic_json(args.out/'status.json',dict(status='failed',pid=os.getpid()))
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['pretrain','ground']);ap.add_argument('--seed',type=int);ap.add_argument('--smoke',action='store_true');args=ap.parse_args();args.out=args.out.resolve()
    if args.stage:
        p=json.loads((args.out/'protocol.json').read_text());worker(args,p)
    else:
        p=json.loads((PACKAGE/'configs/mpe_matching_input_control.json').read_text())
        if args.smoke:p.update(smoke=True,seeds=[45],representation_updates=2,history_updates=2,decoder_updates=2,evaluation_episodes=2)
        manage(args,p)

if __name__=='__main__':main()
