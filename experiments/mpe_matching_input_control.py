"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""
from mte.method_names import report_text

import json

import sys

import time

import numpy as np

import torch

from experiments.mpe_temporal_stages import PACKAGE,digest,ground

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
    (out/'RESULTS_CN.md').write_text(report_text(lines, "control")+'\n',encoding='utf8')
