"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import argparse, hashlib, itertools, json, os, subprocess, sys, time, traceback

from pathlib import Path

import numpy as np

from utils.atomic import atomic_json

from experiments.stage_artifacts import read, digest, location, progress, verify_sources

PACKAGE=Path(__file__).resolve().parents[1]

WORKSPACE=PACKAGE.parents[1]

def jobs(c,phase,kind):
    seeds=c['seeds'][:1] if phase=='smoke' else c['seeds']
    modes=['frozen','trainable'] if phase=='parity' else ['baseonly','auxonly']
    return [(phase,'visual',s,f'{f}_aux_pretrained_{m}',kind) for s in seeds for f in c['families'] for m in modes]

def reference(c,seed,family,mode):
    assert mode in ('frozen','trainable')
    if mode=='frozen' or family=='mif':
        from training.visual_branch_attribution import base_arm
        arm=base_arm(f'{family}_aux_pretrained_frozen') if mode=='frozen' else 'mif_aux_pretrained_trainable'
        rows=read(PACKAGE/c['inventory']/'summary.json')['records']
        result=Path(next(r['source'] for r in rows if r['seed']==seed and r['budget']==2 and r['arm']==arm))
    else:
        root=PACKAGE/c['pilot' if family=='laom' else 'extension']
        result=root/'formal/visual'/f'seed{seed}'/f'evaluate_{family}_aux_pretrained_trainable'/'result.json'
    ground=result.parent.parent/result.parent.name.replace('evaluate_','ground_',1)
    return ground,result

def validate(out,job):
    d=location(out,job);r=read(d/'result.json')
    assert r['complete'] and r['job']==list(job) and not read(d/'access_audit.json')['violations']
    if job[-1]=='ground':
        assert r['native_action_labels_read']==400 and r['partner_labels_read']==0 and r['simulator_queries']==0
        mode=job[3].split('_')[-1]
        assert r['base_trainable']==(mode in ('baseonly','trainable'))
        assert r['aux_trainable']==(mode in ('auxonly','trainable'))
        for name in ['base','aux']:
            assert (r['branch_initial'][name]!=r['branch_final'][name])==r[name+'_trainable']
    return r

def worker(out,job,c):
    import torch
    from closed_loop_lam_v1 import common as C
    from training import visual_label_budget as B
    from training import visual_branch_attribution as T
    phase,_,seed,arm,kind=job;dest=location(out,job);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);C.seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    start=time.time()
    if kind=='ground':
        B.configure(seed,dest.parent,2,updates=2 if phase=='smoke' else c['updates'])
        B.V.progress=lambda phase,**kw:atomic_json(dest/'progress.json',dict(phase=phase,**kw))
        r=T.ground(dest,arm)
    else:
        ground=location(out,(*job[:-1],'ground'));validate(out,(*job[:-1],'ground'))
        if phase=='formal':assert digest(ground/'decoder.pt')==read(out/'ground_freeze.json')[str((ground/'decoder.pt').relative_to(out))]
        from training import visual_supervision_control as S
        from evaluation.visual_label_budget import evaluate
        B.V.controller=S.controller
        seeds=[908601] if phase in ('parity','smoke') else c['evaluation_seeds']
        r=evaluate(dest,arm,seed,dest.parent,PACKAGE/c['inventory']/'assets',seeds)
    r.update(complete=True,job=job,seconds=time.time()-start)
    atomic_json(dest/'result.json',r)

def pairing(out,c):
    checks=[];keys=['history_initial','decoder_initial','normalization','inputs','targets','batches','total_history_parameters','decoder_parameters','base_parameters','aux_parameters']
    for seed in c['seeds']:
        architecture=[]
        for family in c['families']:
            identities=[]
            for mode in ['frozen','baseonly','auxonly','trainable']:
                phase='parity' if mode in ('frozen','trainable') else 'formal'
                d=location(out,(phase,'visual',seed,f'{family}_aux_pretrained_{mode}','ground'))
                r=read(d/'training_identity.json');identities.append(r)
                for branch in ['base','aux']:assert (r['branch_initial'][branch]!=r['branch_final'][branch])==r[branch+'_trainable']
            for key in keys:assert all(x[key]==identities[0][key] for x in identities),(seed,family,key)
            architecture.append(identities[0]);checks.append(dict(seed=seed,family=family,passed=True))
        # Across families, the control has identical architecture, target labels,
        # observation inputs and minibatches, but its own label-free latent scale.
        for key in ['total_history_parameters','decoder_parameters','base_parameters','aux_parameters','decoder_initial','inputs','targets','batches']:
            assert all(x[key]==architecture[0][key] for x in architecture),(seed,key)
        assert all(x['branch_initial']['base']==architecture[0]['branch_initial']['base'] for x in architecture)
    assert len(checks)==25
    atomic_json(out/'pairing_checks.json',dict(passed=True,cells=checks,capacity_matched_families=c['families']))

def stats(values,**fields):
    from scipy.stats import t
    d=np.array(values);assert len(d)==5
    rad=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5));null=np.abs(np.array(list(itertools.product([-1,1],repeat=5)))@d/5)
    return dict(**fields,mean=float(d.mean()),seed_differences=d.tolist(),positive_seeds=int((d>0).sum()),t95=[float(d.mean()-rad),float(d.mean()+rad)],exact_p=float((null>=abs(d.mean())-1e-12).mean()))

def holm(rows):
    bound=0
    for i,r in enumerate(sorted(rows,key=lambda r:r['exact_p'])):
        bound=max(bound,min(1.,r['exact_p']*(len(rows)-i)));r['holm_p']=bound

def aggregate(out,c):
    records=[]
    for seed,family,mode in itertools.product(c['seeds'],c['families'],['frozen','baseonly','auxonly','trainable']):
        reused=mode in ('frozen','trainable')
        if reused:_,path=reference(c,seed,family,mode);r=read(path)
        else:
            job=('formal','visual',seed,f'{family}_aux_pretrained_{mode}','evaluate');r=validate(out,job);path=location(out,job)/'result.json'
        assert [x['seed'] for x in r['rows']]==c['evaluation_seeds']
        value=float(np.mean([x['return_value'] for x in r['rows'] if x['seed'] in c['primary_evaluation_seeds']]))
        records.append(dict(seed=seed,family=family,mode=mode,mean_return=value,reused=reused,source=str(path)))
    assert len(records)==len({(r['seed'],r['family'],r['mode']) for r in records})==100
    value=lambda s,f,m:next(r['mean_return'] for r in records if (r['seed'],r['family'],r['mode'])==(s,f,m))
    contrasts=[]
    formulas={'base_update':{'baseonly':1,'frozen':-1},'aux_update':{'auxonly':1,'frozen':-1},
        'aux_update_given_base':{'trainable':1,'baseonly':-1},'base_update_given_aux':{'trainable':1,'auxonly':-1},
        'interaction':{'trainable':1,'baseonly':-1,'auxonly':-1,'frozen':1}}
    for family in c['families']:
        rows=[stats([sum(w*value(s,family,m) for m,w in weights.items()) for s in c['seeds']],family=family,name=name,test_family='branch_'+family) for name,weights in formulas.items()]
        holm(rows);contrasts+=rows
    rows=[]
    for family,mode in itertools.product(c['mte_families'],['frozen','baseonly','auxonly','trainable']):
        rows.append(stats([value(s,family,mode)-value(s,'laom',mode) for s in c['seeds']],family=family,name='minus_laom_'+mode,test_family='matched_non_mte_control'))
    holm(rows);contrasts+=rows
    verify_sources(out)
    for f,h in read(out/'ground_freeze.json').items():assert digest(out/f)==h,f
    for f in out.rglob('access_audit.json'):assert not read(f)['violations'],f
    check=subprocess.run([sys.executable,str(WORKSPACE/'tools/verify_frozen_foundation.py')],capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
    atomic_json(out/'foundation_verification.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
    atomic_json(out/'summary.json',dict(records=records,contrasts=contrasts,new_results=50,reused_results=50,source_and_weight_checks=True))
    lines=['# 视觉MTE分支归因：B2五种子探索性检验','','## 四格平均回报','','|模型|都冻结|只更新Base|只更新辅助history|两支都更新|','|---|---:|---:|---:|---:|']
    for f in c['families']:lines.append('|'+f+'|'+'|'.join(f'{np.mean([value(s,f,m) for s in c["seeds"]]):.5f}' for m in ['frozen','baseonly','auxonly','trainable'])+'|')
    lines+=['','## 全部预定比较','','|模型|比较|均值差|正向种子|t95|精确p|Holm p|','|---|---|---:|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['family']}|{r['name']}|{r['mean']:+.5f}|{r['positive_seeds']}/5|{r['t95']}|{r['exact_p']:.4f}|{r['holm_p']:.4f}|")
    lines+=['','Base为原visual_laom_target history；辅助分支为MTE或同16维、同参数history架构的LAOM state-adapter，不是官方端到端像素LAOM复现。','所有格均训练decoder；选择性更新只改变history梯度访问。每格400目标动作标签、600更新、相同采样，原前端及预训练原件不改。训练不访问模拟器或伙伴动作。跨方法使用同一种无动作归一化规则，但各自latent统计数值不同。','本轮是观察已有收益后的固定预算探索；MIF为原锁定主模型，Simple/Graph/Tree为家族跟进。分支五比较按每模型Holm族，16项MTE减LAOM组成单独Holm族。重复单位为五上游种子，p最低.0625。所有方向完整保留，不按结果改假设或挑选模型。','分支选择性更新定位适配收益，不等价于删除分支信息。与同容量LAOM相比估计MTE来源history在本任务、预算和训练方式下的相对价值，不单独证明matching构造、Mobius或通用因果机制。原冻结主结果保留，论文图表不自动修改。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=PACKAGE/'results'/c['name'];dest.mkdir(parents=True,exist_ok=False)
    import shutil
    for n in ['RESULTS_CN.md','summary.json','protocol.json','qualification.json','pairing_checks.json','foundation_verification.json']:shutil.copy2(out/n,dest/n)
