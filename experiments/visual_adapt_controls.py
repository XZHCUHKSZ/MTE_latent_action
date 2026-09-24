"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""
from mte.method_names import report_text
from mte.method_names import normalize_config, normalize_job, resolve_control

import hashlib

import itertools

import json

from pathlib import Path

import shutil

import subprocess

import sys

import time

import numpy as np

from utils.atomic import atomic_json

P = Path(__file__).resolve().parents[1]

W = P.parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def path(out, job):
    job = normalize_job(job, 3, "control")
    phase, seed, budget, arm, kind = job
    return out/phase/f'seed{seed}'/f'b{budget}'/(kind+'_'+arm)

def jobs(c, phase, kind):
    c = normalize_config(c, "control")
    if phase == 'smoke':
        return [(phase, c['seeds'][0], 1, a+'_pretrained_trainable', kind) for a in c['arms']]
    if phase == 'reproduce':
        return [(phase,s,2,a+'_pretrained_trainable',kind) for s in c['seeds'] for a in c['arms'] if a.startswith(('lapo_', 'laom_'))]
    return [(phase,s,b,a+'_pretrained_'+('frozen' if phase=='parity' else 'trainable'),kind)
            for s in c['seeds'] for b in c['budgets'] for a in c['arms']
            if phase != 'formal' or b != 2 or not a.startswith(('lapo_', 'laom_'))]

def reference(c, seed, budget, arm, adapted=False):
    c = normalize_config(c, "control")
    arm = resolve_control(arm)
    if adapted:
        root=P/c['pilot']/'formal/visual'/f'seed{seed}'
        return root/('ground_'+arm),root/('evaluate_'+arm)/'result.json'
    from training.visual_adapt_controls import base_arm
    name=base_arm(arm)
    r=next(r for r in read(P/c['inventory']/'summary.json')['records']
           if (r['seed'],r['budget'],r['arm'])==(seed,budget,name))
    result=Path(r['source'])
    if budget==8:
        ground=W/'visual_multiagent_2026_09_10/seed5_v1/runtime'/str(seed)/('ground_'+name)
    else:
        ground=result.parent.parent/('ground_'+name)
    assert ground.is_dir() and result.is_file(), (ground,result)
    return ground,result

def validate(out,job):
    job = normalize_job(job, 3, "control")
    d=path(out,job); r=read(d/'result.json')
    assert r['complete'] and r['job']==list(job),job
    assert not read(d/'access_audit.json')['violations'],job
    if job[-1]=='ground':
        assert r['native_action_labels_read']==job[2]*200
        assert r['partner_labels_read']==r['simulator_queries']==0
    return r

def worker(out,job,c):
    c = normalize_config(c, "control")
    job = normalize_job(job, 3, "control")
    import torch
    from closed_loop_lam_v1 import common as C
    from training import visual_adapt_controls as T, visual_label_budget as B
    phase,seed,budget,arm,kind=job
    dest=path(out,job);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    C.seed_all(seed);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    start=time.time()
    if kind=='ground':
        r=T.ground(dest,dest.parent,seed,budget,arm,2 if phase=='smoke' else c['updates'])
    else:
        g=path(out,(*job[:-1],'ground'));validate(out,(*job[:-1],'ground'))
        if phase=='formal':assert digest(g/'decoder.pt')==read(out/'ground_freeze.json')[(g/'decoder.pt').relative_to(out).as_posix()]
        from evaluation.visual_label_budget import evaluate
        B.V.controller=T.controller
        r=evaluate(dest,arm,seed,dest.parent,P/c['inventory']/'assets',
                   c['evaluation_seeds'] if phase=='formal' else [908601])
    r.update(complete=True,job=job,seconds=time.time()-start)
    atomic_json(dest/'result.json',r)

def verify(out):
    for f,h in read(out/'source_manifest.json').items():assert digest(f)==h,f

def mean(result,c):
    assert [r['seed'] for r in result['rows']]==c['evaluation_seeds']
    return float(np.mean([r['return_value'] for r in result['rows'] if r['seed'] in c['primary_evaluation_seeds']]))

def stats(values):
    from scipy.stats import t
    d=np.asarray(values);assert d.shape==(5,) and np.isfinite(d).all()
    mu=float(d.mean());rad=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5))
    null=np.abs(np.array(list(itertools.product([-1,1],repeat=5)))@d/5)
    return dict(mean_delta=mu,seed_deltas=d.tolist(),positive_seeds=int((d>0).sum()),t95=[mu-rad,mu+rad],exact_p=float((null>=abs(mu)-1e-12).mean()))

def holm(rows):
    bound=0
    for i,r in enumerate(sorted(rows,key=lambda r:r['exact_p'])):
        bound=max(bound,min(1.,r['exact_p']*(len(rows)-i)));r['holm_p']=bound

def aggregate(out,c):
    c = normalize_config(c, "control")
    rows=[]
    for s in c['seeds']:
        for b in c['budgets']:
            for arm in c['arms']:
                for route in ('frozen','trainable'):
                    a=arm+'_pretrained_'+route
                    if route=='frozen':_,e=reference(c,s,b,a);reused=True
                    elif b==2 and arm.startswith(('lapo_','laom_')):_,e=reference(c,s,b,a,True);reused=True
                    else:e=path(out,('formal',s,b,a,'evaluate'))/'result.json';reused=False
                    rows.append(dict(seed=s,budget=b,arm=arm,route=route,mean_return=mean(read(e),c),source=str(e),reused=reused))
    inv=read(P/c['inventory']/'summary.json')['records']
    names={'anchor_solo_edge_cara_mif':('mif_solo','frozen'),'anchor_plus_edge_cara_mif':('mif_aux','frozen'),'mif_solo_pretrained_trainable':('mif_solo','trainable'),'mif_aux_pretrained_trainable':('mif_aux','trainable'),'bc_batch256':('bc','supervised'),'bc_idm_relabel':('idm','supervised')}
    for r in inv:
        if r['arm'] in names:
            arm,route=names[r['arm']];e=Path(r['source']);mu=mean(read(e),c)
            assert abs(mu-r['mean_return'])<1e-9
            rows.append(dict(seed=r['seed'],budget=r['budget'],arm=arm,route=route,mean_return=mu,source=str(e),reused=True))
    assert len(rows)==400 and sum(not r['reused'] for r in rows)==120
    assert len({(r['seed'],r['budget'],r['arm'],r['route']) for r in rows})==400
    def value(s,arm,route):
        rr=[r['mean_return'] for r in rows if r['seed']==s and r['arm']==arm and r['route']==route]
        assert len(rr)==4;return float(np.mean(rr))
    contrasts=[]
    families=c['comparison_families']
    for name,pairs in families.items():
        group=[]
        for a,ra,b,rb in pairs:
            d=[value(s,a,ra)-value(s,b,rb) for s in c['seeds']]
            group.append(dict(family=name,left=[a,ra],right=[b,rb],**stats(d)))
        holm(group);contrasts+=group
    verify(out)
    for f,h in read(out/'ground_freeze.json').items():assert digest(out/f)==h
    for f in out.rglob('access_audit.json'):assert not read(f)['violations'],f
    f=subprocess.run([sys.executable,str(W/'tools/verify_frozen_foundation.py')],text=True,capture_output=True);assert f.returncode==0
    atomic_json(out/'foundation_verification.json',dict(stdout=f.stdout,stderr=f.stderr,returncode=f.returncode))
    atomic_json(out/'summary.json',dict(records=rows,contrasts=contrasts,new_results=120,reused_results=280,source_and_weight_checks=True))
    lines=['# 视觉同预算适配强对照补齐','','120新增、280只读复用；400个唯一身份。主统计先在每seed内平均27评价条件和四预算，再五seed配对；家族与比较在协议中预先固定。','','|比较族|左−右|平均差|正向seed|t95|精确p|Holm|','|---|---|---:|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['family']}|{r['left']} − {r['right']}|{r['mean_delta']:+.5f}|{r['positive_seeds']}/5|{r['t95']}|{r['exact_p']}|{r['holm_p']}|")
    lines+=['','全部预算/方法均值（不按最高回报选择）：','','|配置|B1|B2|B4|B8|','|---|---:|---:|---:|---:|']
    for arm,route in sorted({(r['arm'],r['route']) for r in rows}):
        cells=[]
        for b in c['budgets']:
            v=[r['mean_return'] for r in rows if (r['arm'],r['route'],r['budget'])==(arm,route,b)];assert len(v)==5
            cells.append(f'{np.mean(v):.3f} ± {np.std(v,ddof=1):.3f}')
        lines.append('|'+arm+'/'+route+'|'+'|'.join(cells)+'|')
    lines+=['','旧任务、旧五种子后的预先固定补齐，仍属探索性证据；不是新任务确认。精确双侧p最小.0625，正均值不替代预定检验。','所有预训练保持observation-only，Adapt使用预算内目标动作更新history副本和decoder。Base-duplicate共享同一history，Random-edge仍保留MTE输入。BC/IDM保留其原训练路径，不称所有方法参数完全匹配。','本轮LAPO/LAOM是原state/feature adapters，不冒充新下载的官方像素基线。论文和图未自动更改。']
    (out/'RESULTS_CN.md').write_text(report_text(lines, "control")+'\n',encoding='utf8')
    target=P/'results'/c['name'];target.mkdir(exist_ok=False)
    for n in ('summary.json','RESULTS_CN.md','protocol.json','qualification.json','pairing_checks.json','foundation_verification.json'):shutil.copy2(out/n,target/n)
