"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""
from mte.method_names import report_text
from mte.method_names import normalize_config, resolve_control

import os

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import hashlib

import itertools

import json

from pathlib import Path

import subprocess

import sys

import time

import traceback

import numpy as np

import psutil

import torch

from utils.atomic import atomic_json

PACKAGE = Path(__file__).resolve().parents[1]

WORKSPACE = PACKAGE.parents[1]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def suite(seed):
    sid=f'mamujoco_s{seed}_n90'
    old=WORKSPACE/'observation_scale_restore_2026_09_10/runtime'
    return dict(id=sid,env='mamujoco',seed=seed,n=90,
        train=str(old/(sid+'__prepare/train.npz')),dev=str(old/(sid+'__prepare/dev.npz')),
        endpoints=str(WORKSPACE/'mamujoco_independent_branch_audit_2026_09_12/repair'/sid/'endpoints.npz'),
        endpoint_version='constant_support_repaired',
        base=str(old/(sid+'__pre_entity_target/policy/policy.pt')),
        labels={str(b):str(old/(sid+f'__export_{b}/labels.npz')) for b in [4,8,16,32,64]})

def old_pre(s,arm):
    arm = resolve_control(arm)
    folder='mte_global_aux_scope_2026_09_16' if arm=='global16' else 'mte_family_transfer_2026_09_16'
    name='global' if arm=='global16' else arm
    return WORKSPACE/folder/'runtime'/(s['id']+'__pre_'+name)

def old_result(s,arm,b,ds):
    arm = resolve_control(arm)
    if arm=='global16':
        candidates=[WORKSPACE/'mte_global_aux_scope_2026_09_16/runtime'/f'{s["id"]}__B{b}_base_global_joint16_d{ds}'/'result.json']
    else:
        candidates=[WORKSPACE/'mte_history_edge_transfer_2026_09_16/runtime'/f'{s["id"]}__B{b}_{arm}__latent_d{ds}'/'result.json']
        if arm.startswith('mif_'):
            part='decoder_repeat' if b in [8,32] else 'decoder_allbudgets'
            candidates.append(WORKSPACE/'mte_family_transfer_audit_2026_09_16'/part/f'{s["id"]}_B{b}_{arm}_d{ds}'/'result.json')
    return next((p for p in candidates if p.is_file() and read(p).get('complete')),None)

def clean_audit(folder):
    files=list(folder.glob('*access*.json'))
    assert files, f'Missing access audit: {folder}'
    for p in files: assert not read(p)['violations'],str(p)

def setup(out,config):
    config = normalize_config(config, "control")
    out.mkdir(parents=True,exist_ok=False)
    c=read(config)
    jobs=[]; paths={}; existing={}; inputs={}
    for seed in c['seeds']:
        s=suite(seed)
        for key in ('train','dev','endpoints','base'):
            inputs[s[key]]=digest(s[key])
        for p in s['labels'].values(): inputs[p]=digest(p)  # hash only; no labels are loaded
        paths[str(seed)]={}
        for arm in c['arms']:
            pre=old_pre(s,arm)
            if seed in c['reuse_seeds'] and (pre/'result.json').exists():
                r=read(pre/'result.json');assert r['complete']
                assert r['native_action_labels_read']==r['simulator_queries']==0
                clean_audit(pre)
                files=[pre/'policy/policy.pt',pre/'policy/latents.npz',pre/'result.json']
                if arm!='global16': files.append(pre/'representation.pt')
                for p in files: inputs[str(p)]=digest(p)
                existing[f'{seed}/{arm}']=str(pre)
            else:
                pre=out/'pretraining'/f'seed{seed}_{arm}'
                jobs.append(dict(id=f'pre_s{seed}_{arm}',stage='pre',seed=seed,arm=arm,out=str(pre)))
            paths[str(seed)][arm]=str(pre/'policy/policy.pt')
        for b in c['budgets']:
            for ds in c['decoder_seeds']:
                for arm in c['arms']:
                    source=old_result(s,arm,b,ds) if seed in c['reuse_seeds'] else None
                    j=dict(id=f'ground_s{seed}_B{b}_{arm}_d{ds}',stage='ground',seed=seed,arm=arm,budget=b,decoder_seed=ds,
                           out=str(out/'grounding'/f'seed{seed}_B{b}_{arm}_d{ds}'))
                    if source:
                        r=read(source);clean_audit(source.parent)
                        ev=r['evaluation']
                        assert ev['episode_seeds']==c['evaluation_seeds']['mamujoco']
                        assert len(ev['episode_returns'])==50 and abs(np.mean(ev['episode_returns'])-ev['mean_return'])<1e-8
                        assert r['budget']==b and r['decoder_seed']==ds
                        assert r.get('seed',r.get('upstream_seed'))==seed
                        j['reuse']=str(source);inputs[str(source)]=digest(source)
                    jobs.append(j)
    refs=[]
    for seed in c['seeds']:
        for b in c['budgets']:
            for name in ['anchor_only','lapo_joint','laom_joint_k3','bc_scarce']:
                p=WORKSPACE/'observation_scale_restore_2026_09_10/runtime'/f'mamujoco_s{seed}_n90__ground_{b}_{name}/result.json'
                r=read(p); assert r['complete'] and r['evaluation']['episode_seeds']==c['evaluation_seeds']['mamujoco']
                refs.append(dict(seed=seed,budget=b,method=name,path=str(p),sha256=digest(p),mean_return=r['evaluation']['mean_return']))
                inputs[str(p)]=digest(p)
    codes={str(p):digest(p) for p in PACKAGE.rglob('*.py') if '__pycache__' not in p.parts}
    for name in ['common.py','unified_models.py','training_profiles.py']:
        p=WORKSPACE/'closed_loop_lam_v1'/name;codes[str(p)]=digest(p)
    gate=WORKSPACE/c['teacher_gate'];teacher=WORKSPACE/read(gate)['checkpoint']
    assert read(gate)['passed'] and digest(teacher)==read(gate)['checkpoint_sha256']
    inputs[str(gate)]=digest(gate);inputs[str(teacher)]=digest(teacher)
    atomic_json(out/'protocol.json',c)
    atomic_json(out/'manifest.json',dict(jobs=jobs,policies=paths,reused_pretraining=existing,inputs=inputs,code=codes,
                                       teacher=str(teacher),teacher_gate=str(gate),config_hash=digest(out/'protocol.json')))
    atomic_json(out/'baseline_references.json',refs)
    atomic_json(out/'status.json',dict(status='ready',pretraining_new=sum(j['stage']=='pre' for j in jobs),
        control_total=sum(j['stage']=='ground' for j in jobs),control_reused=sum('reuse' in j for j in jobs)))
    print(json.dumps(read(out/'status.json')))

def check_sources(manifest,inputs=False):
    for p,h in manifest['code'].items(): assert digest(p)==h,f'Changed code: {p}'
    if inputs:
        for p,h in manifest['inputs'].items(): assert digest(p)==h,f'Changed source: {p}'

def ground(s,j,c,out,manifest,progress):
    j = normalize_config(j, "control")
    from training.composition import restore,FrozenPair
    from training.grounding_mamujoco import train_decoder,GroundedPolicy
    from evaluation.mamujoco import evaluate
    from utils.artifacts import data
    x,mask,n=data(s);x=x[:n];mask=mask[:n]
    b=j['budget']
    with np.load(s['labels'][str(b)],allow_pickle=False) as f: actions,ids=f['actions'],f['local_ids']
    assert np.array_equal(ids,np.arange(b))
    val=np.arange(0,b,4);fit=np.setdiff1d(np.arange(b),val)
    base,bk=restore(s['base']);aux,ak=restore(manifest['policies'][str(s['seed'])][j['arm']])
    assert np.array_equal(bk['obs_mean'],ak['obs_mean']) and np.array_equal(bk['obs_std'],ak['obs_std'])
    net=FrozenPair(base,aux).eval()
    assert not any(p.requires_grad for p in net.parameters())
    with torch.no_grad(): z=net(torch.tensor((x[ids,:200]-bk['obs_mean'])/bk['obs_std']))[0].numpy()
    valid=mask[ids];z=np.where(valid[...,None],z,0.)
    progress('decoder',updates=c['decoder_updates']['mamujoco'])
    decoder,diag=train_decoder(z,actions,valid,fit,val,j['decoder_seed'],c['decoder_updates']['mamujoco'])
    torch.save(decoder.state_dict(),out/'decoder.pt')
    progress('closed_loop',episodes=len(c['evaluation_seeds']['mamujoco']))
    ev=evaluate(GroundedPolicy(net,bk['obs_mean'],bk['obs_std'],decoder),c['evaluation_seeds']['mamujoco'],
                c.get('max_steps',200),teacher_checkpoint=manifest['teacher'],teacher_gate=manifest['teacher_gate'])
    return dict(evaluation=ev,training=diag,budget=b,decoder_seed=j['decoder_seed'],seed=s['seed'],
        pretrained_modules_updated=False,latent_dimension=32,fit_episodes=len(fit),validation_episodes=len(val))

def worker(out,jobid):
    torch.set_num_threads(1)
    c=read(out/'protocol.json');manifest=read(out/'manifest.json')
    assert digest(out/'protocol.json')==manifest['config_hash'];check_sources(manifest)
    j=next(j for j in manifest['jobs'] if j['id']==jobid);s=suite(j['seed']);dest=Path(j['out'])
    dest.mkdir(parents=True,exist_ok=False);start=time.time()
    def progress(phase,**kw): atomic_json(dest/'progress.json',dict(phase=phase,elapsed=time.time()-start,updated=time.time(),**kw))
    allowed=[s['train'],s['dev']]
    if j['stage']=='pre':allowed.append(s['endpoints'])
    else:
        gate=read(out/'freeze.json');assert gate['all_modules_frozen']
        allowed += [s['base'],manifest['policies'][str(j['seed'])][j['arm']],s['labels'][str(j['budget'])],manifest['teacher']]
        for p in allowed:
            expected=gate['hashes'].get(p,manifest['inputs'].get(p))
            assert expected and digest(p)==expected,f'Changed input: {p}'
    from utils.access import guard
    audit=guard(dest,allowed,pretraining=j['stage']=='pre')
    try:
        if j['stage']=='pre':
            if j['arm']=='global16':
                from experiments.route_stages import train_global
                r=train_global(s,c,dest,progress)
            else:
                from training import route_representation as route
                # Register the factory's already implemented identity-coordinate mode.
                # Original model, optimizer, losses, histories and train function stay unchanged.
                route.ARMS={**route.ARMS,**{f'{m}_raw':(m,'raw_coordinates') for m in ['simple','graph','mif']}}
                r=route.train(s,j['arm'],c,dest,progress)
        else:r=ground(s,j,c,dest,manifest,progress)
        assert not audit['violations']
        r.update(complete=True,job=j['id'],stage=j['stage'],seed=j['seed'],arm=j['arm'],elapsed_seconds=time.time()-start)
        atomic_json(dest/'access.json',audit);atomic_json(dest/'result.json',r)
    except Exception:
        atomic_json(dest/'access.json',audit);atomic_json(dest/'failure.json',dict(traceback=traceback.format_exc()))
        raise

def freeze(out,manifest):
    from training.composition import restore
    hashes={}
    for seed,arms in manifest['policies'].items():
        for arm,p in arms.items():
            folder=Path(p).parents[1];r=read(folder/'result.json');assert r['complete']
            clean_audit(folder)
            _,ck=restore(p);assert ck['z_dim']==16
            hashes[p]=digest(p)
            s=suite(int(seed));_,bk=restore(s['base'])
            assert np.array_equal(bk['obs_mean'],ck['obs_mean']) and np.array_equal(bk['obs_std'],ck['obs_std'])
            hashes[s['base']]=digest(s['base'])
    check_sources(manifest,True)
    atomic_json(out/'freeze.json',dict(all_modules_frozen=True,hashes=hashes,time=time.time()))

def summarize(out,manifest,c):
    c = normalize_config(c, "control")
    from scipy.stats import t
    rows=[]
    for j in manifest['jobs']:
        if j['stage']!='ground':continue
        p=Path(j.get('reuse',str(Path(j['out'])/'result.json')));r=read(p);ev=r['evaluation']
        assert r['complete'] and ev['episode_seeds']==c['evaluation_seeds']['mamujoco']
        assert abs(np.mean(ev['episode_returns'])-ev['mean_return'])<1e-8
        rows.append(dict(seed=j['seed'],arm=j['arm'],budget=j['budget'],decoder_seed=j['decoder_seed'],
                         mean_return=ev['mean_return'],source=str(p),reused='reuse' in j))
    assert len(rows)==len({(r['seed'],r['arm'],r['budget'],r['decoder_seed']) for r in rows})==875
    contrasts=[]
    for model in ['mif','graph','simple']:
        for kind,control in [('route','global16'),('coordinates',model+'_raw')]:
            diffs=[]
            for seed in c['seeds']:
                means={arm:np.mean([r['mean_return'] for r in rows if r['seed']==seed and r['arm']==arm]) for arm in [model+'_matched',control]}
                diffs.append(means[model+'_matched']-means[control])
            v=np.array(diffs);half=t.ppf(.975,4)*v.std(ddof=1)/np.sqrt(5)
            signs=np.array(list(itertools.product([-1,1],repeat=5)))
            p=float(np.mean(np.abs((signs*v).mean(1))>=abs(v.mean())-1e-12))
            contrasts.append(dict(model=model,family=kind,control=control,mean=float(v.mean()),seed_differences=diffs,
                ci95=[float(v.mean()-half),float(v.mean()+half)],positive_seeds=int((v>0).sum()),exact_p=p))
    for family in ['route','coordinates']:
        selected=sorted([r for r in contrasts if r['family']==family],key=lambda r:r['exact_p']);last=0.
        for i,r in enumerate(selected):last=max(last,min(1.,(3-i)*r['exact_p']));r['holm_p']=last
    check_sources(manifest,True)
    for p,h in read(out/'freeze.json')['hashes'].items():assert digest(p)==h
    atomic_json(out/'summary.json',dict(rows=rows,contrasts=contrasts,source_and_weight_checks=True))
    lines=['# MaMuJoCo 路线与坐标配对补齐结果','','五个上游种子、五预算、五次嵌套decoder重复。0/1沿用已看过的开发模型，是重复性扩展，不是全新任务确认。',
           '两个预定检验族各保留三模型；先对预算及decoder平均，再以五个上游种子配对。', '',
           '|模型|比较|均值差↑|95%区间|正向种子|Holm p|','|---|---|---:|---|---:|---:|']
    for r in contrasts:lines.append(f'|{r["model"]}|{r["control"]}|{r["mean"]:.4f}|{r["ci95"]}|{r["positive_seeds"]}/5|{r["holm_p"]:.4f}|')
    lines+=['','全部875条记录见summary.json；原Base/LAPO/LAOM/BC背景参照见baseline_references.json，不混为同decoder配对。',
            '完整路线的目标/上游成本不同；raw配对保持同表与同目标，只替换Möbius/zeta坐标。该结果不验证实际donor物理语义。','']
    (out/'RESULTS_CN.md').write_text(report_text(lines, "control"),encoding='utf-8')

def manager(out):
    import msvcrt
    lock=(out/'manager.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    c=read(out/'protocol.json');manifest=read(out/'manifest.json');check_sources(manifest,True)
    active={};done=set();failures=[];started=time.time()
    for j in manifest['jobs']:
        f=Path(j['out'])/'result.json'
        if 'reuse' in j:done.add(j['id'])
        elif f.exists():assert read(f)['complete'];clean_audit(f.parent);done.add(j['id'])
        elif f.parent.exists():raise RuntimeError(f'Incomplete stage needs inspection: {f.parent}')
    try:
        while len(done)<len(manifest['jobs']):
            for k,(proc,stream) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                stream.close();j=next(j for j in manifest['jobs'] if j['id']==k)
                f=Path(j['out'])/'result.json'
                if rc or not f.exists():failures.append(dict(job=k,exit_code=rc))
                else:assert read(f)['complete'];clean_audit(f.parent);done.add(k)
                del active[k]
            pre_done=all(j['id'] in done for j in manifest['jobs'] if j['stage']=='pre')
            if pre_done and not (out/'freeze.json').exists() and not failures:freeze(out,manifest)
            stage='ground' if pre_done else 'pre';cap=c['ground_workers'] if pre_done else c['pretraining_workers']
            if not failures and not (out/'PAUSE').exists():
                for j in manifest['jobs']:
                    if j['stage']!=stage or j['id'] in done or j['id'] in active:continue
                    if len(active)>=cap or psutil.virtual_memory().available/2**30<c['minimum_available_gib']:break
                    stream=(out/(j['id']+'.log')).open('ab')
                    proc=subprocess.Popen([sys.executable,'-X','utf8','-m','experiments.mamujoco_route_completion','--out',str(out),'--job',j['id']],
                        cwd=PACKAGE,stdout=stream,stderr=stream,creationflags=subprocess.CREATE_NO_WINDOW)
                    active[j['id']]=(proc,stream)
            atomic_json(out/'status.json',dict(status='failed' if failures else 'paused' if (out/'PAUSE').exists() else 'running',
                stage=stage,pretraining_done=sum(j['id'] in done for j in manifest['jobs'] if j['stage']=='pre'),
                pretraining_new=sum(j['stage']=='pre' for j in manifest['jobs']),
                control_done=sum(j['id'] in done for j in manifest['jobs'] if j['stage']=='ground'),control_total=875,
                reused=sum('reuse' in j for j in manifest['jobs']),running=[dict(job=k,pid=p.pid) for k,(p,_) in active.items()],
                failures=failures,manager_pid=os.getpid(),active_seconds=time.time()-started,updated=time.time()))
            if failures and not active:return
            time.sleep(3)
        summarize(out,manifest,c)
        subprocess.run([sys.executable,str(WORKSPACE/'tools/verify_frozen_foundation.py')],cwd=WORKSPACE,check=True)
        atomic_json(out/'status.json',dict(status='complete',control_done=875,control_total=875,
            reused=sum('reuse' in j for j in manifest['jobs']),source_and_weight_checks=True,active_seconds=time.time()-started,updated=time.time()))
    finally:
        lock.close()
