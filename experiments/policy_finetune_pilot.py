"""Fixed-budget downstream history fine-tuning: visual LAPO/LAOM and MPE."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import psutil
from utils.atomic import atomic_json

PACKAGE=Path(__file__).resolve().parents[1]
WORKSPACE=PACKAGE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def location(out,job):
    phase,suite,seed,arm,kind=job
    return out/phase/suite/f'seed{seed}'/(kind+'_'+arm)


def jobs(config,phase,kind):
    result=[]
    for suite in ['visual','mpe']:
        p=config[suite];seeds=p['seeds'][:1] if phase=='smoke' else p['seeds']
        arms=[f'{m}_{c}_pretrained_'+('frozen' if phase=='parity' else 'trainable') for m in p['families'] for c in p['compositions']] if suite=='visual' else p['methods']
        result += [(phase,suite,s,a,kind) for s in seeds for a in arms]
    return result


def reference(config,suite,seed,arm):
    if suite=='visual':
        from training.policy_finetune_pilot import visual_base
        base=visual_base(arm)
        root=PACKAGE/config['visual']['source']/'formal'/f'seed{seed}'/'b2'
        return root/('ground_'+base),root/('evaluate_'+base)/'result.json'
    rows=read(PACKAGE/config['mpe']['source']/'summary.json')['records']
    row=next(r for r in rows if (r['n'],r['seed'],r['budget'],r['method'])==(700,seed,32,arm))
    f=Path(row['source']);return f.with_suffix('.pt'),f


def validate(out,job):
    dest=location(out,job);r=read(dest/'result.json')
    assert r['complete'] and r['job']==list(job)
    assert not read(dest/'access_audit.json')['violations']
    if job[-1]=='ground':
        assert r['native_action_labels_read']==(400 if job[1]=='visual' else 704)
        assert r['partner_labels_read']==0 and r['simulator_queries']==0
    return r


def worker(out,job,config):
    import torch
    from closed_loop_lam_v1 import common as C
    from training import policy_finetune_pilot as T
    phase,suite,seed,arm,kind=job;dest=location(out,job);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1 if suite=='visual' else 2);torch.set_num_interop_threads(1)
    C.seed_all(seed)
    # Visual uses the previously verified deterministic setup. MPE preserves its
    # existing default backend settings, checked by frozen numerical reproduction.
    if suite=='visual':
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    started=time.time()
    if kind=='ground':
        updates=2 if phase=='smoke' else config[suite]['updates']
        if suite=='visual':r=T.visual_ground(dest,dest.parent,seed,arm,updates)
        else:
            source=PACKAGE/config['mpe']['source']/'n700'/f'seed{seed}'
            r=T.mpe_ground(dest,source,seed,arm,phase!='parity',updates)
    else:
        ground=location(out,(*job[:-1],'ground'));assert read(ground/'result.json')['complete']
        if phase=='formal':assert digest(ground/'decoder.pt')==read(out/'ground_freeze.json')[str((ground/'decoder.pt').relative_to(out))]
        if suite=='visual':
            from training import visual_label_budget as B
            from training import visual_supervision_control as S
            from evaluation.visual_label_budget import evaluate
            B.V.controller=S.controller
            seeds=[908601] if phase in ('parity','smoke') else config['visual']['evaluation_seeds']
            assets=PACKAGE/config['visual']['mif_reference']/'assets'
            r=evaluate(dest,arm,seed,dest.parent,assets,seeds)
        else:
            from utils.access import guard
            from training.grounding_mpe import GroundedPolicy
            from evaluation.mpe import evaluate
            source=PACKAGE/config['mpe']['source']/'n700'/f'seed{seed}'
            allowed=[ground/'decoder.pt']+[source/'pretrain'/m/'policy/policy.pt' for m in arm.split('+')]
            audit=guard(dest,allowed,False)
            try:
                cp=torch.load(ground/'decoder.pt',map_location='cpu',weights_only=False)
                net,ck=T.mpe_net(source,arm);net.load_state_dict(cp['history'])
                dim=net(torch.zeros(1,1,16))[0].shape[-1]
                decoder=C.ActionDecoder(dim,2);decoder.load_state_dict(cp['decoder'])
                seeds=list(range(97220000,97220000+(1 if phase in ('smoke','parity') else 100)))
                r=evaluate(GroundedPolicy(net,cp['mean'],cp['std'],decoder.eval()),seeds)
                assert not audit['violations']
            finally:atomic_json(dest/'access_audit.json',audit)
    r.update(complete=True,job=job,seconds=time.time()-started)
    atomic_json(dest/'result.json',r)


def qualify(out,config):
    import torch
    checks=[]
    for job in jobs(config,'parity','ground'):
        phase,suite,seed,arm,_=job;dest=location(out,job)
        frozen,evaluation=reference(config,suite,seed,arm)
        new=torch.load(dest/'decoder.pt',map_location='cpu',weights_only=False)
        old=torch.load(frozen/'decoder.pt' if suite=='visual' else frozen,map_location='cpu',weights_only=False)
        key='state_dict' if suite=='visual' else 'decoder'
        assert new[key].keys()==old[key].keys()
        error=max(float((new[key][k]-old[key][k]).abs().max()) for k in old[key]);assert error<=1e-6,(job,error)
        if suite=='visual':
            fit_error=float(np.abs(np.load(dest/'fit_predictions.npy')-np.load(frozen/'fit_predictions.npy')).max());assert fit_error<=1e-6
        else:
            assert new['history'].keys()==old['history'].keys()
            assert all(torch.equal(new['history'][k],old['history'][k]) for k in old['history'])
        replay=location(out,(*job[:-1],'evaluate'))
        if suite=='visual':
            with np.load(replay/'episode_908601.npz') as a,np.load(evaluation.parent/'episode_908601.npz') as b:
                assert all(np.array_equal(a[k],b[k]) for k in b.files)
        else:
            a=read(replay/'result.json');b=read(evaluation)
            assert a['episode_seeds']==b['episode_seeds'][:1]
            for key in ['episode_returns','episode_collisions','episode_final_coverage']:assert np.allclose(a[key],b[key][:1],rtol=0,atol=1e-10),(job,key)
        checks.append(dict(job=job,decoder_max_abs=error,passed=True))
    assert len(checks)==35
    atomic_json(out/'qualification.json',dict(passed=True,smokes=7,frozen_parities=35,episode_replays=35,checks=checks))


def snapshot(out,config):
    # Preserve and verify upstream frozen provenance, then extend for this pilot.
    prior=PACKAGE/config['visual']['mif_reference']
    files={Path(f):h for f,h in read(prior/'source_manifest.json').items()}
    for f,h in files.items():assert digest(f)==h,f
    files={f:None for f in files}
    for sub in ['experiments','training','evaluation','mte','visual','environments','utils','closed_loop_lam_v1']:
        files.update({f:None for f in (PACKAGE/sub).rglob('*.py')})
    files[PACKAGE/'configs/policy_finetune_pilot.json']=None
    files[prior/'summary.json']=None
    for seed in config['mpe']['seeds']:
        root=PACKAGE/config['mpe']['source']/'n700'/f'seed{seed}'
        manifest=read(root/'pretrain/complete.json')
        checkpoint_hashes={key.replace(chr(92),'/'):value for key,value in manifest['checkpoints'].items()}
        for method in ['base','graph','lapo','laom']:
            rel=method+'/policy/policy.pt';f=root/'pretrain'/rel
            assert digest(f)==checkpoint_hashes[rel];files[f]=None
        for rel in ['pretrain/complete.json','data/train.npz','data/labels_b32.npz']:files[root/rel]=None
    for job in jobs(config,'parity','ground'):
        _,suite,seed,arm,_=job;ground,result=reference(config,suite,seed,arm)
        for f in ([ground/'decoder.pt',ground/'fit_predictions.npy',result,result.parent/'episode_908601.npz'] if suite=='visual' else [ground,result]):files[f]=None
    for seed in config['mpe']['seeds']:
        _,result=reference(config,'mpe',seed,'bc');files[result]=None
    atomic_json(out/'source_manifest.json',{str(f.resolve()):digest(f) for f in sorted(files)})


def verify_sources(out):
    for f,h in read(out/'source_manifest.json').items():assert digest(f)==h,f


def progress(path):
    try:return read(path)
    except (FileNotFoundError,PermissionError,json.JSONDecodeError):return {'telemetry_unavailable':True}


def run_phase(out,config,title,work):
    active={};done=set();failure=None
    for job in work:
        d=location(out,job)
        if (d/'result.json').exists():validate(out,job);done.add(job)
        elif d.exists():raise RuntimeError(f'Partial retained; inspection needed: {d}')
    while len(done)<len(work) or active:
        for job,(proc,log,created) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();del active[job]
            try:
                assert rc==0,(job,rc);validate(out,job);done.add(job)
            except Exception:failure=traceback.format_exc()
        paused=(out/'PAUSE').exists()
        if not failure and not paused:
            for job in work:
                if len(active)>=config['workers']:break
                if job in done or job in active:continue
                log=(out/'logs'/('__'.join(map(str,job))+f'.{time.time_ns()}.log')).open('w',encoding='utf-8')
                env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                proc=subprocess.Popen([sys.executable,'-X','utf8','-m','experiments.policy_finetune_pilot','worker','--out',str(out),'--job',*map(str,job)],cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                try:created=psutil.Process(proc.pid).create_time()
                except psutil.NoSuchProcess:created=None
                active[job]=(proc,log,created)
        atomic_json(out/'status.json',dict(status='draining_after_failure' if failure else 'pausing' if paused else 'running',phase=title,
            manager_pid=os.getpid(),manager_created=psutil.Process().create_time(),updated_unix=time.time(),completed=len(done),total=len(work),failure=failure,
            new_evaluations=len(list((out/'formal').glob('*/seed*/evaluate_*/result.json'))),
            active=[dict(job=j,pid=p.pid,created=c,progress=progress(location(out,j)/'progress.json')) for j,(p,l,c) in active.items()]))
        if not active and failure:raise RuntimeError(failure)
        if not active and paused:return False
        if len(done)<len(work):time.sleep(3)
    return True


def aggregate(out,config):
    from scipy.stats import t
    records=[];contrasts=[]
    for job in jobs(config,'formal','evaluate'):
        _,suite,seed,arm,_=job
        r=validate(out,job);_,old_path=reference(config,suite,seed,arm);old=read(old_path)
        if suite=='visual':
            assert [e['seed'] for e in r['rows']]==config['visual']['evaluation_seeds']
            mean=lambda x:float(np.mean([e['return_value'] for e in x['rows'] if e['seed'] in config['visual']['primary_evaluation_seeds']]))
        else:
            assert r['episode_seeds']==old['episode_seeds'];mean=lambda x:float(np.mean(x['episode_returns']))
        records.append(dict(suite=suite,seed=seed,arm=arm,frozen=mean(old),finetuned=mean(r),delta=mean(r)-mean(old),frozen_source=str(old_path)))
    for suite in ['visual','mpe']:
        family=[]
        for arm in sorted({r['arm'] for r in records if r['suite']==suite}):
            rr=[r for r in records if r['suite']==suite and r['arm']==arm];d=np.array([r['delta'] for r in rr]);assert len(d)==5
            radius=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5));null=np.abs(np.array(list(itertools.product([-1,1],repeat=5)))@d/5)
            family.append(dict(suite=suite,arm=arm,frozen=float(np.mean([r['frozen'] for r in rr])),finetuned=float(np.mean([r['finetuned'] for r in rr])),mean_delta=float(d.mean()),seed_deltas=d.tolist(),positive_seeds=int((d>0).sum()),t95=[float(d.mean()-radius),float(d.mean()+radius)],exact_p=float((null>=abs(d.mean())-1e-12).mean())))
        bound=0
        for i,r in enumerate(sorted(family,key=lambda r:r['exact_p'])):
            bound=max(bound,min(1.,r['exact_p']*(len(family)-i)));r['holm_p']=bound
        contrasts+=family
    assert len(records)==35 and len({(r['suite'],r['seed'],r['arm']) for r in records})==35
    verify_sources(out)
    for f,h in read(out/'ground_freeze.json').items():assert digest(out/f)==h,f
    for f in out.rglob('access_audit.json'):assert not read(f)['violations'],f
    check=subprocess.run([sys.executable,str(WORKSPACE/'tools/verify_frozen_foundation.py')],capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
    atomic_json(out/'foundation_verification.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
    context=[]
    prior=read(PACKAGE/config['visual']['mif_reference']/'summary.json')
    keep=['anchor_solo_edge_cara_mif','anchor_plus_edge_cara_mif','mif_solo_pretrained_trainable','mif_aux_pretrained_trainable','bc_batch256','bc_idm_relabel']
    for r in prior['records']:
        if r['budget']==2 and r['arm'] in keep:context.append(dict(suite='visual',seed=r['seed'],arm=r['arm'],mean_return=r['mean_return']))
    for seed in config['mpe']['seeds']:
        _,f=reference(config,'mpe',seed,'bc');r=read(f)
        context.append(dict(suite='mpe',seed=seed,arm='bc',mean_return=r['mean_return']))
    atomic_json(out/'summary.json',dict(records=records,contrasts=contrasts,contextual_references=context,new_results=35,reused_results=35,source_and_weight_checks=True))
    lines=['# 下游history微调：视觉与MPE轻量试验','','固定单预算、五既有上游种子；观察性预训练权重不变。新监督路线与原冻结路线分别保留。','','|场景/配置|冻结均值|微调均值|增益|正向种子|t95|精确p|Holm p|','|---|---:|---:|---:|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['suite']} {r['arm']}|{r['frozen']:.5f}|{r['finetuned']:.5f}|{r['mean_delta']:+.5f}|{r['positive_seeds']}/5|{r['t95']}|{r['exact_p']:.4f}|{r['holm_p']:.4f}|")
    lines+=['','视觉B2=400个动作向量，600更新，27主评价条件；MPE N700/B32=704标签（528训练+176验证），1000更新、100共同episode。','本轮为已有任务的固定预算探索，不代表全预算收益或独立新任务。视觉沿用论文的LAPO/LAOM state-adapter，不是官方像素端到端复现。不同模型均可受益于监督适配，不能把共同收益归为MTE独有。','精确双侧p最小.0625；保留正负方向，均值不替代显著性。论文图表未自动修改。']
    lines+=['','## 同预算背景参照（描述性）','','|场景/配置|平均回报|','|---|---:|']
    for suite,arm in sorted({(r['suite'],r['arm']) for r in context}):
        values=[r['mean_return'] for r in context if (r['suite'],r['arm'])==(suite,arm)]
        lines.append(f'|{suite} {arm}|{np.mean(values):.5f}|')
    lines+=['','视觉MIF沿用完全相同的微调规则。MPE原BC为3200更新，与本轮1000更新不同，仅作原协议背景参照。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=PACKAGE/'results'/config['name'];dest.mkdir(parents=True,exist_ok=False)
    import shutil
    for name in ['RESULTS_CN.md','summary.json','protocol.json','qualification.json','foundation_verification.json']:shutil.copy2(out/name,dest/name)


def manage(out,config):
    out.mkdir(parents=True,exist_ok=True);lock=out/'manager.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),f)
    try:
        (out/'logs').mkdir(exist_ok=True)
        if (out/'source_manifest.json').exists():assert read(out/'protocol.json')==config;verify_sources(out)
        else:atomic_json(out/'protocol.json',config);snapshot(out,config)
        for phase,kind,title in [('smoke','ground','smoke_training'),('smoke','evaluate','smoke_rollout'),('parity','ground','frozen_weight_parity'),('parity','evaluate','original_episode_replay'),('formal','ground','history_finetuning'),('formal','evaluate','closed_loop_evaluation')]:
            if not run_phase(out,config,title,jobs(config,phase,kind)):return
            if phase=='parity' and kind=='evaluate':qualify(out,config);verify_sources(out)
            if phase=='formal' and kind=='ground':
                pairs=[]
                for job in jobs(config,'formal','ground'):
                    _,suite,seed,arm,_=job
                    parity_arm=arm.replace('_trainable','_frozen') if suite=='visual' else arm
                    a=read(location(out,job)/'training_identity.json');b=read(location(out,('parity',suite,seed,parity_arm,'ground'))/'training_identity.json')
                    keys=['history_initial','decoder_initial','normalization','inputs','targets','batches']
                    keys+=['total_history_parameters','decoder_parameters'] if suite=='visual' else ['history_parameters','decoder_parameters','fit_ids','validation_ids']
                    assert all(a[k]==b[k] for k in keys),(job,keys)
                    pairs.append(dict(job=job,passed=True))
                atomic_json(out/'pairing_checks.json',pairs)
                atomic_json(out/'ground_freeze.json',{str(f.relative_to(out)):digest(f) for f in (out/'formal').glob('*/seed*/ground_*/decoder.pt')})
        aggregate(out,config)
        atomic_json(out/'status.json',dict(status='complete',new_results=35,reused_results=35,source_and_weight_checks=True,updated_unix=time.time()))
    except Exception:
        atomic_json(out/f'manager_failure.{time.time_ns()}.json',dict(error=traceback.format_exc()))
        atomic_json(out/'status.json',dict(status='failed',manager_pid=os.getpid(),updated_unix=time.time()))
        raise
    finally:lock.unlink()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['manage','worker']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--job',nargs=5)
    args=ap.parse_args();config=read(PACKAGE/'configs/policy_finetune_pilot.json');out=args.out.resolve()
    if args.command=='manage':manage(out,config)
    else:
        phase,suite,seed,arm,kind=args.job;worker(out,(phase,suite,int(seed),arm,kind),config)


if __name__=='__main__':main()
