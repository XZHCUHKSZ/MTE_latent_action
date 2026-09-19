"""Own-model fixed-budget downstream history fine-tuning extension."""
import argparse, hashlib, itertools, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import psutil
from utils.atomic import atomic_json
from experiments.policy_finetune_pilot import read, digest, location, progress, verify_sources
PACKAGE=Path(__file__).resolve().parents[1]
WORKSPACE=PACKAGE.parents[1]


def jobs(c,phase,kind):
    result=[]
    for suite in ['visual','mamujoco']:
        q=c[suite];seeds=q['seeds'][:1] if phase=='smoke' else q['seeds']
        arms=[f'{m}_{comp}_pretrained_'+('frozen' if phase=='parity' else 'trainable') for m in q['families'] for comp in q['compositions']] if suite=='visual' else q['methods']
        result += [(phase,suite,s,a,kind) for s in seeds for a in arms]
    return result


def mamujoco_spec(c,seed,arm):
    from experiments.mamujoco_route_completion import suite
    s=suite(seed);q=c['mamujoco']
    root=PACKAGE/q['native_source' if arm=='mif_native' else 'route_source']
    native='masked_mobius' if arm=='mif_native' else arm
    manifest=read(root/'manifest.json');frozen=read(root/'freeze.json')['hashes']
    auxiliary=manifest['policies'][str(seed)][native]
    for f in [auxiliary,s['base']]:assert digest(f)==frozen[f],f
    row=next(r for r in read(root/'summary.json')['rows'] if (r['seed'],r['arm'],r['budget'],r['decoder_seed'])==(seed,native,8,q['decoder_seed']))
    return dict(train=s['train'],labels=s['labels']['8'],base=s['base'],auxiliary=auxiliary,
        teacher=manifest['teacher'],teacher_gate=manifest['teacher_gate'],decoder_seed=q['decoder_seed'],
        old_result=row['source'],source_root=str(root),arm=arm)


def reference(c,suite,seed,arm):
    if suite=='visual':
        from training.mte_finetune_extension import visual_base
        base=visual_base(arm);root=PACKAGE/c['visual']['source']/'formal'/f'seed{seed}'/'b2'
        return root/('ground_'+base),root/('evaluate_'+base)/'result.json'
    f=Path(mamujoco_spec(c,seed,arm)['old_result']);return f.parent/'decoder.pt',f


def validate(out,job):
    dest=location(out,job);r=read(dest/'result.json')
    assert r['complete'] and r['job']==list(job)
    assert not read(dest/'access_audit.json')['violations']
    if job[-1]=='ground':
        assert r['partner_labels_read']==0 and r['simulator_queries']==0
        if job[1]=='visual':assert r['native_action_labels_read']==400
        else:assert 0<r['native_action_labels_read']<=1600 and r['fit_action_labels']+r['validation_action_labels']==r['native_action_labels_read']
    return r


def worker(out,job,c):
    import torch
    from closed_loop_lam_v1 import common as C
    from training import mte_finetune_extension as T
    phase,suite,seed,arm,kind=job;dest=location(out,job);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);C.seed_all(seed)
    if suite=='visual':
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    start=time.time()
    if kind=='ground':
        updates=2 if phase=='smoke' else c[suite]['updates']
        if suite=='visual':r=T.visual_ground(dest,dest.parent,seed,arm,updates)
        else:r=T.mamujoco_ground(dest,mamujoco_spec(c,seed,arm),phase!='parity',updates)
    else:
        ground=location(out,(*job[:-1],'ground'));validate(out,(*job[:-1],'ground'))
        if phase=='formal':assert digest(ground/'decoder.pt')==read(out/'ground_freeze.json')[str((ground/'decoder.pt').relative_to(out))]
        if suite=='visual':
            from training import visual_label_budget as B
            from training import visual_supervision_control as S
            from evaluation.visual_label_budget import evaluate
            B.V.controller=S.controller
            seeds=[908601] if phase in ('parity','smoke') else c['visual']['evaluation_seeds']
            r=evaluate(dest,arm,seed,dest.parent,PACKAGE/c['visual']['mif_reference']/'assets',seeds)
        else:
            from utils.access import guard
            from training.grounding_mamujoco import GroundedPolicy
            from evaluation.mamujoco import evaluate
            spec=mamujoco_spec(c,seed,arm)
            allowed=[ground/'decoder.pt',spec['base'],spec['auxiliary'],spec['teacher']]
            audit=guard(dest,allowed,False)
            try:
                cp=torch.load(ground/'decoder.pt',map_location='cpu',weights_only=False)
                net,ck=T.mamujoco_net(spec['base'],spec['auxiliary']);net.load_state_dict(cp['history'])
                decoder=C.ActionDecoder(32,2);decoder.load_state_dict(cp['decoder']);decoder.eval()
                seeds=c['mamujoco']['evaluation_seeds'][:1] if phase in ('smoke','parity') else c['mamujoco']['evaluation_seeds']
                r=evaluate(GroundedPolicy(net.eval(),cp['mean'],cp['std'],decoder),seeds,200,spec['teacher'],spec['teacher_gate'])
                assert not audit['violations']
            finally:atomic_json(dest/'access_audit.json',audit)
    r.update(complete=True,job=job,seconds=time.time()-start)
    atomic_json(dest/'result.json',r)


def qualify(out,c):
    import torch
    checks=[]
    for job in jobs(c,'parity','ground'):
        _,suite,seed,arm,_=job;dest=location(out,job);old_ground,old_result=reference(c,suite,seed,arm)
        new=torch.load(dest/'decoder.pt',map_location='cpu',weights_only=False)
        old=torch.load(old_ground/'decoder.pt' if suite=='visual' else old_ground,map_location='cpu',weights_only=False)
        a=new['state_dict' if suite=='visual' else 'decoder'];b=old['state_dict'] if suite=='visual' else old
        assert a.keys()==b.keys();err=max(float((a[k]-b[k]).abs().max()) for k in b);assert err<=1e-6,(job,err)
        replay=location(out,(*job[:-1],'evaluate'))
        if suite=='visual':
            assert np.max(np.abs(np.load(dest/'fit_predictions.npy')-np.load(old_ground/'fit_predictions.npy')))<=1e-6
            with np.load(replay/'episode_908601.npz') as x,np.load(old_result.parent/'episode_908601.npz') as y:
                assert all(np.array_equal(x[k],y[k]) for k in y.files)
        else:
            x=read(replay/'result.json');y=read(old_result)['evaluation']
            assert x['episode_seeds']==y['episode_seeds'][:1]
            assert np.allclose(x['episode_returns'],y['episode_returns'][:1],rtol=0,atol=1e-9),(job,'return replay')
        checks.append(dict(job=job,decoder_max_abs=err,passed=True))
    assert len(checks)==45
    atomic_json(out/'qualification.json',dict(passed=True,smokes=9,frozen_parities=45,episode_replays=45,checks=checks))


def snapshot(out,c):
    prior=PACKAGE/c['visual']['source'];files={Path(f):h for f,h in read(prior/'source_manifest.json').items()}
    for f,h in files.items():assert digest(f)==h,f
    for sub in ['experiments','training','evaluation','mte','visual','environments','utils','closed_loop_lam_v1']:
        files.update({f:None for f in (PACKAGE/sub).rglob('*.py')})
    files[PACKAGE/'configs/mte_finetune_extension.json']=None;files[prior/'summary.json']=None
    for seed in c['mamujoco']['seeds']:
        for arm in c['mamujoco']['methods']:
            s=mamujoco_spec(c,seed,arm);root=Path(s['source_root']);manifest=read(root/'manifest.json')
            for k in ['train','labels','base','auxiliary','teacher','teacher_gate','old_result']:files[Path(s[k])]=None
            for k in ['train','labels','teacher','teacher_gate']:assert digest(s[k])==manifest['inputs'][s[k]],s[k]
            files[Path(s['old_result']).parent/'decoder.pt']=None
            for name in ['manifest.json','freeze.json','summary.json','protocol.json']:files[root/name]=None
    for job in jobs(c,'parity','ground'):
        _,suite,seed,arm,_=job;ground,result=reference(c,suite,seed,arm)
        for f in ([ground/'decoder.pt',ground/'fit_predictions.npy',result,result.parent/'episode_908601.npz'] if suite=='visual' else [ground,result]):files[f]=None
    atomic_json(out/'source_manifest.json',{str(f.resolve()):digest(f) for f in sorted(files)})


def aggregate(out,c):
    from scipy.stats import t
    records=[];contrasts=[]
    for job in jobs(c,'formal','evaluate'):
        _,suite,seed,arm,_=job;r=validate(out,job);_,path=reference(c,suite,seed,arm);old=read(path)
        if suite=='visual':
            assert [e['seed'] for e in r['rows']]==c['visual']['evaluation_seeds']
            mean=lambda x:float(np.mean([e['return_value'] for e in x['rows'] if e['seed'] in c['visual']['primary_evaluation_seeds']]))
        else:
            old=old['evaluation'];assert r['episode_seeds']==old['episode_seeds']
            mean=lambda x:float(np.mean(x['episode_returns']))
        records.append(dict(suite=suite,seed=seed,arm=arm,frozen=mean(old),finetuned=mean(r),delta=mean(r)-mean(old),frozen_source=str(path)))
    for suite in ['visual','mamujoco']:
        family=[]
        for arm in sorted({r['arm'] for r in records if r['suite']==suite}):
            rr=[r for r in records if r['suite']==suite and r['arm']==arm];d=np.array([r['delta'] for r in rr]);assert len(d)==5
            rad=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5));null=np.abs(np.array(list(itertools.product([-1,1],repeat=5)))@d/5)
            family.append(dict(suite=suite,arm=arm,frozen=float(np.mean([r['frozen'] for r in rr])),finetuned=float(np.mean([r['finetuned'] for r in rr])),mean_delta=float(d.mean()),seed_deltas=d.tolist(),positive_seeds=int((d>0).sum()),t95=[float(d.mean()-rad),float(d.mean()+rad)],exact_p=float((null>=abs(d.mean())-1e-12).mean())))
        bound=0
        for i,r in enumerate(sorted(family,key=lambda r:r['exact_p'])):
            bound=max(bound,min(1.,r['exact_p']*(len(family)-i)));r['holm_p']=bound
        contrasts+=family
    assert len(records)==45 and len({(r['suite'],r['seed'],r['arm']) for r in records})==45
    verify_sources(out)
    for f,h in read(out/'ground_freeze.json').items():assert digest(out/f)==h,f
    for f in out.rglob('access_audit.json'):assert not read(f)['violations'],f
    check=subprocess.run([sys.executable,str(WORKSPACE/'tools/verify_frozen_foundation.py')],capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
    atomic_json(out/'foundation_verification.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
    atomic_json(out/'summary.json',dict(records=records,contrasts=contrasts,new_results=45,reused_results=45,source_and_weight_checks=True))
    lines=['# MTE自身模型：固定低标签预算history微调','','|场景/配置|冻结均值|微调均值|增益|正向种子|t95|精确p|Holm p|','|---|---:|---:|---:|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['suite']} {r['arm']}|{r['frozen']:.5f}|{r['finetuned']:.5f}|{r['mean_delta']:+.5f}|{r['positive_seeds']}/5|{r['t95']}|{r['exact_p']:.4f}|{r['holm_p']:.4f}|")
    lines+=['','视觉B2：400动作向量，600更新，27主评价条件；MaMuJoCo B8：8/90轨迹，计入全部有效训练与验证动作，800更新，50共同episode。MaMuJoCo固定decoder种子202609160，重复单位为5个上游种子。','各方法只与自身冻结版本配对。视觉MIF及LAPO/LAOM另轮完整结果可作同预算背景参照。MaMuJoCo MIF为原生masked_mobius；Simple/Graph为accepted matched路线，不把其预训练协议差异当成history微调效应。','观察性预训练和前端权重原件不改。微调更新history副本与decoder；既有冻结主结果不替换。这里只检验下游适配收益，不隔离相对随机初始化的预训练增量，也不是新任务确认。','五种子精确双侧p最低.0625；所有方向与区间完整保留。论文图表未自动修改。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=PACKAGE/'results'/c['name'];dest.mkdir(parents=True,exist_ok=False)
    import shutil
    for name in ['RESULTS_CN.md','summary.json','protocol.json','qualification.json','foundation_verification.json','pairing_checks.json']:shutil.copy2(out/name,dest/name)


def wait_dependency(out,c):
    dependency=PACKAGE/c['wait_for']
    while True:
        state=read(dependency/'status.json')
        if state['status']=='complete':
            assert state['source_and_weight_checks'];return
        if state['status'] in ['failed','draining_after_failure','paused','pausing']:raise RuntimeError('Dependency requires attention: '+str(state))
        manager=psutil.Process(state['manager_pid'])
        assert abs(manager.create_time()-state['manager_created'])<.1
        assert 'experiments.policy_finetune_pilot' in ' '.join(manager.cmdline())
        atomic_json(out/'status.json',dict(status='waiting_dependency',dependency=str(dependency),manager_pid=os.getpid(),manager_created=psutil.Process().create_time(),updated_unix=time.time()))
        time.sleep(10)


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
                proc=subprocess.Popen([sys.executable,'-X','utf8','-m','experiments.mte_finetune_extension','worker','--out',str(out),'--job',*map(str,job)],cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
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



def manage(out,config):
    out.mkdir(parents=True,exist_ok=True);lock=out/'manager.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),f)
    try:
        (out/'logs').mkdir(exist_ok=True)
        wait_dependency(out,config)
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
        atomic_json(out/'status.json',dict(status='complete',new_results=45,reused_results=45,source_and_weight_checks=True,updated_unix=time.time()))
    except Exception:
        atomic_json(out/f'manager_failure.{time.time_ns()}.json',dict(error=traceback.format_exc()))
        atomic_json(out/'status.json',dict(status='failed',manager_pid=os.getpid(),updated_unix=time.time()))
        raise
    finally:lock.unlink()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['manage','worker']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--job',nargs=5)
    args=ap.parse_args();config=read(PACKAGE/'configs/mte_finetune_extension.json');out=args.out.resolve()
    if args.command=='manage':manage(out,config)
    else:
        phase,suite,seed,arm,kind=args.job;worker(out,(phase,suite,int(seed),arm,kind),config)


if __name__=='__main__':main()
