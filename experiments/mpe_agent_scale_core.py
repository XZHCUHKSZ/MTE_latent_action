"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""
from mte.method_names import report_text
from mte.method_names import normalize_config, normalize_job

import argparse,hashlib,itertools,json,os,subprocess,sys,time,traceback

from pathlib import Path

import numpy as np

from utils.atomic import atomic_json

def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))

def digest(f):
    with Path(f).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def out_path(r):return r.parents[2]

def root(out,phase,agents,seed):return out/phase/f'a{agents}'/f'seed{seed}'

def location(out,j):
    j = normalize_job(j, 4, "mpe")
    phase,agents,seed,stage,method,route=j;r=root(out,phase,agents,seed)
    return r/stage if stage in ('collect','pretrain') else r/(stage+'_'+method.replace('+','__')+'_'+route)

def jobs(c,phase,stage):
    c = normalize_config(c, "mpe")
    seeds=c['smoke']['seeds'] if phase=='smoke' else c['seeds']
    arms=[('-', '-')] if stage in ('collect','pretrain') else [(m,r) for m in c['methods'] for r in (['supervised'] if m=='bc' else c['routes'])]
    return [(phase,a,s,stage,m,r) for a in c['agents'] for s in seeds for m,r in arms]

def settings(c,j):
    c = normalize_config(c, "mpe")
    j = normalize_job(j, 4, "mpe")
    p=dict(c,seed=j[2],device='cuda')
    if j[0]=='smoke':
        p.update(train_episodes=c['smoke']['train_episodes'],dev_episodes=c['smoke']['dev_episodes'],evaluation_episodes=1)
        for k in ('frontend_updates','readout_updates','representation_updates','history_updates','decoder_updates','bc_updates'):p[k]=c['smoke']['updates']
    return p

def collect(r,dest,j,c,p,progress):
    c = normalize_config(c, "mpe")
    j = normalize_job(j, 4, "mpe")
    from environments.mpe import make_env,teacher_assignment,teacher_forces
    from environments.native_particle import force_to_direction_id
    phase,agents,seed,*_=j;count=p['train_episodes']+p['dev_episodes']
    rank=(c['smoke']['seeds'] if phase=='smoke' else c['seeds']).index(seed)
    first=c['data_seed_start']+(1000000 if phase=='smoke' else 0)+rank*10000+agents*1000
    x=np.empty((count,26,4*agents),np.float32);labels=np.empty((32,22,2),np.float32)
    for i in range(count):
        env=make_env(first+i,agents);assignment=teacher_assignment(env);rng=np.random.default_rng(first+i)
        try:
            for t in range(25):
                x[i,t]=np.concatenate([env.all_agent_pos().ravel(),env.all_landmark_pos().ravel()])
                forces=teacher_forces(env,'high',rng,assignment)
                if i<32 and 1<=t<=22:labels[i,t-1]=forces[0]
                env.step_with_forces(forces,np.array([force_to_direction_id(v) for v in forces]))
            x[i,25]=np.concatenate([env.all_agent_pos().ravel(),env.all_landmark_pos().ravel()])
        finally:env.env.close()
        if i%25==0:progress('collect',episode=i+1,episodes=count)
    data=r/'data';data.mkdir();curator=r/'curator';curator.mkdir()
    np.savez_compressed(data/'train.npz',positions=x[:p['train_episodes']]);np.savez_compressed(data/'dev.npz',positions=x[p['train_episodes']:])
    np.save(curator/'target_b32.npy',labels)
    return dict(first_data_seed=first,episodes=count,target_action_vectors_curated=704,learner_observation_keys=['positions'],scope='curator only; labels not exported to grounding until global pretraining freeze')

def deny_sim(audit):
    def hook(event,args):
        if event=='import' and args[0].split('.')[0] in ('mpe2','gymnasium','mujoco','pettingzoo'):
            audit['violations'].append(args[0]);raise RuntimeError('Simulator import forbidden in training')
    sys.addaudithook(hook)

def check_hashes(base,values):
    for f,h in values.items():assert digest(base/f)==h,f

def pretrain(r,dest,j,c,p,progress):
    c = normalize_config(c, "mpe")
    j = normalize_job(j, 4, "mpe")
    import torch
    from utils.access import guard
    from mte.temporal_mpe_scaled import frontend,bridge
    from training.temporal_family_mpe_disk import train
    from training.history_mpe_scaled import train_history_policy
    from training.composition import restore
    audit=guard(dest,[r/'data/train.npz',r/'data/dev.npz'],True);deny_sim(audit)
    arrays=[]
    for split in ('train','dev'):
        with np.load(r/'data'/f'{split}.npz') as f:assert f.files==['positions'];arrays.append(f['positions'])
    x=np.concatenate(arrays);n=p['train_episodes'];diags={}
    def history(name,z,diag):
        folder=dest/name;folder.mkdir(exist_ok=True)
        d=train_history_policy(x,z,n,p['seed'],p['history_updates'],folder/'policy',progress,device='cuda')
        net,ck=restore(folder/'policy/policy.pt');obs=torch.tensor((x[:2,:23]-ck['obs_mean'])/ck['obs_std']);changed=obs.clone();changed[:,13:]+=17
        with torch.no_grad():err=float((net(obs)[0][:,:13]-net(changed)[0][:,:13]).abs().max())
        assert err==0;diags[name]=dict(representation=diag,history=d,future_prefix_error=err)
        atomic_json(dest/'diagnostics.json',diags)
    offset=dest/'offset2';offset.mkdir()
    z,fd=frontend(x,n,'entity',2,p,offset/'entity',progress);history('base',z[:,:,0],fd)
    a,b,masks,bd=bridge(x,z,n,p,offset/'bridge',progress);atomic_json(offset/'bridge/diagnostics.json',bd)
    for method in ('mif','graph'):
        folder=dest/method;folder.mkdir();zz,diag=train(x,a,b,masks,n,method,p,folder,progress);history(method,zz,diag)
    del a,b
    for variant in ('global16','laom'):
        zz,diag=frontend(x,n,variant,2,p,offset/variant,progress);history(variant,zz,diag)
    assert not audit['violations'];atomic_json(dest/'access_audit.json',audit)
    return dict(checkpoints={str(f.relative_to(dest)):digest(f) for f in dest.rglob('*.pt')},native_action_labels_read=0,simulator_queries=0,frozen_before_grounding=True,diagnostics=diags)

def ground(r,dest,j,c,p,progress):
    c = normalize_config(c, "mpe")
    j = normalize_job(j, 4, "mpe")
    import torch
    from training.history_adaptation import mpe_ground
    phase,agents,seed,stage,method,route=j
    frozen=read(r/'pretrain/result.json');check_hashes(r/'pretrain',frozen['checkpoints'])
    if method!='bc':
        simulator_audit={'violations':[]}
        deny_sim(simulator_audit)
        result=mpe_ground(dest,r,seed,method,route=='adapt',p['decoder_updates'])
        assert not simulator_audit['violations']
        atomic_json(dest/'simulator_access.json',simulator_audit)
    else:
        from utils.access import guard
        from training.grounding_mpe_scaled import train_supervised_history,ScaledGroundedPolicy
        from closed_loop_lam_v1.common import RecurrentPolicy
        audit=guard(dest,[r/'data/train.npz',r/'data/labels_b32.npz'],False);deny_sim(audit)
        with np.load(r/'data/train.npz') as f:x=f['positions']
        with np.load(r/'data/labels_b32.npz') as f:actions=f['actions'];assert np.array_equal(f['local_ids'],np.arange(32))
        val=np.arange(0,32,4);fit=np.setdiff1d(np.arange(32),val);y=np.zeros((len(x),22,2),np.float32);y[fit]=actions[fit]
        net,mu,sd,diag=train_supervised_history(x,y,fit,x[val],actions[val],seed,p['bc_updates'])
        if phase=='smoke' and agents==4:
            from training.grounding_mpe import train_supervised_history as original
            old,om,osd,od=original(x,y,fit,x[val],actions[val],seed,p['bc_updates'])
            assert np.array_equal(mu,om) and np.array_equal(sd,osd)
            assert all(torch.equal(old.state_dict()[k],v) for k,v in net.state_dict().items())
        with torch.no_grad():
            obs=torch.tensor((x[:2,:23]-mu)/sd);changed=obs.clone();changed[:,13:]+=13
            err=float((net(obs)[0][:,:13]-net(changed)[0][:,:13]).abs().max())
        assert err==0
        torch.save(dict(history=net.state_dict(),decoder=None,mean=mu,std=sd,method='bc',budget=32),dest/'decoder.pt')
        assert not audit['violations'];atomic_json(dest/'access_audit.json',audit)
        result=dict(grounding=diag,native_action_labels_read=704,fit_action_labels=528,validation_action_labels=176,partner_labels_read=0,simulator_queries=0,future_prefix_error=err,history_parameters=sum(q.numel() for q in net.parameters()),a4_original_bc_parity=(phase=='smoke' and agents==4))
    # Parent verifies the full freeze outside the strict per-controller read whitelist.
    return result

def evaluate(r,dest,j,c,p,progress):
    c = normalize_config(c, "mpe")
    j = normalize_job(j, 4, "mpe")
    import torch
    from utils.access import guard
    from training.history_adaptation import mpe_net
    from training.grounding_mpe_scaled import ScaledGroundedPolicy
    from evaluation.mpe_scaled import evaluate as rollout
    from closed_loop_lam_v1.common import RecurrentPolicy,ActionDecoder
    phase,agents,seed,stage,method,route=j;ground=r/('ground_'+method.replace('+','__')+'_'+route)
    ckpath=ground/'decoder.pt';allowed=[ckpath]+([] if method=='bc' else [r/'pretrain'/m/'policy/policy.pt' for m in method.split('+')])
    assert digest(ckpath)==read(out_path(r)/(phase+'_ground_freeze.json'))[str(ckpath.relative_to(out_path(r)))]
    audit=guard(dest,allowed,False);cp=torch.load(ckpath,map_location='cpu',weights_only=False)
    if method=='bc':net=RecurrentPolicy(4*agents,2);decoder=None
    else:
        net,_=mpe_net(r,method)
        with torch.no_grad():dim=net(torch.zeros(1,1,4*agents))[0].shape[-1]
        decoder=ActionDecoder(dim,2);decoder.load_state_dict(cp['decoder']);decoder.eval()
    net.load_state_dict(cp['history']);policy=ScaledGroundedPolicy(net,cp['mean'],cp['std'],decoder)
    start=c['evaluation_seed_start']+(1000000 if phase=='smoke' else 0)
    seeds=list(range(start,start+p['evaluation_episodes']))
    result=rollout(policy,seeds,agents=agents,progress=lambda **kw:progress('evaluate',**kw))
    if phase=='smoke' and agents==4:
        from training.grounding_mpe import GroundedPolicy
        from evaluation.mpe import evaluate as original
        old=original(GroundedPolicy(net,cp['mean'],cp['std'],decoder),seeds)
        for key in ('episode_returns','episode_collisions','episode_final_coverage'):assert np.array_equal(old[key],result[key]),key
        result['a4_original_episode_parity']=True
    assert not audit['violations'];atomic_json(dest/'access_audit.json',audit);return result

def worker(out,j,c):
    c = normalize_config(c, "mpe")
    j = normalize_job(j, 4, "mpe")
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    phase,agents,seed,stage,method,route=j;p=settings(c,j);r=root(out,phase,agents,seed);dest=location(out,j);dest.mkdir(parents=True,exist_ok=False)
    def progress(label,**kw):atomic_json(dest/'progress.json',dict(stage=label,updated_unix=time.time(),**kw))
    try:
        result=globals()[stage](r,dest,j,c,p,progress);result.update(complete=True,job=list(j));atomic_json(dest/'result.json',result)
    except Exception:atomic_json(dest/'failure.json',dict(traceback=traceback.format_exc()));raise

def validate(out,j):
    j = normalize_job(j, 4, "mpe")
    d=location(out,j);r=read(d/'result.json');assert r['complete'] and r['job']==list(j)
    if j[3]!='collect':assert not read(d/'access_audit.json')['violations']
    if j[3]=='ground':assert r['native_action_labels_read']==704 and r['partner_labels_read']==0 and r['simulator_queries']==0
    return r

def freeze_pretrain(out,c,phase):
    c = normalize_config(c, "mpe")
    weights={}
    for j in jobs(c,phase,'pretrain'):
        d=location(out,j);r=validate(out,j);check_hashes(d,r['checkpoints'])
        weights.update({str((d/f).relative_to(out)):h for f,h in r['checkpoints'].items()})
    atomic_json(out/(phase+'_pretrain_freeze.json'),weights)
    for j in jobs(c,phase,'pretrain'):
        r=root(out,*j[:3]);dest=r/'data/labels_b32.npz';assert not dest.exists()
        y=np.load(r/'curator/target_b32.npy',allow_pickle=False);assert y.shape==(32,22,2)
        np.savez_compressed(dest,actions=y,local_ids=np.arange(32))

def pairing(out,c,phase):
    c = normalize_config(c, "mpe")
    checks=[]
    for j in jobs(c,phase,'ground'):
        if j[-1]!='frozen':continue
        other=(*j[:-1],'adapt');a=validate(out,j);b=validate(out,other)
        for key in ('history_initial','decoder_initial','normalization','inputs','targets','batches','fit_ids','validation_ids','history_parameters','decoder_parameters'):assert a[key]==b[key],(j,key)
        checks.append(dict(job=j,passed=True))
    atomic_json(out/(phase+'_pairing.json'),checks)
    weights={str((location(out,j)/'decoder.pt').relative_to(out)):digest(location(out,j)/'decoder.pt') for j in jobs(c,phase,'ground')}
    atomic_json(out/(phase+'_ground_freeze.json'),weights)
    return checks

def aggregate(out,c):
    c = normalize_config(c, "mpe")
    from scipy.stats import t
    rows=[dict(agents=j[1],seed=j[2],method=j[4],route=j[5],**validate(out,j)) for j in jobs(c,'formal','evaluate')]
    assert len(rows)==110 and len({(r['agents'],r['seed'],r['method'],r['route']) for r in rows})==110
    def contrast(family,a,m,route,right,right_route):
        delta=[]
        for seed in c['seeds']:
            get=lambda method,rr:next(r['mean_return'] for r in rows if (r['agents'],r['seed'],r['method'],r['route'])==(a,seed,method,rr))
            delta.append(get(m,route)-get(right,right_route))
        d=np.asarray(delta);mean=float(d.mean());half=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5));null=[abs(np.mean(d*np.array(signs))) for signs in itertools.product([-1,1],repeat=5)]
        return dict(family=family,agents=a,left=[m,route],right=[right,right_route],seed_deltas=delta,mean_delta=mean,positive_seeds=int((d>0).sum()),t95=[mean-half,mean+half],exact_p=sum(v>=abs(mean)-1e-12 for v in null)/32)
    contrasts=[]
    for a in c['agents']:
        for m,family in [('base+mif','primary'),('base+graph','secondary')]:
            for route in c['routes']:
                for right in ('base','base+global16','laom','bc'):contrasts.append(contrast(family,a,m,route,right,'supervised' if right=='bc' else route))
        for m in c['methods']:
            if m!='bc':contrasts.append(contrast('adaptation',a,m,'adapt',m,'frozen'))
    for family in ('primary','secondary','adaptation'):
        group=sorted([r for r in contrasts if r['family']==family],key=lambda r:r['exact_p']);prev=0.
        for i,r in enumerate(group):prev=max(prev,min(1.,(len(group)-i)*r['exact_p']));r['holm_p']=prev
    report=dict(records=rows,contrasts=contrasts,source_and_weight_checks=True,scope=c['scope'])
    atomic_json(out/'summary.json',report)
    lines=['# MPE 4/8-agent 固定预算规模与适配对照','','110独立控制身份；五个新上游种子。N700，B32=704个目标动作向量（528拟合+176预算内验证）。先每seed平均100共同episode，再做五seed配对。不同人数回报不混合。','','|族|人数|左−右|均值差|正向seed|t95|精确p|Holm|','|---|---:|---|---:|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['family']}|{r['agents']}|{r['left']} − {r['right']}|{r['mean_delta']:+.6f}|{r['positive_seeds']}/5|{r['t95']}|{r['exact_p']}|{r['holm_p']}|")
    lines+=['','这是既有MPE任务的规模扩展，不是新环境或跨规模零样本迁移。MIF沿用MPE full-visibility route；LAOM是state adapter。Frozen和预算内监督Adapt分别报告，BC有其原训练路径/参数差异。五seed最小精确双侧p=.0625，不把正均值当预定显著。全部结果保留，论文图表和GitHub未自动修改。']
    (out/'RESULTS_CN.md').write_text(report_text(lines, "mpe")+'\n',encoding='utf-8')
    return report
