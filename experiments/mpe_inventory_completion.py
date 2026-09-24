"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import argparse,concurrent.futures,json,os,shutil,subprocess,sys,time,traceback

import numpy as np

import torch

from experiments.mpe_temporal_stages import PACKAGE,digest,exports,load_x,ground

from utils.atomic import atomic_json
from mte.method_names import resolve_method, resolve_mpe_config

def source(c,n):
    item=c['sources'].get(str(n))
    return (PACKAGE/item).resolve() if item else None

def p_for(c,n):
    c=resolve_mpe_config(c)
    p=json.loads((PACKAGE/'configs/mpe_temporal_full_scale.json').read_text());p.update(c['overrides'])
    p.update(train_episodes=n,budgets=c['budgets'] if n==c['full_n'] else [c['scaling_budget']],
             control_configs=c['methods'],primary_contrasts=[],scope='Full paper inventory on repaired temporal interface; existing valid cells reused.')
    return p

def existing(c,n,seed,b,method):
    method=resolve_method(method, "mpe")
    # Prefer the supplemental outputs; earlier full-scale cells live in their own immutable run.
    roots=[source(c,n)]
    if n==c['full_n']:roots.append((PACKAGE/c['original_full']).resolve())
    for root in roots:
        if root is None:continue
        path=root/f'seed{seed}/grounding/budget_{b}'/f'b{b}_{method.replace("+","__")}.json'
        if path.exists():return path
    return None

def setup(args,c):
    c=resolve_mpe_config(c)
    for n in c['sizes']:
        p=p_for(c,n);dest=args.out/f'n{n}';exports(args.workspace,dest,p)
        if source(c,n):
            sp=json.loads((source(c,n)/'protocol.json').read_text())
            for key in ('seeds','data_seed_map','train_episodes','dev_episodes','frontend_updates','readout_updates',
                        'representation_updates','history_updates','decoder_updates','batch','representation_batch',
                        'evaluation_seed_start','evaluation_episodes'):
                assert sp[key]==p[key],f'Incompatible source {n}/{key}'
            for seed in p['seeds']:
                root=dest/f'seed{seed}';target=root/'pretrain';src=source(c,n)/f'seed{seed}/pretrain'
                if target.exists():continue
                ck=json.loads((src/'complete.json').read_text())['checkpoints']
                assert all(digest(src/f)==h for f,h in ck.items())
                shutil.copytree(src,target,ignore=shutil.ignore_patterns('complete.json'))
                atomic_json(root/'source_checkpoint_manifest.json',dict(source=str(src),checkpoints=ck))
        atomic_json(dest/'protocol.json',p)

def pretrain(root,p,c,progress):
    c=resolve_mpe_config(c)
    from utils.access import guard
    from training.history_mpe import train_history_policy
    from training.composition import restore
    from mte.temporal_mpe import frontend,bridge
    from training.temporal_family_mpe import train as route
    from training.temporal_completion_mpe import train_original
    from training.temporal_inventory_mpe import train as baseline
    out=root/'pretrain';out.mkdir(exist_ok=True)
    audit=guard(out,[root/'data/train.npz',root/'data/dev.npz'],True)
    def deny(event,args):
        if event=='import' and args[0].split('.')[0] in ('mpe2','gymnasium','mujoco','pettingzoo'):
            raise RuntimeError('No simulator in pretraining')
    sys.addaudithook(deny);x,n=load_x(root,p);diagnostics={}
    def history(name,z,info):
        folder=out/name;folder.mkdir(exist_ok=True)
        h=train_history_policy(x,z,n,p['seed'],p['history_updates'],folder/'policy',progress)
        net,ck=restore(folder/'policy/policy.pt')
        with torch.no_grad():
            xx=torch.tensor((x[:2,:23]-ck['obs_mean'])/ck['obs_std']);yy=xx.clone();yy[:,13:]+=13
            err=float((net(xx)[0][:,:13]-net(yy)[0][:,:13]).abs().max())
        assert err==0
        diagnostics[name]=dict(representation=info,history=h,future_prefix_error=err)
        atomic_json(out/'inventory_diagnostics.json',diagnostics)
    if not (out/'offset2').exists():
        f=out/'offset2';f.mkdir();z,info=frontend(x,n,'entity',2,p,f/'entity',progress)
        history('base',z[:,:,0],info);a,b,masks,info=bridge(x,z,n,p,f/'bridge',progress)
        atomic_json(f/'bridge/diagnostics.json',info)
    with np.load(out/'offset2/bridge/endpoints.npz') as f:a,b,masks=f['actual'],f['reference'],f['masks']
    for arm in c['latent_methods']:
        if (out/arm/'policy/policy.pt').exists():continue
        progress('new_latent',arm=arm)
        if arm in ('lapo','laom','global16'):
            z,info=frontend(x,n,arm,2,p,out/'offset2'/arm,progress)
        elif arm=='entity_joint':
            with np.load(out/'offset2/entity/codes.npz') as f:z=f['z'].reshape(len(x),22,64)
            info=dict(source='same frozen four entity codes concatenated; original entity_joint interface')
        else:
            folder=out/arm;folder.mkdir()
            if arm in ('simple','graph','mif'):z,info=route(x,a,b,masks,n,arm,p,folder,progress)
            elif arm in ('edge_h2','edge_h3','tree'):z,info=train_original(x,a,b,masks,n,arm,p,folder,progress)
            else:z,info=baseline(x,a,b,masks,n,arm,p,folder,progress)
        history(arm,z,info)
    assert not audit['violations'];atomic_json(out/'inventory_access_audit.json',audit)
    atomic_json(out/'complete.json',dict(seed=p['seed'],frozen_before_grounding=True,native_action_labels_read=0,
        simulator_queries=0,checkpoints={str(f.relative_to(out)):digest(f) for f in out.rglob('*.pt')}))

def special_ground(root,p,method,b,progress):
    from training.composition import restore,FrozenPair
    from training.grounding_mpe import GroundedPolicy,train_decoder
    from training.temporal_inventory_mpe import train_idm_t2
    from evaluation.mpe import evaluate
    x,n=load_x(root,p);dest=root/f'grounding/budget_{b}';dest.mkdir(parents=True,exist_ok=True)
    with np.load(root/f'data/labels_b{b}.npz') as f:actions=f['actions']
    val=np.arange(0,b,4);fit=np.setdiff1d(np.arange(b),val);decoder=None
    if method=='idm':
        net,mu,sd,diag=train_idm_t2(x[:n],actions,fit,val,p['seed'],p['decoder_updates'],p['history_updates'])
    else:
        assert method=='base_duplicate';base,ck=restore(root/'pretrain/base/policy/policy.pt')
        net=FrozenPair(base,duplicate=True);mu,sd=ck['obs_mean'],ck['obs_std']
        with torch.no_grad():z=net(torch.tensor((x[:b,:23]-mu)/sd))[0][:,1:].numpy()
        assert np.array_equal(z[...,:16],z[...,16:])
        decoder,diag=train_decoder(z,actions,fit,val,p['seed'],p['decoder_updates'])
    progress('rollout',method=method,budget=b)
    row=dict(seed=p['seed'],budget=b,method=method,grounding=diag,
             **evaluate(GroundedPolicy(net,mu,sd,decoder),list(range(p['evaluation_seed_start'],p['evaluation_seed_start']+p['evaluation_episodes']))))
    path=dest/f'b{b}_{method}.json';atomic_json(path,row)
    torch.save(dict(history=net.state_dict(),decoder=None if decoder is None else decoder.state_dict(),mean=mu,std=sd,method=method,budget=b),path.with_suffix('.pt'))

def ground_missing(root,p,c,n,b,progress):
    c=resolve_mpe_config(c)
    frozen=root/'pretrain';ck=json.loads((frozen/'complete.json').read_text())['checkpoints']
    assert all(digest(frozen/f)==h for f,h in ck.items())
    todo=[m for m in c['methods'] if existing(c,n,p['seed'],b,m) is None]
    normal=[m for m in todo if m not in ('idm','base_duplicate')]
    if normal:ground(root,dict(p,budgets=[b],_ground_budget=b,control_configs=normal),progress)
    for m in todo:
        if m in ('idm','base_duplicate') and not (root/f'grounding/budget_{b}/b{b}_{m}.json').exists():special_ground(root,p,m,b,progress)
    assert all(digest(frozen/f)==h for f,h in ck.items())
    atomic_json(root/f'grounding/budget_{b}/inventory_complete.json',dict(new_methods=todo,reused=[m for m in c['methods'] if m not in todo],weights_unchanged=True))

def worker(args,c):
    c=resolve_mpe_config(c)
    root=args.out/f'n{args.n}/seed{args.seed}';p=dict(p_for(c,args.n),seed=args.seed)
    torch.set_num_threads(p['threads_per_process']);name=args.stage+(f'_b{args.budget}' if args.budget else '')
    def progress(stage,**kw):atomic_json(root/f'status_{name}.json',dict(stage=stage,time=time.time(),**kw))
    try:
        if args.stage=='pretrain':pretrain(root,p,c,progress)
        elif args.stage=='ground':ground_missing(root,p,c,args.n,args.budget,progress)
        else:
            from evaluation.temporal_completion_probes import evaluate
            evaluate(root,p,args.workspace,c,progress)
        progress('complete')
    except BaseException:atomic_json(root/f'failure_{name}.json',dict(traceback=traceback.format_exc()));raise

def report(out,c):
    c=resolve_mpe_config(c)
    records=[];newcount=0;missing=[];seeds=p_for(c,c['full_n'])['seeds']
    for n in c['sizes']:
        for s in seeds:
            for b in p_for(c,n)['budgets']:
                for m in c['methods']:
                    old=existing(c,n,s,b,m);path=old or out/f'n{n}/seed{s}/grounding/budget_{b}/b{b}_{m.replace("+","__")}.json'
                    if not path.exists():missing.append([n,s,b,m]);continue
                    r=json.loads(path.read_text());assert (r['seed'],r['budget'],r['method'])==(s,b,m)
                    assert len(r['episode_returns'])==p_for(c,n)['evaluation_episodes']
                    assert np.isclose(np.mean(r['episode_returns']),r['mean_return'])
                    records.append(dict(n=n,seed=s,budget=b,method=m,mean_return=r['mean_return'],source=str(path),reused=bool(old)))
                    newcount+=not bool(old)
    rows=[]
    for n,b,m in sorted({(r['n'],r['budget'],r['method']) for r in records}):
        rr=[r['mean_return'] for r in records if (r['n'],r['budget'],r['method'])==(n,b,m)]
        rows.append(dict(n=n,budget=b,method=m,seeds=len(rr),mean=float(np.mean(rr)),sd=float(np.std(rr,ddof=1)) if len(rr)>1 else None))
    counts=dict(completed_cells=len(records),expected_cells=len(records)+len(missing),new_cells=newcount,reused_cells=len(records)-newcount)
    atomic_json(out/'summary.json',dict(**counts,means=rows,records=records,missing=missing,aliases=c['paper_aliases']))
    lines=['# 原论文模型清单：MPE时间修正版','',json.dumps(counts),'','29个论文标签对应28个唯一配置；同一个Base别名不重复当样本。','',
           '|N|B|配置|种子数|回报 ↑|','|---:|---:|---|---:|---:|']
    for r in rows:lines.append(f"|{r['n']}|{r['budget']}|{r['method']}|{r['seeds']}|{r['mean']:.5f}|")
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return counts
