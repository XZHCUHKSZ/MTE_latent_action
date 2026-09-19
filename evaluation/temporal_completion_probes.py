"""Post-freeze action/interaction readers; diagnostic labels never update policies."""
import json
import numpy as np
import torch
from closed_loop_lam_v1.unified_models import lattice_operators
from evaluation.probes import ridge_probe
from experiments.mpe_temporal_repair import digest
from utils.access import read_rows
from utils.atomic import atomic_json


def score(pred, truth, fit_mean):
    mse=float(np.square(pred.astype(float)-truth).mean())
    null=float(np.square(truth.astype(float)-fit_mean).mean())
    return dict(mse=mse,mean_baseline_mse=null,skill=1-mse/null if null>1e-12 else None)


def evaluate(root,p,workspace,c,progress):
    frozen=root/'pretrain';before=json.loads((frozen/'complete.json').read_text())['checkpoints']
    manifest=json.loads((root/'data/manifest.json').read_text())
    n=p['train_episodes'];d=p['dev_episodes'];cut=d//2
    assert cut>0 and d-cut>0
    raw=manifest['source'];ids=manifest['train_ids']+manifest['dev_ids']
    # Evaluation process only, after all pretraining is frozen.
    actions=read_rows(raw,'actions',ids)[:,1:23]
    methods=c['probe_methods'];features={}
    for name in methods:
        with np.load(frozen/name/'policy/latents.npz') as f:
            features[name]={'R':f['representation_z'],'H':f['deployment_z']}
    for name in c['probe_auxiliaries']:
        features['base+'+name]={stage:np.concatenate([features['base'][stage],features[name][stage]],-1) for stage in ('R','H')}
    rows=[];lambdas=[.0001,.001,.01,.1,1.,10.]
    test=np.arange(n+cut,n+d)
    for budget in c['budgets']:
        val=np.arange(0,budget,4);fit=np.setdiff1d(np.arange(budget),val)
        yy=actions[fit].reshape(-1,2);vy=actions[val].reshape(-1,2);ty=actions[test].reshape(-1,2)
        for name,fs in features.items():
            for stage,z in fs.items():
                zz=lambda ix:z[ix].reshape(-1,z.shape[-1]).astype(float)
                pred,diag,_=ridge_probe(zz(fit),yy,zz(val),vy,zz(test),lambdas)
                rows.append(dict(target='target_action',method=name,stage=stage,budget=budget,
                                 **score(pred,ty,yy.mean(0)),reader=diag))
        progress('action_probe',budget=budget)
    with np.load(frozen/'offset2/bridge/endpoints.npz') as f:
        edge=f['actual']-f['reference'];masks=f['masks']
    M=lattice_operators(torch.tensor(masks))['mobius'].numpy()
    coeff=np.einsum('ij,ethjd->ethid',M,edge,optimize=True)
    rng=np.random.default_rng(918001+p['seed'])
    fi=rng.choice(n*22,min(c['probe_fit_transitions'],n*22),replace=False)
    vi=np.arange(n*22,(n+cut)*22);ti=np.arange((n+cut)*22,(n+d)*22)
    for h in (2,3):
        for partner in (1,2,3):
            ix=np.flatnonzero((masks.sum(1)==1)&masks[:,partner])[0]
            yy=coeff[:,:,h-1,ix].reshape(-1,8).astype(float)
            for name,fs in features.items():
                for stage,z in fs.items():
                    zz=z.reshape(-1,z.shape[-1]).astype(float)
                    pred,diag,_=ridge_probe(zz[fi],yy[fi],zz[vi],yy[vi],zz[ti],lambdas)
                    rows.append(dict(target='learned_single_partner_coefficient',method=name,stage=stage,
                        horizon=h,partner=partner,**score(pred,yy[ti],yy[fi].mean(0)),reader=diag))
        progress('interaction_probe',horizon=h)
    assert all(digest(frozen/f)==v for f,v in before.items())
    atomic_json(root/'probes.json',dict(seed=p['seed'],rows=rows,policy_updates=0,
        split=dict(action_fit='budget prefix minus each fourth episode',action_validation='each fourth budget episode',
                   interaction_fit='training episodes only',interaction_validation=f'first {cut} dev episodes',
                   probe_test=f'last {d-cut} dev episodes; episode-disjoint from reader fit/validation'),
        scope='Development diagnostics. Dev observations previously selected history checkpoints; not untouched confirmation. Native labels are evaluation-only beyond each stated grounding budget.',
        target='Actions use independent native labels; coefficient targets are shared frozen learned tables, not physical interactions.',
        resources='Same ridge family/tuning; feature widths differ for joint baselines and concatenations.',
        weights_unchanged=True))
