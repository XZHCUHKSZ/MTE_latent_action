"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import json

from pathlib import Path

import shutil

import sys

import time

import traceback

import numpy as np

import torch

from experiments.mpe_temporal_stages import PACKAGE, digest, exports, load_x, ground

from utils.atomic import atomic_json

def plan(config, n):
    p = json.loads((PACKAGE/'configs/mpe_temporal_full_scale.json').read_text())
    p.update(config['overrides'])
    p.update(train_episodes=n, inverse_offsets=[2], primary_contrasts=[],
             scope='Supplement to timing repair; new outputs, unchanged original methods; development data reused.')
    if n == config['full_n']:
        p.update(budgets=config['budgets'], control_configs=config['new_control_configs'])
    else:
        p.update(budgets=[config['scaling_budget']], control_configs=config['scaling_control_configs'])
    return p

def pretrain(root, p, config, progress):
    from utils.access import guard
    out = root/'pretrain'; out.mkdir(exist_ok=True)
    audit = guard(out, [root/'data/train.npz', root/'data/dev.npz'], pretraining=True)
    def deny_sim(event, args):
        if event == 'import' and args[0].split('.')[0] in {'mpe2', 'gymnasium', 'mujoco', 'pettingzoo'}:
            raise RuntimeError('No simulator in observation-only training')
    sys.addaudithook(deny_sim)
    from mte.temporal_mpe import frontend, bridge
    from training.temporal_family_mpe import train as route_train
    from training.temporal_completion_mpe import train_original
    from training.history_mpe import train_history_policy
    from training.composition import restore
    x, n = load_x(root, p); diagnostics = {}
    def history(arm, z, info):
        folder=out/arm; folder.mkdir(exist_ok=True)
        h=train_history_policy(x,z,n,p['seed'],p['history_updates'],folder/'policy',progress)
        net,ck=restore(folder/'policy/policy.pt')
        with torch.no_grad():
            xx=torch.tensor((x[:2,:23]-ck['obs_mean'])/ck['obs_std']);yy=xx.clone();yy[:,13:]+=19
            error=float((net(xx)[0][:,:13]-net(yy)[0][:,:13]).abs().max())
        assert error == 0
        diagnostics[arm]=dict(representation=info,history=h,future_prefix_error=error)
        atomic_json(out/'supplement_diagnostics.json',diagnostics)
    if n == config['full_n']:
        with np.load(out/'offset2/bridge/endpoints.npz') as f:a,b,masks=f['actual'],f['reference'],f['masks']
        route_arms=['simple_raw']
        original_arms=config['full_original_arms']
    else:
        froot=out/'offset2';froot.mkdir()
        z,info=frontend(x,n,'entity',2,p,froot/'entity',progress);history('base',z[:,:,0],info)
        a,b,masks,info=bridge(x,z,n,p,froot/'bridge',progress)
        atomic_json(froot/'bridge/diagnostics.json',info)
        route_arms=['simple','graph','mif']
        original_arms=config['scaling_original_arms']
    for arm in route_arms:
        folder=out/arm;folder.mkdir()
        z,info=route_train(x,a,b,masks,n,arm,p,folder,progress);history(arm,z,info)
    for arm in original_arms:
        folder=out/arm;folder.mkdir()
        z,info=train_original(x,a,b,masks,n,arm,p,folder,progress);history(arm,z,info)
    if n != config['full_n']:
        for arm in ('lapo','laom','global16'):
            z,info=frontend(x,n,arm,2,p,out/'offset2'/arm,progress);history(arm,z,info)
    assert not audit['violations']
    atomic_json(out/'supplement_access_audit.json',audit)
    atomic_json(out/'complete.json',dict(checkpoints={str(f.relative_to(out)):digest(f) for f in out.rglob('*.pt')},
        seed=p['seed'],native_action_labels_read=0,simulator_queries=0,frozen_before_grounding=True))

def worker(args,c):
    p=dict(plan(c,args.n),seed=args.seed);root=args.out/f'n{args.n}'/f'seed{args.seed}'
    torch.set_num_threads(p['threads_per_process'])
    name=args.stage+(f'_b{args.budget}' if args.budget else '')
    def progress(stage,**kw):atomic_json(root/f'status_{name}.json',dict(stage=stage,time=time.time(),**kw))
    try:
        if args.stage=='pretrain':pretrain(root,p,c,progress)
        elif args.stage=='ground':ground(root,dict(p,budgets=[args.budget],_ground_budget=args.budget),progress)
        elif args.stage=='probes':
            from evaluation.temporal_completion_probes import evaluate
            evaluate(root,p,args.workspace,c,progress)
        elif args.stage=='donor':
            from evaluation.temporal_donor_effects import evaluate
            evaluate(root,p,args.workspace,c,progress)
        progress('complete')
    except BaseException:
        atomic_json(root/f'failure_{name}.json',dict(traceback=traceback.format_exc()));raise

def prepare(args,c):
    source=Path(c['source_run']).resolve()
    assert source != args.out and not args.out.is_relative_to(source), 'New outputs must not modify completed source run'
    sp=json.loads((source/'protocol.json').read_text())
    fp=plan(c,c['full_n'])
    for key in ('seeds','data_seed_map','train_episodes','dev_episodes','frontend_updates','readout_updates',
                'representation_updates','history_updates','decoder_updates','batch','representation_batch',
                'evaluation_seed_start','evaluation_episodes'):
        assert sp[key]==fp[key], f'Reused evidence has incompatible {key}'
    for n in [c['full_n'],*c['scales']]:
        dest=args.out/f'n{n}';p=plan(c,n);exports(args.workspace,dest,p)
        atomic_json(dest/'protocol.json',p)
        if n==c['full_n']:
            for seed in p['seeds']:
                src=Path(c['source_run'])/f'seed{seed}'/'pretrain';target=dest/f'seed{seed}'/'pretrain'
                if target.exists():continue
                old=json.loads((src/'complete.json').read_text())
                assert all(digest(src/f)==h for f,h in old['checkpoints'].items())
                # Isolated copies: observation-only assets only, no labels/decoder checkpoints.
                shutil.copytree(src,target,ignore=shutil.ignore_patterns('complete.json'))
                atomic_json(dest/f'seed{seed}'/'reused_source.json',dict(source=str(src),checkpoint_hashes=old['checkpoints']))

def report(out,c):
    from evaluation.temporal_completion_report import summarize
    return summarize(out,c)
