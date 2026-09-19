"""Small end-to-end scientific-interface smoke; no performance validation."""
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.mamujoco_route_completion import suite,digest,freeze,read,WORKSPACE
from utils.atomic import atomic_json


def main():
    out=ROOT/'outputs/mamujoco_route_completion_smoke_2026_09_19'
    out.mkdir(parents=True,exist_ok=False)
    c=read(ROOT/'configs/mamujoco_route_completion.json')
    c.update(seeds=[2],reuse_seeds=[],backend_updates={'mamujoco':2},policy_updates=2,
             decoder_updates={'mamujoco':2},max_steps=5,evaluation_seeds={'mamujoco':c['evaluation_seeds']['mamujoco'][:2]})
    c['frontend_config']={**c['frontend_config'],'frontend_updates':2}
    s=suite(2);jobs=[];policies={'2':{}}
    for arm in c['arms']:
        dest=out/'pretraining'/arm
        jobs.append(dict(id='pre_'+arm,stage='pre',seed=2,arm=arm,out=str(dest)))
        policies['2'][arm]=str(dest/'policy/policy.pt')
    for arm in ['mif_matched','mif_raw','global16']:
        jobs.append(dict(id='ground_'+arm,stage='ground',seed=2,arm=arm,budget=4,decoder_seed=202609160,out=str(out/'grounding'/arm)))
    gate=WORKSPACE/c['teacher_gate'];teacher=WORKSPACE/read(gate)['checkpoint']
    allowed=[s['train'],s['dev'],s['endpoints'],s['base'],s['labels']['4'],str(teacher),str(gate)]
    atomic_json(out/'protocol.json',c)
    manifest=dict(jobs=jobs,policies=policies,inputs={p:digest(p) for p in allowed},
        code={str(ROOT/'experiments/mamujoco_route_completion.py'):digest(ROOT/'experiments/mamujoco_route_completion.py')},
        config_hash=digest(out/'protocol.json'),teacher_gate=str(gate),teacher=str(teacher))
    atomic_json(out/'manifest.json',manifest)
    def run(j):
        with (out/(j['id']+'.log')).open('wb') as f:
            r=subprocess.run([sys.executable,'-X','utf8','-m','experiments.mamujoco_route_completion','--out',str(out),'--job',j['id']],cwd=ROOT,stdout=f,stderr=f)
        assert r.returncode==0,f'{j["id"]}: see {out/(j["id"]+".log")}'
        print('PASS',j['id'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[j for j in jobs if j['stage']=='pre']))
    for model in ['simple','graph','mif']:
        a=out/'pretraining'/(model+'_matched');b=out/'pretraining'/(model+'_raw')
        ra,rb=read(a/'result.json'),read(b/'result.json')
        assert ra['parameters']==rb['parameters']
        assert ra['normalization_digest']==rb['normalization_digest']
        ck=torch.load(b/'representation.pt',map_location='cpu',weights_only=False)
        for key in ['mobius','zeta']: assert torch.equal(ck['state_dict'][key],torch.eye(8))
        assert ra['paired_initialization_equal'] and rb['paired_initialization_equal']
    freeze(out,manifest)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[j for j in jobs if j['stage']=='ground']))
    atomic_json(out/'smoke_report.json',dict(status='pass',pretraining_paths=7,grounding_paths=3,
        matched_raw_equal_parameters_and_normalization=True,raw_identity_buffers=True,
        freeze_before_labels=True,updates_per_stage=2,eval_episodes=2,eval_steps=5,
        scope='Interface smoke only; not performance validation or formal results'))
    print('ALL SMOKE CHECKS PASS',flush=True)


if __name__=='__main__':main()
