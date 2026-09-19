"""Native trainer parity plus four-arm end-to-end interface smoke."""
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from experiments.mamujoco_mif_visibility import setup, freeze, MODULE
from experiments.mamujoco_route_completion import read, digest
from utils.atomic import atomic_json


def parity():
    from mte.structured_controls import train as native_train
    from training.mif_visibility_control import representation
    torch.set_num_threads(1)
    rng = np.random.default_rng(550)
    a = rng.normal(size=(3,4,3,8,4,46)).astype(np.float32)
    b = rng.normal(size=a.shape).astype(np.float32)
    masks = np.array([[0, i&1, (i>>1)&1, (i>>2)&1] for i in range(8)],np.float32)
    valid = np.ones((3,4),bool)
    adj = np.ones((4,4),np.float32)
    z0, _, c0 = native_train(a,b,valid,2,masks,adj,'mamujoco',71,2,'mif_edge',lambda *a,**k:None)
    z1, _, c1 = representation(a,b,valid,2,masks,adj,71,2,'masked_mobius',lambda *a,**k:None)
    error = max(float((c0['state_dict'][k].float()-c1['state_dict'][k].float()).abs().max()) for k in c0['state_dict'])
    assert error==0 and np.array_equal(z0,z1)
    return dict(native_two_update_parameter_max_error=error, native_latents_exact=True)


def main():
    out = PACKAGE/'outputs/mamujoco_mif_visibility_smoke_2026_09_19'
    assert not out.exists()
    c = read(PACKAGE/'configs/mamujoco_mif_visibility.json')
    c.update(seeds=[0],budgets=[4],decoder_seeds=[202609160],representation_updates=2,policy_updates=2,
        decoder_updates={'mamujoco':2},max_steps=5,evaluation_seeds={'mamujoco':c['evaluation_seeds']['mamujoco'][:2]})
    config = PACKAGE/'outputs/mamujoco_mif_visibility_smoke_protocol_2026_09_19.json'
    assert not config.exists()
    atomic_json(config,c)
    setup(out,config)
    native = parity()
    manifest = read(out/'manifest.json')
    def run(j):
        with (out/(j['id']+'.log')).open('wb') as f:
            proc = subprocess.run([sys.executable,'-X','utf8','-m',MODULE,'--out',str(out),'--job',j['id']],
                                  cwd=PACKAGE,stdout=f,stderr=f)
        assert proc.returncode==0, j['id']
        print('PASS',j['id'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,[j for j in manifest['jobs'] if j['stage']=='pre']))
    freeze(out,manifest,native_parity=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,[j for j in manifest['jobs'] if j['stage']=='ground']))
    atomic_json(out/'smoke_report.json',dict(status='pass',**native,pretraining_paths=4,grounding_paths=4,
        paired_masks_initialization_normalization_and_parameters=True,raw_identity_buffers=True,
        frozen_before_grounding=True,updates_per_stage=2,evaluation_episodes=2,evaluation_steps=5,
        scope='Interface smoke only; full-length native archive parity is required again before formal grounding.'))
    print('ALL SMOKE CHECKS PASS',flush=True)


if __name__=='__main__': main()
