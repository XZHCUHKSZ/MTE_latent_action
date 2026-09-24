"""Explicit, create-only stages for the manuscript's Frozen Coupled experiment.

Run each training/evaluation stage in a fresh process. This entry has no
background scheduler, continuation, retry, or Adapt experiment.
"""
from mte.method_names import resolve_coupled_ground
import argparse
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=True,indent=2)
def check_freeze(p):
    m=read(p)
    for path,h in m['files'].items():
        if sha(path)!=h:raise ValueError('Frozen artifact changed: '+path)
    return m
def seal(p,files):
    files=sorted(set(Path(q).resolve() for q in files))
    if not files or not all(q.is_file() for q in files):raise ValueError('Incomplete stage artifacts')
    write(p,{'files':{str(q):sha(q) for q in files}})
def note(**kw):pass

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['init','collect','seal-data','pre','seal-pre','labels','ground','seal-ground','evaluate'])
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--data-root',type=Path)
    ap.add_argument('--teacher-root',type=Path)
    ap.add_argument('--teacher',type=int);ap.add_argument('--seed',type=int);ap.add_argument('--arm',help='PC-Field-Solo, other PC-*-Solo/Aux names, or a legacy Frozen arm')
    a=ap.parse_args();r=a.run_dir.resolve()
    if a.stage=='init':
        if not a.data_root or not a.teacher_root:ap.error('init requires --data-root and --teacher-root')
        c=read(P/'configs/coupled_frozen_five_seed.json')
        c.update(data_root=str(a.data_root.resolve()),teacher_root=str(a.teacher_root.resolve()))
        r.mkdir(parents=True,exist_ok=False);write(r/'protocol.json',c)
        files=[p for folder in ['experiments','evaluation','training','mte','visual','environments','utils','closed_loop_lam_v1','third_party/laom'] for p in (P/folder).rglob('*.py')]
        seal(r/'source_freeze.json',files+[r/'protocol.json']);return
    check_freeze(r/'source_freeze.json');c=read(r/'protocol.json')
    from training.coupled_cheetah_frozen import arms,pretrain,ground,export_labels
    from experiments.coupled_frozen_evaluation import evaluate,jobroot
    from experiments.coupled_observation_collection import collect
    assert len(arms())==19 and not any('Adapt' in x for x in arms())
    pairs=[(t,s) for t in c['teacher_seeds'] for s in c['training_seeds']]
    if a.stage=='collect':
        if a.teacher not in c['teacher_seeds']:ap.error('Select one of the two protocol teachers')
        assert len(c['fit_seeds'])==64 and len(c['validation_seeds'])==4
        out=Path(c['data_root'])/f'seed{a.teacher}'/'acquire/data64';out.mkdir(parents=True,exist_ok=False)
        result=collect(c,a.teacher,out,lambda **kw:None);write(out/'result.json',result);return
    if a.stage=='seal-data':
        files=[Path(c['data_root'])/f'seed{t}'/'acquire/data64'/name for t in c['teacher_seeds'] for name in ['rgb.npy','target_actions.npy']]
        seal(r/'data_freeze.json',files);return
    check_freeze(r/'data_freeze.json')
    if a.stage=='seal-pre':
        files=[]
        for t,s in pairs:
            pre=jobroot(r,t,s)/'pre'
            assert read(pre/'result.json')['native_action_labels_read']==0
            assert len(list(pre.glob('*/policy/policy.pt')))==9
            files.extend(pre.rglob('*.pt'));files.extend(pre.glob('*.npz'))
        seal(r/'pretrain_freeze.json',files);return
    if a.stage=='labels':
        check_freeze(r/'pretrain_freeze.json')
        for t in c['teacher_seeds']:
            out=r/'labels'/f'teacher{t}'
            result=export_labels(Path(c['data_root'])/f'seed{t}'/'acquire/data64',out,8)
            write(out/'result.json',result)
        seal(r/'label_freeze.json',list((r/'labels').glob('*/target_actions.npy')));return
    if a.stage=='seal-ground':
        check_freeze(r/'pretrain_freeze.json');check_freeze(r/'label_freeze.json');files=[]
        for t,s in pairs:
            for arm in arms():
                out=jobroot(r,t,s)/'ground'/arm
                result=read(out/'result.json');assert result['arm']==arm and result['updates']==600
                files.append(out/'controller.pt')
                if arm=='idm':files.append(out/'idm.pt')
        assert len(files)==200
        seal(r/'ground_freeze.json',files);return
    if (a.teacher,a.seed) not in pairs:ap.error('Teacher/seed outside the fixed manuscript cohort')
    unit=jobroot(r,a.teacher,a.seed)
    if a.stage=='pre':
        out=unit/'pre';out.mkdir(parents=True,exist_ok=False)
        result=pretrain(c,a.seed,Path(c['data_root'])/f'seed{a.teacher}'/'acquire/data64',out,note)
    elif a.stage=='ground':
        if a.arm is None: ap.error("ground requires --arm")
        a.arm=resolve_coupled_ground(a.arm)
        if a.arm not in arms():ap.error('Select a Frozen arm, bc, or idm')
        check_freeze(r/'pretrain_freeze.json');check_freeze(r/'label_freeze.json')
        out=unit/'ground'/a.arm;out.mkdir(parents=True,exist_ok=False)
        result=ground(c,a.seed,unit/'pre',r/'labels'/f'teacher{a.teacher}'/'target_actions.npy',out,a.arm,note)
        check_freeze(r/'pretrain_freeze.json');check_freeze(r/'label_freeze.json')
    else:
        check_freeze(r/'pretrain_freeze.json');check_freeze(r/'label_freeze.json')
        check_freeze(r/'ground_freeze.json')
        out=unit/'evaluate';out.mkdir(parents=True,exist_ok=False)
        result=evaluate(c,a.teacher,a.seed,r,out,note)
        check_freeze(r/'ground_freeze.json')
    write(out/'result.json',result);check_freeze(r/'source_freeze.json')

if __name__=='__main__':main()
