"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import hashlib

import itertools

import json

from pathlib import Path

import shutil

import subprocess

import sys

import time

import numpy as np

from utils.atomic import atomic_json

PACKAGE = Path(__file__).resolve().parents[1]

WORKSPACE = PACKAGE.parents[1]

ARCHIVE = WORKSPACE/'visual_multiagent_2026_09_10/seed5_v1/runtime'

REPAIRED = WORKSPACE/'visual_multiagent_2026_09_10/seed5_render_repair/runtime'

LOW = PACKAGE/'outputs/visual_low_label_2026_09_19_r1'

PIXEL = WORKSPACE/'visual_multiagent_2026_09_10/pixel_pipeline_v1'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def budget_root(out, phase, seed, budget):
    return out/phase/f'seed{seed}'/f'b{budget}'

def job_path(out, job):
    phase, seed, budget, arm, kind = job
    return budget_root(out, phase, seed, budget)/(kind+'_'+arm)

def compare_checkpoint(new, old):
    import torch
    differences = []

    def walk(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                walk(a[k], b[k])
        elif torch.is_tensor(a) or isinstance(a, np.ndarray):
            a = a.detach().cpu().numpy() if torch.is_tensor(a) else a
            b = b.detach().cpu().numpy() if torch.is_tensor(b) else b
            assert a.shape == b.shape and a.dtype == b.dtype
            differences.append(float(np.max(np.abs(a.astype(float)-b.astype(float)))))
        else:
            assert a == b
    walk(torch.load(new, map_location='cpu', weights_only=False), torch.load(old, map_location='cpu', weights_only=False))
    error = max(differences, default=0.)
    assert error <= 1e-6, f'B8 weight mismatch: {new}: {error}'
    return error

def source_snapshot(out, p):
    frozen = read(ARCHIVE/'global_freeze.json')
    assert frozen['complete'] and frozen['training_seeds'] == p['seeds']
    for f, h in frozen['source_hashes'].items():
        assert digest(f) == h, f
    files = {Path(f) for f in frozen['source_hashes']}
    files.add(ARCHIVE/'global_freeze.json')
    for folder in ['experiments', 'training', 'evaluation', 'visual', 'environments', 'mte', 'utils', 'closed_loop_lam_v1', 'third_party']:
        files.update((PACKAGE/folder).rglob('*.py'))
    files.add(PACKAGE/'configs/visual_supervision_inventory.json')
    files.add(PIXEL/'pilot_protocol.json')
    for split in ['train', 'dev']:
        files.update((PIXEL/f'runtime/pilot/validate/target_labels/{split}').glob('*.npy'))
    config = read(PIXEL/'pilot_protocol.json')
    gatepath = WORKSPACE/config['teacher_gate']
    files.add(gatepath)
    cp = Path(read(gatepath)['checkpoint'])
    files.add(cp if cp.is_absolute() else WORKSPACE/cp)
    images = WORKSPACE/'visual_multiagent_2026_09_10/laom_complexity_v1/assets/DAVIS/JPEGImages/480p'
    for video in config['background_videos']['dev']:
        frames = list((images/video).glob('*.jpg'))
        assert len(frames) >= 2
        files.update(frames)
    for seed in p['seeds']:
        src = ARCHIVE/str(seed)
        assert read(src/'freeze/result.json')['all_sources_frozen']
        files.update([src/'features/features.npz', src/'freeze/result.json'])
        for stage in ['frontend', 'features'] + ['pre_'+m for m in ['visual_laom_target','edge_cara','edge_cara_mobius_simple','edge_cara_mobius_graph','edge_cara_mobius_tree','edge_cara_mif','matched_cf_deepsets','random_edge_encoder','continuous_lam','lapo_state_adapter','laom_state_adapter']]:
            ap = src/stage/'access_audit.json'; rp = src/stage/'result.json'
            assert not read(ap)['violations']
            r = read(rp)
            assert r['native_action_labels_read'] == 0 and r['simulator_queries'] == 0
            files.update([ap, rp])
        for arm in p['arms']:
            g = src/('ground_'+arm)
            assert read(g/'result.json')['complete']
            files.update([g/'decoder.pt', g/'result.json'])
            if arm == 'bc_idm_relabel':
                files.add(g/'idm.pt')
            else:
                files.add(g/'fit_predictions.npy')
            r = REPAIRED/str(seed)/('evaluate_'+arm)
            assert read(r/'result.json')['complete']
            files.update([r/'result.json', r/'episode_908601.npz'])
    assert read(LOW/'status.json')['status']=='complete' and read(LOW/'summary.json')['source_and_weight_checks']
    files.update([LOW/'summary.json',LOW/'source_manifest.json',LOW/'ground_freeze.json',LOW/'qualification.json'])
    files.update((LOW/'formal').rglob('result.json'))
    files.update((LOW/'formal').rglob('decoder.pt'))
    files.update((LOW/'formal').rglob('fit_predictions.npy'))
    files.update((LOW/'formal').rglob('access_audit.json'))
    files.update((out/'assets').rglob('*'))
    return {str(f.resolve()): digest(f) for f in sorted(files) if f.is_file()}

def verify_snapshot(out):
    data = read(out/'source_manifest.json')
    for f, h in data.items():
        assert digest(f) == h, f'Source changed: {f}'

def worker(a, p):
    import torch
    from training import visual_label_budget as B
    phase, seed, budget, arm, kind = a.job
    seed, budget = int(seed), int(budget)
    job = (phase, seed, budget, arm, kind)
    dest = job_path(a.out, job)
    dest.mkdir(parents=True, exist_ok=False)
    root = dest.parent
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    B.V.C.seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic()
    if kind == 'ground':
        B.configure(seed, root, budget, updates=2 if phase == 'smoke' else 600)
        B.V.progress = lambda phase, **kw: atomic_json(dest/'progress.json', dict(phase=phase, updated_unix=time.time(), **kw))
        if arm.startswith('mif_'):
            from training.visual_supervision_control import ground
            r=ground(dest,arm)
        else:
            r = B.idm(dest) if arm == 'bc_idm_relabel' else B.ground(dest, arm)
            if arm=='bc_full1600':
                r['supervised_vectors_per_update']=budget*200
                r['total_label_presentations']=B.UPDATES*budget*200
                r['budget_adaptation']='full batch B*200; original arm identifier retained'

    else:
        assert read(root/('ground_'+arm)/'result.json')['complete']
        if phase == 'formal':
            assert read(a.out/'qualification.json')['passed']
            frozen = read(a.out/'ground_freeze.json')
            cp = root/('ground_'+arm)/'decoder.pt'
            assert digest(cp) == frozen[str(cp.relative_to(a.out))]
        if arm.startswith('mif_'):
            from training.visual_supervision_control import controller
            B.V.controller=controller
        from evaluation.visual_label_budget import evaluate
        r = evaluate(dest, arm, seed, root, a.out/'assets', [908601] if phase in ('parity','smoke') else p['evaluation_seeds'])
        if phase in ('parity','smoke'):
            r['mean_return'] = r['rows'][0]['return_value']
    r.update(complete=True, training_seed=seed, budget=budget, phase=phase,
             seconds=time.monotonic()-start)
    atomic_json(dest/'result.json', r)

def validate_result(out, job):
    dest = job_path(out, job); phase, seed, budget, arm, kind = job
    r = read(dest/'result.json'); audit = read(dest/'access_audit.json')
    assert r['complete'] and (r['training_seed'], r['budget'], r['arm']) == (seed, budget, arm)
    assert not audit['violations']
    if kind == 'ground':
        assert r['native_action_labels_read'] == budget*200 and r['partner_labels_read'] == 0 and r['simulator_queries'] == 0
        labels = [Path(x) for x in audit['numeric_reads'] if 'target_labels' in x]
        assert {x.name for x in labels} == {f'{i:04d}.npy' for i in range(budget)}
        assert all(x.parent.name == 'train' for x in labels)
    if phase=='fair_parity':
        from training.visual_supervision_control import base_arm
        import torch
        base=base_arm(arm)
        source=ARCHIVE/str(seed)/('ground_'+base) if budget==8 else LOW/'formal'/f'seed{seed}'/f'b{budget}'/('ground_'+base)
        ck=torch.load(dest/'decoder.pt',map_location='cpu',weights_only=False)
        ref=torch.load(source/'decoder.pt',map_location='cpu',weights_only=False)
        err=max(float((ck['state_dict'][k]-v).abs().max()) for k,v in ref['state_dict'].items())
        assert err<=1e-6
        pred_error=float(np.max(np.abs(np.load(dest/'fit_predictions.npy')-np.load(source/'fit_predictions.npy'))))
        assert pred_error<=1e-6,pred_error
        atomic_json(dest/'parity.json',dict(passed=True,decoder_max_abs=err,fit_max_abs=pred_error,source=str(source)))
    if phase == 'parity' and kind == 'ground':
        source = ARCHIVE/str(seed)/('ground_'+arm)
        errors = {'decoder':compare_checkpoint(dest/'decoder.pt', source/'decoder.pt')}
        if arm == 'bc_idm_relabel':
            errors['idm'] = compare_checkpoint(dest/'idm.pt', source/'idm.pt')
        else:
            error = float(np.max(np.abs(np.load(dest/'fit_predictions.npy')-np.load(source/'fit_predictions.npy'))))
            assert error <= 1e-6
            errors['fit_predictions'] = error
        atomic_json(dest/'parity.json', dict(passed=True, max_abs_errors=errors))
    if phase == 'parity' and kind == 'evaluate':
        reference = REPAIRED/str(seed)/('evaluate_'+arm)/'episode_908601.npz'
        with np.load(reference) as old, np.load(dest/'episode_908601.npz') as new:
            checks = {k:bool(np.array_equal(old[k],new[k])) for k in old.files}
        atomic_json(dest/'parity.json', dict(passed=all(checks.values()), exact_episode_checks=checks))
        assert all(checks.values()), f'Independent B8 replay differs: {dest}'

def phases(p):
    smoke=[('smoke',p['seeds'][0],1,f'mif_{c}_{i}_{a}','ground') for c in ['solo','aux'] for i in ['pretrained','random'] for a in ['frozen','trainable']]
    fair=[('fair_parity',s,b,f'mif_{c}_pretrained_frozen','ground') for s in p['seeds'] for b in p['budgets'] for c in ['solo','aux']]
    parity=[('parity',s,8,a,'ground') for s in p['seeds'] for a in p['missing_arms']]
    replay=[(*j[:4],'evaluate') for j in parity]
    ground=[('formal',s,b,a,'ground') for b in p['budgets'] for s in p['seeds'] for a in (p['fair_arms']+(p['missing_arms'] if b!=8 else []))]
    return [('factorial_smoke',smoke),('frozen_adapter_parity',fair),('b8_ground_parity',parity),('b8_replay',replay),('low_budget_ground',ground),('low_budget_evaluate',[(*j[:4],'evaluate') for j in ground])]

def process_created(pid):
    """A worker can exit between poll and status serialization."""
    import psutil
    try:
        return psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return None

def factorial_checks(out,p):
    checks=[]
    for seed in p['seeds']:
        for budget in p['budgets']:
            for c in ['solo','aux']:
                rows=[]
                for i,a in [('pretrained','frozen'),('pretrained','trainable'),('random','frozen'),('random','trainable')]:
                    phase='fair_parity' if (i,a)==('pretrained','frozen') else 'formal'
                    path=job_path(out,(phase,seed,budget,f'mif_{c}_{i}_{a}','ground'))/'training_identity.json'
                    rows.append(read(path))
                for key in ['decoder_initial','normalization','batches','targets','inputs','total_history_parameters','decoder_parameters']:
                    assert len({r[key] for r in rows})==1,(seed,budget,c,key)
                assert rows[0]['history_initial']==rows[1]['history_initial']
                assert rows[2]['history_initial']==rows[3]['history_initial']
                assert rows[0]['history_initial']!=rows[2]['history_initial']
                checks.append(dict(seed=seed,budget=budget,composition=c,passed=True))
    atomic_json(out/'factorial_checks.json',checks)

def aggregate(out,p):
    from scipy.stats import t
    from training.visual_supervision_control import base_arm
    records=[]
    for b in p['budgets']:
        for seed in p['seeds']:
            for arm in p['arms']+p['fair_arms']:
                reused=arm in p['arms'] and (b==8 or arm in p['existing_low_label_arms'])
                f=(REPAIRED/str(seed)/('evaluate_'+arm)/'result.json') if reused and b==8 else ((LOW/'formal'/f'seed{seed}'/f'b{b}'/('evaluate_'+arm)/'result.json') if reused else job_path(out,('formal',seed,b,arm,'evaluate'))/'result.json')
                r=read(f);assert [x['seed'] for x in r['rows']]==p['evaluation_seeds']
                mean=float(np.mean([x['return_value'] for x in r['rows'] if x['seed'] in p['primary_evaluation_seeds']]))
                records.append(dict(seed=seed,budget=b,arm=arm,mean_return=mean,reused=reused,source=str(f)))
    assert len({(r['seed'],r['budget'],r['arm']) for r in records})==620
    assert sum(r['reused'] for r in records)==260
    values={(r['seed'],r['budget'],r['arm']):r['mean_return'] for r in records}
    contrasts=[]
    for c in ['solo','aux']:
        pre_f=base_arm(f'mif_{c}_pretrained_frozen');pre_t=f'mif_{c}_pretrained_trainable'
        rand_f=f'mif_{c}_random_frozen';rand_t=f'mif_{c}_random_trainable'
        specs=[('history_action_access',[(pre_t,1),(pre_f,-1)]),('pretrained_vs_random_trainable',[(pre_t,1),(rand_t,-1)]),('initialization_by_action_access',[(pre_t,1),(pre_f,-1),(rand_t,-1),(rand_f,1)])]
        family=[]
        for name,terms in specs:
            d=np.array([np.mean([sum(sign*values[(seed,b,arm)] for arm,sign in terms) for b in p['new_budgets']]) for seed in p['seeds']])
            radius=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5))
            null=np.abs(np.array(list(itertools.product([-1,1],repeat=5)))@d/5)
            family.append(dict(composition=c,name=name,terms=terms,seed_deltas=d.tolist(),mean_delta=float(d.mean()),positive_seeds=int((d>0).sum()),t95=[float(d.mean()-radius),float(d.mean()+radius)],exact_p=float((null>=abs(d.mean())-1e-12).mean())))
        bound=0
        for i,r in enumerate(sorted(family,key=lambda r:r['exact_p'])):
            bound=max(bound,min(1.,r['exact_p']*(len(family)-i)));r['holm_p']=bound
        contrasts+=family
    verify_snapshot(out)
    for f,h in read(out/'ground_freeze.json').items():assert digest(out/f)==h
    for _,jobs in phases(p):
        for job in jobs:validate_result(out,job)
    factorial_checks(out,p)
    check=subprocess.run([sys.executable,str(WORKSPACE/'tools/verify_frozen_foundation.py')],capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr
    atomic_json(out/'foundation_verification.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
    atomic_json(out/'summary.json',dict(records=records,contrasts=contrasts,new_results=360,reused_results=260,source_and_weight_checks=True))
    lines=['# 视觉同架构监督控制与完整家族低标签曲线','','25个原配置四预算五种子，共500单元；另6个监督/初始化控制共120单元。360新增、260只读复用。','随机history与预训练history共用冻结RGB/PCA及预训练导出的无动作normalizer，称随机history初始化对照，不称完全无预训练BC。','每组600更新，主监督控制每步256个动作向量；同架构且训练history的预训练/随机两臂参数完全相同。bc_full1600保留旧名称，低预算每步全批B×200，是次要参照。','','|模型|B1|B2|B4|B8|','|---|---:|---:|---:|---:|']
    for arm in p['arms']+p['fair_arms']:
        cells=[]
        for b in p['budgets']:
            v=[values[(seed,b,arm)] for seed in p['seeds']];cells.append(f'{np.mean(v):.2f} ± {np.std(v,ddof=1):.2f}')
        lines.append('|'+arm+'|'+'|'.join(cells)+'|')
    lines+=['','|预定比较|均值差|正向种子|t95|精确p|Holm p|','|---|---:|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['composition']} {r['name']}|{r['mean_delta']:+.3f}|{r['positive_seeds']}/5|{r['t95']}|{r['exact_p']:.4f}|{r['holm_p']:.4f}|")
    lines+=['','先在每seed内等权平均B1/2/4与27主评价条件，再以5个上游seed配对；Solo三项一个Holm族，Aux三项另一个族。B8及家族逐格比较为描述性结果。','所有方向完整保留；五种子精确双侧p最低.0625。未新增环境、agent数、实体对齐视觉slot；新监督路线不替换原冻结MIF结果。论文图表不自动替换。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=PACKAGE/'results'/p['name'];dest.mkdir(exist_ok=True,parents=True)
    for name in ['RESULTS_CN.md','summary.json','qualification.json','foundation_verification.json','protocol.json','factorial_checks.json']:
        assert not (dest/name).exists();shutil.copy2(out/name,dest/name)
