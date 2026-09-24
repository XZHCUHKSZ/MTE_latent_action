"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""
from mte.method_names import report_text

import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import itertools

import subprocess

import sys

import time

import traceback

from pathlib import Path

import numpy as np

import psutil

import torch

from utils.atomic import atomic_json

from experiments.mamujoco_route_completion import (
    PACKAGE, WORKSPACE, read, digest, suite, ground, clean_audit, check_sources)

from training.mif_visibility_control import ARMS

MODULE = 'experiments.mamujoco_mif_visibility'

def setup(out, config):
    c = read(config)
    assert tuple(c['arms']) == ARMS
    out.mkdir(parents=True, exist_ok=False)
    jobs, policies, inputs = [], {}, {}
    for seed in c['seeds']:
        s = suite(seed)
        for p in [s[k] for k in ('train', 'dev', 'endpoints', 'base')] + list(s['labels'].values()):
            inputs[p] = digest(p)
        policies[str(seed)] = {}
        archive = WORKSPACE / 'mamujoco_independent_branch_audit_2026_09_12/repair' / s['id']
        for name in ['representation.pt', 'policy/policy.pt', 'result.json']:
            p = archive / name
            inputs[str(p)] = digest(p)
        for arm in c['arms']:
            dest = out / 'pretraining' / f'seed{seed}_{arm}'
            jobs.append(dict(id=f'pre_s{seed}_{arm}', stage='pre', seed=seed, arm=arm, out=str(dest)))
            policies[str(seed)][arm] = str(dest / 'policy/policy.pt')
        for budget, ds, arm in itertools.product(c['budgets'], c['decoder_seeds'], c['arms']):
            name = f'seed{seed}_B{budget}_{arm}_d{ds}'
            jobs.append(dict(id='ground_' + name, stage='ground', seed=seed, budget=budget,
                decoder_seed=ds, arm=arm, out=str(out / 'grounding' / name)))
    gate = WORKSPACE / c['teacher_gate']
    teacher = WORKSPACE / read(gate)['checkpoint']
    assert read(gate)['passed'] and digest(teacher) == read(gate)['checkpoint_sha256']
    for p in (gate, teacher, config): inputs[str(p)] = digest(p)
    code = {str(p): digest(p) for p in PACKAGE.rglob('*.py') if '__pycache__' not in p.parts}
    for name in ('common.py', 'unified_models.py', 'training_profiles.py'):
        p = WORKSPACE / 'closed_loop_lam_v1' / name
        code[str(p)] = digest(p)
    atomic_json(out / 'protocol.json', c)
    atomic_json(out / 'manifest.json', dict(jobs=jobs, policies=policies, inputs=inputs, code=code,
        teacher=str(teacher), teacher_gate=str(gate), config_hash=digest(out / 'protocol.json')))
    atomic_json(out / 'status.json', dict(status='ready', pretraining_total=len(c['seeds'])*4,
        control_total=sum(j['stage']=='ground' for j in jobs)))

def worker(out, jobid):
    torch.set_num_threads(1)
    c, manifest = read(out/'protocol.json'), read(out/'manifest.json')
    assert digest(out/'protocol.json') == manifest['config_hash']
    check_sources(manifest)
    j = next(j for j in manifest['jobs'] if j['id'] == jobid)
    s, dest = suite(j['seed']), Path(j['out'])
    dest.mkdir(parents=True, exist_ok=False)
    start = time.time()
    def progress(phase, **kw):
        atomic_json(dest/'progress.json', dict(phase=phase, elapsed=time.time()-start, updated=time.time(), **kw))
    allowed = [s['train'], s['dev']]
    if j['stage'] == 'pre':
        allowed.append(s['endpoints'])
    else:
        frozen = read(out/'freeze.json')
        assert frozen['all_modules_frozen']
        allowed += [s['base'], manifest['policies'][str(j['seed'])][j['arm']],
                    s['labels'][str(j['budget'])], manifest['teacher']]
        for p in allowed:
            assert digest(p) == frozen['hashes'].get(p, manifest['inputs'].get(p)), p
    from utils.access import guard
    audit = guard(dest, allowed, pretraining=j['stage']=='pre')
    try:
        if j['stage'] == 'pre':
            from training.mif_visibility_control import train
            result = train(s, j['arm'], c, dest, progress)
        else:
            result = ground(s, j, c, dest, manifest, progress)
        assert not audit['violations']
        result.update(complete=True, job=jobid, stage=j['stage'], seed=j['seed'],
                      arm=j['arm'], elapsed_seconds=time.time()-start)
        atomic_json(dest/'access.json', audit)
        atomic_json(dest/'result.json', result)
    except Exception:
        atomic_json(dest/'access.json', audit)
        atomic_json(dest/'failure.json', dict(traceback=traceback.format_exc()))
        raise

def freeze(out, manifest, native_parity=True):
    from training.composition import restore
    hashes, parity, pairs = {}, [], []
    for seed, arms in manifest['policies'].items():
        stats = []
        for arm, path in arms.items():
            pre = Path(path).parents[1]
            r = read(pre/'result.json')
            assert r['complete']
            clean_audit(pre)
            stats.append(r['representation'])
            _, ck = restore(path)
            s = suite(int(seed))
            _, base = restore(s['base'])
            assert ck['z_dim']==16 and np.array_equal(base['obs_mean'], ck['obs_mean'])
            assert np.array_equal(base['obs_std'], ck['obs_std'])
            for p in [path, s['base'], str(pre/'representation.pt')]: hashes[p] = digest(p)
            representation = torch.load(pre/'representation.pt', map_location='cpu', weights_only=False)
            if arm.endswith('_raw'):
                for key in ['mobius', 'zeta']:
                    assert torch.equal(representation['state_dict'][key], torch.eye(8))
        for key in ['parameters', 'normalization_sha256', 'initial_parameters_sha256', 'loss_masks_sha256']:
            assert len({s[key] for s in stats}) == 1, (seed, key)
        pairs.append(dict(seed=int(seed), identical_parameters_normalization_initialization_masks=True))
        if native_parity:
            archive = WORKSPACE/'mamujoco_independent_branch_audit_2026_09_12/repair'/suite(int(seed))['id']
            new = Path(arms['masked_mobius']).parents[1]
            errors = {}
            for name in ['representation.pt', 'policy/policy.pt']:
                a = torch.load(archive/name, map_location='cpu', weights_only=False)
                b = torch.load(new/name, map_location='cpu', weights_only=False)
                assert a['state_dict'].keys() == b['state_dict'].keys()
                error = max(float((a['state_dict'][k].float()-b['state_dict'][k].float()).abs().max())
                            for k in a['state_dict'])
                errors[name] = error
                assert error <= 1e-5, f'Native parity failed: {seed}/{name}/{error}'
            parity.append(dict(seed=int(seed), max_absolute_errors=errors))
    check_sources(manifest, True)
    atomic_json(out/'freeze.json', dict(all_modules_frozen=True, hashes=hashes, pairs=pairs,
                archived_native_parity=parity, time=time.time()))

def summarize(out, manifest, c):
    from scipy.stats import t
    rows = []
    for j in manifest['jobs']:
        if j['stage'] != 'ground': continue
        p = Path(j['out'])/'result.json'
        r = read(p)
        ev = r['evaluation']
        assert r['complete'] and ev['episode_seeds'] == c['evaluation_seeds']['mamujoco']
        assert r['budget']==j['budget'] and r['decoder_seed']==j['decoder_seed']
        assert abs(np.mean(ev['episode_returns'])-ev['mean_return']) < 1e-8
        clean_audit(p.parent)
        rows.append(dict(seed=j['seed'], budget=j['budget'], decoder_seed=j['decoder_seed'],
                         arm=j['arm'], mean_return=ev['mean_return'], source=str(p)))
    expected = set(itertools.product(c['seeds'], c['budgets'], c['decoder_seeds'], c['arms']))
    assert len(rows)==len(expected) and {(r['seed'],r['budget'],r['decoder_seed'],r['arm']) for r in rows}==expected
    contrasts = []
    vectors = {}
    for mode in ['masked', 'visible']:
        vectors[mode] = np.array([np.mean([r['mean_return'] for r in rows if r['seed']==seed and r['arm']==mode+'_mobius'])
            - np.mean([r['mean_return'] for r in rows if r['seed']==seed and r['arm']==mode+'_raw']) for seed in c['seeds']])
    vectors['masking_interaction'] = vectors['masked'] - vectors['visible']
    for name, v in vectors.items():
        count = len(v)
        half = t.ppf(.975,count-1)*v.std(ddof=1)/np.sqrt(count)
        signs = np.array(list(itertools.product([-1,1], repeat=count)))
        p = float(np.mean(np.abs((signs*v).mean(1)) >= abs(v.mean())-1e-12))
        contrasts.append(dict(name=name, mean=float(v.mean()), ci95=[float(v.mean()-half),float(v.mean()+half)],
                              seed_differences=v.tolist(), positive_seeds=int((v>0).sum()), exact_p=p))
    last = 0.
    for i, r in enumerate(sorted(contrasts, key=lambda r:r['exact_p'])):
        last = max(last, min(1., (3-i)*r['exact_p']))
        r['holm_p'] = last
    check_sources(manifest, True)
    for p, h in read(out/'freeze.json')['hashes'].items(): assert digest(p)==h
    summary = dict(rows=rows, contrasts=contrasts, source_and_weight_checks=True,
        interpretation='Positive interaction means masking increases Mobius-minus-raw return. Five upstream seeds, decoder repeats nested.')
    atomic_json(out/'summary.json', summary)
    lines = ['# MaMuJoCo MIF 遮蔽 × 坐标配对结果', '',
        '原 masked MIF 与归档权重核对通过。完整可见组只改变编码器可见性，保留原 loss 分组及权重；并非前轮 route 协议。', '',
        '|比较|均值差↑|95%区间|正向种子|精确p|Holm p|', '|---|---:|---|---:|---:|---:|']
    for r in contrasts:
        lines.append(f'|{r["name"]}|{r["mean"]:.4f}|{r["ci95"]}|{r["positive_seeds"]}/5|{r["exact_p"]:.4f}|{r["holm_p"]:.4f}|')
    lines += ['', 'masked/visible：Möbius−raw；masking_interaction：(Möbius−raw)masked−(Möbius−raw)visible。',
        '遮蔽的是各自基底的节点。同索引掩码并不意味着丢掉了相同的原始表信息。',
        '五个既有上游种子的受控扩展；每种子先平均五预算和五次decoder重复。不是新环境验证，均值正不等于稳定显著。',
        '全部500单元和逐种子差保留于summary.json。论文、图表和GitHub未自动更新。', '']
    (out/'RESULTS_CN.md').write_text(report_text(lines, "control"), encoding='utf-8')
    delivery = PACKAGE/'results'/out.name
    delivery.mkdir(parents=True, exist_ok=True)
    (delivery/'RESULTS_CN.md').write_text(report_text(lines, "control"), encoding='utf-8')
    atomic_json(delivery/'summary.json', summary)

def manager(out):
    import msvcrt
    lock = (out/'manager.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    c, manifest = read(out/'protocol.json'), read(out/'manifest.json')
    check_sources(manifest, True)
    done, active, failures = set(), {}, []
    start = time.time()
    for j in manifest['jobs']:
        p = Path(j['out'])/'result.json'
        if p.exists():
            assert read(p)['complete']; clean_audit(p.parent); done.add(j['id'])
        elif p.parent.exists(): raise RuntimeError(f'Inspect partial stage before resume: {p.parent}')
    total = sum(j['stage']=='ground' for j in manifest['jobs'])
    try:
        while len(done)<len(manifest['jobs']):
            for key, (proc, stream) in list(active.items()):
                rc = proc.poll()
                if rc is None: continue
                stream.close()
                j = next(j for j in manifest['jobs'] if j['id']==key)
                p = Path(j['out'])/'result.json'
                if rc or not p.exists(): failures.append(dict(job=key, exit_code=rc))
                else:
                    assert read(p)['complete']; clean_audit(p.parent); done.add(key)
                del active[key]
            pre_done = all(j['id'] in done for j in manifest['jobs'] if j['stage']=='pre')
            if pre_done and not (out/'freeze.json').exists() and not failures: freeze(out, manifest)
            stage = 'ground' if pre_done else 'pre'
            cap = c['ground_workers'] if pre_done else c['pretraining_workers']
            if not failures and not (out/'PAUSE').exists():
                for j in manifest['jobs']:
                    if j['stage']!=stage or j['id'] in done or j['id'] in active: continue
                    if len(active)>=cap or psutil.virtual_memory().available/2**30<c['minimum_available_gib']: break
                    stream = (out/(j['id']+'.log')).open('ab')
                    proc = subprocess.Popen([sys.executable,'-X','utf8','-m',MODULE,'--out',str(out),'--job',j['id']],
                        cwd=PACKAGE, stdout=stream, stderr=stream, creationflags=subprocess.CREATE_NO_WINDOW)
                    active[j['id']] = proc, stream
            atomic_json(out/'status.json', dict(status='failed' if failures else 'paused' if (out/'PAUSE').exists() else 'running',
                stage=stage, pretraining_done=sum(j['id'] in done for j in manifest['jobs'] if j['stage']=='pre'),
                pretraining_total=len(c['seeds'])*len(c['arms']), control_done=sum(j['id'] in done for j in manifest['jobs'] if j['stage']=='ground'),
                control_total=total, running=[dict(job=k,pid=p.pid) for k,(p,_) in active.items()], failures=failures,
                manager_pid=os.getpid(), active_seconds=time.time()-start, updated=time.time()))
            if failures and not active: return
            time.sleep(3)
        summarize(out, manifest, c)
        subprocess.run([sys.executable,str(WORKSPACE/'tools/verify_frozen_foundation.py')], cwd=WORKSPACE, check=True)
        atomic_json(out/'status.json', dict(status='complete', control_done=total, control_total=total,
                    source_and_weight_checks=True, active_seconds=time.time()-start, updated=time.time()))
    except Exception:
        atomic_json(out/'manager_failure.json', dict(traceback=traceback.format_exc(), updated=time.time()))
        raise
    finally:
        lock.close()
