"""X3b: Does the frozen visual MTE route help with fewer action labels?

Stage-isolated jobs, original B8 reproduction gates, resumable completed jobs.
Never rewrites archived evidence or automatically retries a partial job.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from utils.atomic import atomic_json

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[1]
ARCHIVE = WORKSPACE/'visual_multiagent_2026_09_10/seed5_v1/runtime'
REPAIRED = WORKSPACE/'visual_multiagent_2026_09_10/seed5_render_repair/runtime'
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
    files.add(PACKAGE/'configs/visual_low_label.json')
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
        for stage in ['frontend', 'features'] + ['pre_'+m for m in ['visual_laom_target','edge_cara_mif','lapo_state_adapter','laom_state_adapter']]:
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
        r = B.idm(dest) if arm == 'bc_idm_relabel' else B.ground(dest, arm)
    else:
        assert read(root/('ground_'+arm)/'result.json')['complete']
        if phase == 'formal':
            assert read(a.out/'qualification.json')['passed']
            frozen = read(a.out/'ground_freeze.json')
            cp = root/('ground_'+arm)/'decoder.pt'
            assert digest(cp) == frozen[str(cp.relative_to(a.out))]
        from evaluation.visual_label_budget import evaluate
        r = evaluate(dest, arm, seed, root, a.out/'assets', [908601] if phase == 'parity' else p['evaluation_seeds'])
        if phase == 'parity':
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
    smoke = [('smoke',p['seeds'][0],b,a,'ground') for b in [1,4] for a in [p['main_model'],'bc_batch256','bc_idm_relabel']]
    parity = [('parity',s,8,a,'ground') for s in p['seeds'] for a in p['arms']]
    replay = [(*j[:4],'evaluate') for j in parity]
    ground = [('formal',s,b,a,'ground') for b in p['new_budgets'] for s in p['seeds'] for a in p['arms']]
    evaluation = [(*j[:4],'evaluate') for j in ground]
    return [('smoke',smoke),('b8_ground_parity',parity),('b8_replay',replay),('low_budget_ground',ground),('low_budget_evaluate',evaluation)]


def process_created(pid):
    """A worker can exit between poll and status serialization."""
    import psutil
    try:
        return psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return None


def run_phase(out, p, title, jobs):
    import psutil
    running = {}; done = set(); failure = None
    for job in jobs:
        dest = job_path(out,job)
        if (dest/'result.json').exists():
            validate_result(out,job); done.add(job)
        elif dest.exists():
            raise RuntimeError(f'Partial job requires inspection; no automatic overwrite: {dest}')
    while len(done) < len(jobs) or running:
        for job, (proc, log, start) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close(); del running[job]
            try:
                assert rc == 0, f'Worker exited {rc}: {job}'
                validate_result(out,job); done.add(job)
            except Exception:
                failure = traceback.format_exc()
                atomic_json(out/'manager_failure.json',dict(job=job,error=failure))
        paused = (out/'PAUSE').exists()
        if not paused and not failure:
            for job in jobs:
                if len(running) >= p['workers'] or psutil.virtual_memory().available/2**30 < p['minimum_available_gib']:
                    break
                if job in running or job in done:
                    continue
                name = '__'.join(map(str,job))
                log = (out/'logs'/f'{name}.{time.time_ns()}.log').open('w',encoding='utf-8')
                cmd = [sys.executable,'-X','utf8','-m','experiments.visual_low_label','worker','--out',str(out),'--job',*map(str,job)]
                env = os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                proc = subprocess.Popen(cmd,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                running[job] = (proc,log,time.time())
        state = dict(status='draining_after_failure' if failure else 'pausing' if paused else 'running',
            phase=title,manager_pid=os.getpid(),manager_created=psutil.Process().create_time(),updated_unix=time.time(),
            completed_phase_jobs=len(done),total_phase_jobs=len(jobs),failure=failure,
            new_control_results=len(list((out/'formal').glob('seed*/b*/evaluate_*/result.json'))),target_new_control_results=135,
            active=[dict(job=job,pid=proc.pid,created=process_created(proc.pid),seconds=time.time()-start,
                progress=read(job_path(out,job)/'progress.json') if (job_path(out,job)/'progress.json').exists() else {}) for job,(proc,log,start) in running.items()])
        atomic_json(out/'status.json',state)
        if not running and (failure or paused):
            state['status'] = 'failed' if failure else 'paused';atomic_json(out/'status.json',state)
            return False
        time.sleep(3)
    return True


def aggregate(out,p):
    from scipy.stats import t
    records=[]
    for budget in p['budgets']:
        for seed in p['seeds']:
            for arm in p['arms']:
                f = REPAIRED/str(seed)/('evaluate_'+arm)/'result.json' if budget==8 else job_path(out,('formal',seed,budget,arm,'evaluate'))/'result.json'
                r=read(f); assert [x['seed'] for x in r['rows']]==p['evaluation_seeds']
                primary=float(np.mean([x['return_value'] for x in r['rows'] if x['seed'] in p['primary_evaluation_seeds']]))
                records.append(dict(seed=seed,budget=budget,arm=arm,mean_return=primary,reused=budget==8,source=str(f)))
    assert len({(r['seed'],r['budget'],r['arm']) for r in records})==180
    contrasts=[]
    for family,model,controls in [('primary',p['main_model'],p['primary_controls']),('secondary',p['secondary_model'],p['secondary_controls'])]:
        family_rows=[]
        for control in controls:
            def value(s,b,a):return next(r['mean_return'] for r in records if (r['seed'],r['budget'],r['arm'])==(s,b,a))
            d=np.array([np.mean([value(s,b,model)-value(s,b,control) for b in p['new_budgets']]) for s in p['seeds']])
            radius=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5))
            null=np.abs(np.array(list(itertools.product([-1,1],repeat=5)))@d/5)
            family_rows.append(dict(family=family,model=model,control=control,seed_deltas=d.tolist(),mean_delta=float(d.mean()),positive_seeds=int((d>0).sum()),t95=[float(d.mean()-radius),float(d.mean()+radius)],exact_p=float((null>=abs(d.mean())-1e-12).mean())))
        bound=0
        for i, r in enumerate(sorted(family_rows,key=lambda r:r['exact_p'])):
            bound=max(bound,min(1.,r['exact_p']*(len(family_rows)-i)));r['holm_p']=bound
        contrasts+=family_rows
    verify_snapshot(out)
    for f,h in read(out/'ground_freeze.json').items():assert digest(out/f)==h
    for _,jobs in phases(p):
        for job in jobs:validate_result(out,job)
    check=subprocess.run([sys.executable,str(WORKSPACE/'tools/verify_frozen_foundation.py')],capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr
    atomic_json(out/'foundation_verification.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
    atomic_json(out/'summary.json',dict(records=records,contrasts=contrasts,new_results=135,reused_results=45,source_and_weight_checks=True))
    lines=['# 视觉低标签控制曲线','','原五个冻结训练种子、原RGB前端和同一四智能体Ant任务；新增1/2/4条标注轨迹，复用8条结果。每条200个目标动作标签；比例为3.125%/6.25%/12.5%/25%。','各训练种子内先平均27个主评价条件，再汇总5个训练种子。前3个旧开发条件只计入次要30回合结果。','','|模型|B1|B2|B4|B8（复用）|','|---|---:|---:|---:|---:|']
    for arm in p['arms']:
        cells=[]
        for b in p['budgets']:
            v=[r['mean_return'] for r in records if r['arm']==arm and r['budget']==b]
            cells.append(f'{np.mean(v):.2f} ± {np.std(v,ddof=1):.2f}')
        lines.append('|'+arm+'|'+'|'.join(cells)+'|')
    lines+=['','|预定比较（B1/B2/B4等权）|均值差|正向种子|t95|精确p|Holm p|','|---|---:|---:|---|---:|---:|']
    for r in contrasts:lines.append(f"|{r['model']} − {r['control']}|{r['mean_delta']:+.3f}|{r['positive_seeds']}/5|{r['t95']}|{r['exact_p']:.4f}|{r['holm_p']:.4f}|")
    lines+=['','五种子双侧精确检验最低p=.0625；均值方向和统计证据分别报告。不得按低标签结果重选预算或主模型。','标签集合为嵌套轨迹前缀，各方法完全共用；本轮没有重抽数据集或新增训练种子。BC直接训练history，IDM由少量真实标签产生伪标签，潜表示方法冻结history后grounding。','本轮只检验标签效率；未解决agent-aligned视觉slot、新环境/更多agent或MTE各机制的完整归因。论文图表未自动修改，未上传GitHub。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=PACKAGE/'results'/p['name'];dest.mkdir(exist_ok=True,parents=True)
    for name in ['RESULTS_CN.md','summary.json','qualification.json','foundation_verification.json','protocol.json']:
        assert not (dest/name).exists();shutil.copy2(out/name,dest/name)


def manage(a,p):
    import psutil
    out=a.out
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'manager.lock'
    if lock.exists():
        old=read(lock)
        if psutil.pid_exists(old['pid']) and abs(psutil.Process(old['pid']).create_time()-old['created'])<1:
            raise RuntimeError('Existing manager alive; do not launch twice')
        lock.rename(out/f'manager.stale.{time.time_ns()}.json')
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),f)
    try:
        (out/'logs').mkdir(exist_ok=True)
        if not (out/'source_manifest.json').exists():
            assert not (out/'protocol.json').exists(), 'Incomplete preparation requires inspection'
            atomic_json(out/'protocol.json',p)
            assets=out/'assets';(assets/'labels/dev').mkdir(parents=True)
            shutil.copy2(PIXEL/'pilot_protocol.json',assets/'visual_config.json')
            for i in range(8):shutil.copy2(PIXEL/f'runtime/pilot/validate/target_labels/dev/{i:04d}.npy',assets/f'labels/dev/{i:04d}.npy')
            atomic_json(out/'source_manifest.json',source_snapshot(out,p))
        else:
            assert read(out/'protocol.json')==p
            verify_snapshot(out)
        for title,jobs in phases(p):
            if not run_phase(out,p,title,jobs):return
            if title=='b8_replay':
                verify_snapshot(out)
                atomic_json(out/'qualification.json',dict(passed=True,b8_decoder_checks=45,independent_episode_replays=45,low_budget_smokes=6,weight_tolerance=1e-6,replay='exact actions and rewards',time=time.time()))
            if title=='low_budget_ground':
                atomic_json(out/'ground_freeze.json',{str(f.relative_to(out)):digest(f) for f in (out/'formal').glob('seed*/b*/ground_*/*.pt')})
        aggregate(out,p)
        atomic_json(out/'status.json',dict(status='complete',phase='complete',new_control_results=135,reused_results=45,total_results=180,source_and_weight_checks=True,updated_unix=time.time()))
    except Exception:
        atomic_json(out/'manager_failure.json',dict(error=traceback.format_exc()))
        atomic_json(out/'status.json',dict(status='failed',manager_pid=os.getpid(),updated_unix=time.time()))
        raise
    finally:
        lock.unlink()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['manage','worker'])
    ap.add_argument('--out',type=Path,default=PACKAGE/'outputs/visual_low_label_2026_09_19_r1')
    ap.add_argument('--job',nargs=5)
    a=ap.parse_args();a.out=a.out.resolve()
    p=read(PACKAGE/'configs/visual_low_label.json')
    if a.command=='manage':manage(a,p)
    else:worker(a,p)


if __name__=='__main__':main()
