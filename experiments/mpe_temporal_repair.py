"""Bounded MPE timing repair: observation-only children, then budgeted grounding.

Run from package root:
python -m experiments.mpe_temporal_repair --workspace ROOT --out outputs/NEW
Completed stages are resumable under an identical frozen configuration.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from utils.atomic import atomic_json

PACKAGE = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(workspace, seed):
    protocol = json.loads((PACKAGE/'configs/numeric.json').read_text())
    return next(s for s in protocol['suites'] if s['env'] == 'mpe' and s['seed'] == seed and s['n'] == 700)


def exports(workspace, out, p, labels=False):
    """Curator process only. Pretraining never receives the original archive."""
    from utils.access import read_rows
    for seed in p['seeds']:
        data_seed = int(p.get('data_seed_map', {}).get(str(seed), seed))
        s = sources(workspace, data_seed); dst = out/f'seed{seed}'/'data'; dst.mkdir(parents=True, exist_ok=True)
        raw = workspace/'closed_loop_lam_v1/runs/formal_v2/mpe/data'/f'mpe_4agent_H3_seed{data_seed}.npz'
        if labels:
            assert (out/'global_freeze.json').exists()
            for b in p['budgets']:
                dest = dst/f'labels_b{b}.npz'
                if dest.exists(): continue
                y = read_rows(raw, 'actions', s['train_ids'][:b])[:, 1:23]
                np.savez_compressed(dest, actions=y, local_ids=np.arange(b))
        else:
            for name, ids in [('train', s['train_ids'][:p['train_episodes']]), ('dev', s['dev_ids'][:p['dev_episodes']])]:
                dest = dst/f'{name}.npz'
                if dest.exists(): continue
                # Observation member only; never touch actions or old counterfactual arrays.
                o = read_rows(raw, 'obs', ids)
                x = o[..., [0, 1, 4, 5, 8, 9, 12, 13, *range(16, 24)]]
                assert x.shape == (len(ids), 26, 16)
                np.savez_compressed(dest, positions=x)
            atomic_json(dst/'manifest.json', dict(source=str(raw), data_seed=data_seed, training_seed=seed,
                train_ids=s['train_ids'][:p['train_episodes']], dev_ids=s['dev_ids'][:p['dev_episodes']],
                read_members=['obs'], learner_members=['positions'], native_action_labels_read=0))


def load_x(seedout, p):
    arrays = []
    for split in ('train', 'dev'):
        with np.load(seedout/'data'/f'{split}.npz', allow_pickle=False) as f:
            assert set(f.files) == {'positions'}
            arrays.append(f['positions'])
    assert len(arrays[0]) == p['train_episodes'] and len(arrays[1]) == p['dev_episodes']
    return np.concatenate(arrays), len(arrays[0])


def pretrain(seedout, p, progress):
    out = seedout/'pretrain'; out.mkdir(exist_ok=True)
    from utils.access import guard
    audit = guard(out, [seedout/'data/train.npz', seedout/'data/dev.npz'], pretraining=True)
    def deny_sim(event, args):
        if event == 'import' and args[0].split('.')[0] in {'mpe2', 'gymnasium', 'mujoco', 'pettingzoo'}:
            raise RuntimeError('No simulator imports in observation-only pretraining')
    sys.addaudithook(deny_sim)
    from mte.temporal_mpe import frontend, bridge
    from training.temporal_family_mpe import train as train_family
    from training.history_mpe import train_history_policy
    from training.composition import restore
    x, n = load_x(seedout, p)
    diagnostics = {}; representations = {}

    def history(name, z, diag):
        folder = out/name; folder.mkdir(exist_ok=True)
        progress('history_start', arm=name)
        h = train_history_policy(x, z, n, p['seed'], p['history_updates'], folder/'policy', progress)
        net, ck = restore(folder/'policy/policy.pt')
        with torch.no_grad():
            current = torch.tensor((x[:2, :23]-ck['obs_mean'])/ck['obs_std'])
            changed = current.clone(); changed[:, 13:] += 17
            error = float((net(current)[0][:, :13]-net(changed)[0][:, :13]).abs().max())
        assert error == 0, 'Future observation affected causal history'
        diagnostics[name] = dict(representation=diag, history=h, future_prefix_error=error)
        representations[name] = z
        atomic_json(out/'diagnostics.json', diagnostics)

    for offset in (1, 2):
        prefix = 'old_' if offset == 1 else ''
        froot = out/f'offset{offset}'; froot.mkdir()
        z, diag = frontend(x, n, 'entity', offset, p, froot/'entity', progress)
        history(prefix+'base', z[:, :, 0], diag)
        a, b, masks, bd = bridge(x, z, n, p, froot/'bridge', progress)
        atomic_json(froot/'bridge/diagnostics.json', bd)
        arms = ['graph'] if offset == 1 else p['corrected_families']
        for arm in arms:
            folder = out/(prefix+arm); folder.mkdir()
            codes, fd = train_family(x, a, b, masks, n, arm, p, folder, progress)
            history(prefix+arm, codes, fd)
        if offset == 2:
            for variant in ('lapo', 'laom', 'global16'):
                codes, fd = frontend(x, n, variant, offset, p, froot/variant, progress)
                history(variant, codes, fd)
    # Fingerprint every trained weight. Label export is permitted only after this marker.
    checkpoints = {str(f.relative_to(out)): digest(f) for f in out.rglob('*.pt')}
    atomic_json(out/'access_audit.json', audit)
    assert not audit['violations']
    atomic_json(out/'complete.json', dict(checkpoints=checkpoints, native_action_labels_read=0,
        simulator_queries=0, frozen_before_grounding=True, seed=p['seed']))


def ground(seedout, p, progress):
    from training.composition import restore, FrozenPair
    from training.grounding_mpe import train_decoder, train_supervised_history, GroundedPolicy
    from evaluation.mpe import evaluate
    x, n = load_x(seedout, p); frozen = seedout/'pretrain'
    before = json.loads((frozen/'complete.json').read_text())['checkpoints']
    assert all(digest(frozen/f) == h for f, h in before.items())
    eval_seeds = list(range(p['evaluation_seed_start'], p['evaluation_seed_start']+p['evaluation_episodes']))
    dest = seedout/'grounding'
    # A scale scheduler may assign disjoint budgets to independent processes.
    # Per-budget directories prevent result/checkpoint/completion write races.
    if '_ground_budget' in p:
        assert p['budgets'] == [p['_ground_budget']]
        dest = dest/f"budget_{p['_ground_budget']}"
    dest.mkdir(parents=True, exist_ok=True); rows = []
    for b in p['budgets']:
        with np.load(seedout/'data'/f'labels_b{b}.npz', allow_pickle=False) as f:
            assert set(f.files) == {'actions', 'local_ids'}
            actions = f['actions']; assert np.array_equal(f['local_ids'], np.arange(b))
        val = np.arange(0, b, 4); fit = np.setdiff1d(np.arange(b), val)
        for method in p['control_configs']:
            target = dest/f'b{b}_{method.replace("+", "__")}.json'
            if target.exists(): rows.append(json.loads(target.read_text())); continue
            progress('grounding', method=method, budget=b)
            decoder = None
            if method == 'bc':
                y = np.zeros((n, 22, 2), np.float32); y[fit] = actions[fit]
                net, mu, sd, diag = train_supervised_history(x[:n], y, fit, x[val], actions[val],
                                                            p['seed'], p['bc_updates'])
            else:
                members = method.split('+'); net, ck = restore(frozen/members[0]/'policy/policy.pt')
                if len(members) == 2:
                    aux, ac = restore(frozen/members[1]/'policy/policy.pt')
                    assert np.array_equal(ck['obs_mean'], ac['obs_mean']) and np.array_equal(ck['obs_std'], ac['obs_std'])
                    net = FrozenPair(net, aux)
                mu, sd = ck['obs_mean'], ck['obs_std']
                with torch.no_grad():
                    z = net(torch.tensor((x[:b, :23]-mu)/sd))[0][:, 1:].numpy()
                decoder, diag = train_decoder(z, actions, fit, val, p['seed'], p['decoder_updates'])
            policy = GroundedPolicy(net, mu, sd, decoder)
            progress('rollout', method=method, budget=b)
            result = evaluate(policy, eval_seeds)
            row = dict(seed=p['seed'], budget=b, method=method, grounding=diag, **result)
            atomic_json(target, row); rows.append(row)
            torch.save(dict(history=net.state_dict(), decoder=None if decoder is None else decoder.state_dict(),
                            mean=mu, std=sd, method=method, budget=b), target.with_suffix('.pt'))
            atomic_json(dest/'results.json', rows)
    assert all(digest(frozen/f) == h for f, h in before.items())
    atomic_json(dest/'complete.json', dict(rows=len(rows), frozen_checkpoints_unchanged=True))


def worker(args, p):
    seedout = args.out/f'seed{args.seed}'; p = dict(p, seed=args.seed)
    torch.set_num_threads(p['threads_per_process'])
    def progress(stage, **kw):
        row = dict(stage=stage, seed=args.seed, time=time.time(), **kw)
        atomic_json(seedout/'status.json', row); print(json.dumps(row), flush=True)
    try:
        if args.stage == 'pretrain': pretrain(seedout, p, progress)
        elif args.stage == 'ground': ground(seedout, p, progress)
        elif args.stage == 'physical':
            from evaluation.temporal_mpe_effects import evaluate
            evaluate(seedout, p, progress)
        progress('complete', phase=args.stage)
    except BaseException:
        atomic_json(seedout/f'{args.stage}_failure.json', dict(traceback=traceback.format_exc()))
        raise


def summary(out, p):
    rows = []
    for seed in p['seeds']:
        path = out/f'seed{seed}/grounding/results.json'
        if path.exists(): rows.extend(json.loads(path.read_text()))
    means = []; contrasts = []
    for budget in p['budgets']:
        for method in p['control_configs']:
            values = [r for r in rows if r['budget'] == budget and r['method'] == method]
            if values:
                means.append(dict(budget=budget, method=method, seeds=[r['seed'] for r in values],
                    mean_return=float(np.mean([r['mean_return'] for r in values])),
                    mean_collisions=float(np.mean([r['mean_collisions'] for r in values])),
                    mean_final_coverage=float(np.mean([r['mean_final_coverage_distance'] for r in values]))))
        for a, b in p['primary_contrasts'] + [['base+graph', 'old_base+old_graph'], ['base', 'old_base']]:
            differences = []
            for seed in p['seeds']:
                aa = [r for r in rows if r['seed'] == seed and r['budget'] == budget and r['method'] == a]
                bb = [r for r in rows if r['seed'] == seed and r['budget'] == budget and r['method'] == b]
                if aa and bb: differences.append(dict(seed=seed, delta=aa[0]['mean_return']-bb[0]['mean_return']))
            if differences: contrasts.append(dict(budget=budget, plus=a, minus=b, differences=differences,
                mean_delta=float(np.mean([d['delta'] for d in differences]))))
    atomic_json(out/'summary.json', dict(completed_rows=len(rows), expected_rows=len(p['seeds'])*len(p['budgets'])*len(p['control_configs']),
        means=means, contrasts=contrasts, scope=p['scope']))
    lines = ['# MPE 时间接口修正版：开发实验', '', p['scope'], '',
             '原始结果未覆盖。两个训练种子不能当作五种子确认；评估 episodes 不作为独立训练重复。', '',
             '|预算|方法|已完成种子|平均回报 ↑|碰撞 ↓|覆盖距离 ↓|', '|---:|---|---|---:|---:|---:|']
    for r in means: lines.append(f"|{r['budget']}|{r['method']}|{r['seeds']}|{r['mean_return']:.5f}|{r['mean_collisions']:.3f}|{r['mean_final_coverage']:.5f}|")
    lines += ['', '## 预先指定的配对比较与时间窗口检查', '', '|预算|正项|负项|回报差|逐种子差|', '|---:|---|---|---:|---|']
    for r in contrasts: lines.append(f"|{r['budget']}|{r['plus']}|{r['minus']}|{r['mean_delta']:+.5f}|{r['differences']}|")
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def manage(args, p):
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out/'manager.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        protocol = args.out/'protocol.json'
        if protocol.exists(): assert json.loads(protocol.read_text()) == p, 'Cannot resume under changed protocol'
        else: atomic_json(protocol, p)
        atomic_json(args.out/'implementation_sources.json', {str(f.relative_to(PACKAGE)): digest(f)
            for sub in ['mte', 'training', 'experiments', 'evaluation', 'environments', 'closed_loop_lam_v1']
            for f in (PACKAGE/sub).glob('*.py')})
        exports(args.workspace, args.out, p)
        def run(stage, seed):
            folder = args.out/f'seed{seed}'; folder.mkdir(exist_ok=True)
            complete = folder/({'pretrain': 'pretrain/complete.json', 'ground': 'grounding/complete.json', 'physical': 'physical_effects.json'}[stage])
            if complete.exists(): return
            if stage == 'pretrain' and (folder/'pretrain').exists():
                raise RuntimeError('Incomplete pretrain retained for audit. Use a new output directory, not overwrite.')
            cmd = [sys.executable, '-m', 'experiments.mpe_temporal_repair', '--workspace', str(args.workspace),
                   '--out', str(args.out), '--config', str(args.config), '--stage', stage, '--seed', str(seed)]
            with (folder/f'{stage}.log').open('a', encoding='utf-8') as log:
                subprocess.run(cmd, cwd=PACKAGE, stdout=log, stderr=subprocess.STDOUT, check=True,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for stage in ('pretrain', 'ground', 'physical'):
            atomic_json(args.out/'status.json', dict(stage=stage, time=time.time(), status='running'))
            with concurrent.futures.ThreadPoolExecutor(max_workers=p['parallel_seeds']) as pool:
                futures = [pool.submit(run, stage, seed) for seed in p['seeds']]
                for future in concurrent.futures.as_completed(futures): future.result(); summary(args.out, p)
            if stage == 'pretrain':
                atomic_json(args.out/'global_freeze.json', dict(all_pretraining_complete=True, time=time.time()))
                exports(args.workspace, args.out, p, labels=True)
        atomic_json(args.out/'status.json', dict(stage='complete', time=time.time(), status='complete'))
    except BaseException:
        atomic_json(args.out/'status.json', dict(stage='failed', traceback=traceback.format_exc(), time=time.time()))
        raise
    finally: lock.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=PACKAGE/'configs/mpe_temporal_repair.json')
    ap.add_argument('--stage', choices=['pretrain', 'ground', 'physical']); ap.add_argument('--seed', type=int)
    args = ap.parse_args(); args.out = args.out.resolve(); args.workspace = args.workspace.resolve(); args.config = args.config.resolve()
    p = json.loads(args.config.read_text())
    if args.stage: worker(args, p)
    else: manage(args, p)


if __name__ == '__main__': main()
