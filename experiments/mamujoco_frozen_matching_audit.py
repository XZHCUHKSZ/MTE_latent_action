"""Frozen five-checkpoint same-state matching diagnostic; no policy training.

Reuses the accepted evaluation adapters and zero-reference construction.
This does not evaluate historical donor transport or control performance.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from experiments.target_effects import collect, fingerprint, write
from evaluation.frozen_effect_models import Frontend, Predictor
from evaluation.effect_statistics import partition, metrics, target_edges, single_partner_interactions

ROOT = Path(__file__).resolve().parents[1]


def score(workspace, seed, arrays, cfg):
    sid = f'mamujoco_s{seed}_n90'
    old = workspace/'observation_scale_restore_2026_09_10/runtime'
    paths = dict(front=old/(sid+'__front_laom_entity_k3/frontend.pt'),
                 predictor=old/(sid+'__bridge/readout.pt'), data=old/(sid+'__prepare/train.npz'))
    hashes = {str(p): fingerprint(p) for p in paths.values()}
    front = Frontend(paths['front'], 'mamujoco')
    predictor = Predictor(paths['predictor'], 'mamujoco', paths['data'])
    pre, cur, future = (arrays[k] for k in ('previous', 'current', 'factual_future'))
    q = front.encode(pre, cur, future[:, 0])
    ref = front.encode(pre, cur, arrays['cube'][:, 15, 0])
    continuation = np.stack([front.encode(cur, future[:, 0], future[:, 1]),
                             front.encode(future[:, 0], future[:, 1], future[:, 2])], 1)
    aa, bb = [], []
    for mask in range(8):
        za = q.copy()
        for partner in range(1, 4):
            if (mask >> (partner-1)) & 1:
                za[:, partner] = ref[:, partner]
        zb = za.copy(); zb[:, 0] = ref[:, 0]
        assert np.array_equal(za[:, 1:], zb[:, 1:])
        aa.append(predictor.rollout(pre, cur, za, continuation))
        bb.append(predictor.rollout(pre, cur, zb, continuation))
    a, b = np.stack(aa, 2), np.stack(bb, 2)
    edge = a-b
    true = target_edges(arrays['cube']).transpose(0, 2, 1, 3)
    coeff = single_partner_interactions(edge.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    truth_coeff = single_partner_interactions(true.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    splits = partition(arrays['episode'], **cfg['splits'])
    scale = cur[splits[0], :27].std(0).astype(float)
    scale[scale < 1e-4] = 1.
    rows = []
    for h in range(3):
        live = arrays['live'][:, :, :h+1].all((1, 2))
        fit, _, test = (s & live for s in splits)
        y = true[:, h, 0, :27]
        for name, pred in [('matched_edge', edge[:, h, 0, :27]),
                           ('partner_mismatched_edge', (a[:, h, 0]-b[:, h, 7])[:, :27]),
                           ('zero', np.zeros_like(y))]:
            rows.append(dict(seed=seed, horizon=h+1, target='first_order', method=name,
                test_samples=int(test.sum()), test_episodes=int(len(np.unique(arrays['episode'][test]))),
                **metrics(pred[test], y[test], y[fit], scale)))
        for partner in range(3):
            y = truth_coeff[:, h, partner, :27]
            for name, pred in [('matched_coefficient', coeff[:, h, partner, :27]), ('zero', np.zeros_like(y))]:
                rows.append(dict(seed=seed, horizon=h+1, target='second_order', partner=partner+1,
                    method=name, test_samples=int(test.sum()),
                    **metrics(pred[test], y[test], y[fit], scale)))
    assert all(fingerprint(p) == v for p, v in hashes.items())
    return rows, hashes, dict(predicted_edges=edge, true_edges=true, predicted_coefficients=coeff,
                             true_coefficients=truth_coeff, episode=arrays['episode'])


def paired(values):
    v = np.asarray(values, float)
    signs = np.array(list(itertools.product([-1., 1.], repeat=len(v))))
    p = float(np.mean(np.abs((signs*v).mean(1)) >= abs(v.mean())-1e-14))
    half = float(t.ppf(.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return dict(mean=float(v.mean()), ci95=[float(v.mean()-half), float(v.mean()+half)],
                positive_seeds=int((v > 0).sum()), seed_differences=v.tolist(), exact_p=p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'protocol.json', cfg)
    source_paths = [Path(__file__), ROOT/'experiments/target_effects.py',
        ROOT/'evaluation/frozen_effect_models.py', ROOT/'evaluation/effect_statistics.py',
        ROOT/'environments/effect_evaluation.py', ROOT/'environments/mamujoco.py',
        ROOT/'mte/frontends.py', ROOT/'environments/teacher.py']
    sources = {str(p): fingerprint(p) for p in source_paths}
    write(args.out/'status.json', dict(status='running', stage='collect', training_updates=0))
    try:
        arrays = collect(args.workspace, 'mamujoco', cfg, args.out)
        all_rows, inputs = [], {}
        for seed in cfg['upstream_seeds']['mamujoco']:
            rows, hashes, values = score(args.workspace, seed, arrays, cfg)
            all_rows.extend(rows); inputs.update(hashes)
            write(args.out/f'seed{seed}.json', dict(rows=rows, input_hashes=hashes))
            np.savez_compressed(args.out/f'seed{seed}_predictions.npz', **values)
            write(args.out/'status.json', dict(status='running', stage='frozen_prediction', completed_seed=seed))
        contrasts = {}
        for control in ['partner_mismatched_edge', 'zero']:
            diffs = []
            for seed in cfg['upstream_seeds']['mamujoco']:
                r = {x['method']:x['standardized_mse'] for x in all_rows
                     if x['seed']==seed and x['horizon']==1 and x['target']=='first_order'}
                diffs.append(r[control]-r['matched_edge'])
            contrasts[control] = paired(diffs)
        ordered = sorted(contrasts, key=lambda k: contrasts[k]['exact_p'])
        last = 0.
        for i, key in enumerate(ordered):
            last = max(last, min(1., (len(ordered)-i)*contrasts[key]['exact_p']))
            contrasts[key]['holm_p'] = last
        assert all(fingerprint(p)==v for p,v in {**sources,**inputs}.items())
        write(args.out/'source_manifest.json', dict(code=sources, inputs=inputs))
        write(args.out/'summary.json', dict(rows=all_rows, primary_h1_error_reduction=contrasts,
            scope='Five existing checkpoints, fresh evaluation episodes; same-state zero reference, not actual donor or control',
            projection='first 27 observation coordinates; contact channels excluded',
            training_updates=0))
        write(args.out/'status.json', dict(status='complete', seeds=len(cfg['upstream_seeds']['mamujoco']),
            rows=len(all_rows), source_and_weight_checks=True, training_updates=0))
        print(json.dumps(contrasts, ensure_ascii=False))
    except Exception as exc:
        write(args.out/'status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
