"""X8-X10. Frozen target effects, same-slot comparisons, and history transfer.

Development diagnostic, NOT new policy training or confirmatory control evidence.
Run with `python -m experiments.target_effects --help` from the package root.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(workspace, env, seed):
    """Read-only final lineage; paths are explicit in each saved run manifest."""
    n = 700 if env == 'mpe' else 90
    sid = f'{env}_s{seed}_n{n}'
    old = workspace/'observation_scale_restore_2026_09_10/runtime'
    history = {name: old/(sid+'__pre_'+method)/'policy/policy.pt' for name, method in
               [('base', 'entity_target'), ('lapo', 'lapo_joint'), ('laom', 'laom_joint_k3')]}
    route = workspace/'mte_family_transfer_2026_09_16/runtime'
    representations = {}
    for name in ('simple', 'graph', 'mif'):
        d = route/(sid+'__pre_'+name+'_matched')
        representations[name] = d/'representation.pt'
        history[name+'_route'] = d/'policy/policy.pt'
    history['global_joint16'] = workspace/'mte_global_aux_scope_2026_09_16/runtime'/(sid+'__pre_global')/'policy/policy.pt'
    formal = {'edge_h1': 'edge_cara' if env == 'mpe' else 'edge_cara_h1',
              'simple_formal': 'edge_cara_mobius_simple', 'graph_formal': 'edge_cara_mobius_graph',
              'mif_formal': 'edge_cara_mif'}
    for name, method in formal.items():
        p = old/(sid+'__pre_'+method)/'policy/policy.pt'
        if env == 'mamujoco' and name in ('simple_formal', 'graph_formal'):
            p = workspace/'mamujoco_mte_recovery_2026_09_12/runtime'/(sid+'__pre_'+method)/'policy/policy.pt'
        elif env == 'mamujoco' and name == 'mif_formal':
            p = workspace/'mamujoco_independent_branch_audit_2026_09_12/repair'/sid/'policy/policy.pt'
        history[name] = p
    front = {name: old/(sid+'__front_'+variant)/'frontend.pt' for name, variant in
             [('entity', 'laom_entity_k3'), ('lapo', 'lapo_joint'), ('laom', 'laom_joint_k3')]}
    result = dict(env=env, seed=seed, suite=sid, history=history, representations=representations,
                  frontends=front, readout=old/(sid+'__bridge/readout.pt'),
                  training_observations=old/(sid+'__prepare/train.npz'))
    paths = [*history.values(), *representations.values(), *front.values(), result['readout'], result['training_observations']]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2))
    result['files'] = {str(p): fingerprint(p) for p in paths}
    return result


def collect(workspace, env_name, cfg, out):
    from environments.effect_evaluation import EffectEnvironment
    teacher = None
    if env_name == 'mamujoco':
        from environments.teacher import CentralSACTeacher, sha256_file
        gate_path = workspace/cfg['teacher_gate']
        gate = json.loads(gate_path.read_text(encoding='utf-8'))
        path = Path(gate['checkpoint'])
        if not path.is_absolute():
            path = workspace/path
        assert gate['passed'] and gate['scenario'] == 'Ant' and gate['agent_conf'] == '4x2'
        assert sha256_file(path) == gate['checkpoint_sha256']
        teacher = CentralSACTeacher(path, 'Ant')
    rows, trajectories, max_replay_error, total_steps, replay_steps = [], [], 0., 0, 0
    seeds = cfg['episode_seeds'][env_name]
    times = cfg['decision_times']
    length = max(times)+3
    t0 = time.time()
    for episode, seed in enumerate(seeds):
        env = EffectEnvironment(env_name, seed, teacher)
        try:
            obs, ledger, snapshots, valid = [env.observation()], [], [], []
            done = False
            for t in range(length):
                snapshots.append(env.snapshot()); valid.append(not done)
                if not done:
                    a = env.action()
                    # Preserve existing particle evaluation's target warmup.
                    if env_name == 'mpe' and t == 0:
                        a[0] = 0
                    done = env.step(a); total_steps += 1
                else:
                    a = np.zeros((4, 2), np.float32)
                ledger.append(a); obs.append(env.observation())
            obs, ledger = np.array(obs), np.array(ledger)
            trajectories.append(obs)
            for t in times:
                if not all(valid[t:t+3]):
                    continue
                actions = ledger[t:t+3]
                cube, lives = [], []
                for assignment in range(16):
                    seq = actions.copy()
                    for agent in range(4):
                        if (assignment >> agent) & 1:
                            seq[0, agent] = 0.
                    values, live = env.roll(snapshots[t], seq)
                    total_steps += int(live.sum()); cube.append(values); lives.append(live)
                cube, lives = np.array(cube), np.array(lives)
                replay, replay_live = env.roll(snapshots[t], actions)
                replay_steps += int(replay_live.sum())
                error = max(float(np.abs(replay-cube[0]).max()), float(np.abs(replay-obs[t+1:t+4]).max()))
                max_replay_error = max(max_replay_error, error)
                if error > cfg['restore_atol']:
                    raise ValueError(f'Restore/replay mismatch: {env_name} {seed} t={t} error={error}')
                rows.append(dict(episode=episode, time=t, previous=obs[t-1], current=obs[t],
                                 factual_future=obs[t+1:t+4], cube=cube, live=lives,
                                 target_action=actions[0, 0]))
        finally:
            env.close()
        write(out/'progress.json', dict(stage='collect_evaluation_only', environment=env_name,
              episodes=episode+1, total_episodes=len(seeds), samples=len(rows), seconds=time.time()-t0))
    if not rows:
        raise ValueError('No complete natural transitions')
    arrays = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    arrays['trajectories'] = np.array(trajectories)
    arrays['episode_seeds'] = np.array(seeds)
    np.savez_compressed(out/'physical_evaluation_only.npz', **arrays)
    effect = arrays['cube'][:, 0]-arrays['cube'][:, 1]
    physical_dim = 8 if env_name == 'mpe' else 27
    meta = dict(samples=len(rows), episodes=len(seeds), simulator_steps=total_steps,
                verification_replay_steps=replay_steps, total_physics_steps_including_verification=total_steps+replay_steps,
                max_replay_error=max_replay_error, seconds=time.time()-t0,
                target_effect_rms_by_horizon=np.sqrt(np.mean(effect[..., :physical_dim].astype(float)**2, axis=(0, 2))).tolist(),
                reference='Zero native action for selected agents at first step; locked factual continuation',
                evidence='Evaluation-only native-action assignment cube; no training export',
                termination='absorbing hold; live flags saved, h3 scores restricted to all-branch live samples')
    write(out/'collection.json', meta)
    return arrays


def score(bundle, env_name, cfg, arrays, out):
    from evaluation.frozen_effect_models import Frontend, Predictor, history_features, route_representation
    from evaluation.effect_statistics import target_edges, single_partner_interactions, partition, fit_reader, metrics
    pre, cur, future = arrays['previous'], arrays['current'], arrays['factual_future']
    ep, times = arrays['episode'], arrays['time']
    # Models are frozen before any diagnostic reader sees evaluation targets.
    fronts = {k: Frontend(v, env_name) for k, v in bundle['frontends'].items()}
    predictor = Predictor(bundle['readout'], env_name, bundle['training_observations'])
    q = fronts['entity'].encode(pre, cur, future[:, 0])
    ref_future = arrays['cube'][:, 15, 0]
    ref = fronts['entity'].encode(pre, cur, ref_future)
    continuation = np.stack([fronts['entity'].encode(cur, future[:, 0], future[:, 1]),
                             fronts['entity'].encode(future[:, 0], future[:, 1], future[:, 2])], 1)
    actual, reference = [], []
    for mask in range(8):
        za = q.copy()
        for partner in range(1, 4):
            if (mask >> (partner-1)) & 1:
                za[:, partner] = ref[:, partner]
        zb = za.copy(); zb[:, 0] = ref[:, 0]
        assert np.array_equal(za[:, 1:], zb[:, 1:])
        actual.append(predictor.rollout(pre, cur, za, continuation))
        reference.append(predictor.rollout(pre, cur, zb, continuation))
    a, b = np.stack(actual, 2), np.stack(reference, 2)  # N,H,C,D
    edge = a-b
    odim = predictor.out_dim
    true = target_edges(arrays['cube'][..., :odim]).transpose(0, 2, 1, 3)
    physical_coeff = single_partner_interactions(true.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    prediction_coeff = single_partner_interactions(edge.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    split_cfg = cfg['splits']
    splits = partition(ep, **split_cfg)
    # Coordinate scale from calibration natural observations, no test fitting.
    coordinate_scale = cur[splits[0], :odim].std(0).astype(np.float64)
    coordinate_scale[coordinate_scale < 1e-4] = 1.0
    rows, predictions = [], {}
    columns = 8 if env_name == 'mpe' else 27
    def add(name, view, target_name, y, x=None, direct=None, horizon=1, eligible=None):
        keep = np.ones(len(ep), bool) if eligible is None else eligible
        sp = tuple(s[keep] for s in splits)
        if not all(s.any() for s in sp):
            rows.append(dict(method=name, view=view, target=target_name, horizon=horizon, skipped='empty split after horizon validity'))
            return
        y = y[keep].reshape(int(keep.sum()), -1)
        scale = np.tile(coordinate_scale[:columns] if target_name != 'native_action' else np.ones(2), y.shape[1]//(columns if target_name != 'native_action' else 2))
        if direct is None:
            pred, diag = fit_reader(x[keep], y, sp, scale, cfg['reader_alphas'])
        else:
            pred = direct[keep].reshape(len(y), -1)[sp[2]]
            diag = metrics(pred, y[sp[2]], y[sp[0]], scale)
        key = f'p{len(rows):04d}'
        predictions[key] = pred
        rows.append(dict(key=key, method=name, view=view, target=target_name, horizon=horizon,
                         calibration_samples=int(sp[0].sum()), test_samples=int(sp[2].sum()),
                         test_episode_ids=ep[keep][sp[2]].tolist(), **diag))

    # Direct raw operator comparisons do not fit a reader to physical targets.
    for horizon in (1, 2, 3):
        h = horizon-1
        eligible = arrays['live'][:, :, :horizon].all((1, 2))
        y = true[:, h, 0, :columns]
        add('matched_edge', 'direct_prediction', 'physical_target_effect', y, direct=edge[:, h, 0, :columns], horizon=horizon, eligible=eligible)
        add('partner_mismatched_edge', 'direct_prediction', 'physical_target_effect', y, direct=(a[:, h, 0]-b[:, h, 7])[:, :columns], horizon=horizon, eligible=eligible)
        for partner in range(3):
            add('matched_coefficient', 'direct_prediction', f'physical_partner_{partner+1}', physical_coeff[:, h, partner, :columns],
                direct=prediction_coeff[:, h, partner, :columns], horizon=horizon, eligible=eligible)

    features = {'target_slot': q[:, 0], 'target_slot_pair': np.concatenate([q[:, 0], ref[:, 0]], -1),
                'raw_edge_h1': edge[:, 0, 0], 'raw_edge_table_h1': edge[:, 0].reshape(len(ep), -1)}
    for name in ('lapo', 'laom'):
        features[name+'_joint'] = fronts[name].encode(pre, cur, future[:, 0])
        other = fronts[name].encode(pre, cur, ref_future)
        features[name+'_joint_pair'] = np.concatenate([features[name+'_joint'], other], -1)
    for name, path in bundle['representations'].items():
        features[name+'_route'] = route_representation(path, a, b, cur, env_name)
    causality = {}
    histories = {}
    for name, path in bundle['history'].items():
        histories[name], causality[name] = history_features(path, arrays['trajectories'], ep, times)
    # Reuse the exact Base history vector for every concatenation.
    for name in ['simple_route', 'graph_route', 'mif_route', 'simple_formal', 'graph_formal', 'mif_formal', 'global_joint16']:
        histories['base_plus_'+name] = np.concatenate([histories['base'], histories[name]], -1)
    primary_horizon = cfg['physical_primary_horizon'][env_name]
    primary_idx = primary_horizon-1
    primary_live = arrays['live'][:, :, :primary_horizon].all((1, 2))
    for view, fs in [('transition_informed_diagnostic', features), ('history_only', histories)]:
        for name, feature in fs.items():
            add(name, view, 'native_action', arrays['target_action'], x=feature)
            # A zero-reference effect of the naturally recorded action is history-predictable in principle.
            # No arbitrary future action pair is supplied to a history-only policy.
            add(name, view, 'physical_target_effect', true[:, primary_idx, 0, :columns], x=feature, horizon=primary_horizon, eligible=primary_live)
            add(name, view, 'physical_single_partner_interactions', physical_coeff[:, primary_idx, :, :columns], x=feature, horizon=primary_horizon, eligible=primary_live)
    np.savez_compressed(out/'reader_predictions.npz', **predictions)
    np.savez_compressed(out/'frozen_features.npz', **{'transition_'+k:v for k,v in features.items()},
                        **{'history_'+k:v for k,v in histories.items()}, episode=ep, time=times)
    code_difference = q-ref
    write(out/'scores.json', dict(suite=bundle['suite'], rows=rows, causality_prefix_max_errors=causality,
        inverse_one_step_code_change_rms=float(np.sqrt(np.mean(code_difference.astype(float)**2))),
        diagnostic_primary_horizon=primary_horizon,
        horizon_rationale=cfg['horizon_rationale'][env_name],
        evidence='Development diagnostic on fresh episodes, fixed zero-action reference, existing checkpoints',
        route_vs_formal='Route representations use full-visibility loss; formal history policies are separately named.',
        reference_scope='Controlled same-state intervention diagnostic, not original nearest-history-donor pipeline evaluation',
        fairness='Native widths and upstream information differ. Pair controls share paired observations. Readers share splits and ridge grid; not equal-parameter training.',
        history_transfer='Same native action/zero-reference targets, independently calibrated readers; not latent MSE across methods',
        primary_outcome='agent positions' if env_name == 'mpe' else 'first 27 state coordinates (kinematics), contact channels excluded',
        representation_updates=0, policy_updates=0, diagnostic_reader_supervision=True))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=ROOT/'configs/target_effects_pilot.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--environment', choices=['mpe', 'mamujoco'], required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'protocol.json', cfg)
    sources = [Path(__file__), ROOT/'evaluation/frozen_effect_models.py', ROOT/'evaluation/effect_statistics.py',
               ROOT/'environments/effect_evaluation.py']
    write(args.out/'implementation_sources.json', {str(p.relative_to(ROOT)):fingerprint(p) for p in sources})
    started = time.time()
    try:
        bundles = [manifest(args.workspace, args.environment, s) for s in cfg['upstream_seeds'][args.environment]]
        for b in bundles:
            write(args.out/(b['suite']+'_manifest.json'), json.loads(json.dumps(b, default=str)))
        arrays = collect(args.workspace, args.environment, cfg, args.out)
        for b in bundles:
            folder = args.out/b['suite']; folder.mkdir()
            write(args.out/'progress.json', dict(stage='frozen_model_diagnostics', suite=b['suite']))
            rows = score(b, args.environment, cfg, arrays, folder)
            for file, original in b['files'].items():
                if fingerprint(file) != original:
                    raise RuntimeError('Input artifact changed during evaluation: '+file)
            write(folder/'complete.json', dict(complete=True, rows=len(rows), input_artifacts_unchanged=True))
        write(args.out/'complete.json', dict(complete=True, upstream_seeds=len(bundles), seconds=time.time()-started,
              note='Diagnostic readers only; no training of representations, history policies, or controllers'))
    except Exception as exc:
        write(args.out/'failure.json', dict(error=repr(exc), seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
