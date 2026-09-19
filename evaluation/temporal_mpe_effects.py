"""Independent physical checks after all models are frozen; never training data."""
import json
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from mte.frontends import ObservationReadout
from environments.effect_evaluation import EffectEnvironment
from utils.atomic import atomic_json


def evaluate(seedout, p, progress):
    records = []
    episode_count = p.get('physical_evaluation_episodes', 16)
    episode_start = p.get('physical_evaluation_seed_start', 97210000)
    for episode in range(episode_count):
        env = EffectEnvironment('mpe', episode_start+episode)
        try:
            previous = env.observation()
            for t in range(15):
                current = env.observation(); action = env.action()
                if t == 0: action[0] = 0
                if t in (2, 6, 10, 14):
                    snap = env.snapshot()
                    seq = np.repeat(action[None], 3, 0)
                    null = seq.copy(); null[0, 0] = 0
                    allzero = seq.copy(); allzero[0] = 0
                    factual, _ = env.roll(snap, seq)
                    reference, _ = env.roll(snap, null)
                    all_reference, _ = env.roll(snap, allzero)
                    replay, _ = env.roll(snap, seq)
                    assert np.array_equal(factual, replay)
                    assert np.array_equal(factual[0], reference[0])
                    records.append((previous, current, factual, reference, all_reference))
                    env.restore(snap)
                previous = current; env.step(action)
        finally: env.close()
    previous, current, factual, reference, all_reference = [np.stack(a) for a in zip(*records)]
    truth = factual[:, :, :8]-reference[:, :, :8]
    assert np.square(truth[:, 1]).mean() > 1e-10, 'No h2 physical signal'
    rows = []
    for offset in (1, 2):
        root = seedout/'pretrain'/f'offset{offset}'
        fc = torch.load(root/'entity/frontend.pt', map_location='cpu', weights_only=False)
        net = C.LAOMStateAdapter(2, 16, 64); net.load_state_dict(fc['state_dict']); net.eval()
        def code(future):
            a = torch.tensor((current[:, :8].reshape(-1, 2)-fc['mean'])/fc['std'])
            b = torch.tensor((future[:, :8].reshape(-1, 2)-fc['mean'])/fc['std'])
            with torch.no_grad(): return net(a, b)[1].numpy().reshape(-1, 4, 16)
        actual_code = code(factual[:, offset-1]); reference_code = code(all_reference[:, offset-1])
        if offset == 1: assert np.array_equal(actual_code, reference_code)
        bc = torch.load(root/'bridge/readout.pt', map_location='cpu', weights_only=False)
        predictor = ObservationReadout(16, 64, 24); predictor.load_state_dict(bc['state_dict']); predictor.eval()
        h = torch.tensor((np.concatenate([previous, current], -1)-bc['history_mean'])/bc['history_std'])
        def pred(z):
            zz = torch.tensor((z.reshape(-1, 64)-bc['zmean'])/bc['zstd'])
            with torch.no_grad(): return predictor(h, zz).numpy().reshape(-1, 3, 8)*bc['delta_std']+bc['delta_mean']
        matched_code = actual_code.copy(); matched_code[:, 0] = reference_code[:, 0]
        plus = pred(actual_code); matched = plus-pred(matched_code); mismatched = plus-pred(reference_code)
        for horizon in (1, 2, 3):
            i = horizon-1; scale = bc['delta_std'][i]
            rows.append(dict(offset=offset, horizon=horizon,
                physical_effect_rms=float(np.sqrt(np.square(truth[:, i]).mean())),
                code_change_rms=float(np.sqrt(np.square(actual_code-reference_code).mean())),
                matched_mse=float(np.square(matched[:, i]-truth[:, i]).mean()),
                mismatched_mse=float(np.square(mismatched[:, i]-truth[:, i]).mean()),
                zero_mse=float(np.square(truth[:, i]).mean()),
                matched_standardized_mse=float(np.square((matched[:, i]-truth[:, i])/scale).mean()),
                mismatched_standardized_mse=float(np.square((mismatched[:, i]-truth[:, i])/scale).mean())))
        np.savez_compressed(seedout/f'physical_predictions_offset{offset}.npz', truth=truth,
                            matched=matched, mismatched=mismatched, actual_code=actual_code, reference_code=reference_code)
    np.savez_compressed(seedout/'physical_evaluation_only.npz', previous=previous, current=current,
                        factual=factual, reference=reference, all_reference=all_reference)
    atomic_json(seedout/'physical_effects.json', dict(seed=p['seed'], episodes=episode_count,
        episode_seed_start=episode_start, samples=len(records),
        rows=rows, primary_horizon=2, replay_exact=True, training_updates=0,
        reference='Evaluation-only same-state zero action. Not training nearest-history donor.',
        continuation='Repeated factual joint action; only first-step target action is changed.'))
    progress('physical_complete', samples=len(records))
