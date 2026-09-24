from mte.method_names import resolve_method
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
import json
ROOT=Path(__file__).resolve().parents[1]
from closed_loop_lam_v1.training_profiles import resolve_method_training_settings
EDGE_METHODS=frozenset({"edge_cara","edge_cara_mobius_simple","edge_cara_mobius_graph","random_edge_encoder","matched_cf_deepsets","edge_cara_mobius_tree","edge_cara_mif"})
OBSERVATION_METHODS=frozenset({"continuous_lam","lapo_state_adapter","laom_state_adapter"})

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/backend.py:27

def _geometry(positions, valid_mask, n_train):
    x = np.asarray(positions, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] < 1:
        raise ValueError('Expected complete padded observation trajectories [E,T+1,D]')
    if mask.shape != (len(x), x.shape[1]-1) or not 0 < n_train < len(x):
        raise ValueError('Expected transition mask [E,T], nonempty train/dev partitions')
    if not np.isfinite(x).all() or np.any(np.diff(mask.astype(np.int8), axis=1) > 0):
        raise ValueError('Nonfinite observations or non-prefix transition mask')
    target_mask = mask
    if not target_mask[:n_train].any() or not target_mask[n_train:].any():
        raise ValueError('Train and dev must contain real transitions')
    return x, mask, target_mask

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/backend.py:42

def train_backend(positions, valid_mask, actual, reference, masks, method, seed,
                  n_train, updates=3200, device='cuda', rich=None):
    """Return (z [E,T,d], diagnostics), directly using common's trainer.

    Input observations preserve the original D-dimensional interface (currently
    D=105). Predicted h=1 endpoints are [E,T,8,D] for decision times 0..T-1.
    Original baseline history handling is retained, including LAPO's episode-
    local duplication of x0 when no earlier observation exists at t0.
    """
    method = resolve_method(method)
    x, valid, target_mask = _geometry(positions, valid_mask, n_train)
    e, _, obs_dim = x.shape
    steps = x.shape[1]-1
    if method not in EDGE_METHODS | OBSERVATION_METHODS:
        raise ValueError('Method outside registered action-free set: ' + method)
    if updates < 1:
        raise ValueError('Positive training update count required')
    if method in OBSERVATION_METHODS:
        nxt = x[:, 1:]
        data = {'obs': x, 'actions': np.zeros((e, steps, 2), np.float32),
                'valid_step_mask': valid, 'cf_root': nxt,
                'cf_ego_null': np.zeros_like(nxt), 'cf_other_null': nxt,
                'cf_both_null': np.zeros_like(nxt)}
    else:
        a, b = np.asarray(actual, np.float32), np.asarray(reference, np.float32)
        cm = np.asarray(masks, dtype=bool)
        if a.shape != (e, steps, 8, obs_dim) or b.shape != a.shape:
            raise ValueError('Expected two predicted endpoints [E,T,8,D]')
        if cm.shape != (8, 4) or cm[0].any() or cm[:, 0].any():
            raise ValueError('Expected eight partner masks excluding target agent zero')
        if len(np.unique(cm, axis=0)) != 8:
            raise ValueError('Partner masks must enumerate the eight distinct subsets')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite predicted endpoints')
        data = {'obs': x, 'actions': np.zeros((e, steps, 2), np.float32),
                'valid_step_mask': target_mask,
                'cf_root': a[:, :, 0], 'cf_ego_null': b[:, :, 0],
                'cf_other_null': a[:, :, -1], 'cf_both_null': b[:, :, -1],
                'cf_context_root': a, 'cf_context_ego_null': b,
                'cf_context_masks': cm}
    if method=='edge_cara_mif':
        if rich is None: raise ValueError('MIF requires newly predicted temporal entity observations')
        ra,rb=rich
        if ra.shape!=(e,steps,3,8,4,46) or rb.shape!=ra.shape: raise ValueError('Invalid MIF learned-entity schema')
        data.update(cf_horizon_entity_root=ra,cf_horizon_entity_ego_null=rb,
                    mif_entity_adjacency=np.ones((4,4),np.float32),effect_horizons=np.asarray([1,2,3]))
    result = C.train_representation(
        data, method, np.arange(n_train, dtype=np.int64), seed,
        z_dim=16, hidden=json.loads((ROOT/'closed_loop_lam_v1/MAMUJOCO_FORMAL_1M_CAPACITY.json').read_text())['recommended'].get(method,{'hidden':128})['hidden'], updates=updates, batch_size=512,
        device=device, training_profile='matched', requested_policy_updates=1200)
    z = np.asarray(result.z, np.float32).reshape(e, steps, -1)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite representation output')
    # Invalidate padded representations explicitly; normalization and losses
    # below also mask them, so neither zero padding nor terminal repetitions fit.
    z = np.where(target_mask[..., None], z, 0.).astype(np.float32)
    diag = dict(result.diagnostics)
    diag.update(source_function='closed_loop_lam_v1.common.train_representation',
                representation_parameters=int(result.parameter_count),
                representation_train_loss=(float(result.train_loss)
                    if np.isfinite(result.train_loss) else None),
                native_action_labels_read=0, simulator_queries=0,
                original_observation_dim=obs_dim, timestamps=[0, steps-1],
                valid_train_targets=int(target_mask[:n_train].sum()),
                valid_dev_targets=int(target_mask[n_train:].sum()),
                input_evidence=('factual observations only' if method in OBSERVATION_METHODS
                    else 'observation-trained model predicted endpoints; not simulator branches'),
                compatibility_actions='shape-only zeros; no native action values')
    return z, diag

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/backend.py:112

def train_history_policy(positions, z, valid_mask, n_train, seed, updates,
                         folder, progress, device='cuda', method='unspecified'):
    """Freeze original GRU trained on masked observation-derived latent labels.

    Full x0..x(T-1) history is passed through the original RecurrentPolicy.
    Losses use every real transition including t0. No zero-action warmup is
    introduced and there is no temporal truncation.
    Returns diagnostics; writes policy.pt, latents.npz and history.json.
    """
    method = resolve_method(method)
    x, valid, target_mask = _geometry(positions, valid_mask, n_train)
    z = np.asarray(z, np.float32)
    if z.ndim != 3 or z.shape[:2] != (len(x), x.shape[1]-1) or not np.isfinite(z).all():
        raise ValueError('Expected finite latent targets [E,T,d]')
    if updates < 1:
        raise ValueError('Positive policy update count required')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    C.seed_all(seed)
    obs = x[:, :-1]
    om, osd = C.fit_mean_std(obs[:n_train][valid[:n_train]])
    zm, zs = C.fit_mean_std(z[:n_train][target_mask[:n_train]])
    ot = torch.as_tensor((obs-om)/osd, device=device)
    zt = torch.as_tensor((z-zm)/zs, device=device)
    mt = torch.as_tensor(target_mask, device=device)
    net = C.RecurrentPolicy(x.shape[-1], z.shape[-1], hidden=128).to(device)
    settings = resolve_method_training_settings(method, 'matched', updates, updates, 16*200, int(target_mask[:n_train].sum()))
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.policy_lr)
    scheduler = C._profile_scheduler(optimizer, settings.use_linear_warmup_decay, updates)
    rng = np.random.default_rng(seed+101)
    eligible = np.flatnonzero(target_mask[:n_train].any(1))
    best, selected, best_state, history = float('inf'), 0, None, []

    def dev_loss():
        total, count = 0., 0
        net.eval()
        with torch.no_grad():
            for lo in range(n_train, len(x), 8):
                hi = min(lo+8, len(x))
                pred, _ = net(ot[lo:hi])
                values = (pred-zt[lo:hi]).square()[mt[lo:hi]]
                total += values.sum().item()
                count += values.numel()
        return total/count

    for step in range(1, updates+1):
        net.train()
        ix = rng.choice(eligible, min(16, len(eligible)), replace=False)
        prediction, _ = net(ot[ix])
        loss = (prediction-zt[ix]).square()[mt[ix]].mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite masked history-policy loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if scheduler is not None: scheduler.step()
        if step == 1 or step % 100 == 0 or step == updates:
            value = dev_loss()
            if not np.isfinite(value):
                raise ValueError('Nonfinite masked development loss')
            history.append({'update': step, 'train_mse': loss.item(), 'dev_mse': value})
            if value < best:
                best, selected = value, step
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            progress('latent_policy', update=step, updates=updates, dev_mse=value)
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        deployment = torch.cat([net(batch)[0].cpu() for batch in ot.split(8)]).numpy()
    deployment[~target_mask] = 0.
    if not np.isfinite(deployment).all():
        raise ValueError('Nonfinite deployment latent values')
    torch.save({'state_dict': best_state, 'obs_mean': om, 'obs_std': osd,
                'z_mean': zm, 'z_std': zs, 'z_dim': z.shape[-1], 'hidden': 128,
                'obs_dim': x.shape[-1],
                'source_class': 'closed_loop_lam_v1.common.RecurrentPolicy',
                'history': 'x_0 through x_t only; masked latent targets t=0..T-1; no action warmup',
                'selected_update': selected, 'native_action_labels_read': 0,
                'simulator_queries': 0, 'frozen_before_grounding': True}, folder/'policy.pt')
    np.savez_compressed(folder/'latents.npz', representation_z=z,
                        deployment_z=deployment, valid_step_mask=target_mask)
    (folder/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return {'dev_standardized_latent_mse': best, 'selected_update': selected,
            'policy_parameters': sum(p.numel() for p in net.parameters()),
            'checkpoint': str(folder/'policy.pt'), 'z_dim': z.shape[-1],
            'obs_dim': x.shape[-1], 'policy_batch_episodes': 16, 'policy_effective_lr': settings.policy_lr, 'policy_scheduler': settings.use_linear_warmup_decay,
            'policy_updates': updates, 'normalization': 'valid training entries only',
            'valid_train_targets': int(target_mask[:n_train].sum()),
            'valid_dev_targets': int(target_mask[n_train:].sum()),
            'warning': 'Latent MSE is not a ranking of control quality'}
