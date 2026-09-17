import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
OBS_DIM=105
ACTION_DIM=2
from closed_loop_lam_v1.common import ActionDecoder,RecurrentPolicy,InverseDynamics,seed_all

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:24

def _prefix_mask(mask, episodes, steps):
    mask = np.asarray(mask)
    if mask.shape != (episodes, steps) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected binary validity mask aligned to t0..T-1, without slicing')
    mask = mask.astype(bool)
    if np.any(mask[:, 1:] & ~mask[:, :-1]) or np.any(mask.sum(1) == 0):
        raise ValueError('Every trajectory must have a nonempty valid prefix')
    return mask

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:34

def _split(fit, val, budget):
    fit = np.asarray(fit, dtype=np.int64)
    val = np.asarray(val, dtype=np.int64)
    if (fit.ndim != 1 or val.ndim != 1 or not len(fit) or not len(val)
            or sorted(np.concatenate([fit, val]).tolist()) != list(range(budget))):
        raise ValueError('Fit and validation must partition the entire action budget')
    return fit, val

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:43

def _positions(positions, valid_mask):
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != OBS_DIM or positions.shape[1] < 3:
        raise ValueError('Expected complete original 105-dimensional observations')
    if not np.isfinite(positions).all():
        raise ValueError('Nonfinite observation')
    steps = positions.shape[1] - 1
    mask = _prefix_mask(valid_mask, len(positions), steps)
    obs = positions[:, :-1]
    mean = obs[mask].mean(0)
    std = obs[mask].std(0).clip(1e-6)
    return positions, mask, mean, std

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:57

def _labels(actions, shape):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != shape or not np.isfinite(actions).all() or np.abs(actions).max() > 1.00001:
        raise ValueError('Invalid budgeted target-agent continuous action array')
    return actions

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:64

def _loss(pred, target, mask):
    valid = mask.bool()
    if not bool(valid.any()):
        raise ValueError('No valid loss units')
    return (pred[valid] - target[valid]).square().mean()

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:71

def _copy(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:75

def train_decoder(z, actions, mask, fit, val, seed, updates=800, device='cuda'):
    """Original ActionDecoder; checkpoint selection uses only budgeted val."""
    z = np.asarray(z, np.float32)
    if z.ndim != 3 or not np.isfinite(z).all():
        raise ValueError('Latents must be finite (B,T,z_dim)')
    mask = _prefix_mask(mask, len(z), z.shape[1])
    actions = _labels(actions, (*z.shape[:2], ACTION_DIM))
    fit, val = _split(fit, val, len(z))
    if updates < 1:
        raise ValueError('updates must be positive')
    seed_all(seed)
    net = ActionDecoder(z.shape[-1], ACTION_DIM).to(device)
    x = torch.as_tensor(z, device=device)
    y = torch.as_tensor(actions, device=device)
    m = torch.as_tensor(mask, device=device)
    xx, yy = x[fit][m[fit]], y[fit][m[fit]]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 211)
    best, selected, state = float('inf'), 0, None
    for step in range(1, updates + 1):
        ix = rng.integers(0, len(xx), min(512, len(xx)))
        loss = (net(xx[ix]) - yy[ix]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite decoder loss')
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % 25 == 0 or step == updates:
            with torch.no_grad():
                score = _loss(net(x[val]), y[val], m[val]).item()
            if score < best:
                best, selected, state = score, step, _copy(net)
    net.load_state_dict(state)
    return net.cpu().eval(), dict(validation_action_mse=best, selected_update=selected,
        updates=updates, fit_valid_steps=int(mask[fit].sum()),
        validation_valid_steps=int(mask[val].sum()), budget_trajectories=len(z),
        source_class='closed_loop_lam_v1.common.ActionDecoder')

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:112

def _history(positions, mask, targets, training_ids, validation_ids, validation_actions,
             mean, std, seed, updates, device):
    if updates < 1:
        raise ValueError('updates must be positive')
    seed_all(seed)
    x = torch.as_tensor((positions[:, :-1] - mean) / std, device=device)
    y = torch.as_tensor(targets, device=device)
    vy = torch.as_tensor(validation_actions, device=device)
    m = torch.as_tensor(mask, device=device)
    net = RecurrentPolicy(OBS_DIM, ACTION_DIM).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 313)
    best, selected, state = float('inf'), 0, None
    for step in range(1, updates + 1):
        ix = rng.choice(training_ids, min(16, len(training_ids)), replace=False)
        pred, _ = net(x[ix])
        loss = _loss(pred, y[ix], m[ix])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite history policy loss')
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % 25 == 0 or step == updates:
            with torch.no_grad():
                pred, _ = net(x[validation_ids])
                score = _loss(pred, vy, m[validation_ids]).item()
            if score < best:
                best, selected, state = score, step, _copy(net)
    net.load_state_dict(state)
    return net.cpu().eval(), mean, std, dict(validation_action_mse=best,
        selected_update=selected, updates=updates,
        source_class='closed_loop_lam_v1.common.RecurrentPolicy',
        objective='continuous action MSE; all valid t0..T-1; no warmup override',
        normalization='all training observations, valid positions only; no action labels',
        fit_valid_steps=int(mask[training_ids].sum()),
        validation_valid_steps=int(mask[validation_ids].sum()))

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:148

def _budget_inputs(positions, valid_mask, labels, local_ids, fit, val):
    positions, mask, mean, std = _positions(positions, valid_mask)
    local_ids = np.asarray(local_ids, dtype=np.int64)
    if local_ids.ndim != 1 or len(set(local_ids.tolist())) != len(local_ids):
        raise ValueError('Budget episode IDs must be unique')
    if np.any(local_ids < 0) or np.any(local_ids >= len(positions)):
        raise ValueError('Budget episode ID outside training observations')
    labels = _labels(labels, (len(local_ids), mask.shape[1], ACTION_DIM))
    fit, val = _split(fit, val, len(local_ids))
    return positions, mask, mean, std, labels, local_ids, fit, val

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:160

def train_bc(positions, valid_mask, labels, local_ids, fit, val, seed,
             updates=3200, device='cuda'):
    """Original recurrent BC class, with no labels outside the declared budget."""
    p, m, mean, std, labels, ids, fit, val = _budget_inputs(
        positions, valid_mask, labels, local_ids, fit, val)
    targets = np.zeros((*m.shape, ACTION_DIM), np.float32)
    targets[ids[fit]] = labels[fit]
    return _history(p, m, targets, ids[fit], ids[val], labels[val],
                    mean, std, seed, updates, device)

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:171

def train_idm(positions, valid_mask, labels, local_ids, fit, val, seed,
              updates=800, policy_updates=1200, device='cuda'):
    """Original IDM and GRU classes; fit IDM, pseudo-label observations, clone.

    IDM and policy checkpoint selection both use the counted validation labels.
    Pseudo-label BC excludes those validation trajectories from its fitting set.
    """
    p, m, mean, std, labels, ids, fit, val = _budget_inputs(
        positions, valid_mask, labels, local_ids, fit, val)
    if updates < 1:
        raise ValueError('updates must be positive')
    seed_all(seed)
    x = torch.as_tensor((p[:, :-1] - mean) / std, device=device)
    nxt = torch.as_tensor((p[:, 1:] - mean) / std, device=device)
    mask = torch.as_tensor(m, device=device)
    xx = x[ids[fit]][mask[ids[fit]]]
    nn = nxt[ids[fit]][mask[ids[fit]]]
    yy = torch.as_tensor(labels[fit], device=device)[mask[ids[fit]]]
    vy = torch.as_tensor(labels[val], device=device)
    net = InverseDynamics(OBS_DIM, ACTION_DIM).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 919)
    best, selected, state = float('inf'), 0, None
    for step in range(1, updates + 1):
        ix = rng.integers(0, len(xx), min(512, len(xx)))
        loss = (net(xx[ix], nn[ix]) - yy[ix]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite IDM loss')
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % 25 == 0 or step == updates:
            with torch.no_grad():
                score = _loss(net(x[ids[val]], nxt[ids[val]]), vy, mask[ids[val]]).item()
            if score < best:
                best, selected, state = score, step, _copy(net)
    net.load_state_dict(state)
    with torch.no_grad():
        pseudo = net(x, nxt).cpu().numpy()
    training_ids = np.setdiff1d(np.arange(len(p)), ids[val])
    result, om, os, diag = _history(p, m, pseudo, training_ids, ids[val], labels[val],
                                  mean, std, seed + 1, policy_updates, device)
    diag.update(idm_updates=updates, idm_selected_update=selected,
                idm_validation_action_mse=best,
                idm_source_class='closed_loop_lam_v1.common.InverseDynamics',
                validation='counted true-action validation; excluded from pseudo-BC fit')
    return result, om, os, diag

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/grounding.py:218

class GroundedPolicy:
    """Original evaluator interface, acting immediately from the first state."""
    def __init__(self, net, mean, std, decoder=None):
        self.net = net.cpu().eval()
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        if self.mean.shape != (OBS_DIM,) or self.std.shape != (OBS_DIM,) or np.any(self.std <= 0):
            raise ValueError('Expected original 105-dimensional normalization')
        self.decoder = None if decoder is None else decoder.cpu().eval()

    def act_step(self, observation, hidden_state=None, device='cpu'):
        if device != 'cpu':
            raise ValueError('Grounded evaluation is CPU-only')
        obs = np.asarray(observation, np.float32)
        if obs.shape != (OBS_DIM,):
            raise ValueError('Do not slice or change the official observation')
        with torch.no_grad():
            x = torch.as_tensor((obs - self.mean) / self.std).reshape(1, 1, OBS_DIM)
            prediction, hidden = self.net(x, hidden_state)
            value = prediction[:, -1]
            action = value if self.decoder is None else self.decoder(value)
        action = action.numpy().reshape(ACTION_DIM).clip(-1, 1).astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError('Nonfinite deployment action')
        return action, hidden

    def act(self, observation, hidden=None):
        return self.act_step(observation, hidden)
