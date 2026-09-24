"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

from pathlib import Path

import numpy as np

import torch

from utils.artifacts import data, endpoints

from training.representation_mamujoco import train_backend, train_history_policy as ma_history

from mte.entity_view import entities

from mte.structured_controls import train as structured_train

OBSERVATION_METHODS={'continuous_lam','lapo_state_adapter','laom_state_adapter'}

FRONTEND_METHODS={'entity_target','entity_joint','lapo_joint','laom_joint_k3'}

def train_history(s, method, out, progress, p):
    """Original MaMuJoCo representation/history dispatch, restricted to MaMuJoCo.
Caller supplies observation-only exports and installs the access guard."""
    if s['env'] != 'mamujoco':
        raise ValueError('Use mpe_inventory_completion for the corrected MPE protocol')
    x, m, n = data(s)
    seed = s['seed']
    env = s['env']
    out = Path(out)
    if not out.is_dir():
        raise ValueError('Create a fresh stage output directory first')
    updates = p['backend_updates'][env]
    if method in FRONTEND_METHODS:
        with np.load(s['codes'], allow_pickle=False) as f:
            z = f['z']
        if method == 'entity_target':
            z = z[:, :, 0]
        elif method == 'entity_joint':
            z = z.reshape(*z.shape[:2], -1)
        diag = {'source': 'frozen observation frontend', 'latent_dim': z.shape[-1]}
    else:
        obs = method in OBSERVATION_METHODS
        a = b = cm = None
        if not obs:
            with np.load(endpoints(s), allow_pickle=False) as f:
                a, b, cm = (f['actual'], f['reference'], f['masks'])
        if env == 'mamujoco':
            if method == 'edge_cara_mif':
                z, diag, ck = structured_train(entities(a), entities(b), m, n, cm, np.ones((4, 4), np.float32), env, seed, updates, 'mif_edge', progress)
                torch.save(ck, out / 'representation.pt')
            else:
                h = 0 if method == 'edge_cara_h1' else 2
                native = 'edge_cara' if method in {'edge_cara_h1', 'edge_cara_h3'} else method
                z, diag = train_backend(x, m, None if obs else a[:, :, h], None if obs else b[:, :, h], cm, native, seed, n, updates)
        else:
            raise ValueError('Unknown numeric environment')
    native = 'edge_cara' if method in {'edge_cara_h1', 'edge_cara_h3'} else method
    row = ma_history(x, z, m, n, seed, p['policy_updates'], out / 'policy', progress, method=native)
    return dict(history=row, representation=diag, native_action_labels_read=0, simulator_queries=0)

def label_split(budget):
    """B includes fit AND validation episodes; same prefix for every method."""
    val = np.arange(0, budget, 4)
    fit = np.setdiff1d(np.arange(budget), val)
    return (fit, val)
