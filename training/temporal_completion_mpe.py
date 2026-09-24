"""Missing temporal-MPE family members; reuse frozen scientific implementations."""
from mte.method_names import resolve_control
from pathlib import Path
import numpy as np
from closed_loop_lam_v1 import common as C
from training.representation_mpe import endpoint_view


def edge_view(x, a, b, masks, horizon):
    assert horizon in (2, 3)
    view = endpoint_view(x, a, b, masks)
    landmarks = np.broadcast_to(x[:, 1:23, None, 8:], (*a.shape[:2], 8, 8))
    aa = np.concatenate([a[:, :, horizon-1], landmarks], -1)
    bb = np.concatenate([b[:, :, horizon-1], landmarks], -1)
    view.update(cf_root=aa[:, :, 0], cf_ego_null=bb[:, :, 0],
                cf_other_null=aa[:, :, -1], cf_both_null=bb[:, :, -1],
                cf_context_root=aa, cf_context_ego_null=bb)
    # endpoint_view supplies shape placeholders only, never native labels.
    assert not view['actions'].any()
    return view


def train_original(x, a, b, masks, n, arm, p, out, progress):
    arm = resolve_control(arm)
    horizon = 2 if arm == 'edge_h2' else 3
    method = {'edge_h2': 'edge_cara', 'edge_h3': 'edge_cara',
              'edge_h3_root': 'edge_cara_root_only', 'tree': 'edge_cara_mobius_tree'}[arm]
    progress('original_family', arm=arm, horizon=horizon)
    result = C.train_representation(edge_view(x, a, b, masks, horizon), method,
        np.arange(n), p['seed'], z_dim=16, hidden=128,
        updates=p['representation_updates'], batch_size=p['batch'], device='cuda',
        training_profile='matched', requested_policy_updates=p['history_updates'])
    z = result.z.reshape(len(x), 22, -1)
    assert z.shape[-1] == 16 and np.isfinite(z).all()
    return z, dict(source='unchanged common.train_representation', method=method,
        horizon=horizon, parameters=result.parameter_count, settings=result.diagnostics,
        objective='original frozen method objective; not the route-only loss',
        checkpoint_scope='Frozen history and all encoded R/H features; original trainer returns no representation weights')
