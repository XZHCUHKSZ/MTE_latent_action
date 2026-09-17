import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
from closed_loop_lam_v1.common import RecurrentPolicy,seed_all
from utils.atomic import atomic_json

# Final implementation source: mte_observation_only_2026_09_09/g2a_train.py:61

def train_history_policy(positions, z, train_count, seed, updates, folder, progress):
    """Stage-isolated orchestration, reuses original GRU; never receives actions."""
    seed_all(seed)
    folder.mkdir()
    e, t, d = z.shape
    assert t == 22 and positions.shape == (e, 26, 16)
    # Feed observed x_0,...,x_22. Predict targets only at t=1,...,22.
    obs = positions[:, :23]
    om = obs[:train_count].mean((0, 1)); osd = obs[:train_count].std((0, 1)).clip(1e-6)
    zm = z[:train_count].mean((0, 1)); zs = z[:train_count].std((0, 1)).clip(1e-6)
    ot = torch.as_tensor((obs-om)/osd, device='cuda')
    zt = torch.as_tensor((z-zm)/zs, device='cuda')
    policy = RecurrentPolicy(16, d).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed+101)
    best, selected, best_state, history = float('inf'), 0, None, []
    for step in range(1, updates+1):
        ix = rng.choice(train_count, min(16, train_count), replace=False)
        prediction, _ = policy(ot[ix])
        loss = (prediction[:, 1:]-zt[ix]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite history-policy loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == updates:
            with torch.no_grad():
                dev, _ = policy(ot[train_count:])
                value = (dev[:, 1:]-zt[train_count:]).square().mean().item()
            history.append({'update': step, 'train_mse': loss.item(), 'dev_mse': value})
            if value < best:
                best, selected = value, step
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            progress('latent_policy', update=step, updates=updates, dev_mse=value)
    policy.load_state_dict(best_state)
    policy.eval()
    with torch.no_grad():
        deployment = torch.cat([policy(batch)[0][:, 1:] for batch in ot.split(128)]).cpu().numpy()
    assert np.isfinite(deployment).all()
    torch.save({'state_dict': best_state, 'obs_mean': om, 'obs_std': osd,
                'z_mean': zm, 'z_std': zs, 'z_dim': d, 'hidden': 128,
                'source_class': 'closed_loop_lam_v1.common.RecurrentPolicy',
                'history': 'x_0 through x_t only; supervised latent targets t=1..22',
                'selected_update': selected, 'native_action_labels_read': 0,
                'frozen_before_grounding': True}, folder/'policy.pt')
    np.savez_compressed(folder/'latents.npz', representation_z=z, deployment_z=deployment)
    atomic_json(folder/'history.json', history)
    return {'dev_standardized_latent_mse': best, 'selected_update': selected,
            'policy_parameters': sum(p.numel() for p in policy.parameters()), 'z_dim': d,
            'checkpoint': str(folder/'policy.pt'),
            'warning': 'Latent MSE scales and dimensions differ across methods; not a ranking of control quality'}
