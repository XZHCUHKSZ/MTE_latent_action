"""Inference adapters for archived observation-only checkpoints, not new models."""
from pathlib import Path
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from mte.frontends import local_observations, ObservationReadout
from training.composition import restore


def freeze(net):
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class Frontend:
    def __init__(self, path, env):
        self.env = env
        self.ck = torch.load(path, map_location='cpu', weights_only=False)
        if not self.ck.get('frozen'):
            raise ValueError('Frontend must already be frozen')
        ck = self.ck
        self.variant = ck['variant']
        dim = len(ck['mean'])
        sd = ck['state_dict']
        if self.variant == 'lapo_joint':
            self.net = C.LAPOStateAdapter(dim, sd['idm.0.weight'].shape[0])
        else:
            hidden = sd['encoder.0.weight'].shape[0]
            # Native configured widths; no extra projection or truncation.
            zdim = 16 if self.variant == 'laom_entity_k3' else 64
            self.net = C.LAOMStateAdapter(dim, zdim, hidden)
        self.net.load_state_dict(sd)
        freeze(self.net)

    @torch.inference_mode()
    def encode(self, previous, current, future):
        local = self.variant == 'laom_entity_k3'
        arrays = [local_observations(a, self.env) if local else a for a in (previous, current, future)]
        q = [torch.tensor((a-self.ck['mean'])/self.ck['std'], dtype=torch.float32) for a in arrays]
        if self.variant == 'lapo_joint':
            z = self.net(torch.stack(q, 1))[1]
        else:
            cur, nxt = q[1:]
            if local:
                cur, nxt = cur.reshape(-1, cur.shape[-1]), nxt.reshape(-1, nxt.shape[-1])
            z = self.net(cur, nxt)[1]
            if local:
                z = z.reshape(len(current), 4, -1)
        return z.numpy()


class Predictor:
    def __init__(self, path, env, training_observations):
        self.ck = torch.load(path, map_location='cpu', weights_only=False)
        if not self.ck.get('frozen'):
            raise ValueError('Readout must be frozen')
        self.env = env
        d = 16 if env == 'mpe' else 105
        self.out_dim = 8 if env == 'mpe' else 105
        self.net = ObservationReadout(d, 64, self.out_dim)
        self.net.load_state_dict(self.ck['state_dict']); freeze(self.net)
        self.constant = np.zeros(2*d, bool)
        self.fixed = np.zeros(2*d, np.float32)
        if env == 'mamujoco':
            with np.load(training_observations, allow_pickle=False) as f:
                x, valid = f['observations'], f['valid_mask']
            h = np.concatenate([x[:, np.maximum(np.arange(valid.shape[1])-1, 0)], x[:, :-1]], -1)[valid]
            self.constant = np.ptp(h, axis=0) == 0
            self.fixed = (h[0]-self.ck['history_mean'])/self.ck['history_std']

    @torch.inference_mode()
    def rollout(self, previous, current, first_codes, continuation):
        ck = self.ck
        prev, cur, values = previous.copy(), current.copy(), []
        for i in range(3):
            z = first_codes if i == 0 else continuation[:, i-1]
            hh = (np.concatenate([prev, cur], -1)-ck['history_mean'])/ck['history_std']
            hh[:, self.constant] = self.fixed[self.constant]
            zz = (z.reshape(len(z), -1)-ck['zmean'])/ck['zstd']
            inc = self.net(torch.tensor(hh, dtype=torch.float32), torch.tensor(zz, dtype=torch.float32)).numpy()
            nxt = cur.copy(); nxt[:, :self.out_dim] += inc*ck['delta_std']+ck['delta_mean']
            values.append(nxt[:, :self.out_dim].copy()); prev, cur = cur, nxt
        result = np.stack(values, 1)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite frozen prediction')
        return result


@torch.inference_mode()
def history_features(checkpoint, trajectories, episode_index, times):
    net, ck = restore(checkpoint); freeze(net)
    x = torch.tensor((trajectories-ck['obs_mean'])/ck['obs_std'], dtype=torch.float32)
    all_z = net(x)[0].numpy()
    # Check causality with a suffix perturbation; future samples cannot change prefix output.
    cut = int(min(times))+1
    perturbed = x.clone(); perturbed[:, cut:] += 7.0
    error = float(np.abs(net(perturbed)[0].numpy()[:, :cut]-all_z[:, :cut]).max())
    if error > 1e-6:
        raise ValueError('History policy uses future observations')
    return all_z[episode_index, times], error


@torch.inference_mode()
def route_representation(checkpoint, actual, reference, current, env):
    """Load the exact existing route-family class/weights; evaluate fresh tables."""
    from closed_loop_lam_v1.unified_models import MobiusSimple, MobiusGraphEdgeCARA, MIFCARAIncidenceFlow
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model, masks = ck['model'], torch.tensor(ck['masks'])
    if ck['variant'] != 'mobius':
        raise ValueError('Pilot manifest expects original matched coordinate checkpoints')
    if model == 'mif':
        if env == 'mamujoco':
            from mte.entity_view import entities
            aa, bb = entities(actual), entities(reference)
        else:
            def view(a):
                full = np.concatenate([a, np.broadcast_to(current[:, None, None, 8:], a.shape)], -1)
                v = np.zeros((*a.shape[:-1], 8, 4), np.float32)
                v[..., :2] = full.reshape(*a.shape[:-1], 8, 2)
                v[..., :4, 2] = 1; v[..., 4:, 3] = 1
                return v
            aa, bb = view(actual), view(reference)
        y = np.concatenate([aa, aa-bb], -1)
        net = MIFCARAIncidenceFlow(y.shape[-1], masks, torch.tensor(ck['adjacency']), 3, 16, 128 if env == 'mpe' else 108)
    else:
        y = actual[:, 2]-reference[:, 2]
        cls = MobiusSimple if model == 'simple' else MobiusGraphEdgeCARA
        net = cls(y.shape[-1], masks, 16, 128)
    net.load_state_dict(ck['state_dict']); freeze(net)
    q = torch.tensor((y-ck['mean'])/ck['std'], dtype=torch.float32)
    chunks = []
    for batch in q.split(16):
        if model == 'mif':
            z = net(batch, torch.ones(*batch.shape[:-1], 1, dtype=torch.bool))[0]
        else:
            z = net(batch)[0]
        chunks.append(z.numpy())
    return np.concatenate(chunks)
