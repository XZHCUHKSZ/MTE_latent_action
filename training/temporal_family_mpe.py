"""Small timing-repair experiment using unchanged family classes and route loss."""
from pathlib import Path
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from training.route_representation import factory, forward
from training.representation_mpe import endpoint_view
from mte.structured_controls import complement_indices


def train(x, a, b, masks, n, arm, p, out, progress):
    model = arm.split('_')[0]
    variant = 'raw_coordinates' if arm.endswith('_raw') else 'mobius'
    if arm.endswith('_complement'): b = np.take(b, complement_indices(masks), axis=3)
    if model == 'mif':
        view = endpoint_view(x, a, b, masks)
        av, bv = view['cf_horizon_entity_root'], view['cf_horizon_entity_ego_null']
        y = np.concatenate([av, av-bv], -1); adj = view['mif_entity_adjacency']
    else:
        y = a[:, :, 2]-b[:, :, 2]; adj = None
    mu, sd = C.fit_mean_std(y[:n].reshape(-1, y.shape[-1]))
    flat = ((y-mu)/sd).reshape(-1, *y.shape[2:]); C.seed_all(p['seed'])
    net = factory(model, variant, y.shape[-1], masks, adj, 'mpe')
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(p['seed']+17)
    trace = []
    for step in range(1, p['representation_updates']+1):
        ids = rng.integers(n*22, size=p['representation_batch'])
        v = torch.tensor(flat[ids], device='cuda'); z, reco = forward(net, v, model)
        loss = (reco-v).square().mean()+C._variance_floor(z)+.02*C._offdiag_cov(z)
        assert torch.isfinite(loss)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2); opt.step()
        if step == 1 or step % 100 == 0:
            trace.append(dict(update=step, loss=float(loss.detach()))); progress('family', arm=arm, **trace[-1])
    net.eval(); encoded = []
    with torch.no_grad():
        for lo in range(0, len(flat), 64):
            encoded.append(forward(net, torch.tensor(flat[lo:lo+64], device='cuda'), model)[0].cpu().numpy())
    z = np.concatenate(encoded).reshape(len(x), 22, 16); assert np.isfinite(z).all()
    torch.save(dict(state_dict={k: v.cpu() for k, v in net.state_dict().items()},
                    mean=mu, std=sd, masks=masks, adjacency=adj, model=model, variant=variant,
                    objective='existing route full visibility reconstruction plus variance/covariance',
                    frozen=True), Path(out)/'representation.pt')
    return z, dict(trace=trace, parameters=sum(v.numel() for v in net.parameters()),
                   objective='route full visibility, not formal masked MIF objective', variant=variant)
