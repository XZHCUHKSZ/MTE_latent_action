"""Disk-backed data orchestration; unchanged qualified family classes/loss/order.

Source: training/temporal_family_mpe_scaled.py. Only table materialization and
elementwise normalization are chunked. Mean/std use the exact original reducer.
"""
from pathlib import Path
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from training.route_representation import forward
from closed_loop_lam_v1.unified_models import MobiusSimple,MobiusGraphEdgeCARA,MIFCARAIncidenceFlow
from training.representation_mpe_scaled import endpoint_view
from mte.structured_controls import complement_indices



def prepare_disk_table(x, a, b, masks, n, model, out, chunk_episodes=4):
    """Preserve C-order float32 values and the original full training-row reducer."""
    from numpy.lib.format import open_memmap
    assert 0 < n < len(x) and chunk_episodes > 0
    shape = ((len(x), 22, 3, len(masks), x.shape[-1]//2, 8)
             if model == 'mif' else (len(x), 22, len(masks), a.shape[-1]))
    out.mkdir(parents=True, exist_ok=True)
    raw_path, flat_path = out/'table_raw.npy', out/'table_normalized.npy'
    if raw_path.exists() or flat_path.exists():
        raise FileExistsError('Preserve existing partial data; never overwrite')
    raw = open_memmap(raw_path, mode='w+', dtype=np.float32, shape=shape)
    adj = None
    for lo in range(0,len(x),chunk_episodes):
        hi=min(lo+chunk_episodes,len(x))
        if model=='mif':
            view=endpoint_view(x[lo:hi],a[lo:hi],b[lo:hi],masks)
            av,bv=view['cf_horizon_entity_root'],view['cf_horizon_entity_ego_null']
            raw[lo:hi]=np.concatenate([av,av-bv],-1)
            adj=view['mif_entity_adjacency']
            del view,av,bv
        else:raw[lo:hi]=a[lo:hi,:,2]-b[lo:hi,:,2]
    raw.flush()
    # Deliberately do not change reduction dtype/order or replace with online stats.
    mu,sd=C.fit_mean_std(raw[:n].reshape(-1,shape[-1]))
    flat=open_memmap(flat_path,mode='w+',dtype=np.float32,shape=(len(x)*22,*shape[2:]))
    for lo in range(0,len(x),chunk_episodes):
        hi=min(lo+chunk_episodes,len(x))
        flat[lo*22:hi*22]=((raw[lo:hi]-mu)/sd).reshape(-1,*shape[2:])
    flat.flush();raw._mmap.close()
    return flat,adj,mu,sd


def train(x, a, b, masks, n, arm, p, out, progress):
    model = arm.split('_')[0]
    variant = 'raw_coordinates' if arm.endswith('_raw') else 'mobius'
    if arm.endswith('_complement'): b = np.take(b, complement_indices(masks), axis=3)
    flat, adj, mu, sd = prepare_disk_table(x, a, b, masks, n, model, Path(out))
    C.seed_all(p['seed'])
    assert variant=='mobius', 'Qualification currently covers accepted coordinate route only'
    device=p.get('device','cuda');cm=torch.as_tensor(masks,device=device)
    if model=='graph':net=MobiusGraphEdgeCARA(flat.shape[-1],cm,16,128).to(device)
    elif model=='simple':net=MobiusSimple(flat.shape[-1],cm,16,128).to(device)
    elif model=='mif':net=MIFCARAIncidenceFlow(flat.shape[-1],cm,torch.as_tensor(adj,device=device),3,16,128).to(device)
    else:raise ValueError(model)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(p['seed']+17)
    trace = []
    for step in range(1, p['representation_updates']+1):
        ids = rng.integers(n*22, size=p['representation_batch'])
        v = torch.tensor(flat[ids], device=device); z, reco = forward(net, v, model)
        loss = (reco-v).square().mean()+C._variance_floor(z)+.02*C._offdiag_cov(z)
        assert torch.isfinite(loss)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2); opt.step()
        if step == 1 or step % 100 == 0:
            trace.append(dict(update=step, loss=float(loss.detach()))); progress('family', arm=arm, **trace[-1])
    net.eval(); encoded = []
    with torch.no_grad():
        for lo in range(0, len(flat), 4):
            encoded.append(forward(net, torch.tensor(flat[lo:lo+4], device=device), model)[0].cpu().numpy())
    z = np.concatenate(encoded).reshape(len(x), 22, 16); assert np.isfinite(z).all()
    torch.save(dict(state_dict={k: v.cpu() for k, v in net.state_dict().items()},
                    mean=mu, std=sd, masks=masks, adjacency=adj, model=model, variant=variant,
                    objective='existing route full visibility reconstruction plus variance/covariance',
                    frozen=True), Path(out)/'representation.pt')
    return z, dict(trace=trace, parameters=sum(v.numel() for v in net.parameters()),
                   objective='route full visibility, not formal masked MIF objective', variant=variant)
