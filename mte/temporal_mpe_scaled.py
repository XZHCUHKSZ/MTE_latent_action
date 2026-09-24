"""Variable-agent dimension adapter of temporal_mpe.py; same shared classes.

The delayed condition uses p[t+2], not privileged velocity/action channels.
The predictive readout directly predicts physical horizons 1,2,3; it is not
the historical recursive one-step bridge. Both timing conditions use it.
"""
from pathlib import Path
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from mte.frontends import ObservationReadout


def indices(episodes):
    return np.repeat(np.arange(episodes), 22), np.tile(np.arange(1, 23), episodes)


def frontend(x, n, variant, offset, p, out, progress):
    agents=x.shape[-1]//4; device=p.get('device','cuda')
    assert agents in (4,6,8) and x.shape[1:] == (26, 4*agents) and offset in (1,2)
    out = Path(out); out.mkdir()
    local = variant == 'entity'; lapo = variant == 'lapo'
    y = x[..., :2*agents].reshape(len(x), 26, agents, 2) if local else x
    e, t = indices(n)
    norm = np.concatenate([y[e, t], y[e, t+offset]], 0).reshape(-1, y.shape[-1])
    mu, sd = norm.mean(0), norm.std(0).clip(1e-6)
    v = torch.tensor((y-mu)/sd, device=device)
    dim = 16 if variant in ('entity', 'global16') else 64
    C.seed_all(p['seed'])
    net = (C.LAPOStateAdapter(y.shape[-1], 64) if lapo else
           C.LAOMStateAdapter(y.shape[-1], dim, 64)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(p['seed'])

    def call(ee, tt):
        if lapo:
            pred, code, _, vq, _, _ = net(torch.stack([v[ee, tt-1], v[ee, tt], v[ee, tt+offset]], 1))
            return code, (pred-v[ee, tt+offset]).square().mean()+vq
        a, b = v[ee, tt], v[ee, tt+offset]
        if local: a, b = a.reshape(-1, 2), b.reshape(-1, 2)
        pred, code, _ = net(a, b)
        loss = (pred-net.target(b)).square().mean()
        return code.reshape(len(ee), agents, dim) if local else code, loss

    trace = []
    for step in range(1, p['frontend_updates']+1):
        ix = rng.integers(len(e), size=p['batch'])
        _, loss = call(e[ix], t[ix]); assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2); optimizer.step()
        if not lapo: net.update_target(.001)
        if step == 1 or step % 100 == 0:
            trace.append(dict(update=step, loss=float(loss.detach())))
            progress('frontend', variant=variant, offset=offset, **trace[-1])
    net.eval(); ae, at = indices(len(x)); blocks = []
    with torch.no_grad():
        for lo in range(0, len(ae), 256):
            code, _ = call(ae[lo:lo+256], at[lo:lo+256]); blocks.append(code.cpu().numpy())
    z = np.concatenate(blocks).reshape(len(x), 22, *blocks[0].shape[1:])
    assert np.isfinite(z).all()
    torch.save(dict(state_dict=net.cpu().state_dict(), mean=mu, std=sd,
                    variant=variant, offset=offset, source_class=type(net).__name__,
                    native_action_labels_read=0, simulator_queries=0, frozen=True), out/'frontend.pt')
    np.savez_compressed(out/'codes.npz', z=z)
    return z, dict(trace=trace, offset=offset, native_dimension=z.shape[-1],
                   parameters=sum(q.numel() for q in net.parameters()))


def bridge(x, z, n, p, out, progress):
    """Same history and partner codes, only target code changed; donor=train only."""
    out = Path(out); out.mkdir()
    agents=x.shape[-1]//4; device=p.get('device','cuda')
    assert agents in (4,6,8) and x.shape[1:]==(26,4*agents)
    od,cd,zd=4*agents,2*agents,16*agents
    contexts=2**(agents-1)
    assert z.shape==(len(x),22,agents,16)
    count = len(x); hist = np.concatenate([x[:, :22], x[:, 1:23]], -1)
    codes = z.reshape(count, 22, -1)
    target = np.stack([x[:, 1+h:23+h, :cd]-x[:, 1:23, :cd] for h in (1, 2, 3)], 2)
    hm, hs = hist[:n].mean((0, 1)), hist[:n].std((0, 1)).clip(1e-6)
    zm, zs = codes[:n].mean((0, 1)), codes[:n].std((0, 1)).clip(1e-6)
    ym, ys = target[:n].mean((0, 1)), target[:n].std((0, 1)).clip(1e-6)
    ht = torch.tensor(((hist-hm)/hs).reshape(-1, 2*od), device=device)
    zt = torch.tensor(((codes-zm)/zs).reshape(-1, zd), device=device)
    yt = torch.tensor(((target-ym)/ys).reshape(-1, 3*cd), device=device)
    C.seed_all(p['seed']+11); net = ObservationReadout(od, zd, 3*cd).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(p['seed']+11)
    trace = []
    for step in range(1, p['readout_updates']+1):
        ix = rng.integers(n*22, size=p['batch'])
        loss = (net(ht[ix], zt[ix])-yt[ix]).square().mean(); assert torch.isfinite(loss)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % 100 == 0:
            trace.append(dict(update=step, loss=float(loss.detach()))); progress('readout', **trace[-1])
    net.eval()
    bank = rng.choice(n*22, min(1024, n*22), replace=False)
    donor_ids = []
    with torch.no_grad():
        for lo in range(0, len(ht), 256):
            ids = np.arange(lo, min(lo+256, len(ht)))
            dist = torch.cdist(ht[ids], ht[bank])
            dist.masked_fill_(torch.tensor(ids[:, None]//22 == bank[None]//22, device=device), float('inf'))
            donor_ids.extend(bank[dist.argmin(1).cpu().numpy()].tolist())
    donor_ids = np.array(donor_ids); assert (donor_ids < n*22).all()
    assert np.all(donor_ids//22 != np.arange(len(ht))//22)
    base = z.reshape(-1, agents, 16); ref = base[donor_ids]
    masks = np.zeros((contexts, agents), bool)
    for c in range(contexts): masks[c, 1:] = [(c >> j) & 1 for j in range(agents-1)]
    a = np.empty((count*22, 3, contexts, cd), np.float32); b = np.empty_like(a)
    with torch.no_grad():
        for ci, mask in enumerate(masks):
            plus = np.where(mask[None, :, None], ref, base); minus = plus.copy(); minus[:, 0] = ref[:, 0]
            for dest, code in ((a, plus), (b, minus)):
                for lo in range(0, len(ht), 256):
                    cc = torch.tensor((code[lo:lo+256].reshape(-1, zd)-zm)/zs, device=device)
                    inc = net(ht[lo:lo+256], cc).cpu().numpy().reshape(-1, 3, cd)*ys+ym
                    dest[lo:lo+256, :, ci] = x[:, 1:23, :cd].reshape(-1, cd)[lo:lo+256, None]+inc
    a, b = [v.reshape(count, 22, 3, contexts, cd) for v in (a, b)]
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.savez_compressed(out/'endpoints.npz', actual=a, reference=b, masks=masks, donor_flat_ids=donor_ids)
    torch.save(dict(state_dict=net.cpu().state_dict(), history_mean=hm, history_std=hs,
                    zmean=zm, zstd=zs, delta_mean=ym, delta_std=ys, horizons=[1, 2, 3],
                    readout='direct_multi_horizon', frozen=True), out/'readout.pt')
    rows = []
    for h in range(3):
        truth = x[n:, 2+h:24+h, :cd]; prediction = a[n:, :, h, 0]
        rows.append(dict(h=h+1, mse=float(np.square(prediction-truth).mean()),
                         persistence_mse=float(np.square(x[n:, 1:23, :cd]-truth).mean())))
    return a, b, masks, dict(trace=trace, heldout_factual_prediction=rows,
        donor_episodes='training only; same episode excluded', physical_horizons=[1, 2, 3],
        semantics='Predictive contrasts, not identified interventions; h1 is a timing negative control')
