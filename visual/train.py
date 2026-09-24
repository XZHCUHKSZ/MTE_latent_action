import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
import sys, copy, atexit, time, json
from types import SimpleNamespace
from utils.io import read, write as put, digest
from utils.atomic import atomic_json
from utils.access import guard
from training.composition import FrozenPair, restore
from training.representation_mamujoco import train_history_policy
from mte.frontends import ObservationReadout
from .history import history, OnlineHistory
from .official import load, load_augmenter
V = G = A = sys.modules[__name__]
E = SimpleNamespace(guard=guard, digest=digest, Path=Path)
W = SimpleNamespace(DATA=None)
RT = OLD = None
SEED = None
OBS = {'continuous_lam', 'lapo_state_adapter', 'laom_state_adapter'}
FAMILY = ['edge_cara', 'edge_cara_mobius_simple', 'edge_cara_mobius_graph', 'edge_cara_mobius_tree', 'edge_cara_mif', 'matched_cf_deepsets', 'random_edge_encoder', 'continuous_lam', 'lapo_state_adapter', 'laom_state_adapter']
METHODS = ['visual_laom_target', *FAMILY]
ARMS = ['anchor_only', 'anchor_duplicate'] + ['anchor_solo_' + m for m in FAMILY] + ['anchor_plus_' + m for m in FAMILY] + ['bc_batch256', 'bc_full1600', 'bc_idm_relabel']

def configure(seed, run_dir, data_dir, label_root):
    global RT, OLD, SEED
    if seed not in range(908711, 908716):
        raise ValueError('Use one of the five paper training seeds')
    RT = Path(run_dir)
    OLD = Path(label_root)
    SEED = seed
    W.DATA = Path(data_dir)
    RT.mkdir(parents=True, exist_ok=True)

def progress(stage, **kw):
    put(RT / 'progress.json', dict(stage=stage, **kw))

def source(method):
    return RT / ('pre_' + method) / 'policy/policy.pt'

def policy_paths():
    return [source(m) for m in METHODS]

def audit(out, paths, pre=True):
    a = E.guard(out, paths, pre)
    atexit.register(lambda: put(out / 'access_audit.json', a))
    return a

def frontend_path():
    return V.RT / 'frontend/model.pt'

def visual_net():
    ck = torch.load(frontend_path(), map_location='cuda', weights_only=False)
    assert ck['training_seed'] == V.SEED and ck['native_action_labels_read'] == 0 and (ck['simulator_queries'] == 0)
    net = load()['LAOM']((9, 64, 64), latent_act_dim=32, encoder_channels=(16, 32, 64), encoder_num_res_blocks=1, act_head_dim=128, obs_head_dim=128).cuda()
    net.load_state_dict(ck['state_dict'])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net

def frontend(out):
    files = sorted((W.DATA / 'rgb').rglob('*.npy'))
    a = audit(out, files)
    p = read(V.OLD / 'visual_config.json')
    rng = np.random.default_rng(V.SEED)
    data = {s: [np.load(f, allow_pickle=False) for f in files if f.parent.name == s] for s in ['train', 'dev']}
    assert len(data['train']) == 32 and len(data['dev']) == 8
    net = load()['LAOM']((9, 64, 64), latent_act_dim=32, encoder_channels=tuple(p['channels']), encoder_num_res_blocks=1, act_head_dim=128, obs_head_dim=128).cuda()
    target = copy.deepcopy(net).eval()
    for q in target.parameters():
        q.requires_grad_(False)
    aug = load_augmenter()
    opt = torch.optim.Adam(net.parameters(), lr=p['learning_rate'])
    ids = np.array([(e, t, j) for e, x in enumerate(data['train']) for t in range(len(x) - 1) for j in range(4)])

    def tensor(xs):
        return torch.from_numpy(np.stack(xs)).cuda().float() / 127.5 - 1
    for step in range(p['frontend_updates']):
        batch = ids[rng.integers(len(ids), size=p['batch_size'])]
        obs = tensor([history(data['train'][e], t, j) for e, t, j in batch])
        nxt = tensor([history(data['train'][e], t + 1, j) for e, t, j in batch])
        ks = [rng.integers(1, min(p['frontend_max_offset'], len(data['train'][e]) - 1 - t) + 1) for e, t, j in batch]
        future = tensor([history(data['train'][e], t + int(k), j) for (e, t, j), k in zip(batch, ks)])
        pred, _, _ = net(aug(obs), aug(future))
        with torch.no_grad():
            yt = target.encoder(aug(nxt))
        loss = F.mse_loss(pred, yt)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
        opt.step()
        with torch.no_grad():
            for tp, q in zip(target.parameters(), net.parameters()):
                tp.lerp_(q, p['ema_tau'])
        if step % 100 == 0:
            V.progress('frontend', update=step + 1, updates=p['frontend_updates'])
    torch.save(dict(state_dict=net.cpu().state_dict(), training_seed=V.SEED, native_action_labels_read=0, simulator_queries=0, method='compact archived LAOM visual frontend', frozen=True), out / 'model.pt')
    assert not a['violations']
    return dict(native_action_labels_read=0, simulator_queries=0, updates=p['frontend_updates'], device='cuda', selection='fixed final update')

@torch.no_grad()
def features(out):
    prev = read(V.RT / 'frontend/result.json')
    pa = read(V.RT / 'frontend/access_audit.json')
    assert prev['native_action_labels_read'] == 0 and prev['simulator_queries'] == 0 and (not pa['violations']) and pa['pretraining']
    from pathlib import Path
    assert all((Path(p).resolve().is_relative_to((W.DATA / 'rgb').resolve()) for p in pa['numeric_reads']))
    files = [f for s in ['train', 'dev'] for f in sorted((W.DATA / 'rgb' / s).glob('*.npy'))]
    a = audit(out, files + [frontend_path()])
    net = visual_net()
    fs = []
    zs = []
    for e, p in enumerate(files):
        rgb = np.load(p, allow_pickle=False)
        assert rgb.shape == (201, 4, 64, 64, 3) and rgb.dtype == np.uint8
        ff = np.zeros((201, 4, 256), np.float32)
        zz = np.zeros((200, 4, 32), np.float32)
        for j in range(4):
            xs = torch.from_numpy(np.stack([history(rgb, t, j) for t in range(201)])).cuda().float() / 127.5 - 1
            for lo in range(0, 201, 32):
                hi = min(lo + 32, 201)
                ff[lo:hi, j] = V.pooled(net, xs[lo:hi]).cpu().numpy()
                if lo < 200:
                    stop = min(hi, 200)
                    zz[lo:stop, j] = net.label(xs[lo:stop], xs[lo + 1:stop + 1]).cpu().numpy()
        fs.append(ff)
        zs.append(zz)
        V.progress('features', episode=e + 1, episodes=40)
    f = torch.from_numpy(np.stack(fs)).cuda()
    train = f[:32].reshape(-1, 256)
    mean = train.mean(0)
    cov = (train - mean).T @ (train - mean) / (len(train) - 1)
    eig, vec = torch.linalg.eigh(cov)
    basis = vec[:, -32:]
    scale = eig[-32:].clamp_min(1e-08).sqrt()
    x = ((f - mean) @ basis / scale).cpu().numpy().astype(np.float32)
    z = np.stack(zs)
    assert np.isfinite(x).all() and np.isfinite(z).all()
    np.savez_compressed(out / 'features.npz', x=x, z=z)
    torch.save(dict(mean=mean.cpu(), basis=basis.cpu(), scale=scale.cpu(), native_action_labels_read=0, training_seed=V.SEED), out / 'projection.pt')
    assert not a['violations']
    return dict(native_action_labels_read=0, simulator_queries=0, shape=list(x.shape))

@torch.no_grad()
def freeze(out):
    a = audit(out, G.policy_paths() + [V.RT / 'features/features.npz'])
    for m in METHODS:
        r = read(V.RT / ('pre_' + m) / 'result.json')
        pa = read(V.RT / ('pre_' + m) / 'access_audit.json')
        assert r['native_action_labels_read'] == 0 and r['simulator_queries'] == 0 and (not pa['violations'])
    x = np.load(V.RT / 'features/features.npz')['x'][:1, :32, 0]
    rows = {}
    for arm in ARMS:
        if arm.startswith('bc'):
            continue
        net, ck = G.pair(arm)
        assert ck['native_action_labels_read'] == 0 and ck['simulator_queries'] == 0
        inp = torch.tensor((x - ck['obs_mean']) / ck['obs_std'], device='cuda')
        full = net(inp)[0]
        h = None
        parts = []
        for t in range(32):
            v, h = net(inp[:, t:t + 1], h)
            parts.append(v)
        err = float((full - torch.cat(parts, 1)).abs().max())
        assert err < 1e-05 and torch.isfinite(full).all()
        rows[arm] = dict(sequential_error=err)
    assert not a['violations']
    return dict(all_sources_frozen=True, native_action_labels_read=0, simulator_queries=0, arms=rows)

@torch.no_grad()
def pooled(net, x):
    f = net.encoder(x).reshape(len(x), 64, 8, 8)
    return F.adaptive_avg_pool2d(f, (2, 2)).flatten(1)

def bridge(out):
    src = RT / 'features/features.npz'
    a = audit(out, [src])
    f = np.load(src)
    x = f['x'].reshape(40, 201, 128)
    z = f['z'].reshape(40, 200, 128)
    h = np.concatenate([x[:, np.maximum(np.arange(200) - 1, 0)], x[:, :200]], -1)
    y = x[:, 1:] - x[:, :-1]

    def norm(v):
        m = v[:32].reshape(-1, v.shape[-1]).mean(0)
        s = v[:32].reshape(-1, v.shape[-1]).std(0).clip(0.0001)
        return (torch.tensor((v - m) / s, device='cuda'), m, s)
    ht, hm, hs = norm(h)
    zt, zm, zs = norm(z)
    yt, ym, ys = norm(y)
    net = ObservationReadout(128, 128, 128).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    rng = np.random.default_rng(SEED)
    trh = ht[:32].reshape(-1, 256)
    trz = zt[:32].reshape(-1, 128)
    try_ = yt[:32].reshape(-1, 128)
    for step in range(600):
        ix = torch.tensor(rng.integers(6400, size=256), device='cuda')
        loss = F.mse_loss(net(trh[ix], trz[ix]), try_[ix])
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            progress('bridge', update=step + 1, updates=600, loss=float(loss.detach()))
    net.eval()
    with torch.no_grad():
        dh = ht[32:].reshape(-1, 256)
        dz = zt[32:].reshape(-1, 4, 32)
        dy = yt[32:].reshape(-1, 128)
        ix = torch.arange(len(dh), device='cuda').roll(137)
        pred = net(dh, dz.flatten(1))
        target_shuf = dz.clone()
        target_shuf[:, 0] = dz[ix, 0]
        partner_shuf = dz.clone()
        partner_shuf[:, 1:] = dz[ix, 1:]
        diagnostics = dict(dev_increment_mse=float(F.mse_loss(pred, dy)), shuffled_target_mse=float(F.mse_loss(net(dh, target_shuf.flatten(1)), dy)), shuffled_partners_mse=float(F.mse_loss(net(dh, partner_shuf.flatten(1)), dy)), persistence_mse=float(torch.tensor(y[32:] / ys, device='cuda').square().mean()))
        bank = rng.choice(6400, 1024, replace=False)
        bank_ep = bank // 200
        bank_z = trz[bank].reshape(-1, 4, 32)
        masks = np.zeros((8, 4), bool)
        for c in range(8):
            masks[c, 1:] = [c >> j & 1 for j in range(3)]
        aa = np.zeros((8000, 8, 128), np.float32)
        bb = np.zeros_like(aa)
        donor_ids = []
        flat_h = ht.reshape(-1, 256)
        flat_z = zt.reshape(-1, 4, 32)
        for lo in range(0, 8000, 128):
            hi = min(lo + 128, 8000)
            hh = flat_h[lo:hi]
            zz = flat_z[lo:hi]
            dist = torch.cdist(hh, trh[bank])
            qep = np.arange(lo, hi) // 200
            dist.masked_fill_(torch.tensor(qep[:, None] == bank_ep[None], device='cuda'), float('inf'))
            assert torch.isfinite(dist.min(1).values).all()
            donor = dist.argmin(1)
            rr = bank_z[donor]
            donor_ids.extend(bank[donor.cpu().numpy()].tolist())
            for ci, mask in enumerate(masks):
                za = zz.clone()
                za[:, mask] = rr[:, mask]
                zb = za.clone()
                zb[:, 0] = rr[:, 0]
                assert torch.equal(za[:, 1:], zb[:, 1:])
                aa[lo:hi, ci] = net(hh, za.flatten(1)).cpu().numpy()
                bb[lo:hi, ci] = net(hh, zb.flatten(1)).cpu().numpy()
            if lo % 1024 == 0:
                progress('matched_endpoints', completed=hi, total=8000)
        assert torch.equal(net(dh[:32], dz[:32].flatten(1)), net(dh[:32], dz[:32].flatten(1)))
    assert np.isfinite(aa).all() and np.isfinite(bb).all() and (np.abs(aa - bb).mean() > 1e-09)
    np.savez_compressed(out / 'endpoints.npz', actual=aa.reshape(40, 200, 8, 128), reference=bb.reshape(40, 200, 8, 128), masks=masks, donor_indices=np.asarray(donor_ids).reshape(40, 200))
    torch.save(dict(state_dict=net.cpu().state_dict(), history_mean=hm, history_std=hs, code_mean=zm, code_std=zs, delta_mean=ym, delta_std=ys), out / 'readout.pt')
    assert not a['violations']
    return dict(diagnostics=diagnostics, mean_absolute_edge=float(np.abs(aa - bb).mean()), native_action_labels_read=0, simulator_queries=0, horizon=1, endpoint='predicted standardized common visual-feature increments; not simulator outcomes', matching_checks='same history/partners; target-only replacement; training donors exclude same episode', unresolved='View-associated latent replacement is not established as isolated native-agent action replacement')

def pretrain_base(out, method):
    paths = [RT / 'features/features.npz']
    if method != 'visual_laom_target':
        paths.append(RT / 'bridge/endpoints.npz')
    a = audit(out, paths)
    f = np.load(paths[0])
    x = f['x']
    valid = np.ones((40, 200), bool)
    if method == 'visual_laom_target':
        z = f['z'][:, :, 0]
        diag = dict(source='frozen official visual LAOM labels for target camera', latent_dim=32)
    else:
        ep = np.load(paths[1])
        aa, bb, masks = (ep['actual'], ep['reference'], ep['masks'])
        view = dict(obs=x.reshape(40, 201, 128), actions=np.zeros((40, 200, 2), np.float32), valid_step_mask=valid, cf_root=aa[:, :, 0], cf_ego_null=bb[:, :, 0], cf_other_null=aa[:, :, -1], cf_both_null=bb[:, :, -1], cf_context_root=aa, cf_context_ego_null=bb, cf_context_masks=masks)
        progress('original_representation', method=method, updates=256)
        result = C.train_representation(view, method, np.arange(32), SEED, z_dim=16, hidden=128, updates=256, batch_size=128, device='cuda', training_profile='matched', requested_policy_updates=256)
        z = result.z.reshape(40, 200, -1)
        diag = dict(result.diagnostics, parameters=result.parameter_count, source_function='closed_loop_lam_v1.common.train_representation', endpoint_evidence='learned visual endpoints; exact Mobius refers to algebra, not true simulator effects', action_field='shape-only zeros required by legacy API; no true actions')

    def note(phase, **kw):
        progress(phase, method=method, **kw)
    row = train_history_policy(x[:, :, 0], z, valid, 32, SEED, 256, out / 'policy', note, method='edge_cara' if method == 'visual_laom_target' else method)
    assert not a['violations']
    return dict(method=method, representation=diag, policy=row, native_action_labels_read=0, simulator_queries=0, deploy_input='target camera feature history only; no online partner observations', selection='original GRU trainer uses only unlabelled development latent MSE')

def put(p, d):
    V.atomic_json(p, d)

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def pair(arm):
    if arm.startswith('anchor_solo_'):
        net, ck = restore(source(arm[len('anchor_solo_'):]))
        return (FrozenPair(net).cuda().eval(), ck)
    net, ck = restore(source('visual_laom_target'))
    aux = None
    if arm.startswith('anchor_plus_'):
        aux, ac = restore(source(arm[len('anchor_plus_'):]))
        assert np.array_equal(ck['obs_mean'], ac['obs_mean']) and np.array_equal(ck['obs_std'], ac['obs_std'])
    return (FrozenPair(net, aux, arm == 'anchor_duplicate').cuda().eval(), ck)

def rich(out):
    paths = [V.RT / 'features/features.npz', V.RT / 'bridge/endpoints.npz', V.RT / 'bridge/readout.pt']
    audit = V.audit(out, paths)
    f = np.load(paths[0])
    x = f['x'].reshape(40, 201, 128)
    z = f['z'].reshape(40, 200, 128)
    ep = np.load(paths[1])
    ck = torch.load(paths[2], map_location='cpu', weights_only=False)
    h = np.concatenate([x[:, np.maximum(np.arange(200) - 1, 0)], x[:, :200]], -1)
    ht = torch.tensor((h - ck['history_mean']) / ck['history_std'], device='cuda')
    zt = torch.tensor((z - ck['code_mean']) / ck['code_std'], device='cuda')
    donor = ep['donor_indices']
    assert np.all(donor < 6400) and np.all(donor // 200 != np.arange(40)[:, None])
    masks = ep['masks']
    assert not masks[:, 0].any()
    aa = np.empty((40, 200, 3, 8, 4, 32), np.float32)
    bb = np.empty_like(aa)
    aa[:, :, 0] = ep['actual'].reshape(40, 200, 8, 4, 32)
    bb[:, :, 0] = ep['reference'].reshape(40, 200, 8, 4, 32)
    rows = []
    rng = np.random.default_rng(V.SEED)
    for horizon in (2, 3):
        steps = 201 - horizon
        y = x[:, horizon:] - x[:, :steps]
        ym = y[:32].reshape(-1, 128).mean(0)
        ys = y[:32].reshape(-1, 128).std(0).clip(0.0001)
        yt = torch.tensor((y - ym) / ys, device='cuda')
        hh = ht[:32, :steps].reshape(-1, 256)
        zz = zt[:32, :steps].reshape(-1, 128)
        yy = yt[:32].reshape(-1, 128)
        C.seed_all(V.SEED + horizon)
        net = ObservationReadout(128, 128, 128).cuda()
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        for step in range(600):
            ix = torch.tensor(rng.integers(len(hh), size=256), device='cuda')
            loss = F.mse_loss(net(hh[ix], zz[ix]), yy[ix])
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                progress('multihorizon_readout', horizon=horizon, update=step + 1, updates=600)
        net.eval()
        with torch.no_grad():
            pred = net(ht[32:, :steps].reshape(-1, 256), zt[32:, :steps].reshape(-1, 128))
            rows.append(dict(horizon=horizon, valid_fit_transitions=32 * steps, dev_factual_mse=float(F.mse_loss(pred, yt[32:].reshape(-1, 128)))))
            fh = ht.reshape(-1, 256)
            fz = zt.reshape(-1, 4, 32)
            dr = fz[torch.tensor(donor.reshape(-1), device='cuda')]
            ra = np.empty((8000, 8, 128), np.float32)
            rb = np.empty_like(ra)
            for lo in range(0, 8000, 128):
                hi = min(lo + 128, 8000)
                for ci, mask in enumerate(masks):
                    za = fz[lo:hi].clone()
                    za[:, mask] = dr[lo:hi, mask]
                    zb = za.clone()
                    zb[:, 0] = dr[lo:hi, 0]
                    assert torch.equal(za[:, 1:], zb[:, 1:])
                    ra[lo:hi, ci] = net(fh[lo:hi], za.flatten(1)).cpu().numpy()
                    rb[lo:hi, ci] = net(fh[lo:hi], zb.flatten(1)).cpu().numpy()
            aa[:, :, horizon - 1] = ra.reshape(40, 200, 8, 4, 32)
            bb[:, :, horizon - 1] = rb.reshape(40, 200, 8, 4, 32)
        torch.save(dict(state_dict=net.cpu().state_dict(), delta_mean=ym, delta_std=ys, horizon=horizon), out / f'readout_h{horizon}.pt')
    assert np.isfinite(aa).all() and np.isfinite(bb).all()
    assert np.array_equal(aa[:, :, 0].reshape(40, 200, 8, 128), ep['actual'])
    np.savez_compressed(out / 'endpoints.npz', actual=aa, reference=bb, masks=masks, adjacency=np.ones((4, 4), np.float32), horizons=np.array([1, 2, 3]))
    assert not audit['violations']
    return dict(native_action_labels_read=0, simulator_queries=0, shape=list(aa.shape), diagnostics=rows, entity_definition='four separately encoded camera views,32 shared-PCA coordinates each; not physical entity segmentation', adjacency='fixed complete view graph, no simulator topology or state read', horizon_definition='direct predicted standardized visual increments from same current history/code to t+h; no simulator rollout or fixed continuation claim', tails='readout fits only observed valid t+h; MIF inputs at last2 timestamps are model extrapolations, not copied future frames')

def pretrain(out, m):
    paths = [V.RT / 'features/features.npz']
    if m not in OBS:
        paths.append(V.RT / 'bridge/endpoints.npz')
    if m == 'edge_cara_mif':
        paths.append(RT / 'rich/endpoints.npz')
    audit = V.audit(out, paths)
    f = np.load(paths[0])
    x = f['x']
    valid = np.ones((40, 200), bool)
    flat = x.reshape(40, 201, 128)
    if m in OBS:
        aa = flat[:, 1:]
        bb = np.zeros_like(aa)
        view = dict(cf_root=aa, cf_ego_null=bb, cf_other_null=aa, cf_both_null=bb)
    else:
        ep = np.load(paths[1])
        aa, bb, masks = (ep['actual'], ep['reference'], ep['masks'])
        view = dict(cf_root=aa[:, :, 0], cf_ego_null=bb[:, :, 0], cf_other_null=aa[:, :, -1], cf_both_null=bb[:, :, -1], cf_context_root=aa, cf_context_ego_null=bb, cf_context_masks=masks)
    view.update(obs=flat, actions=np.zeros((40, 200, 2), np.float32), valid_step_mask=valid)
    if m == 'edge_cara_mif':
        r = np.load(paths[2])
        view.update(cf_horizon_entity_root=r['actual'], cf_horizon_entity_ego_null=r['reference'], mif_entity_adjacency=r['adjacency'], effect_horizons=r['horizons'])
    progress('original_representation', method=m, updates=256)
    result = C.train_representation(view, m, np.arange(32), V.SEED, z_dim=16, hidden=128, updates=256, batch_size=128, device='cuda', training_profile='matched', requested_policy_updates=256)
    z = result.z.reshape(40, 200, -1)
    assert np.isfinite(z).all()

    def note(stage, **kw):
        progress(stage, method=m, **kw)
    row = V.train_history_policy(x[:, :, 0], z, valid, 32, V.SEED, 256, out / 'policy', note, method=m)
    assert not audit['violations']
    return dict(method=m, native_action_labels_read=0, simulator_queries=0, policy=row, representation=dict(result.diagnostics, parameters=result.parameter_count, latent_dim=z.shape[-1]), source='unchanged common.train_representation and original history trainer; visual feature adapter', baseline_scope='feature-space adaptation, not full official pixel LAPO/LAOM replication' if m in OBS else 'learned visual matching')

def idm(out):
    assert read(RT / 'freeze/result.json')['all_sources_frozen']
    files = [V.RT / 'features/features.npz'] + [V.OLD / f'labels/train/{i:04d}.npy' for i in range(8)]
    audit = A.watch(out, files, 'budgeted_supervised_IDM_and_pseudolabel_policy')
    x = np.load(files[0])['x'][:32, :, 0]
    y = torch.tensor(np.stack([np.load(p) for p in files[1:]]), device='cuda')
    mean = x[:, :200].reshape(-1, 32).mean(0)
    scale = x[:, :200].reshape(-1, 32).std(0).clip(0.0001)
    xx = torch.tensor((x - mean) / scale, device='cuda')
    C.seed_all(V.SEED)
    net = C.InverseDynamics(32, 2).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    rng = np.random.default_rng(V.SEED)
    cur = xx[:8, :200].reshape(-1, 32)
    nxt = xx[:8, 1:].reshape(-1, 32)
    for step in range(600):
        ix = torch.tensor(rng.integers(1600, size=256), device='cuda')
        loss = F.mse_loss(net(cur[ix], nxt[ix]), y.reshape(-1, 2)[ix])
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pseudo = net(xx[:, :200], xx[:, 1:]).detach()
    torch.save(dict(state_dict=net.cpu().state_dict(), mean=mean, scale=scale), out / 'idm.pt')
    C.seed_all(V.SEED)
    pol = C.RecurrentPolicy(32, 2).cuda()
    opt = torch.optim.Adam(pol.parameters(), lr=0.001)
    for step in range(600):
        ids = rng.integers(32, size=8)
        pred = pol(xx[ids, :200])[0]
        loss = F.mse_loss(pred, pseudo[ids])
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            progress('idm_pseudo_policy', update=step + 1, updates=600)
    pol.eval()
    with torch.no_grad():
        pred = pol(xx[:8, :200])[0]
    torch.save(dict(state_dict=pol.cpu().state_dict(), mean=mean, scale=scale, arm='bc_idm_relabel', latent_dim=32), out / 'decoder.pt')
    assert not audit['violations']
    return dict(arm='bc_idm_relabel', native_action_labels_read=1600, partner_labels_read=0, simulator_queries=0, total_label_presentations=153600, trainable_parameters=sum((p.numel() for p in pol.parameters())), idm_trainable_parameters=sum((p.numel() for p in net.parameters())), pseudo_policy_updates=600, full_fit_action_mse=float(F.mse_loss(pred, y)), scope='supervised IDM baseline; not observation-only pretraining; same distinct B8 labels')

def watch(out, paths, phase):
    a = V.audit(out, paths)
    a['phase'] = phase
    a['pretraining'] = phase == 'freeze_verification'
    return a

def ground(out, arm):
    assert read(RT / 'freeze/result.json')['all_sources_frozen']
    labels_paths = [V.OLD / f'labels/train/{i:04d}.npy' for i in range(8)]
    paths = [V.RT / 'features/features.npz'] + labels_paths
    if arm.startswith('anchor'):
        paths += policy_paths()
    audit = watch(out, paths, 'budgeted_action_grounding')
    x = np.load(paths[0])['x'][:, :, 0]
    y = torch.tensor(np.stack([np.load(p) for p in labels_paths]), device='cuda')
    assert y.shape == (8, 200, 2)
    is_bc = arm.startswith('bc')
    if is_bc:
        mean = x[:32, :200].reshape(-1, 32).mean(0)
        scale = x[:32, :200].reshape(-1, 32).std(0).clip(0.0001)
        inputs = torch.tensor((x[:8, :200] - mean) / scale, device='cuda')
        C.seed_all(V.SEED)
        model = C.RecurrentPolicy(32, 2).cuda()
        frozen_parameters = 0
    else:
        net, ck = pair(arm)
        inputs = torch.tensor((x[:8, :200] - ck['obs_mean']) / ck['obs_std'], device='cuda')
        with torch.no_grad():
            values = net(inputs)[0]
            values = values.reshape(-1, values.shape[-1])
        mean = values.mean(0)
        scale = values.std(0).clamp_min(0.0001)
        values = (values - mean) / scale
        C.seed_all(V.SEED)
        model = C.ActionDecoder(values.shape[-1], 2, hidden=128).cuda()
        frozen_parameters = sum((p.numel() for p in net.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(V.SEED)
    curve = []
    for step in range(600):
        ix = torch.tensor(rng.integers(1600, size=256), device='cuda')
        if is_bc:
            pred = model(inputs)[0].reshape(-1, 2)
            loss = F.mse_loss(pred, y.reshape(-1, 2)) if arm == 'bc_full1600' else F.mse_loss(pred[ix], y.reshape(-1, 2)[ix])
        else:
            loss = F.mse_loss(model(values[ix]), y.reshape(-1, 2)[ix])
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 599:
            curve.append(dict(update=step + 1, sampled_loss=float(loss.detach())))
            progress('grounding', arm=arm, update=step + 1, updates=600)
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)[0] if is_bc else model(values).reshape(8, 200, 2)
        train_mse = float(F.mse_loss(prediction, y))
        output_std = float(prediction.reshape(-1, 2).std(0).mean())
    torch.save(dict(state_dict=model.cpu().state_dict(), mean=mean, scale=scale, arm=arm, latent_dim=32 if is_bc else values.shape[-1]), out / 'decoder.pt')
    np.save(out / 'fit_predictions.npy', prediction.cpu().numpy())
    put(out / 'curve.json', curve)
    assert not audit['violations']
    return dict(arm=arm, budget_trajectories=8, native_action_labels_read=1600, partner_labels_read=0, simulator_queries=0, updates=600, supervised_vectors_per_update=1600 if arm == 'bc_full1600' else 256, total_label_presentations=600 * (1600 if arm == 'bc_full1600' else 256), trainable_parameters=sum((p.numel() for p in model.parameters())), frozen_policy_parameters=frozen_parameters, full_fit_action_mse=train_mse, fit_output_std=output_std, selection='fixed final update; no action-dev or return selection')

def controller(arm):
    d = torch.load(RT / ('ground_' + arm) / 'decoder.pt', map_location='cuda', weights_only=False)
    if arm.startswith('bc'):
        net = C.RecurrentPolicy(32, 2).cuda()
        net.load_state_dict(d['state_dict'])
        net.eval()

        def predict(features, h=None):
            return net((features - torch.as_tensor(d['mean'], device='cuda')) / torch.as_tensor(d['scale'], device='cuda'), h)
    else:
        net, ck = pair(arm)
        decoder = C.ActionDecoder(d['latent_dim'], 2, hidden=128).cuda()
        decoder.load_state_dict(d['state_dict'])
        decoder.eval()

        def predict(features, h=None):
            z, h = net((features - torch.as_tensor(ck['obs_mean'], device='cuda')) / torch.as_tensor(ck['obs_std'], device='cuda'), h)
            return (decoder((z - d['mean']) / d['scale']), h)
    return predict
