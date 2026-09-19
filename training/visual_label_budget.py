"""Budget-only adapter for the accepted RGB grounding routines.

Scientific model classes, objectives and optimizer order come from visual.train.
Only the number of labelled training episodes changes. B8 parity is a required
all-five-seed gate before formal low-budget results can be dispatched.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from visual import train as V
from closed_loop_lam_v1 import common as C
from utils.atomic import atomic_json
from utils.io import read

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[1]
ARCHIVE = WORKSPACE / 'visual_multiagent_2026_09_10/seed5_v1/runtime'
ASSETS = WORKSPACE / 'visual_multiagent_2026_09_10/pixel_pipeline_v1'
LABELS = ASSETS / 'runtime/pilot/validate/target_labels/train'
BUDGET = 8
UPDATES = 600
SOURCE = None

def configure(seed, job_root, budget, updates=600):
    global BUDGET, UPDATES, SOURCE
    assert budget in (1,2,4,8) and updates in (2,600)
    BUDGET, UPDATES = budget, updates
    SOURCE = ARCHIVE / str(seed)
    assert read(SOURCE / 'freeze/result.json')['all_sources_frozen']
    V.SEED, V.RT = seed, Path(job_root)
    V.source = lambda method: SOURCE / ('pre_'+method) / 'policy/policy.pt'
    V.frontend_path = lambda: SOURCE / 'frontend/model.pt'
    V.progress = lambda phase, **kw: atomic_json(Path(job_root)/'progress.json', dict(phase=phase, **kw))

# Adapted from visual/train.py:ground; method bodies remain in the frozen classes.
def ground(out, arm):
    assert read(SOURCE / 'freeze/result.json')['all_sources_frozen']
    labels_paths = [LABELS / f'{i:04d}.npy' for i in range(BUDGET)]
    paths = [SOURCE / 'features/features.npz'] + labels_paths
    if arm.startswith('anchor'):
        paths += V.policy_paths()
    audit = V.watch(out, paths, 'budgeted_action_grounding')
    x = np.load(paths[0])['x'][:, :, 0]
    y = torch.tensor(np.stack([np.load(p) for p in labels_paths]), device='cuda')
    assert y.shape == (BUDGET, 200, 2)
    is_bc = arm.startswith('bc')
    if is_bc:
        mean = x[:32, :200].reshape(-1, 32).mean(0)
        scale = x[:32, :200].reshape(-1, 32).std(0).clip(0.0001)
        inputs = torch.tensor((x[:BUDGET, :200] - mean) / scale, device='cuda')
        C.seed_all(V.SEED)
        model = C.RecurrentPolicy(32, 2).cuda()
        frozen_parameters = 0
    else:
        net, ck = V.pair(arm)
        inputs = torch.tensor((x[:BUDGET, :200] - ck['obs_mean']) / ck['obs_std'], device='cuda')
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
    for step in range(UPDATES):
        ix = torch.tensor(rng.integers(BUDGET*200, size=256), device='cuda')
        if is_bc:
            pred = model(inputs)[0].reshape(-1, 2)
            loss = F.mse_loss(pred, y.reshape(-1, 2)) if arm == 'bc_full1600' else F.mse_loss(pred[ix], y.reshape(-1, 2)[ix])
        else:
            loss = F.mse_loss(model(values[ix]), y.reshape(-1, 2)[ix])
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == UPDATES-1:
            curve.append(dict(update=step + 1, sampled_loss=float(loss.detach())))
            V.progress('grounding', arm=arm, update=step + 1, updates=UPDATES)
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)[0] if is_bc else model(values).reshape(BUDGET, 200, 2)
        train_mse = float(F.mse_loss(prediction, y))
        output_std = float(prediction.reshape(-1, 2).std(0).mean())
    torch.save(dict(state_dict=model.cpu().state_dict(), mean=mean, scale=scale, arm=arm, latent_dim=32 if is_bc else values.shape[-1]), out / 'decoder.pt')
    np.save(out / 'fit_predictions.npy', prediction.cpu().numpy())
    atomic_json(out / 'curve.json', curve)
    assert not audit['violations']
    return dict(arm=arm, budget_trajectories=BUDGET, native_action_labels_read=BUDGET*200, partner_labels_read=0, simulator_queries=0, updates=UPDATES, supervised_vectors_per_update=1600 if arm == 'bc_full1600' else 256, total_label_presentations=UPDATES * (1600 if arm == 'bc_full1600' else 256), trainable_parameters=sum((p.numel() for p in model.parameters())), frozen_policy_parameters=frozen_parameters, full_fit_action_mse=train_mse, fit_output_std=output_std, selection='fixed final update; no action-dev or return selection')


# Adapted from visual/train.py:idm; pseudo-policy batch remains 8 episodes.
def idm(out):
    assert read(SOURCE / 'freeze/result.json')['all_sources_frozen']
    files = [SOURCE / 'features/features.npz'] + [LABELS / f'{i:04d}.npy' for i in range(BUDGET)]
    audit = V.watch(out, files, 'budgeted_supervised_IDM_and_pseudolabel_policy')
    x = np.load(files[0])['x'][:32, :, 0]
    y = torch.tensor(np.stack([np.load(p) for p in files[1:]]), device='cuda')
    mean = x[:, :200].reshape(-1, 32).mean(0)
    scale = x[:, :200].reshape(-1, 32).std(0).clip(0.0001)
    xx = torch.tensor((x - mean) / scale, device='cuda')
    C.seed_all(V.SEED)
    net = C.InverseDynamics(32, 2).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    rng = np.random.default_rng(V.SEED)
    cur = xx[:BUDGET, :200].reshape(-1, 32)
    nxt = xx[:BUDGET, 1:].reshape(-1, 32)
    for step in range(UPDATES):
        ix = torch.tensor(rng.integers(BUDGET*200, size=256), device='cuda')
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
    for step in range(UPDATES):
        ids = rng.integers(32, size=8)
        pred = pol(xx[ids, :200])[0]
        loss = F.mse_loss(pred, pseudo[ids])
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            V.progress('idm_pseudo_policy', update=step + 1, updates=UPDATES)
    pol.eval()
    with torch.no_grad():
        pred = pol(xx[:BUDGET, :200])[0]
    torch.save(dict(state_dict=pol.cpu().state_dict(), mean=mean, scale=scale, arm='bc_idm_relabel', latent_dim=32), out / 'decoder.pt')
    assert not audit['violations']
    return dict(arm='bc_idm_relabel', native_action_labels_read=BUDGET*200, partner_labels_read=0, simulator_queries=0, total_label_presentations=UPDATES*256, trainable_parameters=sum((p.numel() for p in pol.parameters())), idm_trainable_parameters=sum((p.numel() for p in net.parameters())), pseudo_policy_updates=UPDATES, full_fit_action_mse=float(F.mse_loss(pred, y)), scope='supervised IDM baseline; not observation-only pretraining; same distinct budgeted target labels')

