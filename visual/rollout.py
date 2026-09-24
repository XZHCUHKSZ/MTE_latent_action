import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C
from . import train as V
from .train import controller, read
from .render import teacher, render
SEEDS = list(range(908601, 908631))
ARMS = V.ARMS
RT = None
progress = V.progress

@torch.no_grad()
def evaluate(out, arm):
    assert all(((RT / ('ground_' + a) / 'result.json').exists() for a in ARMS))
    from .render import teacher, render
    from environments.mamujoco import make_env
    predict = controller(arm)
    x = torch.tensor(np.load(V.RT / 'features/features.npz')['x'][32:, :200, 0], device='cuda')
    truth = torch.tensor(np.stack([np.load(V.OLD / f'labels/dev/{e:04d}.npy') for e in range(8)]), device='cuda')
    pred = predict(x)[0]
    h = None
    seq = []
    for t in range(200):
        a, h = predict(x[:1, t:t + 1], h)
        seq.append(a)
    err = float((pred[:1] - torch.cat(seq, 1)).abs().max())
    assert err < 1e-05
    metrics = dict(action_mse=float(F.mse_loss(pred, truth)), executed_action_mse=float(F.mse_loss(pred.clamp(-1, 1), truth)), output_std=float(pred.reshape(-1, 2).std(0).mean()), sequential_error=err)
    net = V.visual_net()
    pc = torch.load(V.RT / 'features/projection.pt', map_location='cuda', weights_only=False)
    p = read(V.OLD / 'visual_config.json')
    expert = teacher(p)
    rows = []
    for i, seed in enumerate(SEEDS):
        env = make_env(seed)
        r = None
        history = V.OnlineHistory()
        h = None
        rewards = []
        actions = []
        try:
            r, _ = render(env, p, 'dev', seed + 10000, samples=0, target_only=True)
            for step in range(200):
                pix = history.observe(r.pixels())[0]
                ff = (V.pooled(net, torch.from_numpy(pix[None]).cuda().float() / 127.5 - 1) - pc['mean']) @ pc['basis'] / pc['scale']
                pred, h = predict(ff[:, None], h)
                a = expert.joint_action(env)
                a[0] = pred[0, 0].cpu().numpy().clip(-1, 1)
                result = env.step({ag: a[j].copy() for j, ag in enumerate(env.possible_agents)})
                rewards.append(result[1]['agent_0'])
                actions.append(a[0].copy())
                r.advance()
                if step % 50 == 0:
                    progress('evaluation', arm=arm, episode=i + 1, episodes=len(SEEDS), step=step + 1)
                if not env.agents:
                    break
            rows.append(dict(seed=seed, return_value=float(sum(rewards)), steps=len(rewards)))
            np.savez_compressed(out / f'episode_{seed}.npz', rewards=np.asarray(rewards), target_actions=np.asarray(actions))
        finally:
            if r:
                r.close()
            env.close()
    return dict(arm=arm, metrics=metrics, rows=rows, mean_return=float(np.mean([r['return_value'] for r in rows if r['seed'] >= 908604])), all30_mean=float(np.mean([r['return_value'] for r in rows])), scope='One of five training seeds; 27 new evaluation conditions; 3 earlier development conditions excluded from primary mean')
