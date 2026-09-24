"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""
from mte.method_names import report_text

import hashlib

import json

from pathlib import Path

import time

import numpy as np

from environments.effect_evaluation import EffectEnvironment

from utils.atomic import atomic_json

PACKAGE = Path(__file__).resolve().parents[1]

def second_difference(factual, target_zero, partner_zero, both_zero):
    # Subtract in float64, so cancellation does not add float32 arithmetic error.
    return (factual.astype(np.float64) - target_zero.astype(np.float64)
            - partner_zero.astype(np.float64) + both_zero.astype(np.float64))

def run(config, out):
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'protocol.json', config)
    started = time.time()
    atomic_json(out/'status.json', dict(status='running', episodes=0, started=started))
    rows = []
    raw = []
    for ep in range(config['episodes']):
        env = EffectEnvironment('mpe', config['episode_seed_start'] + ep)
        try:
            for t in range(max(config['sample_times'])+1):
                action = env.action()
                if t == 0:
                    action[0] = 0
                if t in config['sample_times']:
                    snap = env.snapshot()
                    seq = np.repeat(action[None], max(config['horizons']), axis=0)
                    f, live = env.roll(snap, seq)
                    replay, live2 = env.roll(snap, seq)
                    assert np.array_equal(f, replay) and np.array_equal(live, live2)
                    assert live.all(), 'Branch horizon crossed episode boundary'
                    target_seq = seq.copy(); target_seq[0, 0] = 0
                    target, _ = env.roll(snap, target_seq)
                    assert np.array_equal(f[0], target[0]), 'h1 position control failed'
                    for partner in config['partners']:
                        ps = seq.copy(); ps[0, partner] = 0
                        bs = target_seq.copy(); bs[0, partner] = 0
                        p, _ = env.roll(snap, ps)
                        b, _ = env.roll(snap, bs)
                        assert np.array_equal(f[0], p[0]) and np.array_equal(f[0], b[0])
                        interaction = second_difference(f[:,:8], target[:,:8], p[:,:8], b[:,:8])
                        target_effect = f[:,:8].astype(float) - target[:,:8].astype(float)
                        # Constant landmarks must never masquerade as an interaction.
                        assert not np.any(second_difference(f[:,8:], target[:,8:], p[:,8:], b[:,8:]))
                        raw.append(dict(episode=ep, t=t, partner=partner,
                                        branches=np.stack([f,target,p,b])[:,:,:8]))
                        for h in config['horizons']:
                            effect = interaction[h-1]
                            rows.append(dict(episode=ep, time=t, partner=partner, horizon=h,
                                interaction_mse=float(np.mean(effect**2)),
                                interaction_max_abs=float(np.max(np.abs(effect))),
                                target_effect_mse=float(np.mean(target_effect[h-1]**2))))
                    env.restore(snap)
                env.step(action)
        finally:
            env.close()
        atomic_json(out/'status.json', dict(status='running', episodes=ep+1,
                    total=config['episodes'], elapsed_seconds=time.time()-started))
    stats=[]
    for h in config['horizons']:
        rr=[r for r in rows if r['horizon']==h]
        ep_mse=[float(np.mean([r['interaction_mse'] for r in rr if r['episode']==e]))
                for e in range(config['episodes'])]
        stats.append(dict(horizon=h, interaction_rms=float(np.sqrt(np.mean(ep_mse))),
            target_effect_rms=float(np.sqrt(np.mean([r['target_effect_mse'] for r in rr]))),
            max_abs=float(max(r['interaction_max_abs'] for r in rr)),
            fraction_nonzero={str(q):float(np.mean([r['interaction_max_abs']>q for r in rr]))
                              for q in config['nonzero_thresholds']},
            episode_interaction_mse=ep_mse))
    np.savez_compressed(out/'physical_branches_evaluation_only.npz',
        branches=np.stack([r['branches'] for r in raw]),
        episode=[r['episode'] for r in raw],time=[r['t'] for r in raw],
        partner=[r['partner'] for r in raw],branch_names=['factual','target_zero','partner_zero','both_zero'])
    hashes={str(f.relative_to(PACKAGE)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),PACKAGE/'environments/effect_evaluation.py',
                      PACKAGE/'environments/mpe.py',PACKAGE/'environments/native_particle.py']}
    atomic_json(out/'summary.json',dict(protocol=config, horizons=stats, rows=rows,
        replay_exact=True,h1_position_effect_zero=True, source_hashes=hashes,
        training_updates=0, learned_models_evaluated=False,
        elapsed_seconds=time.time()-started))
    report=['# MPE真实伙伴交互出现时间：评估诊断','',
      '固定同一状态，目标与伙伴第一步动作各保留/置零，后续动作完全相同。',
      '这是模拟器测量审计，不是任何PC模型的效果验证，也不修改原h1–h3模型。','',
      '|h|交互RMS|目标一阶效应RMS|交互最大绝对值|超过1e-7比例|',
      '|---|---:|---:|---:|---:|']
    report += [f"|{r['horizon']}|{r['interaction_rms']:.6g}|{r['target_effect_rms']:.6g}|{r['max_abs']:.6g}|{r['fraction_nonzero']['1e-07']:.1%}|" for r in stats]
    report += ['',f"{config['episodes']}个episode；每个4个状态、3个伙伴。状态/伙伴不是独立训练种子。",
      '保留全部h1–h8及三档浮点阈值；数据不回流训练，不基于结果选择论文模型或确认检验。']
    (out/'RESULTS_CN.md').write_text(report_text(report, "control")+'\n',encoding='utf-8')
    atomic_json(out/'status.json',dict(status='complete',episodes=config['episodes'],
                replay_exact=True,training_updates=0,elapsed_seconds=time.time()-started))
    print(json.dumps(stats,ensure_ascii=False))
