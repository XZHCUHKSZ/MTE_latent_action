"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import hashlib

import json

import time

from pathlib import Path

import numpy as np

import torch

from closed_loop_lam_v1 import common as C

from mte.frontends import ObservationReadout

from environments.effect_evaluation import EffectEnvironment

from evaluation.temporal_donor_effects import donor_bank

from utils.access import read_rows

from utils.atomic import atomic_json

PACKAGE = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scores(truth, predictions, scale):
    rows = []
    for h in range(3):
        energy = np.square(truth[:, h, :2].astype(float)).mean(1)
        # Outcome-defined strata are diagnostic, not a new primary endpoint.
        ordered = np.argsort(energy, kind='stable')
        groups = [('all', np.arange(len(truth)))]
        groups += [(f'truth_energy_quartile_{i+1}', ix) for i, ix in enumerate(np.array_split(ordered, 4))]
        groups += [('effect_nonzero_1e-7', np.flatnonzero(np.sqrt(energy) > 1e-7))]
        for group, ix in groups:
            if not len(ix):
                continue
            for scope, sl in [('all_agents', slice(None)), ('target_only', slice(0, 2))]:
                y = truth[ix, h, sl].astype(float)
                for name, pred in predictions.items():
                    z = pred[ix, h, sl].astype(float)
                    mse = float(np.square(z-y).mean())
                    zero_mse = float(np.square(y).mean())
                    rows.append(dict(horizon=h+1, group=group, scope=scope, method=name,
                        n=len(ix), mse=mse, zero_mse=zero_mse,
                        skill_vs_zero=None if zero_mse <= 1e-20 else 1-mse/zero_mse,
                        standardized_mse=float(np.square((z-y)/scale[h, sl]).mean()),
                        fraction_samples_better_than_zero=float((np.square(z-y).mean(1) < np.square(y).mean(1)).mean()),
                        true_rms=float(np.sqrt(zero_mse))))
    return rows

def run_seed(seed, cfg, out, episodes):
    torch.set_num_threads(2)
    root = PACKAGE/cfg['source_run']/f'seed{seed}'
    pre = root/'pretrain'
    paths = [root/'data/train.npz', root/'data/manifest.json', root/'donor_effects.json',
        root/'donor_predictions_evaluation_only.npz', pre/'complete.json',
        pre/'offset2/entity/frontend.pt', pre/'offset2/entity/codes.npz',
        pre/'offset2/bridge/readout.pt', pre/'offset2/bridge/endpoints.npz',
        root.parent/'protocol.json']
    meta = json.loads((root/'data/manifest.json').read_text())
    paths += [Path(meta['source'])]
    before = {str(p): digest(p) for p in paths}
    frozen_hashes = {Path(k).as_posix(): v for k,v in json.loads((pre/'complete.json').read_text())['checkpoints'].items()}
    for rel in ['offset2/entity/frontend.pt', 'offset2/bridge/readout.pt']:
        assert digest(pre/rel) == frozen_hashes[rel]
    p = dict(json.loads((root.parent/'protocol.json').read_text()), seed=seed)
    bc = torch.load(pre/'offset2/bridge/readout.pt', map_location='cpu', weights_only=False)
    fc = torch.load(pre/'offset2/entity/frontend.pt', map_location='cpu', weights_only=False)
    assert fc['offset'] == 2 and bc['horizons'] == [1, 2, 3]
    entity = C.LAOMStateAdapter(2, 16, 64); entity.load_state_dict(fc['state_dict']); entity.eval()
    predictor = ObservationReadout(16, 64, 24); predictor.load_state_dict(bc['state_dict']); predictor.eval()
    with np.load(root/'data/train.npz') as f: x = f['positions']
    with np.load(pre/'offset2/entity/codes.npz') as f: z = f['z'][:len(x)].reshape(-1,4,16)
    bank = donor_bank(len(x), p)
    hist = np.concatenate([x[:,:22], x[:,1:23]], -1).reshape(-1,32)
    ht = (hist-bc['history_mean'])/bc['history_std']; bh = ht[bank]
    with np.load(pre/'offset2/bridge/endpoints.npz') as f: stored = f['donor_flat_ids']
    td = torch.cdist(torch.tensor(ht[:64]), torch.tensor(bh))
    td.masked_fill_(torch.tensor(np.arange(64)[:,None]//22 == bank[None]//22), float('inf'))
    assert np.array_equal(bank[td.argmin(1).numpy()], stored[:64])
    joint = read_rows(meta['source'], 'joint_actions', meta['train_ids'])[:,1:23].reshape(-1,4,2)
    old = dict(np.load(root/'donor_predictions_evaluation_only.npz'))
    old_meta = json.loads((root/'donor_effects.json').read_text())
    assert old_meta['episodes'] == cfg['episodes']
    # Independent metric reconstruction of all 18 archived raw/standardized errors.
    for ki, kind in enumerate(['nearest_history', 'random_donor']):
        y = old['truth'][ki::2].astype(float)
        for mode in ['matched', 'mismatched', 'zero']:
            pred = np.zeros_like(y) if mode == 'zero' else old[mode][ki::2].astype(float)
            for h in range(3):
                r = next(r for r in old_meta['rows'] if r['kind']==kind and r['mode']==mode and r['horizon']==h+1)
                assert np.isclose(np.square(pred[:,h]-y[:,h]).mean(), r['mse'], rtol=2e-6, atol=1e-10)
                assert np.isclose(np.square((pred[:,h]-y[:,h])/bc['delta_std'][h]).mean(), r['standardized_mse'], rtol=2e-6, atol=1e-10)
    def code(now, future):
        aa = torch.tensor((now[:8].reshape(-1,2)-fc['mean'])/fc['std'])
        bb = torch.tensor((future[:8].reshape(-1,2)-fc['mean'])/fc['std'])
        with torch.no_grad(): return entity(aa,bb)[1].numpy()
    def predict(h, zz):
        zz = torch.tensor((zz.reshape(1,64)-bc['zmean'])/bc['zstd'])
        with torch.no_grad(): return predictor(torch.tensor(h[None]),zz).numpy().reshape(3,8)*bc['delta_std']+bc['delta_mean']
    records = []; rng = np.random.default_rng(918101+seed)
    for ep in range(episodes):
        env = EffectEnvironment('mpe', cfg['episode_seed_start']+ep)
        try:
            previous = env.observation()
            for t in range(15):
                current = env.observation(); action = env.action()
                if t == 0: action[0] = 0
                if t in (2,6,10,14):
                    snap = env.snapshot(); seq = np.repeat(action[None],3,0)
                    factual,_ = env.roll(snap,seq); replay,_ = env.roll(snap,seq)
                    assert np.array_equal(factual,replay)
                    h = (np.concatenate([previous,current])-bc['history_mean'])/bc['history_std']
                    di = bank[torch.cdist(torch.tensor(h[None]),torch.tensor(bh)).argmin().item()]
                    rdi = int(rng.choice(bank)); actual = code(current,factual[1]); plus = predict(h,actual)
                    assert np.array_equal(plus-predict(h,actual.copy()), np.zeros((3,8)))
                    for kind, donor in [('nearest_history',di),('random_donor',rdi)]:
                        seqminus = seq.copy(); seqminus[0,0] = joint[donor,0]
                        ref,_ = env.roll(snap,seqminus); truth = factual[:,:8]-ref[:,:8]
                        assert np.array_equal(factual[0],ref[0])
                        minus = actual.copy(); minus[0] = z[donor,0]
                        matched = plus-predict(h,minus); mismatched = plus-predict(h,z[donor])
                        reencoded = code(current,ref[1]); query = actual.copy(); query[0] = reencoded[0]
                        # h2 partner positions, hence local partner codes, are exactly unchanged.
                        assert np.array_equal(actual[1:],reencoded[1:])
                        diagnostic = plus-predict(h,query)
                        i = len(records)
                        assert int(old['episode'][i]) == ep and int(old['donor'][i]) == donor
                        for name,value in [('truth',truth),('matched',matched),('mismatched',mismatched),('history',h)]:
                            np.testing.assert_allclose(value,old[name][i],rtol=2e-6,atol=1e-7,err_msg=f'{seed}/{ep}/{t}/{name}')
                        # Independently verify code/action indexing from original t+2 position windows.
                        de, dt = divmod(int(donor),22); dt += 1
                        check_code = code(x[de,dt],x[de,dt+2])
                        np.testing.assert_allclose(check_code,z[donor],rtol=2e-5,atol=2e-6)
                        records.append(dict(episode=ep,t=t,donor=int(donor),kind=kind,truth=truth,
                            matched=matched,mismatched=mismatched,query_reencoded=diagnostic,
                            code_gap=float(np.square(z[donor,0]-reencoded[0]).mean()),
                            history_distance=float(np.linalg.norm(h-ht[donor])),
                            action_gap=float(np.square(action[0]-joint[donor,0]).mean())))
                    env.restore(snap)
                previous = current; env.step(action)
        finally: env.close()
        atomic_json(out/'progress.json',dict(seed=seed,episode=ep+1,episodes=episodes,time=time.time()))
    arrays = {k:np.stack([r[k] for r in records]) for k in ['truth','matched','mismatched','query_reencoded','episode','t','donor','code_gap','history_distance','action_gap']}
    np.savez_compressed(out/'predictions.npz', **arrays)
    rows=[]; distribution=[]
    for ki,kind in enumerate(['nearest_history','random_donor']):
        truth=arrays['truth'][ki::2]
        predictions={k:arrays[k][ki::2] for k in ['matched','mismatched','query_reencoded']}
        predictions['zero']=np.zeros_like(truth)
        rows.extend([dict(seed=seed,kind=kind,**r) for r in scores(truth,predictions,bc['delta_std'])])
        energy=np.sqrt(np.square(truth[:,1,:2].astype(float)).mean(1))
        distribution.append(dict(kind=kind,h2_nonzero_fraction=float((energy>1e-7).mean()),
            h2_target_rms_quantiles=np.quantile(energy,[0,.25,.5,.75,1]).tolist(),
            mean_code_gap=float(arrays['code_gap'][ki::2].mean()),
            zero_action_gap_count=int((arrays['action_gap'][ki::2]<1e-14).sum())))
    assert all(digest(f)==v for f,v in before.items())
    result=dict(seed=seed,episodes=episodes,rows=rows,distribution=distribution,
        archived_metrics_reproduced=True,replay_arrays_verified=True,donor_code_index_verified=True,
        all_source_hashes_unchanged=True,source_hashes=before,training_updates=0,
        diagnostic_oracle_uses_simulator=True,source_models='frozen original entity encoder and observation readout')
    atomic_json(out/'result.json',result)
    return result

def summarize(out, results, cfg):
    from scipy.stats import t as student_t
    means=[]; keys=['kind','horizon','group','scope','method']
    for template in results[0]['rows']:
        rs=[next(r for r in result['rows'] if all(r[k]==template[k] for k in keys)) for result in results]
        means.append(dict(**{k:template[k] for k in keys},mse=float(np.mean([r['mse'] for r in rs])),
            zero_mse=float(np.mean([r['zero_mse'] for r in rs])),
            standardized_mse=float(np.mean([r['standardized_mse'] for r in rs])),
            seed_mses=[r['mse'] for r in rs]))
    contrasts=[]
    for method in ['matched','query_reencoded']:
        delta=[]
        for result in results:
            r=next(r for r in result['rows'] if r['kind']=='nearest_history' and r['horizon']==2 and r['group']=='all' and r['scope']=='all_agents' and r['method']==method)
            delta.append(r['zero_mse']-r['mse'])
        d=np.array(delta); mean=float(d.mean()); ci=None; p=None
        if len(d)>1:
            radius=float(student_t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)));ci=[mean-radius,mean+radius]
            import itertools
            null=np.abs(np.array(list(itertools.product([-1,1],repeat=len(d))))@d/len(d))
            p=float((null>=abs(mean)-1e-15).mean())
        contrasts.append(dict(method=method,metric='zero MSE minus method MSE; positive is better',mean=mean,ci95=ci,exact_two_sided_p=p,seed_deltas=delta,scope='diagnostic, not confirmatory superiority test'))
    report=dict(status='complete',seeds=[r['seed'] for r in results],episodes_per_seed=results[0]['episodes'],
        means=means,contrasts=contrasts,source_and_weight_checks=all(r['all_source_hashes_unchanged'] for r in results),
        all_legacy_values_verified=True,training_updates=0,scope=cfg['scope'])
    atomic_json(out/'summary.json',report)
    lines=['# Zero-effect 与真实训练 donor：公平性和输运审计','','本报告没有修改训练、donor 选择或原结果。query_reencoded 使用模拟器观测重编码，只是诊断 oracle，不能作为标准部署优势。',
        '原18项原始/标准化MSE均重新计算，所有重放样本和donor身份逐项通过核验；t+2 code索引通过独立窗口重编码核验。','',
        '|donor|horizon|坐标|原MTE MSE ↓|同状态重编码 MSE ↓|zero MSE ↓|','|---|---:|---|---:|---:|---:|']
    def lookup(kind,h,scope,method,group='all'):
        return next(r for r in means if r['kind']==kind and r['horizon']==h and r['scope']==scope and r['method']==method and r['group']==group)
    for kind in ['nearest_history','random_donor']:
        for h in [1,2,3]:
            for scope in ['all_agents','target_only']:
                values=[lookup(kind,h,scope,m)['mse'] for m in ['matched','query_reencoded','zero']]
                lines.append(f'|{kind}|{h}|{scope}|'+ '|'.join(f'{v:.8g}' for v in values)+'|')
    lines+=['','## h2目标坐标：所有效应大小分组均保留','','分位组按真实效应排序定义，仅作事后误差定位，不作为替换全样本主指标的理由。',
        '|组|原MTE/zero MSE比|重编码/zero MSE比|','|---|---:|---:|']
    for group in ['all',*[f'truth_energy_quartile_{i}' for i in range(1,5)],'effect_nonzero_1e-7']:
        r=lookup('nearest_history',2,'target_only','matched',group)
        q=lookup('nearest_history',2,'target_only','query_reencoded',group)
        lines.append(f"|{group}|{r['mse']/r['zero_mse']:.4f}|{q['mse']/q['zero_mse']:.4f}|")
    lines+=['','## 五种子h2全坐标差值','','正数表示优于zero；负数表示差于zero。五个模型共享评估episode集合，不把这些episode当新训练重复。']
    for r in contrasts: lines.append(f"- {r['method']}: Δ={r['mean']:+.8g}; t95={r['ci95']}; exact p={r['exact_two_sided_p']}; seeds={r['seed_deltas']}")
    lines+=['','## 解释边界','',
        '- h1是位置即时效应为零的负对照，zero胜出不否定控制表征；h2/h3有非零物理效应，必须单独检验。',
        '- zero是物理效应MSE的有效无学习基准，不是控制策略；差于它不能直接推出MTE闭环控制无用。',
        '- 若query重编码改善，只能支持跨状态code输运是该测试的重要误差来源；没有证明可部署的donor修复或控制收益。',
        '- 本诊断沿用已分析过的状态，不是新环境、独立确认或对原负结果的替换。',
        '- 全表与逐种子数据见summary.json和seed*/result.json；论文和图表未自动更改。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    atomic_json(out/'status.json',dict(status='complete',seeds=report['seeds'],source_and_weight_checks=True))
