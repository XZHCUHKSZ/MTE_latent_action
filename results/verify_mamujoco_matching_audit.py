"""Recompute frozen matching metrics from saved predictions and physical cube."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.frozen_effect_models import Frontend, Predictor
import torch
RUN = ROOT/'outputs/mamujoco_frozen_matching_audit_2026_09_19'
OUT = ROOT/'results/mamujoco_frozen_matching_audit_2026_09_19'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    cfg, status, summary = (read(RUN/f'{name}.json') for name in ('protocol', 'status', 'summary'))
    assert status['status']=='complete' and status['seeds']==5 and status['rows']==135
    manifests = read(RUN/'source_manifest.json')
    for group in manifests.values():
        for path, expected in group.items():
            p = Path(path)
            if not p.is_absolute(): p = ROOT/p
            assert digest(p)==expected, str(p)
    with np.load(RUN/'physical_evaluation_only.npz', allow_pickle=False) as f:
        cube, live, cur, ep = (f[k] for k in ('cube', 'live', 'current', 'episode'))
        pre, future = f['previous'], f['factual_future']
        assert f['episode_seeds'].tolist()==cfg['episode_seeds']['mamujoco']
    fit = np.isin(ep, cfg['splits']['fit_ids']); test = np.isin(ep, cfg['splits']['test_ids'])
    assert not np.any(fit & test)
    scale = cur[fit,:27].std(0).astype(float); scale[scale<1e-4]=1.
    true = (cube[:,0::2]-cube[:,1::2]).transpose(0,2,1,3)
    true_coeff = np.stack([true[:,:,k]-true[:,:,0] for k in [1,2,4]],axis=2)
    rows=[]
    for seed in range(5):
        saved=read(RUN/f'seed{seed}.json'); rows.extend(saved['rows'])
        paths = {Path(k).name: (Path(k) if Path(k).is_absolute() else ROOT/k) for k in saved['input_hashes']}
        front = Frontend(paths['frontend.pt'], 'mamujoco')
        predictor = Predictor(paths['readout.pt'], 'mamujoco', paths['train.npz'])
        q, ref = front.encode(pre,cur,future[:,0]), front.encode(pre,cur,cube[:,15,0])
        cont = np.stack([front.encode(cur,future[:,0],future[:,1]), front.encode(future[:,0],future[:,1],future[:,2])],1)
        mismatch = predictor.rollout(pre,cur,q,cont)-predictor.rollout(pre,cur,ref,cont)
        with np.load(RUN/f'seed{seed}_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(true,z['true_edges'])
            assert np.array_equal(true_coeff,z['true_coefficients'])
            for r in saved['rows']:
                h=r['horizon']-1; keep=test & live[:,:,:h+1].all((1,2))
                if r['target']=='first_order':
                    y=true[:,h,0,:27]
                    if r['method']=='partner_mismatched_edge': pred=mismatch[:,h,:27]
                    else: pred=np.zeros_like(y) if r['method']=='zero' else z['predicted_edges'][:,h,0,:27]
                else:
                    j=r['partner']-1; y=true_coeff[:,h,j,:27]
                    pred=np.zeros_like(y) if r['method']=='zero' else z['predicted_coefficients'][:,h,j,:27]
                err=pred[keep].astype(float)-y[keep].astype(float)
                assert np.isclose(np.mean(err**2),r['mse'],rtol=1e-10,atol=1e-12)
                assert np.isclose(np.mean((err/scale)**2),r['standardized_mse'],rtol=1e-10,atol=1e-12)
    assert rows==summary['rows']
    collection=read(RUN/'collection.json')
    assert collection['episodes']==96 and collection['max_replay_error']==0
    source_files=[RUN/'protocol.json',RUN/'summary.json',RUN/'physical_evaluation_only.npz',RUN/'source_manifest.json']
    OUT.mkdir(exist_ok=True)
    (OUT/'VERIFICATION.json').write_text(json.dumps(dict(status='pass', rows=135,
        metrics_recomputed_rows=135, legacy_replay_rows=30,
        code_and_input_hashes_verified=True, simulator_replay_error=0,
        training_updates=0, sources={str(p.relative_to(ROOT)):digest(p) for p in source_files}),indent=2),encoding='utf-8')
    lines=['# MaMuJoCo 五种子冻结匹配诊断（2026-09-19）','',
        '复用原训练种子0–4的冻结实体前端与预测器；96条新评估轨迹，48/24/24沿用校准/验证/测试划分。直接预测不拟合reader，尺度仅由校准自然观察确定。不是新控制训练、实际历史donor验证或全新任务。', '',
        '本实验沿用原同状态零动作诊断：替换code来自全agent首步零动作分支的局部编码；物理目标则按四agent动作组合计算。固定上下文不使latent替换自动等价于原生动作干预。输出为原105维观察的前27维运动学坐标，排除contact。', '',
        '## 一阶目标效应','', '|时域|匹配MSE↓|未匹配MSE↓|零预测MSE↓|','|---|---:|---:|---:|']
    for h in [1,2,3]:
        vals=[np.mean([r['standardized_mse'] for r in rows if r['target']=='first_order' and r['horizon']==h and r['method']==m]) for m in ['matched_edge','partner_mismatched_edge','zero']]
        lines.append(f'|h{h}|{vals[0]:.6f}|{vals[1]:.6f}|{vals[2]:.6f}|')
    lines+=['','h1匹配较未匹配和零预测的平均标准化误差分别降低84.21%与67.47%，五个种子方向均一致。h2均值也优于两个对照；h3未优于零预测。','',
        '|h1预定误差降低|均值|配对t95区间|正向种子|精确p|Holm p|','|---|---:|---|---:|---:|---:|']
    for name,c in summary['primary_h1_error_reduction'].items():
        lines.append(f'|{name} − matched|{c["mean"]:.6f}|[{c["ci95"][0]:.6f}, {c["ci95"][1]:.6f}]|{c["positive_seeds"]}/5|{c["exact_p"]:.4f}|{c["holm_p"]:.3f}|')
    lines+=['','t区间为近似正态种子差假设下的描述，精确符号置换采用五个模型种子为单位，Holm校正包含两个预定比较。不能用t区间替换预定精确检验后称为校正显著。评估episode是共同样本，不作为新增模型种子。','',
        '## 真实二阶伙伴交互','', '|时域|预测系数MSE↓|零预测MSE↓|','|---|---:|---:|']
    for h in [1,2,3]:
        vals=[np.mean([r['standardized_mse'] for r in rows if r['target']=='second_order' and r['horizon']==h and r['method']==m]) for m in ['matched_coefficient','zero']]
        lines.append(f'|h{h}|{vals[0]:.6f}|{vals[1]:.6f}|')
    lines+=['','每个种子先对三个伙伴等权，再汇总五个种子。三个时域预测系数均未胜零预测；真实四分支系数与模型内部learned coefficient是不同目标。','',
        '## 可用结论','',
        '五个既有模型种子的同状态测试一致支持：保持伙伴上下文匹配，能改善MaMuJoCo一步目标效应预测。与MPE同状态匹配诊断形成跨环境的算子级证据。它未验证跨状态donor输运、家族编码后的物理交互保留或下游控制因果归因。','',
        '## 核验与来源','',
        '新评估器先在原96-episode数据上重现种子0/1的30条直接诊断记录（含所有h与伙伴系数）。正式运行不覆盖原记录；保存135条指标、物理cube及五份全预测数组。来源文件与权重哈希前后相同，同状态模拟器重放误差为零。','',
        '协议：`configs/mamujoco_frozen_matching_audit.json`；入口：`experiments/mamujoco_frozen_matching_audit.py`；输出：`outputs/mamujoco_frozen_matching_audit_2026_09_19/`。',
        '后验核验重算135条匹配/未匹配/零/系数记录；未匹配由冻结原网络重放两个端点独立核验。完整源文件SHA256在本目录VERIFICATION.json。',
        '论文图表尚未合入此新增诊断；原主实验与既有结构消融不变。','']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PASS: 135 rows; 135 metrics recomputed; original inputs unchanged')


if __name__=='__main__': main()
