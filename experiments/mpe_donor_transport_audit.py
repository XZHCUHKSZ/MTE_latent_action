"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import json

import hashlib

from pathlib import Path

import numpy as np

from utils.atomic import atomic_json

PACKAGE=Path(__file__).resolve().parents[1]

def main():
    out=PACKAGE/'outputs/mpe_donor_transport_audit_2026_09_18'
    out.mkdir(parents=True,exist_ok=False)
    root=PACKAGE/'outputs/mpe_evidence_completion_2026_09_18/n700'
    rows=[];sources={}
    for seed in range(45,50):
        p=root/f'seed{seed}';meta=json.loads((p/'donor_effects.json').read_text())
        for f in [p/'donor_effects.json',p/'donor_predictions_evaluation_only.npz']:
            sources[str(f.relative_to(PACKAGE))]=hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(p/'donor_predictions_evaluation_only.npz') as saved:
            assert np.array_equal(saved['episode'][::2],saved['episode'][1::2])
            # Archive writer interleaves nearest_history and random_donor;
            # verify that interpretation against every saved metric before use.
            for start,kind in enumerate(['nearest_history','random_donor']):
                truth=saved['truth'][start::2].astype(float)
                for mode in ['matched','mismatched','zero']:
                    pred=np.zeros_like(truth) if mode=='zero' else saved[mode][start::2].astype(float)
                    for h in [1,2,3]:
                        reported=next(r['mse'] for r in meta['rows'] if r['kind']==kind and r['mode']==mode and r['horizon']==h)
                        assert np.isclose(np.mean((pred[:,h-1]-truth[:,h-1])**2),reported,rtol=2e-6,atol=1e-10)
            y=saved['truth'][::2,1].astype(float);z=saved['matched'][::2,1].astype(float)
            ey=float(np.mean(y*y));ez=float(np.mean(z*z));cross=float(np.mean(y*z))
            err=float(np.mean((z-y)**2));assert np.isclose(err,ez+ey-2*cross)
            rows.append(dict(seed=seed,truth_energy=ey,predicted_energy=ez,cross_moment=cross,
                mse=err,rms_ratio=float(np.sqrt(ez/ey)),
                energy_normalized_alignment=float(cross/np.sqrt(ey*ez)),
                partner_true_effect_max=float(abs(y[:,2:]).max()),
                partner_predicted_energy_fraction=float(np.square(z[:,2:]).sum()/np.square(z).sum()),
                **meta['transport_diagnostics']))
    assert all(hashlib.sha256((PACKAGE/f).read_bytes()).hexdigest()==h for f,h in sources.items())
    atomic_json(out/'summary.json',dict(rows=rows,sources=sources,source_values_verified=True,
        training_updates=0,new_rollouts=0,scope='Post-hoc diagnosis of existing h2 nearest-donor data; no causal attribution or new confirmation.'))
    lines=['# 实际训练donor：h2误差分解','',
       '只读复核原五种子的归档数组；先核对nearest/random的全部18项MSE，再分析。',
       '这不是新的优势实验，没有新训练或新rollout。','',
       '|种子|预测/真实RMS比|整体方向对齐|预测能量落在伙伴坐标比例|目标code输运差MSE|',
       '|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['seed']}|{r['rms_ratio']:.3f}|{r['energy_normalized_alignment']:.4f}|{r['partner_predicted_energy_fraction']:.2%}|{r['target_code_transport_gap']:.4f}|")
    lines+=['','这里的方向对齐为整体内积除以整体范数，并非逐样本余弦均值。',
      '实际问题同时涉及幅度和方向：不是只缩放预测就能确认物理归因。',
      'h2真实伙伴坐标效应为零；预测能量约98%也在目标坐标，伙伴编码变化为零。',
      '因此不能把误差简单归因于伙伴slot污染。目标donor code与在query状态执行相同donor动作后重新编码得到的code明显不同。',
      '这是跨状态代码含义变化的线索，不是已识别的唯一原因；预测器重组误差仍可能共同贡献。',
      '下一步可用同一donor动作在query状态重编码作为evaluation-only对照，保持训练donor规则不变，分离输运与读出问题。',
      '这种模拟器重编码不能用于observation-only训练，也不能冒充标准部署。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    atomic_json(out/'status.json',dict(status='complete',source_values_verified=True,training_updates=0))
    print('Complete',out)

if __name__=='__main__':main()
