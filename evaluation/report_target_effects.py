"""Report every frozen diagnostic, without pooling incompatible evidence."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    lines = ['# 前三项审查问题：冻结模型诊断', '',
             '这是独立评估读出，不是新控制训练、正式回报或五种子确认。每环境两个已有上游种子；reader 标注用于诊断校准，不计入原控制实验的标签预算，也不据此宣称标签效率提升。', '',
             '真实分支只在评估中构造。代码使用原一阶 inverse inference，不因物理评估结果更改输入。参考分支是固定零动作；不等同于原方法的最近历史 donor 选择。', '']
    all_rows = []
    for env in ('mpe', 'mamujoco'):
        folder = args.run/env
        if not (folder/'complete.json').exists():
            lines += [f'{env}：尚未完成。', '']; continue
        meta = json.loads((folder/'collection.json').read_text())
        lines += [f'## {env}', '', f'新评估 episode：{meta["episodes"]}；转移：{meta["samples"]}；重放最大误差：{meta["max_replay_error"]}。',
                  f'真实目标效应 RMS（h1/h2/h3）：{meta["target_effect_rms_by_horizon"]}。', '']
        if env == 'mpe':
            lines += ['本机 MPE 积分器先用旧速度更新位置，本步动作到 h2 才改变位置。h1=0 是机械退化，不能用来给模型排名。该实验也发现原一步 inverse code 在同状态零动作分支上不变；它不能在这个诊断中表示当前动作的直接变化。这不能直接推翻历史条件 donor 比较或既有控制回报，但要求单独审计时间语义。', '']
        lines += ['|训练种子|视图|模型|目标|h|标准化 MSE↓|相对校准均值 skill↑|', '|---|---|---|---|---:|---:|---:|']
        for p in sorted(folder.glob('*/scores.json')):
            obj = json.loads(p.read_text())
            for r in obj['rows']:
                all_rows.append(dict(environment=env, suite=obj['suite'], **r))
                if 'skipped' in r:
                    continue
                if r['target'] not in ('physical_target_effect', 'native_action'):
                    continue
                if r['view'] == 'direct_prediction' or r['method'] in ['target_slot','target_slot_pair','simple_route','graph_route','mif_route','base','lapo','laom','base_plus_simple_route','base_plus_graph_route','base_plus_mif_route','base_plus_global_joint16']:
                    val = r['skill_vs_mean']
                    skill = '退化/无定义' if val is None else f'{val:.4f}'
                    lines.append(f'|{obj["suite"]}|{r["view"]}|{r["method"]}|{r["target"]}|{r["horizon"]}|{r["standardized_mse"]:.7f}|{skill}|')
        lines += ['', '完整结果含正式版 history、其他基线、伙伴交互和所有符号，见各 suite/scores.json；不跨环境平均原始误差。', '']
    lines += ['## 解释边界', '',
              '- transition-informed 特征可使用评估分支的未来观察；history-only 特征只使用自然轨迹的当前及过去观察。两层分别比较，不把未来可见的结果当作部署表现。',
              '- 家族 route 表示使用既有 full-visibility 目标；formal history 使用原正式训练目标，名称分开，不拼成同一算法证据。',
              '- 配对 reader 共享 episode 划分与正则候选，但特征原生维度不同；属于读出诊断，不是等参数重训练消融。',
              '- 近零 interaction 的 skill 没有解释性，需依据 informative_target 与方差判断；不把零目标上的低误差当作分解成功。',
              '- 本轮没有训练或修改表示、history policy、controller；模型文件前后指纹一致。', '']
    (args.run/'RESULTS_CN.md').write_text('\n'.join(lines), encoding='utf-8')
    (args.run/'all_scores.json').write_text(json.dumps(all_rows, indent=2, allow_nan=False), encoding='utf-8')
    print(str(args.run/'RESULTS_CN.md'))


if __name__ == '__main__':
    main()
