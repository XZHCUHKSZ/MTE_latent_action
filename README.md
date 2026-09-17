# 与当前论文对应的实验代码

本包以 `iclr2027_control_ablation_figures_2026_09_17/main.tex` 及其最终证据表为准。它将散落在多轮开发目录中的最终函数归并到功能模块，并去掉历史看板、调度器和试跑入口；没有改写原仓库，也没有重新训练。

## 从这里开始

```text
python run.py --list
python run.py --report numeric
python run.py --report visual
python run.py --report route
python run.py --check-results
```

以上命令只读包内结果，不启动实验。训练实现与配置可从下表进入；不要直接运行旧日期目录下的 worker。

| 实验小标题 / 要验证什么 | 整理后的入口 | 核心实现 | 论文证据 |
|---|---|---|---|
| X1 有限动作标签下，MTE family 能否支持目标 agent 控制？ | `experiments/limited_labels.py` | `mte/`、`training/` | `results/numeric/limited_labels.json` |
| X2 标签固定时，增加无标签观测是否有帮助？ | `experiments/unlabelled_scaling.py` | 同一正式训练流程、N 专属端点和质量检查 | `results/numeric/unlabelled_scaling.json` |
| X3 从共享物理环境的 RGB 观测中能否学习控制？ | `experiments/visual_control.py` | `visual/train.py`、`visual/render.py`、`visual/rollout.py` | `results/visual/summary.json` |
| X4 相比完整系统辅助表示，MTE 结构学习路线是否有控制收益？ | `experiments/route_ablation.py` | `training/route_representation.py`、Global-Joint16 | `results/route/` |
| X5 冻结的历史特征保留多少目标动作信息？ | `experiments/action_information.py` | `evaluation/probes.py` | `results/probes/` |
| X6 subset / Möbius 结构是否保留 learned partner interaction？ | `experiments/partner_interactions.py` | `mte/coordinate_controls.py`、固定 ridge reader | `results/interactions/` |
| X7 各输入和结构组件在资源控制下起什么作用？ | `experiments/resource_controls.py` | `mte/structured_controls.py` | `results/structural/` |

## 代码层次

```text
experiments/          按论文问题划分的入口与实验接口
configs/              最终训练预算、划分、方法与评估配置
mte/                  frontend、匹配端点、实体投影与结构对照
training/             表示训练、history policy、grounding、冻结组合
evaluation/           环境回报及动作/interaction probes
environments/         环境适配和仅评估使用的 teacher
visual/               RGB frontend、MTE、family、最终渲染与 rollout
closed_loop_lam_v1/   三个冻结核心文件的逐字副本（检查点兼容）
third_party/          实际使用的第三方源码与许可证
results/              当前论文采用的结果，不是重新训练所得
provenance/           每个函数及每条已核对数值的原始来源
tests/                重组检查与 CPU 一致性检查
```

## 整理的边界

算法函数从产生最终结果的代码中抽取，替换模块导入与文件寻址；实验入口将正式版和诊断版明确分开。三份受保护核心文件保持逐字不变，类和检查点命名空间也不变。因此其中保留一些兼容分支；它们不是本包推荐的独立实验。

这是研究代码的功能重组与结果核对包，不是已在新目录完整重训通过的一键复现发行版。已完成的检查列在 `provenance/validation.json`。训练入口需要显式提供冻结的数据、端点、检查点和标签出口，并在独立进程内安装读取审计。重组后尚未进行整套 GPU 重训，不把静态/CPU 检查说成完整复现。

原论文/代码中的主要结果及有效不利对照都按各自作用保留。旧局部双分支调度实验不是当前 X4 的入口；当前 X4 是完整系统 Global-Joint16 对照。结果整理不会把两者说成同一个消融。

详见 `版本归并说明.md` 和 `模型与数据边界.md`。

## 简易冒烟测试

已完成数值短训练、grounding、8 条结构对照路线、环境执行和视觉模块测试。结果与具体覆盖范围见 `冒烟测试说明.md` / `provenance/smoke_report.json`；完整 GPU 复现实验仍未重新运行。
