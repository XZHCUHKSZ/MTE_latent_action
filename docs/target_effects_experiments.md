# 前三项审查意见对应的实验

本次实现依次针对：已知 slot 的贡献、真实动作效应参照、history 迁移。没有替换现有模型定义，也没有把评估模拟器分支用于预训练。

## 文件与入口

| 文件 | 用途 |
|---|---|
| `experiments/target_effects.py` | X8–X10 的 manifest、冻结评估与分层读出入口 |
| `environments/effect_evaluation.py` | 仅评估使用的两环境快照、恢复及固定续行 |
| `evaluation/frozen_effect_models.py` | 原 frontend、predictor、family、history checkpoints 的只读推理 |
| `evaluation/effect_statistics.py` | 真实 edge/单伙伴差分、episode 划分、ridge 与误差 |
| `evaluation/report_target_effects.py` | 全符号报告，分别呈现环境、种子、协议和层次 |
| `configs/target_effects_pilot.json` | 24 episode 接口诊断；两个已有上游种子 |
| `configs/target_effects_development.json` | 固定后扩到 96 新 episode；48/24/24 校准/验证/测试 |
| `tests/check_target_effects.py` | 分支方向、索引、数据隔离和真实 MPE 脉冲测试 |

在此代码包根目录运行（将 WORKSPACE 改为保有原始 checkpoints 的路径）：

```text
python tests/check_target_effects.py
python -m experiments.target_effects --workspace WORKSPACE --environment mpe --config configs/target_effects_development.json --out outputs/new_effect_run/mpe
python -m experiments.target_effects --workspace WORKSPACE --environment mamujoco --config configs/target_effects_development.json --out outputs/new_effect_run/mamujoco
python -m evaluation.report_target_effects outputs/new_effect_run
```

输出目录必须不存在；所有旧运行保留。该入口本地读取已有 checkpoint，不下载模型。运行所需 teacher/data/weights 不包含在公共代码包内。两环境可在独立进程并行，默认每个进程两个 CPU 计算线程，不使用 GPU 训练。

## 测量对象与边界

同一自然轨迹状态上生成 16 种四-agent 首步动作配置：每个 agent 选择记录动作或零动作，后两步均重放记录动作。target 的正向减参考形成 8 个 partner-context edges。单伙伴 interaction 为 g({i})−g(empty)，与原定义方向一致。

这是 **same-state zero-reference evaluation**。原 MTE 用历史相近的 donor；本次并未把训练 donor 改成零动作，也不把本诊断结果称作原 nearest-donor 流程的完整验证。物理 reference 分支产生的未来观察只用于独立诊断的 inverse 输入，不能用于 deployment。

原模型 code inference、预测器、实体投影和 family 参数原封不动读取。Simple/Graph/MIF 的可加载表示来自已归档 route 实验，使用 full-visibility 目标；原正式模型的 history 则单独标为 formal。两类不可合并成一种训练协议。

三个问题分别由以下输出回答：

1. **slot 与 MTE：**在相同评估样本上比较 target-slot、paired target-slot、joint LAPO/LAOM、raw edge、Simple/Graph/MIF；history 层比较 Base、家族单路、Base+家族、Base+Global-Joint16。它们共享校准和测试集，不宣称原生维度/原训练计算完全相同。
2. **真实效应：**MTE 直接输出误差不拟合额外 reader。未匹配对照仅换参考端点的 partner assignments，保留 history 和 target codes。统一 reader 则另列为诊断监督，输出真实动作、真实目标效应和真实单伙伴效应。
3. **history 迁移：**当前与过去观察独立送入各冻结 history policy，和 transition-informed 表示分别读出同一目标；每个网络做未来后缀扰动测试。相同 Base features 复用于全部拼接，不额外训练第二套 Base。

全部 reader 仅在校准集拟合、验证集选 ridge，在 episode-disjoint 测试集评分。测试标签不参与拟合与调参。reader 使用的真实动作/效应是额外评估监督，不能套用为原文 B 个标签的控制表现。

## MPE 时间接口审计

本机安装的 MPE2 `World.integrate_state` 先执行 position += old_velocity * dt，再由当前 force 更新 velocity。于是从相同状态改变当前动作：

- h1 的位置相同，真实位置 effect 严格为零；
- h2 起才能看到当前动作引起的位置变化；
- 通过伙伴物理交互传播的影响可能更晚出现，h1/h2 的 interaction 也可能退化。

真实 pulse-action 测试和自然轨迹评估均验证了这一点。初始 h1 试跑保留；后续协议明确 MPE 主物理 horizon 为 h2，MaMuJoCo 为 h1，全部 h1–h3 直接结果仍保存。该调整依据机械更新时间，不依据哪个结果更正向。

**没有将 MPE inverse 输入改为两步，也没有改动原模型。** 因此在该同状态分支诊断中，原一步 inverse code 可能完全不变。这是需要单独解决的时间语义问题，不能把 history 控制可预测性和即时动作识别混为一谈；也不能直接宣布旧回报失效。后续若改 inference offset，必须另建协议和新模型标签，不能覆盖原实验。

## 证据强度与后续

本轮使用两个已有上游种子，评估 episode 为新生成；96 个 episode 不等于 96 个训练种子。结果属于开发诊断，不计算宣称五种子确认的 p 值。通过它可以发现时间接口和语义瓶颈，并检查外部物理参照；尚未完成 nearest-history-donor 的物理配对、五个 fresh upstream seeds、等资源重训练或新控制收益实验。

每次输出包含 frozen source manifests、物理 cube、原始特征、reader predictions、完整 scores、future-leakage 测试误差和源文件前后指纹。接口失败写 failure.json；仅全部完成后写 complete.json。所有生成数组位于被 `.gitignore` 排除的 outputs 目录，禁止并入无标签训练出口。
