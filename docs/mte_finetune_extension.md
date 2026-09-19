# MTE自身模型的固定预算微调扩展

用户于2026-09-20明确要求在视觉和MaMuJoCo中轻量测试自己的模型。本试验回答：保留既有无动作预训练后，允许少量动作标签更新history策略，会怎样改变各方法的闭环回报？原冻结主试验保持原样。

## 固定规模与来源

- 视觉：Simple、Graph、Tree各Solo/Aux；五个上游种子908711–715；固定B2，即32条训练轨迹中的2条（6.25%），400个二维目标动作标签。共30个新增结果。原MIF的Solo/Aux同规则B2结果已存在，不重算。
- MaMuJoCo状态输入：Simple/Graph matched路线和原生masked Möbius MIF，均为Base+辅助history的32维组合；种子0–4，N90，固定B8（8.89%训练轨迹），1600个填充槽位中仅有效前缀计入动作标签与损失。训练/验证轨迹为6/2，全部计入B8预算。仅用预先指定decoder seed 202609160，共15个新增结果。
- MaMuJoCo MIF复用通过native parity的`mamujoco_mif_visibility_2026_09_19`中`masked_mobius`，不使用前轮整表MSE、full-visibility路线MIF。Simple/Graph来自`mamujoco_route_completion_2026_09_19`的matched版本。跨方法预训练差异保留，每个微调结果只与自身准确来源的冻结结果配对。
- 45新增+45原冻结参照，共90个成对路线身份。已有任务与既有种子，是固定预算后续研究，不是新任务确认。

## 学习边界与公平性

只更新history网络的副本和动作decoder。latent encoder、RGB/PCA前端、旧history原件、数据和所有旧结果保持不变。不读取伙伴动作，不在训练中查询模拟器，不使用评价回报挑选权重、预算或方法。

视觉直接复用上一轮MIF监督适配器：同一方法的初始化、无动作统计得到的归一化、标签、256个采样索引、600更新完全一致，采用最终更新，无验证回报选模。MaMuJoCo复用原ActionDecoder与GRU类型，800更新、Adam 0.001、有效步采样min(512,n_fit)、原每25步的预算内验证规则。history和decoder保存于同一个最佳验证步；不另增加验证标签。

监督微调增加可训练history参数。这是受控地改变下游适配方式，不宣称两条路径可训练参数数目相同，也不能仅凭此对照估计预训练相对随机初始化的增量。MaMuJoCo的一个decoder种子不是五个decoder重复；统计重复单位始终是五个上游种子。

## 门槛、审计和调度

独立入口`experiments/mte_finetune_extension.py`、适配器`training/mte_finetune_extension.py`、协议`configs/mte_finetune_extension.json`。最多四路，先等待`policy_finetune_pilot_2026_09_20_r1`完整结束，避免叠加八路。

预检包含九条真实2更新路径及九个原环境单回合，另外复现seed0三条MaMuJoCo完整冻结训练权重与原首episode回报。正式门槛为九smoke、45冻结权重复现（maxabs≤1e-6）、45原episode重放；视觉逐步动作/奖励与原件逐项一致，MaMuJoCo首episode回报误差≤1e-9。全部通过才派发正式微调。

训练后核对45组history与decoder初始化、归一化、输入、标签、采样哈希、架构参数数目及MaMuJoCo预算划分相等；所有新权重冻结后再评价。来源、原件、新权重和访问审计完成后运行foundation verifier。任何资格失败停止派发并保留现场，不放宽门槛，不覆盖partial。

视觉30个共同条件中908604–630的27个用于主统计；MaMuJoCo50个原共同episode，每个最多200步。每个上游种子先平均episode，再计算五种子微调−冻结配对均值、t95、精确双侧符号置换p。视觉六项和MaMuJoCo三项分别为Holm族。五种子精确p最低0.0625，全部正负方向保留。

## 文献依据和表述

[LAPA第3.3节](https://arxiv.org/html/2410.11758v2#S3.SS3)先从无动作视频学习latent action并训练预测策略，再用少量带真实动作轨迹微调。它替换latent action head，冻结视觉编码器并解冻语言模型主体。这支持“无动作预训练后做少标签策略适配”的研究范式；其VLA架构和动作离散化与本试验不同，不称为逐项复现。

[LAPO原论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/27985d21f0b751b933d675930aa25022-Paper-Conference.pdf)的少标签离线路线主要训练latent到真实动作的decoder，并讨论在线强化学习适配。不能把它的离线实验说成与本轮history监督微调完全相同。

本轮可以报告“observation-only pretraining followed by budgeted supervised policy adaptation”。微调阶段使用真实动作标签，不能称全流程observation-only。它评估部署控制可塑性；原冻结路线继续衡量固定latent/history的可用性，两个问题分开呈现。通用微调收益不自动归因于MTE独有优势。

结果写入`results/mte_finetune_extension_2026_09_20/RESULTS_CN.md`。论文图表不自动更新，不上传GitHub。
